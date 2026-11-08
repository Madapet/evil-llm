"""Persistent Python executor that runs INSIDE the sandbox container.

The harness starts the container with `docker run -i` and talks JSON lines over stdin/stdout:
  request : {"id": 1, "code": "...", "timeout": 600}
  response: {"id": 1, "stdout": "...", "stderr": "...", "error": null|"traceback", "timed_out": bool,
             "truncated": bool, "probes_used": int|null}
Globals persist between calls. Protocol I/O uses private dup'd fds so user code printing to
fd 0/1 cannot corrupt or read the channel. Output is capped per call.
"""
import contextlib
import io
import json
import os
import signal
import sys
import traceback
import types

MAX_OUT = int(os.environ.get("EXEC_MAX_OUT", 20000))
DEFAULT_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", 600))


class Capped(io.StringIO):
    def __init__(self):
        super().__init__()
        self.truncated = False

    def write(self, s):
        room = MAX_OUT - self.tell()
        if room <= 0:
            self.truncated = True
            return len(s)
        if len(s) > room:
            self.truncated = True
            s = s[:room]
        return super().write(s)


class Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise Timeout()


def main() -> None:
    proto_in = os.fdopen(os.dup(0), "r")
    proto_out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    signal.signal(signal.SIGALRM, _alarm)
    env: dict = {"__name__": "__sandbox__"}

    # Host channel: lets wsp.blind_probe ask the harness (host side) for probe results mid-call.
    # The host decides what is allowed and keeps the budget, so sandbox code cannot extend it.
    def host_call(payload: dict) -> dict:
        proto_out.write(json.dumps({"host_call": payload}) + "\n")
        proto_out.flush()
        return json.loads(proto_in.readline())

    host_mod = types.ModuleType("wsp_host")
    host_mod.call = host_call
    sys.modules["wsp_host"] = host_mod

    for line in proto_in:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if "code" not in req:      # stale host-call reply after a timeout: ignore
            continue
        out, err = Capped(), Capped()
        error, timed_out = None, False
        signal.alarm(int(req.get("timeout") or DEFAULT_TIMEOUT))
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(req["code"], "<run_python>", "exec"), env)
        except Timeout:
            timed_out, error = True, "TimeoutError: call exceeded time limit"
        except BaseException:
            error = traceback.format_exc(limit=-8)[-MAX_OUT:]
        finally:
            signal.alarm(0)
        pa = sys.modules.get("wsp.probe_api")
        proto_out.write(json.dumps({
            "id": req.get("id"), "stdout": out.getvalue(), "stderr": err.getvalue(),
            "error": error, "timed_out": timed_out, "truncated": out.truncated or err.truncated,
            "probes_used": getattr(pa, "_used", None) if pa else None,
        }) + "\n")
        proto_out.flush()


if __name__ == "__main__":
    main()
