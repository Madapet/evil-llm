import json
import subprocess
import sys
from pathlib import Path

import pytest

from harness.sandbox import Sandbox, SandboxConfig, build_run_cmd

EXECUTOR = Path(__file__).resolve().parents[1] / "sandbox" / "executor.py"


def test_run_cmd_isolation_flags(tmp_path):
    cmd = build_run_cmd(SandboxConfig(weights_dir=str(tmp_path)))
    joined = " ".join(cmd)
    for needed in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                   "--pids-limit", "--memory", "--cpus", "--user 10001:10001", ":/model:ro"):
        assert needed in joined
    assert "--gpus" not in cmd                        # 0-pass conditions get no GPU
    assert "--privileged" not in cmd
    assert sum(1 for a in cmd if a == "-v") == 1      # weights are the only mount
    assert not any("docker.sock" in a or ".env" in a or "secrets" in a for a in cmd)


def test_gpu_only_when_requested(tmp_path):
    cmd = build_run_cmd(SandboxConfig(weights_dir=str(tmp_path), gpu=True, gpu_device="0"))
    assert cmd[cmd.index("--gpus") + 1] == "device=0"


@pytest.mark.parametrize("bad", [".ssh", "secrets", ".env"])
def test_refuses_dangerous_mount(tmp_path, bad):
    p = tmp_path / bad
    p.mkdir()
    with pytest.raises(ValueError):
        build_run_cmd(SandboxConfig(weights_dir=str(p)))


class LocalExec:
    """Drive executor.py as a plain subprocess (same protocol the container uses)."""

    def __init__(self, **env):
        import os
        self.p = subprocess.Popen([sys.executable, str(EXECUTOR)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env={**os.environ, **env})
        self.i = 0

    def run(self, code, timeout=5):
        self.i += 1
        self.p.stdin.write(json.dumps({"id": self.i, "code": code, "timeout": timeout}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        self.p.kill()


@pytest.fixture()
def ex():
    e = LocalExec(EXEC_MAX_OUT="100")
    yield e
    e.close()


def test_executor_state_persists_and_captures(ex):
    assert ex.run("x = 41")["error"] is None
    r = ex.run("print(x + 1)")
    assert r["stdout"].strip() == "42" and r["error"] is None


def test_executor_reports_exceptions_and_survives(ex):
    r = ex.run("1/0")
    assert "ZeroDivisionError" in r["error"]
    assert ex.run("print('still alive')")["stdout"].strip() == "still alive"


def test_executor_timeout(ex):
    r = ex.run("while True: pass", timeout=1)
    assert r["timed_out"] is True
    assert ex.run("print(1)")["stdout"].strip() == "1"


def test_executor_output_cap(ex):
    r = ex.run("print('a' * 10000)")
    assert r["truncated"] is True and len(r["stdout"]) <= 100


def test_executor_protocol_channel_cannot_be_polluted(ex):
    r = ex.run("import os; os.write(1, b'garbage not json\\n'); print('ok')")
    assert r["stdout"].strip() == "ok"                 # response still parsed as clean JSON
    assert ex.run("print(open(0).read() if False else 'fine')")["stdout"].strip() == "fine"


def test_sandbox_class_handles_dead_container():
    sb = Sandbox(SandboxConfig(weights_dir="/tmp", runtime="definitely-not-a-runtime"))
    with pytest.raises(FileNotFoundError):
        sb.start()
