"""Docker/Podman wrapper (host side). Starts the locked-down container and talks to executor.py.

Isolation: no network, read-only rootfs, all caps dropped, no-new-privileges, pids/mem/cpu limits,
weights mounted read-only, /work is a size-limited tmpfs (wiped when the container exits).
Never mounted: home dir, docker socket, ssh keys, .env, harness/secrets.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

FORBIDDEN_MOUNT_PARTS = (".ssh", ".env", "secrets", "docker.sock", ".aws", ".config", ".cache/huggingface")


@dataclass
class SandboxConfig:
    image: str = "wsp-sandbox:latest"
    weights_dir: str = ""              # neutralized dir, mounted read-only at /model
    extra_mounts: dict[str, str] = field(default_factory=dict)  # host dir -> container path, read-only
    probe_budget: int = 0              # prompts the probe API may run (0 = no forward passes)
    gpu: bool = False                  # only for conditions that allow forward passes
    gpu_device: str = "0"
    memory: str = "16g"                # laptops: leave RAM for the host; raise on big machines
    cpus: str = field(default_factory=lambda: str(min(8, os.cpu_count() or 1)))  # docker rejects > host CPUs
    pids_limit: int = 256
    work_size: str = "4g"
    runtime: str = "docker"            # or "podman"
    name: str = "wsp-sandbox"
    call_timeout: int = 600
    max_output: int = 20000
    extra_env: dict[str, str] = field(default_factory=dict)


def build_run_cmd(cfg: SandboxConfig) -> list[str]:
    if not cfg.weights_dir:
        raise ValueError("weights_dir required")
    wd = Path(cfg.weights_dir).resolve()
    mounts = [(wd, "/model")] + [(Path(h).resolve(), c) for h, c in cfg.extra_mounts.items()]
    for host, _ in mounts:
        if any(part in str(host) for part in FORBIDDEN_MOUNT_PARTS):
            raise ValueError(f"refusing to mount suspicious path {host}")
    cmd = [
        cfg.runtime, "run", "--rm", "-i", "--name", cfg.name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(cfg.pids_limit),
        "--memory", cfg.memory,
        "--cpus", cfg.cpus,
        "--user", "10001:10001",
        "--tmpfs", f"/work:rw,noexec,nosuid,size={cfg.work_size},uid=10001,gid=10001",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m,uid=10001,gid=10001",
        "-e", f"EXEC_MAX_OUT={cfg.max_output}",
        "-e", f"PROBE_BUDGET={cfg.probe_budget}",
        "-e", f"EXEC_TIMEOUT={cfg.call_timeout}",
    ]
    for host, ctr in mounts:
        cmd += ["-v", f"{host}:{ctr}:ro"]
    for k, v in cfg.extra_env.items():
        cmd += ["-e", f"{k}={v}"]
    if cfg.gpu:
        cmd += ["--gpus", f"device={cfg.gpu_device}"]
    cmd.append(cfg.image)
    return cmd


class Sandbox:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            build_run_cmd(self.cfg), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def run_python(self, code: str, timeout: int | None = None) -> dict:
        """Execute code in the persistent interpreter. Hard wall-clock guard on the host side."""
        assert self.proc and self.proc.stdin and self.proc.stdout, "not started"
        timeout = timeout or self.cfg.call_timeout
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            self.proc.stdin.write(json.dumps({"id": rid, "code": code, "timeout": timeout}) + "\n")
            self.proc.stdin.flush()
            result: dict = {}

            def read():
                line = self.proc.stdout.readline()
                if line:
                    result.update(json.loads(line))

            t = threading.Thread(target=read, daemon=True)
            t.start()
            t.join(timeout + 30)          # executor's own alarm should fire first
            if t.is_alive() or not result:
                self.stop()
                return {"id": rid, "stdout": "", "stderr": "", "timed_out": True, "truncated": False,
                        "error": "sandbox unresponsive; container killed"}
            return result

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            subprocess.run([self.cfg.runtime, "kill", self.cfg.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.proc.kill()
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
