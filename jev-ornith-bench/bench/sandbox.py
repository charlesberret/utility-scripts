"""A disposable container for the agentic sysadmin tasks.

Two reasons this is not optional. The obvious one: the tasks under test are
things like "clean up the full disk" and "fix the broken unit file", and a
model that misjudges one should wreck a container rather than your laptop.

The less obvious one: several tasks in the sysadmin suite are traps, where the
correct behaviour is to refuse or escalate. You cannot score a refusal unless
the model genuinely had the ability to do the damage. Blocking the action at
the tool layer would score the harness, not the model -- so the sandbox gives
the model real capability inside a boundary that makes the real capability
safe.

Network is off by default: it stops a task quietly succeeding by fetching the
answer, and it keeps a misjudged `curl | sh` inside the box.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass


class SandboxError(RuntimeError):
    pass


@dataclass
class Exec:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def output(self) -> str:
        parts = [p for p in (self.stdout.strip(), self.stderr.strip()) if p]
        return "\n".join(parts) or "(no output)"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        ).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


class Sandbox:
    """A container that lives for one task."""

    def __init__(
        self,
        image: str = "debian:bookworm-slim",
        *,
        network: bool = False,
        memory: str = "512m",
        pids: int = 256,
        workdir: str = "/work",
    ) -> None:
        self.image = image
        self.network = network
        self.memory = memory
        self.pids = pids
        self.workdir = workdir
        self.name = f"jevbench-{uuid.uuid4().hex[:12]}"
        self._up = False

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not docker_available():
            raise SandboxError("docker is not available; agentic tasks need it")
        command = [
            "docker", "run", "-d", "--rm",
            "--name", self.name,
            "--network", "bridge" if self.network else "none",
            "--memory", self.memory,
            "--pids-limit", str(self.pids),
            "-w", self.workdir,
            self.image,
            "sleep", "3600",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise SandboxError(f"could not start sandbox: {result.stderr.strip()}")
        self._up = True
        self.run(f"mkdir -p {self.workdir}", timeout=15)

    def run(self, command: str, timeout: int = 60) -> Exec:
        """Run a shell command inside the container."""
        if not self._up:
            raise SandboxError("sandbox is not running")
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.name, "sh", "-lc", command],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # A hung command is a task failure, not a harness failure. Tasks
            # that wait on input are a real way a small model gets stuck.
            return Exec(124, "", f"timed out after {timeout}s", timed_out=True)
        # Long outputs are truncated in the middle: the head carries the
        # error and the tail carries the summary, and the middle of a
        # thousand-line log is rarely what decides the next action.
        return Exec(result.returncode, _clip(result.stdout), _clip(result.stderr))

    def stop(self) -> None:
        if not self._up:
            return
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
        self._up = False


def _clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n... [{omitted} characters omitted] ...\n{text[-half:]}"
