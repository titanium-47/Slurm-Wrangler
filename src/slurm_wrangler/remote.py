"""SSH / rsync helpers. All remote access goes through the user's ~/.ssh/config
aliases (klone-login, tillicum) and rides the ControlMaster socket so Duo is
answered at most once per session.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from typing import List, Optional


class RemoteError(Exception):
    pass


def _ssh_base(login: str) -> List[str]:
    # BatchMode would break first-time Duo; rely on the user's ssh config.
    return ["ssh", "-o", "ConnectTimeout=15", login]


def ssh_run(login: str, command: str, *, check: bool = True, capture: bool = True,
            input_text: Optional[str] = None, timeout: Optional[int] = 600) -> subprocess.CompletedProcess:
    """Run a shell command on the remote host."""
    proc = subprocess.run(
        _ssh_base(login) + [command],
        input=input_text,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        err = (proc.stderr or "").strip() if capture else ""
        raise RemoteError(f"ssh {login} '{command[:120]}...' failed (rc={proc.returncode}): {err}")
    return proc


def ssh_check_socket(login: str) -> bool:
    """True if a ControlMaster socket is already up (no auth needed)."""
    proc = subprocess.run(["ssh", "-O", "check", login], capture_output=True, text=True)
    return proc.returncode == 0


def ssh_test(login: str, test_expr: str) -> bool:
    """Run a remote `test` expression, e.g. '-f /path/file'."""
    proc = ssh_run(login, f"test {test_expr}", check=False)
    return proc.returncode == 0


def ssh_mkdirs(login: str, paths: List[str]) -> None:
    if paths:
        quoted = " ".join(shlex.quote(p) for p in paths)
        ssh_run(login, f"mkdir -p {quoted}")


def rsync(src: str, dest: str, args: List[str], *, quiet: bool = False,
          timeout: Optional[int] = None) -> None:
    cmd = ["rsync"] + args + [src, dest]
    stdout = subprocess.DEVNULL if quiet else None
    proc = subprocess.run(cmd, stdout=stdout, timeout=timeout)
    if proc.returncode != 0:
        raise RemoteError(f"rsync failed (rc={proc.returncode}): {' '.join(shlex.quote(c) for c in cmd)}")


def run_local(cmd: List[str], *, check: bool = True, capture: bool = False,
              cwd: Optional[str] = None, env=None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=capture, text=True, cwd=cwd, env=env)
    if check and proc.returncode != 0:
        detail = (proc.stderr or "").strip()[-2000:] if capture else ""
        raise RemoteError(f"command failed (rc={proc.returncode}): {' '.join(cmd)}\n{detail}")
    return proc


def info(msg: str) -> None:
    print(f"[sw] {msg}")


def warn(msg: str) -> None:
    print(f"\033[33m[sw] WARNING: {msg}\033[0m", file=sys.stderr)


def die(msg: str, code: int = 1) -> "None":
    print(f"\033[31m[sw] ERROR: {msg}\033[0m", file=sys.stderr)
    sys.exit(code)
