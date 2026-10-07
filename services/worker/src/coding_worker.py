"""Bounded coding verification worker: named tools only, never a generic shell."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationResult:
    tool: str
    returncode: int
    output: str
    timed_out: bool = False


_ALLOWED = {
    "pytest": ("pytest",),
    "ruff": ("ruff",),
    "mypy": ("mypy",),
}


def run_verification(
    tool: str, workspace: Path, args: list[str] | None = None, timeout_s: int = 300
) -> VerificationResult:
    if tool not in _ALLOWED:
        raise ValueError("unsupported verification tool")
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    if any("\x00" in str(arg) for arg in (args or [])):
        raise ValueError("invalid argument")
    safe_args = [str(arg) for arg in (args or [])][:32]
    if any(
        arg.startswith("-") and arg not in {"-q", "-x", "--strict", "--check"} for arg in safe_args
    ):
        raise ValueError("unsupported verification option")
    argv = [*_ALLOWED[tool], *safe_args]
    try:
        cp = subprocess.run(
            argv,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1, min(timeout_s, 900)),
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "")[-32_768:] if isinstance(exc.stdout, str) else ""
        return VerificationResult(tool, 124, out, True)
    return VerificationResult(tool, cp.returncode, cp.stdout[-32_768:])
