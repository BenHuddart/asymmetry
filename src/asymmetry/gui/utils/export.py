"""Cancellable GLE subprocess invocation shared by export paths."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from asymmetry.gui.tasks import TaskCancelledError


def compile_gle(
    gle_executable: str,
    gle_file: str | Path,
    output_format: str | Sequence[str],
    *,
    cwd: str | Path,
    timeout: float = 120.0,
    cancelled: Callable[[], bool] | None = None,
) -> subprocess.CompletedProcess:
    """Compile once; terminate and reap on cancellation or timeout."""
    devices = output_format if isinstance(output_format, str) else ",".join(output_format)
    args = [gle_executable, "-d", devices, str(gle_file)]
    if cancelled is None:
        return subprocess.run(
            args, capture_output=True, text=True, check=True, cwd=str(cwd), timeout=timeout
        )
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd)
    ) as process:
        deadline = time.monotonic() + timeout
        try:
            while True:
                if cancelled():
                    raise TaskCancelledError
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
