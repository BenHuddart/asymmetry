"""Shared GLE subprocess-invocation helper for export/preview code paths."""

from __future__ import annotations

import subprocess
from pathlib import Path


def compile_gle(
    gle_executable: str,
    gle_file: str | Path,
    output_format: str,
    *,
    cwd: str | Path,
) -> subprocess.CompletedProcess:
    """Run GLE to compile *gle_file* to *output_format* in *cwd*.

    Centralizes the ``gle -d <fmt> <file>`` invocation (capture_output, text,
    check=True) shared by every GLE export/preview path. Raises
    ``subprocess.CalledProcessError`` on non-zero exit (callers handle it).
    Returns the ``CompletedProcess``.
    """
    return subprocess.run(
        [gle_executable, "-d", output_format, str(gle_file)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(cwd),
    )
