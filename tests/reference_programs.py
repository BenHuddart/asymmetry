"""Public CLI identity boundary for optional musrfit executions."""

import os
import shutil
import subprocess

import pytest

MUSRFIT_VERSION = (
    "musrfit version: 1.12.1, git-branch: root6, git-rev: "
    "Sun Oct 11 10:28:11 2026 +0200 - 708f777b (Release), ROOT version: 6.40.04"
)


def musrfit_executable(tool: str = "musrfit") -> str:
    """Use an explicit command or PATH; selected but broken references fail."""
    executable = os.environ.get(f"ASYMMETRY_{tool.upper()}") or shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} executable is not installed; set ASYMMETRY_{tool.upper()}")
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True, timeout=15
    )
    expected = MUSRFIT_VERSION.replace("musrfit", tool, 1)
    assert result.stdout.strip() == expected, (
        f"{tool} identity mismatch: expected {expected!r}, got {result.stdout!r}"
    )
    return executable
