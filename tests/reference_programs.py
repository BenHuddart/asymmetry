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


def musrfit_kt_objective(directory, time, values, theory, parameters):
    """Evaluate a fixed KT theory on owned supplied-error type-8 observations."""
    import re
    from pathlib import Path

    executable = musrfit_executable()
    fixture = Path(__file__).parent / "fixtures/reference_cases/constant"
    shutil.copy(fixture / "musrfit_startup.xml", directory / "musrfit_startup.xml")
    (directory / "owned.dat").write_text(
        "HEADER\nTITLE: Independent KT integral control\nDATA\n"
        + "".join(f"{t:.17g}, {value:.17g}, 1\n" for t, value in zip(time, values, strict=True))
    )
    parameter_text = "\n".join(
        f"{i} {name} {value:.17g} 0 none" for i, (name, value) in enumerate(parameters, start=1)
    )
    request = (fixture / "owned.msr").read_text().replace("1 a 3 0.1 none", parameter_text)
    request = request.replace("asymmetry 1", f"asymmetry 1\n{theory}")
    request = request.replace("fit -0.5 2.5", f"fit {min(time) - 0.1:.17g} {max(time) + 0.1:.17g}")
    request = request.replace("range 0 2", f"range {min(time):.17g} {max(time):.17g}")
    (directory / "owned.msr").write_text(request)
    result = subprocess.run(
        [executable, "owned.msr", "--chisq-only", "--use-no-of-threads", "1"],
        cwd=directory,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (directory / "native_stdout.txt").write_text(result.stdout)
    (directory / "native_stderr.txt").write_text(result.stderr)
    match = re.search(r">> chisq = ([^,]+)", result.stdout)
    assert match is not None, result.stdout + result.stderr
    return float(match[1])
