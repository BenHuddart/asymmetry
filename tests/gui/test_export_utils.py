from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from asymmetry.gui.utils.export import compile_gle  # noqa: E402

pytestmark = pytest.mark.gui


def test_compile_gle_builds_expected_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    gle_file = tmp_path / "plot.gle"

    result = compile_gle("gle", gle_file, "png", cwd=tmp_path)

    mock_run.assert_called_once_with(
        ["gle", "-d", "png", str(gle_file)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
        timeout=120.0,
    )
    assert result is mock_run.return_value


def test_compile_gle_accepts_string_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0))
    monkeypatch.setattr(subprocess, "run", mock_run)

    gle_file = str(tmp_path / "plot.gle")

    compile_gle("/usr/local/bin/gle", gle_file, "eps", cwd=str(tmp_path))

    mock_run.assert_called_once_with(
        ["/usr/local/bin/gle", "-d", "eps", gle_file],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(tmp_path),
        timeout=120.0,
    )


def test_compile_gle_propagates_called_process_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args[0], stderr="boom")

    monkeypatch.setattr(subprocess, "run", _raise)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        compile_gle("gle", tmp_path / "plot.gle", "png", cwd=tmp_path)

    assert exc_info.value.stderr == "boom"


@pytest.mark.parametrize("cancel", [True, False])
def test_compiler_cancellation_and_timeout_reap_process(monkeypatch, tmp_path, cancel):
    import sys

    from asymmetry.gui.tasks import TaskCancelledError

    original = subprocess.Popen
    processes = []

    def launch(args, **kwargs):
        process = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(TaskCancelledError if cancel else subprocess.TimeoutExpired):
        compile_gle(
            "gle",
            tmp_path / "plot.gle",
            ("eps", "pdf"),
            cwd=tmp_path,
            timeout=0.05,
            cancelled=lambda: cancel,
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None
