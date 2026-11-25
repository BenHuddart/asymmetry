"""Real-process application teardown must finish while Qt is still running."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("mode", ["normal", "smoke", "cancel"])
def test_application_destroys_shell_before_return(tmp_path, mode):
    script = """
import atexit
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid
from asymmetry.gui import app
from asymmetry.gui.shell import ProjectShell

atexit.register(lambda: print("interpreter-exit", flush=True))
class ClosingShell(ProjectShell):
    def __init__(self):
        super().__init__()
        self.destroyed.connect(lambda: print("shell-destroyed", flush=True))
        QApplication.instance().aboutToQuit.connect(
            lambda: print("about-to-quit", flush=True))
    def show(self):
        super().show()
        if sys.argv[0] == "cancel":
            QTimer.singleShot(0, self.cancel_then_close)
        elif sys.argv[0] == "normal":
            QTimer.singleShot(0, self.close)
    def cancel_then_close(self):
        page = self.pages()[0]
        page._maybe_save = lambda _context: False
        assert not self.close()
        assert isValid(self) and self.isVisible()
        assert self.pages() == [page]
        print("close-cancelled", flush=True)
        page._maybe_save = lambda _context: True
        QTimer.singleShot(0, self.close)
app.ProjectShell = ClosingShell
sys.argv = [sys.argv[1]] + (["--smoke-test"] if sys.argv[1] == "smoke" else [])
try:
    app.main()
except SystemExit as exc:
    assert exc.code == 0
print("application-returned", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, mode],
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    markers = result.stdout.splitlines()
    assert set(markers[-4:-2]) == {"shell-destroyed", "about-to-quit"}
    assert markers[-2:] == ["application-returned", "interpreter-exit"]
    if mode == "cancel":
        assert "close-cancelled" in markers
    assert "Traceback (most recent call last):" not in result.stderr
    assert "Exception ignored" not in result.stderr
    assert "screen_guard:" not in result.stderr
