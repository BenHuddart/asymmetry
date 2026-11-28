"""Exercise real exception hooks without changing the pytest process."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pytest

pytestmark = pytest.mark.gui

_STARTUP = """
import sys
from pathlib import Path
from asymmetry.gui.utils import app_data

data_dir = Path(sys.argv[1]) / "app-data"
app_data.app_data_dir = lambda: data_dir
"""


def _run_gui(tmp_path, script, *args):
    return subprocess.run(
        [sys.executable, "-c", _STARTUP + script, str(tmp_path), *args],
        env={
            **os.environ,
            "QT_QPA_PLATFORM": "offscreen",
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("mode", ["console", "windowed"])
def test_uncaught_timer_slot_reaches_log_and_window(tmp_path, mode):
    result = _run_gui(
        tmp_path,
        """
from PySide6.QtCore import QTimer, Slot, qCritical, qInstallMessageHandler, qWarning
from PySide6.QtWidgets import QApplication, QMessageBox
from asymmetry.gui import app

show_splash = app._create_splash_screen

def downstream_handler(kind, context, message):
    print(f"Qt downstream: {message}", flush=True)

qInstallMessageHandler(downstream_handler)
if sys.argv[2] == "windowed":
    sys.stderr = None

@Slot()
def raise_from_timer():
    print("timer slot entered", flush=True)
    qWarning("injected Qt warning")
    qCritical("injected Qt critical")
    qWarning("This plugin does not support propagateSizeHints() [filtered probe]")
    raise ValueError("injected timer error")

def check_splash(qapp, logo):
    splash = show_splash(qapp, logo)
    QTimer.singleShot(0, raise_from_timer)
    QTimer.singleShot(0, arm_notification_check)
    return splash

def arm_notification_check():
    QTimer.singleShot(0, find_notification)

def find_notification():
    for widget in QApplication.topLevelWidgets():
        if isinstance(widget, QMessageBox) and widget.isVisible():
            assert not widget.isModal()
            print(widget.windowTitle(), widget.text(), flush=True)
    QApplication.instance().quit()

app._create_splash_screen = check_splash
app._start_project_window = lambda qapp, *_args, **_kwargs: qapp.exec()
sys.argv = ["asymmetry-gui", "--smoke-test"]
app.main()
""",
        mode,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "timer slot entered" in result.stdout, result.stdout + result.stderr
    log_path = tmp_path / "app-data" / "logs" / "asymmetry.log"
    assert log_path.is_file(), result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in log
    assert "in raise_from_timer" in log
    assert "ValueError: injected timer error" in log
    assert "WARNING Qt [default] injected Qt warning" in log
    assert "ERROR Qt [default] injected Qt critical" in log
    assert "[filtered probe]" in log
    assert "Qt downstream: injected Qt warning" in result.stdout
    assert "Qt downstream: injected Qt critical" in result.stdout
    assert "[filtered probe]" not in result.stdout
    assert "Unexpected Error" in result.stdout
    assert "ValueError: injected timer error" in result.stdout
    assert "Help → Open Log Folder" in result.stdout
    assert str(log_path) in result.stdout


def test_thread_exception_and_native_stream(tmp_path):
    result = _run_gui(
        tmp_path,
        """
import faulthandler
import os
from threading import Thread

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMessageBox

from asymmetry.gui import crash_log
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.tasks import TaskRunner
from tests._qt_helpers import wait_for

app = QApplication([])
shown = []

class CheckedMessageBox(QMessageBox):
    def show(self):
        assert QThread.currentThread() is app.thread()
        shown.append(self.text())
        super().show()

crash_log.QMessageBox = CheckedMessageBox
log_dir = data_dir / "logs"
log_dir.mkdir(parents=True)
native_path = log_dir / "native-crash.log"
native_path.write_text("earlier session\\n", encoding="utf-8")
reporter = crash_log.CrashLog(app)
assert faulthandler.is_enabled()

panel = LogPanel()
panel.log("Opened run file.nxs", tag="load")

def raise_from_thread():
    raise RuntimeError("injected thread error")

thread = Thread(target=raise_from_thread)
thread.start()
thread.join()
assert not shown
wait_for(lambda: shown, app)
assert len(shown) == 1
assert "RuntimeError: injected thread error" in shown[0]

def raise_handled(worker):
    raise ValueError("handled task error")

handled = []
runner = TaskRunner(app)
runner.start(raise_handled, on_error=handled.append)
wait_for(lambda: handled, app)
runner.shutdown()
app.processEvents()
assert handled == ["handled task error"]
assert len(shown) == 1
assert os.path.samestat(os.fstat(reporter._native_stream.fileno()), native_path.stat())
""",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    log_dir = tmp_path / "app-data" / "logs"
    log = (log_dir / "asymmetry.log").read_text(encoding="utf-8")
    for marker in ("Session started", "Asymmetry", "Python", "Qt", "PySide6"):
        assert marker in log
    assert "[LOAD] Opened run file.nxs" in log
    assert "Traceback (most recent call last):" in log
    assert "in raise_from_thread" in log
    assert "RuntimeError: injected thread error" in log
    assert "handled task error" not in log

    native_log = (log_dir / "native-crash.log").read_text(encoding="utf-8")
    previous, session = native_log.strip().split("\n\n")
    assert previous == "earlier session"
    timestamp, label = session.split(" ", 1)
    assert datetime.fromisoformat(timestamp).tzinfo is not None
    assert label == "Session started"


def test_help_opens_application_log_folder(qapp, monkeypatch, tmp_path):
    from PySide6.QtGui import QDesktopServices

    from asymmetry.gui.mainwindow import MainWindow

    urls = []
    monkeypatch.setattr(QDesktopServices, "openUrl", urls.append)
    window = MainWindow()
    help_action = next(action for action in window.menuBar().actions() if action.text() == "&Help")
    help_menu = help_action.menu()

    next(action for action in help_menu.actions() if action.text() == "Open &Log Folder").trigger()

    assert len(urls) == 1
    assert urls[0].isLocalFile()
    assert urls[0].toLocalFile() == str(tmp_path / "app-data" / "logs")
