"""Exercise real exception hooks without changing the pytest process."""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


def _gui_environment(tmp_path):
    return {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONIOENCODING": "utf-8:backslashreplace",
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }


def _run_gui(tmp_path, script, *args):
    return subprocess.run(
        [sys.executable, "-c", _STARTUP + script, str(tmp_path), *args],
        env=_gui_environment(tmp_path),
        capture_output=True,
        encoding="utf-8",
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
windowed = sys.argv[2] == "windowed"
if windowed:
    sys.stderr = None

@Slot()
def raise_from_timer():
    print("timer slot entered", flush=True)
    qWarning("injected Qt warning")
    qCritical("injected Qt critical")
    qWarning("This plugin does not support propagateSizeHints() [filtered probe]")
    suffix = f" {chr(0xDCFF)}" if windowed else ""
    raise ValueError("injected timer error" + suffix)

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
    if mode == "windowed":
        assert r"\udcff" in log
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


def test_two_running_apps_share_the_rotating_log(tmp_path):
    observer_script = """
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.gui.crash_log import CrashLog

app = QApplication([])
sys.stderr = None
CrashLog(app)
(data_dir / "ready").touch()
sys.stdin.readline()

def raise_after_rotation():
    raise ValueError("fresh error after another instance rotated the log")

QTimer.singleShot(0, raise_after_rotation)
QTimer.singleShot(0, app.quit)
sys.exit(app.exec())
"""
    with subprocess.Popen(
        [sys.executable, "-c", _STARTUP + observer_script, str(tmp_path)],
        env=_gui_environment(tmp_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    ) as observer:
        try:
            deadline = time.monotonic() + 10
            ready = tmp_path / "app-data" / "ready"
            while not ready.is_file() and observer.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.is_file(), "The first app did not open its log"

            rotator = _run_gui(
                tmp_path,
                """
from PySide6.QtWidgets import QApplication

from asymmetry.gui.crash_log import CrashLog, session_log

app = QApplication([])
sys.stderr = None
CrashLog(app)
handler = session_log.handlers[0]
for index in range(handler.backupCount + 3):
    session_log.info("%d %s", index, "x" * (handler.maxBytes // 2))
""",
            )
            assert rotator.returncode == 0, rotator.stdout + rotator.stderr
            stdout, stderr = observer.communicate(input="\n", timeout=10)
            assert observer.returncode == 0, stdout + stderr
        finally:
            if observer.poll() is None:
                observer.kill()

    log_dir = tmp_path / "app-data" / "logs"
    assert (log_dir / "asymmetry.log.2").is_file()
    log = (log_dir / "asymmetry.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in log
    assert "in raise_after_rotation" in log
    assert "ValueError: fresh error after another instance rotated the log" in log


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
