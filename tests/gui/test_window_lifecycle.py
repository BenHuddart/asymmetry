from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Qt, QTimer, Slot
from PySide6.QtWidgets import QDockWidget, QMainWindow

from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.ui_manager import _ui_scale_broadcast
from tests._qt_helpers import dispose_qt_widgets, flush_deferred_deletes

pytestmark = pytest.mark.gui
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["main", "synthetic", "unlink"])
def test_window_lifecycle_survives_in_subprocess(mode: str) -> None:
    node = (
        "test_close_with_unremovable_autosave_still_closes"
        if mode == "unlink"
        else f"test_lifecycle_probe[{mode}]"
    )
    result = subprocess.run(
        [
            sys.executable,
            "tools/harness.py",
            "test",
            "--no-parallel",
            "--",
            f"tests/gui/test_window_lifecycle.py::{node}",
            "-s",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Error calling Python override" not in output


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["main", "synthetic"])
def test_lifecycle_probe(qapp, mode: str) -> None:
    class SyntheticWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.callback = lambda: None
            for index in range(5):
                dock = QDockWidget(str(index), self)
                self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
                dock.installEventFilter(self)

        def eventFilter(self, watched, event):  # noqa: N802
            return False

    for index in range(25):
        window = MainWindow() if mode == "main" else SyntheticWindow()
        if mode == "main":
            window._ui_manager.set_ui_scale(0.9 if index % 2 else 1.1)
        window.show()
        qapp.processEvents()
        assert window.close()
        dispose_qt_widgets(qapp)


@pytest.mark.slow
def test_close_with_unremovable_autosave_still_closes(qapp, monkeypatch) -> None:
    window = MainWindow()
    window.show()
    messages: list[str] = []
    shutdowns: list[bool] = []
    shutdown = window._shutdown_workers

    def record_shutdown():
        shutdown()
        shutdowns.append(True)

    def deny_unlink(path, **kwargs):
        raise PermissionError("synthetic unwritable autosave")

    monkeypatch.setattr(window, "_shutdown_workers", record_shutdown)
    monkeypatch.setattr(window._log_panel, "log", messages.append)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", deny_unlink)
        assert window.close()
    assert not window.isVisible()
    assert shutdowns == [True]
    assert not window._autosave_timer.isActive()
    assert any(message.startswith("Could not remove autosave:") for message in messages)


def test_no_callbacks_after_window_deletion(qapp) -> None:
    events: list[str] = []

    class Receiver(QObject):
        @Slot()
        def tick(self):
            events.append("late timer")

    signal = "2ui_scale_broadcast(double)"
    baseline = _ui_scale_broadcast.receivers(signal)
    window = MainWindow()
    assert _ui_scale_broadcast.receivers(signal) > baseline
    window._schedule_memory_settle()
    receiver = Receiver(window)
    QTimer.singleShot(0, receiver, receiver.tick)
    assert window.close()
    window.deleteLater()
    flush_deferred_deletes(qapp)
    assert _ui_scale_broadcast.receivers(signal) == baseline
    _ui_scale_broadcast.ui_scale_broadcast.emit(1.1)
    qapp.processEvents()
    assert events == []


def test_teardown_clears_dirty_grouping_drafts(qapp, monkeypatch) -> None:
    from PySide6.QtWidgets import QMessageBox

    from asymmetry.gui.windows.grouping_dialog import GroupingDialog
    from tests.gui.test_beta_section import _dataset

    questions: list[str] = []

    def cancel_question(*args):
        questions.append("discard prompt")
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(cancel_question))
    dialog = GroupingDialog([_dataset()])
    dialog._beta_section.set_value(0.9)
    assert dialog._draft_dirty
    dispose_qt_widgets(qapp)
    assert questions == []
