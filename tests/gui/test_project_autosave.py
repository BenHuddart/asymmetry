"""D9: crash-safe autosave — timer arming, the write itself, and recovery on open.

``_mark_dirty`` arms a single-shot ``QTimer`` (interval from the
``project/autosave_interval_minutes`` QSettings key, default 5 minutes, ``0``
disables); firing it writes ``<stem>.autosave.asymp`` beside the project
through the same ``save_project`` used for a real save, without touching
``_current_project_path``, the window title or the dirty flag. A real save
deletes any stale autosave; opening a project whose autosave is newer offers
to recover it.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.real_save_guard]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QCloseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.project.schema import save_project as core_save_project  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from tests._qt_helpers import wait_for  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    w = MainWindow()
    w.show()
    qapp.processEvents()
    return w


def _patch_recovery_choice(monkeypatch, choice: str) -> None:
    """Patch the D9 recovery dialog in mainwindow to auto-select *choice*.

    *choice* is ``load`` / ``open`` / ``cancel``, matching the "Load
    autosave" / "Open saved file" / "Cancel" buttons.
    """
    label = {
        "load": "Load autosave",
        "open": "Open saved file",
        "cancel": "Cancel",
    }[choice]

    class _Box:
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole
        StandardButton = QMessageBox.StandardButton

        def __init__(self, *args, **kwargs) -> None:
            self._buttons: dict[str, object] = {}
            self._clicked: object = None

        def setIcon(self, *a) -> None: ...  # noqa: N802
        def setWindowTitle(self, *a) -> None: ...  # noqa: N802
        def setText(self, *a) -> None: ...  # noqa: N802
        def setDefaultButton(self, *a) -> None: ...  # noqa: N802

        def addButton(self, arg, role=None) -> object:  # noqa: N802
            btn = object()
            key = arg if isinstance(arg, str) else "Cancel"
            self._buttons[key] = btn
            return btn

        def exec(self) -> int:  # noqa: A003
            self._clicked = self._buttons[label]
            return 0

        def clickedButton(self) -> object:  # noqa: N802
            return self._clicked

    monkeypatch.setattr(mw_module, "QMessageBox", _Box)


# ── timer arming ──────────────────────────────────────────────────────────


def test_mark_dirty_arms_timer_with_configured_interval(win: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(mw_module, "autosave_interval_minutes", lambda: 7)
    assert not win._autosave_timer.isActive()

    win._mark_dirty()

    assert win._autosave_timer.isActive()
    assert win._autosave_timer.interval() == 7 * 60_000


def test_mark_dirty_does_not_rearm_an_already_running_timer(win: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(mw_module, "autosave_interval_minutes", lambda: 5)
    win._mark_dirty()
    win._autosave_timer.start(123)  # simulate a timer already ticking down

    win._mark_dirty()

    assert win._autosave_timer.interval() == 123


def test_interval_zero_never_arms_the_timer(win: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(mw_module, "autosave_interval_minutes", lambda: 0)

    win._mark_dirty()

    assert not win._autosave_timer.isActive()


def test_clear_dirty_stops_the_timer(win: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(mw_module, "autosave_interval_minutes", lambda: 5)
    win._mark_dirty()
    assert win._autosave_timer.isActive()

    win._clear_dirty()

    assert not win._autosave_timer.isActive()


# ── firing the timer ────────────────────────────────────────────────────


def test_autosave_timeout_writes_snapshot_and_leaves_session_unchanged(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path = tmp_path / "proj.asymp"
    win._current_project_path = str(proj_path)
    win._mark_dirty()
    original_title = win.windowTitle()

    recorded: list[str] = []

    def _record(state, path):
        recorded.append(str(path))
        core_save_project(state, path)

    monkeypatch.setattr(mw_module, "save_project", _record)

    win._on_autosave_timeout()
    wait_for(lambda: not win._project_save_active, qapp, timeout_s=5.0)

    expected = tmp_path / "proj.autosave.asymp"
    assert recorded == [str(expected)]
    assert expected.exists()
    assert not proj_path.exists(), "autosave must never touch the real project file"
    assert win._dirty is True
    assert win._current_project_path == str(proj_path)
    assert win.windowTitle() == original_title


def test_autosave_timeout_does_nothing_when_clean(win: MainWindow, monkeypatch) -> None:
    called = []
    monkeypatch.setattr(mw_module, "save_project", lambda *a, **k: called.append(1))

    win._on_autosave_timeout()

    assert called == []


def test_autosave_timeout_skips_and_rearms_when_a_save_is_active(
    win: MainWindow, monkeypatch
) -> None:
    win._current_project_path = "unused.asymp"
    win._mark_dirty()
    win._project_save_active = True
    called = []
    monkeypatch.setattr(win, "collect_project_state", lambda: called.append(1))
    win._autosave_timer.stop()

    win._on_autosave_timeout()

    assert called == [], "must not collect/write while a real save is in flight"
    assert win._autosave_timer.isActive(), "must retry once the active save clears"


# ── interaction with a real save ────────────────────────────────────────


def test_successful_save_deletes_existing_autosave(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path = tmp_path / "proj.asymp"
    autosave_path = tmp_path / "proj.autosave.asymp"
    autosave_path.write_text("{}")
    monkeypatch.setattr(mw_module, "save_project", lambda *a, **k: None)

    win._write_project(str(proj_path))
    wait_for(lambda: not win._project_save_active, qapp, timeout_s=5.0)

    assert not autosave_path.exists()


def test_clean_close_deletes_leftover_autosave(win: MainWindow, tmp_path) -> None:
    win._current_project_path = str(tmp_path / "proj.asymp")
    autosave_path = tmp_path / "proj.autosave.asymp"
    autosave_path.write_text("{}")
    assert win._dirty is False

    event = QCloseEvent()
    win.closeEvent(event)

    assert not autosave_path.exists()


def test_dirty_close_via_discard_leaves_autosave_in_place(
    win: MainWindow, tmp_path, monkeypatch
) -> None:
    win._current_project_path = str(tmp_path / "proj.asymp")
    autosave_path = tmp_path / "proj.autosave.asymp"
    autosave_path.write_text("{}")
    win._mark_dirty()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )

    event = QCloseEvent()
    win.closeEvent(event)

    assert autosave_path.exists()


# ── recovery on open ─────────────────────────────────────────────────────


def _write_project_pair(win: MainWindow, tmp_path):
    """Write a real project file and a newer autosave sibling; return their paths."""
    proj_path = tmp_path / "proj.asymp"
    autosave_path = tmp_path / "proj.autosave.asymp"
    state = win.collect_project_state()
    core_save_project(state, proj_path)
    core_save_project(state, autosave_path)
    future = time.time() + 10
    os.utime(autosave_path, (future, future))
    return proj_path, autosave_path


def test_open_with_newer_autosave_prompts_and_load_reads_the_autosave(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path, autosave_path = _write_project_pair(win, tmp_path)
    real_load_project = mw_module.load_project
    calls: list[str] = []

    def _recording_load(path):
        calls.append(str(path))
        return real_load_project(path)

    monkeypatch.setattr(mw_module, "load_project", _recording_load)
    _patch_recovery_choice(monkeypatch, "load")

    win._open_project_file(str(proj_path))
    qapp.processEvents()

    assert calls == [str(autosave_path)]
    assert win._current_project_path == str(proj_path)
    assert win._dirty is True


def test_open_with_newer_autosave_open_saved_file_reads_the_project(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path, autosave_path = _write_project_pair(win, tmp_path)
    real_load_project = mw_module.load_project
    calls: list[str] = []

    def _recording_load(path):
        calls.append(str(path))
        return real_load_project(path)

    monkeypatch.setattr(mw_module, "load_project", _recording_load)
    _patch_recovery_choice(monkeypatch, "open")

    win._open_project_file(str(proj_path))
    qapp.processEvents()

    assert calls == [str(proj_path)]
    assert win._current_project_path == str(proj_path)
    assert win._dirty is False
    assert autosave_path.exists(), "declining recovery must leave the autosave in place"


def test_open_with_newer_autosave_cancel_aborts_the_open(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path, _autosave_path = _write_project_pair(win, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(mw_module, "load_project", lambda path: calls.append(path))
    _patch_recovery_choice(monkeypatch, "cancel")
    win._current_project_path = None

    win._open_project_file(str(proj_path))
    qapp.processEvents()

    assert calls == []
    assert win._current_project_path is None


def test_open_without_a_newer_autosave_opens_normally_without_prompting(
    win: MainWindow, qapp: QApplication, tmp_path, monkeypatch
) -> None:
    proj_path = tmp_path / "solo.asymp"
    core_save_project(win.collect_project_state(), proj_path)

    class _NoPromptExpected:
        def __init__(self, *a, **k) -> None:
            raise AssertionError("must not prompt when there is no autosave to recover")

    monkeypatch.setattr(mw_module, "QMessageBox", _NoPromptExpected)

    win._open_project_file(str(proj_path))
    qapp.processEvents()

    assert win._current_project_path == str(proj_path)
    assert win._dirty is False
