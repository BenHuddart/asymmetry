"""F4 / P0-2 + P0-3: unsaved-changes guard and keyboard shortcuts.

The GUI previously had zero dirty tracking — closing the app or starting/opening
a project after hours of fitting silently dropped everything — and no keyboard
shortcuts at all. These tests pin the new behaviour:

* mutating user actions (load data, fit) mark the session modified;
* save / open / new clear the modified flag;
* a project restore does not mark the freshly-opened project modified;
* close / new / open prompt Save/Discard/Cancel when modified and honour Cancel;
* the project + fit actions carry standard keyboard shortcuts.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui, pytest.mark.real_save_guard]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402


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


def _ds(run_number: int = 11) -> MuonDataset:
    n = 8
    meta = {"run_number": run_number, "field": 100.0, "temperature": 10.0}
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, 110.0), bin_width=0.01),
            Histogram(counts=np.full(n, 90.0), bin_width=0.01),
        ],
        metadata=dict(meta),
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
        },
    )
    t = np.arange(n) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros(n),
        error=np.full(n, 0.01),
        metadata=dict(meta),
        run=run,
    )


# ── dirty tracking ──────────────────────────────────────────────────────


def test_fresh_window_is_clean(win: MainWindow) -> None:
    assert win._dirty is False
    assert win.isWindowModified() is False
    # Title carries Qt's [*] placeholder so setWindowModified can show "*".
    assert "[*]" in win.windowTitle()


def test_adding_data_marks_dirty(win: MainWindow, qapp: QApplication) -> None:
    assert win._dirty is False
    win._data_browser.add_dataset(_ds())
    qapp.processEvents()
    assert win._dirty is True
    assert win.isWindowModified() is True


def test_fit_completion_signals_wired_to_dirty(win: MainWindow) -> None:
    # Verify _mark_dirty is connected to the fit-result signals without firing
    # the heavy real handlers (which need a valid FitResult payload). disconnect
    # raises RuntimeError if the slot was never connected, so a clean disconnect
    # is the assertion.
    win._fit_panel.fit_completed.disconnect(win._mark_dirty)
    win._fit_panel.global_fit_completed.disconnect(win._mark_dirty)
    win._fit_panel.grouped_fit_completed.disconnect(win._mark_dirty)


def test_restore_does_not_mark_dirty(win: MainWindow, qapp: QApplication) -> None:
    # While a project loads, replayed mutations must not dirty the new project.
    win._restoring_project = True
    try:
        win._data_browser.add_dataset(_ds(12))
        # _mark_dirty is also invoked directly during restore replays.
        win._mark_dirty()
        qapp.processEvents()
    finally:
        win._restoring_project = False
    assert win._dirty is False
    assert win.isWindowModified() is False


def test_clear_dirty_resets_flag_and_title(win: MainWindow) -> None:
    win._mark_dirty()
    assert win.isWindowModified() is True
    win._clear_dirty()
    assert win._dirty is False
    assert win.isWindowModified() is False


def test_save_clears_dirty(win: MainWindow, qapp: QApplication, tmp_path, monkeypatch) -> None:
    from tests._qt_helpers import wait_for

    win._data_browser.add_dataset(_ds())
    qapp.processEvents()
    assert win._dirty is True

    monkeypatch.setattr(mw_module, "save_project", lambda *a, **k: None)
    win._write_project(str(tmp_path / "p.asymp"))
    wait_for(lambda: not win._project_save_active, qapp, timeout_s=5.0)
    assert win._dirty is False
    assert win.isWindowModified() is False


# ── Save/Discard/Cancel prompts ─────────────────────────────────────────


def test_maybe_save_no_prompt_when_clean(win: MainWindow, monkeypatch) -> None:
    called = {"n": 0}
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: called.__setitem__("n", 1))
    )
    assert win._maybe_save("closing") is True
    assert called["n"] == 0, "clean session must not prompt"


def test_maybe_save_discard_proceeds(win: MainWindow, monkeypatch) -> None:
    win._mark_dirty()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    assert win._maybe_save("closing") is True


def test_maybe_save_cancel_aborts(win: MainWindow, monkeypatch) -> None:
    win._mark_dirty()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    assert win._maybe_save("closing") is False


def test_close_when_dirty_and_cancel_keeps_window_open(
    win: MainWindow, qapp: QApplication, monkeypatch
) -> None:
    win._data_browser.add_dataset(_ds())
    qapp.processEvents()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is False, "Cancel must veto the close"


def test_close_when_clean_proceeds_without_prompt(
    win: MainWindow, qapp: QApplication, monkeypatch
) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("clean close must not prompt")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(_boom))
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted() is True


def test_new_project_cancel_preserves_state(
    win: MainWindow, qapp: QApplication, monkeypatch
) -> None:
    win._data_browser.add_dataset(_ds())
    qapp.processEvents()
    assert win._data_browser.all_datasets()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    win._on_new_project()
    assert win._data_browser.all_datasets(), "Cancel must not clear the session"
    assert win._dirty is True


def test_new_project_discard_clears_and_resets_dirty(
    win: MainWindow, qapp: QApplication, monkeypatch
) -> None:
    win._data_browser.add_dataset(_ds())
    qapp.processEvents()
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    win._on_new_project()
    assert not win._data_browser.all_datasets()
    assert win._dirty is False
    assert win.isWindowModified() is False


# ── keyboard shortcuts ──────────────────────────────────────────────────


def _shortcut_keys(win: MainWindow) -> set[str]:
    keys: set[str] = set()
    for action in win.menuBar().findChildren(QAction):
        for seq in action.shortcuts():
            keys.add(seq.toString(QKeySequence.SequenceFormat.PortableText))
    return keys


def test_standard_shortcuts_are_wired(win: MainWindow) -> None:
    keys = _shortcut_keys(win)
    # Save / Open / New / SaveAs all resolve to non-empty bindings on this
    # platform; assert by the portable text of the standard sequences.
    for std in (
        QKeySequence.StandardKey.Save,
        QKeySequence.StandardKey.Open,
        QKeySequence.StandardKey.New,
        QKeySequence.StandardKey.SaveAs,
    ):
        expected = QKeySequence(std).toString(QKeySequence.SequenceFormat.PortableText)
        assert expected in keys, f"missing shortcut for {std}: {expected}"


def test_fit_and_quit_shortcuts_present(win: MainWindow) -> None:
    keys = _shortcut_keys(win)
    assert "Ctrl+Return" in keys, "Fit shortcut missing"
    assert "Ctrl+Q" in keys, "Quit shortcut missing"
