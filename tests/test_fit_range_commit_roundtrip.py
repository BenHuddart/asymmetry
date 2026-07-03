"""Characterization: fit-range commit round-trip for GlobalFitTab.

``SingleFitTab``'s fit-range spinbox <-> engine round-trip is already covered
by ``tests/test_fit_panel_phase5_range.py``. ``GlobalFitTab`` shares the same
``FloatLimitField`` widgets and the same ``_apply_fit_range_display`` /
``fit_range_edit_committed`` plumbing (see ``fit_panel.py``), but had no
dedicated coverage. These tests pin the observable contract only (spinbox
values, the ``fit_range_edit_committed`` signal payload, and the plot's fit
range via ``MainWindow``) so they survive the later ``fit_panel.py`` split.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import asymmetry.gui.mainwindow as mw_module
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fit_panel import GlobalFitTab
from asymmetry.gui.widgets.axis_limits import FloatLimitField


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tab(qapp: QApplication) -> GlobalFitTab:
    return GlobalFitTab()


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    w = MainWindow()
    w.show()
    qapp.processEvents()
    return w


# ── Unit tests on GlobalFitTab ────────────────────────────────────────────────


def test_global_fit_range_spinboxes_exist(tab: GlobalFitTab) -> None:
    assert isinstance(tab._fit_range_min_spin, FloatLimitField)
    assert isinstance(tab._fit_range_max_spin, FloatLimitField)


def test_global_set_fit_range_display_updates_spinboxes(tab: GlobalFitTab) -> None:
    tab.set_fit_range_display(0.15, 7.50)
    assert abs(tab._fit_range_min_spin.value() - 0.15) < 1e-6
    assert abs(tab._fit_range_max_spin.value() - 7.50) < 1e-6


def test_global_set_fit_range_display_does_not_emit_signal(tab: GlobalFitTab) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.20, 8.00)
    assert emitted == [], "set_fit_range_display must not emit fit_range_edit_committed"


def test_global_set_fit_range_display_none_disables_spinboxes(tab: GlobalFitTab) -> None:
    tab.set_fit_range_display(None, None)
    assert not tab._fit_range_min_spin.isEnabled()
    assert not tab._fit_range_max_spin.isEnabled()


def test_global_spinbox_commit_emits_signal(tab: GlobalFitTab, qapp: QApplication) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_min_spin.setValue(0.25)
    tab._fit_range_min_spin.editingFinished.emit()
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.25) < 1e-6
    assert abs(emitted[0][1] - 5.00) < 1e-6


def test_global_spinbox_commit_emits_both_bounds(tab: GlobalFitTab, qapp: QApplication) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_max_spin.setValue(6.00)
    tab._fit_range_max_spin.editingFinished.emit()
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.10) < 1e-6
    assert abs(emitted[0][1] - 6.00) < 1e-6


# ── Integration tests via MainWindow (Batch/Global tab) ───────────────────────


def _global_tab(win: MainWindow) -> GlobalFitTab:
    """Return the GlobalFitTab ("Batch" tab) from the FitPanel wrapper."""
    return win._fit_panel._global_tab


def test_plot_fit_range_change_updates_global_tab_spinboxes(
    win: MainWindow, qapp: QApplication
) -> None:
    win._plot_panel.set_fit_range(0.30, 6.50)
    qapp.processEvents()
    gt = _global_tab(win)
    assert abs(gt._fit_range_min_spin.value() - 0.30) < 1e-3
    assert abs(gt._fit_range_max_spin.value() - 6.50) < 1e-3


def test_global_tab_spinbox_commit_updates_plot_fit_range(
    win: MainWindow, qapp: QApplication
) -> None:
    win._plot_panel.set_fit_range(0.10, 5.00)
    qapp.processEvents()
    gt = _global_tab(win)
    gt._fit_range_min_spin.setValue(0.40)
    gt._fit_range_min_spin.editingFinished.emit()
    qapp.processEvents()
    lo, _hi = win._plot_panel.get_fit_range()
    assert lo is not None and abs(lo - 0.40) < 1e-3
