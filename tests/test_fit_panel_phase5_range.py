"""Phase 5a: fit-range spinbox display and bidirectional sync tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fit_panel import SingleFitTab


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tab(qapp: QApplication) -> SingleFitTab:
    return SingleFitTab()


@pytest.fixture
def win(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    w = MainWindow()
    w.show()
    qapp.processEvents()
    return w


def _make_dataset(run_number: int = 1001) -> MuonDataset:
    n = 200
    time = np.linspace(0.0, 3.2, n)
    asym = np.exp(-time / 2.0) * 0.25
    err = np.full(n, 0.01)
    counts = np.array([100.0, 95.0, 90.0, 85.0] * (n // 4), dtype=float)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts, bin_width=0.016),
            Histogram(counts=counts * 0.8, bin_width=0.016),
        ],
        metadata={"run_number": run_number, "field": 100.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
            "deadtime_correction": False,
        },
    )
    return MuonDataset(
        time=time,
        asymmetry=asym,
        error=err,
        run=run,
        metadata={"run_number": run_number, "field": 100.0},
    )


# ── Unit tests on SingleFitTab ────────────────────────────────────────────────


def test_fit_range_spinboxes_exist(tab: SingleFitTab) -> None:
    assert isinstance(tab._fit_range_min_spin, QDoubleSpinBox)
    assert isinstance(tab._fit_range_max_spin, QDoubleSpinBox)


def test_set_fit_range_display_updates_spinboxes(tab: SingleFitTab) -> None:
    tab.set_fit_range_display(0.15, 7.50)
    assert abs(tab._fit_range_min_spin.value() - 0.15) < 1e-6
    assert abs(tab._fit_range_max_spin.value() - 7.50) < 1e-6


def test_set_fit_range_display_does_not_emit_signal(tab: SingleFitTab) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.20, 8.00)
    assert emitted == [], "set_fit_range_display must not emit fit_range_edit_committed"


def test_set_fit_range_display_none_disables_spinboxes(tab: SingleFitTab) -> None:
    tab.set_fit_range_display(None, None)
    assert not tab._fit_range_min_spin.isEnabled()
    assert not tab._fit_range_max_spin.isEnabled()


def test_spinbox_commit_emits_signal(tab: SingleFitTab, qapp: QApplication) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_min_spin.setValue(0.25)
    tab._fit_range_min_spin.editingFinished.emit()
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.25) < 1e-6
    assert abs(emitted[0][1] - 5.00) < 1e-6


def test_spinbox_commit_emits_both_bounds(tab: SingleFitTab, qapp: QApplication) -> None:
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_max_spin.setValue(6.00)
    tab._fit_range_max_spin.editingFinished.emit()
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.10) < 1e-6
    assert abs(emitted[0][1] - 6.00) < 1e-6


# ── Integration tests via MainWindow ─────────────────────────────────────────


def _single_tab(win: MainWindow) -> SingleFitTab:
    """Return the SingleFitTab from the FitPanel wrapper."""
    return win._fit_panel._single_tab


def test_plot_fit_range_change_updates_panel_spinboxes(win: MainWindow, qapp: QApplication) -> None:
    win._plot_panel.set_fit_range(0.30, 6.50)
    qapp.processEvents()
    st = _single_tab(win)
    assert abs(st._fit_range_min_spin.value() - 0.30) < 1e-3
    assert abs(st._fit_range_max_spin.value() - 6.50) < 1e-3


def test_panel_spinbox_commit_updates_plot_fit_range(win: MainWindow, qapp: QApplication) -> None:
    win._plot_panel.set_fit_range(0.10, 5.00)
    qapp.processEvents()
    st = _single_tab(win)
    st._fit_range_min_spin.setValue(0.40)
    st._fit_range_min_spin.editingFinished.emit()
    qapp.processEvents()
    lo, _hi = win._plot_panel.get_fit_range()
    assert lo is not None and abs(lo - 0.40) < 1e-3


def test_no_feedback_loop_on_spinbox_commit(win: MainWindow, qapp: QApplication) -> None:
    """Committing a spinbox should not cause multiple extra emissions."""
    emission_count: list[int] = [0]

    def _count(_a: float, _b: float) -> None:
        emission_count[0] += 1

    win._fit_panel._single_tab.fit_range_edit_committed.connect(_count)
    win._plot_panel.set_fit_range(0.10, 5.00)
    qapp.processEvents()
    st = _single_tab(win)
    st._fit_range_max_spin.setValue(7.00)
    st._fit_range_max_spin.editingFinished.emit()
    qapp.processEvents()
    # One emission from the user commit; the plot's response must not re-emit
    assert emission_count[0] == 1


def test_dataset_switch_refreshes_fit_range_display(win: MainWindow, qapp: QApplication) -> None:
    win._plot_panel.set_fit_range(0.20, 5.00)
    qapp.processEvents()
    dataset = _make_dataset(9901)
    win._data_browser.add_dataset(dataset)
    win._data_browser.dataset_selected.emit(9901)
    qapp.processEvents()
    st = _single_tab(win)
    assert abs(st._fit_range_min_spin.value() - 0.20) < 1e-3
    assert abs(st._fit_range_max_spin.value() - 5.00) < 1e-3
