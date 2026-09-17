"""The Batch tab's fit window belongs to its series, not to the plot (D8).

``SingleFitTab``'s spinbox <-> plot round-trip is covered by
``tests/test_fit_panel_phase5_range.py`` and is unchanged: the Single tab edits
the project-wide range the plot owns. ``GlobalFitTab`` shares the same
``FloatLimitField`` widgets but no longer shares that range — its window is the
open series' ``recipe["fit_range"]`` (docs/plans/series-workflow.md, D8), so a
commit there reports ``batch_fit_range_changed`` and moves only the plot's range
*guides*, and a plot-side range change never writes back into the tab.
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
    tab.batch_fit_range_changed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.20, 8.00)
    assert emitted == [], "set_fit_range_display must not report a user edit"


def test_global_set_fit_range_display_none_disables_spinboxes(tab: GlobalFitTab) -> None:
    tab.set_fit_range_display(None, None)
    assert not tab._fit_range_min_spin.isEnabled()
    assert not tab._fit_range_max_spin.isEnabled()


def test_global_spinbox_commit_reports_the_batch_window(
    tab: GlobalFitTab, qapp: QApplication
) -> None:
    emitted: list[tuple[float, float]] = []
    tab.batch_fit_range_changed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    # A programmatic setValue commits on its own (commit_on_set_value).
    tab._fit_range_min_spin.setValue(0.25)
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.25) < 1e-6
    assert abs(emitted[0][1] - 5.00) < 1e-6


def test_global_spinbox_commit_reports_both_bounds(tab: GlobalFitTab, qapp: QApplication) -> None:
    emitted: list[tuple[float, float]] = []
    tab.batch_fit_range_changed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_max_spin.setValue(6.00)
    qapp.processEvents()
    assert len(emitted) == 1
    assert abs(emitted[0][0] - 0.10) < 1e-6
    assert abs(emitted[0][1] - 6.00) < 1e-6


def test_global_spinbox_commit_never_moves_the_project_range(
    tab: GlobalFitTab, qapp: QApplication
) -> None:
    """D8: the runs surface does not speak the plot-owned range's signal at all."""
    emitted: list[tuple[float, float]] = []
    tab.fit_range_edit_committed.connect(lambda a, b: emitted.append((a, b)))
    tab.set_fit_range_display(0.10, 5.00)
    tab._fit_range_min_spin.setValue(0.25)
    qapp.processEvents()
    assert emitted == []


# ── Integration tests via MainWindow (Batch tab) ──────────────────────────────


def _global_tab(win: MainWindow) -> GlobalFitTab:
    """Return the GlobalFitTab ("Batch" tab) from the FitPanel wrapper."""
    return win._fit_panel._global_tab


def test_plot_fit_range_change_leaves_the_batch_window_alone(
    win: MainWindow, qapp: QApplication
) -> None:
    gt = _global_tab(win)
    gt.set_fit_range_display(0.20, 8.00)
    win._plot_panel.set_fit_range(0.30, 6.50)
    qapp.processEvents()
    assert abs(gt._fit_range_min_spin.value() - 0.20) < 1e-3
    assert abs(gt._fit_range_max_spin.value() - 8.00) < 1e-3


def test_batch_spinbox_commit_leaves_the_plot_fit_range_alone(
    win: MainWindow, qapp: QApplication
) -> None:
    win._plot_panel.set_fit_range(0.10, 5.00)
    qapp.processEvents()
    gt = _global_tab(win)
    gt._fit_range_min_spin.setValue(0.40)
    gt._fit_range_max_spin.setValue(6.25)
    qapp.processEvents()
    lo, hi = win._plot_panel.get_fit_range()
    assert lo is not None and abs(lo - 0.10) < 1e-3
    assert hi is not None and abs(hi - 5.00) < 1e-3


def test_range_guides_follow_the_visible_tab(win: MainWindow, qapp: QApplication) -> None:
    """The guides show whichever window is being edited, and hand back on exit."""
    win._plot_panel.set_fit_range(0.10, 5.00)
    gt = _global_tab(win)
    win._fit_panel._tabs.setCurrentWidget(gt)
    gt._fit_range_min_spin.setValue(0.40)
    gt._fit_range_max_spin.setValue(6.25)
    qapp.processEvents()
    assert win._plot_panel._fit_guide_window() == pytest.approx((0.40, 6.25))

    win._fit_panel._tabs.setCurrentWidget(win._fit_panel._single_tab)
    qapp.processEvents()
    assert win._plot_panel._fit_guide_window() == pytest.approx((0.10, 5.00))
