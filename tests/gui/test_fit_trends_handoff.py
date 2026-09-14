"""The results card's ``Trends →`` hand-off (GUI).

Phase 4 of the Fit tab refresh (``docs/plans/fit-tab-refresh.md``): a completed
batch arms the hand-off, and pressing it brings the Parameters dock forward on
that batch's own series rather than leaving the user to find the pill.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from asymmetry.gui.mainwindow import MainWindow  # noqa: E402
from asymmetry.gui.panels.fit.global_tab import TRENDS_ACTION, GlobalFitTab  # noqa: E402

_BATCH_ID = "batch-trends-1"

_ROWS = [
    {
        "run_number": run,
        "run_label": str(run),
        "field": 100.0,
        "temperature": float(run),
        "values": {"Lambda": 0.1 * run},
        "errors": {"Lambda": 0.01},
    }
    for run in (1, 2, 3)
]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_trends_action_rests_disabled_until_a_batch_is_recorded(qapp) -> None:
    tab = GlobalFitTab(member_kind="runs")
    try:
        assert tab._results_card._actions[TRENDS_ACTION].isEnabled() is False
        tab.set_trends_available(True)
        assert tab._results_card._actions[TRENDS_ACTION].isEnabled() is True
    finally:
        tab.close()
        tab.deleteLater()


def test_trends_request_raises_the_parameters_dock_on_the_batch_series(qapp) -> None:
    window = MainWindow()
    try:
        # Stand in for a completed batch: the series the panel holds, and the
        # bookkeeping _on_global_fit_completed does when one is recorded.
        window._fit_parameters_panel.load_representation_series([(_BATCH_ID, "Batch · 1–3", _ROWS)])
        window._remember_trends_batch("batch", _BATCH_ID, window._fit_panel)
        window._parameters_stack.setCurrentWidget(window._alc_analysis_widget)
        window._fit_parameters_panel.select_series([])

        window._fit_panel._global_tab._results_card.action_triggered.emit(TRENDS_ACTION)

        assert window._parameters_stack.currentWidget() is window._fit_parameters_panel
        assert window._fit_parameters_panel._active_group_id == _BATCH_ID
        assert window._fit_panel._global_tab._results_card._actions[TRENDS_ACTION].isEnabled()
    finally:
        window.close()
        window.deleteLater()


def test_a_grouped_single_fit_records_no_batch_and_leaves_trends_disarmed(qapp) -> None:
    """Only a recorded series arms the hand-off — a single grouped fit records none."""
    window = MainWindow()
    try:
        batch_tab = window._multi_group_fit_window._batch_fit_tab
        assert batch_tab._results_card._actions[TRENDS_ACTION].isEnabled() is False

        window._remember_trends_batch("grouped", None, window._multi_group_fit_window)

        assert batch_tab._results_card._actions[TRENDS_ACTION].isEnabled() is False
        assert "grouped" not in window._last_batch_id_by_surface
    finally:
        window.close()
        window.deleteLater()
