"""Density regressions for the fit panel on a 13-inch screen.

Covers the F3 GUI-review findings, as the Fit tab refresh now answers them:

* **P1-2** — the model actions are one wrapping row (``Edit…`` · ``Wizard…``)
  rather than a stack of full-width buttons, so the PARAMETERS table and the
  Fit button stay above the fold. The hand-offs that used to hide in a
  "⋯ More…" overflow menu are the results card's own buttons; ``Drop
  background`` is retired (the term is removed in the function editor).
* **P1-4** — the resting parameter table fits the inspector dock without a
  horizontal scrollbar, and the tab's minimum width is set by the fit-range
  row rather than by the table.
* **P1-5** — the fit wizards must open no larger than the available screen.
* **P2-3** — ``ModelFitDialog`` must size against the available screen.
* **P3-3** — the trend panel must show a "load a batch series" hint while empty.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QToolButton

from asymmetry.gui.panels.fit import global_tab as global_tab_module
from asymmetry.gui.panels.fit.single_tab import (
    ADD_TO_SERIES_ACTION,
    DIAGNOSTIC_ACTION,
    SEND_TO_BATCH_ACTION,
)
from asymmetry.gui.panels.fit_panel import GlobalFitTab, SingleFitTab
from asymmetry.gui.styles.metrics import char_width

_DECK_DEFAULT_WIDTH = 360  # INSPECTOR_DOCK_DEFAULT_WIDTH

#: The inspector dock the Fit tab is designed against, in characters (~320 px).
#: The resting column set is 41 characters wide, so the tab needs that plus its
#: own layout margins and the table's frame.
_DOCK_CHARS = 46

#: Labels that must not reappear as buttons or menu entries on the Single tab.
_RETIRED_LABELS = {"Drop background", "More…", "Send to Batch", "Add to Series..."}


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _find_button(widget, label: str) -> QPushButton:
    for btn in widget.findChildren(QPushButton):
        if btn.text() == label:
            return btn
    raise AssertionError(f"{label!r} button not found")


def test_retired_model_actions_are_gone_from_the_single_tab(app):
    """`More…` and the actions it hid are no longer on the tab at all."""
    tab = SingleFitTab()
    try:
        labels = {b.text() for b in tab.findChildren(QPushButton)}
        labels |= {b.text() for b in tab.findChildren(QToolButton)}
        labels |= {a.text() for m in tab.findChildren(QMenu) for a in m.actions()}
        assert not (_RETIRED_LABELS & labels), _RETIRED_LABELS & labels

        # The model row is `Edit…` then `Wizard…`; the hand-offs live on the card.
        assert {"Edit…", "Wizard…"} <= {b.text() for b in tab.findChildren(QPushButton)}
        assert set(tab._results_card._actions) == {
            DIAGNOSTIC_ACTION,
            ADD_TO_SERIES_ACTION,
            SEND_TO_BATCH_ACTION,
        }
    finally:
        tab.close()
        tab.deleteLater()


def test_model_row_wraps_instead_of_stacking(app):
    """`Edit…` and `Wizard…` share one row, so PARAMETERS keeps its height."""
    tab = SingleFitTab()
    try:
        tab.resize(_DECK_DEFAULT_WIDTH, 1200)
        tab.show()
        app.processEvents()

        edit = _find_button(tab, "Edit…")
        wizard = _find_button(tab, "Wizard…")
        assert (
            edit.mapTo(tab, edit.rect().topLeft()).y()
            == wizard.mapTo(tab, wizard.rect().topLeft()).y()
        ), "the model actions should sit on one row at the deck's width"

        fit_btn = _find_button(tab, "Fit")
        param_top = tab._param_table.mapTo(tab, tab._param_table.rect().topLeft()).y()
        assert param_top < fit_btn.mapTo(tab, fit_btn.rect().bottomLeft()).y()
    finally:
        tab.close()
        tab.deleteLater()


def test_resting_parameter_table_does_not_scroll_sideways_in_the_dock(app):
    """At rest the table shows Name·Value·Fix·Min·Max without a scrollbar.

    The tab's *minimum* width stays under ``char_width(40)`` — the fit-range
    row, not the table, is what the dock can never be narrower than.
    """
    settings = QSettings("AsymmetryTest", "FitPanelDensity")
    settings.clear()
    tab = SingleFitTab(settings=settings)
    try:
        tab.resize(char_width(_DOCK_CHARS), 1200)
        tab.show()
        app.processEvents()

        table = tab._param_table
        assert table.column_group_visible("bounds") is True
        assert table.column_group_visible("links") is False
        assert table.column_group_visible("batch") is False
        assert table.horizontalScrollBar().maximum() == 0
        assert tab.minimumSizeHint().width() <= char_width(40)
    finally:
        tab.close()
        tab.deleteLater()
        settings.clear()


def test_batch_tab_parameter_tables_do_not_scroll_sideways_in_the_dock(app):
    """The Batch tab's tables fit the same dock as the Single tab's.

    Parameter · Seed · Type is what rests visible; the rail's ``Bounds`` chip is
    off, so Min and Max are not part of the resting budget.
    """
    settings = QSettings("AsymmetryTest", "BatchPanelDensity")
    settings.clear()
    tab = GlobalFitTab(member_kind="runs", settings=settings)
    try:
        tab.resize(char_width(_DOCK_CHARS), 1200)
        tab.show()
        app.processEvents()

        assert tab._param_table.isColumnHidden(global_tab_module._COL_MIN)
        assert tab._param_table.isColumnHidden(global_tab_module._COL_MAX)
        assert tab._param_table.horizontalScrollBar().maximum() == 0
        assert tab.minimumSizeHint().width() <= char_width(_DOCK_CHARS)
    finally:
        tab.close()
        tab.deleteLater()
        settings.clear()


def _available_size():
    screen = QGuiApplication.primaryScreen()
    geo = screen.availableGeometry()
    return geo.width(), geo.height()


def test_fit_wizard_window_fits_available_screen(app):
    from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

    avail_w, avail_h = _available_size()
    win = FitWizardWindow()
    try:
        assert win.height() <= avail_h, (win.height(), avail_h)
        assert win.width() <= avail_w, (win.width(), avail_w)
        # Never larger than the preferred 13-inch-friendly cap.
        assert win.height() <= 740
        assert win.width() <= 1180
    finally:
        win.close()
        win.deleteLater()


def test_global_fit_wizard_window_fits_available_screen(app):
    from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

    avail_w, avail_h = _available_size()
    win = GlobalFitWizardWindow()
    try:
        assert win.height() <= avail_h, (win.height(), avail_h)
        assert win.width() <= avail_w, (win.width(), avail_w)
        assert win.height() <= 740
        assert win.width() <= 1180
    finally:
        win.close()
        win.deleteLater()


def test_model_fit_dialog_fits_available_screen(app):
    from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog

    avail_w, avail_h = _available_size()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.8, 0.6, 0.5])
    yerr = np.array([0.05, 0.05, 0.05, 0.05])
    dlg = ModelFitDialog("lambda", "field", x, y, yerr)
    try:
        assert dlg.height() <= avail_h, (dlg.height(), avail_h)
        assert dlg.width() <= avail_w, (dlg.width(), avail_w)
    finally:
        dlg.close()
        dlg.deleteLater()


def test_trend_panel_empty_state_hint_toggles_with_rows(app):
    """The trend panel shows a 'load a batch series' hint until rows arrive."""
    from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow

    panel = FitParametersPanel()
    try:
        # Empty on construction → the hint is visible.
        assert panel._empty_state_hint.isVisibleTo(panel)

        panel._rows = [
            _FitRow(
                run_number=1,
                run_label="1",
                field=100.0,
                temperature=10.0,
                values={"A0": 0.2},
                errors={"A0": 0.01},
            )
        ]
        panel._varying_params = ["A0"]
        panel._refresh_plot()
        # Rows present → the hint hides.
        assert not panel._empty_state_hint.isVisibleTo(panel)

        panel.clear()
        # Cleared → the hint returns.
        assert panel._empty_state_hint.isVisibleTo(panel)
    finally:
        panel.close()
        panel.deleteLater()
