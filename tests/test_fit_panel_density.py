"""Vertical-density regressions for the fit panel on a 13-inch screen.

Covers the F3 GUI-review findings:

* **P1-2** — the four advanced model actions (Drop background / Share with Group
  / Send to Batch / Add to Series) must be folded into a single "⋯ More…"
  overflow menu so the PARAMETERS table and the Fit button rise into view instead
  of falling below the fold.
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

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from asymmetry.gui.panels.fit_panel import SingleFitTab

_DECK_DEFAULT_WIDTH = 360  # INSPECTOR_DOCK_DEFAULT_WIDTH

#: A representative content height for the right inspector dock on a 960×640 /
#: 1280×820 "small screen" case (the screen height minus the menu/tool/status
#: chrome and the tab bar). The MODEL controls + PARAMETERS table + **Fit**
#: button must fit within this so the primary action is reachable without
#: scrolling. Pre-fix the four advanced button rows pushed the Fit button to
#: ~403 px; folding them into the overflow menu lifts it to ~330 px.
_REPRESENTATIVE_DOCK_CONTENT_HEIGHT = 360

_ADVANCED_ACTION_LABELS = {
    "Drop background",
    "Share with Group",
    "Send to Batch",
    "Add to Series...",
}


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _find_button(widget, label: str) -> QPushButton:
    for btn in widget.findChildren(QPushButton):
        if btn.text() == label:
            return btn
    raise AssertionError(f"{label!r} button not found")


def test_advanced_actions_folded_into_overflow_menu(app):
    """The four advanced actions live in the More… menu, not as button rows."""
    tab = SingleFitTab()
    try:
        # No standalone push buttons for the advanced actions any more.
        button_labels = {b.text() for b in tab.findChildren(QPushButton)}
        assert not (_ADVANCED_ACTION_LABELS & button_labels), (
            "advanced actions should be menu items, not full-width buttons: "
            f"{_ADVANCED_ACTION_LABELS & button_labels}"
        )

        # They are reachable via the single "More…" overflow tool button.
        more = tab._more_btn
        assert isinstance(more, QToolButton)
        assert more.menu() is not None
        menu_labels = {a.text() for a in more.menu().actions()}
        assert _ADVANCED_ACTION_LABELS <= menu_labels, menu_labels
    finally:
        tab.close()
        tab.deleteLater()


def test_parameters_and_fit_button_reachable_on_small_screen(app):
    """At the deck width the param table + Fit button stay above the fold."""
    tab = SingleFitTab()
    try:
        tab.resize(_DECK_DEFAULT_WIDTH, 1200)
        tab.show()
        app.processEvents()

        param_top = tab._param_table.mapTo(tab, tab._param_table.rect().topLeft()).y()
        fit_btn = _find_button(tab, "Fit")
        fit_bottom = fit_btn.mapTo(tab, fit_btn.rect().bottomLeft()).y()

        assert fit_bottom <= _REPRESENTATIVE_DOCK_CONTENT_HEIGHT, (
            f"Fit button bottom at {fit_bottom}px exceeds the representative dock "
            f"content height {_REPRESENTATIVE_DOCK_CONTENT_HEIGHT}px — it falls "
            "below the fold on a 13-inch screen."
        )
        # The parameter table must sit above the Fit button (sanity on ordering).
        assert param_top < fit_bottom
    finally:
        tab.close()
        tab.deleteLater()


def test_advanced_action_enable_state_still_tracks_domain(app):
    """Folding into a menu preserves the enable/disable wiring of the actions."""
    tab = SingleFitTab()
    try:
        # Drop-background is offered for the default time-domain Exp+Const model.
        assert tab._drop_background_action.isEnabled()
        tab.set_domain("frequency")
        # In the frequency domain there is nothing to drop and no group sharing.
        assert not tab._drop_background_action.isEnabled()
        assert not tab._share_group_action.isEnabled()
    finally:
        tab.close()
        tab.deleteLater()


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
