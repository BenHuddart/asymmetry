"""Vertical-density regressions for the fit panel on a 13-inch screen.

Covers the F3 GUI-review findings:

* **P1-2** — the advanced model actions (Drop background / Send to Batch /
  Add to Series) must be folded into a single "⋯ More…" overflow menu so the
  PARAMETERS table and the Fit button rise into view instead of falling below
  the fold. ("Share with Group" was a fourth folded action here until D5
  (`docs/studies/datagroup-fitseries-unification.md`) retired it in favour of
  refresh-unless-fitted carry-forward.)
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

_ADVANCED_ACTION_LABELS = {
    "Drop background",
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


def _model_button_layout(tab):
    """The QVBoxLayout holding Edit Function / Fit Wizard / More…."""
    from PySide6.QtWidgets import QVBoxLayout

    for layout in tab.findChildren(QVBoxLayout):
        if layout.indexOf(tab._more_btn) >= 0:
            return layout
    raise AssertionError("model button layout not found")


def test_overflow_menu_lifts_fit_button_vs_inline_buttons(app):
    """Folding the four advanced actions lifts PARAMETERS + Fit by ~3 button rows.

    Asserted as a *relative* lift measured in the same environment (folded vs. a
    reconstructed pre-fix inline layout), so it is independent of per-platform
    font metrics — an absolute pixel budget is not portable between Windows and
    the Linux CI runner.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton

    tab = SingleFitTab()
    try:
        tab.resize(_DECK_DEFAULT_WIDTH, 1200)
        tab.show()
        app.processEvents()

        fit_btn = _find_button(tab, "Fit")
        param_top = tab._param_table.mapTo(tab, tab._param_table.rect().topLeft()).y()
        folded_bottom = fit_btn.mapTo(tab, fit_btn.rect().bottomLeft()).y()
        # The parameter table sits above the Fit button (ordering sanity).
        assert param_top < folded_bottom

        # Reconstruct the pre-fix layout: the four advanced actions as button
        # rows below the existing controls. The Fit button must drop by roughly
        # the height of those four rows — i.e. folding lifts it by ≥3 rows.
        layout = _model_button_layout(tab)
        extras = [QPushButton(f"advanced {i}") for i in range(4)]
        for btn in extras:
            layout.addWidget(btn, 0, Qt.AlignmentFlag.AlignLeft)
        app.processEvents()

        inline_bottom = fit_btn.mapTo(tab, fit_btn.rect().bottomLeft()).y()
        row_height = extras[0].sizeHint().height()
        assert folded_bottom <= inline_bottom - 3 * row_height, (
            f"folding saved only {inline_bottom - folded_bottom}px; expected "
            f"≥3 button rows (~{3 * row_height}px). The overflow menu is not "
            "lifting PARAMETERS/Fit as intended (P1-2)."
        )
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
        # In the frequency domain there is nothing to drop.
        assert not tab._drop_background_action.isEnabled()
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
