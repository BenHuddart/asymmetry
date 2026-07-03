"""Characterization tests for the two ``_FloatLimitField`` implementations.

Phase 0 of the shared-foundations audit: pin the *observable* contract of both
variants before Phase 1a converges them into a single ``FloatLimitField``.

- ``asymmetry.gui.panels.fit_panel._FloatLimitField`` (the FEATURED variant):
  clamps to its validator range, commits on Return/Enter, and forces a commit
  on focus-out even for "Intermediate" (not-yet-acceptable) input.
- ``asymmetry.gui.panels.plot_panel._FloatLimitField`` (the plain variant):
  no clamping, no commit-on-Return; relies on the base ``QLineEdit``
  ``editingFinished`` (which only fires for acceptable input).

Per PLAN.md Phase 1a, ``plot_panel``'s field is expected to GAIN clamping and
commit-on-Return, converging on the fit_panel behavior (with per-call-site
width/decimals preserved as parameters). Tests that pin plot_panel's field as
lacking those two behaviors are marked ``_current_behavior`` with an
EXPECTED-TO-CHANGE comment so Phase 1a can update them knowingly instead of
misreading a red test as a regression. The fit_panel-variant tests (and any
shared-contract assertions) are NOT marked and must stay green throughout.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from asymmetry.gui.panels.fit_panel import _FloatLimitField as FitPanelFloatLimitField
from asymmetry.gui.panels.plot_panel import _FloatLimitField as PlotPanelFloatLimitField


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ─────────────────────────────────────────────────────────────────────────────
# fit_panel._FloatLimitField — the FEATURED variant (clamp + commit-on-Return)
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_panel_field_default_value_and_format(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    assert field.value() == pytest.approx(0.0)
    assert field.decimals() == 3
    assert field.text() == "0.000"


def test_fit_panel_field_set_value_formats_to_decimals(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    field.setValue(1.5)
    assert field.text() == "1.500"
    assert field.value() == pytest.approx(1.5)


def test_fit_panel_field_set_decimals_reformats_current_value(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    field.setValue(1.23456)
    field.setDecimals(1)
    assert field.decimals() == 1
    assert field.text() == "1.2"


def test_fit_panel_field_clamps_set_value_to_validator_range(qapp: QApplication) -> None:
    """setValue clamps to the validator's [-1000, 1000] default range."""
    field = FitPanelFloatLimitField()
    field.setValue(2000.0)
    assert field.value() == pytest.approx(1000.0)
    field.setValue(-2000.0)
    assert field.value() == pytest.approx(-1000.0)


def test_fit_panel_field_set_range_narrows_clamp(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    field.setRange(0.0, 10.0)
    field.setValue(50.0)
    assert field.value() == pytest.approx(10.0)
    field.setValue(-5.0)
    assert field.value() == pytest.approx(0.0)


def test_fit_panel_field_empty_input_falls_back_to_last_value(qapp: QApplication) -> None:
    """A blank field (unparsable text) returns the last committed value()."""
    field = FitPanelFloatLimitField()
    field.setValue(3.5)
    field.clear()
    assert field.value() == pytest.approx(3.5)


def test_fit_panel_field_set_unset_shows_placeholder_and_clears_text(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    field.setValue(3.5)
    field.set_unset("full spectrum")
    assert field.text() == ""
    assert field.placeholderText() == "full spectrum"


def test_fit_panel_field_return_key_commits_typed_value(qapp: QApplication) -> None:
    """Typing a value and pressing Return commits it (Round-10 #9 regression)."""
    field = FitPanelFloatLimitField()
    field.setValue(5.0)
    emitted: list[float] = []
    field.editingFinished.connect(lambda: emitted.append(field.value()))

    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "7.25")
    QTest.keyClick(field, Qt.Key.Key_Return)
    qapp.processEvents()

    assert emitted, "Return did not fire a commit"
    assert emitted[-1] == pytest.approx(7.25)
    assert field.value() == pytest.approx(7.25)


def test_fit_panel_field_return_key_is_consumed(qapp: QApplication) -> None:
    """Return must be consumed (accepted) so it cannot bubble to a default button."""
    field = FitPanelFloatLimitField()
    field.setValue(5.0)
    emitted: list[float] = []
    field.editingFinished.connect(lambda: emitted.append(field.value()))

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    field.keyPressEvent(event)

    assert event.isAccepted(), "Return must be consumed, not propagated"
    assert emitted == [pytest.approx(5.0)]


def test_fit_panel_field_enter_key_also_commits(qapp: QApplication) -> None:
    """The numpad Enter key commits identically to Return."""
    field = FitPanelFloatLimitField()
    field.setValue(5.0)
    emitted: list[float] = []
    field.editingFinished.connect(lambda: emitted.append(field.value()))

    event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Enter, Qt.KeyboardModifier.NoModifier)
    field.keyPressEvent(event)

    assert event.isAccepted()
    assert emitted == [pytest.approx(5.0)]


def test_fit_panel_field_focus_out_forces_commit_for_intermediate_input(
    qapp: QApplication,
) -> None:
    """An out-of-range ('Intermediate') value committed via focus-out still fires.

    A bare QLineEdit only emits editingFinished when hasAcceptableInput() is
    true; the fit_panel variant forces the commit regardless so an
    Intermediate value doesn't silently revert on the next external refresh.

    Offscreen platforms don't reliably deliver a real window-focus-out via
    clearFocus(), so drive the Qt override directly with a synthetic
    QFocusEvent, matching how test_fit_panel_phase5_range.py drives
    keyPressEvent directly for the analogous Return-key contract.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QFocusEvent

    field = FitPanelFloatLimitField()
    field.setValue(5.0)
    emitted: list[float] = []
    field.editingFinished.connect(lambda: emitted.append(field.value()))

    # 5000 exceeds the default validator top (1000) -> Intermediate, not Acceptable.
    field.setText("5000")
    assert not field.hasAcceptableInput()

    event = QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason)
    field.focusOutEvent(event)

    assert emitted, "focus-out did not force a commit for Intermediate input"
    assert emitted[-1] == pytest.approx(1000.0), "value must be clamped on forced commit"


# ─────────────────────────────────────────────────────────────────────────────
# plot_panel._FloatLimitField — the plain variant (no clamp, no commit-on-Return)
# ─────────────────────────────────────────────────────────────────────────────


def test_plot_panel_field_default_value_and_format(qapp: QApplication) -> None:
    field = PlotPanelFloatLimitField(2.5)
    assert field.value() == pytest.approx(2.5)
    assert field.text() == "2.500"


def test_plot_panel_field_set_value_formats_to_decimals(qapp: QApplication) -> None:
    field = PlotPanelFloatLimitField(0.0, decimals=1)
    field.setValue(1.23456)
    assert field.text() == "1.2"
    assert field.value() == pytest.approx(1.2)


def test_plot_panel_field_default_width_is_76px(qapp: QApplication) -> None:
    field = PlotPanelFloatLimitField(0.0)
    assert field.minimumWidth() == 76


def test_plot_panel_field_empty_input_falls_back_to_last_value(qapp: QApplication) -> None:
    field = PlotPanelFloatLimitField(3.5)
    field.clear()
    assert field.value() == pytest.approx(3.5)


def test_plot_panel_field_has_no_clamp_method_current_behavior(qapp: QApplication) -> None:
    # EXPECTED-TO-CHANGE in Phase 1a: plot_panel field gains clamp + commit-on-Return.
    # Unlike the fit_panel variant, this field exposes no _clamp helper at all.
    field = PlotPanelFloatLimitField(0.0)
    assert not hasattr(field, "_clamp")


def test_plot_panel_field_set_value_does_not_clamp_current_behavior(qapp: QApplication) -> None:
    # EXPECTED-TO-CHANGE in Phase 1a: plot_panel field gains clamp + commit-on-Return.
    # setValue accepts values far outside any fit-range-sized bound unclamped;
    # only the QDoubleValidator's very wide (-1e6, 1e6) range would reject a
    # *keystroke*, and setValue()/programmatic writes bypass the validator
    # entirely.
    field = PlotPanelFloatLimitField(0.0)
    field.setValue(2000.0)
    assert field.value() == pytest.approx(2000.0)
    field.setValue(-5000.0)
    assert field.value() == pytest.approx(-5000.0)


def test_plot_panel_field_return_key_does_not_force_commit_current_behavior(
    qapp: QApplication,
) -> None:
    # EXPECTED-TO-CHANGE in Phase 1a: plot_panel field gains clamp + commit-on-Return.
    # There is no keyPressEvent override, so Return only commits when Qt's
    # default QLineEdit behavior judges the input Acceptable AND a default
    # button/return-triggered slot doesn't consume it first. We characterize
    # the narrower, verifiable fact: the class defines no keyPressEvent
    # override at all (behavior is entirely inherited from QLineEdit).
    assert "keyPressEvent" not in PlotPanelFloatLimitField.__dict__


def test_plot_panel_field_has_no_focus_out_override_current_behavior(
    qapp: QApplication,
) -> None:
    # EXPECTED-TO-CHANGE in Phase 1a: plot_panel field gains clamp + commit-on-Return.
    # No focusOutEvent override means an Intermediate/unacceptable value typed
    # and left via focus-out does NOT force a commit (unlike fit_panel's field).
    assert "focusOutEvent" not in PlotPanelFloatLimitField.__dict__


def test_plot_panel_field_editing_finished_fires_for_acceptable_input(
    qapp: QApplication,
) -> None:
    """Baseline (unmarked): acceptable input still commits via the base QLineEdit."""
    field = PlotPanelFloatLimitField(0.0)
    emitted: list[float] = []
    field.editingFinished.connect(lambda: emitted.append(field.value()))

    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "4.5")
    QTest.keyClick(field, Qt.Key.Key_Return)
    qapp.processEvents()

    assert emitted, "Return with acceptable input should still commit via base QLineEdit"
    assert emitted[-1] == pytest.approx(4.5)
