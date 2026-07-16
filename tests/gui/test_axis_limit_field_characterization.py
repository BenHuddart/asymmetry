"""Characterization tests for the converged ``FloatLimitField`` widget.

Phase 0 of the shared-foundations audit pinned the *observable* contract of
the two independent ``_FloatLimitField`` implementations
(``asymmetry.gui.panels.fit_panel`` and ``asymmetry.gui.panels.plot_panel``)
before Phase 1a converged them into a single
``asymmetry.gui.widgets.axis_limits.FloatLimitField``:

- The fit_panel variant was the FEATURED one: it clamps to its validator
  range, commits on Return/Enter, and forces a commit on focus-out even for
  "Intermediate" (not-yet-acceptable) input.
- The plot_panel variant was the plain one: no clamping, no commit-on-Return;
  it relied on the base ``QLineEdit`` ``editingFinished`` (which only fires
  for acceptable input).

Phase 1a converged both call sites onto the fit_panel behavior. The tests
below that constructed a "plot_panel-style" field (default decimals=3,
width=76, no ``value_range`` override i.e. the converged class's default
±1000 range) now assert the FEATURED behavior (clamp + commit-on-Return)
instead of its absence — this is the deliberate, expected flip; each test
that changed carries an ``# UPDATED in Phase 1a`` comment explaining the new
assertion. Tests that only ever exercised shared, name-agnostic behavior
(format, decimals, empty-input fallback) are unchanged aside from repointing
imports to the single converged class.
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

from asymmetry.gui.widgets.axis_limits import FloatLimitField

# Both former call-site styles now construct the same converged class; keep
# two names in this file purely to preserve the "fit_panel-style" vs.
# "plot_panel-style" construction grouping below.
FitPanelFloatLimitField = FloatLimitField
PlotPanelFloatLimitField = FloatLimitField


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ─────────────────────────────────────────────────────────────────────────────
# fit_panel-style construction — FloatLimitField() with defaults (clamp + commit)
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


def test_field_rejects_nan_set_value_keeping_last_good_value(qapp: QApplication) -> None:
    """setValue(nan) keeps the last good value — NaN slips through min/max
    clamping, and once resident it round-trips to QSettings at shutdown and
    crashes the next startup in Axes.set_ylim."""
    field = FitPanelFloatLimitField()
    field.setValue(3.5)
    field.setValue(float("nan"))
    assert field.value() == pytest.approx(3.5)
    assert field.text() == "3.500"


def test_field_clamps_infinite_set_value_to_range_ends(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField()
    field.setValue(float("inf"))
    assert field.value() == pytest.approx(1000.0)
    field.setValue(float("-inf"))
    assert field.value() == pytest.approx(-1000.0)


def test_field_constructed_with_nan_falls_back_to_zero(qapp: QApplication) -> None:
    field = FitPanelFloatLimitField(float("nan"))
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
    assert field.is_unset() is True


def test_plain_field_blank_commit_rewrites_stored_value(qapp: QApplication) -> None:
    """A field never given an unset placeholder keeps the legacy revert-on-blank."""
    field = FitPanelFloatLimitField()
    field.setValue(3.5)
    field.clear()
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert field.text() == "3.500"
    assert field.is_unset() is False


def test_unset_capable_field_blank_commit_stays_blank(qapp: QApplication) -> None:
    """Committing a cleared unset-capable field keeps the unset (blank) state.

    ``_normalise_text`` used to rewrite the stored value back into the field
    on every commit, so clearing a field after a ``setValue`` (e.g. a project
    restore) visibly snapped back and "blank = auto" was unreachable.
    """
    field = FitPanelFloatLimitField()
    field.set_unset("Auto")
    field.setValue(2.5)
    assert field.is_unset() is False

    field.clear()
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert field.text() == ""
    assert field.placeholderText() == "Auto"
    assert field.is_unset() is True

    # setValue re-fills the field and leaves the unset state.
    field.setValue(4.0)
    assert field.text() == "4.000"
    assert field.is_unset() is False


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
    true; this field forces the commit regardless so an Intermediate value
    doesn't silently revert on the next external refresh.

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
# plot_panel-style construction — FloatLimitField(value, decimals=..., ...)
# Phase 1a converged this call style onto the same clamp + commit-on-Return
# behavior as the fit_panel style above (per-call-site width/decimals/range
# are still parameters, but the class itself is now one implementation).
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
    field = PlotPanelFloatLimitField(0.0, minimum_width=76)
    assert field.minimumWidth() == 76


def test_plot_panel_field_empty_input_falls_back_to_last_value(qapp: QApplication) -> None:
    field = PlotPanelFloatLimitField(3.5)
    field.clear()
    assert field.value() == pytest.approx(3.5)


def test_plot_panel_field_has_clamp_method(qapp: QApplication) -> None:
    # UPDATED in Phase 1a: the converged field always exposes _clamp, since
    # plot_panel's call sites now share the fit_panel implementation.
    field = PlotPanelFloatLimitField(0.0)
    assert hasattr(field, "_clamp")


def test_plot_panel_field_set_value_clamps_to_its_value_range(qapp: QApplication) -> None:
    # UPDATED in Phase 1a: setValue now clamps to the field's value_range,
    # matching the fit_panel behavior. plot_panel/alc call sites construct
    # with value_range=(-1e6, 1e6) (preserving their historical wide axis
    # range) rather than the converged class's ±1000 default, so exercise
    # that explicitly here alongside the default-range case.
    field = PlotPanelFloatLimitField(0.0)  # default value_range=(-1000, 1000)
    field.setValue(2000.0)
    assert field.value() == pytest.approx(1000.0)
    field.setValue(-5000.0)
    assert field.value() == pytest.approx(-1000.0)

    axis_field = PlotPanelFloatLimitField(0.0, value_range=(-1e6, 1e6))
    axis_field.setValue(2000.0)
    assert axis_field.value() == pytest.approx(2000.0)
    axis_field.setValue(-5000.0)
    assert axis_field.value() == pytest.approx(-5000.0)
    axis_field.setValue(2_000_000.0)
    assert axis_field.value() == pytest.approx(1e6)


def test_plot_panel_field_return_key_commits(qapp: QApplication) -> None:
    # UPDATED in Phase 1a: the converged field now overrides keyPressEvent and
    # commits on Return, matching the fit_panel behavior.
    assert "keyPressEvent" in FloatLimitField.__dict__
    field = PlotPanelFloatLimitField(0.0, value_range=(-1e6, 1e6))
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


def test_plot_panel_field_has_focus_out_override(qapp: QApplication) -> None:
    # UPDATED in Phase 1a: the converged field now overrides focusOutEvent so
    # an Intermediate/unacceptable value typed and left via focus-out DOES
    # force a commit, matching the fit_panel behavior.
    assert "focusOutEvent" in FloatLimitField.__dict__


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
