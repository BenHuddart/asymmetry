"""Characterization tests for the shared ``AxisLimitControls`` row widget.

Phase 1a extracted the X/Y axis-limit row (min/max fields + Auto X/Y toggles,
optional unit labels) that plot_panel and alc_panel built independently into
one ``asymmetry.gui.widgets.axis_limits.AxisLimitControls`` holder. These
tests pin its construction contract so Phase 2/3 (and future call sites) can
build on it safely: attribute presence, per-call-site parameters (field
width, unit labels, auto default state, initial values, value range), and the
holder-not-wiring behavior (it exposes widgets; callers wire their own
signals).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from asymmetry.gui.widgets.axis_limits import AxisLimitControls, FloatLimitField  # noqa: E402


def test_builds_four_fields_and_two_auto_buttons(qapp: object) -> None:
    controls = AxisLimitControls()
    for name in ("x_min", "x_max", "y_min", "y_max"):
        assert isinstance(getattr(controls, name), FloatLimitField)
    assert controls.auto_x_btn.isCheckable()
    assert controls.auto_y_btn.isCheckable()


def test_show_units_toggles_unit_labels(qapp: object) -> None:
    with_units = AxisLimitControls(show_units=True)
    assert with_units.x_unit_label is not None
    assert with_units.y_unit_label is not None

    without = AxisLimitControls(show_units=False)
    assert without.x_unit_label is None
    assert without.y_unit_label is None


def test_auto_checked_default_state(qapp: object) -> None:
    # alc_panel uses auto_checked=True (auto on by default); plot_panel False.
    on = AxisLimitControls(auto_checked=True)
    assert on.auto_x_btn.isChecked()
    assert on.auto_y_btn.isChecked()

    off = AxisLimitControls(auto_checked=False)
    assert not off.auto_x_btn.isChecked()
    assert not off.auto_y_btn.isChecked()


def test_initial_values_seed_the_fields(qapp: object) -> None:
    controls = AxisLimitControls(initial_values=(0.0, 1.0, 0.0, 1.0))
    assert controls.x_min.value() == 0.0
    assert controls.x_max.value() == 1.0
    assert controls.y_min.value() == 0.0
    assert controls.y_max.value() == 1.0


def test_field_width_and_value_range_are_applied(qapp: object) -> None:
    # A large axis range must NOT be clamped to the fit-field default of +/-1000.
    controls = AxisLimitControls(field_width=64, value_range=(-1e6, 1e6))
    assert controls.x_min.minimumWidth() == 64
    controls.x_max.setValue(5000.0)
    assert controls.x_max.value() == 5000.0


def test_is_a_holder_wiring_none_by_default(qapp: object) -> None:
    # The widget exposes the buttons/fields but wires no external handlers of
    # its own; each panel connects its own slots. Constructing it must not
    # raise even though nothing is connected.
    controls = AxisLimitControls()
    # Toggling the auto button programmatically is safe (no side effects here).
    controls.auto_x_btn.setChecked(True)
    assert controls.auto_x_btn.isChecked()
