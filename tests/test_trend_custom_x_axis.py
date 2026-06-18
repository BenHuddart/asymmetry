"""Custom data-browser columns selectable as the parameter-trend x-axis (M4)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _row(run_number: int, custom: dict[str, str]) -> _FitRow:
    return _FitRow(
        run_number=run_number,
        run_label=str(run_number),
        field=100.0 + run_number,
        temperature=10.0,
        values={"A0": 0.2},
        errors={"A0": 0.01},
        custom_values=dict(custom),
    )


def test_x_value_coerces_custom_numeric(qapp):
    panel = FitParametersPanel()
    row = _row(1, {"custom:abc": "12.5"})
    assert panel._x_value(row, "custom:abc") == pytest.approx(12.5)


def test_x_value_nan_for_empty_or_non_numeric(qapp):
    panel = FitParametersPanel()
    assert np.isnan(panel._x_value(_row(1, {"custom:abc": ""}), "custom:abc"))
    assert np.isnan(panel._x_value(_row(2, {"custom:abc": "annealed"}), "custom:abc"))
    assert np.isnan(panel._x_value(_row(3, {}), "custom:abc"))


def test_x_value_inf_string_is_dropped(qapp):
    panel = FitParametersPanel()
    assert np.isnan(panel._x_value(_row(1, {"custom:abc": "inf"}), "custom:abc"))
    assert np.isnan(panel._x_value(_row(2, {"custom:abc": "-inf"}), "custom:abc"))


def test_custom_column_offered_in_x_combo_with_label(qapp):
    panel = FitParametersPanel()
    panel.set_custom_x_fields([("Anneal", "custom:abc")])
    assert panel._x_combo.findData("custom:abc") >= 0
    assert panel._x_axis_label_mpl("custom:abc") == "Anneal"


def test_selecting_custom_column_sets_effective_x_key(qapp):
    # Regression: _effective_x_key must translate a custom: combo selection into
    # the custom key, otherwise the trend silently plots against the inferred
    # field/temperature/run axis (the whole feature is a no-op).
    panel = FitParametersPanel()
    panel.set_custom_x_fields([("Anneal", "custom:abc")])
    idx = panel._x_combo.findData("custom:abc")
    panel._x_combo.setCurrentIndex(idx)
    assert panel._effective_x_key() == "custom:abc"


def test_skip_note_counts_dropped_runs(qapp):
    panel = FitParametersPanel()
    # 2 of 3 runs have empty/non-numeric values for this custom column.
    panel._update_custom_x_skip_note("custom:abc", np.array([12.0, np.nan, np.nan]))
    assert "2/3" in panel._x_auto_hint.text()
    # All numeric → no note.
    panel._update_custom_x_skip_note("custom:abc", np.array([1.0, 2.0, 3.0]))
    assert panel._x_auto_hint.text() == ""
    # A built-in axis never gets a custom-skip note.
    panel._x_auto_hint.setText("(B)")
    panel._update_custom_x_skip_note("field", np.array([np.nan]))
    assert panel._x_auto_hint.text() == "(B)"


def test_custom_values_round_trip_through_state(qapp):
    panel = FitParametersPanel()
    panel._rows = [_row(1, {"custom:abc": "12.5"}), _row(2, {"custom:abc": ""})]
    panel._varying_params = ["A0"]

    state = panel.get_state()
    assert state["rows"][0]["custom_values"] == {"custom:abc": "12.5"}

    restored = FitParametersPanel()
    restored.restore_state(state)
    assert restored._rows[0].custom_values == {"custom:abc": "12.5"}
    assert restored._x_value(restored._rows[0], "custom:abc") == pytest.approx(12.5)


# --- First-class Angle (°) x-axis --------------------------------------------


def test_angle_axis_is_first_class_after_run(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    # Listed with the fixed run-level axes, right after "Run".
    fixed = [panel._x_combo.itemText(i) for i in range(5)]
    assert fixed == ["Auto", "𝐵 (G)", "𝑇 (K)", "Run", "Angle (°)"]
    assert panel._x_combo.itemData(4) == "angle"
    assert panel._x_axis_label_mpl("angle") == "Angle (°)"


def test_angle_axis_resolves_angle_values_not_run_number(qapp):
    # The Angle key has no "custom:" prefix; it must resolve from the row's
    # per-run angle value, not silently fall through to the run number.
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    assert panel._x_value(_row(7, {"angle": "30"}), "angle") == pytest.approx(30.0)
    assert panel._x_value(_row(7, {"angle": "-45.5"}), "angle") == pytest.approx(-45.5)
    # Empty / non-numeric / non-finite all drop to NaN (not the run number).
    assert np.isnan(panel._x_value(_row(7, {"angle": ""}), "angle"))
    assert np.isnan(panel._x_value(_row(7, {"angle": "tilt"}), "angle"))
    assert np.isnan(panel._x_value(_row(7, {"angle": "inf"}), "angle"))


def test_selecting_angle_axis_sets_effective_x_key(qapp):
    # Regression: _effective_x_key must translate an Angle combo selection into
    # the "angle" key. The Angle item data has no "param:"/"custom:" prefix, so
    # without an explicit match it silently falls through to the inferred
    # field/temperature/run axis and the angle axis becomes a no-op (the bug
    # observed in grouped-fit trending).
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    idx = panel._x_combo.findData("angle")
    panel._x_combo.setCurrentIndex(idx)
    assert panel._effective_x_key() == "angle"


def test_angle_axis_gets_skip_note(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    panel._update_custom_x_skip_note("angle", np.array([0.0, np.nan, 30.0]))
    assert "1/3" in panel._x_auto_hint.text()


def test_clearing_angle_field_removes_it_from_combo(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    assert panel._x_combo.findData("angle") >= 0
    panel.set_angle_x_field(None)
    assert panel._x_combo.findData("angle") < 0


# --- Angle folding / periodicity (Phase 4) -----------------------------------


def test_angle_fold_folds_x_value(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    panel._angle_fold_combo.setCurrentIndex(panel._angle_fold_combo.findData(180.0))
    assert panel._x_value(_row(1, {"angle": "190"}), "angle") == pytest.approx(10.0)
    assert panel._x_value(_row(2, {"angle": "-10"}), "angle") == pytest.approx(170.0)
    # A generic custom column is never folded, even with the same numeric value.
    panel.set_custom_x_fields([("Anneal", "custom:abc")])
    assert panel._x_value(_row(3, {"custom:abc": "190"}), "custom:abc") == pytest.approx(190.0)


def test_angle_fold_off_returns_raw_value(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    assert panel._x_value(_row(1, {"angle": "190"}), "angle") == pytest.approx(190.0)


def test_fold_control_visible_only_for_angle_axis(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    assert panel._angle_fold_combo.isHidden()  # default Auto axis
    panel._x_combo.setCurrentIndex(panel._x_combo.findData("angle"))
    assert not panel._angle_fold_combo.isHidden()


def test_clear_resets_angle_fold(qapp):
    # New Project (clear) must not leak the fold into the next project.
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    panel._angle_fold_combo.setCurrentIndex(panel._angle_fold_combo.findData(180.0))
    assert panel._angle_wrap_period == 180.0
    panel.clear()
    assert panel._angle_wrap_period is None
    assert panel._angle_fold_combo.currentData() is None


def test_angle_fold_round_trips_through_state(qapp):
    panel = FitParametersPanel()
    panel.set_angle_x_field(("Angle (°)", "angle"))
    panel._angle_fold_combo.setCurrentIndex(panel._angle_fold_combo.findData(360.0))
    state = panel.get_state()
    assert state["angle_wrap_period"] == 360.0

    restored = FitParametersPanel()
    restored.set_angle_x_field(("Angle (°)", "angle"))
    restored.restore_state(state)
    assert restored._angle_wrap_period == 360.0
    assert restored._angle_fold_combo.currentData() == 360.0
