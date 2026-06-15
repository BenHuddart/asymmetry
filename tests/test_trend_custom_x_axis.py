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


def test_custom_column_offered_in_x_combo_with_label(qapp):
    panel = FitParametersPanel()
    panel.set_custom_x_fields([("Anneal", "custom:abc")])
    assert panel._x_combo.findData("custom:abc") >= 0
    assert panel._x_axis_label_mpl("custom:abc") == "Anneal"


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
