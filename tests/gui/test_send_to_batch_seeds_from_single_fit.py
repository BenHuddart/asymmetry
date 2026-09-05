"""B8c: "Send to Batch" must seed the batch from the current single-fit values.

Live-testing finding (Round-3): after fitting a single run with a fixed
frequency (5.397 MHz), clicking "Send to Batch" populated the batch Parameter
Classification with a STALE frequency seed (1.355, a leftover from a previous
analysis) rather than the value the user just set. ``send_single_model_to_batch``
copies only the composite model, never the single tab's current parameter
values, so the batch falls back to defaults / preserved state.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.gui.panels.fit_panel import FitPanel

pytestmark = [pytest.mark.gui]


def _row_for(table, name: str) -> int:
    for r in range(table.rowCount()):
        item = table.item(r, 0)
        if item is None:
            continue
        key = item.data(Qt.ItemDataRole.UserRole)
        if key == name or (not isinstance(key, str) and item.text() == name):
            return r
    raise AssertionError(f"parameter row {name!r} not found")


def test_send_to_batch_carries_single_fit_seed(qapp: QApplication) -> None:
    panel = FitPanel()
    panel._single_tab._set_composite_model(
        CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"])
    )

    # User sets a deliberate, non-default frequency seed in the single-fit table.
    single_tbl = panel._single_tab._param_table
    freq_row = _row_for(single_tbl, "frequency")
    single_tbl.item(freq_row, 1).setText("5.397")

    assert panel.send_single_model_to_batch() is True

    # The batch Parameter-Classification seed (column 1) must reflect 5.397,
    # not the model default (1.0) or any stale leftover.
    batch_tbl = panel._global_tab._param_table
    batch_freq_row = _row_for(batch_tbl, "frequency")
    seed = float(batch_tbl.item(batch_freq_row, 1).text())
    assert seed == pytest.approx(5.397), (
        f"batch frequency seed is {seed}, not the single-fit value 5.397 — "
        "Send-to-Batch did not carry the current seed"
    )


def test_send_to_batch_carries_single_fit_bounds(qapp: QApplication) -> None:
    """Round-10 #9: Send to Batch must carry parameter min/max, not just seeds."""
    panel = FitPanel()
    panel._single_tab._set_composite_model(
        CompositeModel(["Oscillatory", "Gaussian", "Constant"], operators=["*", "+"])
    )

    # User sets an amplitude floor (A_1 >= 1) in the single-fit table: Min=col 3,
    # Max=col 4 on the single tab's parameter table.
    single_tbl = panel._single_tab._param_table
    amp_row = _row_for(single_tbl, "A_1")
    single_tbl.item(amp_row, single_tbl.COL_MIN).setText("1")
    single_tbl.item(amp_row, single_tbl.COL_MAX).setText("50")

    assert panel.send_single_model_to_batch() is True

    # The batch table stores bounds as one "min, max" string in column 3.
    batch_tbl = panel._global_tab._param_table
    batch_amp_row = _row_for(batch_tbl, "A_1")
    bounds_text = batch_tbl.item(batch_amp_row, 3).text()
    lo, hi = (part.strip() for part in bounds_text.split(","))
    assert float(lo) == pytest.approx(1.0), (
        f"batch A_1 lower bound is {lo!r}, not the single-fit floor 1 — "
        "Send-to-Batch dropped the parameter bounds"
    )
    assert float(hi) == pytest.approx(50.0)


def test_send_to_batch_preserves_batch_role_across_resend(qapp: QApplication) -> None:
    """A batch Type override must survive a re-send of the same-named model.

    ``send_single_model_to_batch`` rebuilds the batch table by calling
    ``_set_composite_model`` again; before it passed ``aligned_origins`` this
    reset every row's Type combo to the model's default assignment (Fixed for
    ``VortexLattice``'s ``field``), clobbering a role the user had deliberately
    chosen (``Local``) on the batch surface itself. Aligning by name means the
    role is keyed to the surviving component, not recomputed from scratch.
    """
    panel = FitPanel()
    panel._single_tab._set_composite_model(
        CompositeModel(["VortexLattice", "Constant"], operators=["+"])
    )
    assert panel.send_single_model_to_batch() is True

    batch_tbl = panel._global_tab._param_table
    field_combo = batch_tbl.cellWidget(_row_for(batch_tbl, "field"), 2)
    assert isinstance(field_combo, QComboBox)
    assert field_combo.currentText() == "Fixed"  # VortexLattice's declared default
    field_combo.setCurrentText("Local")

    # User then sets a new field value in Single and sends the same model again.
    single_tbl = panel._single_tab._param_table
    single_tbl.item(_row_for(single_tbl, "field"), 1).setText("777.0")
    assert panel.send_single_model_to_batch() is True

    field_row = _row_for(batch_tbl, "field")
    field_combo = batch_tbl.cellWidget(field_row, 2)
    assert field_combo.currentText() == "Local", (
        "resending the model reset the batch Type combo to its default role "
        "instead of carrying the user's override"
    )
    assert float(batch_tbl.item(field_row, 1).text()) == pytest.approx(777.0), (
        "resending the model did not carry the single tab's new value"
    )
