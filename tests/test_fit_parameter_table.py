"""Unit tests for the shared FitParameterTable widget.

The table is the reusable Name·Value·Fix·Min·Max·Batch·Link·Tie parameter
editor shared by the single-fit panel and the single grouped (individual-groups)
fit. These tests exercise it directly (no host panel).
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.gui.panels.fit_panel import FitParameterTable, _format_param_label


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _model() -> CompositeModel:
    return CompositeModel(["Exponential", "Constant"], operators=["+"])


def _fix_checkbox(table: FitParameterTable, row: int) -> QCheckBox:
    return table.cellWidget(row, table.COL_FIX).findChild(QCheckBox)


def test_populate_creates_one_row_per_param_with_fixed_defaults(qapp):
    model = _model()
    table = FitParameterTable()
    table.populate(model, fixed_names={"A_bg"})

    assert table.rowCount() == len(model.param_names)
    names = [table.item(r, table.COL_NAME).data(Qt.ItemDataRole.UserRole) for r in range(3)]
    assert names == list(model.param_names)
    # The named default is checked; others are not.
    fixed = {n: _fix_checkbox(table, r).isChecked() for r, n in enumerate(names)}
    assert fixed["A_bg"] is True
    assert fixed["Lambda"] is False


def test_name_cells_carry_full_label_tooltip(qapp):
    """The Name column is kept narrow and clips "name (unit)" labels, so each name
    cell must expose the full formatted label as a tooltip (Bug #6)."""
    model = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    table = FitParameterTable()
    table.populate(model)

    for row, pname in enumerate(model.param_names):
        item = table.item(row, table.COL_NAME)
        full = _format_param_label(pname)
        # The full label is the displayed text AND the tooltip, so even when the
        # narrow column elides it (e.g. "f (MHz)" → "f (MH…") it is readable on hover.
        assert item.text() == full
        assert item.toolTip() == full
    # The labels really do carry units that overflow a narrow column.
    labels = [_format_param_label(p) for p in model.param_names]
    assert any("(" in label for label in labels)


def test_read_parameter_set_reflects_value_fix_and_bounds(qapp):
    model = _model()
    table = FitParameterTable()
    table.populate(model)

    table.item(0, table.COL_VALUE).setText("0.3")
    table.item(0, table.COL_MIN).setText("0")
    table.item(0, table.COL_MAX).setText("1")
    _fix_checkbox(table, 1).setChecked(True)

    ps = table.read_parameter_set()
    assert ps["A_1"].value == pytest.approx(0.3)
    assert ps["A_1"].min == pytest.approx(0.0)
    assert ps["A_1"].max == pytest.approx(1.0)
    assert ps["A_1"].fixed is False
    assert ps["Lambda"].fixed is True


def test_invalid_value_raises(qapp):
    table = FitParameterTable()
    table.populate(_model())
    table.item(0, table.COL_VALUE).setText("not-a-number")
    with pytest.raises(ValueError):
        table.read_parameter_set()


def test_fix_and_link_are_mutually_exclusive(qapp):
    table = FitParameterTable()
    table.populate(_model())
    fix = _fix_checkbox(table, 0)
    link = table.cellWidget(0, table.COL_LINK)
    assert isinstance(link, QComboBox)

    fix.setChecked(True)
    assert not link.isEnabled()
    fix.setChecked(False)
    assert link.isEnabled()

    # Selecting a link group clears + disables Fix.
    link.setCurrentIndex(1)  # first real group
    assert fix.isChecked() is False
    assert not fix.isEnabled()


def test_parameters_state_round_trips_through_restore(qapp):
    model = _model()
    table = FitParameterTable()
    table.populate(model)
    table.item(0, table.COL_VALUE).setText("0.25")
    _fix_checkbox(table, 2).setChecked(True)

    state = {s["name"]: s for s in table.parameters_state()}
    assert set(state["A_1"]) >= {"name", "value", "fixed", "min", "max", "link_group", "tie"}

    restored = FitParameterTable()
    restored.populate(model)
    restored.restore_parameters(state)
    ps = restored.read_parameter_set()
    assert ps["A_1"].value == pytest.approx(0.25)
    assert ps["A_bg"].fixed is True


def test_current_seed_values_skips_non_numeric(qapp):
    table = FitParameterTable()
    table.populate(_model())
    table.item(0, table.COL_VALUE).setText("1.5")
    table.item(1, table.COL_VALUE).setText("oops")
    seeds = table.current_seed_values()
    assert seeds.get("A_1") == "1.5"
    assert "Lambda" not in seeds


def test_populate_clears_stale_auxiliary_params_on_model_change(qapp):
    # Auxiliary (non-model) params from a prior restore must not survive a model
    # rebuild — otherwise they resurrect as ghost params in read/state.
    model = _model()
    table = FitParameterTable()
    table.populate(model)
    # A restore that carries an auxiliary param with no table row.
    state = {s["name"]: s for s in table.parameters_state()}
    state["delta"] = {"name": "delta", "value": 0.5, "fixed": False, "min": "-inf", "max": "inf"}
    table.restore_parameters(state)
    assert any(p.name == "delta" for p in table.read_parameter_set())

    # Rebuilding for a (possibly different) model drops the stale auxiliary.
    table.populate(model)
    assert all(p.name != "delta" for p in table.read_parameter_set())
    assert all(s["name"] != "delta" for s in table.parameters_state())


def test_populate_param_names_restricts_rows(qapp):
    # The grouped physics table renders only a subset (nuisance amplitudes omitted).
    model = _model()
    table = FitParameterTable()
    subset = [n for n in model.param_names if n != "A_bg"]
    table.populate(model, param_names=subset)
    names = [
        table.item(r, table.COL_NAME).data(Qt.ItemDataRole.UserRole)
        for r in range(table.rowCount())
    ]
    assert names == subset
    assert "A_bg" not in {p.name for p in table.read_parameter_set()}


def test_batch_column_can_be_hidden(qapp):
    table = FitParameterTable()
    table.populate(_model())
    assert not table.isColumnHidden(table.COL_BATCH)
    table.set_batch_column_visible(False)
    assert table.isColumnHidden(table.COL_BATCH)
