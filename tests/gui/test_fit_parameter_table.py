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
from asymmetry.core.fitting.seeding import Seed
from asymmetry.gui.panels.fit_panel import FitParameterTable
from asymmetry.gui.utils.formatting import format_param_label as _format_param_label


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _model() -> CompositeModel:
    return CompositeModel(["Exponential", "Constant"], operators=["+"])


def _fix_checkbox(table: FitParameterTable, row: int) -> QCheckBox:
    return table.cellWidget(row, table.COL_FIX).findChild(QCheckBox)


def test_populate_creates_one_row_per_param_from_its_seeds(qapp):
    model = _model()
    table = FitParameterTable()
    table.populate(model, seeds={"A_bg": Seed(value=0.05, fixed=True, min=0.0)})

    assert table.rowCount() == len(model.param_names)
    names = [table.item(r, table.COL_NAME).data(Qt.ItemDataRole.UserRole) for r in range(3)]
    assert names == list(model.param_names)
    # The seed carries value, Fix state and bounds onto its row; a parameter the
    # caller says nothing about keeps the model's own static seed.
    fixed = {n: _fix_checkbox(table, r).isChecked() for r, n in enumerate(names)}
    assert fixed["A_bg"] is True
    assert fixed["Lambda"] is False
    bg_row = names.index("A_bg")
    assert float(table.item(bg_row, table.COL_VALUE).text()) == pytest.approx(0.05)
    assert table.item(bg_row, table.COL_MIN).text() == "0.0"
    assert float(table.item(names.index("Lambda"), table.COL_VALUE).text()) == pytest.approx(
        model.param_defaults["Lambda"]
    )


def test_populate_stamps_every_value_as_seeded_and_an_edit_makes_it_the_users(qapp):
    """Provenance is what makes a re-seed safe: it replaces seeds, not values."""
    model = _model()
    table = FitParameterTable()
    table.populate(model)

    assert all(entry["seeded"] is True for entry in table.parameters_state())

    table.item(0, table.COL_VALUE).setText("0.42")
    state = {entry["name"]: entry for entry in table.parameters_state()}
    assert state["A_1"]["seeded"] is False
    assert state["Lambda"]["seeded"] is True


def test_reseed_replaces_seeded_values_and_leaves_typed_ones(qapp):
    model = _model()
    table = FitParameterTable()
    table.populate(model)
    table.item(0, table.COL_VALUE).setText("0.42")  # the user's own guess

    table.reseed({"A_1": Seed(value=9.0), "Lambda": Seed(value=7.0)})

    assert float(table.item(0, table.COL_VALUE).text()) == pytest.approx(0.42)
    assert float(table.item(1, table.COL_VALUE).text()) == pytest.approx(7.0)


def test_run_bound_cells_are_handed_back_to_the_seeder(qapp):
    """A value that described the previous run is not a user value for this one."""
    model = _model()
    table = FitParameterTable()
    table.populate(model)
    table.item(0, table.COL_VALUE).setText("0.42")
    table.item(1, table.COL_VALUE).setText("3.3")

    table.mark_run_bound_as_seeded(["Lambda"])
    table.reseed({"A_1": Seed(value=9.0), "Lambda": Seed(value=7.0)})

    assert float(table.item(0, table.COL_VALUE).text()) == pytest.approx(0.42)
    assert float(table.item(1, table.COL_VALUE).text()) == pytest.approx(7.0)


def test_provenance_survives_a_state_round_trip(qapp):
    """The identity carry-over runs through parameters_state/restore_parameters."""
    model = _model()
    table = FitParameterTable()
    table.populate(model)
    table.item(0, table.COL_VALUE).setText("0.42")

    carried = FitParameterTable()
    carried.populate(model)
    carried.restore_parameters({s["name"]: s for s in table.parameters_state()})

    carried.reseed({name: Seed(value=9.0) for name in model.param_names})

    assert float(carried.item(0, carried.COL_VALUE).text()) == pytest.approx(0.42)
    assert float(carried.item(1, carried.COL_VALUE).text()) == pytest.approx(9.0)


def test_restored_state_without_provenance_is_the_users(qapp):
    """A project file written before provenance existed holds user values."""
    model = _model()
    table = FitParameterTable()
    table.populate(model)

    table.restore_parameters({"A_1": {"name": "A_1", "value": 0.42}})
    table.reseed({"A_1": Seed(value=9.0)})

    assert float(table.item(0, table.COL_VALUE).text()) == pytest.approx(0.42)


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


def test_format_value_error_matches_precision_to_uncertainty():
    from asymmetry.gui.utils.formatting import format_value_error

    assert format_value_error(0.46674, 0.00163) == "0.4667 ± 0.0016"
    assert format_value_error(12.5001, 0.28) == "12.50 ± 0.28"
    assert format_value_error(-0.8797, 0.0011) == "-0.8797 ± 0.0011"
    assert format_value_error(199.83, 12.0) == "200 ± 12"


def test_format_value_error_degenerate_inputs():
    from asymmetry.gui.utils.formatting import format_value_error

    assert format_value_error(0.4, 0.0) == "0.4"
    assert format_value_error(0.4, float("nan")) == "0.4"
    assert format_value_error(float("nan"), 0.1) == "—"
    # Pathological scale mismatch falls back to independent rounding rather
    # than an unreadable 16-decimal matched-precision string.
    assert format_value_error(1.0, 1e-15) == "1 ± 1e-15"
