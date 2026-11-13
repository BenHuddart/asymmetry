"""Actual editor restoration preserves historical content and refit safeguards."""

from copy import deepcopy

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project.schema import migrate_to_current
from asymmetry.gui.panels.fit.global_tab import GlobalFitTab
from asymmetry.gui.panels.fit.single_tab import SingleFitTab

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("tab_class", [SingleFitTab, GlobalFitTab])
def test_project_constraint_history_survives_editor_restore_and_resave(qapp, tab_class):
    # The schema10 single form and later global form persisted the same raw rows.
    state = {
        "composite_model": {
            "component_names": ["Exponential", "Gaussian", "Constant"],
            "operators": ["+", "+"],
            "open_parentheses": [1, 0, 0],
            "close_parentheses": [0, 0, 1],
            "fraction_groups": [[0, 2]],
        },
        "parameters": [
            {"name": "A_1", "value": 20.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "sigma", "value": 0.3},
            {"name": "fraction_1", "value": 0.5, "tie": {"main": "x", "scale": 1.0}},
            {"name": "fraction_2", "value": 0.25},
            {"name": "fraction_3", "value": 0.25},
            {"name": "x", "value": 0.5},
        ],
    }
    if tab_class is GlobalFitTab:
        state.update(group_parameters=[], group_model_parameters=[])
    original = deepcopy(state)
    project = {
        "schema_version": 10,
        "datasets": [],
        "single_fit_state": state,
        "fit_states": {"time": {"single_fit_state": state}},
    }
    migrated = migrate_to_current(project)
    state = migrated["fit_states"]["time"]["single_fit_state"]
    tab = tab_class()
    tab.restore_state(state)
    assert "Historical constraints" in tab._results_card.content_html(), (
        state.get("analysis_history"),
        tab.get_state()["parameters"],
    )
    restored = tab.get_state()
    assert restored["analysis_history"]["original"] == original
    parser = (
        tab._parameter_set_from_table
        if tab_class is SingleFitTab
        else tab._parse_parameter_configuration
    )
    with pytest.raises(ValueError, match="Historical constraints"):
        parser()
    second = tab_class()
    second.restore_state(restored)
    assert second.get_state()["analysis_history"]["original"] == original
    assert (
        second._composite_model.component_names
        == CompositeModel.from_dict(original["composite_model"]).component_names
    )


def test_slot_mediator_retains_archive_without_saved_form(qapp):
    from types import SimpleNamespace

    from asymmetry.core.project.analysis_migration import fit_slot
    from asymmetry.core.representation import FitSlot, RepresentationType
    from asymmetry.gui.mainwindow import MainWindow

    record = {
        "model": CompositeModel.from_expression("(Exponential * Gaussian)").to_dict(),
        "parameters": [
            {"name": "A_1", "value": 2.0, "tie": {"main": "x", "scale": 2.0}},
            {"name": "A_2", "value": 3.0},
            {"name": "x", "value": 1.0},
        ],
        "result": {"parameters": {"A_1": 2.0, "A_2": 3.0}, "computation_revision": None},
    }
    fit_slot(record)
    slot = FitSlot.from_dict(record)
    fake = SimpleNamespace(
        _active_representation_type=lambda: RepresentationType.TIME_FB_ASYMMETRY,
        _current_single_fit_projection=lambda: None,
        _project_model=SimpleNamespace(
            representation=lambda run, rep: SimpleNamespace(fit_for=lambda projection: slot)
        ),
        _fit_panel=SimpleNamespace(
            build_single_fit_payload_from_slot=lambda model, parameters, result: {
                "composite_model": model,
                "parameters": parameters,
                "result": result,
            }
        ),
    )
    payload = MainWindow._single_fit_restore_payload(fake, SimpleNamespace(run_number=7))
    tab = SingleFitTab()
    tab.restore_state(payload)
    assert tab.get_state()["analysis_history"] == slot.analysis_history
    with pytest.raises(ValueError, match="Historical constraints"):
        tab._parameter_set_from_table()


def test_real_reconstructed_form_preserves_roles_auxiliaries_and_edit_intent(qapp):
    from PySide6.QtCore import Qt

    from asymmetry.core.project.analysis_migration import fit_slot
    from asymmetry.gui.panels.fit.panel import FitPanel

    record = {
        "model": CompositeModel.from_expression("(Exponential * Gaussian)").to_dict(),
        "parameters": [
            {"name": "A_1", "value": 2.0, "min": 0.0, "max": 2.0, "role": "Global"},
            {"name": "A_2", "value": 3.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "sigma", "value": 0.3},
            {"name": "x", "value": 1.0, "fixed": True},
        ],
        "result": {
            "parameters": {"A_1": 2.0, "A_2": 3.0, "Lambda": 0.5, "sigma": 0.3},
            "success": True,
            "computation_revision": "older",
        },
    }
    fit_slot(record)
    panel = FitPanel()
    state = panel.build_single_fit_payload_from_slot(
        record["model"], record["parameters"], record["result"]
    )
    state["analysis_history"] = record["analysis_history"]
    tab = SingleFitTab()
    tab.restore_state(state)
    with pytest.raises(ValueError, match="Historical constraints"):
        tab._parameter_set_from_table()
    rows = {
        tab._param_table.item(i, 0).data(Qt.ItemDataRole.UserRole): i
        for i in range(tab._param_table.rowCount())
    }
    tab._param_table.item(rows["A_1"], tab._param_table.COL_MIN).setText("0")
    assert tab._parameter_set_from_table()["x"].fixed is True
    edited = tab.get_state()
    second = SingleFitTab()
    second.restore_state(edited)
    second._parameter_set_from_table()
    assert (
        second.get_state()["analysis_history"]["original"] == record["analysis_history"]["original"]
    )


def test_seeded_draft_uses_the_seed_series_history(qapp):
    from asymmetry.core.representation.series import FitSeries

    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    project = {
        "schema_version": 38,
        "datasets": [],
        "batches": [
            {
                "batch_id": "old",
                "rep_type": "time_fb_asymmetry",
                "canonical_model": model.to_dict(),
                "param_roles": {"A_1": "global", "A_2": "local"},
                "recipe": {
                    "parameters": [
                        {"name": "A_1", "value": 2.0},
                        {"name": "A_2", "value": 3.0},
                        {"name": "Lambda", "value": 0.5},
                        {"name": "sigma", "value": 0.3},
                    ]
                },
            }
        ],
    }
    series = FitSeries.from_dict(migrate_to_current(project)["batches"][0])
    tab = GlobalFitTab()
    tab.open_draft(datasets=[], seed_from=series)
    assert tab.get_state()["analysis_history"] == series.analysis_history
    with pytest.raises(ValueError, match="Historical constraints"):
        tab._parse_parameter_configuration()


def test_grouped_inner_table_uses_implicit_fraction_model(qapp):
    outer = CompositeModel(["Exponential", "Gaussian", "Constant"], operators=["+", "+"])
    state = {
        "composite_model": outer.to_dict(),
        "parameters": [],
        "group_parameters": [],
        "group_model_parameters": [
            {"name": "A_1", "value": 20.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "sigma", "value": 0.3},
            {"name": "fraction_1", "value": 2.0},
            {"name": "fraction_2", "value": 1.0},
            {"name": "fraction_3", "value": 1.0},
        ],
    }
    project = {"schema_version": 38, "datasets": [], "multi_group_fit_state": {"single": state}}
    current = migrate_to_current(project)["multi_group_fit_state"]["single"]
    assert current["composite_model"]["fraction_groups"] == []
    tab = GlobalFitTab(member_kind="groups", grouped_single=True)
    tab.restore_state(current)
    values = {row["name"]: row["value"] for row in tab.get_state()["group_model_parameters"]}
    assert values["f_Exponential"] == 0.5
    assert values["f_Gaussian"] == 0.25


def test_unchecking_fixed_is_an_explicit_constraint_edit(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox

    from asymmetry.core.project.analysis_migration import fit_editor
    from asymmetry.gui.panels.fit.panel import FitPanel

    state = {
        "composite_model": CompositeModel.from_expression("(Exponential * Gaussian)").to_dict(),
        "parameters": [
            {"name": "A_1", "value": 2.0, "type": "Fixed"},
            {"name": "A_2", "value": 3.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "sigma", "value": 0.3},
        ],
    }
    fit_editor(state)
    canonical = FitPanel().build_single_fit_payload_from_slot(
        state["composite_model"],
        state["parameters"],
        {"parameters": {row["name"]: row["value"] for row in state["parameters"]}},
    )
    canonical["analysis_history"] = state["analysis_history"]
    state = canonical
    tab = SingleFitTab()
    tab.restore_state(state)
    with pytest.raises(ValueError, match="Historical constraints"):
        tab._parameter_set_from_table()
    row = next(
        i
        for i in range(tab._param_table.rowCount())
        if tab._param_table.item(i, 0).data(Qt.ItemDataRole.UserRole) == "A_1"
    )
    # Batch role metadata is retained; this checkbox is the Single-tab constraint.
    checkbox = tab._param_table.cellWidget(row, tab._param_table.COL_FIX).findChild(QCheckBox)
    checkbox.setChecked(False)
    tab._parameter_set_from_table()


def _legacy_single_fraction_state() -> dict:
    """A pre-rework single-fit state blob with legacy fraction_<k> keys."""
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    return {
        "composite_model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": 20.0, "fixed": False, "min": "0", "max": "inf"},
            {"name": "Lambda", "value": 0.5, "fixed": False, "min": "0", "max": "inf"},
            {"name": "fraction_1", "value": 2.0, "fixed": False, "min": "0", "max": "1"},
            {"name": "sigma", "value": 0.3, "fixed": False, "min": "0", "max": "inf"},
            {"name": "fraction_2", "value": 1.0, "fixed": False, "min": "0", "max": "1"},
            {"name": "fraction_3", "value": 1.0, "fixed": False, "min": "0", "max": "1"},
        ],
    }


def test_single_fit_restore_migrates_legacy_fraction_state(qapp: QApplication) -> None:
    tab = SingleFitTab()
    from asymmetry.core.project.schema import migrate_to_current

    payload = migrate_to_current(
        {
            "schema_version": 38,
            "datasets": [],
            "fit_states": {"time": {"single_fit_state": _legacy_single_fraction_state()}},
        }
    )
    tab.restore_state(payload["fit_states"]["time"]["single_fit_state"])

    by_name = {p["name"]: p for p in tab.get_state()["parameters"]}
    # Legacy fraction_1/2 → new free names (normalised 2:1:1 → 0.5/0.25); the
    # dropped fraction_3 term becomes the display-only derived remainder.
    assert "f_Exponential" in by_name
    assert "f_Gaussian" in by_name
    assert not any(name.startswith("fraction_") for name in by_name)
    assert float(by_name["f_Exponential"]["value"]) == pytest.approx(0.5)
    assert float(by_name["f_Gaussian"]["value"]) == pytest.approx(0.25)

    row_by_name = {
        tab._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._param_table.rowCount())
    }
    # The derived remainder display row is present and shows 1 - 0.5 - 0.25 = 0.25.
    assert "f_Constant" in row_by_name
    assert tab._param_table.item(row_by_name["f_Constant"], 1).text() == "0.25"


def test_global_fit_restore_migrates_legacy_fraction_state(qapp: QApplication) -> None:
    tab = GlobalFitTab()
    from asymmetry.core.project.schema import migrate_to_current

    payload = migrate_to_current(
        {
            "schema_version": 38,
            "datasets": [],
            "fit_states": {"time": {"single_fit_state": _legacy_single_fraction_state()}},
        }
    )
    tab.restore_state(payload["fit_states"]["time"]["single_fit_state"])

    by_name = {p["name"]: p for p in tab.get_state()["parameters"]}
    assert "f_Exponential" in by_name
    assert "f_Gaussian" in by_name
    assert not any(name.startswith("fraction_") for name in by_name)

    row_by_name = {
        tab._param_table.item(row, 0).data(Qt.ItemDataRole.UserRole): row
        for row in range(tab._param_table.rowCount())
    }
    assert "f_Constant" in row_by_name
    assert tab._param_table.item(row_by_name["f_Constant"], 1).text() == "0.25"
