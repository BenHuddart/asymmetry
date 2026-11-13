"""Released analysis names become canonical before domain restoration."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project.schema import migrate_to_current
from tests.project.test_saved_single_fits_migration import _v24_project


def test_v24_fraction_fit_is_canonical_at_project_boundary():
    model = CompositeModel.from_expression("(Exponential + Gaussian + Constant){frac}")
    project = _v24_project()
    project["datasets"][0]["representations"]["time_fb_asymmetry"]["fit"] = {
        "model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": 20.0},
            {"name": "Lambda", "value": 0.5},
            {"name": "fraction_1", "value": 2.0, "min": 0.0, "max": 3.0},
            {"name": "sigma", "value": 0.3},
            {"name": "fraction_2", "value": 1.0, "fixed": True},
            {"name": "fraction_3", "value": 1.0},
        ],
        "result": {"computation_revision": "older"},
        "provenance": "single",
    }
    original = deepcopy(project)
    migrated = migrate_to_current(project)
    fit = migrated["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""][
        "fits"
    ][0]
    entries = {entry["name"]: entry for entry in fit["parameters"]}
    assert entries["f_Exponential"]["value"] == 0.5
    assert entries["f_Gaussian"]["value"] == 0.25
    assert entries["f_Gaussian"]["fixed"] is True
    assert fit["result"]["computation_revision"] == "older"
    assert project == original
    assert migrate_to_current(migrated) == migrated
    t = np.array([0.0, 0.2, 1.0, 3.0])
    expected = 20 * (0.5 * np.exp(-0.5 * t) + 0.25 * np.exp(-((0.3 * t) ** 2)) + 0.25)
    np.testing.assert_allclose(
        model.function(t, **{name: row["value"] for name, row in entries.items()}), expected
    )


def restore_slot(data):
    from asymmetry.core.project.analysis_migration import fit_slot
    from asymmetry.core.representation import FitSlot

    fit_slot(data)
    return FitSlot.from_dict(data)


def test_fit_slot_migrates_legacy_fraction_parameters():
    """Released positional fractions migrate before FitSlot construction."""
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    slot = restore_slot(
        {
            "model": model.to_dict(),
            "parameters": [
                {"name": "A_1", "value": 20.0},
                {"name": "Lambda", "value": 0.5},
                {"name": "fraction_1", "value": 2.0, "fixed": False},
                {"name": "sigma", "value": 0.3},
                {"name": "fraction_2", "value": 1.0, "fixed": True},
                {"name": "fraction_3", "value": 1.0},
            ],
            "provenance": "single",
        }
    )
    names = [entry["name"] for entry in slot.parameters]
    assert names == ["A_1", "Lambda", "f_Exponential", "sigma", "f_Gaussian"]
    by_name = {entry["name"]: entry for entry in slot.parameters}
    assert by_name["f_Exponential"]["value"] == pytest.approx(0.5)
    assert by_name["f_Gaussian"]["value"] == pytest.approx(0.25)
    assert by_name["f_Gaussian"]["fixed"] is True
    assert not any(name.startswith("fraction_") for name in names)


def test_project_boundary_rejects_malformed_legacy_fraction_value():
    """Corrupt fractions fail at persistence rather than inventing weights."""
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    with pytest.raises(ValueError, match="stored fraction"):
        restore_slot(
            {
                "model": model.to_dict(),
                "parameters": [
                    {"name": "A_1", "value": 20.0},
                    {"name": "Lambda", "value": 0.5},
                    {"name": "fraction_1", "value": None, "fixed": False},
                    {"name": "sigma", "value": 0.3},
                    {"name": "fraction_2", "value": "garbage", "fixed": True},
                    {"name": "fraction_3", "value": 1.0},
                ],
                "provenance": "single",
            }
        )


def test_released_affine_constraints_remain_historical_until_explicit_edit():
    from asymmetry.core.fitting.analysis_history import analysis_block_reason
    from asymmetry.core.project.analysis_migration import fit_slot
    from asymmetry.core.representation import FitSlot

    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    raw = {
        "model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": 2.0, "tie": {"main": "x", "scale": 2.0}},
            {"name": "A_2", "value": 3.0, "fixed": False},
            {"name": "Lambda", "value": 0.5},
            {"name": "sigma", "value": 0.3},
            {"name": "x", "value": 1.0, "fixed": True},
        ],
        "result": {"computation_revision": "older", "parameters": {"A_1": 2.0, "A_2": 3.0}},
    }
    original = deepcopy(raw)
    fit_slot(raw)
    restored = FitSlot.from_dict(raw)
    assert restored.analysis_history["original"] == original
    assert analysis_block_reason(restored.analysis_history, restored.parameters) is not None
    values = {p["name"]: p["value"] for p in restored.parameters}
    t = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(model.function(t, **values), 6 * np.exp(-0.5 * t - (0.3 * t) ** 2))
    # A new seed is not an explicit constraint edit.
    restored.parameters[0]["value"] = 8.0
    assert analysis_block_reason(restored.analysis_history, restored.parameters) is not None
    restored.parameters[0]["fixed"] = True
    assert analysis_block_reason(restored.analysis_history, restored.parameters) is None
    assert FitSlot.from_dict(restored.to_dict()).analysis_history["original"] == original


def test_missing_model_deferred_conversion_survives_actual_slot_codec(monkeypatch):
    from asymmetry.core.fitting.composite import COMPONENTS
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION
    from asymmetry.core.representation import FitSlot

    descriptor = {
        "component_names": ["Exponential", "UnavailableScale"],
        "operators": ["*"],
        "open_parentheses": [1, 0],
        "close_parentheses": [0, 1],
        "fraction_groups": [],
    }
    record = {
        "model": descriptor,
        "parameters": [{"name": "A_1", "value": 2.0}, {"name": "A_2", "value": 3.0}],
        "result": {"computation_revision": None},
    }
    project = {
        "schema_version": 38,
        "datasets": [
            {"representations": {"time_fb_asymmetry": {"single_fits": {"": {"fits": [record]}}}}}
        ],
    }
    migrated = migrate_to_current(project)
    slot = migrated["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""][
        "fits"
    ][0]
    assert slot["analysis_history"]["reason"] == "missing_model"
    slot = FitSlot.from_dict(slot).to_dict()
    migrated["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""]["fits"][
        0
    ] = slot
    monkeypatch.setitem(COMPONENTS, "UnavailableScale", COMPONENTS["Gaussian"])
    resolved = migrate_to_current(migrated)
    slot = resolved["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""][
        "fits"
    ][0]
    assert resolved["schema_version"] == CURRENT_SCHEMA_VERSION
    assert slot.get("analysis_history") is None
    assert slot["parameters"] == [{"name": "A_1", "value": 6.0}]
    assert slot["result"]["computation_revision"] is None
    assert migrate_to_current(resolved) == resolved


def test_workdir_wizard_boundary_versions_payload_and_preserves_execution(tmp_path):
    import json

    from asymmetry.core.workflow.workdir import WorkDir

    workdir = WorkDir(tmp_path)
    workdir.ensure()
    template = {
        "key": "old",
        "model": {
            "component_names": ["Exponential", "Gaussian"],
            "operators": ["*"],
            "open_parentheses": [1, 0],
            "close_parentheses": [0, 1],
            "fraction_groups": [],
        },
    }
    payload = {
        "computation_revision": "older",
        "recommendation": {
            "assessments": [
                {
                    "template": template,
                    "fit_result": {
                        "parameters": [
                            {"name": "A_1", "value": 2.0},
                            {"name": "A_2", "value": 3.0},
                        ],
                        "uncertainties": {"A_1": 0.2, "A_2": 0.3},
                        "computation_revision": None,
                    },
                }
            ]
        },
    }
    workdir.wizard_path(7).write_text(json.dumps(payload))
    restored = workdir.read_wizard(7, require_current=False)
    recommendation = restored["recommendation"]
    assert recommendation["analysis_payload_version"] == 1
    result = recommendation["assessments"][0]["fit_result"]
    assert result["parameters"] == [{"name": "A_1", "value": 6.0}]
    assert result["uncertainties"]["A_1"] == pytest.approx(np.sqrt(0.72))
    assert result["computation_revision"] is None
    assert restored["computation_revision"] == "older"
    workdir.write_wizard(7, restored)
    assert workdir.read_wizard(7, require_current=False) == restored | {
        "schema": 8,
        "asymmetry_version": __import__("asymmetry").__version__,
    }


def test_canonical_auxiliary_tie_and_draft_values_are_not_legacy():
    from asymmetry.core.project.analysis_migration import fit_slot

    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    record = {
        "model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": "", "tie": {"main": "x"}},
            {"name": "Lambda", "value": ""},
            {"name": "sigma", "value": 0.3},
            {"name": "x", "value": 2.0},
        ],
    }
    original = deepcopy(record)
    fit_slot(record)
    assert record == original


def test_bounded_product_retains_original_bounds_and_snapshot():
    from asymmetry.core.fitting.analysis_history import analysis_block_reason
    from asymmetry.core.project.analysis_migration import fit_slot

    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    record = {
        "model": model.to_dict(),
        "parameters": [
            {"name": "A_1", "value": 2.0, "min": 0.0, "max": 2.0},
            {"name": "A_2", "value": 3.0, "min": 0.0, "max": 3.0},
        ],
    }
    original = deepcopy(record)
    fit_slot(record)
    assert record["analysis_history"]["original"] == original
    assert record["parameters"][0]["value"] == 6.0
    assert record["parameters"][0]["max"] == float("inf")
    assert analysis_block_reason(record["analysis_history"], record["parameters"]) is not None


@pytest.mark.parametrize("value", [None, "garbage", float("nan"), float("inf")])
def test_corrupt_product_scale_is_rejected_at_boundary(value):
    from asymmetry.core.project.analysis_migration import fit_slot

    record = {
        "model": CompositeModel.from_expression("(Exponential * Gaussian)").to_dict(),
        "parameters": [{"name": "A_1", "value": 2.0}, {"name": "A_2", "value": value}],
    }
    with pytest.raises(ValueError, match="stored product amplitude"):
        fit_slot(record)


def test_series_constraint_archive_survives_series_codec():
    from asymmetry.core.fitting.analysis_history import analysis_block_reason
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
                    "parameters": [{"name": "A_1", "value": 2.0}, {"name": "A_2", "value": 3.0}]
                },
                "results_by_run": {
                    "7": {"parameters": {"A_1": 2.0, "A_2": 3.0}, "computation_revision": "older"}
                },
            }
        ],
    }
    series = FitSeries.from_dict(migrate_to_current(project)["batches"][0])
    assert analysis_block_reason(series.analysis_history, series.recipe["parameters"]) is not None
    stored = FitSeries.from_dict(series.to_dict())
    assert stored.analysis_history == series.analysis_history
    assert stored.results_by_run[7]["parameters"] == {"A_1": 6.0}
    assert stored.results_by_run[7]["computation_revision"] == "older"


def test_joint_pending_conversion_retains_earlier_member_mapping(monkeypatch):
    from asymmetry.core.fitting.composite import COMPONENTS
    from asymmetry.core.representation.joint_fit import JointFit

    model = CompositeModel.from_expression("(Exponential * Gaussian)").to_dict()
    missing = {"component_names": ["UnavailableScale"], "operators": []}
    project = {
        "schema_version": 38,
        "datasets": [],
        "batches": [
            {
                "batch_id": "a",
                "canonical_model": model,
                "param_roles": {"A_1": "local", "A_2": "local"},
                "recipe": {
                    "parameters": [{"name": "A_1", "value": 2.0}, {"name": "A_2", "value": 3.0}]
                },
            },
            {"batch_id": "b", "canonical_model": missing, "recipe": {"parameters": []}},
        ],
        "joint_fits": [
            {
                "joint_id": "j",
                "label": None,
                "rep_type": "time_fb_asymmetry",
                "member_batch_ids": ["a", "b"],
                "shared": [
                    {
                        "name": "s",
                        "members": {"a": "A_1", "b": "A"},
                        "value": 2.0,
                        "min": None,
                        "max": None,
                    }
                ],
            }
        ],
    }
    first = migrate_to_current(project)
    joint = JointFit.from_dict(first["joint_fits"][0])
    assert joint.analysis_history["reason"] == "missing_model"
    first["joint_fits"][0] = joint.to_dict()
    monkeypatch.setitem(COMPONENTS, "UnavailableScale", COMPONENTS["Exponential"])
    second = migrate_to_current(first)["joint_fits"][0]
    assert second["shared"] == []
    assert second["analysis_history"]["reason"] == "constraints"
    assert second["analysis_history"]["original"]["shared"][0]["members"]["a"] == "A_1"


def test_signed_fixed_product_factor_is_retained_as_historical():
    from asymmetry.core.fitting.analysis_history import analysis_block_reason
    from asymmetry.core.project.analysis_migration import fit_slot

    record = {
        "model": CompositeModel.from_expression("Constant * Exponential").to_dict(),
        "parameters": [
            {"name": "A_bg", "value": -2.0, "fixed": True},
            {"name": "A_1", "value": 3.0, "min": 0.0, "max": float("inf")},
            {"name": "Lambda", "value": 0.5},
        ],
    }
    original = deepcopy(record)
    fit_slot(record)
    assert record["analysis_history"]["original"] == original
    assert analysis_block_reason(record["analysis_history"], record["parameters"]) is not None
    assert record["parameters"][0]["value"] == -6.0


def test_folded_result_minos_band_remains_original_and_unavailable_canonically():
    from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
    from asymmetry.core.project.analysis_migration import analysis_result

    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    result = {
        "parameters": {"A_1": 2.0, "A_2": 3.0, "Lambda": 0.5, "sigma": 0.3},
        "uncertainties_asymmetric": {"A_1": [-0.1, 0.2], "Lambda": [-0.01, 0.01]},
        "computation_revision": "older",
    }
    original = deepcopy(result)
    analysis_result(result, model)
    assert result["parameters"]["A_1"] == 6.0
    assert result["uncertainties_asymmetric"] == {"Lambda": [-0.01, 0.01]}
    assert (
        result["parameter_history"]["uncertainties_asymmetric"]
        == original["uncertainties_asymmetric"]
    )
    # Wizard result codecs use row lists; the original statistics travel separately.
    result["parameters"] = [
        {"name": name, "value": value} for name, value in result["parameters"].items()
    ]
    restored = _serialize_fit_result(_deserialize_fit_result(result))
    assert restored["parameter_history"] == result["parameter_history"]
    assert restored["uncertainties_asymmetric"] == result["uncertainties_asymmetric"]
    assert restored["computation_revision"] == "older"
