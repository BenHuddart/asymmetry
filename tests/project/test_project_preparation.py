"""Canonical owned shapes, live links, and deliberately unavailable analysis history."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.project import (
    CURRENT_SCHEMA_VERSION,
    GroupingProfile,
    ProfileFingerprint,
    load_prepared_project,
    load_project,
    prepare_project,
    save_project,
)
from asymmetry.core.representation import FitSeries, FitSlot, RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from tests.project.test_saved_single_fits_migration import _v24_project


def project():
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [{"run_number": 11, "source_file": "missing.nxs", "profile": "sample"}],
        "grouping_profiles": [GroupingProfile("sample", ProfileFingerprint("EMU", 2)).to_dict()],
    }
    model = ProjectModel()
    rep = model.ensure_dataset(11).ensure(RepresentationType.TIME_FB_ASYMMETRY)
    rep.fit = FitSlot(
        fit_id="fit-owned",
        model=CompositeModel.from_expression("Exponential").to_dict(),
        parameters=[{"name": "A", "value": 0.2}, {"name": "Lambda", "value": 0.5}],
        result={"parameters": {"A": 0.2, "Lambda": 0.5}, "computation_revision": "older"},
        provenance="single",
    )
    model.add_batch(
        FitSeries("series", RepresentationType.TIME_FB_ASYMMETRY, member_run_numbers=[11, 99])
    )
    model.set_active_series(RepresentationType.TIME_FB_ASYMMETRY, "series")
    model.write_to_project_state(state)
    return state


REP = ("datasets", 0, "representations", "time_fb_asymmetry")
FIT_SET = (*REP, "single_fits", "")


@pytest.mark.parametrize(
    "path,value",
    [
        (("datasets",), "bad"),
        (("datasets",), {}),
        (("datasets", 0, "source_file"), []),
        (("datasets", 0, "source_file"), None),
        (("datasets", 0, "run_number"), True),
        (("datasets", 0, "metadata_overrides"), []),
        (("datasets", 0, "grouping_overrides"), []),
        (("grouping_profiles", 0, "groups"), []),
        (("grouping_profiles", 0, "alpha_policy"), []),
        (("grouping_profiles", 0, "calibration_records"), "bad"),
        (
            ("grouping_profiles", 0, "alpha_policy"),
            {"calibration_id": "missing", "coefficient": "alpha"},
        ),
        (("grouping_profiles", 0, "background_policy"), {"details": {"background_run": []}}),
        (REP, []),
        ((*REP, "recipe"), []),
        ((*FIT_SET, "fits"), "bad"),
        ((*FIT_SET, "fits", 0, "parameters"), ["bad"]),
        ((*FIT_SET, "open_id"), "absent"),
        (("batches", 0, "results_by_run"), []),
        (("batches", 0, "member_run_numbers"), [True]),
        (("active_series", "time_fb_asymmetry"), "absent"),
        (("plot_state",), []),
        (("browser_state",), None),
        (("global_fit_studies",), ["bad"]),
        (
            ("joint_fits",),
            [{"joint_id": "j", "rep_type": "time_fb_asymmetry", "member_batch_ids": ["a"]}],
        ),
        (("combined_datasets",), [{"combined_run_number": -1, "source_run_numbers": "bad"}]),
    ],
)
def test_malformed_owned_fields_are_contextual(path, value):
    state = project()
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    before = deepcopy(state)
    with pytest.raises(ValueError, match=path[0]):
        prepare_project(state)
    assert state == before


@pytest.mark.parametrize("key", ["datasets", "grouping_profiles", "batches"])
def test_duplicate_identity_is_rejected_before_registry_construction(key):
    state = project()
    state[key].append(deepcopy(state[key][0]))
    with pytest.raises(ValueError, match=f"{key}.*duplicate"):
        prepare_project(state)


@pytest.mark.parametrize("root", [[], "bad", None, {"datasets": [], "schema_version": 999}])
def test_envelope_is_rejected(root):
    with pytest.raises(ValueError):
        prepare_project(root)


def test_preparation_preserves_current_history_and_caller_input(tmp_path):
    state = project()
    state["datasets"][0]["metadata_overrides"] = {"custom": {"arbitrary": [1, None, {}]}}
    slot = state["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""]["fits"][
        0
    ]
    slot["analysis_history"] = {
        "reason": "authored",
        "original": {
            "parameters": [{"name": "fraction_1", "value": 2, "tie": {"main": "archived_missing"}}]
        },
    }
    slot["result"]["parameter_history"] = {"uncertainty": None, "old_name": 2}
    before = deepcopy(state)
    prepared = prepare_project(state)
    assert state == before
    assert prepared.state == before
    prepared.profiles[0].name = "detached"
    assert state == before
    path = tmp_path / "project.asymp"
    save_project(state, path)
    first = load_project(path)
    save_project(first, path)
    assert load_project(path) == first
    restored = load_prepared_project(path).model.representation(
        11, RepresentationType.TIME_FB_ASYMMETRY
    )
    assert restored.fit.result["computation_revision"] == "older"
    assert restored.fit.analysis_history == slot["analysis_history"]
    assert restored.fit.result["parameter_history"] == slot["result"]["parameter_history"]
    assert prepared.model.batch("series").member_run_numbers == [11, 99]


@pytest.mark.parametrize(
    "expression,rows,expected",
    [
        (
            "(Exponential + Gaussian + Constant){frac}",
            [
                {"name": "A_1", "value": 20.0},
                {"name": "Lambda", "value": 0.5},
                {"name": "sigma", "value": 0.3},
                {"name": "fraction_1", "value": 2.0},
                {"name": "fraction_2", "value": 1.0},
                {"name": "fraction_3", "value": 1.0},
            ],
            lambda t: 20 * (0.5 * np.exp(-0.5 * t) + 0.25 * np.exp(-((0.3 * t) ** 2)) + 0.25),
        ),
        (
            "(Exponential * Gaussian)",
            [
                {"name": "A_1", "value": 2.0},
                {"name": "A_2", "value": 3.0},
                {"name": "Lambda", "value": 0.5},
                {"name": "sigma", "value": 0.3},
            ],
            lambda t: 6 * np.exp(-0.5 * t) * np.exp(-((0.3 * t) ** 2)),
        ),
    ],
)
def test_released_fit_preparation_preserves_independent_curves(expression, rows, expected):
    state = _v24_project()
    model = CompositeModel.from_expression(expression)
    state["datasets"][0]["representations"]["time_fb_asymmetry"]["fit"].update(
        model=model.to_dict(), parameters=rows
    )
    prepared = prepare_project(state)
    slot = prepared.model.representation(10, RepresentationType.TIME_FB_ASYMMETRY).fit
    restored = CompositeModel.from_dict(slot.model)
    time = np.array([0.0, 0.2, 1.0, 3.0])
    np.testing.assert_allclose(
        restored.function(time, **{p["name"]: p["value"] for p in slot.parameters}),
        expected(time),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("results_by_run", {"11": ["bad"]}),
        ("recipe", {"parameters": ["bad"]}),
    ],
)
def test_series_does_not_silently_drop_malformed_owned_content(key, value):
    state = project()
    state["batches"][0][key] = value
    with pytest.raises(ValueError, match="batches"):
        prepare_project(state)


def test_joint_shared_numeric_row_is_not_silently_dropped():
    state = project()
    state["joint_fits"] = [
        {
            "joint_id": "joint",
            "rep_type": "time_fb_asymmetry",
            "member_batch_ids": ["series", "stale"],
            "shared": [
                {
                    "name": "rate",
                    "members": {"series": "Lambda", "stale": "Lambda"},
                    "value": ["bad"],
                }
            ],
        }
    ]
    with pytest.raises(ValueError, match="joint_fits.*shared.*value"):
        prepare_project(state)


def test_missing_constituents_and_historical_links_are_supported():
    state = project()
    state["datasets"][0]["profile"] = "removed profile"
    state["combined_datasets"] = [{"combined_run_number": -1, "source_run_numbers": [11, 999]}]
    state["joint_fits"] = [
        {
            "joint_id": "joint",
            "rep_type": "time_fb_asymmetry",
            "member_batch_ids": ["series", "removed"],
            "shared": [],
        }
    ]
    prepared = prepare_project(state)
    assert prepared.model.joint_fits["joint"].member_batch_ids == ["series", "removed"]
    assert prepared.state["combined_datasets"] == state["combined_datasets"]


@pytest.mark.parametrize("version", ["0.5.0", "0.17.1"])
def test_actual_released_writer_fixture_prepares_and_resaves(version, tmp_path):
    from pathlib import Path

    source = Path(__file__).parents[1] / "fixtures" / "projects" / f"released-v{version}.asymp"
    prepared = load_prepared_project(source)
    assert prepared.state["schema_version"] == CURRENT_SCHEMA_VERSION
    path = tmp_path / "resaved.asymp"
    save_project(prepared.state, path)
    assert load_project(path) == prepared.state


def test_study_does_not_discard_a_malformed_group():
    from tests.core.test_global_fit_study import _study

    state = project()
    raw = _study().to_dict()
    raw["groups"][0]["x"] = ["bad"]
    state["global_fit_studies"] = [raw]
    with pytest.raises(ValueError, match="global_fit_studies.*groups"):
        prepare_project(state)


def test_missing_model_can_register_after_a_prepared_round_trip(monkeypatch, tmp_path):
    from asymmetry.core.fitting.composite import COMPONENTS

    state = project()
    state["schema_version"] = 38
    slot = state["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""]["fits"][
        0
    ]
    slot["model"] = {
        "component_names": ["Exponential", "UnavailableScale"],
        "operators": ["*"],
        "open_parentheses": [1, 0],
        "close_parentheses": [0, 1],
        "fraction_groups": [],
    }
    slot["parameters"] = [{"name": "A_1", "value": 2.0}, {"name": "A_2", "value": 3.0}]
    first = prepare_project(state)
    assert (
        first.model.representation(11, RepresentationType.TIME_FB_ASYMMETRY).fit.analysis_history[
            "reason"
        ]
        == "missing_model"
    )
    path = tmp_path / "missing.asymp"
    save_project(first.state, path)
    monkeypatch.setitem(COMPONENTS, "UnavailableScale", COMPONENTS["Gaussian"])
    resolved = (
        load_prepared_project(path)
        .model.representation(11, RepresentationType.TIME_FB_ASYMMETRY)
        .fit
    )
    assert resolved.analysis_history is None
    assert resolved.parameters == [{"name": "A_1", "value": 6.0}]
    assert resolved.result["computation_revision"] == "older"


def test_combined_released_slot_uses_the_same_migration_owner():
    state = _v24_project()
    entry = state["datasets"][0]
    state["datasets"] = []
    state["combined_datasets"] = [
        {
            "combined_run_number": -1,
            "source_run_numbers": [10, 20],
            "representations": entry["representations"],
        }
    ]
    # The v24 fit-set transition is dataset-owned in that released schema;
    # exercise the v38 analysis-name converter on an already saved set.
    state["schema_version"] = 38
    model = CompositeModel.from_expression("(Exponential * Gaussian)")
    slot = FitSlot(
        fit_id="combined",
        model=model.to_dict(),
        provenance="single",
        parameters=[{"name": "A_1", "value": 2}, {"name": "A_2", "value": 3}],
    )
    state["combined_datasets"][0]["representations"] = {
        "time_fb_asymmetry": {
            "single_fits": {"": {"fits": [slot.to_dict()], "open_id": "combined"}}
        }
    }
    prepared = prepare_project(state)
    restored = prepared.model.representation(-1, RepresentationType.TIME_FB_ASYMMETRY).fit
    assert restored.parameters == [{"name": "A_1", "value": 6.0}]


def test_profile_names_are_scoped_by_the_instrument_fingerprint():
    state = project()
    other = deepcopy(state["grouping_profiles"][0])
    other["fingerprint"]["instrument"] = "MUSR"
    state["grouping_profiles"].append(other)
    assert len(prepare_project(state).profiles) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("alpha_policy", {"value": []}),
        ("beta_policy", {"value": {}}),
        ("deadtime_policy", {"mode": "manual", "values": [[]]}),
    ],
)
def test_profile_policy_does_not_replace_malformed_numbers_with_defaults(key, value):
    state = project()
    state["grouping_profiles"][0][key] = value
    with pytest.raises(ValueError, match=f"grouping_profiles.*{key}"):
        prepare_project(state)


def test_global_blank_and_unset_draft_round_trip_is_exact(tmp_path):
    state = project()
    state["fit_states"] = {
        "time": {
            "domain": "time",
            "global_fit_state": {
                "composite_model": CompositeModel.from_expression("Exponential").to_dict(),
                "parameters": [{"name": "A", "value": ""}, {"name": "Lambda", "value": None}],
            },
        }
    }
    prepared = prepare_project(state)
    assert prepared.state == state
    path = tmp_path / "draft.asymp"
    save_project(state, path)
    assert load_project(path) == state


def test_nonfinite_scientific_uncertainty_and_archive_remain_supported(tmp_path):
    import math

    state = project()
    slot = state["datasets"][0]["representations"]["time_fb_asymmetry"]["single_fits"][""]["fits"][
        0
    ]
    slot["result"]["uncertainties"] = {"A_1": float("nan")}
    slot["analysis_history"] = {"reason": "authored", "original": {"uncertainty": float("inf")}}
    path = tmp_path / "uncertainty.asymp"
    save_project(state, path)
    restored = (
        load_prepared_project(path)
        .model.representation(11, RepresentationType.TIME_FB_ASYMMETRY)
        .fit
    )
    assert math.isnan(restored.result["uncertainties"]["A_1"])
    assert math.isinf(restored.analysis_history["original"]["uncertainty"])
