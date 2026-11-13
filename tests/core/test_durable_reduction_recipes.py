"""Effective history travels with stored observations, without granting raw replay."""

import json

import numpy as np
import pytest

from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.transform.integral import build_field_scan, differentiate_scan
from asymmetry.core.workflow.integral_scan import field_scan_from_payload, field_scan_payload
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from asymmetry.core.workflow.workdir import ReducedEntry, WorkDir
from tests.core.test_workflow_reduction import _continuous_run


def test_workdir_retains_captured_recipe_and_curve_only_limits(tmp_path):
    dataset = reduce_run(_continuous_run(0), ReductionSettings(alpha=1.0, alpha_source="user"))
    entry = ReducedEntry(
        7,
        "owned",
        "owned.bin",
        dataset.n_points,
        ReductionSettings(),
        dict(dataset.metadata),
        1.0,
        "none",
        1,
        2,
        [],
    )
    workdir = WorkDir(tmp_path)
    workdir.write_reduced(dataset, entry)
    restored = workdir.reduced(7)
    assert restored.effective_recipe == dataset.effective_recipe
    assert restored.histogram_observation is None
    assert restored.counting_noise is None
    for stored, original in zip(
        (restored.time, restored.asymmetry, restored.error),
        (dataset.time, dataset.asymmetry, dataset.error),
    ):
        np.testing.assert_array_equal(stored, original)
    assert restored.metadata["correction_outcomes"] == dataset.metadata["correction_outcomes"]


@pytest.mark.parametrize("derivative", [False, True])
def test_scan_codecs_retain_point_and_scan_history_and_eligibility(derivative):
    datasets = [
        reduce_run(_continuous_run(i), ReductionSettings(alpha=1.0, alpha_source="user"))
        for i in range(3)
    ]
    for i, dataset in enumerate(datasets):
        dataset.metadata["run_number"] = i + 7
    scan = build_field_scan(datasets, method="integral", t_min=0.3, t_max=3.0, order_key="run")
    if derivative:
        scan = differentiate_scan(scan)
    restored = field_scan_from_payload(json.loads(json.dumps(field_scan_payload(scan))))
    assert restored.effective_recipe == scan.effective_recipe
    assert [p.effective_recipe for p in restored.points] == [
        p.effective_recipe for p in scan.points
    ]
    assert restored.derivative == scan.derivative
    assert restored.correlated_pairs == scan.correlated_pairs
    assert restored.observations == scan.observations
    assert restored.units == scan.units
    np.testing.assert_array_equal(restored.value, scan.value)


def test_legacy_workdir_and_scan_remain_recipe_unknown(tmp_path):
    dataset = reduce_run(_continuous_run(0), ReductionSettings())
    workdir = WorkDir(tmp_path)
    workdir.write_reduced(
        dataset,
        ReducedEntry(
            7,
            "owned",
            "owned.bin",
            dataset.n_points,
            ReductionSettings(),
            dict(dataset.metadata),
            1.0,
            "none",
            1,
            2,
            [],
        ),
    )
    path = workdir.reduced_dir / "7.json"
    sidecar = json.loads(path.read_text())
    sidecar.pop("effective_recipe")
    path.write_text(json.dumps(sidecar))
    assert workdir.reduced(7).effective_recipe is None
    scan = build_field_scan([dataset], method="integral", t_min=0.3, t_max=3.0, order_key="run")
    payload = field_scan_payload(scan)
    payload.pop("effective_recipe")
    restored = field_scan_from_payload(payload)
    assert restored.effective_recipe is None
    assert restored.points[0].effective_recipe is None


def test_nested_publishable_recipe_preserves_identity_without_mutating_history():
    from asymmetry.core.data.recipe_export import disclose_result_recipes, exported_recipe
    from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe

    recipe = EffectiveReductionRecipe.from_dict(
        {
            "version": 1,
            "units": "percent",
            "calibration": {
                "records": [
                    {
                        "source_file": "/private/calibration.bin",
                        "message": "failed at /private/calibration.bin",
                        "value": 1.2,
                    }
                ]
            },
            "reference": {
                "source_file": "/private/reference.bin",
                "reason": "from /private/reference.bin",
            },
            "constituents": [
                {"source": {"source_file": "/private/input.bin", "content_identity": None}}
            ],
            "reason": 2,
        }
    )
    payload = {"effective_recipe": recipe.to_dict(), "waterfall_offset": 3.0}
    public = disclose_result_recipes({"fits": [payload]})["fits"][0]
    assert "/private/" not in json.dumps(public)
    assert (
        EffectiveReductionRecipe.from_dict(public["effective_recipe"]).identity == recipe.identity
    )
    assert "/private/" in recipe.canonical_json
    exported = exported_recipe(public, x_range=(2.0, 1.0), bake_waterfall_offset=False)
    assert exported["curve_operations"][-1] == {
        "operation": "export_curve",
        "x_range": [1.0, 2.0],
        "applied_value_offset": 0.0,
        "plotted_value_offset": 3.0,
    }


@pytest.mark.parametrize(
    "change", ["counts", "errors", "support", "alpha", "title", "oversampling", "content"]
)
def test_fit_attribution_identifies_observations_separately_from_recipe_and_labels(change):
    import copy

    from asymmetry.core.data.ratio_validity import RatioValidity
    from asymmetry.core.fitting.input_attribution import fit_input_attribution
    from asymmetry.core.fitting.likelihood import dataset_observations

    run = _continuous_run(0)
    dataset = reduce_run(run, ReductionSettings(alpha=1.0, alpha_source="user"))
    other = copy.deepcopy(dataset)
    if change == "counts":
        run.histograms[0].counts[:] += 1
        other = reduce_run(run, ReductionSettings(alpha=1.0, alpha_source="user"))
        assert other.effective_recipe.identity == dataset.effective_recipe.identity
    elif change == "errors":
        other.error = other.error * 1.1
    elif change == "support":
        reasons = other.ratio_validity.reasons.copy()
        reasons[2] = 1
        other.ratio_validity = RatioValidity(reasons)
    elif change == "alpha":
        other = reduce_run(run, ReductionSettings(alpha=1.1, alpha_source="user"))
    elif change == "content":
        value = other.effective_recipe.to_dict()
        value["periods"]["red"]["source"]["content_identity"] = {"sha256": "a" * 64}
        other.effective_recipe = EffectiveReductionRecipe.from_dict(value)
    elif change == "title":
        other.metadata["title"] = "A renamed display label"
    original = fit_input_attribution(dataset_observations(dataset), dataset)
    changed = fit_input_attribution(
        dataset_observations(other),
        other,
        error_oversampling=2.0 if change == "oversampling" else 1.0,
    )
    assert (original["identity"] == changed["identity"]) == (change in {"title", "content"})


def test_real_fit_boundary_retains_recipe_and_exact_observations():
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.input_attribution import saved_fit_matches
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fitting.result_summary import fit_result_summary
    from asymmetry.core.representation.base import FitSlot

    dataset = reduce_run(_continuous_run(0), ReductionSettings(alpha=1.0, alpha_source="user"))
    crop = dataset.time_range(0.3, 0.5)
    fit = FitEngine().fit(
        crop, lambda t, c: np.full_like(t, c), ParameterSet([Parameter("c", 0.0)])
    )
    summary = fit_result_summary(fit)
    assert fit.success
    assert summary["input_attribution"]["effective_recipe"] == crop.effective_recipe.to_dict()
    slot = FitSlot(
        result=summary | {"npar": 1, "ndof": fit.dof}, fit_range={"min": 0.3, "max": 0.5}
    )
    assert saved_fit_matches(slot, dataset)
    weights = 1.0 / crop.error**2
    assert fit.parameters["c"].value == pytest.approx(
        np.average(crop.asymmetry, weights=weights), abs=1e-5
    )


@pytest.mark.parametrize("strategy", ["joint", "profiled", "least_squares"])
@pytest.mark.parametrize("oversampling", [1.0, 2.0])
def test_global_fit_attribution_matches_actual_window_and_statistical_assumptions(
    strategy, oversampling
):
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.input_attribution import fit_input_attribution
    from asymmetry.core.fitting.likelihood import dataset_observations
    from asymmetry.core.fitting.models import MODELS
    from tests.core.test_engine_profiled_fit import _make_series

    datasets, initial = _make_series(n_datasets=2, a0_true=20.0, lambdas=[0.3, 0.5], seed=21)
    results, _ = FitEngine().global_fit(
        datasets,
        MODELS["ExponentialRelaxation"].function,
        ["A0"],
        ["Lambda"],
        initial,
        t_min=0.5,
        t_max=6.0,
        strategy=strategy,
        error_oversampling=oversampling,
    )
    for dataset in datasets:
        result = results[dataset.run_number]
        assert result.success
        assert result.input_attribution == fit_input_attribution(
            dataset_observations(dataset, 0.5, 6.0),
            dataset,
            window=(0.5, 6.0),
            error_oversampling=oversampling,
        )
