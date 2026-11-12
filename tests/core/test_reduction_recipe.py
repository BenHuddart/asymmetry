"""Effective scientific requests survive mutable live sources."""

import numpy as np

from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_workflow_reduction import _continuous_run


def test_effective_alpha_recipe_is_distinct_with_identical_metadata():
    run = _continuous_run(background_per_bin=0)
    one, two = [reduce_run(run, ReductionSettings(alpha=a, alpha_source="user")) for a in (1, 2)]
    assert one.metadata == two.metadata
    assert one.run is two.run is run
    assert not np.array_equal(one.asymmetry, two.asymmetry)
    assert one.histogram_observation.identity != two.histogram_observation.identity
    assert one.effective_recipe.identity != two.effective_recipe.identity


def test_recipe_capture_is_canonical_detached_and_survives_fit_transfer():
    import json
    import pickle
    from dataclasses import FrozenInstanceError

    import pytest

    from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe

    run = _continuous_run(background_per_bin=0)
    result = reduce_run(run, ReductionSettings(alpha=1.3))
    recipe = result.effective_recipe
    before = recipe.canonical_json
    run.grouping.update(alpha=9, groups={1: [2], 2: [1]})
    run.histograms[0].t0_bin = 3
    run.metadata["title"] = "Changed display"
    view = recipe.to_dict()
    view["periods"]["red"]["alpha"] = 9
    assert recipe.canonical_json == before
    reverse = {key: value for key, value in reversed(json.loads(before).items())}
    assert EffectiveReductionRecipe.from_dict(reverse).identity == recipe.identity
    assert pickle.loads(pickle.dumps(result.fit_record())).effective_recipe == recipe
    assert result.fit_record().histogram_observation is None
    with pytest.raises(FrozenInstanceError):
        recipe.canonical_json = "{}"


def test_actual_count_merge_crop_and_curve_mean_have_distinct_meanings():
    from asymmetry.core.data.combine import reduce_combined_run
    from asymmetry.core.transform.preparation import prepare_wizard_dataset
    from tests.core.test_binning_entry_points import _run

    result = reduce_combined_run(_run([90, 1, 80, 2, 70, 3], [10, 9, 20, 8, 30, 7]))
    crop = result.time_range(0.1, 0.4)
    counts = prepare_wizard_dataset(crop, 2)
    curve = crop.merge_curve_bins(2)
    np.testing.assert_allclose(counts.asymmetry, [100 * (81 - 29) / 110, 100 * (72 - 38) / 110])
    np.testing.assert_allclose(curve.asymmetry, crop.asymmetry.reshape(2, 2).mean(axis=1))
    assert counts.effective_recipe.identity != curve.effective_recipe.identity
    period = counts.effective_recipe.to_dict()["periods"]["red"]
    assert period["binning"]["selections"] == [[1, 5, 1], [0, 4, 2]]
    assert period["aligned_bin_edges"] == [1, 3, 5]
    assert (
        curve.effective_recipe.to_dict()["curve_operations"][-1]["operation"] == "mean_curve_bins"
    )
    assert crop.effective_recipe.identity != result.effective_recipe.identity


def test_historical_curve_never_acquires_histogram_provenance():
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.transform.preparation import prepare_wizard_dataset

    curve = MuonDataset(np.arange(6.0), np.arange(6.0), np.ones(6), run=_continuous_run(0))
    for result in (
        curve.time_range(1, 4),
        curve.merge_curve_bins(2),
        prepare_wizard_dataset(curve, 2),
    ):
        assert result.effective_recipe is None
        assert result.histogram_observation is None


def test_display_and_locator_changes_are_outside_numerical_identity():
    from dataclasses import replace

    run = _continuous_run(background_per_bin=0)
    first = reduce_run(run, ReductionSettings())
    run = replace(
        run,
        source_file="relocated.bin",
        metadata={**run.metadata, "title": "New", "run_label": "Other"},
    )
    second = reduce_run(run, ReductionSettings())
    assert first.effective_recipe.identity == second.effective_recipe.identity
    assert second.effective_recipe.to_dict()["periods"]["red"]["source"]["content_identity"] is None


def test_vector_rotation_records_both_captured_inputs_and_frame():
    from asymmetry.core.transform.projections import reduce_selected_projections
    from asymmetry.core.transform.rotating_frame import RotatingFrame, rotate_transverse
    from tests.core.test_projection_noise import shared_run

    pair = reduce_selected_projections(shared_run()[0])
    px, py = pair["P_x"], pair["P_y"]
    frame = RotatingFrame.typed_frequency(0.1, 1)
    rotated = rotate_transverse(px, py, frame, weights=(1,))
    assert all(ds.histogram_observation is None for ds in rotated)
    for ds in rotated:
        record = ds.effective_recipe.to_dict()
        assert record["inputs"] == [px.effective_recipe.to_dict(), py.effective_recipe.to_dict()]
        assert record["frame"]["frequency_mhz"] == 0.1
        assert ds.merge_curve_bins(2).effective_recipe.identity != ds.effective_recipe.identity


def test_scalar_scan_and_derivative_record_actual_observable_and_selection():
    from asymmetry.core.transform.integral import (
        build_field_scan,
        differentiate_scan,
        reduce_period_integral,
    )
    from tests.core.test_binning_entry_points import _run

    runs = [_run([90, 1], [10, 9]) for _ in range(3)]
    for number, run in enumerate(runs, 1):
        run.run_number = number
        run.metadata["field"] = number * 10.0
    integral = reduce_period_integral(runs[0])
    mean = reduce_period_integral(runs[0], method="differential")
    assert integral.effective_recipe.identity != mean.effective_recipe.identity
    assert integral.effective_recipe.to_dict()["units"] == "fraction"
    np.testing.assert_allclose(integral.value, 72 / 110)
    np.testing.assert_allclose(mean.value, 0.0)
    scan = build_field_scan(runs)
    selected = scan.select(np.array([0, 2]))
    derivative = differentiate_scan(selected)
    assert selected.effective_recipe.to_dict()["curve_operations"][-1]["indices"] == [0, 2]
    assert selected.points[1].effective_recipe == scan.points[2].effective_recipe
    facts = derivative.effective_recipe.to_dict()["curve_operations"][-1]
    assert facts["operation"] == "forward_difference"
    assert facts["fit_eligibility"] == "display_export_only"
    assert derivative.effective_recipe.identity != selected.effective_recipe.identity


def test_scan_recipe_keeps_known_noise_eligibility():
    from copy import deepcopy

    from asymmetry.core.transform.integral import build_field_scan
    from tests.core.test_binning_entry_points import _run

    run = _run([90, 1], [10, 9])
    run.metadata["field"] = 10.0
    shared = build_field_scan([run, run])
    independent = build_field_scan([run, deepcopy(run)])
    assert shared.correlated_pairs == ((0, 1),)
    assert independent.correlated_pairs == ()
    assert shared.effective_recipe.identity != independent.effective_recipe.identity


def test_fresh_fraction_representation_gets_its_actual_recipe():
    from asymmetry.core.representation.time import TimeFBAsymmetry
    from tests.core.test_binning_entry_points import _run

    result = TimeFBAsymmetry().compute(_run([90, 1], [10, 9]))[0]
    np.testing.assert_allclose(result.asymmetry, [0.8, -0.8])
    assert result.effective_recipe is not None
    assert result.effective_recipe.to_dict()["units"] == "fraction"


def test_equivalent_live_dictionary_order_produces_the_same_canonical_identity():
    from copy import deepcopy

    run = _continuous_run(0)
    reordered = deepcopy(run)
    reordered.grouping = dict(reversed(tuple(run.grouping.items())))
    reordered.grouping["groups"] = dict(reversed(tuple(run.grouping["groups"].items())))
    one = reduce_run(run, ReductionSettings())
    two = reduce_run(reordered, ReductionSettings())
    assert one.effective_recipe.identity == two.effective_recipe.identity
    np.testing.assert_array_equal(one.asymmetry, two.asymmetry)


def test_recipe_keeps_numeric_ratio_reason_and_omits_diagnostic_prose_from_identity():
    from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe

    usable = EffectiveReductionRecipe.from_dict({"version": 1, "observation": {"reason": 0}})
    zero_variance = EffectiveReductionRecipe.from_dict({"version": 1, "observation": {"reason": 2}})
    assert usable.identity != zero_variance.identity
    first = usable.with_facts(correction={"status": "skipped", "reason": "diagnosis one"})
    second = usable.with_facts(correction={"status": "skipped", "reason": "diagnosis two"})
    assert first.identity == second.identity


def test_raw_fraction_inspection_keeps_unknown_original_reduction_provenance():
    from asymmetry.core.data.combine import combine_runs
    from asymmetry.core.representation.time import TimeFBAsymmetry
    from tests.core.test_binning_entry_points import _run

    combined = combine_runs([_run([90, 80], [60, 40]), _run([30, 20], [10, 20])])
    output = TimeFBAsymmetry().compute(combined)[0]
    np.testing.assert_allclose(output.asymmetry, [(120 - 70) / 190, (100 - 60) / 160])
    assert output.effective_recipe is None
