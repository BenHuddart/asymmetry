"""Acquired and mapped observations retain resolved recipe facts."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.combine import combine_runs, reduce_combined_run
from asymmetry.core.io.periods import combine_mapped_periods, select_period
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
from asymmetry.core.transform.reference_scale import ReferenceScaleRequest
from tests.core.test_binning_entry_points import _run
from tests.core.test_constituent_preparation import acquired


def prepare(run, reference=None):
    return prepare_histogram_asymmetry(run, merge=1, facility=None, reference=reference)


@pytest.mark.parametrize("change", ["alpha", "beta", "groups", "t0", "support", "binning"])
def test_effective_request_facts_change_identity_without_changing_the_source(change):
    run = _run([90, 20, 80, 30, 70, 40], [10, 80, 20, 70, 30, 60])
    before = prepare(run)
    if change in {"alpha", "beta"}:
        run.grouping[change] = 1.2
    elif change == "groups":
        run.grouping.update(groups={1: [2], 2: [1]})
    elif change == "t0":
        run.grouping["effective_detector_t0_bins"] = [1, 1]
    elif change == "support":
        run.grouping["first_good_bin"] = 1
    else:
        run.grouping["bunching_factor"] = 2
    after = prepare(run)
    assert before.effective_recipe.identity != after.effective_recipe.identity
    assert before.effective_recipe.to_dict()["periods"]["red"]["alpha"] == 1


@pytest.mark.parametrize("missing", [False, True])
def test_actual_applied_or_skipped_corrections_are_recorded(missing):
    source = acquired(1, [600] * 8, [400] * 8).run
    if missing:
        source.grouping["deadtime_file_us"] = []
        source.grouping["dead_time_us"] = []
        with pytest.warns(UserWarning, match="not applied"):
            result = prepare(source)
    else:
        result = prepare(source)
    facts = result.effective_recipe.to_dict()["periods"]["red"]
    assert facts["corrections"]["deadtime"] == result.corrections["red"].deadtime.to_dict()
    assert facts["corrections"]["deadtime"]["status"] == ("skipped" if missing else "applied")
    assert facts["uncertainty"]["calibration"] == "conditional"


def test_reference_scale_source_and_clock_belong_to_the_actual_result():
    source = _run([90] * 8, [60] * 8, background_correction=True, background_mode="reference_run")
    reference = BackgroundReference(
        [replace(h, counts=np.full(8, 10.0)) for h in source.histograms],
        ReferenceScaleRequest("manual", 0.5).resolve(),
        run_number=99,
        period_number=2,
    )
    first = prepare(source, reference)
    second = prepare(
        source, replace(reference, scaling=ReferenceScaleRequest("manual", 0.25).resolve())
    )
    other = prepare(source, replace(reference, run_number=100))
    assert len({r.effective_recipe.identity for r in (first, second, other)}) == 3
    facts = first.effective_recipe.to_dict()["periods"]["red"]
    assert facts["reference"]["scale"] == 0.5
    assert facts["reference"]["period_number"] == 2
    assert "reference_alignment" in facts["corrections"]["background"]["effective"]
    np.testing.assert_allclose(first.asymmetry, 100 * (85 - 55) / 140)


def test_mapped_periods_keep_acquired_coefficients_and_cached_selection():
    sources = [acquired(i, [60] * 8, [40] * 8) for i in (1, 2, 3)]
    for source in sources:
        source.run.grouping["deadtime_correction"] = False
    mapped = combine_mapped_periods(sources, {1: "red", 2: "red", 3: "green"})
    first = prepare(mapped.run)
    mapped.run.grouping["period_mode"] = "green_minus_red"
    difference = prepare(mapped.run)
    record = difference.effective_recipe.to_dict()
    assert record["period_coefficients"] == {"red": -1, "green": 1}
    assert [m["input_number"] for m in record["periods"]["red"]["constituents"]] == [1, 2]
    assert [m["input_number"] for m in record["periods"]["green"]["constituents"]] == [3]
    assert first.effective_recipe.identity != difference.effective_recipe.identity
    assert select_period(mapped, "green").effective_recipe is not None


def test_signed_constituents_record_weights_before_ratio_and_supplied_noise():
    runs = [acquired(i, [90] * 8, [60] * 8).run for i in (1, 2)]
    for run in runs:
        run.grouping["deadtime_correction"] = False
    combined = combine_runs(runs, sign=-1, scales=(1, 0.5))
    result = reduce_combined_run(combined)
    facts = result.effective_recipe.to_dict()["periods"]["red"]
    assert [m["coefficient"] for m in facts["constituents"]] == [1, -0.5]
    assert all(m["acquired_arithmetic"] for m in facts["constituents"])
    np.testing.assert_allclose(result.asymmetry, 20)


def test_quick_recipe_reports_effective_manual_clock():
    from asymmetry.core.data.dataset import Histogram
    from asymmetry.core.io.quick_asymmetry import quick_asymmetry

    quick = quick_asymmetry(
        [Histogram(np.full(8, v), 0.1, t0_bin=2, t0_time_us=0.25) for v in (60, 40)],
        {"groups": {1: [1], 2: [2]}, "t0_policy": {"mode": "manual", "offset_bins": 2}},
    )
    assert quick.unavailable_reason is None
    assert quick.file_grouping["effective_detector_t0_bins"] == [4, 4]
    recipe = quick.effective_recipe.to_dict()
    assert [h["t0_bin"] for h in recipe["constituents"][0]["detectors"]] == [4, 4]


def test_range_estimator_noise_is_not_declared_fixed():
    run = _run(
        [8, 8, 100, 100, 100, 100],
        [8, 8, 80, 80, 80, 80],
        first_good_bin=2,
        last_good_bin=5,
        background_correction=True,
        background_mode="range",
        background_ranges=[0, 1],
    )
    result = prepare_histogram_asymmetry(run, merge=4, facility=None, reference=None)
    f, b = 368, 288
    vf, vb = 400 + 16 * 4, 320 + 16 * 4
    np.testing.assert_allclose(
        result.error, [200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2]
    )
    recipe = result.effective_recipe.to_dict()["periods"]["red"]
    assert recipe["uncertainty"].get("correction_parameters") != "conditional"
