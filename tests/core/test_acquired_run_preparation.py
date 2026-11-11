"""Independent acquired-run arithmetic oracles through actual reduction consumers."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.combine import combine_runs, reduce_combined_run
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.count_domain import build_count_groups
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.io.periods import combine_mapped_periods, period_run
from asymmetry.core.transform.preparation import (
    histogram_observation,
    prepare_histogram_asymmetry,
    prepare_wizard_dataset,
)


def acquired(number, pair, *, anchor=0, correct=False, frames=100000, tau=0.05):
    histograms = [
        Histogram(
            np.full(16, v, dtype=float),
            0.1,
            t0_bin=anchor,
            good_bin_start=anchor,
            good_bin_end=anchor + 8,
        )
        for v in pair
    ]
    return Run(
        number,
        histograms=histograms,
        source_file=f"synthetic-{number}.nxs",
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "beta": 1.0,
            "good_frames": frames,
            "dead_time_us": [tau, tau],
            "deadtime_file_us": [tau, tau],
            "deadtime_correction": correct,
            "deadtime_mode": "file",
            "background_correction": False,
            "good_window_policy": {"mode": "from_file"},
            "t0_source": "file",
        },
    )


def prepare(run, merge=1):
    return prepare_histogram_asymmetry(run, merge=merge, facility=None, reference=None)


def own_corrected(runs, coefficients):
    """Published inverse N/(1-N tau/(F dt)); noise is retained acquired variance."""
    values, noise = np.zeros(2), np.zeros(2)
    for run, c in zip(runs, coefficients, strict=True):
        for i, h in enumerate(run.histograms):
            n = h.counts[h.t0_bin]
            tau = run.grouping["deadtime_file_us"][i]
            values[i] += c * n / (1 - n * tau / (run.grouping["good_frames"] * h.bin_width))
            noise[i] += c * c * (n if h.count_variance is None else h.count_variance[h.t0_bin])
    return values, noise


def ratio(values, noise, alpha=1.0, beta=1.0, bins=1):
    f, b = values
    vf, vb = noise / bins
    denominator = beta * f + alpha * b
    return (
        100 * (f - alpha * b) / denominator,
        100 * alpha * (1 + beta) * np.sqrt(b * b * vf + f * f * vb) / denominator**2,
    )


def test_correct_each_acquisition_before_pooling():
    runs = [acquired(1, (80000, 20000), correct=True), acquired(2, (10000, 90000), correct=True)]
    values, noise = own_corrected(runs, [1, 1])
    expected, error = ratio(values, noise)
    assert expected == pytest.approx(-12.737826507578193)
    result = reduce_combined_run(combine_runs(runs))
    np.testing.assert_allclose(result.asymmetry, expected, atol=1e-10)
    np.testing.assert_allclose(result.error, error)


@pytest.mark.parametrize("coefficients", [(1.0, 1.0), (2.0, -0.5), (0.2, 0.7), (-1.0, -2.0)])
@pytest.mark.parametrize("supplied", [False, True])
def test_unequal_original_facts_weights_noise_and_permutation(coefficients, supplied):
    runs = [
        acquired(1, (8000, 2000), correct=True, frames=100000, tau=0.1),
        acquired(2, (1000, 9000), correct=True, frames=200000, tau=0.4),
    ]
    if supplied:
        for run in runs:
            run.histograms = [
                replace(h, count_variance=np.full(16, v))
                for h, v in zip(run.histograms, (25.0, 49.0), strict=True)
            ]
    values, noise = own_corrected(runs, coefficients)
    expected, error = ratio(values, noise, alpha=0.7, beta=1.3, bins=3)
    for sources, weights in ((runs, coefficients), (runs[::-1], coefficients[::-1])):
        combined = combine_runs(sources, scales=weights)
        combined.grouping.update(alpha=0.7, beta=1.3)
        result = prepare(combined, merge=3)
        np.testing.assert_allclose(result.asymmetry, expected)
        np.testing.assert_allclose(result.error, abs(error))
        assert result.time.size == 3


def test_own_clocks_and_file_good_support_precede_pooling():
    runs = [acquired(1, (60, 40), anchor=0), acquired(2, (90, 10), anchor=3)]
    result = prepare(combine_runs(runs))
    np.testing.assert_allclose(result.time, np.arange(9) * 0.1, atol=1e-14)
    np.testing.assert_allclose(result.asymmetry, 50.0, atol=1e-12)


@pytest.mark.parametrize("consumer", ["count", "fourier"])
def test_group_consumers_use_original_correction_and_common_support(consumer):
    runs = [
        acquired(1, (8000, 2000), correct=True),
        acquired(2, (1000, 9000), anchor=3, correct=True, frames=200000, tau=0.1),
    ]
    combined = combine_runs(runs, scales=[2, 0.5])
    expected, noise = own_corrected(runs, [2, 0.5])
    data = reduce_combined_run(combined)
    if consumer == "count":
        group = build_count_groups(data, [1], lifetime_corrected=False)[0]
        time, values, errors = group.time, group.counts, group.error
    else:
        signal = build_group_signal_dataset(
            combined, 1, center_signal=False, apply_lifetime_correction=False
        )
        time, values, errors = signal.time, signal.asymmetry, signal.error
    np.testing.assert_allclose(time, np.arange(9) * 0.1, atol=1e-14)
    np.testing.assert_allclose(values, expected[0])
    np.testing.assert_allclose(errors, np.sqrt(noise[0]))


def test_snapshot_replay_owns_originals_and_new_source_edits_change_identity():
    runs = [acquired(1, (8000, 2000), correct=True), acquired(2, (1000, 9000), correct=True)]
    combined = combine_runs(runs)
    data = reduce_combined_run(combined)
    snapshot = data.histogram_observation
    assert (
        snapshot.source.grouping["period_inputs"][0].histograms[0].counts
        is not runs[0].histograms[0].counts
    )
    assert not snapshot.source.grouping["period_inputs"][0].histograms[0].counts.flags.writeable
    before = prepare_wizard_dataset(data, 3)
    runs[0].histograms[0].counts[:] += 100
    runs[0].grouping["deadtime_file_us"][0] = 0.07
    assert histogram_observation(combined).identity != snapshot.identity
    replay = prepare_wizard_dataset(data, 3)
    np.testing.assert_array_equal(replay.asymmetry, before.asymmetry)
    assert data.fit_record().run.grouping.get("period_inputs") is None


@pytest.mark.parametrize("missing", [False, True])
def test_zero_weight_source_has_no_correction_or_noise_requirements(missing):
    runs = [acquired(1, (60, 40), correct=True), acquired(2, (-10, 1e10), correct=True)]
    if missing:
        runs[1].grouping["deadtime_file_us"] = []
    result = prepare(combine_runs(runs, scales=[1, 0]))
    np.testing.assert_allclose(result.asymmetry, prepare(runs[0]).asymmetry)


def test_irrelevant_bins_and_detectors_do_not_enter_saturation_domain():
    runs = [acquired(1, (60, 40), correct=True), acquired(2, (90, 10), correct=True)]
    runs[0].histograms[0].counts[15] = 1e10
    for run in runs:
        run.histograms.append(Histogram(np.full(16, 1e10), 0.1))
        run.grouping["deadtime_file_us"].append(0.05)
    assert prepare(combine_runs(runs)).time.size == 9
    combined = combine_runs(runs)
    combined.grouping["groups"][1] = [1, 3]
    with pytest.raises(ValueError, match="saturat|occupancy|denominator"):
        prepare(combined)


def test_missing_required_file_table_skips_entire_result():
    runs = [acquired(1, (8000, 2000), correct=True), acquired(2, (1000, 9000), correct=True)]
    runs[1].grouping["deadtime_file_us"] = []
    runs[1].grouping["dead_time_us"] = []
    with pytest.warns(UserWarning):
        result = prepare(combine_runs(runs))
    np.testing.assert_allclose(result.asymmetry, -10.0)
    assert result.corrections["red"].deadtime.status == "skipped"


def test_counterpart_periods_keep_file_identity_and_own_clocks():
    runs = [
        acquired(1, (8000, 2000), correct=True),
        acquired(2, (1000, 9000), anchor=3, correct=True),
    ]
    for run in runs:
        run.grouping.update(
            period_histograms=[
                run.histograms,
                [replace(h, counts=h.counts * 2) for h in run.histograms],
            ],
            period_good_frames=[100000, 300000],
            period_dead_time_us=[[0.05] * 2, [0.1] * 2],
            period_mode="green",
        )
    combined = combine_runs(runs, scales=[2, 0.5])
    green = period_run(combined, 1)
    result = prepare(green)
    inputs = green.grouping["period_inputs"]
    assert {r.source_file for r in inputs} == {r.source_file for r in runs}
    assert [r.metadata["data_period_number"] for r in inputs] == [1, 2, 1, 2]
    np.testing.assert_allclose(result.time, np.arange(9) * 0.1, atol=1e-14)
    values = sum(
        c
        * np.array([8000, 2000] if j == 0 else [1000, 9000])
        * 2
        / (1 - np.array([8000, 2000] if j == 0 else [1000, 9000]) * 2 * 0.1 / (300000 * 0.1))
        for j, c in enumerate([2, 0.5])
    )
    np.testing.assert_allclose(result.asymmetry, 100 * (values[0] - values[1]) / values.sum())


def test_live_mapped_sets_and_ignored_acquisitions_keep_coefficients():
    datasets = []
    for number in (1, 2):
        originals = [
            acquired(number * 10 + i, pair, anchor=i, correct=True)
            for i, pair in enumerate([(8000, 2000), (1000, 9000), (-10, 1e10)])
        ]
        datasets.append(
            combine_mapped_periods(
                [MuonDataset(np.zeros(1), np.zeros(1), np.ones(1), run=r) for r in originals],
                {1: "red", 2: "green", 3: "ignore"},
            )
        )
    combined = combine_runs([d.run for d in datasets], scales=[2, 0.5])
    combined.grouping["period_mode"] = "green_plus_red"
    result = prepare(combined)
    assert result.time.size == 9
    values, _ = own_corrected([acquired(1, (8000, 2000)), acquired(2, (1000, 9000))], [2.5, 2.5])
    np.testing.assert_allclose(result.asymmetry, 100 * (values[0] - values[1]) / values.sum())


@pytest.mark.parametrize("weights,known", [([1, 1], True), ([1, 2], False), ([1, -1], False)])
def test_measured_exposure_is_separate_from_deadtime_normalisation(weights, known):
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    runs = [acquired(1, (60, 40)), acquired(2, (90, 10))]
    for run, exposure in zip(runs, [100, 200], strict=True):
        run.grouping["reference_exposure"] = ReferenceExposure(
            exposure, "synthetic_frames"
        ).to_dict()
    combined = combine_runs(runs, scales=weights)
    assert ReferenceExposure.from_grouping(combined.grouping).value == (300 if known else None)
    for run in runs:
        run.grouping.pop("reference_exposure")
        run.grouping["good_frames"] = 1
    assert ReferenceExposure.from_grouping(combine_runs(runs).grouping).value is None


@pytest.mark.parametrize("consumer", ["asymmetry", "fourier"])
def test_pooled_group_background_is_applied_once(consumer):
    runs = [acquired(1, (60, 40)), acquired(2, (90, 10))]
    combined = combine_runs(runs)
    combined.grouping.update(
        background_correction=True, background_mode="fixed", background_fixed_values=[10.0, 5.0]
    )
    if consumer == "asymmetry":
        np.testing.assert_allclose(prepare(combined).asymmetry, 100 * (140 - 45) / (140 + 45))
    else:
        signal = build_group_signal_dataset(
            combined, 1, center_signal=False, apply_lifetime_correction=False
        )
        np.testing.assert_allclose(signal.asymmetry, 140)


def test_reference_is_corrected_and_subtracted_once_after_pooled_sample():
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    runs = [acquired(1, (8000, 2000), correct=True), acquired(2, (1000, 9000), correct=True)]
    combined = combine_runs(runs)
    combined.grouping.update(background_correction=True, background_mode="reference_run")
    ref = acquired(3, (100, 50), anchor=2, frames=200000, tau=0.2)
    reference = BackgroundReference(
        ref.histograms,
        ReferenceScaleRequest("manual", 2.0).resolve(),
        deadtime_normaliser=200000,
        file_dead_time_us=(0.2, 0.2),
        source_file=ref.source_file,
        run_number=3,
    )
    result = prepare_histogram_asymmetry(combined, merge=1, facility=None, reference=reference)
    sample, noise = own_corrected(runs, [1, 1])
    reference_values, reference_noise = own_corrected([ref], [1])
    value, error = ratio(sample - 2 * reference_values, noise + 4 * reference_noise)
    np.testing.assert_allclose(result.asymmetry, value)
    np.testing.assert_allclose(result.error, error)


def test_actual_output_packing_excludes_unknown_tail_before_correction():
    runs = [acquired(1, (60, 40), correct=True), acquired(2, (90, 10), correct=True)]
    runs[1].histograms[0].counts[8] = 0.5
    combined = combine_runs(runs)
    assert prepare(combined, merge=2).time.size == 4
    with pytest.raises(ValueError, match="unknown independent count noise"):
        prepare(combined)


def test_manual_timing_offsets_keep_own_file_clocks_and_fractional_residuals():
    runs = [acquired(1, (60, 40), anchor=1), acquired(2, (90, 10), anchor=3)]
    runs[0].histograms = [replace(h, t0_time_us=0.14) for h in runs[0].histograms]
    runs[1].histograms = [replace(h, t0_time_us=0.36) for h in runs[1].histograms]
    combined = combine_runs(runs)
    combined.grouping["t0_policy"] = {"mode": "manual", "offset_bins": 1}
    result = prepare(combined)
    np.testing.assert_allclose(result.time, np.arange(9) * 0.1 - 0.11, atol=1e-14)
    assert result.alignments["red"].timing_spread_us == pytest.approx(0.02)


def test_extracted_green_runs_keep_acquired_period_identity_and_measured_exposure():
    from asymmetry.core.transform.reference_scale import ReferenceExposure

    runs = [acquired(1, (60, 40)), acquired(2, (90, 10))]
    for run in runs:
        run.metadata["period_number"] = 2
        run.grouping.update(
            period_mode="green",
            reference_exposure=ReferenceExposure(
                100, "synthetic_frames", period_number=2
            ).to_dict(),
        )
    combined = combine_runs(runs)
    assert ReferenceExposure.from_grouping(combined.grouping).value == 200
    assert (
        combined.grouping["period_inputs"][0].histograms[0].counts is runs[0].histograms[0].counts
    )
    result = prepare(combined)
    assert {period for _, period, _ in result.alignments["green"].acquired_detectors} == {2}


def test_nonrepresentable_fitted_deadtime_has_typed_actionable_diagnosis():
    from asymmetry.core.fitting.count_inputs import count_model_frame_norm
    from asymmetry.core.fitting.likelihood import FitInputError

    data = reduce_combined_run(combine_runs([acquired(1, (60, 40)), acquired(2, (90, 10))]))
    group = build_count_groups(data, [1], lifetime_corrected=False)[0]
    with pytest.raises(FitInputError, match="original acquisitions|asymmetry fit"):
        count_model_frame_norm(data, group)


def test_raw_unit_sum_preserves_poisson_on_narrow_acquired_good_support():
    from asymmetry.core.fitting.grouped_time_domain import group_observations
    from asymmetry.core.fitting.likelihood import CountSource, Objective

    combined = combine_runs([acquired(1, (60, 40)), acquired(2, (90, 10), anchor=3)])
    group = build_count_groups(reduce_combined_run(combined), [1], lifetime_corrected=False)[0]
    assert group.count_source is CountSource.RAW
    assert len(group_observations(group, Objective.POISSON).time) == 9


def test_extracted_counterpart_quicklook_applies_signed_coefficients_once():
    from asymmetry.core.io.periods import select_period

    runs = [acquired(1, (100, 40)), acquired(2, (20, 10), anchor=3)]
    for run in runs:
        run.grouping.update(
            period_histograms=[
                run.histograms,
                [replace(h, counts=h.counts * 2) for h in run.histograms],
            ],
            period_mode="red",
        )
    combined = combine_runs(runs, scales=[2, -0.5])
    selected = select_period(reduce_combined_run(combined), 2)
    np.testing.assert_allclose(selected.asymmetry, 100 * (380 - 150) / (380 + 150))
    np.testing.assert_allclose(selected.time, np.arange(9) * 0.1, atol=1e-14)


def test_conflicting_derived_reference_requests_require_a_shared_target():
    from asymmetry.core.data.combine import CombineError

    combined = [combine_runs([acquired(i, (60, 40)), acquired(i + 1, (90, 10))]) for i in (1, 3)]
    for run, reference in zip(combined, (101, 102), strict=True):
        run.grouping.update(
            background_correction=True,
            background_mode="reference_run",
            background_run={"run_number": reference, "scale_request": {"mode": "automatic"}},
        )
    with pytest.raises(CombineError, match="one shared reference"):
        combine_runs(combined)
