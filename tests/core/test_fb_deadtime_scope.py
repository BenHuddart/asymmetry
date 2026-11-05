"""Required raw observations define the F/B inverse and File availability."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.corrections import CorrectionStatus
from asymmetry.core.transform.deadtime import DeadtimeCorrectionError
from asymmetry.core.transform.reduce import corrected_grouped_counts


def reduce_counts(arrays, table, *, first=0, last=None, backward=1, background=False, mode="file"):
    histograms = [Histogram(np.asarray(a, dtype=float), bin_width=1, t0_bin=0) for a in arrays]
    grouping = {
        "first_good_bin": first,
        "last_good_bin": len(arrays[0]) - 1 if last is None else last,
        "dead_time_us": table,
        "good_frames": 100,
        "background_mode": "range",
        "background_range": [0, 1],
    }
    result = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[backward],
        use_deadtime=True,
        deadtime_mode=mode,
        use_background=background,
        facility="",
    )
    for histogram, original in zip(histograms, arrays, strict=True):
        np.testing.assert_array_equal(histogram.counts, original)
        assert histogram.count_variance is None
    return result


@pytest.mark.parametrize("defect", ["detector", "prefix", "gap", "missing"])
def test_unused_observations_do_not_block_or_suppress_correction(defect):
    arrays = [[10, 8, 6], [5, 4, 3], [100, 100, 100]]
    table = [0.1, -0.1, 1.0]
    options = {}
    expected = np.array([10, 8, 6]) / (1 - np.array([10, 8, 6]) * 0.1 / 100)
    variance = [10, 8, 6]
    if defect == "prefix":
        arrays = [[100, 10, 8, 6], [50, 5, 4, 3]]
        table = [1.0, -0.1]
        options = {"first": 1}
        expected = np.array([10, 8, 6]) / (1 - np.array([10, 8, 6]) / 100)
    elif defect == "gap":
        arrays = [[2, 2, 100, 10, 8, 6], [1, 1, 5, 5, 4, 3]]
        table = [1.0, -0.1]
        options = {"first": 3, "background": True}
        expected = np.array([10, 8, 6]) / (1 - np.array([10, 8, 6]) / 100) - 2 / 0.98
    elif defect == "missing":
        table = [0.1, -0.1, None]
    result = reduce_counts(arrays, table, **options)
    assert result.corrections.deadtime.status == CorrectionStatus.APPLIED
    np.testing.assert_allclose(result.forward.counts[result.window.slice], expected, rtol=1e-13)
    np.testing.assert_array_equal(result.forward.variance[result.window.slice], variance)
    assert result.window.first == options.get("first", 0)
    assert result.alignment.detector_offsets == ((0, 0), (1, 0))


@pytest.mark.parametrize("defect", ["detector", "prefix", "gap", "missing"])
def test_required_observation_controls(defect):
    if defect == "missing":
        with pytest.warns(UserWarning, match="File deadtime correction not applied"):
            result = reduce_counts([[10, 8], [5, 4], [4, 4]], [0.1, -0.1, None], backward=2)
        assert result.corrections.deadtime.status == CorrectionStatus.SKIPPED
        np.testing.assert_array_equal(result.forward.counts, [10, 8])
        return
    arrays, table, options, detector, index = {
        "detector": ([[10, 8], [5, 4], [100, 100]], [0.1, -0.1, 1], {"backward": 2}, 2, 0),
        "prefix": ([[100, 10, 8], [50, 5, 4]], [1, -0.1], {}, 0, 0),
        "gap": ([[2, 2, 100, 10], [1, 1, 5, 5]], [1, -0.1], {"first": 2, "background": True}, 0, 2),
    }[defect]
    with pytest.raises(DeadtimeCorrectionError) as error:
        reduce_counts(arrays, table, **options)
    assert error.value.detector == detector
    assert error.value.bin_index == index


@pytest.mark.parametrize("policy", ["from_file", "manual"])
def test_staggered_timing_and_inclusive_window_preserve_original_coordinates(policy):
    from asymmetry.core.transform.grouping import group_forward_backward
    from asymmetry.core.transform.t0 import EFFECTIVE_DETECTOR_T0_KEY

    histograms = [
        Histogram(
            np.array([100, 10, 11, 12, 100]),
            1,
            t0_bin=1,
            good_bin_start=1,
            good_bin_end=3,
            count_variance=np.arange(5.0) + 2,
        ),
        Histogram(
            np.array([100, 100, 5, 6, 7, 100]),
            1,
            t0_bin=2,
            good_bin_start=2,
            good_bin_end=4,
            count_variance=np.arange(6.0) + 4,
        ),
        Histogram(np.full(8, 100.0), 1, t0_bin=0),
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "dead_time_us": [1, 1, None],
        "good_frames": 100,
        "detector_t0_bins": [1, 2, 0],
        "detector_first_good_bins": [1, 2, 0],
        "detector_last_good_bins": [3, 4, 7],
        EFFECTIVE_DETECTOR_T0_KEY: [1, 2, 0],
        "good_window_policy": {"mode": policy},
    }
    if policy == "manual":
        grouping["good_window_policy"].update(first_offset_bins=0, last_offset_bins=2)
    raw = group_forward_backward(histograms, grouping).counts
    result = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    assert result.alignment == raw.alignment
    assert result.window == raw.window
    assert result.t0_time_us == raw.t0_time_us
    for index, group in enumerate((result.forward, result.backward)):
        start = result.window.first - dict(result.alignment.detector_offsets)[index]
        bins = np.arange(start, start + 3)
        observed = histograms[index].counts[bins]
        np.testing.assert_allclose(
            group.counts[result.window.slice], observed / (1 - observed / 100)
        )
        np.testing.assert_array_equal(
            group.variance[result.window.slice], histograms[index].count_variance[bins]
        )
    scope = result.corrections.deadtime.to_dict()["effective"]["scope"]["raw_bin_ranges"]
    assert scope == {"0": [[1, 3]], "1": [[2, 4]]}


@pytest.mark.parametrize("required_missing", [False, True])
@pytest.mark.parametrize("period_mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_period_cohort_uses_each_required_domain(period_mode, required_missing):
    from contextlib import nullcontext

    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    periods = [
        [
            Histogram(np.array(a, dtype=float), 1, t0_bin=t0)
            for a in ([100, 10, 8, 6], [100, 5, 4, 3], [100] * 4)
        ]
        for t0 in (0, 0)
    ]
    tables = [[1, -0.1, None], [0.5, -0.2, None]]
    if required_missing:
        tables[1][1] = None
    grouping = {
        "period_histograms": periods,
        "period_dead_time_us": tables,
        "period_good_frames": [100, 200],
        "first_good_bin": 1,
        "last_good_bin": 3,
    }
    options = dict(
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        deadtime_mode="file",
        use_background=False,
    )
    skipped = required_missing and period_mode != "red"
    with pytest.warns(UserWarning, match="Continuing uncorrected") if skipped else nullcontext():
        result = reduce_period_asymmetry(
            periods[0], grouping, period_mode, use_deadtime=True, **options
        )
    assert all(
        c.deadtime.status == ("skipped" if skipped else "applied")
        for c in result.corrections.values()
    )
    expected, variances = [], []
    for index in (0, 1):
        f, b = np.array([10.0, 8.0, 6.0]), np.array([5.0, 4.0, 3.0])
        if not skipped:
            f = f / (1 - f * tables[index][0] / [100, 200][index])
            if tables[index][1] is not None:
                b = b / (1 - b * tables[index][1] / [100, 200][index])
        expected.append(100 * (f - b) / (f + b))
        variances.append(
            100**2
            * ((2 * b / (f + b) ** 2) ** 2 * [10, 8, 6] + (2 * f / (f + b) ** 2) ** 2 * [5, 4, 3])
        )
    oracle = {
        "red": expected[0],
        "green": expected[1],
        "green_minus_red": expected[1] - expected[0],
        "green_plus_red": expected[1] + expected[0],
    }
    np.testing.assert_allclose(result.asymmetry, oracle[period_mode])
    noise = (
        variances[0]
        if period_mode == "red"
        else variances[1]
        if period_mode == "green"
        else variances[0] + variances[1]
    )
    np.testing.assert_allclose(result.error, np.sqrt(noise))


@pytest.mark.parametrize("required_missing", [False, True])
def test_reference_uses_own_exposure_coefficients_and_selected_samples(required_missing):
    from contextlib import nullcontext

    from asymmetry.core.transform.corrections import BackgroundReference

    sample = [
        Histogram(np.array(a, dtype=float), 1) for a in ([100, 10, 8], [100, 5, 4], [100] * 3)
    ]
    reference = BackgroundReference(
        [Histogram(np.array(a, dtype=float), 1) for a in ([100, 2, 2], [100, 1, 1], [100] * 3)],
        scale=2,
        good_frames_reference=50,
        file_dead_time_us=(0.5, None if required_missing else -0.1, None),
    )
    grouping = {
        "dead_time_us": [1, 0.1, None],
        "good_frames": 100,
        "first_good_bin": 1,
        "last_good_bin": 2,
        "background_mode": "reference_run",
    }
    with (
        pytest.warns(UserWarning, match="Continuing uncorrected")
        if required_missing
        else nullcontext()
    ):
        result = corrected_grouped_counts(
            histograms=sample,
            grouping=grouping,
            forward_idx=[0],
            backward_idx=[1],
            use_deadtime=True,
            deadtime_mode="file",
            use_background=True,
            reference_resolver=lambda _: reference,
        )
    assert result.corrections.deadtime.status == ("skipped" if required_missing else "applied")
    expected = (
        np.array([10, 8]) - 4
        if required_missing
        else np.array([10, 8]) / (1 - np.array([10, 8]) / 100) - 4 / 0.98
    )
    np.testing.assert_allclose(result.forward.counts[result.window.slice], expected)
    np.testing.assert_array_equal(result.forward.variance[result.window.slice], [18, 16])
    record = result.corrections.background.to_dict()["effective"]["reference"]["deadtime"]
    assert record["status"] == ("skipped" if required_missing else "applied")
    if not required_missing:
        assert record["effective"]["good_frames"] == 50
        assert record["effective"]["tau_us"] == [0.5, -0.1, None]


def test_tail_estimator_samples_outside_output_are_required_but_inactive_ranges_are_not():
    histograms = [Histogram(np.full(30, value), 1) for value in (10.0, 5.0)]
    histograms[0].counts[15] = 100
    grouping = {
        "first_good_bin": 25,
        "last_good_bin": 29,
        "dead_time_us": [1, 1],
        "good_frames": 100,
        "background_mode": "tail_fit",
        "background_range": [0, 200],
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="manual",
    )
    result = corrected_grouped_counts(**options, use_background=False)
    assert result.corrections.deadtime.status == "applied"
    with pytest.raises(DeadtimeCorrectionError) as error:
        corrected_grouped_counts(**options, use_background=True)
    assert error.value.bin_index == 15


def test_unused_manual_values_remain_validated_but_exposure_is_only_required_for_nonzero_selected_coefficients():
    result = reduce_counts([[10, 8], [5, 4], [100, 100]], [0, 0, 1])
    assert result.corrections.deadtime.effective["good_frames"] is None
    with pytest.raises(DeadtimeCorrectionError, match="detector 3"):
        reduce_counts([[10, 8], [5, 4], [100, 100]], [0, 0, float("nan")], mode="manual")


@pytest.mark.parametrize("method", ["integral", "differential"])
def test_integral_explicit_window_excludes_unused_saturation(method):
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.transform.integral import integrate_run

    run = Run(
        histograms=[
            Histogram(np.array(a, dtype=float), 1)
            for a in ([100, 10, 10, 100], [100, 5, 5, 100], [100] * 4)
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "deadtime_correction": True,
            "deadtime_mode": "file",
            "dead_time_us": [1, -1, None],
            "good_frames": 100,
        },
    )
    value, error = integrate_run(run, t_min=1, t_max=2, method=method)
    f, b = 10 / 0.9, 5 / 1.05
    assert value == pytest.approx((f - b) / (f + b))
    assert error == pytest.approx(
        np.sqrt(((2 * b / (f + b) ** 2) ** 2 * 10 + (2 * f / (f + b) ** 2) ** 2 * 5) / 2)
    )
    with pytest.raises(DeadtimeCorrectionError):
        integrate_run(run, method=method)


def test_scoped_outcome_round_trips_original_ordinals_and_missing_unused_entries():
    import json
    import pickle

    from asymmetry.core.transform.corrections import CorrectionOutcomes

    result = reduce_counts([[10, 8], [5, 4], [100, 100]], [0.1, -0.1, None])
    assert result.corrections.deadtime.effective["tau_us"] == (0.1, -0.1, None)
    assert (
        CorrectionOutcomes.from_dict(json.loads(json.dumps(result.corrections.to_dict())))
        == result.corrections
    )
    assert pickle.loads(pickle.dumps(result.corrections)) == result.corrections


def test_active_background_pole_remains_required():
    with pytest.raises(DeadtimeCorrectionError) as error:
        reduce_counts([[100, 2, 1, 10, 8], [1, 1, 5, 5, 4]], [1, -0.1], first=3, background=True)
    assert error.value.bin_index == 0


def test_beam_trim_excludes_only_samples_not_used_by_range_estimator():
    histograms = [Histogram(np.full(70, level), 0.001) for level in (2.0, 1.0)]
    histograms[0].counts[45] = 100
    grouping = {
        "first_good_bin": 60,
        "last_good_bin": 69,
        "dead_time_us": [1, 1],
        "good_frames": 100000,
        "background_mode": "range",
        "background_range": [0, 49],
    }
    options = dict(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="file",
        use_background=True,
    )
    result = corrected_grouped_counts(**options, facility="PSI")
    np.testing.assert_allclose(result.forward.counts[result.window.slice], 0)
    assert result.corrections.background.effective["ranges"] == ((0, 39), (0, 39))
    assert result.corrections.deadtime.effective["scope"]["raw_bin_ranges"]["0"] == (
        (0, 39),
        (60, 69),
    )
    with pytest.raises(DeadtimeCorrectionError) as error:
        corrected_grouped_counts(**options, facility="")
    assert error.value.bin_index == 45


@pytest.mark.parametrize("unused", [float("nan"), float("inf"), "unavailable"])
def test_unused_unavailable_file_member_has_serialisable_effective_identity(unused):
    result = reduce_counts([[10, 8], [5, 4], [100, 100]], [0.1, -0.1, unused])
    assert result.corrections.deadtime.effective["tau_us"] == (0.1, -0.1, None)
    assert result.corrections.deadtime.status == "applied"


def test_incomplete_binning_tail_is_not_an_output_dependency():
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_grouped_asymmetry

    result = reduce_grouped_asymmetry(
        histograms=[Histogram(np.array(a, dtype=float), 1) for a in ([10, 10, 100], [5, 5, 5])],
        grouping={"dead_time_us": [1, -1], "good_frames": 100},
        forward_idx=[0],
        backward_idx=[1],
        binning=BinningSpec(factor=2),
        alpha=1,
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    f, b = 10 / 0.9, 5 / 1.05
    np.testing.assert_allclose(result.asymmetry, [100 * (f - b) / (f + b)])
    np.testing.assert_array_equal(result.time, [0.5])
    assert result.corrections.deadtime.effective["scope"]["raw_bin_ranges"] == {
        "0": ((0, 1),),
        "1": ((0, 1),),
    }


def test_layout_does_not_sum_raw_counts_before_signed_detector_inverse():
    histograms = [
        Histogram(np.array([value]), 1, count_variance=np.ones(1)) for value in (1e308, 1e308, 10.0)
    ]
    result = corrected_grouped_counts(
        histograms=histograms,
        grouping={"dead_time_us": [-1, -1, 0], "good_frames": 1},
        forward_idx=[0, 1],
        backward_idx=[2],
        use_deadtime=True,
        deadtime_mode="manual",
        use_background=False,
    )
    np.testing.assert_allclose(result.forward.counts, [2.0])
    np.testing.assert_array_equal(result.forward.variance, [2.0])
    assert histograms[0].counts[0] == 1e308


def test_workflow_time_window_limits_inverse_but_bandwidth_estimator_keeps_its_inputs():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    run = Run(
        histograms=[
            Histogram(np.array(a, dtype=float), 1, good_bin_end=3)
            for a in ([100, 10, 10, 100], [5, 5, 5, 5])
        ],
        grouping={"groups": {1: [1], 2: [2]}, "dead_time_us": [1, -1], "good_frames": 100},
    )
    settings = ReductionSettings(deadtime="from_file", t_min=1.0, t_max=2.0)
    result = reduce_run(run, settings, allow_aliasing=True)
    f, b = 10 / 0.9, 5 / 1.05
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_array_equal(result.time, [1.0, 2.0])
    with pytest.raises(DeadtimeCorrectionError):
        reduce_run(run, replace(settings, rebin=2))


@pytest.mark.parametrize("required_missing", [False, True])
def test_combined_period_reference_cohort_skips_only_required_missing_members(required_missing):
    from contextlib import nullcontext

    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    sample = [Histogram(np.array([100.0, 10.0, 8.0]), 1) for _ in range(3)]
    reference_histograms = [Histogram(np.array([100.0, 2.0, 1.0]), 1) for _ in range(3)]
    references = {
        frames: BackgroundReference(
            reference_histograms,
            1.0,
            good_frames_reference=frames,
            file_dead_time_us=(0.1, None if required_missing and frames == 200 else -0.1, None),
        )
        for frames in (100, 200)
    }
    options = dict(
        histograms=sample,
        grouping={
            "period_histograms": [sample, sample],
            "period_dead_time_us": [[0.1, -0.1, None], [0.2, -0.2, None]],
            "period_good_frames": [100, 200],
            "background_mode": "reference_run",
            "first_good_bin": 1,
            "last_good_bin": 2,
        },
        period_mode="green_plus_red",
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1.0,
        deadtime_mode="file",
        use_background=True,
        reference_resolver=lambda grouping: references[grouping["good_frames"]],
    )
    with (
        pytest.warns(UserWarning, match="Continuing uncorrected")
        if required_missing
        else nullcontext()
    ):
        result = reduce_period_asymmetry(**options, use_deadtime=True)
    assert all(
        c.deadtime.status == ("skipped" if required_missing else "applied")
        for c in result.corrections.values()
    )
    assert all(
        c.background.effective["reference"]["deadtime"]["status"]
        == ("skipped" if required_missing else "applied")
        for c in result.corrections.values()
    )
    if required_missing:
        uncorrected = reduce_period_asymmetry(**options, use_deadtime=False)
        np.testing.assert_array_equal(result.asymmetry, uncorrected.asymmetry)
        np.testing.assert_array_equal(result.error, uncorrected.error)


def test_each_period_resolves_its_own_original_raw_bins_and_t0():
    from asymmetry.core.transform.rebin import BinningSpec
    from asymmetry.core.transform.reduce import reduce_period_asymmetry

    periods = [
        [
            Histogram(
                np.r_[np.full(t0 + 1, 100.0), [10.0, 8.0, 6.0]],
                1,
                t0_bin=t0,
                good_bin_start=t0 + 1,
                good_bin_end=t0 + 3,
            )
            for _ in range(2)
        ]
        for t0 in (0, 2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "good_window_policy": {"mode": "from_file"},
        "detector_t0_bins": [0, 0],
        "detector_first_good_bins": [1, 1],
        "detector_last_good_bins": [3, 3],
        "period_histograms": periods,
        "period_dead_time_us": [[1, -1], [1, -1]],
        "period_good_frames": [100, 100],
    }
    result = reduce_period_asymmetry(
        periods[0],
        grouping,
        "green_minus_red",
        binning=BinningSpec(),
        forward_idx=[0],
        backward_idx=[1],
        alpha=1,
        use_deadtime=True,
        deadtime_mode="file",
        use_background=False,
    )
    np.testing.assert_allclose(result.asymmetry, 0, atol=0)
    np.testing.assert_array_equal(result.time, [1.0, 2.0, 3.0])
    for label, start in (("red", 1), ("green", 3)):
        assert result.corrections[label].deadtime.effective["scope"]["raw_bin_ranges"]["0"] == (
            (start, start + 2),
        )
