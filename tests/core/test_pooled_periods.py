"""Pooled observations through production preparation and scan estimators."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.transform.integral import reduce_period_integral
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry


def two_periods(red=(60, 40), green=(90, 10), *, t0=(0, 0), length=8):
    periods = [
        [Histogram(np.full(length, value, dtype=float), 0.1, t0_bin=anchor) for value in pair]
        for pair, anchor in zip((red, green), t0, strict=True)
    ]
    return Run(
        1,
        histograms=periods[0],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "period_histograms": periods,
            "period_mode": "green_plus_red",
            "good_window_policy": {"mode": "from_file"},
            "t0_source": "file",
        },
    )


def prepare(run, merge=1):
    return prepare_histogram_asymmetry(run, merge=merge, facility=None, reference=None)


@pytest.mark.parametrize("green,expected", [((90, 10), 50), ((270, 30), 65)])
def test_pooled_ratio_and_unchanged_independent_difference(green, expected):
    run = two_periods(green=green)
    np.testing.assert_allclose(prepare(run).asymmetry, expected)
    run.grouping["period_mode"] = "green_minus_red"
    np.testing.assert_allclose(prepare(run).asymmetry, 60)


@pytest.mark.parametrize("method", ["integral", "differential"])
def test_integral_pools_before_selected_estimator(method):
    result = reduce_period_integral(two_periods(), method=method)
    assert result.value == pytest.approx(0.5)


def test_background_once_nonunity_calibration_and_retained_variance():
    run = two_periods()
    run.grouping.update(
        alpha=0.7,
        beta=1.2,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[4, 2],
    )
    for period, variance in zip(run.grouping["period_histograms"], (25, 49), strict=True):
        period[:] = [replace(h, count_variance=np.full(8, variance)) for h in period]
    original = [h.counts.copy() for p in run.grouping["period_histograms"] for h in p]
    result = prepare(run, 2)
    f, b, vf, vb = 292, 96, 148, 148
    denominator = 1.2 * f + 0.7 * b
    expected = 100 * (f - 0.7 * b) / denominator
    error = 100 * 0.7 * 2.2 * np.sqrt(b * b * vf + f * f * vb) / denominator**2
    np.testing.assert_allclose(result.asymmetry, expected)
    np.testing.assert_allclose(result.error, error)
    for old, hist in zip(
        original, [h for p in run.grouping["period_histograms"] for h in p], strict=True
    ):
        np.testing.assert_array_equal(old, hist.counts)


@pytest.mark.parametrize("mode", ["green_plus_red", "green_minus_red"])
def test_cross_set_common_support_precedes_binning(mode):
    run = two_periods(t0=(1, 3))
    run.grouping["period_mode"] = mode
    result = prepare(run, 2)
    assert len(result.time) == 3
    np.testing.assert_allclose(result.time, [-0.05, 0.15, 0.35], atol=1e-15)
    np.testing.assert_allclose(result.asymmetry, 50 if mode == "green_plus_red" else 60)


def test_individually_undefined_period_is_usable_after_pooling():
    run = two_periods(red=(0, 0))
    result = prepare(run)
    assert result.ratio_validity.report().used == 8
    np.testing.assert_allclose(result.asymmetry, 80)


@pytest.mark.parametrize("manual", [False, True])
def test_pooled_reference_exposure_scale_and_variance_once(manual):
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest

    run = two_periods()
    run.grouping.update(
        background_correction=True,
        background_mode="reference_run",
        period_reference_exposure=[ReferenceExposure(v, "test").to_dict() for v in (10, 20)],
    )
    request = ReferenceScaleRequest("manual", 0.5) if manual else ReferenceScaleRequest()
    reference = BackgroundReference(
        [Histogram(np.full(8, v), 0.1, count_variance=np.full(8, 9)) for v in (4, 2)],
        request.resolve(reference=ReferenceExposure(10, "test")),
        run_number=99,
    )
    result = prepare_histogram_asymmetry(run, merge=1, facility=None, reference=reference)
    scale = 0.5 if manual else 3
    f, b = 150 - scale * 4, 50 - scale * 2
    vf, vb = 150 + scale**2 * 9, 50 + scale**2 * 9
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.error, 200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2)
    assert result.corrections["green_plus_red"].background.effective["reference"]["scale"] == scale


def test_mapped_pooled_corrects_each_selected_acquisition_and_cancels():
    from asymmetry.core.io.periods import combine_mapped_periods
    from tests.core.test_constituent_preparation import acquired

    sources = [
        acquired(1, [8000] * 8, [2000] * 8, frames=10000, tau=0.01, t0=1),
        acquired(2, [1000] * 8, [9000] * 8, frames=20000, tau=0.03, t0=2),
        acquired(3, [100] * 8, [300] * 8, frames=4000, tau=0.02, t0=3),
        acquired(4, [1e12] * 8, [1e12] * 8),
    ]
    mapped = combine_mapped_periods(sources, {1: "red", 2: "green", 3: "green", 4: "ignore"})
    mapped.run.grouping.update(period_mode="green_plus_red", alpha=0.7, beta=1.2)
    f, b = sum(
        np.array(v) / (1 - np.array(v) * tau / (0.1 * frames))
        for v, tau, frames in [
            ((8000, 2000), 0.01, 10000),
            ((1000, 9000), 0.03, 20000),
            ((100, 300), 0.02, 4000),
        ]
    )
    result = prepare(mapped.run, 2)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - 0.7 * b) / (1.2 * f + 0.7 * b))
    assert len(result.time) == 3
    assert {p for _, p, _ in result.alignments["green_plus_red"].acquired_detectors} == {1, 2, 3}
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 4:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        prepare_histogram_asymmetry(
            mapped.run, merge=2, facility=None, reference=None, cancel_check=cancel
        )


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_pooled_rotation_removes_equal_baseline_once_and_rejects_unequal(value):
    import json

    from asymmetry.core.transform.rotating_frame import (
        FrameSnapshot,
        Provenance,
        RotatingFrame,
        period_weights,
    )

    frame = RotatingFrame.typed_frequency(1, 2)
    frame = frame.with_baseline(0, Provenance.TYPED, value, -value)
    frame = frame.with_baseline(1, Provenance.TYPED, value, -value)
    weights = period_weights("green_plus_red", 2)
    assert frame.combined_baseline(weights) == (value, -value)
    snapshot = FrameSnapshot(frame, weights, "test")
    assert FrameSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict()))).same_rotation(
        snapshot
    )
    unequal = frame.with_baseline(1, Provenance.TYPED, np.nextafter(value, np.inf), -value)
    with pytest.raises(ValueError, match="G\\+R rotation unavailable"):
        unequal.combined_baseline(weights)
    assert unequal.rotation_unavailable(period_weights("green_minus_red", 2)) is None
    # Historical explicitly linear frame snapshots retain their original meaning.
    assert frame.combined_baseline((1, 1)) == (2 * value, -2 * value)


def test_rf_fourier_and_fit_use_the_actual_pooled_observation():
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet
    from asymmetry.core.fourier.fb_input import build_fb_signal_dataset
    from asymmetry.core.io.periods import build_rf_difference_scan
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    run = two_periods()
    run.metadata["field"] = 10
    # Deliberately stale curves cannot define a fresh count-backed scan.
    run.grouping["period_reduced"] = [(np.arange(8.0), np.full(8, 900.0), np.ones(8))] * 2
    dataset = reduce_run(run, ReductionSettings(period="green_plus_red"))
    scan = build_rf_difference_scan([dataset], mode="green_plus_red")
    np.testing.assert_allclose(scan.value, 0.5)
    np.testing.assert_allclose(build_fb_signal_dataset(dataset.run).asymmetry, 50)

    def constant(time, c):
        return np.full_like(time, c)

    fit = FitEngine().fit(dataset, constant, ParameterSet([Parameter("c", 40)]))
    assert fit.success
    assert fit.parameters["c"].value == pytest.approx(50)
    assert fit.dof == 7
    assert fit.observation_report.used == 8
    np.testing.assert_allclose(fit.residuals, 0, atol=1e-7)


@pytest.mark.parametrize("method", ["integral", "differential"])
def test_public_integrate_run_respects_the_requested_period_statistic(method):
    from asymmetry.core.transform.integral import integrate_run

    run = two_periods()
    assert integrate_run(run, method=method)[0] == pytest.approx(0.5)
    run.grouping["period_mode"] = "green_minus_red"
    assert integrate_run(run, method=method)[0] == pytest.approx(0.6)


def test_signed_pooling_requires_supplied_variance_only_in_consumed_bins():
    run = two_periods(red=(-5, 10), green=(20, 10))
    with pytest.raises(ValueError, match="signed acquired counts requires supplied variance"):
        prepare(run)
    for period in run.grouping["period_histograms"]:
        period[:] = [replace(h, count_variance=np.full(8, 25.0)) for h in period]
    result = prepare(run)
    np.testing.assert_allclose(result.asymmetry, -100 / 7)
    np.testing.assert_allclose(result.error, 200 * np.sqrt(20**2 * 50 + 15**2 * 50) / 35**2)
    run = two_periods()
    run.grouping["good_window_policy"] = {
        "mode": "manual",
        "first_good_offset_bins": 1,
        "last_good_offset_bins": 7,
    }
    # Keep explicit legacy bounds so the unused signed raw bin remains inspectable.
    run.grouping.pop("good_window_policy")
    run.grouping.update(first_good_bin=1, last_good_bin=7)
    run.grouping["period_histograms"][0][0].counts[0] = -5
    np.testing.assert_allclose(prepare(run).asymmetry, 50)


def test_pooled_missing_reference_file_coefficient_skips_whole_request():
    from asymmetry.core.transform.corrections import BackgroundReference
    from asymmetry.core.transform.reference_scale import ReferenceScaleRequest

    run = two_periods()
    run.grouping.update(
        deadtime_correction=True,
        deadtime_mode="file",
        period_good_frames=[1000, 2000],
        period_dead_time_us=[[0.01, 0.02], [0.03, 0.04]],
        background_correction=True,
        background_mode="reference_run",
    )
    reference = BackgroundReference(
        [Histogram(np.full(8, v), 0.1) for v in (4, 2)],
        ReferenceScaleRequest("manual", 0.5).resolve(),
        run_number=99,
        deadtime_normaliser=5000,
        file_dead_time_us=(0.01, None),
    )
    with pytest.warns(UserWarning, match="File deadtime correction not applied"):
        result = prepare_histogram_asymmetry(run, merge=1, facility=None, reference=reference)
    outcome = result.corrections["green_plus_red"]
    assert outcome.deadtime.status == "skipped"
    assert outcome.background.effective["reference"]["deadtime"]["status"] == "skipped"
    np.testing.assert_allclose(result.asymmetry, 100 * (148 - 49) / (148 + 49))


def test_explicit_automatic_alpha_uses_one_pooled_target_and_fixed_stays_fixed():
    from asymmetry.core.project.profiles import (
        AlphaPolicy,
        GroupingProfile,
        ProfileFingerprint,
        resolve_effective_grouping,
    )

    run = two_periods()
    profile = GroupingProfile(
        "test",
        ProfileFingerprint("", 2),
        groups={1: [1], 2: [2]},
        forward_group=1,
        backward_group=2,
        period_mode="green_plus_red",
        alpha_policy=AlphaPolicy(mode="per_run_estimate"),
    )
    automatic = resolve_effective_grouping(profile, run)
    assert automatic["alpha"] == pytest.approx(3)
    selected = resolve_effective_grouping(replace(profile, period_mode="red"), run)
    assert selected["alpha"] == pytest.approx(1.5)
    for mode in ("fixed", "calibrated"):
        fixed = resolve_effective_grouping(
            replace(profile, alpha_policy=AlphaPolicy(mode=mode, value=1.17)), run
        )
        assert fixed["alpha"] == 1.17


def test_implicit_mapped_file_window_uses_acquired_support_not_held_quick_bounds():
    from asymmetry.core.io.periods import combine_mapped_periods
    from tests.core.test_constituent_preparation import acquired

    k = np.arange(20.0)
    sources = [
        acquired(i, 100 + 3 * k + k * k, 80 + 2 * k, t0=anchor, first=anchor, last=anchor + 10)
        for i, anchor in enumerate((1, 3, 2), 1)
    ]
    for source in sources:
        source.run.grouping.pop("good_window_policy")
    mapped = combine_mapped_periods(sources, {1: "red", 2: "green", 3: "red"})
    mapped.run.grouping.update(period_mode="green_plus_red", deadtime_correction=False)
    result = prepare(mapped.run, 2)
    f, b = (
        sum(
            np.array([h.counts[anchor : anchor + 10] for h in source.run.histograms])
            for source, anchor in zip(sources, (1, 3, 2), strict=True)
        )
        .reshape(2, 5, 2)
        .sum(axis=2)
    )
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.time, np.arange(5) * 0.2 + 0.05, atol=1e-15)


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("mixed_variance", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_unused_signed_context_does_not_supply_variance_or_fail_pooling(
    correction, mixed_variance, background
):
    from copy import deepcopy

    run = two_periods()
    run.grouping.pop("good_window_policy")
    run.grouping.update(
        first_good_bin=4,
        last_good_bin=7,
        deadtime_correction=correction,
        deadtime_mode="file",
        period_good_frames=[1000, 2000],
        period_dead_time_us=[[0.01, 0.02], [0.03, 0.04]],
        background_correction=background,
        background_mode="range",
        background_ranges=[0, 1],
    )
    if mixed_variance:
        run.grouping["period_histograms"][1][:] = [
            replace(h, count_variance=np.full(8, 9.0)) for h in run.grouping["period_histograms"][1]
        ]
    control = prepare(deepcopy(run))
    original = run.grouping["period_histograms"][0]
    original[0].counts[2] = -1000
    original[1].counts[3] = -1000
    result = prepare(run)
    np.testing.assert_array_equal(result.asymmetry, control.asymmetry)
    np.testing.assert_array_equal(result.error, control.error)
    assert original[0].counts[2] == -1000
    assert original[1].counts[3] == -1000


@pytest.mark.parametrize("correct", [False, True])
def test_scoped_variance_excludes_raw_context_from_disjoint_domains(correct):
    from asymmetry.core.transform.deadtime import DeadtimeDomain
    from asymmetry.core.transform.fb_counts import BinWindow
    from asymmetry.core.transform.grouping import resolve_alignment
    from asymmetry.core.transform.scoped_counts import ScopedCounts

    source = [
        Histogram(np.array([-1000.0, 3, 4, -2000, 6, 7, -3000, 9]), 0.1),
        Histogram(np.full(8, 2.0), 0.1, count_variance=np.full(8, 5.0)),
    ]
    domain = DeadtimeDomain(tuple((i, (BinWindow(1, 2), BinWindow(4, 5))) for i in (0, 1)))
    scoped = ScopedCounts.prepare(
        source,
        {"good_frames": 1000, "dead_time_us": [0.01, 0.01]},
        domain,
        use_deadtime=correct,
        mode="manual",
    )
    aligned = scoped.aligned([0, 1], resolve_alignment(source, [0], [1]))
    np.testing.assert_array_equal(aligned.variance, [0, 8, 9, 0, 11, 12, 0, 0])
    assert aligned.counts[0] == -998
    assert source[0].counts[3] == -2000
    assert source[0].count_variance is None
