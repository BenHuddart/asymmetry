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
