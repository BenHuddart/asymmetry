"""Resolved histogram preparation preserves the selected count observation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest
from tests.core.test_binning_entry_points import _run
from tests.core.test_selected_pair_balances import pair_run


def test_preparation_count_ratio_and_independent_error():
    run = _run([90, 1], [10, 9])
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    np.testing.assert_allclose(result.time, [0.05], atol=1e-15)
    np.testing.assert_allclose(result.asymmetry, [100 * 72 / 110], rtol=1e-12)
    np.testing.assert_allclose(
        result.error, [200 * np.sqrt(19**2 * 91 + 91**2 * 19) / 110**2], rtol=1e-12
    )


@pytest.mark.parametrize("merge", [1, 2, 3, 8])
def test_preparation_keeps_base_packing_and_complete_groups(merge):
    f, b = np.arange(1.0, 12), np.arange(21.0, 32)
    run = _run(f, b, bunching_factor=2)
    result = prepare_histogram_asymmetry(run, merge=merge, facility=None, reference=None)
    width = 2 * merge
    n = len(f) // width * width
    forward = f[:n].reshape(-1, width).sum(axis=1)
    backward = b[:n].reshape(-1, width).sum(axis=1)
    total = forward + backward
    np.testing.assert_allclose(result.asymmetry, 100 * (forward - backward) / total)
    np.testing.assert_allclose(
        result.error, 200 * np.sqrt(backward**2 * forward + forward**2 * backward) / total**2
    )
    np.testing.assert_allclose(result.time, (np.arange(n).reshape(-1, width).mean(axis=1)) * 0.1)
    np.testing.assert_array_equal(run.histograms[0].counts, f)
    assert run.grouping["bunching_factor"] == 2


@pytest.mark.parametrize("mode", ["fixed", "variable", "constant_error"])
def test_preparation_all_base_modes_keep_actual_midpoints_and_exact_t0(mode):
    f, b = np.arange(1.0, 12), np.arange(21.0, 32)
    run = _run(f, b, binning_mode=mode, bunching_factor=3, bin0_us=0.25, bin10_us=0.25)
    run.histograms = [replace(h, t0_time_us=0.07) for h in run.histograms]
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    # Native widths: fixed [3,3,3], variable [3,3,3,2], lifetime [3,3,3,2].
    edges = [0, 6] if mode == "fixed" else [0, 6, 11]
    forward = np.array([f[a:z].sum() for a, z in zip(edges[:-1], edges[1:])])
    backward = np.array([b[a:z].sum() for a, z in zip(edges[:-1], edges[1:])])
    total = forward + backward
    np.testing.assert_allclose(result.asymmetry, 100 * (forward - backward) / total)
    np.testing.assert_allclose(
        result.error, 200 * np.sqrt(backward**2 * forward + forward**2 * backward) / total**2
    )
    np.testing.assert_allclose(
        result.time, [(a + z) * 0.05 - 0.07 for a, z in zip(edges[:-1], edges[1:])], atol=1e-15
    )


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("pair,alpha", [((1, 2), 0.75), ((3, 4), 1.1)])
def test_preparation_uses_selected_calibrated_pair(canonical, pair, alpha):
    run = pair_run(canonical)
    run.grouping.update(forward_group=pair[0], backward_group=pair[1])
    before = deepcopy(run.grouping)
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    f, b = (100, 80) if pair == (1, 2) else (120, 60)
    np.testing.assert_allclose(result.asymmetry, 100 * (f - alpha * b) / (f + alpha * b))
    np.testing.assert_allclose(
        result.error,
        100
        * alpha
        * np.sqrt((2 * b) ** 2 * 2 * f + (2 * f) ** 2 * 2 * b)
        * 2
        / (2 * f + alpha * 2 * b) ** 2,
    )
    assert run.grouping == before


@pytest.mark.parametrize("mode", ["red", "green", "green_minus_red", "green_plus_red"])
def test_preparation_reduces_each_period_before_combining(mode):
    red, green = _run([90, 1], [10, 9]), _run([2, 80], [8, 10])
    run = _run(
        [1, 1],
        [1, 1],
        period_mode=mode,
        period_histograms=[red.histograms, green.histograms],
        period_reduced=[{}, {}],
    )
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    ra, ga = 100 * 72 / 110, 100 * 64 / 100
    rv, gv = 40000 * 91 * 19 / 110**3, 40000 * 82 * 18 / 100**3
    expected = {"red": ra, "green": ga, "green_minus_red": ga - ra, "green_plus_red": ga + ra}
    variance = {"red": rv, "green": gv, "green_minus_red": gv + rv, "green_plus_red": gv + rv}
    np.testing.assert_allclose(result.asymmetry, expected[mode])
    np.testing.assert_allclose(result.error, np.sqrt(variance[mode]))


def test_preparation_shared_background_variance_and_selected_good_support():
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
    vf, vb = 400 + 4**2 * 4, 320 + 4**2 * 4
    np.testing.assert_allclose(result.asymmetry, [100 * (f - b) / (f + b)])
    np.testing.assert_allclose(
        result.error, [200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2]
    )
    assert result.ratio_validity.report().used == 1
    np.testing.assert_allclose(result.time, [0.35])


@pytest.mark.parametrize("scale_mode", ["manual", "automatic"])
def test_preparation_reference_own_clock_scaling_and_both_supplied_variances(scale_mode):
    run = _run(
        [200] * 8,
        [150] * 8,
        first_good_bin=2,
        last_good_bin=5,
        background_correction=True,
        background_mode="reference_run",
    )
    run.histograms = [
        replace(h, count_variance=np.full(8, v)) for h, v in zip(run.histograms, [17, 23])
    ]
    ref_run = _run(np.arange(10.0, 20), np.arange(20.0, 30))
    ref_hists = [
        replace(h, t0_bin=1, count_variance=np.full(10, v))
        for h, v in zip(ref_run.histograms, [5, 7])
    ]
    sample_exp = ReferenceExposure(200, "file")
    ref_exp = ReferenceExposure(100, "file")
    run.grouping["reference_exposure"] = sample_exp.to_dict()
    request = (
        ReferenceScaleRequest("manual", 2) if scale_mode == "manual" else ReferenceScaleRequest()
    )
    reference = BackgroundReference(ref_hists, request.resolve(sample_exp, ref_exp), run_number=2)
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=reference)
    f, b = np.array([346.0, 338.0]), np.array([206.0, 198.0])
    vf, vb = 2 * 17 + 2**2 * 2 * 5, 2 * 23 + 2**2 * 2 * 7
    np.testing.assert_allclose(result.asymmetry, 100 * (f - b) / (f + b))
    np.testing.assert_allclose(result.error, 200 * np.sqrt(b * b * vf + f * f * vb) / (f + b) ** 2)
    np.testing.assert_allclose(result.time, [0.25, 0.45])
    assert result.ratio_validity.report().used == 2
    assert result.ratio_validity.report().omitted == 0
    np.testing.assert_array_equal(reference.histograms[0].counts, np.arange(10, 20))
    np.testing.assert_array_equal(run.histograms[0].counts, np.full(8, 200))


def test_preparation_signed_supplied_variance_and_scalar_beta():
    run = _run([-2, 4], [3, 2], beta=1.2, alpha=0.8)
    run.histograms = [replace(h, count_variance=np.array([4.0, 9.0])) for h in run.histograms]
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    f, b, alpha, beta = 2, 5, 0.8, 1.2
    np.testing.assert_allclose(result.asymmetry, [100 * (f - alpha * b) / (beta * f + alpha * b)])
    np.testing.assert_allclose(
        result.error,
        [100 * alpha * (1 + beta) * np.sqrt(b * b * 13 + f * f * 13) / (beta * f + alpha * b) ** 2],
    )


def test_preparation_deadtime_before_fixed_background_keeps_measured_variance():
    run = _run(
        [100, 80],
        [60, 50],
        deadtime_correction=True,
        deadtime_mode="manual",
        dead_time_us=[0.01, 0.02],
        good_frames=1000,
        background_correction=True,
        background_mode="fixed",
        background_fixed_values=[7, 3],
    )
    before = deepcopy(run.grouping)
    result = prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
    f = sum(c / (1 - c * 0.01 / (1000 * 0.1)) for c in [100, 80]) - 2 * 7
    b = sum(c / (1 - c * 0.02 / (1000 * 0.1)) for c in [60, 50]) - 2 * 3
    np.testing.assert_allclose(result.asymmetry, [100 * (f - b) / (f + b)])
    np.testing.assert_allclose(
        result.error, [200 * np.sqrt(b * b * 180 + f * f * 110) / (f + b) ** 2]
    )
    assert result.ratio_validity.report().used == 1
    assert result.ratio_validity.report().omitted == 0
    assert run.grouping == before
    np.testing.assert_array_equal(run.histograms[0].counts, [100, 80])


def test_preparation_cancels_before_reading_source():
    class CancelledError(Exception):
        pass

    def cancel():
        raise CancelledError

    with pytest.raises(CancelledError):
        prepare_histogram_asymmetry(
            _run([], []), merge=2, facility=None, reference=None, cancel_check=cancel
        )


def test_preparation_missing_histograms_is_an_operation_error():
    run = _run([1], [1])
    run.histograms.clear()
    with pytest.raises(ValueError, match="preparation requires raw histograms"):
        prepare_histogram_asymmetry(run, merge=2, facility=None, reference=None)
