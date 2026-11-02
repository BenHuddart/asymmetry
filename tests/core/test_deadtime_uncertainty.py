"""Independent measured-count variance checks through correction consumers."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_groups
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input
from asymmetry.core.transform.asymmetry import estimate_alpha_detailed
from asymmetry.core.transform.background import fit_tail_background
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.fb_counts import linear_combination_variance
from asymmetry.core.transform.grouping import align_forward_backward_counts
from asymmetry.core.transform.integral import integrate_asymmetry
from asymmetry.core.transform.rebin import BinningSpec, binned_fb_asymmetry
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _constant_run(*, coefficient=0.2, bunching=1):
    return Run(
        histograms=[Histogram(np.full(32, count), bin_width=1.0) for count in (100.0, 25.0)],
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": 31,
            "deadtime_correction": True,
            "deadtime_mode": "manual",
            "dead_time_us": [coefficient, coefficient],
            "good_frames": 100.0,
            "bunching_factor": bunching,
        },
    )


def _counts(run, *, background=False, **kwargs):
    return corrected_grouped_counts(
        histograms=run.histograms,
        grouping=run.grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_deadtime=True,
        deadtime_mode="manual",
        use_background=background,
        facility="",
        **kwargs,
    )


@pytest.mark.parametrize("variance", [[1.0], [-1.0, 2.0], [np.nan, 2.0], [[1.0, 2.0]], [1j, 2j]])
def test_histogram_rejects_invalid_count_variance(variance):
    with pytest.raises(ValueError, match="variance"):
        Histogram(np.array([1.0, 2.0]), bin_width=1.0, count_variance=variance)


@pytest.mark.parametrize("coefficient", [-0.2, 0.0, 0.2])
def test_preparation_retains_supplied_variance_and_timing(coefficient):
    variance = np.array([49.0, 9.0])
    histogram = Histogram(
        np.array([100.0, 25.0]), 1.0, t0_bin=1, t0_time_us=1.2, count_variance=variance
    )
    prepared, outcome = prepare_histograms_with_deadtime(
        [histogram], {"dead_time_us": [coefficient], "good_frames": 100.0}, True, mode="manual"
    )
    variance[:] = 999.0
    np.testing.assert_array_equal(prepared[0].count_variance, [49.0, 9.0])
    np.testing.assert_array_equal(prepared[0].time_axis, histogram.time_axis)
    np.testing.assert_array_equal(histogram.counts, [100.0, 25.0])
    assert "coefficient" in outcome.uncertainty


@pytest.mark.parametrize("mode", ["fixed", "variable", "constant_error"])
@pytest.mark.parametrize("coefficient", [-0.2, 0.0, 0.2])
def test_binning_uses_production_measured_count_variance(mode, coefficient):
    run = _constant_run(coefficient=coefficient)
    counts = _counts(run)
    spec = BinningSpec(mode=mode, factor=4, bin0_us=2.0, bin10_us=4.0)
    _, _, error = binned_fb_asymmetry(counts, binning=spec, alpha=1.0)
    edges = spec.edges(32, 1.0, 0.0)
    widths = np.diff(edges)
    f = 100.0 / (1.0 - coefficient)
    b = 25.0 / (1.0 - coefficient / 4.0)
    expected = 2 * np.sqrt(b * b * 100.0 + f * f * 25.0) / (f + b) ** 2 / np.sqrt(widths)
    np.testing.assert_allclose(error, expected, rtol=1e-12)


@pytest.mark.parametrize("window, expected_variance", [([0, 1], 464.0), ([1, 2], 432.0)])
def test_background_overlap_retains_measured_variance(window, expected_variance):
    run = _constant_run(coefficient=0.01)
    run.histograms = [
        Histogram(np.array(row, dtype=float), 1.0)
        for row in ([8, 8, 100, 100, 100, 100], [8, 8, 80, 80, 80, 80])
    ]
    run.grouping.update(
        first_good_bin=2, last_good_bin=5, background_mode="range", background_range=window
    )
    counts = _counts(run, background=True)
    assert linear_combination_variance(counts.forward, [2, 6])[0] == pytest.approx(
        expected_variance
    )
    f, b = counts.forward.counts[2:].sum(), counts.backward.counts[2:].sum()
    vb = 384.0 if window == [0, 1] else 352.0
    expected_error = 2 * np.sqrt(b * b * expected_variance + f * f * vb) / (f + b) ** 2
    _, error = integrate_asymmetry(
        counts.forward, counts.backward, time=np.arange(6.0), t_min=2.0, t_max=5.0
    )
    assert error == pytest.approx(expected_error)


def test_fixed_background_preserves_measured_variance():
    run = _constant_run()
    run.grouping.update(background_mode="fixed", background_fixed_values=[3.0, 4.0])
    counts = _counts(run, background=True)
    np.testing.assert_array_equal(counts.forward.variance, np.full(32, 100.0))
    np.testing.assert_array_equal(counts.backward.variance, np.full(32, 25.0))


def test_reference_background_uses_both_measured_variances_and_own_exposure():
    run = _constant_run()
    run.grouping["background_mode"] = "reference_run"
    reference = BackgroundReference(
        [Histogram(np.full(32, n), 1.0) for n in (10.0, 5.0)],
        scale=2.0,
        good_frames_sample=100.0,
        good_frames_reference=50.0,
    )
    counts = _counts(run, background=True, reference_resolver=lambda _: reference)
    np.testing.assert_array_equal(counts.forward.variance, np.full(32, 140.0))
    np.testing.assert_array_equal(counts.backward.variance, np.full(32, 45.0))
    np.testing.assert_allclose(counts.forward.counts, 125.0 - 2 * 10.0 / 0.96)


def test_alignment_shifts_pads_and_crops_variance_like_counts():
    raw = [
        Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 1.0, t0_bin=1),
        Histogram(np.array([2.0, 4.0, 6.0]), 1.0),
        Histogram(np.full(6, 8.0), 1.0),
    ]
    prepared, _ = prepare_histograms_with_deadtime(
        raw, {"dead_time_us": [0.2, 0.0, -0.1], "good_frames": 100.0}, True, mode="manual"
    )
    f, b, crop = align_forward_backward_counts(prepared, [0, 1], [2])
    np.testing.assert_array_equal(f.variance, [10.0, 22.0, 34.0, 46.0])
    np.testing.assert_array_equal(b.variance, [0.0, 8.0, 8.0, 8.0])
    assert crop.length == 4


def test_ratio_alpha_uses_selected_count_variance():
    counts = _counts(_constant_run())
    estimate = estimate_alpha_detailed(counts.forward, counts.backward, method="ratio")
    expected = (125.0 / (25.0 / 0.95)) * np.sqrt(
        3200.0 / 4000.0**2 + 800.0 / (32 * 25.0 / 0.95) ** 2
    )
    assert estimate.alpha_error == pytest.approx(expected)


def test_tail_fit_propagates_supplied_variance_through_its_influence():
    time = np.arange(120) * 0.05
    raw = 200.0 * np.exp(-time / MUON_LIFETIME_US) + 20.0
    kwargs = dict(bin_width_us=0.05, t0_bin=0, fit_start_bin=0)
    baseline = fit_tail_background(raw, **kwargs)
    fitted = fit_tail_background(raw, count_variance=3 * baseline.expected_counts, **kwargs)
    assert fitted.ok
    assert fitted.rate_per_us == baseline.rate_per_us
    assert fitted.rate_error_per_us == pytest.approx(np.sqrt(3) * baseline.rate_error_per_us)
    np.testing.assert_allclose(fitted.count_covariance, 3 * baseline.count_covariance)


def test_gaussian_and_spectral_inputs_retain_count_variance():
    run = _constant_run(bunching=4)
    dataset = MuonDataset(np.arange(32.0), np.zeros(32), np.ones(32), run=run)
    group = build_grouped_time_domain_groups(dataset, lifetime_corrected=False)[0]
    np.testing.assert_array_equal(group.counts, np.full(8, 500.0))
    np.testing.assert_array_equal(group.error, np.full(8, 20.0))
    spectrum = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_array_equal(spectrum.error, np.full(8, 20.0))
    config = MaxEntConfig(n_spectrum_points=32, f_min_mhz=0.01, f_max_mhz=0.1, auto_window=False)
    inputs = build_maxent_input(run, config)
    np.testing.assert_allclose(inputs.groups[0].sigma / (inputs.groups[0].signal + 1), 20.0 / 500.0)


def test_spectral_range_uses_estimation_window_even_with_cached_values():
    run = _constant_run(coefficient=0.01, bunching=4)
    run.histograms = [Histogram(np.array([8.0, 8.0, 100.0, 100.0, 100.0, 100.0]), 1.0)] * 2
    run.metadata["facility"] = "PSI"
    run.grouping.update(
        first_good_bin=2,
        last_good_bin=5,
        background_mode="range",
        background_range=[0, 1],
        background_values=[8 / 0.9992] * 2,
        background_correction=True,
    )
    signal = build_group_signal_dataset(
        run, 1, center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_allclose(signal.error, [np.sqrt(464.0)])
