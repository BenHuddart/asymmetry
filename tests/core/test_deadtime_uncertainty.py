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
from asymmetry.core.transform.fb_counts import GroupCounts, linear_combination_variance
from asymmetry.core.transform.grouping import align_forward_backward_counts
from asymmetry.core.transform.integral import integrate_asymmetry
from asymmetry.core.transform.rebin import BinningSpec, binned_fb_asymmetry
from asymmetry.core.transform.reduce import corrected_grouped_counts
from asymmetry.core.transform.reference_scale import ReferenceScaleRequest
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


@pytest.mark.parametrize("variance", [[1.0], [-1.0, 2.0], [np.inf, 2.0], [[1.0, 2.0]], [1j, 2j]])
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
        scaling=ReferenceScaleRequest("manual", 2.0).resolve(),
        deadtime_normaliser=50.0,
    )
    counts = _counts(run, background=True, reference_resolver=lambda _: reference)
    np.testing.assert_array_equal(counts.forward.variance, np.full(32, 140.0))
    np.testing.assert_array_equal(counts.backward.variance, np.full(32, 45.0))
    np.testing.assert_allclose(counts.forward.counts, 125.0 - 2 * 10.0 / 0.96)


def test_alignment_shifts_and_crops_observed_variance_like_counts():
    raw = [
        Histogram(np.array([10.0, 20.0, 30.0, 40.0]), 1.0, t0_bin=1),
        Histogram(np.array([2.0, 4.0, 6.0]), 1.0),
        Histogram(np.full(6, 8.0), 1.0),
    ]
    prepared, _ = prepare_histograms_with_deadtime(
        raw, {"dead_time_us": [0.2, 0.0, -0.1], "good_frames": 100.0}, True, mode="manual"
    )
    f, b, crop = align_forward_backward_counts(prepared, [0, 1], [2])
    np.testing.assert_array_equal(f.variance, [22.0, 34.0, 46.0])
    np.testing.assert_array_equal(b.variance, [8.0, 8.0, 8.0])
    assert crop.length == 3


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
    fitted = fit_tail_background(GroupCounts(raw, 3 * baseline.expected_counts), **kwargs)
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


@pytest.mark.parametrize("declared_t0", [False, True])
@pytest.mark.parametrize("range_form", ["shared", "paired", "named", "implicit"])
def test_spectral_background_preserves_acquisition_bins_with_later_extra_group(
    declared_t0, range_form
):
    run = _constant_run(coefficient=0.01, bunching=4)
    run.histograms = [
        Histogram(np.array([8.0, 8.0] + [100.0] * 30), 1.0, t0_bin=2),
        Histogram(np.array([12.0, 12.0] + [80.0] * 30), 1.0, t0_bin=2),
        Histogram(np.full(32, 40.0), 1.0, t0_bin=6),
    ]
    run.metadata["facility"] = "PSI"
    run.grouping.update(
        groups={1: [1], 2: [2], 3: [3]},
        dead_time_us=[0.01] * 3,
        first_good_bin=2,
        last_good_bin=17,
        background_correction=True,
        background_mode="range",
    )
    if declared_t0:
        run.grouping["t0_bin"] = 2
    if range_form == "shared":
        run.grouping["background_range"] = [0, 1]
    elif range_form == "paired":
        run.grouping["background_ranges"] = [[0, 1], [0, 1]]
    elif range_form == "named":
        run.grouping.update(background_forward_range=[0, 1], background_backward_range=[0, 1])
    # A two-bin source t0 gives the same implicit pre-t0 window [0, 1].
    reduced = _counts(run, background=True)
    if range_form != "implicit":
        run.grouping["background_values"] = reduced.corrections.background.effective["values"]
    for gid, group, expected_variance in (
        (1, reduced.forward, 464.0),
        (2, reduced.backward, 416.0),
    ):
        signal = build_group_signal_dataset(
            run, gid, center_signal=False, apply_lifetime_correction=False
        )
        expected = group.counts[2:18].reshape(4, 4).sum(axis=1)
        np.testing.assert_allclose(signal.asymmetry, expected, rtol=1e-12)
        np.testing.assert_allclose(signal.error**2, expected_variance, rtol=1e-12)
        np.testing.assert_allclose(signal.time, [1.5, 5.5, 9.5, 13.5])
    inputs = build_maxent_input(
        run, MaxEntConfig(n_spectrum_points=32, f_min_mhz=0.01, f_max_mhz=0.1, auto_window=False)
    )
    np.testing.assert_allclose(
        inputs.groups[0].sigma / (inputs.groups[0].signal + 1),
        np.sqrt(464.0) / reduced.forward.counts[2:6].sum(),
    )


def test_tail_fit_rejects_zero_propagated_uncertainty():
    time = np.arange(120) * 0.05
    counts = 200 * np.exp(-time / MUON_LIFETIME_US) + 20
    result = fit_tail_background(
        GroupCounts(counts, np.zeros(120)), bin_width_us=0.05, t0_bin=0, fit_start_bin=0
    )
    assert not result.ok
    assert "variance" in result.message
