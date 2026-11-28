"""Reduction windows preserve declared limits and the selected observations."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.data.ratio_validity import ObservationReport, RatioReason, RatioValidity
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.data.time_window import TIME_WINDOW_ATOL_US, time_window_mask
from asymmetry.core.fitting.count_inputs import count_time_selection
from asymmetry.core.fitting.grouped_time_domain import build_count_group
from asymmetry.core.fitting.input_attribution import fit_input_attribution
from asymmetry.core.fitting.likelihood import dataset_observations, gaussian_selection
from asymmetry.core.fourier.fft import prepare_fft_arrays
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.maxent import MaxEntConfig, build_maxent_input
from asymmetry.core.maxent.engine import estimate_maxent_workload
from asymmetry.core.project.profiles import (
    GoodWindowPolicy,
    resolve_good_window,
    run_file_good_window,
)
from asymmetry.core.transform.background import apply_grouped_background_correction
from asymmetry.core.transform.fb_counts import BinWindow, native_time_window
from asymmetry.core.transform.integral import integrate_curve
from asymmetry.core.transform.rebin import BinningSpec, fb_binning_edges
from asymmetry.core.workflow.fourier import FourierSettings, fourier_spectrum
from tests.counts import aligned_counts


@pytest.mark.parametrize("path", ["dataset", "gaussian", "observations", "counts"])
def test_time_crop_keeps_centres_one_ulp_outside_either_bound(path):
    time = np.arange(5, dtype=float) * 0.125
    lower = np.nextafter(time[1], np.inf)
    upper = np.nextafter(time[3], -np.inf)
    dataset = MuonDataset(time, np.arange(5, dtype=float), np.ones(5))
    if path == "dataset":
        selected = dataset.time_range(lower, upper).time
    elif path == "gaussian":
        selected = time[gaussian_selection(dataset, lower, upper).mask]
    elif path == "observations":
        selected = dataset_observations(dataset, lower, upper).time
    else:
        centres, mask = count_time_selection(0, 4, 1, 0, 0.125, 0, lower, upper, None)
        selected = centres[mask]
    np.testing.assert_array_equal(selected, time[1:4])


@pytest.mark.parametrize("bound", [0.0, 128.0])
def test_open_time_crop_uses_a_fixed_absolute_margin(bound):
    time = bound + np.array([-2, -0.5, 0, 0.5, 2]) * TIME_WINDOW_ATOL_US
    np.testing.assert_array_equal(time_window_mask(time, bound, None), [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(time_window_mask(time, None, bound), [1, 1, 1, 1, 0])
    assert not time_window_mask(time, np.nextafter(bound, np.inf), bound).any()


def test_time_crop_preserves_bounds_and_aligns_observation_support():
    time = np.arange(1.0, 6.0)
    values, errors = time * 8, time / 4
    bounds = (np.nextafter(time[1], np.inf), np.nextafter(time[3], -np.inf))
    validity = RatioValidity(
        [
            RatioReason.ZERO_VARIANCE,
            RatioReason.USABLE,
            RatioReason.UNDEFINED,
            RatioReason.USABLE,
            RatioReason.ZERO_VARIANCE,
        ]
    )
    dataset = MuonDataset(
        time,
        values,
        errors,
        ratio_validity=validity,
        effective_recipe=EffectiveReductionRecipe.from_dict({"version": 1}),
    )

    crop = dataset.time_range(*bounds)
    np.testing.assert_array_equal(crop.time, time[1:4])
    np.testing.assert_array_equal(crop.asymmetry, values[1:4])
    np.testing.assert_array_equal(crop.error, errors[1:4])
    np.testing.assert_array_equal(crop.ratio_validity.reasons, validity.reasons[1:4])
    (operation,) = crop.effective_recipe.to_dict()["curve_operations"]
    assert (operation["t_min"], operation["t_max"]) == bounds
    assert (operation["first"], operation["stop"]) == (1, 4)

    selection = gaussian_selection(dataset, *bounds)
    np.testing.assert_array_equal(selection.mask, [0, 1, 0, 1, 0])
    assert selection.report == ObservationReport(used=2, undefined_ratio=1)
    observations = dataset_observations(dataset, *bounds)
    np.testing.assert_array_equal(observations.time, time[[1, 3]])
    np.testing.assert_array_equal(observations.values, values[[1, 3]])
    np.testing.assert_array_equal(observations.sigma, errors[[1, 3]])
    attribution = fit_input_attribution(observations, dataset, window=bounds)
    assert (attribution["min"], attribution["max"]) == (time[1], time[3])
    assert attribution["identity"] == fit_input_attribution(observations, crop)["identity"]


def test_frequency_window_keeps_exact_membership():
    frequency = np.arange(1.0, 4.0)
    dataset = MuonDataset(frequency, np.ones(3), np.ones(3), {"plot_domain": "frequency"})
    bounds = (np.nextafter(frequency[0], np.inf), np.nextafter(frequency[-1], -np.inf))
    np.testing.assert_array_equal(gaussian_selection(dataset, *bounds).mask, [0, 1, 0])
    np.testing.assert_array_equal(dataset.time_range(*bounds).time, [2.0])
    np.testing.assert_array_equal(dataset_observations(dataset, *bounds).time, [2.0])


def test_bunched_count_window_keeps_contributing_counts_and_errors_aligned():
    run = Run(
        run_number=1,
        histograms=[Histogram(np.arange(1.0, 9.0), bin_width=0.125)],
        grouping={
            "groups": {1: [1]},
            "first_good_bin": 0,
            "last_good_bin": 7,
            "bunching_factor": 2,
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
    centres = np.array([0.3125, 0.5625])
    bounds = np.nextafter(centres, [np.inf, -np.inf])
    count = build_count_group(
        dataset, 1, t_min=bounds[0], t_max=bounds[1], lifetime_corrected=False
    )
    np.testing.assert_array_equal(count.time, centres)
    np.testing.assert_array_equal(count.counts, [7.0, 11.0])
    np.testing.assert_array_equal(count.detector_counts, [[7.0, 11.0]])
    np.testing.assert_array_equal(count.error, np.sqrt([7.0, 11.0]))
    assert count.metadata["count_fit_raw_windows"] == ((2, 3), (4, 5))

    signal = build_group_signal_dataset(
        run, 1, time_range=tuple(bounds), center_signal=False, apply_lifetime_correction=False
    )
    np.testing.assert_array_equal(signal.time, centres)
    np.testing.assert_array_equal(signal.asymmetry, count.counts)
    np.testing.assert_array_equal(signal.error, count.error)


def test_fft_and_integral_use_the_same_time_centres():
    time = np.arange(5) * 0.125
    values, errors = time * 8, np.ones(5)
    bounds = np.nextafter(time[[1, 3]], [np.inf, -np.inf])
    fft_input = prepare_fft_arrays(
        time, values, errors, t_min=bounds[0], t_max=bounds[1], subtract_average_signal=False
    )
    assert fft_input.time_start_us == time[1]
    np.testing.assert_array_equal(fft_input.signal, values[1:4])
    np.testing.assert_array_equal(fft_input.sample_errors, errors[1:4])
    mean, error = integrate_curve(time, values, errors, t_min=bounds[0], t_max=bounds[1])
    assert mean == 2.0
    assert error == pytest.approx(1 / np.sqrt(3))
    assert native_time_window(time, BinWindow(1, 3), (np.nextafter(time[3], np.inf), None)) == (
        BinWindow(3, 3)
    )


def test_binning_time_window_keeps_complete_bins():
    counts = aligned_counts(np.ones(8), np.ones(8), bin_width_us=0.125)
    bounds = tuple(np.nextafter([0.3125, 0.5625], [np.inf, -np.inf]))
    spec = BinningSpec(factor=2, time_range=bounds)
    np.testing.assert_array_equal(spec.edges(8, 0.125, 0.0), [2, 4, 6])
    edges, _ = fb_binning_edges(counts, BinningSpec(factor=2), time_range=bounds)
    np.testing.assert_array_equal(edges, [2, 4, 6])
    assert spec.time_range == bounds


def test_maxent_workload_and_reconstruction_share_window_centres():
    time = np.arange(5) * 0.125
    run = Run(
        run_number=1,
        histograms=[Histogram(np.arange(1.0, 6.0), bin_width=0.125) for _ in range(2)],
        grouping={"groups": {1: [1], 2: [2]}, "first_good_bin": 0, "last_good_bin": 4},
    )
    lower, upper = np.nextafter(time[[1, 3]], [np.inf, -np.inf])
    config = MaxEntConfig(
        t_min_us=lower, t_max_us=upper, selected_group_ids=[1], auto_phase_seed=False
    )
    estimate = estimate_maxent_workload(run, config)
    (group,) = build_maxent_input(run, config).groups
    np.testing.assert_array_equal(group.time_us[group.mask], time[1:4])
    assert estimate.time_points_per_group == (3,)


def test_fft_resolution_uses_the_cropped_time_span():
    time = np.arange(67) * 0.125
    values = np.cos(2 * np.pi * 1.1 * time)
    bounds = np.nextafter(time[[1, -2]], [np.inf, -np.inf])
    spectrum = fourier_spectrum(
        MuonDataset(time, values, np.ones(time.size)),
        FourierSettings(t_min=bounds[0], t_max=bounds[1]),
    )
    assert spectrum.resolution_mhz == 0.125


@pytest.mark.parametrize("facility", ["", "PSI"])
@pytest.mark.parametrize("window", [[4, 1], [-1, 2], [0, 9], [0, 1.5], ["bad", 1], [1]])
def test_declared_background_window_is_rejected(window, facility):
    with pytest.raises(ValueError, match="window|range|index"):
        apply_grouped_background_correction(
            np.ones(8),
            np.ones(8),
            grouping={"background_range": window},
            t0_bin=6,
            bin_width_us=0.01,
            facility=facility,
        )


def test_scalar_file_limits_are_not_reordered_before_manual_repair():
    run = Run(run_number=1, histograms=[Histogram(np.ones(20), bin_width=0.01)], metadata={})
    assert run_file_good_window(
        run, {"first_good_bin": 7, "last_good_bin": 3}, common_t0_bin=0
    ) == (7, 3)
    with pytest.raises(ValueError, match="window"):
        resolve_good_window(
            GoodWindowPolicy(), t0_bin=0, file_first_good=7, file_last_good=3, n_bins=20
        )


@pytest.mark.parametrize("window", [("bad", 3), (0, 1.5), (False, 3)])
def test_scalar_file_limits_reject_noninteger_indices(window):
    with pytest.raises(ValueError, match="window"):
        resolve_good_window(
            GoodWindowPolicy(),
            t0_bin=0,
            file_first_good=window[0],
            file_last_good=window[1],
            n_bins=20,
        )


def test_grouped_fourier_rejects_a_reversed_background_window():
    from asymmetry.core.fourier.group_background import _apply_group_background_correction

    with pytest.raises(ValueError, match="background range.*reversed"):
        _apply_group_background_correction(
            np.ones(8),
            grouping={
                "background_correction": True,
                "background_mode": "range",
                "background_range": [4, 1],
            },
            group_id=1,
            t0_bin=6,
            bin_width=0.01,
            metadata={"facility": "PSI"},
            source_file="synthetic.bin",
        )
