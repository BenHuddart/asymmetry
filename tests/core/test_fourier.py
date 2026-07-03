"""Tests for Fourier analysis modules."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.fft import (
    average_fourier_display_values,
    canonical_fourier_display_mode,
    estimate_fft_phase,
    exclude_frequency_ranges,
    fft_asymmetry,
    fft_complex_asymmetry,
    fourier_display_values,
    fourier_mode_uses_entropy_optimizer,
    fourier_mode_uses_phase_correction,
    optimize_phase_entropy,
)
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.fourier.maxent import maxent
from asymmetry.core.fourier.window import apply_fft_filter, apply_window


def _dataset(n: int = 128) -> MuonDataset:
    t = np.linspace(0.0, 10.0, n)
    a = 0.2 * np.cos(2 * np.pi * 0.4 * t)
    e = np.full(n, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 1})


def _phase_dataset(phase_degrees: float, n: int = 256) -> tuple[MuonDataset, float]:
    dt = 0.05
    cycles = 12
    time = np.arange(n, dtype=float) * dt
    frequency = cycles / (n * dt)
    phase_radians = np.deg2rad(phase_degrees)
    asymmetry = 0.2 * np.cos(2 * np.pi * frequency * time + phase_radians)
    error = np.full(n, 0.01)
    dataset = MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={"run_number": 2},
    )
    return dataset, frequency


def _angle_distance(actual: float, expected: float) -> float:
    return float(abs(np.angle(np.exp(1j * (actual - expected)))))


@pytest.mark.parametrize("name", ["gaussian", "hann", "cosine", "lorentzian"])
def test_apply_window_supported(name: str) -> None:
    signal = np.ones(32, dtype=float)
    out = apply_window(signal, name)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))


def test_apply_window_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown window"):
        apply_window(np.ones(16, dtype=float), "not-a-window")


def test_apply_fft_filter_lorentzian_matches_wimda_exponential_case() -> None:
    time = np.array([0.0, 0.5, 1.0, 2.0], dtype=float)
    signal = np.ones_like(time)

    out = apply_fft_filter(signal, time, mode="lorentzian", start_time_us=0.0, time_constant_us=1.5)

    assert out == pytest.approx(np.exp(-time / 1.5))


def test_apply_fft_filter_gaussian_start_time_delays_rolloff() -> None:
    time = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
    signal = np.ones_like(time)

    out = apply_fft_filter(signal, time, mode="gaussian", start_time_us=1.0, time_constant_us=0.5)
    expected = (1.0 + np.exp((1.0 / 0.5) ** 2)) / (1.0 + np.exp(((time - 1.0) / 0.5) ** 2))

    assert out == pytest.approx(expected)
    assert out[0] == pytest.approx(1.0)
    assert out[-1] < out[2]


def test_fft_asymmetry_basic_shape() -> None:
    ds = _dataset(128)
    f, real, mag = fft_asymmetry(ds)
    assert len(f) == 65
    assert len(real) == 65
    assert len(mag) == 65
    assert np.all(mag >= 0)


def test_fft_asymmetry_padding_and_time_window() -> None:
    ds = _dataset(120)
    f, real, mag = fft_asymmetry(ds, window="hann", padding_factor=2, t_min=2.0, t_max=8.0)
    assert len(f) == len(real) == len(mag)
    assert len(f) > 60


def test_fft_complex_asymmetry_manual_phase_correction_rotates_peak() -> None:
    phase_degrees = 47.5
    ds, expected_frequency = _phase_dataset(phase_degrees)

    freqs, raw_spectrum = fft_complex_asymmetry(ds)
    corrected_freqs, corrected_spectrum = fft_complex_asymmetry(ds, phase_degrees=phase_degrees)

    peak_index = int(np.argmax(np.abs(raw_spectrum[1:])) + 1)
    assert corrected_freqs[peak_index] == pytest.approx(freqs[peak_index])
    assert freqs[peak_index] == pytest.approx(expected_frequency)
    assert _angle_distance(np.angle(raw_spectrum[peak_index]), np.deg2rad(phase_degrees)) < 1e-9
    assert _angle_distance(np.angle(corrected_spectrum[peak_index]), 0.0) < 1e-9
    assert abs(corrected_spectrum.imag[peak_index]) < 1e-9


def test_fft_complex_asymmetry_t0_offset_matches_frequency_phase_term() -> None:
    ds, expected_frequency = _phase_dataset(0.0)
    t0_offset_us = 0.125

    freqs, raw_spectrum = fft_complex_asymmetry(ds)
    corrected_freqs, corrected_spectrum = fft_complex_asymmetry(ds, t0_offset_us=t0_offset_us)

    peak_index = int(np.argmax(np.abs(raw_spectrum[1:])) + 1)
    expected_angle = 2.0 * np.pi * expected_frequency * t0_offset_us

    assert corrected_freqs[peak_index] == pytest.approx(freqs[peak_index])
    assert freqs[peak_index] == pytest.approx(expected_frequency)
    assert _angle_distance(np.angle(corrected_spectrum[peak_index]), -expected_angle) < 1e-9


def test_fft_complex_asymmetry_subtract_average_signal_removes_dc_component() -> None:
    time = np.arange(256, dtype=float) * 0.05
    frequency = 12.0 / (time.size * 0.05)
    dataset = MuonDataset(
        time=time,
        asymmetry=3.0 + 0.2 * np.cos(2.0 * np.pi * frequency * time),
        error=np.full_like(time, 0.1),
        metadata={"run_number": 3},
    )

    freqs_raw, spectrum_raw = fft_complex_asymmetry(dataset, subtract_average_signal=False)
    freqs_sub, spectrum_sub = fft_complex_asymmetry(dataset, subtract_average_signal=True)

    assert freqs_sub[0] == pytest.approx(freqs_raw[0])
    assert abs(spectrum_raw[0]) > 100.0
    assert abs(spectrum_sub[0]) < 1e-9


def test_estimate_fft_phase_recovers_peak_and_average_methods() -> None:
    phase_degrees = 36.0
    ds, _expected_frequency = _phase_dataset(phase_degrees)

    freqs, spectrum = fft_complex_asymmetry(ds)

    assert estimate_fft_phase(freqs, spectrum, method="peak") == pytest.approx(phase_degrees)
    assert estimate_fft_phase(freqs, spectrum, method="average") == pytest.approx(phase_degrees)


def test_estimate_fft_phase_can_limit_the_frequency_window() -> None:
    freqs = np.array([0.05, 338.75], dtype=float)
    spectrum = np.array(
        [100.0 * np.exp(1j * np.deg2rad(-25.0)), 5.0 * np.exp(1j * np.deg2rad(110.0))],
        dtype=np.complex128,
    )

    assert estimate_fft_phase(freqs, spectrum, method="peak") == pytest.approx(-25.0)
    assert estimate_fft_phase(
        freqs,
        spectrum,
        method="peak",
        min_frequency=330.0,
        max_frequency=345.0,
    ) == pytest.approx(110.0)


def test_fourier_display_values_exposes_all_supported_channels() -> None:
    spectrum = np.array([1.0 + 1.0j, -2.0j, 0.0 + 0.0j], dtype=np.complex128)

    assert np.allclose(fourier_display_values(spectrum, display="Real"), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(
        fourier_display_values(spectrum, display="Imaginary"),
        np.array([1.0, -2.0, 0.0]),
    )
    assert np.allclose(
        fourier_display_values(spectrum, display="Magnitude"),
        np.array([np.sqrt(2.0), 2.0, 0.0]),
    )
    assert np.allclose(
        fourier_display_values(spectrum, display="Power"),
        np.array([2.0, 4.0, 0.0]),
    )
    assert np.allclose(
        fourier_display_values(spectrum, display="Phase"),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.allclose(
        fourier_display_values(spectrum, display="(Power)^1/2"),
        np.array([np.sqrt(2.0), 2.0, 0.0]),
    )
    assert np.allclose(
        fourier_display_values(spectrum, display="Phase Spectrum"),
        np.array([45.0, -90.0, 0.0]),
    )
    assert np.allclose(fourier_display_values(spectrum, display="Cos"), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(
        fourier_display_values(spectrum, display="Sin"),
        np.array([1.0, -2.0, 0.0]),
    )


def test_fourier_mode_helpers_cover_wimda_modes_and_legacy_aliases() -> None:
    assert canonical_fourier_display_mode("(Power)^1/2") == "power_sqrt"
    assert canonical_fourier_display_mode("Phase Spectrum") == "phase_spectrum"
    assert canonical_fourier_display_mode("Cos") == "cos"
    assert canonical_fourier_display_mode("Sin") == "sin"
    assert canonical_fourier_display_mode("Phase") == "phase_corrected"
    assert canonical_fourier_display_mode("Real") == "real"
    assert fourier_mode_uses_phase_correction("Phase") is True
    assert fourier_mode_uses_phase_correction("(Power)^1/2") is False
    assert fourier_mode_uses_phase_correction("Cos") is False


def test_exclude_frequency_ranges_zeroes_bins_inside_requested_windows() -> None:
    freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    filtered = exclude_frequency_ranges(freqs, values, [(1.5, 0.6), (4.0, 0.1)])

    assert np.allclose(filtered, np.array([5.0, 0.0, 0.0, 2.0, 0.0]))


def test_average_fourier_display_values_can_estimate_wimda_style_error() -> None:
    averaged, error = average_fourier_display_values(
        [
            np.array([1.0, 2.0, 4.0]),
            np.array([3.0, 4.0, 6.0]),
        ],
        estimate_error=True,
    )

    assert np.allclose(averaged, np.array([2.0, 3.0, 5.0]))
    assert np.allclose(error, np.array([np.sqrt(0.5), np.sqrt(0.5), np.sqrt(0.5)]))


def test_build_group_signal_dataset_centers_group_counts() -> None:
    counts_a = np.array([100.0, 120.0, 140.0, 120.0, 100.0], dtype=float)
    counts_b = np.array([90.0, 92.0, 94.0, 92.0, 90.0], dtype=float)
    run = Run(
        run_number=42,
        histograms=[
            Histogram(counts=counts_a, bin_width=0.01, t0_bin=1),
            Histogram(counts=counts_b, bin_width=0.01, t0_bin=1),
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Left", 2: "Right"},
            "first_good_bin": 1,
            "last_good_bin": 4,
            "deadtime_correction": False,
        },
        metadata={"run_number": 42},
    )

    dataset = build_group_signal_dataset(run, 1)

    assert dataset.metadata["group_id"] == 1
    assert dataset.metadata["group_name"] == "Left"
    assert dataset.metadata["run_label"] == "42 Left"
    assert dataset.time[0] == pytest.approx(0.0)
    assert np.mean(dataset.asymmetry) == pytest.approx(0.0)
    assert np.all(dataset.error > 0.0)


def test_build_group_signal_dataset_uses_common_run_t0_reference() -> None:
    counts = np.array([100.0, 120.0, 180.0, 120.0, 100.0, 90.0], dtype=float)
    run = Run(
        run_number=43,
        histograms=[
            Histogram(counts=counts, bin_width=0.01, t0_bin=0),
            Histogram(counts=counts, bin_width=0.01, t0_bin=2),
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Early", 2: "Late"},
            "first_good_bin": 0,
            "last_good_bin": 5,
            "deadtime_correction": False,
        },
        metadata={"run_number": 43},
    )

    early = build_group_signal_dataset(run, 1)
    late = build_group_signal_dataset(run, 2)

    assert early.time[0] == pytest.approx(late.time[0])
    assert int(np.argmax(early.asymmetry)) == int(np.argmax(late.asymmetry)) + 2


def test_build_group_signal_dataset_applies_bunching_factor_to_group_counts() -> None:
    run = Run(
        run_number=44,
        histograms=[
            Histogram(counts=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), bin_width=0.01, t0_bin=0),
        ],
        grouping={
            "groups": {1: [1]},
            "group_names": {1: "Grouped"},
            "first_good_bin": 0,
            "last_good_bin": 5,
            "bunching_factor": 2,
            "deadtime_correction": False,
        },
        metadata={"run_number": 44},
    )

    dataset = build_group_signal_dataset(run, 1, center_signal=False)

    expected_scale = np.exp(dataset.time / 2.1969811)

    assert np.allclose(dataset.time, np.array([0.005, 0.025, 0.045]))
    assert np.allclose(dataset.asymmetry, np.array([3.0, 7.0, 11.0]) * expected_scale)
    assert np.allclose(dataset.error, np.sqrt(np.array([3.0, 7.0, 11.0])) * expected_scale)


def test_build_group_signal_dataset_applies_wimda_style_lifetime_correction() -> None:
    run = Run(
        run_number=45,
        histograms=[
            Histogram(counts=np.array([10.0, 10.0, 10.0]), bin_width=0.5, t0_bin=0),
        ],
        grouping={
            "groups": {1: [1]},
            "group_names": {1: "Grouped"},
            "first_good_bin": 0,
            "last_good_bin": 2,
            "deadtime_correction": False,
        },
        metadata={"run_number": 45},
    )

    dataset = build_group_signal_dataset(run, 1, center_signal=False)

    expected_scale = np.exp(dataset.time / 2.1969811)

    assert dataset.metadata["fourier_lifetime_corrected"] is True
    assert np.allclose(dataset.asymmetry, 10.0 * expected_scale)
    assert np.allclose(dataset.error, np.sqrt(10.0) * expected_scale)


def test_build_group_signal_dataset_applies_group_background_correction() -> None:
    run = Run(
        run_number=46,
        histograms=[
            Histogram(counts=np.array([12.0, 12.0, 50.0, 50.0]), bin_width=0.01, t0_bin=0),
            Histogram(counts=np.array([21.0, 21.0, 70.0, 70.0]), bin_width=0.01, t0_bin=0),
        ],
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Forward", 2: "Backward"},
            "forward_group": 1,
            "backward_group": 2,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "deadtime_correction": False,
            "background_correction": True,
            "background_values": [12.0, 21.0],
        },
        metadata={"run_number": 46, "facility": "PSI"},
    )

    forward = build_group_signal_dataset(
        run,
        1,
        center_signal=False,
        apply_lifetime_correction=False,
    )
    backward = build_group_signal_dataset(
        run,
        2,
        center_signal=False,
        apply_lifetime_correction=False,
    )

    np.testing.assert_allclose(forward.asymmetry, np.array([0.0, 0.0, 38.0, 38.0]))
    np.testing.assert_allclose(backward.asymmetry, np.array([0.0, 0.0, 49.0, 49.0]))
    assert forward.metadata["fourier_background_corrected"] is True
    assert backward.metadata["fourier_background_corrected"] is True
    assert forward.metadata["fourier_background_value"] == pytest.approx(12.0)
    assert backward.metadata["fourier_background_value"] == pytest.approx(21.0)


def _pulsed_tail_fit_run(seed: int = 1, background_per_bin: float = 6.0) -> Run:
    """An ISIS-style pulsed synthetic run with a flat uncorrelated background."""
    from asymmetry.core.simulate import build_builtin_template, simulate_run

    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template,
        lambda t: 20.0 * np.cos(2.0 * np.pi * 1.0 * t) * np.exp(-0.3 * t),
        total_events=4.0e7,
        seed=seed,
        background_per_bin=background_per_bin,
    )
    run.grouping["background_correction"] = True
    run.grouping["background_mode"] = "tail_fit"
    return run


def test_fourier_tail_fit_matches_grouping_side_fit() -> None:
    # The Fourier-input tail fit must agree with the time-domain reduction's
    # tail fit on the same forward grouped counts — the same Poisson estimator,
    # the same data.
    from asymmetry.core.transform.background import (
        apply_grouped_background_correction,
        fit_tail_background,
    )
    from asymmetry.core.transform.grouping import group_forward_backward

    run = _pulsed_tail_fit_run()
    grouping = run.grouping

    forward = build_group_signal_dataset(run, 1, apply_lifetime_correction=True)
    assert forward.metadata["fourier_background_corrected"] is True
    fourier_value = forward.metadata["fourier_background_value"]

    fb = group_forward_backward(run.histograms, grouping)
    bin_width = float(run.histograms[0].bin_width)
    direct = fit_tail_background(
        fb.forward,
        bin_width_us=bin_width,
        t0_bin=int(grouping["t0_bin"]),
        last_good_bin=int(grouping["last_good_bin"]),
    )
    assert direct.ok
    assert fourier_value == pytest.approx(direct.rate_per_us * bin_width, rel=1e-9)

    reduction = apply_grouped_background_correction(
        fb.forward,
        fb.backward,
        grouping=grouping,
        t0_bin=int(grouping["t0_bin"]),
        bin_width_us=bin_width,
        last_good_bin=int(grouping["last_good_bin"]),
    )
    assert reduction.method == "tail_fit"
    assert fourier_value == pytest.approx(reduction.values[0], rel=1e-9)


def test_fourier_tail_fit_improves_baseline() -> None:
    # Subtracting the flat background before the lifetime correction removes the
    # growing exp(t/tau) ramp it would otherwise become, cutting the spurious
    # low-frequency (DC) power in the FFT input.
    run = _pulsed_tail_fit_run()

    corrected = build_group_signal_dataset(run, 1, center_signal=False)
    run.grouping["background_correction"] = False
    uncorrected = build_group_signal_dataset(run, 1, center_signal=False)

    # The lifetime-corrected signal's offset (its DC component) is far smaller
    # once the flat background is removed.
    assert abs(float(np.mean(corrected.asymmetry))) < abs(float(np.mean(uncorrected.asymmetry)))


def test_fourier_tail_fit_not_hijacked_by_stray_range_keys() -> None:
    # tail_fit re-fits each group; a stray background_ranges key left from an
    # earlier mode must not silently turn it into a range subtraction.
    run = _pulsed_tail_fit_run()
    run.grouping["background_ranges"] = [[0, 5], [0, 5]]

    forward = build_group_signal_dataset(run, 1, apply_lifetime_correction=True)
    assert forward.metadata["fourier_background_corrected"] is True
    # A range over bins 0..5 (deep in the muon pulse) would subtract a huge
    # value; the tail-fit rate is far smaller, so a mismatch proves the mode
    # was not hijacked.
    from asymmetry.core.transform.background import fit_tail_background
    from asymmetry.core.transform.grouping import group_forward_backward

    fb = group_forward_backward(run.histograms, run.grouping)
    bin_width = float(run.histograms[0].bin_width)
    direct = fit_tail_background(
        fb.forward,
        bin_width_us=bin_width,
        t0_bin=int(run.grouping["t0_bin"]),
        last_good_bin=int(run.grouping["last_good_bin"]),
    )
    assert forward.metadata["fourier_background_value"] == pytest.approx(
        direct.rate_per_us * bin_width, rel=1e-9
    )


def test_maxent_wrapper_requires_run_with_histograms() -> None:
    # The Fourier-module wrapper delegates to asymmetry.core.maxent, which is
    # a grouped raw-count algorithm: datasets without a Run are rejected.
    with pytest.raises(ValueError, match="raw detector histograms"):
        maxent(_dataset(32))


# ── optimize_phase_entropy ─────────────────────────────────────────────────────


def _spectrum_with_known_phase(
    *,
    n: int = 256,
    dt_us: float = 0.05,
    bin_index: int = 12,
    phase_degrees: float,
) -> np.ndarray:
    """Return a complex FFT spectrum of a cosine at a known phase offset.

    Uses an on-grid frequency (bin_index / (n * dt_us)) so the signal lands
    exactly at one bin with no spectral leakage.
    """
    frequency_mhz = bin_index / (n * dt_us)
    time = np.arange(n, dtype=float) * dt_us
    signal = np.cos(2.0 * np.pi * frequency_mhz * time + np.deg2rad(phase_degrees))
    signal -= signal.mean()
    return np.fft.rfft(signal, n=n)


def test_optimize_phase_entropy_corrects_large_negative_phase_offset() -> None:
    """Optimizer should remove a phase that causes the dominant bin to be negative.

    At 135° offset the raw real part of the dominant bin is ≈ −0.707·|F|, a
    large negative value that drives up the penalty term.  The optimizer should
    find c0 ≈ −135° and produce a clearly positive dominant-bin real part.

    Uses an on-grid frequency so the signal lands exactly at bin 12 with no
    spectral leakage, giving the optimizer a clean cost landscape.
    """
    true_phase = 135.0  # degrees — raw spectrum has large negative real part
    n = 256
    bin_idx = 12
    spectrum = _spectrum_with_known_phase(n=n, bin_index=bin_idx, phase_degrees=true_phase)

    real_opt, _c0, _c1 = optimize_phase_entropy(spectrum, min_bin=1)

    # Dominant bin has large negative real, small at other bins
    assert spectrum[bin_idx].real < -50.0  # confirm raw is negative
    # After optimization the dominant bin should be clearly positive
    assert real_opt[bin_idx] > 50.0


def test_optimize_phase_entropy_returns_same_length_as_input() -> None:
    spectrum = _spectrum_with_known_phase(n=128, bin_index=8, phase_degrees=55.0)
    real_opt, c0, c1 = optimize_phase_entropy(spectrum)

    assert real_opt.shape == spectrum.shape
    assert real_opt.dtype == np.float64
    assert np.isfinite(c0)
    assert np.isfinite(c1)


def test_optimize_phase_entropy_handles_empty_spectrum() -> None:
    real_opt, c0, c1 = optimize_phase_entropy(np.zeros(0, dtype=np.complex128))
    assert real_opt.size == 0
    assert c0 == 0.0
    assert c1 == 0.0


def test_optimize_phase_entropy_zero_phase_input_is_stable() -> None:
    """A spectrum already on the real axis should produce a positive dominant bin."""
    n = 256
    bin_idx = 12
    spectrum = _spectrum_with_known_phase(n=n, bin_index=bin_idx, phase_degrees=0.0)
    real_opt, _c0, _c1 = optimize_phase_entropy(spectrum, min_bin=1)
    # Dominant bin is already large and positive — optimizer should keep it positive
    assert real_opt[bin_idx] > 50.0


def test_fourier_mode_uses_entropy_optimizer() -> None:
    assert fourier_mode_uses_entropy_optimizer("phaseOptReal")
    assert fourier_mode_uses_entropy_optimizer("phaseoptreal")
    assert fourier_mode_uses_entropy_optimizer("phase_opt_real")
    assert not fourier_mode_uses_entropy_optimizer("Phase")
    assert not fourier_mode_uses_entropy_optimizer("(Power)^1/2")
    assert not fourier_mode_uses_entropy_optimizer("Cos")


def test_canonical_fourier_display_mode_phase_opt_real() -> None:
    assert canonical_fourier_display_mode("phaseOptReal") == "phase_opt_real"
    assert canonical_fourier_display_mode("phaseoptreal") == "phase_opt_real"
    assert canonical_fourier_display_mode("phase_opt_real") == "phase_opt_real"


def test_fourier_display_values_phase_opt_real_returns_real_part() -> None:
    """fourier_display_values with phase_opt_real should return the real part."""
    spectrum = np.array([1.0 + 2.0j, 3.0 - 1.5j, 0.5 + 0.5j], dtype=np.complex128)
    result = fourier_display_values(spectrum, display="phaseOptReal")
    np.testing.assert_array_equal(result, spectrum.real)
