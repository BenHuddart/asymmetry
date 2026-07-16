"""Tests for the matched-apodisation suggestion.

The estimator must round-trip synthetic truth: a decaying cosine of known
relaxation rate yields the matching filter time constant within leakage
tolerance, and every "no physical width to match" case returns ``None``
rather than a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier import fft_complex_asymmetry, suggest_matched_apodisation


def _spectrum_of(signal: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unapodised padded magnitude spectrum of a synthetic time signal."""
    dataset = MuonDataset(
        time=time,
        asymmetry=signal,
        error=np.full_like(time, 0.01),
        metadata={},
    )
    freqs, spectrum = fft_complex_asymmetry(
        dataset,
        window="none",
        padding_factor=8,
        subtract_average_signal=True,
    )
    return freqs, np.abs(spectrum)


def _time_axis(n: int = 2048, dt: float = 0.02) -> np.ndarray:
    return np.arange(n, dtype=float) * dt


def test_lorentzian_line_recovers_matched_time_constant() -> None:
    time = _time_axis()
    rate = 0.5  # µs⁻¹ → matched Lorentzian τ = 2.0 µs
    signal = np.exp(-rate * time) * np.cos(2.0 * np.pi * 3.0 * time)
    freqs, magnitude = _spectrum_of(signal, time)

    suggestion = suggest_matched_apodisation(freqs, magnitude, window="lorentzian")

    assert suggestion is not None
    assert suggestion.window == "lorentzian"
    assert suggestion.line_frequency_mhz == pytest.approx(3.0, abs=0.05)
    assert suggestion.time_constant_us == pytest.approx(1.0 / rate, rel=0.15)


def test_gaussian_line_recovers_matched_time_constant() -> None:
    time = _time_axis()
    sigma = 0.8  # µs⁻¹ → matched Gaussian (exp(-(t/τ)²)) τ = √2/σ ≈ 1.768 µs
    signal = np.exp(-0.5 * (sigma * time) ** 2) * np.cos(2.0 * np.pi * 3.0 * time)
    freqs, magnitude = _spectrum_of(signal, time)

    suggestion = suggest_matched_apodisation(freqs, magnitude, window="gaussian")

    assert suggestion is not None
    assert suggestion.window == "gaussian"
    assert suggestion.time_constant_us == pytest.approx(np.sqrt(2.0) / sigma, rel=0.15)


def test_featureless_noise_returns_none() -> None:
    rng = np.random.default_rng(9)
    time = _time_axis()
    freqs, magnitude = _spectrum_of(rng.normal(0.0, 0.05, time.size), time)

    assert suggest_matched_apodisation(freqs, magnitude) is None


def test_resolution_limited_line_returns_none() -> None:
    """An undamped cosine's width is the transform's, not the sample's."""
    time = _time_axis()
    signal = np.cos(2.0 * np.pi * 3.0 * time)
    dataset = MuonDataset(
        time=time,
        asymmetry=signal,
        error=np.full_like(time, 0.01),
        metadata={},
    )
    # No padding: the line then spans ~1 bin and must be rejected as
    # resolution-limited rather than matched to a huge time constant.
    freqs, spectrum = fft_complex_asymmetry(
        dataset,
        window="none",
        padding_factor=1,
        subtract_average_signal=True,
    )

    assert suggest_matched_apodisation(freqs, np.abs(spectrum)) is None


def test_frequency_window_selects_the_intended_line() -> None:
    """With two lines, the search window picks the one it brackets."""
    time = _time_axis()
    rate_slow, rate_fast = 0.3, 1.2
    signal = np.exp(-rate_slow * time) * np.cos(2.0 * np.pi * 2.0 * time) + np.exp(
        -rate_fast * time
    ) * np.cos(2.0 * np.pi * 8.0 * time)
    freqs, magnitude = _spectrum_of(signal, time)

    low = suggest_matched_apodisation(
        freqs, magnitude, min_frequency_mhz=1.0, max_frequency_mhz=4.0
    )
    high = suggest_matched_apodisation(
        freqs, magnitude, min_frequency_mhz=6.0, max_frequency_mhz=10.0
    )

    assert low is not None and high is not None
    assert low.line_frequency_mhz == pytest.approx(2.0, abs=0.1)
    assert high.line_frequency_mhz == pytest.approx(8.0, abs=0.1)
    assert low.time_constant_us == pytest.approx(1.0 / rate_slow, rel=0.2)
    assert high.time_constant_us == pytest.approx(1.0 / rate_fast, rel=0.2)


def test_unknown_window_kind_raises() -> None:
    freqs = np.linspace(0.0, 10.0, 128)
    with pytest.raises(ValueError):
        suggest_matched_apodisation(freqs, np.ones_like(freqs), window="hann")


def _raw_prominence_ratio(
    freqs: np.ndarray, magnitude: np.ndarray, fmin: float, fmax: float
) -> float:
    """Peak/median power ratio in the search window (mirrors the fast path)."""
    power = magnitude**2
    mask = (freqs > fmin) & (freqs <= fmax)
    windowed = power[mask]
    return float(windowed.max() / np.median(windowed))


def test_buried_lorentzian_line_detected_by_matched_scan() -> None:
    """A line too weak for the raw-prominence fast path is still found.

    Padding factor 16, a fast lifetime relaxation (Gamma ~ 1.6 MHz, matching
    the real dataset that validated the fix), and additive white noise scaled so the
    raw peak sits below ``_LINE_PROMINENCE_POWER`` (16x) — the fast path
    must decline before the matched-scan fallback ever runs.
    """
    time = _time_axis()
    rate = 5.0  # us^-1 -> Lorentzian Gamma = rate / pi ~= 1.59 MHz, tau = 0.2 us
    rng = np.random.default_rng(1)
    signal = 0.1 * np.exp(-rate * time) * np.cos(2.0 * np.pi * 3.0 * time) + rng.normal(
        0.0, 0.05, time.size
    )
    dataset = MuonDataset(time=time, asymmetry=signal, error=np.full_like(time, 0.01), metadata={})
    freqs, spectrum = fft_complex_asymmetry(
        dataset, window="none", padding_factor=16, subtract_average_signal=True
    )
    magnitude = np.abs(spectrum)

    # The fast path must be genuinely unable to fire on the raw spectrum —
    # otherwise a green result would reflect the unchanged fast path, not the
    # new matched-scan fallback.
    assert _raw_prominence_ratio(freqs, magnitude, 1.0, 6.0) < 16.0

    suggestion = suggest_matched_apodisation(
        freqs, magnitude, window="lorentzian", min_frequency_mhz=1.0, max_frequency_mhz=6.0
    )

    assert suggestion is not None
    assert suggestion.window == "lorentzian"
    assert suggestion.time_constant_us == pytest.approx(1.0 / rate, rel=0.3)


def test_buried_gaussian_line_detected_by_matched_scan() -> None:
    """The matched-scan fallback also works for the Gaussian window kind."""
    time = _time_axis()
    sigma = 2.0  # us^-1 -> Gaussian tau = sqrt(2) / sigma
    rng = np.random.default_rng(13)
    signal = 0.1 * np.exp(-0.5 * (sigma * time) ** 2) * np.cos(
        2.0 * np.pi * 3.0 * time
    ) + rng.normal(0.0, 0.05, time.size)
    dataset = MuonDataset(time=time, asymmetry=signal, error=np.full_like(time, 0.01), metadata={})
    freqs, spectrum = fft_complex_asymmetry(
        dataset, window="none", padding_factor=16, subtract_average_signal=True
    )
    magnitude = np.abs(spectrum)

    assert _raw_prominence_ratio(freqs, magnitude, 1.0, 6.0) < 16.0

    suggestion = suggest_matched_apodisation(
        freqs, magnitude, window="gaussian", min_frequency_mhz=1.0, max_frequency_mhz=6.0
    )

    assert suggestion is not None
    assert suggestion.window == "gaussian"
    assert suggestion.time_constant_us == pytest.approx(np.sqrt(2.0) / sigma, rel=0.3)


def test_pure_noise_returns_none_through_both_stages_at_heavy_padding() -> None:
    """Heavy zero-padding must not trip the matched-scan fallback on noise.

    This is the false positive the intrinsic-resolution anchoring exists to
    prevent: at padding 16 the padded grid is far finer than one resolution
    element, so a fallback anchored to the *grid* bin width (rather than the
    intrinsic resolution) would scan sub-resolution kernels and chase
    correlated noise. Seed 9 is pinned (like the existing featureless-noise
    test) because scanning ~10 kernel widths against thousands of bins is
    itself a look-elsewhere search with a small residual false-positive rate
    on adversarial noise draws — see ``_MATCHED_SCAN_SNR_THRESHOLD``'s
    docstring; this seed's fallback SNR sits comfortably below threshold.
    """
    from asymmetry.core.fourier.apodisation import (
        _DC_CUT_FRACTION,
        _MATCHED_SCAN_SNR_THRESHOLD,
        _estimate_intrinsic_resolution,
        _matched_scan_best,
    )

    time = _time_axis()
    rng = np.random.default_rng(9)
    signal = rng.normal(0.0, 0.05, time.size)
    dataset = MuonDataset(time=time, asymmetry=signal, error=np.full_like(time, 0.01), metadata={})
    freqs, spectrum = fft_complex_asymmetry(
        dataset, window="none", padding_factor=16, subtract_average_signal=True
    )
    magnitude = np.abs(spectrum)

    assert suggest_matched_apodisation(freqs, magnitude, window="lorentzian") is None

    # Confirm the fallback was actually reachable (a resolution estimate was
    # available) and its own SNR — not just the end-to-end None — stays well
    # below the detection threshold, so a green result is not an accident of
    # the resolution estimate declining early.
    power = magnitude**2
    f_max = float(np.max(freqs))
    mask = freqs > f_max * _DC_CUT_FRACTION
    f_win = freqs[mask]
    v_win = power[mask]
    bin_width = float(np.median(np.diff(f_win)))
    resolution = _estimate_intrinsic_resolution(v_win, bin_width)
    assert resolution is not None
    best = _matched_scan_best(f_win, v_win, "lorentzian", bin_width, resolution)
    assert best is not None
    assert best[0] < _MATCHED_SCAN_SNR_THRESHOLD - 2.0  # comfortable margin, not marginal


def test_resolution_limited_deconvolved_width_returns_none_via_fallback() -> None:
    """An undamped line is resolution-limited even through the matched scan.

    Unpadded (``padding_factor=1``, matching the existing fast-path
    resolution-limited test) so the line has no physical width beyond the
    transform's own response — but weak enough relative to noise that the
    raw-prominence fast path declines, forcing the matched-scan fallback to
    run. The fallback detects the (very strong, once smoothed) candidate at
    high SNR, but every kernel's deconvolved width collapses to (near) zero
    once the kernel's own contribution is removed, so the same
    ``_MIN_FWHM_BINS`` guard used by the fast path must still return
    ``None`` — proving the guard applies to the fallback's DECONVOLVED
    width, not just its raw detection SNR.
    """
    from asymmetry.core.fourier.apodisation import (
        _DC_CUT_FRACTION,
        _estimate_intrinsic_resolution,
        _matched_scan_best,
        _prominence_line,
    )

    time = _time_axis()
    rng = np.random.default_rng(1)
    signal = 0.1 * np.cos(2.0 * np.pi * 3.0 * time) + rng.normal(0.0, 0.05, time.size)
    dataset = MuonDataset(time=time, asymmetry=signal, error=np.full_like(time, 0.01), metadata={})
    freqs, spectrum = fft_complex_asymmetry(
        dataset, window="none", padding_factor=1, subtract_average_signal=True
    )
    magnitude = np.abs(spectrum)

    # Confirm the fallback (not the fast path) is what's being exercised,
    # and that it does find a high-SNR candidate before the width guard
    # rejects it — otherwise a green result could just mean "nothing found".
    power = magnitude**2
    f_max = float(np.max(freqs))
    mask = freqs > f_max * _DC_CUT_FRACTION
    f_win = freqs[mask]
    v_win = power[mask]
    bin_width = float(np.median(np.diff(f_win)))
    assert _prominence_line(f_win, v_win, "lorentzian", bin_width) is None
    resolution = _estimate_intrinsic_resolution(v_win, bin_width)
    assert resolution is not None
    best = _matched_scan_best(f_win, v_win, "lorentzian", bin_width, resolution)
    assert best is not None and best[0] >= 8.0

    assert suggest_matched_apodisation(freqs, magnitude, window="lorentzian") is None


def test_matched_scan_floor_excludes_the_detecting_kernels_peak_region() -> None:
    """A strong buried line is still detected — its own peak cannot inflate the floor.

    Directly exercises ``_matched_scan_best``'s exclusion window by comparing
    the (correct) SNR it reports — floor computed with the peak region
    excluded — against what a naive median/MAD over the WHOLE smoothed
    spectrum would give. A strong enough line pulls its own naive floor up
    and its own naive MAD up, suppressing the naive SNR well below
    threshold; the exclusion is what keeps the real detection intact.
    """
    from asymmetry.core.fourier.apodisation import _matched_scan_best

    time = _time_axis()
    rate = 5.0
    rng = np.random.default_rng(1)
    signal = 0.3 * np.exp(-rate * time) * np.cos(2.0 * np.pi * 3.0 * time) + rng.normal(
        0.0, 0.05, time.size
    )
    dataset = MuonDataset(time=time, asymmetry=signal, error=np.full_like(time, 0.01), metadata={})
    freqs, spectrum = fft_complex_asymmetry(
        dataset, window="none", padding_factor=16, subtract_average_signal=True
    )
    magnitude = np.abs(spectrum)
    power = magnitude**2
    lower, upper = 1.0, 6.0
    mask = (freqs > lower) & (freqs <= upper)
    f_win = freqs[mask]
    v_win = power[mask]
    bin_width = float(np.median(np.diff(f_win)))
    resolution = 16.0 * bin_width  # matches the padding factor used above

    best = _matched_scan_best(f_win, v_win, "lorentzian", bin_width, resolution)
    assert best is not None
    snr, _kernel_fwhm, smoothed, peak_index, _floor_median = best
    assert snr >= 8.0

    naive_median = float(np.median(smoothed))
    naive_mad = float(np.median(np.abs(smoothed - naive_median))) * 1.4826
    naive_snr = (float(smoothed[peak_index]) - naive_median) / naive_mad

    assert naive_snr < snr
    assert naive_snr < 8.0


def test_spectrum_metadata_records_window() -> None:
    """Computed spectra carry the apodisation they were made with (moments caveat)."""
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.fourier import GroupSpectrumConfig, compute_average_group_spectrum

    time = _time_axis(512, 0.04)
    counts = 4000.0 * np.exp(-time / 2.1969811) * (1.0 + 0.2 * np.cos(2.0 * np.pi * 2.7 * time))
    run = Run(
        run_number=9,
        histograms=[
            Histogram(counts=counts, bin_width=0.04),
            Histogram(counts=counts * 0.9, bin_width=0.04),
        ],
        metadata={"field": 200.0},
        grouping={"groups": {1: [1], 2: [2]}, "deadtime_correction": False},
    )
    plain = compute_average_group_spectrum(run, GroupSpectrumConfig(window="none"))
    filtered = compute_average_group_spectrum(
        run, GroupSpectrumConfig(window="lorentzian", filter_time_constant_us=1.8)
    )

    assert plain.metadata["fourier_window"] == "none"
    assert filtered.metadata["fourier_window"] == "lorentzian"
    assert filtered.metadata["fourier_filter_time_constant_us"] == pytest.approx(1.8)
