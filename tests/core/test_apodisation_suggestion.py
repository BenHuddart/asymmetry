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
