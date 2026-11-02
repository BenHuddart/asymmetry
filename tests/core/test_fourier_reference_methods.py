"""Analytic checks of FFT phase rotation and grouped-count conventions."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.fft import fft_complex_asymmetry
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _make_dataset(
    *, frequency_mhz: float, phase_degrees: float, n: int = 256, dt_us: float = 0.05
) -> MuonDataset:
    time = np.arange(n, dtype=float) * dt_us
    phase_radians = np.deg2rad(phase_degrees)
    asymmetry = 0.2 * np.cos(2 * np.pi * frequency_mhz * time + phase_radians)
    error = np.full(n, 0.01)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=error,
        metadata={"run_number": 7001},
    )


def _wimda_projected_real(
    freqs_mhz: np.ndarray,
    spectrum: np.ndarray,
    *,
    phase_degrees: float,
    t0_offset_us: float = 0.0,
) -> np.ndarray:
    """Return WiMDA-style projected real spectrum.

    WiMDA projects cosine and sine components into one real-valued spectrum
    using a manual phase and, optionally, a time-zero-derived frequency term.
    With NumPy's FFT sign convention this is equivalent to taking the real part
    of the complex spectrum after a phase rotation by ``phase + 2π f t0``.
    """

    angles = np.deg2rad(phase_degrees) + 2.0 * np.pi * freqs_mhz * t0_offset_us
    return np.real(spectrum * np.exp(-1j * angles))


def _wimda_group_lifetime_correction(signal: np.ndarray, time_us: np.ndarray) -> np.ndarray:
    """Apply WiMDA's grouped-count FFT lifetime correction."""

    return np.asarray(signal, dtype=float) * np.exp(
        np.asarray(time_us, dtype=float) / MUON_LIFETIME_US
    )


def test_wimda_manual_phase_projection_matches_asymmetry_rotation() -> None:
    dataset = _make_dataset(frequency_mhz=0.9375, phase_degrees=33.0)
    freqs, raw_spectrum = fft_complex_asymmetry(dataset)
    _rotated_freqs, rotated_spectrum = fft_complex_asymmetry(dataset, phase_degrees=33.0)

    projected = _wimda_projected_real(freqs, raw_spectrum, phase_degrees=33.0)

    np.testing.assert_allclose(rotated_spectrum.real, projected, atol=1e-12)


def test_wimda_t0_phase_projection_matches_asymmetry_rotation() -> None:
    dataset = _make_dataset(frequency_mhz=0.9375, phase_degrees=12.0)
    t0_offset_us = 0.1375
    freqs, raw_spectrum = fft_complex_asymmetry(dataset)
    _rotated_freqs, rotated_spectrum = fft_complex_asymmetry(
        dataset,
        t0_offset_us=t0_offset_us,
    )

    projected = _wimda_projected_real(
        freqs,
        raw_spectrum,
        phase_degrees=0.0,
        t0_offset_us=t0_offset_us,
    )

    np.testing.assert_allclose(rotated_spectrum.real, projected, atol=1e-12)


def test_grouped_fft_source_uses_decay_corrected_counts() -> None:
    raw_counts = np.array([120.0, 90.0, 70.0, 55.0], dtype=float)
    run = Run(
        run_number=7002,
        histograms=[Histogram(counts=raw_counts, bin_width=0.4, t0_bin=0)],
        grouping={
            "groups": {1: [1]},
            "group_names": {1: "Grouped"},
            "first_good_bin": 0,
            "last_good_bin": 3,
            "deadtime_correction": False,
        },
        metadata={"run_number": 7002},
    )

    dataset = build_group_signal_dataset(run, 1, center_signal=False)
    expected = _wimda_group_lifetime_correction(raw_counts, dataset.time)

    np.testing.assert_allclose(dataset.asymmetry, expected, rtol=1e-12, atol=1e-12)
