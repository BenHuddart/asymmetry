"""Reference-method comparison tests for Fourier porting work.

These tests encode meaningful behavioral differences between the existing
reference programs so later Asymmetry implementation work can verify the right
target rather than only matching a generic FFT surface.
"""

from __future__ import annotations

import math

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.fft import fft_complex_asymmetry
from asymmetry.core.fourier.grouped import build_group_signal_dataset
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def _make_dataset(*, frequency_mhz: float, phase_degrees: float, n: int = 256, dt_us: float = 0.05) -> MuonDataset:
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

    return np.asarray(signal, dtype=float) * np.exp(np.asarray(time_us, dtype=float) / MUON_LIFETIME_US)


def _musrfit_lifetime_correction(signal: np.ndarray, dt_us: float, fudge: float = 1.0) -> np.ndarray:
    """Apply musrfit's theory-free lifetime correction to one trace."""

    indices = np.arange(signal.size, dtype=float)
    corrected = signal * np.exp(indices * dt_us / MUON_LIFETIME_US)
    n0 = corrected.mean() * fudge
    return (corrected - n0) / n0


def _musrfit_linear_phase_real(spectrum: np.ndarray, c0: float, c1: float) -> np.ndarray:
    """Return musrfit's linearly phase-corrected real Fourier spectrum."""

    n = spectrum.size
    weights = np.arange(n, dtype=float) / float(n)
    angles = c0 + c1 * weights
    return spectrum.real * np.cos(angles) - spectrum.imag * np.sin(angles)


def _mantid_phase_quad(
    detector_signals: np.ndarray,
    *,
    asymmetries: np.ndarray,
    phases_radians: np.ndarray,
    n0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a simplified Mantid PhaseQuad combination for residual signals.

    This mirrors the coefficient construction in ``PhaseQuadMuon::squash`` for
    already decay-corrected detector residuals.
    """

    if detector_signals.ndim != 2:
        raise ValueError("detector_signals must be a 2D array")
    if n0 is None:
        n0 = np.ones(detector_signals.shape[0], dtype=float)

    max_asymmetry = float(np.max(asymmetries))
    if max_asymmetry <= 0.0:
        raise ValueError("asymmetries must contain a positive value")

    x = n0 * (asymmetries / max_asymmetry) * np.cos(phases_radians)
    y = n0 * (asymmetries / max_asymmetry) * np.sin(phases_radians)
    mu_lambda = np.array(
        [
            [np.sum(x * x), np.sum(x * y)],
            [np.sum(x * y), np.sum(y * y)],
        ],
        dtype=float,
    )
    coefficients = np.linalg.inv(mu_lambda) @ np.vstack([x, y])
    aj = coefficients[0]
    bj = coefficients[1]
    real = np.tensordot(aj, detector_signals, axes=(0, 0))
    imag = np.tensordot(bj, detector_signals, axes=(0, 0))
    return real, imag


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


def test_wimda_grouped_fft_source_uses_decay_corrected_counts() -> None:
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
    musrfit_like = _musrfit_lifetime_correction(raw_counts, 0.4)

    np.testing.assert_allclose(dataset.asymmetry, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(dataset.asymmetry, musrfit_like)


def test_musrfit_lifetime_correction_flattens_decay_before_fft() -> None:
    dt_us = 0.01
    time = np.arange(1200, dtype=float) * dt_us
    raw_counts = np.exp(-time / MUON_LIFETIME_US) * (1.0 + 0.18 * np.cos(2 * np.pi * 0.8 * time))
    raw_zero_mean = raw_counts - raw_counts.mean()
    corrected = _musrfit_lifetime_correction(raw_counts, dt_us)

    half = raw_counts.size // 2
    raw_ratio = float(np.std(raw_zero_mean[half:]) / np.std(raw_zero_mean[:half]))
    corrected_ratio = float(np.std(corrected[half:]) / np.std(corrected[:half]))

    assert raw_ratio < 0.35
    assert corrected_ratio > 0.8


def test_musrfit_linear_phase_profile_is_not_a_single_wimda_phase() -> None:
    spectrum = np.zeros(32, dtype=np.complex128)
    spectrum[3] = 1.0 + 0.8j
    spectrum[11] = 0.65 - 0.55j
    spectrum[19] = 0.2 + 0.4j

    target = _musrfit_linear_phase_real(spectrum, c0=0.2, c1=1.7)

    best_rmse = math.inf
    for phase_degrees in np.linspace(-180.0, 180.0, 1441):
        candidate = _wimda_projected_real(
            np.zeros(spectrum.size, dtype=float),
            spectrum,
            phase_degrees=float(phase_degrees),
        )
        rmse = float(np.sqrt(np.mean((candidate - target) ** 2)))
        best_rmse = min(best_rmse, rmse)

    assert best_rmse > 0.05


def test_mantid_phase_table_recovers_quadratures_lost_by_group_sum() -> None:
    time = np.linspace(0.0, 8.0, 256, dtype=float)
    cosine = np.cos(2 * np.pi * 0.5 * time)
    sine = np.sin(2 * np.pi * 0.5 * time)
    detector_signals = np.vstack([cosine, sine, -cosine, -sine])
    grouped_trace = detector_signals.sum(axis=0)

    real, imag = _mantid_phase_quad(
        detector_signals,
        asymmetries=np.ones(4, dtype=float),
        phases_radians=np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]),
    )

    assert np.max(np.abs(grouped_trace)) < 1e-12
    np.testing.assert_allclose(real, cosine, atol=1e-12)
    np.testing.assert_allclose(imag, sine, atol=1e-12)