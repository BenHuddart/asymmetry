"""Residual statistics and separately attributed fixed-grid spectral diagnostics."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.likelihood import dataset_observations
from asymmetry.core.fitting.residual_statistics import (
    max_abs_autocorrelation as _max_abs_autocorrelation,
)
from asymmetry.core.fitting.residual_statistics import (
    runs_test_z_score as _runs_test_z_score,
)
from asymmetry.core.fitting.results import FitResult
from asymmetry.core.fourier.apodisation import ApodisationEarlySignalWarning
from asymmetry.core.fourier.diagnostic import prepare_diagnostic_observation
from asymmetry.core.fourier.fft import fft_arrays
from asymmetry.core.fourier.sampling import is_uniform_sampling

_EPS = 1e-12
_DEFAULT_FFT_WINDOW = "hann"
_DEFAULT_FFT_PADDING = 4


def _dominant_peak_metrics(
    frequencies: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> tuple[float, float]:
    if frequencies.size < 2 or magnitude.size < 2:
        return 0.0, 0.0

    freqs = np.asarray(frequencies, dtype=float)
    mags = np.asarray(magnitude, dtype=float)
    positive = np.flatnonzero(freqs > 0.0)
    if positive.size == 0:
        return 0.0, 0.0

    positive_mags = mags[positive]
    peak_idx_local = int(np.argmax(positive_mags))
    peak_idx = int(positive[peak_idx_local])
    peak_mag = float(mags[peak_idx])

    noise = np.delete(positive_mags, peak_idx_local)
    noise_floor = float(np.median(noise)) if noise.size else 0.0
    snr = peak_mag / max(noise_floor, _EPS)
    return float(freqs[peak_idx]), float(snr)


def _residual_diagnostics(
    dataset: MuonDataset,
    fit_result: FitResult,
    model: Callable[..., NDArray[np.float64]] | None = None,
    *,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[float, float, float, float]:
    if cancel_check is not None:
        cancel_check()
    if fit_result.residuals is None or len(fit_result.residuals) == 0:
        return float("inf"), 0.0, 0.0, float("inf")

    residuals = np.asarray(fit_result.residuals, dtype=float)
    observations = dataset_observations(dataset)
    safe_errors = observations.sigma
    if residuals.size != safe_errors.size:
        raise ValueError("Residuals must describe the selected Gaussian observations.")
    standardized = residuals / safe_errors
    residual_rms = (
        float(np.sqrt(np.mean(np.square(standardized)))) if standardized.size else float("inf")
    )

    runs_z_score = _runs_test_z_score(standardized)
    if observations.report is not None and observations.report.omitted:
        message = "Residual autocorrelation and FFT unavailable: Gaussian selection omits acquisition bins."
        if message not in fit_result.warnings:
            fit_result.warnings.append(message)
        return residual_rms, runs_z_score, float("nan"), float("nan")
    uniform_original = is_uniform_sampling(observations.time)
    max_abs_autocorrelation = (
        _max_abs_autocorrelation(standardized) if uniform_original else float("nan")
    )
    if uniform_original and (
        residual_rms <= 0.1 or float(np.max(np.abs(standardized), initial=0.0)) <= 0.25
    ):
        return residual_rms, runs_z_score, max_abs_autocorrelation, 0.0

    if uniform_original:
        time, spectral_residuals, errors = observations.time, residuals, safe_errors
        view = prepare_diagnostic_observation(dataset, cancel_check=cancel_check)
    elif model is None:
        fit_result.warnings.append(
            "Residual FFT unavailable: an evaluable fitted model is required on the fixed diagnostic grid."
        )
        return residual_rms, runs_z_score, max_abs_autocorrelation, float("nan")
    else:
        view = prepare_diagnostic_observation(dataset, cancel_check=cancel_check)
        time, errors = view.time, view.error
        predicted = np.asarray(
            model(time, **{p.name: p.value for p in fit_result.parameters}), dtype=float
        )
        spectral_residuals = view.asymmetry - predicted
    if cancel_check is not None:
        cancel_check()
    fit_result.spectral_residual_attribution = view.window_attribution(len(time), "hann") | {
        "observation": "original_fitted" if uniform_original else "separate_fixed_count_reduction",
        "model_parameters": {p.name: p.value for p in fit_result.parameters},
    }
    # Only the spectral view changes; RMS and runs retain the fitted observations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApodisationEarlySignalWarning)
        frequencies, _real, magnitude = fft_arrays(
            time,
            spectral_residuals,
            errors,
            window=_DEFAULT_FFT_WINDOW,
            padding_factor=_DEFAULT_FFT_PADDING,
        )
    if cancel_check is not None:
        cancel_check()
    _peak_freq, residual_fft_peak_snr = _dominant_peak_metrics(frequencies, magnitude)
    return residual_rms, runs_z_score, max_abs_autocorrelation, residual_fft_peak_snr


@dataclass(frozen=True)
class RunResidualDiagnostic:
    """Residual diagnostics for one dataset inside a global-fit candidate."""

    run_number: int
    run_label: str
    axis_value: float
    residual_rms: float
    runs_z_score: float
    max_abs_autocorrelation: float
    residual_fft_peak_snr: float
    gate_passed: bool
    gate_reasons: tuple[str, ...]

    @property
    def available(self) -> bool:
        """Whether all recorded residual diagnostics were available."""
        return all(
            np.isfinite(v)
            for v in (
                self.residual_rms,
                self.runs_z_score,
                self.max_abs_autocorrelation,
                self.residual_fft_peak_snr,
            )
        )
