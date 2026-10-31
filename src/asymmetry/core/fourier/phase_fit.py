"""Fit a sampled real common-phase signal with nonnegative spectral amplitudes.

The complex response retains both real-signal sidebands and the known mean
operator. Complex fitting is motivated by Isaac et al.,
https://doi.org/10.1016/j.jmr.2025.107991; amplitude elimination follows the variable
projection principle of Golub and Pereyra, https://doi.org/10.1088/0266-5611/19/2/201.
The uncertainty is conditional on the local active dictionary and excludes
model error, changing active sets, and timing/background systematics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar, nnls

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
CancelCallback = Callable[[], bool] | None
PHASE_FIT_METHOD = "sampled_common_phase_v1"
_UNCERTAINTY_SCOPE = (
    "Conditional local active-set covariance with correlated FFT noise from independent sample errors; "
    "excludes model misspecification, active-set changes and timing/background systematics"
)


@dataclass(frozen=True)
class PhaseAcquisition:
    """Known sampling and preprocessing, with prepared samples before windowing."""

    time_start_us: float
    sample_interval_us: float
    window_weights: FloatArray
    sample_errors: FloatArray
    mean_weights: FloatArray | None
    fft_scale: float = 1.0
    unwindowed_signal: FloatArray = field(kw_only=True)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_start_us):
            raise ValueError("The acquisition start must be finite")
        if not np.isfinite(self.sample_interval_us) or self.sample_interval_us <= 0:
            raise ValueError("The sample interval must be finite and positive")
        if not np.isfinite(self.fft_scale) or self.fft_scale <= 0:
            raise ValueError("The FFT scale must be finite and positive")
        for name, source in (
            ("window_weights", self.window_weights),
            ("sample_errors", self.sample_errors),
            ("unwindowed_signal", self.unwindowed_signal),
            ("mean_weights", self.mean_weights),
        ):
            if source is None and name == "mean_weights":
                continue
            values = np.array(source, dtype=float, copy=True)
            if values.ndim != 1 or len(values) < 12 or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain at least 12 finite samples")
            if name != "unwindowed_signal" and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative")
            if name in ("window_weights", "mean_weights") and not np.any(values > 0):
                raise ValueError(f"{name} must have positive sum")
            if name == "mean_weights":
                values /= values.max()
                values /= values.sum()
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        n = len(self.window_weights)
        if len(self.sample_errors) != n or len(self.unwindowed_signal) != n:
            raise ValueError("Acquisition arrays must have equal lengths")
        if self.mean_weights is not None and len(self.mean_weights) != n:
            raise ValueError("Acquisition arrays must have equal lengths")


@dataclass(frozen=True)
class PhaseFitConfig:
    """Frequency band and maximum tolerated phase sensitivity."""

    minimum_frequency_mhz: float | None = None
    maximum_frequency_mhz: float | None = None
    tolerance_degrees: float = 5.0

    def __post_init__(self) -> None:
        for value in (self.minimum_frequency_mhz, self.maximum_frequency_mhz):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Frequency bounds must be finite and nonnegative")
        if (
            self.minimum_frequency_mhz is not None
            and self.maximum_frequency_mhz is not None
            and self.minimum_frequency_mhz >= self.maximum_frequency_mhz
        ):
            raise ValueError("The lower frequency bound must be below the upper bound")
        if not np.isfinite(self.tolerance_degrees) or not 0 < self.tolerance_degrees <= 45:
            raise ValueError("The phase tolerance must be positive and at most 45 degrees")


@dataclass(frozen=True)
class PhaseEstimate:
    """Accepted physical cosine phase at absolute time zero."""

    phase_radians: float
    standard_error_radians: float
    band_mhz: tuple[float, float]
    maximum_change_radians: float
    signal_to_noise: float
    residual_ratio: float
    grid_size: int
    grid_density: int
    observation_band_mhz: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": PHASE_FIT_METHOD,
            "status": "accepted",
            "phase_radians": self.phase_radians,
            "standard_error_radians": self.standard_error_radians,
            "band_mhz": list(self.band_mhz),
            "observation_band_mhz": list(self.observation_band_mhz),
            "maximum_change_radians": self.maximum_change_radians,
            "signal_to_noise": self.signal_to_noise,
            "signal_to_noise_kind": "conditional_total_positive_amplitude_with_phase_nuisance",
            "residual_ratio": self.residual_ratio,
            "grid_size": self.grid_size,
            "grid_density": self.grid_density,
            "uncertainty_scope": _UNCERTAINTY_SCOPE,
        }


@dataclass(frozen=True)
class PhaseRejection:
    """An unsuitable measurement has no phase value."""

    reason: str
    message: str
    band_mhz: tuple[float, float] | None = None
    observation_band_mhz: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": PHASE_FIT_METHOD,
            "status": "rejected",
            "reason": self.reason,
            "message": self.message,
            "band_mhz": None if self.band_mhz is None else list(self.band_mhz),
            "observation_band_mhz": None
            if self.observation_band_mhz is None
            else list(self.observation_band_mhz),
        }


class PhaseFitCancelled(Exception):  # noqa: N818
    """The caller cancelled a phase fit cooperatively."""


class PhaseFitRejectedError(ValueError):
    """Expose a typed rejection at front-end boundaries requiring an exception."""

    def __init__(self, report: PhaseRejection, message: str | None = None) -> None:
        super().__init__(report.message if message is None else message)
        self.report = report


def _check_cancel(callback: CancelCallback) -> None:
    if callback is not None and callback():
        raise PhaseFitCancelled("Phase estimation cancelled")


@dataclass(frozen=True)
class _PhaseSolution:
    phase: float
    amplitudes: FloatArray
    residual: FloatArray
    columns: FloatArray
    derivative: FloatArray
    centers: FloatArray
    bins: NDArray[np.int64]
    competing_ratio: float


def _observation_bins(n: int, duration: float, band: tuple[float, float]) -> NDArray[np.int64]:
    """Constrain finite-window leakage beyond the positive component support."""
    guard = max(8, int(np.ceil((band[1] - band[0]) * duration * 3)))
    return np.arange(
        max(1, int(np.ceil(band[0] * duration)) - guard),
        min(n // 2, int(np.floor(band[1] * duration)) + guard) + 1,
    )


def _sampled_response(
    acquisition: PhaseAcquisition,
    window: FloatArray,
    band: tuple[float, float],
    density: int,
    callback: CancelCallback = None,
) -> tuple[ComplexArray, ComplexArray, FloatArray, NDArray[np.int64]]:
    n = len(window)
    duration = n * acquisition.sample_interval_us
    bins = _observation_bins(n, duration, band)
    lattice = np.arange(
        max(1, int(np.ceil(band[0] * duration * density))),
        int(np.floor(band[1] * duration * density)) + 1,
    )
    centers = lattice / (density * duration)
    _check_cancel(callback)
    transform = np.fft.fft(window, n=density * n)
    origin = np.exp(2j * np.pi * centers * acquisition.time_start_us)
    mean = np.zeros(len(centers), dtype=complex)
    if acquisition.mean_weights is not None:
        _check_cancel(callback)
        mean = origin * np.fft.fft(acquisition.mean_weights, n=density * n)[-lattice]
    carrier = transform[density * bins]
    positive = origin * transform[(density * bins[:, None] - lattice) % (density * n)]
    negative = origin.conj() * transform[(density * bins[:, None] + lattice) % (density * n)]
    return (
        0.5 * (positive - carrier[:, None] * mean),
        0.5 * (negative - carrier[:, None] * mean.conj()),
        centers,
        bins,
    )


def _phase_search(
    data: ComplexArray,
    acquisition: PhaseAcquisition,
    window: FloatArray,
    band: tuple[float, float],
    density: int,
    callback: CancelCallback,
) -> _PhaseSolution:
    _check_cancel(callback)
    positive, negative, centers, bins = _sampled_response(
        acquisition, window, band, density, callback
    )
    observed = np.concatenate((data[bins].real, data[bins].imag))

    def solve(phase: float) -> tuple[float, FloatArray, FloatArray, FloatArray, FloatArray]:
        _check_cancel(callback)
        rotation = np.exp(1j * phase)
        response = rotation * positive + rotation.conjugate() * negative
        derivative = 1j * (rotation * positive - rotation.conjugate() * negative)
        columns = np.concatenate((response.real, response.imag))
        amplitudes, norm = nnls(columns, observed, maxiter=max(1000, 10 * len(centers)))
        return (
            norm * norm,
            amplitudes,
            observed - columns @ amplitudes,
            columns,
            np.concatenate((derivative.real, derivative.imag)) @ amplitudes,
        )

    phases = np.linspace(-np.pi, np.pi, 48, endpoint=False)
    objectives = np.array([solve(float(p))[0] for p in phases])
    minima = [
        i
        for i in range(len(phases))
        if objectives[i] <= objectives[i - 1] and objectives[i] <= objectives[(i + 1) % len(phases)]
    ]
    step = 2 * np.pi / len(phases)
    refined = [
        minimize_scalar(
            lambda p: solve(float(p))[0],
            bounds=(phases[i] - step, phases[i] + step),
            method="bounded",
            options={"xatol": 1e-9},
        )
        for i in minima
    ]
    refined.sort(key=lambda result: result.fun)
    best = refined[0]
    _, amplitudes, residual, columns, derivative = solve(float(best.x))
    competing = float("inf") if len(refined) == 1 else float(refined[1].fun / max(best.fun, 1e-30))
    return _PhaseSolution(
        float(np.angle(np.exp(1j * best.x))),
        amplitudes,
        residual,
        columns,
        derivative,
        centers,
        bins,
        competing,
    )


def _noise_power(
    acquisition: PhaseAcquisition,
    window: FloatArray,
    bins: NDArray[np.int64],
    callback: CancelCallback = None,
    normalization: float = 1.0,
) -> float:
    _check_cancel(callback)
    variance = (acquisition.sample_errors / normalization * acquisition.fft_scale) ** 2
    power = np.full(len(bins), np.sum(window**2 * variance))
    if acquisition.mean_weights is not None:
        mean = acquisition.mean_weights
        transform = np.fft.rfft(window)[bins]
        power += np.abs(transform) ** 2 * np.sum(mean**2 * variance)
        _check_cancel(callback)
        power -= 2 * (transform.conj() * np.fft.rfft(window * mean * variance)[bins]).real
    return float(np.maximum(power, 0).sum())


def _sample_noise_standard_error(
    gradient: FloatArray,
    bins: NDArray[np.int64],
    acquisition: PhaseAcquisition,
    normalization: float,
    callback: CancelCallback,
) -> float:
    """Apply the transpose of the complete window/mean Fourier noise operator."""
    _check_cancel(callback)
    n = len(acquisition.window_weights)
    spectrum = np.zeros(n, dtype=complex)
    count = len(bins)
    spectrum[bins] = gradient[:count] + 1j * gradient[count:]
    _check_cancel(callback)
    influence = n * np.fft.ifft(spectrum).real * acquisition.window_weights
    if acquisition.mean_weights is not None:
        influence -= acquisition.mean_weights * influence.sum()
    normalized_errors = acquisition.sample_errors / normalization * acquisition.fft_scale
    return float(np.linalg.norm(normalized_errors * influence))


def _conditional_standard_error(
    solution: _PhaseSolution,
    acquisition: PhaseAcquisition,
    normalization: float,
    callback: CancelCallback = None,
) -> float:
    _check_cancel(callback)
    active = solution.amplitudes > max(solution.amplitudes.max() * 1e-7, 1e-15)
    columns = solution.columns[:, active]
    projected = (
        solution.derivative
        - columns @ np.linalg.lstsq(columns, solution.derivative, rcond=1e-10)[0]
    )
    curvature = float(projected @ projected)
    if curvature <= 1e-20:
        return float("inf")
    return (
        _sample_noise_standard_error(projected, solution.bins, acquisition, normalization, callback)
        / curvature
    )


def _conditional_amplitude_standard_error(
    solution: _PhaseSolution,
    acquisition: PhaseAcquisition,
    normalization: float,
    callback: CancelCallback = None,
) -> float:
    """Propagate noise for total positive amplitude, with phase as a nuisance.

    The functional must lie in the retained row space of the joint Jacobian;
    a pseudoinverse alone would give false precision for unidentified amplitude.
    Column scaling prevents amplitude units from deciding the numerical rank.
    """
    _check_cancel(callback)
    active = solution.amplitudes > max(solution.amplitudes.max() * 1e-7, 1e-15)
    jacobian = np.column_stack((solution.columns[:, active], solution.derivative))
    norms = np.linalg.norm(jacobian, axis=0)
    if not np.all(norms > 0):
        return float("inf")
    left, singular, right = np.linalg.svd(jacobian / norms, full_matrices=False)
    retained = singular > singular[0] * 1e-10
    functional = np.concatenate((np.ones(int(active.sum())), [0.0])) / norms
    coordinates = right[retained] @ functional
    if np.linalg.norm(functional - right[retained].T @ coordinates) > 1e-8 * np.linalg.norm(
        functional
    ):
        return float("inf")
    gradient = left[:, retained] @ (coordinates / singular[retained])
    return _sample_noise_standard_error(
        gradient, solution.bins, acquisition, normalization, callback
    )


def fit_fft_phase(
    frequency_mhz: FloatArray,
    spectrum: ComplexArray,
    acquisition: PhaseAcquisition,
    config: PhaseFitConfig = PhaseFitConfig(),
    *,
    cancel_callback: CancelCallback = None,
) -> PhaseEstimate | PhaseRejection:
    """Estimate phase using native bins and explicit acquisition-response fits.

    Suitability screens use conservative engineering thresholds, not calibrated
    confidence levels: conditional total-positive-amplitude SNR >= 8,
    residual/noise <= 5, edge amplitude <= 5%, and
    phase sensitivity and conditional error within the requested tolerance.
    SNR divides total fitted positive amplitude by its correlated-noise standard
    error from the joint active-amplitude/phase Jacobian; it does not divide by
    accumulated noise energy in otherwise empty observation guards. This local
    uncertainty excludes active-set selection and is not a detection confidence.
    The support band uses at most 257 base lattice centers; three support-band
    widths of leakage guard on either side bound the native fitting grid to 598
    bins. Refinement is capped at 1025 centers and density 24.
    """
    _check_cancel(cancel_callback)
    frequency = np.asarray(frequency_mhz, dtype=float)
    supplied = np.asarray(spectrum, dtype=complex)
    n = len(acquisition.window_weights)
    if frequency.ndim != 1 or supplied.shape != frequency.shape or len(frequency) < 2:
        raise ValueError("Spectrum and frequency must be matching one-dimensional rFFT arrays")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(supplied)):
        raise ValueError("Spectrum and frequencies must be finite")
    padded = (
        int(round(1 / ((frequency[1] - frequency[0]) * acquisition.sample_interval_us)))
        if frequency[1] > frequency[0]
        else 0
    )
    if padded < n or padded % n or len(frequency) != padded // 2 + 1:
        raise ValueError("The spectrum must use an integer-padded rFFT grid for the acquisition")
    if not np.allclose(
        frequency, np.fft.rfftfreq(padded, acquisition.sample_interval_us), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Frequency values do not match the acquisition rFFT grid")
    native = supplied[:: padded // n]
    _check_cancel(cancel_callback)
    expected = (
        np.fft.rfft(acquisition.unwindowed_signal * acquisition.window_weights)
        * acquisition.fft_scale
    )
    if not np.allclose(
        native, expected, rtol=1e-8, atol=max(float(np.max(np.abs(expected))) * 1e-10, 1e-300)
    ):
        raise ValueError(
            "The supplied spectrum must match the prepared unwindowed samples and window"
        )
    duration = n * acquisition.sample_interval_us
    resolution = 1 / duration
    nyquist = 0.5 / acquisition.sample_interval_us
    requested_low = 0.0 if config.minimum_frequency_mhz is None else config.minimum_frequency_mhz
    requested_high = (
        nyquist if config.maximum_frequency_mhz is None else config.maximum_frequency_mhz
    )
    if requested_low >= nyquist or requested_high > nyquist:
        return PhaseRejection(
            "invalid_band",
            "Choose a nonempty band within the positive Nyquist range",
            (float(requested_low), float(requested_high)),
        )
    native_frequency = np.fft.rfftfreq(n, acquisition.sample_interval_us)
    eligible = np.flatnonzero(
        (native_frequency >= max(3 * resolution, requested_low))
        & (native_frequency <= requested_high)
    )
    if not len(eligible) or not np.any(np.abs(native[eligible]) > 0):
        return PhaseRejection("no_oscillation", "No resolved oscillatory signal is available")
    peak = int(eligible[np.argmax(np.abs(native[eligible]))])
    height = abs(native[peak])
    left = right = peak
    while left > 1 and abs(native[left - 1]) > height / 2:
        left -= 1
    while right < len(native) - 1 and abs(native[right + 1]) > height / 2:
        right += 1
    radius = min(max(8, 3 * (right - left + 1)), 40) * resolution
    low = (
        max(resolution, peak * resolution - radius)
        if config.minimum_frequency_mhz is None
        else config.minimum_frequency_mhz
    )
    high = (
        min(nyquist, peak * resolution + radius)
        if config.maximum_frequency_mhz is None
        else config.maximum_frequency_mhz
    )
    band = (float(low), float(high))
    if low >= high or low >= nyquist or high > nyquist:
        return PhaseRejection(
            "invalid_band", "Choose a nonempty band within the positive Nyquist range", band
        )
    if (high - low) * duration * 3 > 256:
        return PhaseRejection(
            "band_too_wide", "Narrow the frequency band to at most 257 lattice centers", band
        )
    if (high - low) * duration < 4:
        return PhaseRejection(
            "insufficient_band", "Include at least four native frequency intervals", band
        )
    observation_bins = _observation_bins(n, duration, band)
    observation_band = (
        float(observation_bins[0] / duration),
        float(observation_bins[-1] / duration),
    )
    normalization = float(np.max(np.abs(native)))
    observed = native / normalization
    base = _phase_search(
        observed, acquisition, acquisition.window_weights, band, 3, cancel_callback
    )
    noise = max(
        np.sqrt(
            _noise_power(
                acquisition, acquisition.window_weights, base.bins, cancel_callback, normalization
            )
        ),
        np.finfo(float).eps * np.sqrt(2 * len(base.bins)),
    )
    dense = _phase_search(
        observed, acquisition, acquisition.window_weights, band, 6, cancel_callback
    )
    solutions = [base, dense]
    solution, density = dense, 6
    ratio = float(np.linalg.norm(solution.residual) / noise)
    for candidate_density in (12, 24):
        if ratio <= 5:
            break
        count = (
            int(np.floor(high * duration * candidate_density))
            - max(1, int(np.ceil(low * duration * candidate_density)))
            + 1
        )
        if count > 1025:
            break
        solution = _phase_search(
            observed,
            acquisition,
            acquisition.window_weights,
            band,
            candidate_density,
            cancel_callback,
        )
        density = candidate_density
        solutions.append(solution)
        ratio = float(np.linalg.norm(solution.residual) / noise)
    if ratio > 5:
        return PhaseRejection(
            "incompatible_or_resolution_limited",
            "The positive common-phase model remains incompatible with the sample noise at the bounded lattice resolution; narrow the band or inspect the signal model",
            band,
            observation_band,
        )
    if solution.competing_ratio < 1.2:
        return PhaseRejection(
            "numerical_resolution_limited" if density > 6 else "ambiguous_phase",
            "Competing common-phase solutions are not sufficiently separated",
            band,
            observation_band,
        )
    amplitude_sum = solution.amplitudes.sum()
    amplitude_error = _conditional_amplitude_standard_error(
        solution, acquisition, normalization, cancel_callback
    )
    if not np.isfinite(amplitude_error):
        return PhaseRejection(
            "unidentifiable_amplitude",
            "The total positive amplitude is not identifiable with phase as a nuisance",
            band,
            observation_band,
        )
    snr = float(amplitude_sum / max(amplitude_error, np.finfo(float).eps * amplitude_sum))
    if snr < 8:
        return PhaseRejection(
            "low_signal_to_noise",
            "The total positive amplitude is too weak relative to its conditional sample-noise uncertainty",
            band,
            observation_band,
        )
    edge = (solution.centers < low + resolution) | (solution.centers > high - resolution)
    if solution.amplitudes[edge].sum() > 0.05 * amplitude_sum:
        return PhaseRejection(
            "edge_supported",
            "Widen the band: substantial fitted signal lies at its edges",
            band,
            observation_band,
        )
    if np.sum(solution.amplitudes[solution.centers <= 3 * resolution]) > 0.05 * amplitude_sum:
        return PhaseRejection(
            "unresolved_oscillation",
            "The fitted signal must resolve more than three cycles",
            band,
            observation_band,
        )
    standard_error = _conditional_standard_error(
        solution, acquisition, normalization, cancel_callback
    )
    tolerance = np.deg2rad(config.tolerance_degrees)
    if not np.isfinite(standard_error) or standard_error > tolerance:
        return PhaseRejection(
            "numerical_resolution_limited"
            if density > 6 and not np.isfinite(standard_error)
            else "ill_conditioned",
            "The conditional phase uncertainty exceeds the requested tolerance",
            band,
            observation_band,
        )
    _check_cancel(cancel_callback)
    center = float(np.sum(solution.centers * solution.amplitudes) / amplitude_sum)
    narrow = (max(low, center - (high - low) / 3), min(high, center + (high - low) / 3))
    narrowed = _phase_search(
        observed, acquisition, acquisition.window_weights, narrow, density, cancel_callback
    )
    _check_cancel(cancel_callback)
    window = (
        np.ones(n)
        if np.ptp(acquisition.window_weights) > 1e-10 * np.max(acquisition.window_weights)
        else np.exp(-np.arange(n) / n)
    )
    alternative_data = (
        np.fft.rfft(acquisition.unwindowed_signal * window) * acquisition.fft_scale / normalization
    )
    alternative = _phase_search(
        alternative_data, acquisition, window, band, density, cancel_callback
    )
    solutions.extend((narrowed, alternative))
    maximum_change = max(
        abs(float(np.angle(np.exp(1j * (first.phase - second.phase)))))
        for first in solutions
        for second in solutions
    )
    if maximum_change > tolerance:
        return PhaseRejection(
            "unstable_phase",
            "Phase changes excessively with lattice density, band or window",
            band,
            observation_band,
        )
    return PhaseEstimate(
        solution.phase,
        standard_error,
        band,
        maximum_change,
        snr,
        ratio,
        len(solution.centers),
        density,
        observation_band,
    )
