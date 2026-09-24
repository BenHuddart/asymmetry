"""Agent-facing Fourier transform with quantitative peak reporting.

Peaks are detected on the **whole** spectrum and then restricted to the
requested band: the detector's noise floor is a property of the spectrum, and
estimating it from a narrow zoom around a line measures the line itself as
noise, so a zoomed transform would otherwise report nothing exactly where it
was asked to look.

When no line passes the detector, the strongest local maxima in the band are
still reported, with their height over that noise floor, as *candidates* — a
weak line that a time-domain fit can confirm or refute, never a detection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.peak_detection import detect_peaks_in_spectrum, serialize_peak_analysis
from asymmetry.core.fourier.fft import fft_arrays


@dataclass(frozen=True)
class FourierSettings:
    """Choices that determine a stored frequency spectrum."""

    window: str = "none"
    padding_factor: int = 4
    t_min: float | None = None
    t_max: float | None = None
    phase_degrees: float = 0.0
    filter_time_constant_us: float = 1.5
    f_min: float = 0.0
    f_max: float | None = None
    max_peaks: int = 6

    def __post_init__(self) -> None:
        if self.padding_factor < 1:
            raise ValueError("Padding factor must be at least 1.")
        if self.t_min is not None and self.t_max is not None and self.t_min >= self.t_max:
            raise ValueError("Fourier t_min must be below t_max.")
        if self.f_max is not None and self.f_min >= self.f_max:
            raise ValueError("Fourier f_min must be below f_max.")
        if self.max_peaks < 1:
            raise ValueError("Peak count must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "padding_factor": self.padding_factor,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "phase_degrees": self.phase_degrees,
            "filter_time_constant_us": self.filter_time_constant_us,
            "f_min": self.f_min,
            "f_max": self.f_max,
            "max_peaks": self.max_peaks,
        }


#: Sub-threshold maxima reported when no line passes the detector.
_CANDIDATE_MAXIMA = 3


@dataclass(frozen=True)
class FourierOutcome:
    frequency: np.ndarray
    real: np.ndarray
    magnitude: np.ndarray
    settings: FourierSettings
    resolution_mhz: float
    peaks: dict[str, Any]
    #: ``{"frequency_mhz", "height_over_noise"}`` for the strongest maxima in
    #: the band — filled only when no peak was detected there.
    candidate_maxima: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "resolution_mhz": self.resolution_mhz,
            "n_points": int(self.frequency.size),
            "frequency_min_mhz": float(self.frequency[0]),
            "frequency_max_mhz": float(self.frequency[-1]),
            "peak_analysis": self.peaks,
            "candidate_maxima": [dict(entry) for entry in self.candidate_maxima],
        }


def fourier_spectrum(dataset: MuonDataset, settings: FourierSettings) -> FourierOutcome:
    """Transform one reduced spectrum and quantify its resolved peaks."""
    frequency, real, magnitude = fft_arrays(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        window=settings.window,
        padding_factor=settings.padding_factor,
        t_min=settings.t_min,
        t_max=settings.t_max,
        phase_degrees=settings.phase_degrees,
        filter_start_us=0.0,
        filter_time_constant_us=settings.filter_time_constant_us,
        subtract_average_signal=True,
    )
    full_frequency = np.asarray(frequency, dtype=np.float64)
    full_magnitude = np.asarray(magnitude, dtype=np.float64)
    mask = full_frequency >= settings.f_min
    if settings.f_max is not None:
        mask &= full_frequency <= settings.f_max
    frequency = full_frequency[mask]
    real = np.asarray(real[mask], dtype=np.float64)
    magnitude = full_magnitude[mask]
    if frequency.size < 2:
        raise ValueError("The requested Fourier frequency window contains fewer than two bins.")

    time = np.asarray(dataset.time, dtype=np.float64)
    time_mask = np.ones(time.size, dtype=bool)
    if settings.t_min is not None:
        time_mask &= time >= settings.t_min
    if settings.t_max is not None:
        time_mask &= time <= settings.t_max
    selected_time = time[time_mask]
    if selected_time.size < 2:
        raise ValueError("The requested Fourier time window contains fewer than two points.")
    duration = float(selected_time[-1] - selected_time[0])
    resolution = 1.0 / duration if duration > 0.0 else float(np.median(np.diff(frequency)))
    analysis = detect_peaks_in_spectrum(
        full_frequency,
        full_magnitude,
        resolution_mhz=resolution,
        max_peaks=max(settings.max_peaks, int(full_frequency.size)),
        source="fft",
        leakage_profile="hann" if settings.window == "hann" else "rect",
    )
    f_hi = np.inf if settings.f_max is None else settings.f_max
    in_band = [peak for peak in analysis.peaks if settings.f_min <= peak.frequency_mhz <= f_hi]
    analysis = replace(analysis, peaks=tuple(in_band[: settings.max_peaks]))
    return FourierOutcome(
        frequency=frequency,
        real=real,
        magnitude=magnitude,
        settings=settings,
        resolution_mhz=resolution,
        peaks=serialize_peak_analysis(analysis),
        candidate_maxima=[] if in_band else _strongest_maxima(frequency, magnitude, analysis),
    )


def _strongest_maxima(
    frequency: np.ndarray, magnitude: np.ndarray, analysis: Any
) -> list[dict[str, float]]:
    """The band's highest interior local maxima, with height over the noise floor."""
    interior = (
        np.flatnonzero((magnitude[1:-1] > magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:])) + 1
    )
    strongest = interior[np.argsort(magnitude[interior])[::-1][:_CANDIDATE_MAXIMA]]
    return [
        {
            "frequency_mhz": float(frequency[index]),
            "height_over_noise": float(magnitude[index] / analysis.noise_floor),
        }
        for index in sorted(strongest, key=lambda i: -magnitude[i])
    ]


__all__ = ["FourierOutcome", "FourierSettings", "fourier_spectrum"]
