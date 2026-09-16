"""Agent-facing Fourier transform with quantitative peak reporting."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FourierOutcome:
    frequency: np.ndarray
    real: np.ndarray
    magnitude: np.ndarray
    settings: FourierSettings
    resolution_mhz: float
    peaks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "resolution_mhz": self.resolution_mhz,
            "n_points": int(self.frequency.size),
            "frequency_min_mhz": float(self.frequency[0]),
            "frequency_max_mhz": float(self.frequency[-1]),
            "peak_analysis": self.peaks,
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
    mask = frequency >= settings.f_min
    if settings.f_max is not None:
        mask &= frequency <= settings.f_max
    frequency = np.asarray(frequency[mask], dtype=np.float64)
    real = np.asarray(real[mask], dtype=np.float64)
    magnitude = np.asarray(magnitude[mask], dtype=np.float64)
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
        frequency,
        magnitude,
        resolution_mhz=resolution,
        max_peaks=settings.max_peaks,
        source="fft",
        leakage_profile="hann" if settings.window == "hann" else "rect",
    )
    return FourierOutcome(
        frequency=frequency,
        real=real,
        magnitude=magnitude,
        settings=settings,
        resolution_mhz=resolution,
        peaks=serialize_peak_analysis(analysis),
    )


__all__ = ["FourierOutcome", "FourierSettings", "fourier_spectrum"]
