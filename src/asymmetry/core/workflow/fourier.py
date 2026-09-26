"""Agent-facing Fourier transform with quantitative peak reporting.

Peaks are detected on the **whole** spectrum and then restricted to the
requested band: the detector's noise floor is a property of the spectrum, and
estimating it from a narrow zoom around a line measures the line itself as
noise, so a zoomed transform would otherwise report nothing exactly where it
was asked to look.

When no line passes the detector, the strongest local maxima in the band are
still reported, with their height over that noise floor, as *candidates* — a
weak line that a time-domain fit can confirm or refute, never a detection.

The muoniated-radical correlation spectrum (:func:`correlation_spectrum`) is
quantified the same way on its own axis, the hyperfine coupling A_µ in MHz; it
is built from the run's detector groups, not from the reduced curve.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.peak_detection import detect_peaks_in_spectrum, serialize_peak_analysis
from asymmetry.core.fourier.correlation import DEFAULT_CORR_ORDER
from asymmetry.core.fourier.fft import fft_arrays
from asymmetry.core.fourier.spectrum import GroupSpectrumConfig, compute_average_group_spectrum


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
class CorrelationSettings:
    """The Breit–Rabi pairing a correlation spectrum is built with.

    ``field_gauss`` is the transverse field the line pairs are computed at;
    ``order`` is WiMDA's ``CorrFn`` ratio-penalty order.
    """

    field_gauss: float
    order: int = DEFAULT_CORR_ORDER

    def __post_init__(self) -> None:
        if self.field_gauss <= 0.0:
            raise ValueError(
                f"The correlation spectrum pairs lines in a transverse field; got "
                f"{self.field_gauss} G."
            )
        if self.order < 1:
            raise ValueError(f"Correlation order must be at least 1, got {self.order}.")

    def to_dict(self) -> dict[str, Any]:
        return {"field_gauss": self.field_gauss, "order": self.order}


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
    #: Set for a correlation spectrum, whose ``frequency`` axis — and every
    #: peak's ``frequency_mhz`` — is the hyperfine coupling A_µ.
    correlation: CorrelationSettings | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": "frequency" if self.correlation is None else "hyperfine_coupling",
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
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
    return _quantified(frequency, real, magnitude, dataset.time, settings, source="fft")


def correlation_spectrum(
    run: Run,
    time: np.ndarray,
    settings: FourierSettings,
    correlation: CorrelationSettings,
    *,
    group_ids: list[int],
) -> FourierOutcome:
    """The muoniated-radical correlation spectrum of *run*'s detector groups, quantified.

    :func:`compute_average_group_spectrum` in its ``correlation`` display: the
    averaged amplitude spectrum of *group_ids*, on the run's own bins and under
    the corrections its grouping payload carries, mapped onto the hyperfine
    coupling axis, where a radical's line pair (ν₁₂, ν₃₄) peaks at A_µ. *time*
    is the reduced record of the same run; it sets the resolution the peak
    detector works to. ``settings.phase_degrees`` does not apply — the
    correlation pairs amplitudes.
    """
    spectrum = compute_average_group_spectrum(
        run,
        GroupSpectrumConfig(
            display="correlation",
            window=settings.window,
            padding=settings.padding_factor,
            filter_time_constant_us=settings.filter_time_constant_us,
            t_min_us=settings.t_min,
            t_max_us=settings.t_max,
            selected_group_ids=group_ids,
            correlation_reference_field_gauss=correlation.field_gauss,
            correlation_order=correlation.order,
        ),
    )
    if spectrum is None or spectrum.time.size < 2:
        raise ValueError(
            f"Run {run.run_number} gives no correlation spectrum: its groups "
            f"{', '.join(str(group) for group in group_ids)} leave no frequency axis to pair."
        )
    outcome = _quantified(
        spectrum.time, spectrum.asymmetry, spectrum.asymmetry, time, settings, source="correlation"
    )
    return replace(outcome, correlation=correlation)


def _quantified(
    frequency: np.ndarray,
    real: np.ndarray,
    magnitude: np.ndarray,
    time: np.ndarray,
    settings: FourierSettings,
    *,
    source: str,
) -> FourierOutcome:
    """A spectrum restricted to the band, with the peaks detected on the whole of it."""
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

    time = np.asarray(time, dtype=np.float64)
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
        source=source,
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


__all__ = [
    "CorrelationSettings",
    "FourierOutcome",
    "FourierSettings",
    "correlation_spectrum",
    "fourier_spectrum",
]
