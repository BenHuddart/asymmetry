"""Time-domain fit-and-subtract of a diamagnetic precession signal.

In a transverse field the unshifted diamagnetic muon line is often the dominant
feature and can swamp weaker shifted or radical lines.  WiMDA removes it by
fitting a damped cosine to the time-domain signal *before* the FFT and
subtracting it (``Plot.pas``).  The fitted frequency, converted back to field,
also provides an independent read of the applied field.

The model is a single damped cosine on a constant offset,

.. math::

    s(t) = A\\,\\cos\\!\\big(2\\pi(\\nu t + \\phi)\\big)\\,e^{-\\lambda t} + c,

with :math:`\\nu` in MHz, :math:`t` in µs, :math:`\\lambda` in µs⁻¹.  The fitted
field is :math:`B = 2\\pi\\nu/\\gamma_\\mu`.  The fit seeds its frequency from the
run's applied field, so it locks onto the diamagnetic line rather than a shifted
satellite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fourier.units import gauss_to_mhz, mhz_to_gauss


@dataclass
class DiamagneticFit:
    """Result of a diamagnetic damped-cosine fit."""

    amplitude: float
    frequency_mhz: float
    phase: float
    damping_per_us: float
    offset: float
    field_gauss: float
    success: bool

    def model(self, time_us: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the fitted damped cosine over *time_us*."""
        t = np.asarray(time_us, dtype=np.float64)
        return (
            self.amplitude
            * np.cos(2.0 * np.pi * (self.frequency_mhz * t + self.phase))
            * np.exp(-self.damping_per_us * t)
            + self.offset
        )


def _model(
    t: NDArray[np.float64],
    amplitude: float,
    frequency_mhz: float,
    phase: float,
    damping: float,
    offset: float,
) -> NDArray[np.float64]:
    return (
        amplitude * np.cos(2.0 * np.pi * (frequency_mhz * t + phase)) * np.exp(-damping * t)
        + offset
    )


def fit_diamagnetic(
    time_us: NDArray[np.float64],
    signal: NDArray[np.float64],
    *,
    seed_frequency_mhz: float,
    error: NDArray[np.float64] | None = None,
) -> DiamagneticFit:
    """Fit a damped cosine to *signal*, seeded at *seed_frequency_mhz*.

    Returns a :class:`DiamagneticFit`; ``success`` is false when the fit does not
    converge, in which case the seed values are returned unchanged.
    """
    t = np.asarray(time_us, dtype=np.float64)
    y_raw = np.asarray(signal, dtype=np.float64)
    # Normalise so the fit is well conditioned on grouped counts (amplitude ~1),
    # which leaves the frequency — and hence the reported field — unchanged.
    scale = float(np.max(np.abs(y_raw))) if y_raw.size else 1.0
    scale = scale if scale > 0.0 else 1.0
    y = y_raw / scale
    seed_offset = float(np.mean(y)) if y.size else 0.0
    seed_amp = float(np.std(y) * np.sqrt(2.0)) if y.size else 1.0
    seed = [seed_amp or 1.0, float(seed_frequency_mhz), 0.0, 0.1, seed_offset]

    fallback = DiamagneticFit(
        amplitude=seed[0] * scale,
        frequency_mhz=seed[1],
        phase=seed[2],
        damping_per_us=seed[3],
        offset=seed[4] * scale,
        field_gauss=float(mhz_to_gauss(seed_frequency_mhz)),
        success=False,
    )
    if t.size < 6:
        return fallback

    try:
        from scipy.optimize import curve_fit  # noqa: PLC0415

        sigma = None
        if error is not None:
            err = np.asarray(error, dtype=np.float64) / scale
            if err.shape == y.shape and np.all(np.isfinite(err)) and np.all(err > 0.0):
                sigma = err
        # Bound the frequency to a window around the applied-field seed so the
        # fit cannot run away to a high-frequency alias.
        freq_hi = max(5.0 * float(seed_frequency_mhz), float(seed_frequency_mhz) + 1.0)
        bounds = (
            [0.0, 0.0, -1.0, 0.0, -np.inf],
            [np.inf, freq_hi, 1.0, np.inf, np.inf],
        )
        popt, _ = curve_fit(_model, t, y, p0=seed, sigma=sigma, bounds=bounds, maxfev=10000)
    except Exception:
        return fallback

    amplitude, frequency_mhz, phase, damping, offset = (float(v) for v in popt)
    amplitude *= scale
    offset *= scale
    return DiamagneticFit(
        amplitude=amplitude,
        frequency_mhz=frequency_mhz,
        phase=phase,
        damping_per_us=damping,
        offset=offset,
        field_gauss=float(mhz_to_gauss(frequency_mhz)),
        success=True,
    )


def fit_and_subtract_diamagnetic(
    dataset: MuonDataset,
    *,
    seed_field_gauss: float,
) -> tuple[MuonDataset, DiamagneticFit]:
    """Fit and subtract the diamagnetic line from *dataset*'s time signal.

    Returns the cleaned dataset (the fitted cosine removed, the offset kept) and
    the :class:`DiamagneticFit`.  On a failed fit the dataset is returned
    unchanged.
    """
    seed_freq = float(gauss_to_mhz(seed_field_gauss))
    fit = fit_diamagnetic(
        np.asarray(dataset.time, dtype=np.float64),
        np.asarray(dataset.asymmetry, dtype=np.float64),
        seed_frequency_mhz=seed_freq,
        error=np.asarray(dataset.error, dtype=np.float64) if dataset.error is not None else None,
    )
    if not fit.success:
        return dataset, fit

    # Subtract the oscillatory part only, keeping the fitted constant offset so
    # the spectrum's baseline is unchanged.
    t = np.asarray(dataset.time, dtype=np.float64)
    oscillatory = fit.model(t) - fit.offset
    cleaned_signal = np.asarray(dataset.asymmetry, dtype=np.float64) - oscillatory
    cleaned = MuonDataset(
        time=t,
        asymmetry=cleaned_signal,
        error=dataset.error,
        metadata=dict(dataset.metadata) if isinstance(dataset.metadata, dict) else {},
        run=getattr(dataset, "run", None),
    )
    return cleaned, fit
