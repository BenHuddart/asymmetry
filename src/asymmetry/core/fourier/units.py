"""Frequency ↔ field unit conversions for spectral displays.

A muon precession line at frequency ν corresponds to a local field B through the
Larmor relation ν = γ_μ B / 2π, with γ_μ/2π = 135.538817 MHz/T, so a MaxEnt or
FFT spectrum can be displayed against frequency (MHz) or field (Gauss / Tesla)
interchangeably (Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy*,
OUP 2022, §15.5).

These are pure scalar/array converters built on the CODATA constants in
:mod:`asymmetry.core.utils.constants`.  The gyromagnetic ratio is a parameter
defaulting to the muon value, so other probes (¹⁹F, ¹H) can pass their own.

This is the shared units helper for the frequency-domain display work; the
``frequency-domain-finishers`` project reuses this API verbatim.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)


class FieldUnit(str, Enum):
    """A spectrum display axis unit."""

    MHZ = "mhz"
    GAUSS = "gauss"
    TESLA = "tesla"

    @classmethod
    def coerce(cls, value: object, default: FieldUnit | None = None) -> FieldUnit:
        """Return *value* as a :class:`FieldUnit`, falling back to *default*."""
        fallback = cls.MHZ if default is None else default
        if isinstance(value, FieldUnit):
            return value
        if isinstance(value, str) and value.strip().lower() in {u.value for u in cls}:
            return cls(value.strip().lower())
        return fallback


def _ratio_mhz_per_gauss(gyromagnetic_ratio_mhz_per_t: float) -> float:
    return float(gyromagnetic_ratio_mhz_per_t) * GAUSS_TO_TESLA


def mhz_to_gauss(
    mhz: ArrayLike,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert a frequency (MHz) to a field (Gauss) via ν = γ B / 2π."""
    return np.asarray(mhz, dtype=float) / _ratio_mhz_per_gauss(gyromagnetic_ratio_mhz_per_t)


def gauss_to_mhz(
    gauss: ArrayLike,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert a field (Gauss) to a frequency (MHz)."""
    return np.asarray(gauss, dtype=float) * _ratio_mhz_per_gauss(gyromagnetic_ratio_mhz_per_t)


def mhz_to_tesla(
    mhz: ArrayLike,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert a frequency (MHz) to a field (Tesla)."""
    return np.asarray(mhz, dtype=float) / float(gyromagnetic_ratio_mhz_per_t)


def tesla_to_mhz(
    tesla: ArrayLike,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert a field (Tesla) to a frequency (MHz)."""
    return np.asarray(tesla, dtype=float) * float(gyromagnetic_ratio_mhz_per_t)


def gauss_to_tesla(gauss: ArrayLike) -> NDArray[np.float64]:
    """Convert Gauss to Tesla."""
    return np.asarray(gauss, dtype=float) * GAUSS_TO_TESLA


def tesla_to_gauss(tesla: ArrayLike) -> NDArray[np.float64]:
    """Convert Tesla to Gauss."""
    return np.asarray(tesla, dtype=float) / GAUSS_TO_TESLA


def convert(
    value: ArrayLike,
    frm: FieldUnit | str,
    to: FieldUnit | str,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert *value* from unit *frm* to unit *to*, pivoting through MHz."""
    source = FieldUnit.coerce(frm)
    target = FieldUnit.coerce(to)
    to_mhz = {
        FieldUnit.MHZ: lambda v: np.asarray(v, dtype=float),
        FieldUnit.GAUSS: lambda v: gauss_to_mhz(
            v, gyromagnetic_ratio_mhz_per_t=gyromagnetic_ratio_mhz_per_t
        ),
        FieldUnit.TESLA: lambda v: tesla_to_mhz(
            v, gyromagnetic_ratio_mhz_per_t=gyromagnetic_ratio_mhz_per_t
        ),
    }
    from_mhz = {
        FieldUnit.MHZ: lambda v: np.asarray(v, dtype=float),
        FieldUnit.GAUSS: lambda v: mhz_to_gauss(
            v, gyromagnetic_ratio_mhz_per_t=gyromagnetic_ratio_mhz_per_t
        ),
        FieldUnit.TESLA: lambda v: mhz_to_tesla(
            v, gyromagnetic_ratio_mhz_per_t=gyromagnetic_ratio_mhz_per_t
        ),
    }
    return from_mhz[target](to_mhz[source](value))


def axis_label(unit: FieldUnit | str) -> str:
    """Return a plot-axis label for the given display unit."""
    return {
        FieldUnit.MHZ: "Frequency (MHz)",
        FieldUnit.GAUSS: "Field (G)",
        FieldUnit.TESLA: "Field (T)",
    }[FieldUnit.coerce(unit)]


def frequency_resolution_mhz(bin_width_us: float, n_spectrum_points: int) -> float:
    """Return the spectrum frequency resolution 1/(2·Δt·N) in MHz.

    *bin_width_us* is the (rebinned) time-channel width and *n_spectrum_points*
    the spectrum length, matching WiMDA's ``fres = 1/(2·tres·bunch·nptsME)``.
    """
    denom = 2.0 * float(bin_width_us) * int(n_spectrum_points)
    return float("inf") if denom <= 0.0 else 1.0 / denom
