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


def _validated_ratio(gyromagnetic_ratio_mhz_per_t: float) -> float:
    """Return a positive, finite gyromagnetic ratio or raise.

    The ratio is a denominator on the field→ frequency override path, so a
    non-positive or non-finite override (a bad probe constant) would silently
    yield ``inf``/sign-flipped fields — fail fast instead.
    """
    ratio = float(gyromagnetic_ratio_mhz_per_t)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError(
            f"gyromagnetic_ratio_mhz_per_t must be positive and finite, got {ratio!r}."
        )
    return ratio


def _ratio_mhz_per_gauss(gyromagnetic_ratio_mhz_per_t: float) -> float:
    return _validated_ratio(gyromagnetic_ratio_mhz_per_t) * GAUSS_TO_TESLA


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
    return np.asarray(mhz, dtype=float) / _validated_ratio(gyromagnetic_ratio_mhz_per_t)


def tesla_to_mhz(
    tesla: ArrayLike,
    *,
    gyromagnetic_ratio_mhz_per_t: float = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
) -> NDArray[np.float64]:
    """Convert a field (Tesla) to a frequency (MHz)."""
    return np.asarray(tesla, dtype=float) * _validated_ratio(gyromagnetic_ratio_mhz_per_t)


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


#: Scale factor turning a dimensionless fractional shift into parts-per-million,
#: matching the Knight-shift ppm convention in ``core.fitting.knight_shift``.
PPM_SCALE = 1.0e6


def shift_from_reference(x_mhz: ArrayLike, reference_mhz: float) -> NDArray[np.float64]:
    """Return the absolute shift ``x − x₀`` (in MHz) about a reference.

    The Larmor relation ν = γ B / 2π is linear, so the MHz shift converts to a
    field shift (Gauss / Tesla) by the same :func:`convert` scaling downstream —
    a ``ν − ν₀`` in MHz is a ``B − B₀`` in field once unit-converted.
    """
    return np.asarray(x_mhz, dtype=float) - float(reference_mhz)


def relative_shift_ppm(x_mhz: ArrayLike, reference_mhz: float) -> NDArray[np.float64]:
    """Return the fractional shift ``(x − x₀)/x₀`` in ppm about a reference.

    The result is dimensionless (identical whether *x* and *reference* are read
    as frequencies or fields, since the γ factor cancels), scaled to
    parts-per-million by :data:`PPM_SCALE`. This mirrors the Knight-shift
    fractional convention (Blundell et al., *Muon Spectroscopy*, OUP 2022,
    §5.6). *reference_mhz* is a denominator, so a non-positive or non-finite
    reference is rejected — the caller (GUI) checks first and routes a missing
    reference to its untransformed fallback rather than emitting ``inf``/``nan``.
    """
    ref = float(reference_mhz)
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError(f"reference_mhz must be positive and finite for a ppm shift, got {ref!r}.")
    return (np.asarray(x_mhz, dtype=float) - ref) / ref * PPM_SCALE


def shift_axis_label(unit: FieldUnit | str) -> str:
    """Return a plot-axis label for an absolute-shift (``x − x₀``) axis."""
    return {
        FieldUnit.MHZ: "Frequency shift ν − ν₀ (MHz)",
        FieldUnit.GAUSS: "Field shift B − B₀ (G)",
        FieldUnit.TESLA: "Field shift B − B₀ (T)",
    }[FieldUnit.coerce(unit)]


def relative_shift_axis_label() -> str:
    """Return the plot-axis label for the dimensionless ppm relative-shift axis."""
    return "Relative shift (B − B₀)/B₀ (ppm)"


def frequency_resolution_mhz(bin_width_us: float, n_spectrum_points: int) -> float:
    """Return the spectrum frequency resolution 1/(2·Δt·N) in MHz.

    *bin_width_us* is the (rebinned) time-channel width and *n_spectrum_points*
    the spectrum length, matching WiMDA's ``fres = 1/(2·tres·bunch·nptsME)``.
    """
    denom = 2.0 * float(bin_width_us) * int(n_spectrum_points)
    return float("inf") if denom <= 0.0 else 1.0 / denom
