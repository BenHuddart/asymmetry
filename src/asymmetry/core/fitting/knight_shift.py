"""Muon Knight-shift conversion from fitted precession frequencies.

The Knight shift is the relative frequency shift of the muon precession away from
the free-muon Larmor frequency of the applied field:

    K = (ν_µ − ν_ref) / ν_ref          (dimensionless)

with two reference conventions (both supported here):

* **Applied field** — ``ν_ref = γ_µ · B_ext``, the free-muon Larmor frequency of
  the applied field. This needs no measured reference line, so it is the default
  for low-background systems. (Amato, *Intro to MuSR*, Eq. 5.54; Blundell,
  *Muon Spectroscopy*, Eq. 16.14: ``K = (B_µ − B_ext)·B_ext / B_ext²``.)
* **Designated component** — ``ν_ref`` is the fitted frequency of a reference
  oscillation component measured in the same fit (e.g. a diamagnetic line). The
  two frequencies are then correlated, so the covariance is carried through the
  error propagation.

This module computes the directly-measured ``K_exp``. The Lorentz/demagnetizing
correction to the intrinsic ``K_µ`` (Amato Eqs. 5.59–5.60) needs sample geometry
and susceptibility and is intentionally out of scope.

Knight shifts span a wide range — tens of ppm for diamagnets up to a few percent
for paramagnets (Amato p.194) — so :func:`resolve_auto_unit` picks a sensible
display unit from the data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)

#: Free-muon Larmor frequency per unit applied field, γ_µ/(2π) in MHz/G
#: (≈ 0.013554 MHz/G). Derived from the canonical MHz/T constant so the two
#: cannot drift.
MUON_LARMOR_MHZ_PER_G = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA


def larmor_frequency_mhz(field_gauss: float) -> float:
    """Free-muon Larmor frequency (MHz) for an applied field in Gauss."""
    return MUON_LARMOR_MHZ_PER_G * float(field_gauss)


def knight_shift(
    nu: float,
    nu_ref: float,
    *,
    sigma_nu: float = 0.0,
    sigma_ref: float = 0.0,
    cov: float = 0.0,
) -> tuple[float, float]:
    """Knight shift ``K = ν/ν_ref − 1`` and its 1σ uncertainty.

    ``cov`` is the covariance between ``nu`` and ``nu_ref`` — non-zero only when
    both are fitted in the same fit (the designated-component reference). For the
    applied-field reference ``nu_ref = γ_µ·B`` is treated as exact, so the caller
    leaves ``sigma_ref`` and ``cov`` at zero.

    Returns ``(nan, nan)`` when ``nu_ref`` is zero (no reference to shift against).
    The error propagation is the standard first-order expansion of the ratio:

        σ_K² = σ_ν²/ν_ref²  +  ν²·σ_ref²/ν_ref⁴  −  2·ν·cov/ν_ref³
    """
    nu = float(nu)
    nu_ref = float(nu_ref)
    if nu_ref == 0.0 or not math.isfinite(nu_ref):
        return float("nan"), float("nan")

    k = nu / nu_ref - 1.0

    # Partial derivatives: ∂K/∂ν = 1/ν_ref, ∂K/∂ν_ref = −ν/ν_ref².
    d_nu = 1.0 / nu_ref
    d_ref = -nu / (nu_ref * nu_ref)
    variance = (
        d_nu * d_nu * float(sigma_nu) ** 2
        + d_ref * d_ref * float(sigma_ref) ** 2
        + 2.0 * d_nu * d_ref * float(cov)
    )
    # Round-off (or a pathological negative covariance) can push the variance
    # slightly below zero; clamp rather than emit a NaN sigma.
    sigma_k = math.sqrt(variance) if variance > 0.0 else 0.0
    return k, sigma_k


class KnightShiftUnit(Enum):
    """Display unit for a Knight shift (stored internally as a fraction)."""

    FRACTION = "fraction"
    PPM = "ppm"
    PERCENT = "percent"
    AUTO = "auto"


#: Multiplicative scale from the internal fraction to each concrete display unit.
_UNIT_SCALE = {
    KnightShiftUnit.FRACTION: 1.0,
    KnightShiftUnit.PPM: 1.0e6,
    KnightShiftUnit.PERCENT: 100.0,
}

#: Short axis-label suffix per concrete unit.
_UNIT_LABEL = {
    KnightShiftUnit.FRACTION: "",
    KnightShiftUnit.PPM: "ppm",
    KnightShiftUnit.PERCENT: "%",
}

#: |K| below this (fraction) reads naturally in ppm; above it, in percent.
#: A diamagnet sits at tens of ppm, a paramagnet at up to a few percent.
_AUTO_PPM_MAX = 1.0e-3


def resolve_auto_unit(values: Iterable[float]) -> KnightShiftUnit:
    """Pick ppm vs percent for a set of Knight shifts (fractions).

    Returns :attr:`KnightShiftUnit.PPM` when the largest finite |K| is below
    :data:`_AUTO_PPM_MAX`, else :attr:`KnightShiftUnit.PERCENT`. An empty or
    all-non-finite set defaults to ppm.
    """
    peak = 0.0
    for value in values:
        v = abs(float(value))
        if math.isfinite(v) and v > peak:
            peak = v
    return KnightShiftUnit.PPM if peak < _AUTO_PPM_MAX else KnightShiftUnit.PERCENT


def concrete_unit(unit: KnightShiftUnit, values: Iterable[float]) -> KnightShiftUnit:
    """Resolve ``AUTO`` against the data; pass concrete units through unchanged."""
    if unit is KnightShiftUnit.AUTO:
        return resolve_auto_unit(values)
    return unit


def scale_for_unit(unit: KnightShiftUnit) -> float:
    """Factor to multiply an internal fraction by for display in ``unit``."""
    return _UNIT_SCALE[unit]


def label_for_unit(unit: KnightShiftUnit) -> str:
    """Short unit suffix for an axis/legend label (empty for a bare fraction)."""
    return _UNIT_LABEL[unit]


#: Reference conventions for the Knight shift.
REFERENCE_APPLIED_FIELD = "applied_field"
REFERENCE_COMPONENT = "component"


@dataclass
class KnightShiftConfig:
    """User configuration for converting fitted frequencies to Knight shifts.

    ``reference_mode`` is ``"applied_field"`` (ν_ref = γ_µ·B, needs no reference
    line) or ``"component"`` (ν_ref is ``reference_component``'s fitted frequency).
    ``components`` lists the oscillation-frequency parameter names to convert; an
    empty tuple means "all discovered components". ``unit`` is the display unit
    (the conversion is stored internally as a dimensionless fraction).
    """

    enabled: bool = False
    reference_mode: str = REFERENCE_APPLIED_FIELD
    reference_component: str | None = None
    unit: KnightShiftUnit = KnightShiftUnit.AUTO
    components: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "reference_mode": str(self.reference_mode),
            "reference_component": self.reference_component,
            "unit": self.unit.value,
            "components": list(self.components),
        }

    @classmethod
    def from_dict(cls, data: object) -> KnightShiftConfig:
        if not isinstance(data, dict):
            return cls()
        try:
            unit = KnightShiftUnit(str(data.get("unit", KnightShiftUnit.AUTO.value)))
        except ValueError:
            unit = KnightShiftUnit.AUTO
        mode = str(data.get("reference_mode", REFERENCE_APPLIED_FIELD))
        if mode not in (REFERENCE_APPLIED_FIELD, REFERENCE_COMPONENT):
            mode = REFERENCE_APPLIED_FIELD
        ref = data.get("reference_component")
        components = data.get("components") or []
        return cls(
            enabled=bool(data.get("enabled", False)),
            reference_mode=mode,
            reference_component=str(ref) if ref else None,
            unit=unit,
            components=tuple(str(c) for c in components),
        )
