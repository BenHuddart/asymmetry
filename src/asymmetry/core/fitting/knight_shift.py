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
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from asymmetry.core.fitting.derived_uncertainty import first_order_uncertainty
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz

#: Free-muon Larmor frequency per unit applied field, γ_µ/(2π) in MHz/G
#: (≈ 0.013554 MHz/G). Derived from the single γ_µ·B conversion in
#: :mod:`asymmetry.core.fitting.spectral` so the Larmor slope cannot drift from
#: the frequency-domain field axis.
MUON_LARMOR_MHZ_PER_G = field_gauss_to_frequency_mhz(1.0)


def larmor_frequency_mhz(field_gauss: float) -> float:
    """Free-muon Larmor frequency (MHz) for an applied field in Gauss."""
    return field_gauss_to_frequency_mhz(float(field_gauss))


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
    return _shift_with_uncertainty(
        nu,
        nu_ref,
        "sample",
        "reference",
        {"sample": sigma_nu, "reference": sigma_ref},
        {"sample": {"reference": cov}},
    )


def fitted_knight_shift(
    values: Mapping[str, float],
    errors: Mapping[str, float],
    *,
    component: str,
    kind: str,
    field_gauss: float,
    config: KnightShiftConfig,
    covariance: Mapping[str, Mapping[str, float]] | None = None,
    fixed_parameters: Collection[str] = (),
) -> tuple[float, float]:
    """Derive a shift from a fit, using an exact field or a co-fitted reference."""
    nu = values.get(component, math.nan)
    reference = None
    if config.reference_mode == REFERENCE_APPLIED_FIELD:
        nu_ref = field_gauss if kind == "field" else larmor_frequency_mhz(field_gauss)
    else:
        reference = config.reference_component
        nu_ref = values.get(reference, math.nan) if reference is not None else math.nan
    return _shift_with_uncertainty(
        nu, nu_ref, component, reference, errors, covariance, fixed_parameters
    )


def _shift_with_uncertainty(
    nu: float,
    nu_ref: float,
    component: str,
    reference: str | None,
    errors: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]] | None,
    fixed_parameters: Collection[str] = (),
) -> tuple[float, float]:
    if nu_ref == 0.0 or not math.isfinite(nu_ref) or not math.isfinite(nu):
        return math.nan, math.nan
    ratio = nu / nu_ref
    gradient = {component: 1.0 / nu_ref}
    if reference is not None:
        gradient[reference] = gradient.get(reference, 0.0) - ratio / nu_ref
    sigma = first_order_uncertainty(gradient, errors, covariance, fixed_parameters=fixed_parameters)
    return ratio - 1.0, sigma


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

    Returns :attr:`KnightShiftUnit.PPM` when the largest finite :math:`|K|` is below
    ``asymmetry.core.fitting.knight_shift._AUTO_PPM_MAX``, else :attr:`KnightShiftUnit.PERCENT`. An empty or
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


@dataclass(frozen=True)
class ClogstonJaccarinoResult:
    """Linear K-vs-χ fit (Clogston–Jaccarino), with temperature implicit.

    ``slope`` is ``dK/dχ`` — proportional to the muon hyperfine coupling at the
    site — and ``intercept`` is the χ-independent (orbital/diamagnetic) shift K0.
    ``slope_err`` and ``intercept_err`` are 1σ uncertainties and are ``nan``
    for an unweighted fit with only two points. ``n`` is the number of points used.
    """

    slope: float
    intercept: float
    slope_err: float
    intercept_err: float
    n: int


def clogston_jaccarino_fit(
    chi: Sequence[float],
    knight: Sequence[float],
    sigma_knight: Sequence[float] | None = None,
) -> ClogstonJaccarinoResult:
    """Linear fit of Knight shift vs susceptibility (API-only).

    Fits ``K = slope·χ + intercept`` with temperature as the implicit parameter —
    the Clogston–Jaccarino construction — where the slope ``dK/dχ`` measures the
    muon hyperfine coupling and the intercept the χ-independent shift.

    Supplied ``sigma_knight`` values are absolute standard errors in the same
    units as ``knight``. The fit uses inverse-variance weights, and the coefficient
    errors follow from those weights even with two usable points. Without
    ``sigma_knight``, the fit uses equal weights and estimates the residual
    variance as the sum of squared residuals divided by ``n - 2``. With two
    usable points this variance is unavailable and both coefficient errors are
    ``nan``. See `NIST's least-squares reference
    <https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm>`_.

    The fit drops points with non-finite χ or K, and points with non-finite or
    non-positive supplied standard errors. It raises ``ValueError`` with fewer
    than two usable points or no spread in χ.

    There is no GUI entry point because the muon experiment does not measure χ;
    pair the trended Knight shift (exported per :ref:`knight-shift`) with an
    independent susceptibility and call this directly.
    """
    x = np.asarray(chi, dtype=float)
    y = np.asarray(knight, dtype=float)
    if x.shape != y.shape:
        raise ValueError("chi and knight must have the same length")
    if sigma_knight is None:
        w = np.ones_like(x)
    else:
        sig = np.asarray(sigma_knight, dtype=float)
        if sig.shape != x.shape:
            raise ValueError("sigma_knight must match chi/knight length")
        # Guard zero/negative/non-finite errors so a single bad point cannot make
        # the weight blow up or go negative. np.where evaluates both branches, so
        # 1/σ² is computed for the bad entries too — suppress the resulting
        # divide-by-zero and invalid (NaN σ) warnings; those entries are masked out.
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.isfinite(sig) & (sig > 0.0), 1.0 / np.square(sig), np.nan)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]
    n = int(x.size)
    if n < 2:
        raise ValueError("Clogston–Jaccarino fit needs at least two finite points")

    s = float(np.sum(w))
    sx = float(np.sum(w * x))
    sy = float(np.sum(w * y))
    sxx = float(np.sum(w * x * x))
    sxy = float(np.sum(w * x * y))
    delta = s * sxx - sx * sx
    if delta <= 0.0:
        raise ValueError("Clogston–Jaccarino fit needs spread in χ (χ is constant)")

    slope = (s * sxy - sx * sy) / delta
    intercept = (sxx * sy - sx * sxy) / delta
    if sigma_knight is not None:
        variance_scale = 1.0
    elif n == 2:
        variance_scale = math.nan
    else:
        residuals = y - (slope * x + intercept)
        variance_scale = float(residuals @ residuals) / (n - 2)

    return ClogstonJaccarinoResult(
        slope=slope,
        intercept=intercept,
        slope_err=math.sqrt(variance_scale * (s / delta)),
        intercept_err=math.sqrt(variance_scale * (sxx / delta)),
        n=n,
    )
