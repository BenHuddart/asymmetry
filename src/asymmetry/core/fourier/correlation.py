"""Muoniated-radical correlation spectrum (WiMDA ``Corr`` / ``AvCorr``).

A muon that adds to a double bond or aromatic ring forms a *muoniated radical*,
whose muon spin couples to the molecule's unpaired electron with an isotropic
hyperfine coupling ``A_µ``.  In a transverse field the muon--electron
(Breit--Rabi) two-spin system precesses as a **pair** of lines whose sum is
``A_µ`` (Blundell/De Renzi/Lancaster/Pratt, *Muon Spectroscopy*, OUP 2022, §4.4;
I. McKenzie, *Annu. Rep. Prog. Chem. Sect. C* **109**, 65 (2013)).

The correlation spectrum is a matched filter over the transverse-field FFT power
spectrum that collapses each genuine Breit--Rabi line-pair onto a single peak at
the hyperfine-coupling value ``A_µ`` -- the standard frequency-domain route to
identifying a radical and pinning its coupling.

This is a faithful port of WiMDA's correlation analysis (``Plot.pas`` ``rmatch``
515--523, ``CorrFn`` 1387--1394, the ``Corr``/``AvCorr`` generation loop
2149--2230), with one deliberate, documented divergence: rather than
transliterate WiMDA's approximate closed-form inverse ``rmatch`` (which carries
constants rounded at the 5th significant figure and drifts ``A_µ`` by
~0.01--0.03 MHz), we build the spectrum by the **exact Breit--Rabi forward
map** -- scanning the hyperfine axis ``A`` directly and obtaining the exact pair
``(ν₁₂, ν₃₄)`` from :func:`asymmetry.core.fitting.muonium._tf_levels`, for which
``ν₁₂ + ν₃₄ = A`` to machine precision (textbook eqn 4.65, high transverse
field).  See ``docs/porting/radical-correlation-spectrum/comparison.md``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core.fitting.muonium import _tf_levels

#: WiMDA's ``CorrOrder`` default (``FFTPar.dfm:817``).
DEFAULT_CORR_ORDER = 2


def corr_fn(y1: ArrayLike, y2: ArrayLike, order: int = DEFAULT_CORR_ORDER) -> NDArray[np.float64]:
    """Order-weighted line-pair combiner (WiMDA ``CorrFn``, ``Plot.pas:1387-1394``).

    Returns the product ``|y₁·y₂|`` weighted by an order-``n`` ratio penalty
    ``2 / (rⁿ + r⁻ⁿ)`` with ``r = |y₁/y₂|``.  The penalty is 1 when the two
    amplitudes are equal and falls toward 0 as they diverge, so a genuine pair
    (both lines present, comparable amplitude) is rewarded and a spurious pair
    (one line in the noise) is suppressed -- increasingly so for larger
    ``order``.  ``order ≤ 0`` reduces to the plain product ``|y₁·y₂|`` (WiMDA's
    ``order = 0`` / one-line-zero fallback).
    """
    a = np.abs(np.asarray(y1, dtype=float))
    b = np.abs(np.asarray(y2, dtype=float))
    product = a * b
    if order <= 0:
        return product
    both = (a > 0.0) & (b > 0.0)
    # r and 1/r only evaluated where both are positive; elsewhere `product` is
    # already zero (one factor vanished), matching WiMDA's `else` branch.
    ratio = np.divide(a, b, out=np.ones_like(product), where=both)
    inverse = np.divide(b, a, out=np.ones_like(product), where=both)
    # A vanishing amplitude sends the ratio penalty to ∞ and the result to 0 —
    # the intended "totally unequal pair is suppressed" limit, so the overflow
    # is expected and silenced rather than warned.
    with np.errstate(over="ignore", invalid="ignore"):
        denom = np.power(ratio, order) + np.power(inverse, order)
        weighted = 2.0 * product / denom
    return np.where(both & np.isfinite(weighted), weighted, np.where(both, 0.0, product))


def breit_rabi_pair(field_gauss: float, a_mhz: float) -> tuple[float, float]:
    """Return the exact high-field Breit--Rabi pair ``(ν₁₂, ν₃₄)`` in MHz.

    Thin wrapper over :func:`asymmetry.core.fitting.muonium._tf_levels` (textbook
    eqn 4.54): for an isotropic muon--electron system with hyperfine coupling
    ``a_mhz`` at transverse field ``field_gauss`` (Gauss), the two observable
    high-field precession frequencies are ``ν₁₂ = |E₁−E₂|`` and
    ``ν₃₄ = |E₃−E₄|``, and ``ν₁₂ + ν₃₄ = a_mhz`` exactly (eqn 4.65).  Reuses the
    shared Breit--Rabi machinery -- the relation is not re-derived here.
    """
    _delta, e1, e2, e3, e4 = _tf_levels(float(field_gauss), float(a_mhz))
    return abs(e1 - e2), abs(e3 - e4)


def _pair_frequencies(
    field_gauss: float, a_axis: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorise :func:`breit_rabi_pair` over a hyperfine axis ``a_axis``."""
    if a_axis.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    pairs = [breit_rabi_pair(field_gauss, float(a)) for a in a_axis]
    nu12 = np.fromiter((p[0] for p in pairs), dtype=float, count=len(pairs))
    nu34 = np.fromiter((p[1] for p in pairs), dtype=float, count=len(pairs))
    return nu12, nu34


def correlation_spectrum(
    freqs: ArrayLike,
    power: ArrayLike,
    *,
    field_gauss: float,
    order: int = DEFAULT_CORR_ORDER,
    a_axis: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the muoniated-radical correlation spectrum from a power spectrum.

    For each candidate hyperfine coupling ``A`` on the output axis, obtain the
    exact Breit--Rabi pair ``(ν₁₂, ν₃₄)`` (which sums to ``A``), linearly
    interpolate the transverse-field power spectrum ``power`` at both
    frequencies, and combine them with :func:`corr_fn`.  A genuine radical
    line-pair produces a peak at its true ``A_µ``; everything else contributes
    background.

    Parameters
    ----------
    freqs, power
        The transverse-field FFT frequency axis (MHz) and a real, non-negative
        amplitude/power channel (e.g. ``(Power)^1/2``).
    field_gauss
        The transverse field used for the Breit--Rabi pairing (WiMDA's
        ``CorrField``); normally the run's applied field.
    order
        ``CorrFn`` ratio-penalty order (default 2).
    a_axis
        Optional explicit hyperfine axis (MHz).  When omitted, a uniform axis at
        the spectrum's frequency resolution is built, bounded above by the field
        for which the upper line ``ν₃₄`` reaches the spectrum's Nyquist
        frequency (partners beyond the data are unmeasurable -- WiMDA's
        ``i2 ≤ nf`` guard).

    Returns
    -------
    ``(a_mhz, corr)``
        The hyperfine axis (MHz) and the correlation amplitude.  Empty arrays at
        zero field or for a degenerate spectrum.
    """
    frequencies = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(power, dtype=float)
    field = abs(float(field_gauss))
    if frequencies.size < 2 or spectrum.size != frequencies.size or field <= 0.0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    resolution = float(np.median(np.diff(frequencies)))
    f_max = float(frequencies.max())
    f_min = float(frequencies.min())
    if resolution <= 0.0 or not np.isfinite(resolution):
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    if a_axis is None:
        # ν₃₄ ≥ A/2 at high field, so A ≤ 2·ν₃₄ ≤ 2·f_max bounds the scan.
        grid = np.arange(resolution, 2.0 * f_max + resolution, resolution)
        nu12, nu34 = _pair_frequencies(field, grid)
        valid = np.isfinite(nu12) & np.isfinite(nu34) & (nu34 <= f_max) & (nu12 >= f_min)
        if not valid.any():
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        # Keep the contiguous samplable region (ν₃₄ rises monotonically with A).
        last = int(np.nonzero(valid)[0][-1])
        a_mhz = grid[: last + 1]
        nu12 = nu12[: last + 1]
        nu34 = nu34[: last + 1]
    else:
        a_mhz = np.asarray(a_axis, dtype=float)
        nu12, nu34 = _pair_frequencies(field, a_mhz)

    s12 = np.interp(nu12, frequencies, spectrum, left=0.0, right=0.0)
    s34 = np.interp(nu34, frequencies, spectrum, left=0.0, right=0.0)
    corr = corr_fn(s12, s34, order)
    return a_mhz, np.asarray(corr, dtype=float)


__all__ = [
    "DEFAULT_CORR_ORDER",
    "breit_rabi_pair",
    "correlation_spectrum",
    "corr_fn",
]
