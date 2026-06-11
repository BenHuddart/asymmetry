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

from asymmetry.core.fitting.muonium import (
    G_E_MHZ_PER_G,
    G_MU_MHZ_PER_G,
    _tf_levels,
)

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
    # When either amplitude vanishes the ratio penalty diverges and the weighted
    # value is non-finite (inf/nan); falling back to `product` (which is 0 there)
    # reproduces WiMDA's `else` branch. A diverging-but-finite penalty for a very
    # unequal pair correctly drives the result toward 0. The divide/overflow are
    # expected on those paths, so they are silenced rather than warned.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        weighted = 2.0 * product / (np.power(a / b, order) + np.power(b / a, order))
    return np.where(np.isfinite(weighted), weighted, product)


def breit_rabi_pair(field_gauss: float, a_mhz: float) -> tuple[float, float]:
    """Return the high-field Breit--Rabi pair ``(ν₁₂, ν₃₄)`` in MHz.

    Thin scalar wrapper over :func:`asymmetry.core.fitting.muonium._tf_levels`
    (textbook eqn 4.54): for an isotropic muon--electron system with hyperfine
    coupling ``a_mhz`` at transverse field ``field_gauss`` (Gauss), the two
    observable high-field precession frequencies are ``ν₁₂ = |E₁−E₂|`` and
    ``ν₃₄ = |E₃−E₄|``.  In the high-field (Paschen--Back) regime
    ``a_mhz ≫ (g_e+g_µ)·field`` their sum equals ``a_mhz`` (eqn 4.65); off that
    regime the sum departs from ``a_mhz`` (the *difference* tends to it instead).
    What matters for the correlation spectrum is that these are the two
    *observed* line positions, so a real pair peaks at its true ``a_mhz`` on the
    coupling axis regardless of regime.  Reuses the shared Breit--Rabi machinery
    -- the relation is not re-derived here.  :func:`_pair_frequencies` is the
    array-valued form (pinned equal to this wrapper by tests).
    """
    _delta, e1, e2, e3, e4 = _tf_levels(float(field_gauss), float(a_mhz))
    return abs(e1 - e2), abs(e3 - e4)


def _pair_frequencies(
    field_gauss: float, a_axis: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Array-valued form of :func:`breit_rabi_pair` over a hyperfine axis.

    Vectorised transcription of :func:`asymmetry.core.fitting.muonium._tf_levels`
    (``w12``/``w34`` only) over ``a_axis`` -- the scalar ``_tf_levels`` cannot
    take an array directly because of its ``A_hf > 0`` branch.  Pinned equal to
    the scalar :func:`breit_rabi_pair` by tests; built on the same shared
    ``G_E_MHZ_PER_G`` / ``G_MU_MHZ_PER_G`` constants.
    """
    a = np.asarray(a_axis, dtype=float)
    if a.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(a > 0.0, (G_E_MHZ_PER_G + G_MU_MHZ_PER_G) * field_gauss / a, 1.0e20)
    x = np.clip(x, -1.0e20, 1.0e20)
    d = (G_E_MHZ_PER_G - G_MU_MHZ_PER_G) / (G_E_MHZ_PER_G + G_MU_MHZ_PER_G)
    root = np.sqrt(1.0 + x * x)
    quarter = a / 4.0
    e1 = quarter * (1.0 + 2.0 * d * x)
    e2 = quarter * (-1.0 + 2.0 * root)
    e3 = quarter * (1.0 - 2.0 * d * x)
    e4 = quarter * (-1.0 - 2.0 * root)
    return np.abs(e1 - e2), np.abs(e3 - e4)


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
    Breit--Rabi line pair ``(ν₁₂, ν₃₄)`` that coupling would produce at this
    field, linearly interpolate the transverse-field power spectrum ``power`` at
    both frequencies, and combine them with :func:`corr_fn`.  A genuine radical
    line-pair produces a peak at its true ``A_µ`` (the output axis is the
    candidate coupling itself, so the peak lands at ``A_µ`` independent of the
    high-field sum rule); everything else contributes background.

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

    # The diamagnetic muon line sits at γ_µ·B, and as A → 0 both radical lines
    # collapse onto it. A candidate is only a genuine radical pair if its *lower*
    # line ν₁₂ has cleared the diamagnetic line (the upper line is always higher
    # still); below that the forward map just pairs the strong diamagnetic peak
    # with itself and raises a spurious low-A artifact. Requiring ν₁₂ above the
    # diamagnetic line is the WiMDA-faithful behaviour (WiMDA scans candidate
    # frequencies starting above the diamagnetic line, Plot.pas i0, on a spectrum
    # whose diamagnetic line has been excluded) and also removes the ν₁₂ → 0 dip
    # (the lower line buried in DC/baseline).
    diamag_mhz = G_MU_MHZ_PER_G * field
    lower_floor = max(f_min, diamag_mhz + 2.0 * resolution)

    def resolvable(nu12: NDArray[np.float64], nu34: NDArray[np.float64]) -> NDArray[np.bool_]:
        """True where the pair is a measurable radical line pair."""
        return (nu12 >= lower_floor) & (nu34 <= f_max) & np.isfinite(nu34)

    if a_axis is None:
        # ν₃₄ rises monotonically with A and ν₃₄ ≥ A/2 at high field, so
        # A ≤ 2·ν₃₄ ≤ 2·f_max bounds the scan; keep the contiguous band of
        # couplings whose pair is resolvable (lower line above the diamagnetic
        # line, upper line within the spectrum's Nyquist — WiMDA's i2 ≤ nf).
        grid = np.arange(resolution, 2.0 * f_max + resolution, resolution)
        nu12, nu34 = _pair_frequencies(field, grid)
        samplable = resolvable(nu12, nu34)
        if not samplable.any():
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        kept = np.nonzero(samplable)[0]
        first, last = int(kept[0]), int(kept[-1])
        a_mhz = grid[first : last + 1]
        nu12 = nu12[first : last + 1]
        nu34 = nu34[first : last + 1]
    else:
        a_mhz = np.asarray(a_axis, dtype=float)
        nu12, nu34 = _pair_frequencies(field, a_mhz)

    s12 = np.interp(nu12, frequencies, spectrum, left=0.0, right=0.0)
    s34 = np.interp(nu34, frequencies, spectrum, left=0.0, right=0.0)
    corr = np.asarray(corr_fn(s12, s34, order), dtype=float)
    # Zero any candidate whose pair is not resolvable (also covers an explicit
    # a_axis, where the band trim above does not apply).
    corr = np.where(resolvable(nu12, nu34), corr, 0.0)
    return a_mhz, corr


__all__ = [
    "DEFAULT_CORR_ORDER",
    "breit_rabi_pair",
    "correlation_spectrum",
    "corr_fn",
]
