"""Statistical moments of a muon field/frequency spectrum.

Once a MaxEnt or phase-corrected FFT spectrum exists, its lineshape can be
reduced to a handful of numbers that answer concrete physics questions:

* ``b_rms_mean`` — the RMS width of the field distribution ``p(B)``; in the
  London limit of a type-II superconductor's mixed state this sets the magnetic
  penetration depth (``B_rms ∝ 1/λ²``) and hence the superfluid density.
* ``skewness`` / ``beta`` — the *asymmetry* of ``p(B)``: the long high-field tail
  from the vortex cores and the sharp low-field cutoff at the lattice saddle
  point, which probe the vortex-lattice geometry and its disorder.
* ``b_pk`` / ``b_ave`` — where the line sits, and its diamagnetic shift.

This module is the Qt-free, array-in / array-out core. It is *unit-agnostic*: the
caller passes an axis ``x`` in whatever unit it likes (the GUI defaults to Gauss,
the penetration-depth reading) and the returned ``B_*`` moments share that unit,
while ``skewness`` and ``beta`` are dimensionless and invariant under the linear
field↔frequency rescaling.

The arithmetic follows WiMDA's ``Moments.pas`` as a behavioural oracle — the
amplitude-weighted central moments, the five-point parabolic peak refinement, the
cutoff-as-fraction-of-peak gate — with documented divergences (see
``docs/porting/spectral-moments/comparison.md``): a consistent ``0…n-1`` index
range (WiMDA's peak search has an off-by-one), the textbook skewness ``γ₁``
reported alongside WiMDA's cube-root ``α``, real per-moment uncertainties (WiMDA
gives none), and a discrete peak taken over the analysis *range* rather than the
whole spectrum so the cutoff threshold and ``b_pk`` stay self-consistent.

Physics reference: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An
Introduction* (OUP, 2022), for the mixed-state field distribution; E. H. Brandt
for the vortex-lattice ``p(B)`` whose positive skew fixes the sign of ``beta``.

**``b_pk`` is the fragile member of the set.** It is a parabola fitted to the five
spectral points around the discrete maximum; on a noisy or near-flat spectrum the
maximum hops between bins and the vertex can swing wide. ``b_diff`` and especially
``beta`` (which is referenced to ``b_pk``) inherit that fragility. The robust
members are ``b_ave``, ``b_rms_mean`` and ``skewness`` — amplitude-weighted
integrals that average noise down. The bootstrap uncertainties make this visible
rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Minimum points required for the five-point parabolic peak refinement.
_PARABOLA_POINTS = 5

#: Uncertainty methods accepted by :func:`spectrum_moments`.
UNCERTAINTY_METHODS = ("bootstrap", "propagate", "none")


@dataclass(frozen=True)
class SpectrumMoments:
    """The WiMDA moment set for one spectrum, with per-moment uncertainties.

    ``B_*`` fields carry the axis unit the caller passed (Gauss by default);
    ``skewness``, ``skewness_g1`` and ``beta`` are dimensionless. ``*_err`` are
    1σ uncertainties, ``NaN`` when unavailable (no spectrum errors, or a moment
    the chosen method cannot propagate). An empty window yields ``n_sample == 0``
    and all-``NaN`` moments.
    """

    b_pk: float
    b_ave: float
    b_diff: float
    b_rms_mean: float
    b_rms_peak: float
    skewness: float
    skewness_g1: float
    beta: float
    n_sample: int
    peak_refined: bool
    b_pk_err: float = float("nan")
    b_ave_err: float = float("nan")
    b_diff_err: float = float("nan")
    b_rms_mean_err: float = float("nan")
    b_rms_peak_err: float = float("nan")
    skewness_err: float = float("nan")
    skewness_g1_err: float = float("nan")
    beta_err: float = float("nan")
    #: Peak amplitude inside the window (the cutoff reference); ``NaN`` if empty.
    window_peak_amplitude: float = float("nan")
    #: Extraction provenance: range, cutoff, unit, mode, uncertainty seed/method.
    recipe: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the window held points and a finite mean was found."""
        return self.n_sample > 0 and np.isfinite(self.b_ave)


# Ordered moment columns shared by the trend-row builder and the GUI readout.
MOMENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("B_pk", "b_pk"),
    ("B_ave", "b_ave"),
    ("B_diff", "b_diff"),
    ("B_rms_mean", "b_rms_mean"),
    ("B_rms_peak", "b_rms_peak"),
    ("skewness", "skewness"),
    ("skewness_g1", "skewness_g1"),
    ("beta", "beta"),
)

_ERR_ATTR = {
    "b_pk": "b_pk_err",
    "b_ave": "b_ave_err",
    "b_diff": "b_diff_err",
    "b_rms_mean": "b_rms_mean_err",
    "b_rms_peak": "b_rms_peak_err",
    "skewness": "skewness_err",
    "skewness_g1": "skewness_g1_err",
    "beta": "beta_err",
}


# ── core moment kernel ──────────────────────────────────────────────────────


def _parabolic_peak(x: NDArray[np.float64], amp: NDArray[np.float64], ipk: int) -> float | None:
    """Return the five-point parabolic-vertex peak position, or ``None``.

    Fits a quadratic to the five points centred on *ipk* (in normalised
    coordinates for conditioning, as WiMDA does) and returns its vertex. Returns
    ``None`` when *ipk* is within two points of either end (the edge guard) or the
    fitted parabola is degenerate (opens the wrong way / is flat), so the caller
    falls back to the discrete peak bin.
    """
    n = x.size
    half = _PARABOLA_POINTS // 2
    if ipk < half or ipk > n - half - 1:
        return None
    lo, hi = ipk - half, ipk + half + 1
    x0 = float(x[ipk])
    dx = float(x[ipk + 1] - x[ipk])
    if dx == 0.0:
        return None
    xn = (x[lo:hi] - x0) / dx
    yn = amp[lo:hi]
    # Quadratic least squares: y = a·xn² + b·xn + c.
    coeffs = np.polyfit(xn, yn, 2)
    a, b = float(coeffs[0]), float(coeffs[1])
    if a >= 0.0:  # not a downward parabola → no meaningful maximum
        return None
    vertex_n = -b / (2.0 * a)
    vertex = vertex_n * dx + x0
    # Clamp to the fitted five-point span: a near-flat parabola can place the
    # vertex far outside the points it was fitted to, which is unphysical for a
    # peak. WiMDA leaves this unclamped (divergence D8); we bound it.
    span_lo, span_hi = float(x[lo]), float(x[hi - 1])
    return min(max(vertex, span_lo), span_hi)


def _moment_scalars(
    x: NDArray[np.float64], amp: NDArray[np.float64], cutoff_fraction: float
) -> dict[str, float] | None:
    """Compute the moment set on an already range-restricted spectrum.

    *x* / *amp* are the in-range axis and amplitude. Applies the
    cutoff-vs-discrete-peak mask, then the amplitude-weighted moments. Returns
    ``None`` when the window is empty (no point above cutoff, or non-positive
    total weight) so the caller can emit an empty result.
    """
    if x.size == 0:
        return None
    ipk = int(np.argmax(amp))
    peak_amp = float(amp[ipk])
    if not np.isfinite(peak_amp) or peak_amp <= 0.0:
        return None

    refined = _parabolic_peak(x, amp, ipk)
    b_pk = float(x[ipk]) if refined is None else float(refined)

    threshold = float(cutoff_fraction) * peak_amp
    mask = amp > threshold
    p = amp[mask]
    b = x[mask]
    m0 = float(p.sum())
    if p.size == 0 or m0 <= 0.0:
        return None

    b_ave = float((p * b).sum() / m0)
    d = b - b_ave
    m2 = float((p * d * d).sum() / m0)
    m3 = float((p * d * d * d).sum() / m0)
    dpk = b - b_pk
    m2pk = float((p * dpk * dpk).sum() / m0)

    b_rms_mean = float(np.sqrt(m2)) if m2 > 0.0 else 0.0
    b_rms_peak = float(np.sqrt(m2pk)) if m2pk > 0.0 else 0.0
    # WiMDA's cube-root skewness α = sign(m₃)·∛|m₃| / √m₂ (dimensionless).
    skewness = (
        float(np.sign(m3) * np.cbrt(abs(m3)) / b_rms_mean) if b_rms_mean > 0.0 else float("nan")
    )
    # Textbook standardised skewness γ₁ = m₃ / m₂^{3/2}.
    skewness_g1 = float(m3 / m2**1.5) if m2 > 0.0 else float("nan")
    beta = float((b_ave - b_pk) / b_rms_peak) if b_rms_peak > 0.0 else float("nan")

    return {
        "b_pk": b_pk,
        "b_ave": b_ave,
        "b_diff": b_ave - b_pk,
        "b_rms_mean": b_rms_mean,
        "b_rms_peak": b_rms_peak,
        "skewness": skewness,
        "skewness_g1": skewness_g1,
        "beta": beta,
        "n_sample": float(p.size),
        "peak_refined": 1.0 if refined is not None else 0.0,
        "window_peak": peak_amp,
    }


# ── uncertainties ───────────────────────────────────────────────────────────

_BOOTSTRAP_KEYS = (
    "b_pk",
    "b_ave",
    "b_diff",
    "b_rms_mean",
    "b_rms_peak",
    "skewness",
    "skewness_g1",
    "beta",
)


def _bootstrap_errors(
    x: NDArray[np.float64],
    amp: NDArray[np.float64],
    sigma: NDArray[np.float64],
    cutoff_fraction: float,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Std of each moment over noise realisations ``amp + N(0, σ)``.

    Resampling re-finds the peak each draw, so ``b_pk``/``beta`` error bars
    inflate on noisy, flat-topped spectra — the fragility made measurable.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in _BOOTSTRAP_KEYS}
    for _ in range(int(n_bootstrap)):
        noisy = amp + rng.normal(0.0, sigma)
        res = _moment_scalars(x, noisy, cutoff_fraction)
        if res is None:
            continue
        for k in _BOOTSTRAP_KEYS:
            draws[k].append(res[k])
    out: dict[str, float] = {}
    for k, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return out


def _propagate_errors(
    x: NDArray[np.float64],
    amp: NDArray[np.float64],
    sigma: NDArray[np.float64],
    cutoff_fraction: float,
    base: dict[str, float],
) -> dict[str, float]:
    """First-order (linear) propagation of σ through the *integral* moments.

    Deterministic and cheap, but only the amplitude-weighted integral moments are
    differentiable in the weights; the parabolic ``b_pk`` and the ``b_pk``-derived
    ``b_diff``/``beta`` are left ``NaN`` (use bootstrap for those). Mirrors the
    intent of WiMDA's commented ``errsread`` block.
    """
    out = dict.fromkeys(_BOOTSTRAP_KEYS, float("nan"))
    ipk = int(np.argmax(amp))
    peak_amp = float(amp[ipk])
    if peak_amp <= 0.0:
        return out
    mask = amp > cutoff_fraction * peak_amp
    p, b, s = amp[mask], x[mask], sigma[mask]
    m0 = float(p.sum())
    if m0 <= 0.0:
        return out
    b_ave = base["b_ave"]
    # var(b_ave) = Σ [(B_i - b_ave)/m0]² σ_i²
    out["b_ave"] = float(np.sqrt(np.sum(((b - b_ave) / m0) ** 2 * s**2)))
    # m2 = Σ p_i d_i²/m0 with d fixed → var(m2) = Σ (d_i²/m0)² σ_i²; δ√m2 = δm2/(2√m2)
    d = b - b_ave
    m2 = base["b_rms_mean"] ** 2
    if m2 > 0.0:
        var_m2 = float(np.sum((d**2 / m0) ** 2 * s**2))
        out["b_rms_mean"] = float(np.sqrt(var_m2) / (2.0 * np.sqrt(m2)))
    dpk = b - base["b_pk"]
    m2pk = base["b_rms_peak"] ** 2
    if m2pk > 0.0:
        var_m2pk = float(np.sum((dpk**2 / m0) ** 2 * s**2))
        out["b_rms_peak"] = float(np.sqrt(var_m2pk) / (2.0 * np.sqrt(m2pk)))
    # γ₁ = m₃/m₂^{3/2}: propagate the m₃ term (dominant), m₂ held.
    m2_15 = m2**1.5
    if m2_15 > 0.0:
        var_m3 = float(np.sum((d**3 / m0) ** 2 * s**2))
        out["skewness_g1"] = float(np.sqrt(var_m3) / m2_15)
    return out


# ── public API ──────────────────────────────────────────────────────────────


def _empty(recipe: dict[str, Any]) -> SpectrumMoments:
    nan = float("nan")
    return SpectrumMoments(
        b_pk=nan,
        b_ave=nan,
        b_diff=nan,
        b_rms_mean=nan,
        b_rms_peak=nan,
        skewness=nan,
        skewness_g1=nan,
        beta=nan,
        n_sample=0,
        peak_refined=False,
        recipe=recipe,
    )


def spectrum_moments(
    x: ArrayLike,
    amplitude: ArrayLike,
    *,
    x_range: tuple[float, float] | None,
    cutoff_fraction: float,
    errors: ArrayLike | None = None,
    uncertainty: str = "bootstrap",
    n_bootstrap: int = 256,
    seed: int = 0,
    unit: str | None = None,
    mode: str | None = None,
) -> SpectrumMoments:
    """Return the moment set of a spectrum over a window above an amplitude cutoff.

    Parameters
    ----------
    x, amplitude:
        The spectrum axis (field or frequency, in the caller's unit) and its
        baseline-subtracted amplitude. Need not be sorted; non-finite samples are
        dropped.
    x_range:
        ``(lo, hi)`` analysis window in *x* units, or ``None`` for the full axis.
        The discrete peak (and so the cutoff threshold and ``b_pk``) is taken
        *within* this window — a divergence from WiMDA's whole-spectrum peak that
        keeps the windowed analysis self-consistent.
    cutoff_fraction:
        Points with ``amplitude <= cutoff_fraction · peak`` are excluded (the
        WiMDA "% of peak" cutoff, as a fraction in ``[0, 1)``).
    errors:
        Optional per-point 1σ amplitude errors enabling uncertainties.
    uncertainty:
        ``"bootstrap"`` (default; resample over *errors*, propagates through every
        moment incl. the nonlinear ones), ``"propagate"`` (linear, integral
        moments only), or ``"none"``. Falls back to all-``NaN`` errors when
        *errors* is absent or non-positive.
    n_bootstrap, seed:
        Bootstrap draw count and RNG seed (recorded in ``recipe`` for
        reproducibility).
    unit, mode:
        Provenance only — the display unit and spectrum mode, stored in ``recipe``.
    """
    if uncertainty not in UNCERTAINTY_METHODS:
        raise ValueError(f"uncertainty must be one of {UNCERTAINTY_METHODS}, got {uncertainty!r}")
    cutoff_fraction = float(cutoff_fraction)
    if not np.isfinite(cutoff_fraction) or not (0.0 <= cutoff_fraction < 1.0):
        raise ValueError(f"cutoff_fraction must be in [0, 1), got {cutoff_fraction!r}")

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(amplitude, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and amplitude must be 1-D arrays of equal length")
    sig = None if errors is None else np.asarray(errors, dtype=float)
    if sig is not None and sig.shape != xa.shape:
        raise ValueError("errors must match x/amplitude length")

    recipe: dict[str, Any] = {
        "x_range": None if x_range is None else [float(x_range[0]), float(x_range[1])],
        "cutoff_fraction": cutoff_fraction,
        "unit": unit,
        "mode": mode,
        "uncertainty": uncertainty,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }

    finite = np.isfinite(xa) & np.isfinite(ya)
    if sig is not None:
        finite &= np.isfinite(sig)
    xa, ya = xa[finite], ya[finite]
    sig = None if sig is None else sig[finite]

    # Sort by axis so the parabolic-peak neighbours are spatial neighbours.
    order = np.argsort(xa, kind="stable")
    xa, ya = xa[order], ya[order]
    sig = None if sig is None else sig[order]

    if x_range is not None:
        lo, hi = float(x_range[0]), float(x_range[1])
        if lo > hi:
            lo, hi = hi, lo
        in_range = (xa >= lo) & (xa <= hi)
        xa, ya = xa[in_range], ya[in_range]
        sig = None if sig is None else sig[in_range]

    base = _moment_scalars(xa, ya, cutoff_fraction)
    if base is None:
        return _empty(recipe)

    errs: dict[str, float] = dict.fromkeys(_BOOTSTRAP_KEYS, float("nan"))
    # Sanitise the error array before it reaches the noise model: a stray
    # negative or non-finite σ (a derived/subtracted spectrum, a bad loader) would
    # make ``rng.normal(0, σ)`` raise; clamp such points to zero (unperturbed).
    sig_safe = None if sig is None else np.where(np.isfinite(sig) & (sig > 0.0), sig, 0.0)
    have_sigma = sig_safe is not None and bool(np.any(sig_safe > 0.0))
    if uncertainty != "none" and have_sigma:
        if uncertainty == "bootstrap":
            errs = _bootstrap_errors(xa, ya, sig_safe, cutoff_fraction, n_bootstrap, seed)
        else:
            errs = _propagate_errors(xa, ya, sig_safe, cutoff_fraction, base)

    return SpectrumMoments(
        b_pk=base["b_pk"],
        b_ave=base["b_ave"],
        b_diff=base["b_diff"],
        b_rms_mean=base["b_rms_mean"],
        b_rms_peak=base["b_rms_peak"],
        skewness=base["skewness"],
        skewness_g1=base["skewness_g1"],
        beta=base["beta"],
        n_sample=int(base["n_sample"]),
        peak_refined=bool(base["peak_refined"]),
        b_pk_err=errs["b_pk"],
        b_ave_err=errs["b_ave"],
        b_diff_err=errs["b_diff"],
        b_rms_mean_err=errs["b_rms_mean"],
        b_rms_peak_err=errs["b_rms_peak"],
        skewness_err=errs["skewness"],
        skewness_g1_err=errs["skewness_g1"],
        beta_err=errs["beta"],
        window_peak_amplitude=base["window_peak"],
        recipe=recipe,
    )


def moments_trend_row(
    moments: SpectrumMoments,
    *,
    run_number: int,
    run_label: str | None = None,
    field: float | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build a ``fit_result_summary``-shaped row for one spectrum's moments.

    The shape (``success`` / ``parameters`` / ``uncertainties`` + explicit
    ``field`` / ``temperature`` / ``run_label`` / ``run_number``) is exactly what
    a computed :class:`~asymmetry.core.representation.series.FitSeries`'
    ``results_by_run`` consumes, so a moments series trends like any fit. A
    non-finite coordinate is stored as ``None`` (JSON null → "off this axis").
    """

    def _coord(v: float | None) -> float | None:
        if v is None or not np.isfinite(float(v)):
            return None
        return float(v)

    values = {col: float(getattr(moments, attr)) for col, attr in MOMENT_COLUMNS}
    errors = {col: float(getattr(moments, _ERR_ATTR[attr])) for col, attr in MOMENT_COLUMNS}
    return {
        "success": bool(moments.ok),
        "parameters": values,
        "uncertainties": errors,
        "run_number": int(run_number),
        "run_label": str(run_label) if run_label is not None else None,
        "field": _coord(field),
        "temperature": _coord(temperature),
    }
