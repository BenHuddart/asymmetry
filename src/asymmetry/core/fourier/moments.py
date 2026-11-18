"""Cutoff-selected moments of a displayed field/frequency spectrum.

The caller supplies the axis unit; widths and positions retain it, while
skewness and beta are invariant under linear field/frequency rescaling.
The selected, range-limited displayed spectrum is the target, not automatically
an unwindowed physical full-distribution moment. Windowing and reconstruction
can change that target.

Amplitude-weighted moments, a five-point parabolic peak and a fraction-of-peak
cutoff follow the documented spectral-moment convention. The peak and cutoff
are chosen within the requested range. Peak-bin/support selection can be unstable
under noise, including for the centroid and RMS width.

Errors are heuristic independent-bin approximations with frozen processing.
Spectral covariance, estimated phase and reconstruction uncertainty are not fully
propagated; scalar padding rescaling does not establish coverage. Bootstrap
reselects spectral support/peak; first-order propagation freezes their indices.
See the spectral-moments reference for definitions, sources and limits.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from asymmetry.core import computation
from asymmetry.core.fourier.uncertainty import MOMENT_UNCERTAINTY_SCOPE

#: Minimum points required for the five-point parabolic peak refinement.
_PARABOLA_POINTS = 5

#: Uncertainty methods accepted by :func:`spectrum_moments`.
UNCERTAINTY_METHODS = ("bootstrap", "propagate", "none")


class MomentsCancelledError(Exception):
    """A spectral-moment calculation was cooperatively cancelled."""


@dataclass(frozen=True)
class SpectrumMoments:
    """Selected displayed-spectrum moments with heuristic approximate errors.

    ``B_*`` fields carry the axis unit the caller passed (Gauss by default);
    ``skewness``, ``skewness_g1`` and ``beta`` are dimensionless. ``*_err`` are
    heuristic approximate errors, ``NaN`` when unavailable (no spectrum errors,
    or a moment the chosen method cannot propagate). An empty window yields ``n_sample == 0``
    and all-``NaN`` moments. Propagation assumes fixed support and peak bin,
    independent errors, and leaves cube-root alpha unavailable.
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
    computation_revision: str | None = None

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


@dataclass(frozen=True)
class _ParabolicVertex:
    position: float
    gradient: NDArray[np.float64] | None


def _parabolic_peak(
    x: NDArray[np.float64], amp: NDArray[np.float64], ipk: int
) -> _ParabolicVertex | None:
    """Five-point vertex and its amplitude derivative on a fixed peak bin.

    Normalised coordinates keep the quadratic operator conditioned. An edge,
    nonconcave or clamped vertex has no differentiable peak uncertainty.
    """
    half = _PARABOLA_POINTS // 2
    if ipk < half or ipk > x.size - half - 1:
        return None
    lo, hi = ipk - half, ipk + half + 1
    x0 = float(x[ipk])
    dx = float(x[ipk + 1] - x[ipk])
    if dx == 0.0:
        return None
    xn = (x[lo:hi] - x0) / dx
    operator = np.linalg.pinv(np.column_stack((xn**2, xn, np.ones_like(xn))))
    a, b, _ = operator @ amp[lo:hi]
    if a >= 0.0:
        return None
    vertex = x0 - dx * b / (2 * a)
    span_lo, span_hi = float(x[lo]), float(x[hi - 1])
    if vertex <= span_lo or vertex >= span_hi:
        return _ParabolicVertex(float(np.clip(vertex, span_lo, span_hi)), None)
    gradient = np.zeros_like(amp)
    gradient[lo:hi] = dx * (-operator[1] / (2 * a) + b * operator[0] / (2 * a**2))
    return _ParabolicVertex(float(vertex), gradient)


def _support(amp: NDArray[np.float64], cutoff: float) -> tuple[int, NDArray[np.bool_]]:
    """Fix the discrete peak and the samples strictly above its cutoff."""
    ipk = int(np.argmax(amp))
    return ipk, amp > cutoff * amp[ipk]


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
    ipk, mask = _support(amp, cutoff_fraction)
    peak_amp = float(amp[ipk])
    if not np.isfinite(peak_amp) or peak_amp <= 0.0:
        return None

    refined = _parabolic_peak(x, amp, ipk)
    b_pk = float(x[ipk]) if refined is None else refined.position

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
    cancel_callback: Callable[[], bool] | None,
) -> dict[str, float]:
    """Std of each moment over noise realisations ``amp + N(0, σ)``.

    Resampling re-finds the peak each draw, so ``b_pk``/``beta`` error bars
    inflate on noisy, flat-topped spectra — the fragility made measurable.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in _BOOTSTRAP_KEYS}
    for _ in range(int(n_bootstrap)):
        if cancel_callback is not None and cancel_callback():
            raise MomentsCancelledError("Spectral moments cancelled.")
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


def _moment_jacobian(
    x: NDArray[np.float64],
    amp: NDArray[np.float64],
    cutoff_fraction: float,
    base: dict[str, float],
) -> dict[str, NDArray[np.float64]]:
    """First-order rows with fixed support and peak bin; missing means unavailable.

    Normalisation, central moments and the refined peak all vary with amplitude.
    JCGM 100:2008 section 5 gives the uncertainty contraction for independent
    errors. Cube-root alpha is excluded: its derivative diverges at symmetry.
    """
    ipk, mask = _support(amp, cutoff_fraction)
    total = float(amp[mask].sum())
    m2 = base["b_rms_mean"] ** 2
    if m2 <= 0:
        return {}
    d = x - base["b_ave"]
    mean = np.where(mask, d / total, 0.0)
    second = np.where(mask, (d**2 - m2) / total, 0.0)
    gamma = base["skewness_g1"]
    third = np.where(mask, (d**3 - gamma * m2**1.5 - 3 * m2 * d) / total, 0.0)
    rows = {
        "b_ave": mean,
        "b_rms_mean": second / (2 * np.sqrt(m2)),
        "skewness_g1": third / m2**1.5 - 1.5 * gamma / m2 * second,
    }
    vertex = _parabolic_peak(x, amp, ipk)
    if vertex is None or vertex.gradient is None:
        return rows
    peak = vertex.gradient
    m2pk = base["b_rms_peak"] ** 2
    e = x - base["b_pk"]
    second_peak = np.where(mask, (e**2 - m2pk) / total, 0.0) - 2 * base["b_diff"] * peak
    rows["b_pk"] = peak
    rows["b_diff"] = mean - peak
    rows["b_rms_peak"] = second_peak / (2 * np.sqrt(m2pk))
    rows["beta"] = (mean - peak) / np.sqrt(m2pk) - base["beta"] * second_peak / (2 * m2pk)
    return rows


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
        computation_revision=computation.COMPUTATION_REVISION,
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
    error_oversampling: float = 1.0,
    cancel_callback: Callable[[], bool] | None = None,
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
        ``"bootstrap"`` (default; perturb independent spectral bins, reselecting
        support and peak), ``"propagate"`` (first order with fixed support and
        peak bin, independent errors; cube-root alpha unavailable),
        or ``"none"``. Errors are all ``NaN`` when *errors* is absent or
        non-positive. Unrefined or clamped peaks have no peak-derived errors.
    n_bootstrap, seed:
        Bootstrap draw count and RNG seed (recorded in ``recipe`` for
        reproducibility).
    unit, mode:
        Provenance only — the display unit and spectrum mode, stored in ``recipe``.
    cancel_callback:
        Optional cooperative cancellation predicate, polled before calculation,
        between bootstrap draws and before returning. Raises ``MomentsCancelledError``.
    error_oversampling:
        Scalar heuristic: finite output errors are multiplied by
        ``sqrt(error_oversampling)`` when greater than one; central values stay
        unchanged. Pass the FFT padding factor. This does not recover spectral
        covariance and can understate or overstate errors. Neither this scaling
        nor independent-bin perturbations establish nominal coverage.
    """
    if cancel_callback is not None and cancel_callback():
        raise MomentsCancelledError("Spectral moments cancelled.")
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
        recipe["uncertainty_scope"] = MOMENT_UNCERTAINTY_SCOPE
        recipe["error_oversampling"] = max(float(error_oversampling), 1.0)
        if uncertainty == "bootstrap":
            errs = _bootstrap_errors(
                xa, ya, sig_safe, cutoff_fraction, n_bootstrap, seed, cancel_callback
            )
        else:
            for key, row in _moment_jacobian(xa, ya, cutoff_fraction, base).items():
                errs[key] = float(np.linalg.norm(row * sig_safe))

    if error_oversampling > 1.0:
        # Scalar padding rescaling is heuristic; it does not reconstruct covariance.
        scale = math.sqrt(float(error_oversampling))
        errs = {
            key: (value * scale if np.isfinite(value) else value) for key, value in errs.items()
        }

    if cancel_callback is not None and cancel_callback():
        raise MomentsCancelledError("Spectral moments cancelled.")
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
        computation_revision=computation.COMPUTATION_REVISION,
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
        "computation_revision": moments.computation_revision,
        "parameters": values,
        "uncertainties": errors,
        "moments_recipe": dict(moments.recipe),
        "warnings": [moments.recipe["uncertainty_scope"]]
        if moments.recipe.get("uncertainty_scope")
        else [],
        "run_number": int(run_number),
        "run_label": str(run_label) if run_label is not None else None,
        "field": _coord(field),
        "temperature": _coord(temperature),
    }
