"""Compute the μSR asymmetry from grouped histograms.

The standard asymmetry is defined as

    A(t) = [N_F(t) − α N_B(t)] / [N_F(t) + α N_B(t)]

where N_F and N_B are the forward and backward group counts and α is
the balance parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from asymmetry.core.utils.constants import MUON_LIFETIME_US


def compute_asymmetry(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    alpha: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry and its statistical error.

    Parameters
    ----------
    forward, backward
        Counts in the forward and backward detector groups (same length).
    alpha
        Balance parameter (α).

    Returns
    -------
    asymmetry, error
        Arrays of the same length as the inputs.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)

    numerator = f - alpha * b
    denominator = f + alpha * b
    # Only divide on non-zero denominator.
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    # Default error 1.0 is a "no information" sentinel used for both the
    # zero-denominator bins and the degenerate one-sided bins below.
    err = np.ones_like(f)

    asym[safe] = numerator[safe] / denominator[safe]

    # Exact Poisson propagation of A = (F - alpha B)/(F + alpha B) with
    # var(F) = F, var(B) = B, keeping the cov(num, den) = F - alpha^2 B term:
    #   var(A) = 4 alpha^2 F B (F + B) / (F + alpha B)^4
    #          = (1 - A^2)/(F + B)        at alpha = 1.
    # This is the textbook / WiMDA / musrfit result. (The older Mantid
    # AsymmetryCalc model propagated num and den as *independent*, dropping
    # that covariance and over-estimating sigma_A by (1 + A^2)/(1 - A^2);
    # see docs/porting/asymmetry-error-propagation/.)
    #
    # One-sided bins (F * B == 0) have a degenerate zero first-order variance
    # (A is pinned to +/-1), which is useless as a fit weight, so they keep
    # the 1.0 sentinel rather than a zero error.
    informative = safe & (f * b > 0.0)
    if np.any(informative):
        den = denominator[informative]
        fi = f[informative]
        bi = b[informative]
        # np.maximum guards against negative radicands from out-of-contract
        # (e.g. background-subtracted) counts, matching the clamp in
        # compute_asymmetry_with_count_errors.
        radicand = np.maximum(fi * bi * (fi + bi), 0.0)
        err[informative] = 2.0 * abs(float(alpha)) * np.sqrt(radicand) / (den * den)

    return asym, err


def compute_asymmetry_with_count_errors(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    forward_error: NDArray[np.float64],
    backward_error: NDArray[np.float64],
    alpha: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry from counts with supplied count uncertainties.

    This is used for musrfit-style background-corrected histograms, where the
    count uncertainties are formed before or during background subtraction.
    The asymmetry convention remains Asymmetry's convention, with ``alpha``
    multiplying the backward group.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    ef = np.asarray(forward_error, dtype=np.float64)
    eb = np.asarray(backward_error, dtype=np.float64)

    n = min(f.size, b.size, ef.size, eb.size)
    f = f[:n]
    b = b[:n]
    ef = ef[:n]
    eb = eb[:n]

    numerator = f - alpha * b
    denominator = f + alpha * b
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    err = np.ones_like(f)

    asym[safe] = numerator[safe] / denominator[safe]
    if np.any(safe):
        den_safe = denominator[safe]
        variance_term = (b[safe] * ef[safe]) ** 2 + (f[safe] * eb[safe]) ** 2
        err[safe] = (
            2.0
            * abs(float(alpha))
            * np.sqrt(np.maximum(variance_term, 0.0))
            / (den_safe * den_safe)
        )

    return asym, err


def slice_to_good_window(
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    grouping: dict,
    *,
    common_t0: int,
    bin_width: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Slice reduced arrays to the grouping's good-bin window and build the axis.

    The shared tail of every F-B reduction: clip ``asymmetry``/``error`` to the
    inclusive ``[first_good_bin, last_good_bin]`` window (falling back to the
    full range when the window is degenerate) and form the time axis in
    microseconds, measured from ``common_t0``. Used by both the loader-style
    reduction (:func:`asymmetry.core.simulate._reduce_histograms`) and the
    run-arithmetic subtraction reduction so the two agree by construction.
    """
    size = int(asymmetry.size)
    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", size - 1))
    except (TypeError, ValueError):
        last_good = size - 1
    last_good = min(last_good, size - 1)
    if first_good > last_good:
        first_good, last_good = 0, size - 1

    asymmetry = asymmetry[first_good : last_good + 1]
    error = error[first_good : last_good + 1]
    time = (np.arange(asymmetry.size, dtype=float) + first_good - int(common_t0)) * float(
        bin_width
    )
    return time, asymmetry, error


def estimate_alpha(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    first_good_bin: int | None = None,
    last_good_bin: int | None = None,
) -> float:
    r"""Estimate the detector-balance parameter ``alpha`` from grouped counts.

    This follows the same approach used by Mantid's ``AlphaCalc`` algorithm:

    .. math::

        \alpha = \frac{\sum_i F_i}{\sum_i B_i}

    where :math:`F_i` and :math:`B_i` are forward and backward grouped counts
    integrated over the selected good-bin window.

    Parameters
    ----------
    forward, backward
        Forward and backward grouped count arrays.
    first_good_bin, last_good_bin
        Optional inclusive bin range for integration. If omitted, the full
        overlap of the two arrays is used.

    Returns
    -------
    float
        Estimated alpha value. Returns ``1.0`` when the backward integral is
        not positive or when no valid bins are available.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    n = min(len(f), len(b))
    if n <= 0:
        return 1.0

    lo = 0 if first_good_bin is None else max(0, int(first_good_bin))
    hi = n - 1 if last_good_bin is None else min(n - 1, int(last_good_bin))
    if lo > hi:
        return 1.0

    fs = float(np.sum(f[lo : hi + 1]))
    bs = float(np.sum(b[lo : hi + 1]))
    if bs <= 0.0:
        return 1.0
    return fs / bs


ALPHA_ESTIMATION_METHODS = ("diamagnetic", "general", "ratio")

# Search bounds for the continuous optimiser, expressed on ln(alpha) so the
# bounds are symmetric about alpha = 1. WiMDA's grid walk clamps trial alpha
# to [0.1, 10]; a wider window costs nothing with a bounded scalar minimiser.
_LN_ALPHA_BOUNDS = (float(np.log(0.01)), float(np.log(100.0)))

# Bootstrap replicas sit close to the base estimate, so the diamagnetic
# minimiser searches only a narrow ln window around it with a relaxed tolerance
# — far cheaper than a full bounded minimisation per replica (~200×).
_BOOTSTRAP_LN_WINDOW = 1.0
_BOOTSTRAP_XATOL = 1e-4


@dataclass(frozen=True)
class AlphaEstimate:
    """Result of an alpha estimation.

    ``alpha_error`` is a seeded Poisson-bootstrap standard deviation (``None``
    when bootstrapping was disabled or too few replicas survived).
    ``objective_value`` is the minimised objective for the optimising methods
    and ``None`` for the ``ratio`` method.
    """

    alpha: float
    alpha_error: float | None
    method: str
    n_bins_used: int
    objective_value: float | None
    ok: bool
    message: str = ""


def _diamagnetic_objective(alpha: float, f: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """WiMDA's diamagnetic objective: Σ (A_i / σ_i)².

    Transcribed from ``Group.pas EstimateButtonClick`` (method = diamag).
    The per-bin error is the exact Poisson propagation of the asymmetry,
    σ = 2α√(fb(f+b)) / (f+αb)², written in the source as
    ``2 a (f/b) sqrt(1/f + 1/b) / (f/b + a)²``. On a transverse-field run
    the oscillation is symmetric about zero exactly when alpha balances the
    detector efficiencies, which minimises this weighted asymmetry power.
    """
    asym = (f - alpha * b) / (f + alpha * b)
    err = 2.0 * alpha * np.sqrt(f * b * (f + b)) / np.square(f + alpha * b)
    return float(np.sum(np.square(asym / err)))


def _general_two_window_alpha(
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    time_us: NDArray[np.float64],
) -> float | None:
    """General-method alpha: flatness of the lifetime-corrected balance.

    The combination N(α) = (F/√α + B√α)·exp(t/τ_μ) is flat in time exactly
    when alpha equals the true efficiency ratio — the polarization term
    cancels for any P(t) — which is the principle of WiMDA's "general"
    estimate (``Group.pas EstimateButtonClick``). WiMDA minimises a weighted
    relative-scatter functional of N; that functional has no interior
    minimum at realistic statistics (study divergence D14), so Asymmetry
    solves the equivalent flatness condition in closed form instead.

    Equating the mean lifetime-corrected count density between two
    equal-statistics time windows W₁, W₂ gives

        A₁/√α + B₁√α = A₂/√α + B₂√α  ⇒  α = (A₁ − A₂)/(B₂ − B₁)

    where A_k, B_k are smoothly weighted window means of F·exp(t/τ_μ) and
    B·exp(t/τ_μ) (weights exp(−2t/τ_μ), the inverse variance scale of a
    corrected count). Everything is linear in the counts, so the estimate
    is unbiased; it requires the polarization to *relax* between the
    windows — on non-relaxing data the denominator is consistent with zero
    and the method reports failure rather than a number.
    """
    if f.size < 4:
        return None
    combined = np.clip(f, 0.0, None) + np.clip(b, 0.0, None)
    total = float(np.sum(combined))
    if total <= 0.0:
        return None
    cum = np.cumsum(combined)
    split = int(np.searchsorted(cum, total / 2.0)) + 1
    if split < 2 or split > f.size - 2:
        split = f.size // 2
    weight = np.exp(-2.0 * time_us / MUON_LIFETIME_US)
    corrected = np.exp(time_us / MUON_LIFETIME_US)

    def window_density(values: NDArray[np.float64], sl: slice) -> float:
        return float(np.sum(weight[sl] * values[sl] * corrected[sl]) / np.sum(weight[sl]))

    early = slice(0, split)
    late = slice(split, None)
    a1 = window_density(f, early)
    a2 = window_density(f, late)
    b1 = window_density(b, early)
    b2 = window_density(b, late)
    denominator = b2 - b1
    if denominator == 0.0:
        return None
    alpha = (a1 - a2) / denominator
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    return float(alpha)


def _alpha_window(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    time_us: NDArray[np.float64] | None,
    first_good_bin: int | None,
    last_good_bin: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Slice counts (and optional times) to the good-bin window."""
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    n = min(f.size, b.size)
    if time_us is not None:
        t = np.asarray(time_us, dtype=np.float64)
        n = min(n, t.size)
    lo = 0 if first_good_bin is None else max(0, int(first_good_bin))
    hi = n - 1 if last_good_bin is None else min(n - 1, int(last_good_bin))
    if n <= 0 or lo > hi:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, (empty if time_us is not None else None)
    f = f[lo : hi + 1]
    b = b[lo : hi + 1]
    t_sel = None
    if time_us is not None:
        t_sel = np.asarray(time_us, dtype=np.float64)[lo : hi + 1]
    return f, b, t_sel


# The optimising estimators run on internally packed bins so the objective's
# Poisson noise floor does not swamp the alpha-mismatch signal on fine raw
# binning (WiMDA runs on the live display bins, which are already bunched;
# packing deterministically from the data keeps the result independent of
# display settings — study divergence D3). Target combined counts per packed
# bin, and the minimum number of packed bins to keep:
_PACK_TARGET_COUNTS = 200.0
_PACK_MIN_BINS = 10


def _pack_for_estimation(
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    t: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Sum windowed counts into packed bins of ~equal combined statistics.

    Edges are placed where the cumulative combined count crosses multiples of
    the per-bin target, so every packed bin carries roughly the same counts.
    Uniform-width packing would leave near-empty bins in the exponential tail
    whose positive-count selection (E[N | N > 0] ≫ E[N]) biases the
    lifetime-corrected General objective. Packed times are count-weighted
    centroids. A trailing remainder below half the target is dropped.
    """
    n = f.size
    combined = np.clip(f, 0.0, None) + np.clip(b, 0.0, None)
    total = float(np.sum(combined))
    if n == 0 or total <= _PACK_TARGET_COUNTS:
        return f, b, t
    n_out = int(total / _PACK_TARGET_COUNTS)
    n_out = max(_PACK_MIN_BINS, min(n_out, n))
    if n_out >= n:
        return f, b, t
    cum = np.cumsum(combined)
    targets = (np.arange(1, n_out) * total) / n_out
    edges = np.concatenate(([0], np.searchsorted(cum, targets) + 1, [n]))
    edges = np.unique(edges)

    f_packed = np.add.reduceat(f, edges[:-1])
    b_packed = np.add.reduceat(b, edges[:-1])
    if float(f_packed[-1] + b_packed[-1]) < 0.5 * _PACK_TARGET_COUNTS and len(edges) > 2:
        f_packed = f_packed[:-1]
        b_packed = b_packed[:-1]
        edges = edges[:-1]
    t_packed = None
    if t is not None:
        weights = np.add.reduceat(combined, edges[:-1])
        weighted_t = np.add.reduceat(combined * t, edges[:-1])
        plain_t = np.array([float(np.mean(t[lo:hi])) for lo, hi in zip(edges[:-1], edges[1:])])
        with np.errstate(invalid="ignore", divide="ignore"):
            t_packed = np.where(weights > 0.0, weighted_t / np.maximum(weights, 1e-300), plain_t)
    return f_packed, b_packed, t_packed


def _positive_mask(
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    t: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """WiMDA parity: only bins with positive counts in both groups contribute."""
    mask = np.isfinite(f) & np.isfinite(b) & (f > 0.0) & (b > 0.0)
    if t is not None:
        mask &= np.isfinite(t)
        t = t[mask]
    return f[mask], b[mask], t


def _minimise_alpha(
    objective,
    *,
    center: float | None = None,
    xatol: float = 1e-10,
) -> float:
    """Minimise an objective of alpha on ln(alpha) within the search bounds.

    With *center* given (a finite, positive alpha), the bracket is narrowed to
    a small ln window around ``ln(center)`` (clamped to the global bounds) — the
    bootstrap path passes the base estimate so each replica solves a tight,
    relaxed-tolerance problem instead of a full bounded minimisation.
    """
    lo, hi = _LN_ALPHA_BOUNDS
    if center is not None and np.isfinite(center) and center > 0.0:
        c = float(np.log(center))
        lo = max(lo, c - _BOOTSTRAP_LN_WINDOW)
        hi = min(hi, c + _BOOTSTRAP_LN_WINDOW)
    result = minimize_scalar(
        lambda ln_alpha: objective(float(np.exp(ln_alpha))),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": xatol},
    )
    return float(np.exp(result.x))


def _single_alpha_estimate(
    method: str,
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    t: NDArray[np.float64] | None,
    *,
    center: float | None = None,
    xatol: float = 1e-10,
) -> float | None:
    """One alpha estimate on prepared counts; ``None`` when degenerate.

    ``ratio`` and ``general`` expect raw windowed counts (both are linear in
    the data); ``diamagnetic`` expects packed, positive-masked counts.
    ``center``/``xatol`` narrow the diamagnetic minimiser for bootstrap replicas.
    """
    if f.size == 0:
        return None
    if method == "ratio":
        bs = float(np.sum(b))
        return float(np.sum(f)) / bs if bs > 0.0 else None
    if method == "diamagnetic":
        return _minimise_alpha(
            lambda a: _diamagnetic_objective(a, f, b), center=center, xatol=xatol
        )
    return _general_two_window_alpha(f, b, t)


def estimate_alpha_detailed(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    method: str = "diamagnetic",
    time_us: NDArray[np.float64] | None = None,
    first_good_bin: int | None = None,
    last_good_bin: int | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> AlphaEstimate:
    """Estimate alpha with an uncertainty, by one of three methods.

    Methods (see ``docs/user_guide/data_reduction/alpha_calibration``):

    - ``"diamagnetic"`` — minimise the weighted asymmetry power Σ(A/σ)² over
      a transverse-field calibration run (WiMDA's diamagnetic estimate, run
      on internally packed equal-statistics bins).
    - ``"general"`` — flatness of the lifetime-corrected balanced count
      (F/√α + B√α)·exp(t/τ_μ), solved in closed form between two
      equal-statistics time windows; works on *relaxing* LF/ZF data, where
      no zero-mean oscillation exists, and fails informatively when the
      polarization does not relax. Requires ``time_us`` (bin centres
      relative to t0, microseconds).
    - ``"ratio"`` — ΣF/ΣB (Mantid ``AlphaCalc``; the legacy
      :func:`estimate_alpha`). Only unbiased when the polarization
      integrates to zero over the window (many-cycle TF data).

    The uncertainty is a Poisson bootstrap: per-bin counts are resampled as
    Poisson(observed) ``n_bootstrap`` times with a seeded generator and the
    estimator re-run; ``alpha_error`` is the robust (percentile) standard
    error of the replicas (``None`` when ``n_bootstrap`` is 0 or fewer than
    10 replicas survive). WiMDA reports a bare number; the uncertainty is an
    Asymmetry improvement (study divergence D2).

    Parameters mirror :func:`estimate_alpha`.
    """
    if method not in ALPHA_ESTIMATION_METHODS:
        raise ValueError(
            f"Unknown alpha estimation method {method!r}; "
            f"expected one of {ALPHA_ESTIMATION_METHODS}"
        )
    if method == "general" and time_us is None:
        raise ValueError("The 'general' method requires time_us (bin centres in µs)")

    f, b, t = _alpha_window(forward, backward, time_us, first_good_bin, last_good_bin)
    if method == "diamagnetic":
        # The diamagnetic objective runs on packed, positive-masked bins;
        # ratio and general are linear in counts and use the raw window.
        f, b, t = _pack_for_estimation(f, b, t)
        f, b, t = _positive_mask(f, b, t)
    n_bins = int(f.size)
    if n_bins == 0 or (method == "ratio" and float(np.sum(b)) <= 0.0):
        return AlphaEstimate(
            alpha=1.0,
            alpha_error=None,
            method=method,
            n_bins_used=0,
            objective_value=None,
            ok=False,
            message="No usable bins (need positive counts in both groups)",
        )

    alpha = _single_alpha_estimate(method, f, b, t)
    if alpha is None:
        message = "Estimate is degenerate on this data"
        if method == "general":
            message = (
                "No polarization contrast between the early and late windows "
                "— the General method needs visibly relaxing data"
            )
        return AlphaEstimate(
            alpha=1.0,
            alpha_error=None,
            method=method,
            n_bins_used=n_bins,
            objective_value=None,
            ok=False,
            message=message,
        )

    objective_value: float | None = None
    if method == "diamagnetic":
        objective_value = _diamagnetic_objective(alpha, f, b)

    alpha_error: float | None = None
    message = ""
    ok = True
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        f_lam = np.clip(f, 0.0, None)
        b_lam = np.clip(b, 0.0, None)
        replicas: list[float] = []
        if method == "ratio":
            # The ratio only sees the two window sums, and a sum of
            # independent Poisson counts is Poisson(sum of means) — draw the
            # totals directly instead of full per-bin replicas.
            totals_f = rng.poisson(float(np.sum(f_lam)), int(n_bootstrap)).astype(np.float64)
            totals_b = rng.poisson(float(np.sum(b_lam)), int(n_bootstrap)).astype(np.float64)
            replicas = [float(tf / tb) for tf, tb in zip(totals_f, totals_b) if tb > 0.0]
        else:
            for _ in range(int(n_bootstrap)):
                fr = rng.poisson(f_lam).astype(np.float64)
                br = rng.poisson(b_lam).astype(np.float64)
                if method == "diamagnetic":
                    fr, br, t_rep = _positive_mask(fr, br, t)
                    replica = _single_alpha_estimate(
                        method, fr, br, t_rep, center=alpha, xatol=_BOOTSTRAP_XATOL
                    )
                else:
                    replica = _single_alpha_estimate(method, fr, br, t)
                if replica is not None and np.isfinite(replica):
                    replicas.append(replica)
        if len(replicas) >= 10:
            # Robust standard error: half-width of the central 68.27%
            # interval — equals the standard deviation for well-behaved
            # replicas, but is not blown up by the heavy tails the General
            # estimator develops near its identifiability limit.
            lo_q, hi_q = np.percentile(replicas, [15.865, 84.135])
            alpha_error = float((hi_q - lo_q) / 2.0)
        if method == "general" and len(replicas) < 0.9 * int(n_bootstrap):
            ok = False
            message = (
                "Polarization contrast is marginal — the General estimate is "
                "unreliable on this data "
                f"({int(n_bootstrap) - len(replicas)}/{int(n_bootstrap)} "
                "bootstrap replicas failed)"
            )

    return AlphaEstimate(
        alpha=float(alpha),
        alpha_error=alpha_error,
        method=method,
        n_bins_used=n_bins,
        objective_value=objective_value,
        ok=ok,
        message=message,
    )
