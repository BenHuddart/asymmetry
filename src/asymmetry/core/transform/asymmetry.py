"""Compute the μSR asymmetry from grouped histograms.

The corrected asymmetry is defined as

    A(t) = [N_F(t) − α N_B(t)] / [β N_F(t) + α N_B(t)]

where N_F and N_B are the forward and backward group counts, α is the
count-rate balance (detector efficiencies × solid angles) and β is the
intrinsic-asymmetry balance β = A_{0,B}/A_{0,F} (musrfit asymmetry fit
type 2; β multiplies the *forward* counts in our α-on-backward convention,
which is algebraically identical to musrfit's α-on-forward form with
α_musrfit = 1/α — see docs/porting/beta-correction/). β = 1 recovers the
standard formula A = (F − αB)/(F + αB).

Scale: every asymmetry this module returns is the dimensionless **fraction**
``A ∈ [-1, 1]``, never percent. See
:mod:`asymmetry.core.transform.units` and "Asymmetry units across the API" in the
documentation for the whole picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from asymmetry.core.transform.corrections import CorrectionUnavailableError
from asymmetry.core.transform.fb_counts import (
    BinWindow,
    GroupCounts,
    count_timing,
    good_window,
    linear_combination_variance,
    validated_count_arrays,
)
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def compute_asymmetry(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry and its statistical error.

    Scale
    -----
    This primitive returns the dimensionless **fraction** ``A ∈ [-1, 1]``
    (and a fractional error), matching the textbook definition above. Note
    this differs from the loaded :class:`~asymmetry.core.data.dataset.MuonDataset`
    surface: ``ds.asymmetry`` / ``ds.error`` are on the **percent** scale
    (the loaders multiply this fraction by 100, the WiMDA-style convention the
    time-domain fit models also use). When you need an explicit scale from a
    dataset, use ``ds.asymmetry_fraction`` / ``ds.asymmetry_percent`` rather
    than assuming one — seeding a fit with fraction-scale amplitudes against
    percent-scale data converges to the wrong minimum.

    Parameters
    ----------
    forward, backward
        Counts in the forward and backward detector groups (same length).
    alpha
        Count-rate balance parameter (α).
    beta
        Intrinsic-asymmetry balance parameter (β = A_{0,B}/A_{0,F}); the
        default 1.0 is the standard formula.

    Returns
    -------
    asymmetry, error
        **As a fraction** (``A ∈ [-1, 1]``), with the error on the same scale —
        *not* percent. Arrays of the same length as the inputs. Scale by 100 (or
        :func:`asymmetry.core.transform.units.to_percent`) before handing these to
        anything percent-scale: a
        :class:`~asymmetry.core.data.dataset.MuonDataset`, a fit model's ``A0``,
        :meth:`~asymmetry.core.fitting.FitEngine.fit_arrays`.
    """
    f, b = validated_count_arrays(forward, backward)

    numerator = f - alpha * b
    denominator = beta * f + alpha * b
    # Only divide on non-zero denominator.
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    # Default error 1.0 is a "no information" sentinel used for both the
    # zero-denominator bins and the degenerate one-sided bins below.
    err = np.ones_like(f)

    asym[safe] = numerator[safe] / denominator[safe]

    # First-order Poisson propagation of A = (F - alpha B)/(beta F + alpha B) with
    # var(F) = F, var(B) = B, keeping the num/den covariance:
    #   dA/dF = alpha B (1 + beta)/D^2,  dA/dB = -alpha F (1 + beta)/D^2
    #   var(A) = alpha^2 (1 + beta)^2 F B (F + B) / (beta F + alpha B)^4
    #          = (1 - A^2)/(F + B)        at alpha = beta = 1.
    # At beta = 1 this is the textbook / WiMDA / musrfit result. (The older
    # Mantid AsymmetryCalc model propagated num and den as *independent*,
    # dropping that covariance and over-estimating sigma_A by
    # (1 + A^2)/(1 - A^2); see docs/porting/asymmetry-error-propagation/.
    # musrfit itself weights fits with the *raw* asymmetry error, independent
    # of alpha/beta — a recorded divergence, see docs/porting/beta-correction/.)
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
        scale = abs(float(alpha)) * (1.0 + float(beta))
        err[informative] = scale * np.sqrt(radicand) / (den * den)

    return asym, err


def compute_asymmetry_with_count_errors(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    forward_error: NDArray[np.float64],
    backward_error: NDArray[np.float64],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry from counts with supplied count uncertainties.

    This is used for musrfit-style background-corrected histograms, where the
    count uncertainties are formed before or during background subtraction.
    The asymmetry convention remains Asymmetry's convention, with ``alpha``
    multiplying the backward group (and ``beta`` the forward group in the
    denominator).

    Parameters
    ----------
    forward, backward, forward_error, backward_error
        Counts and their 1σ count uncertainties (count domain — no asymmetry
        scale involved on the way in).
    alpha, beta
        As in :func:`compute_asymmetry`.

    Returns
    -------
    asymmetry, error
        **As a fraction** (``A ∈ [-1, 1]``), exactly as
        :func:`compute_asymmetry` — *not* percent.
    """
    f, b, ef, eb = validated_count_arrays(forward, backward, forward_error, backward_error)
    if np.any(ef < 0) or np.any(eb < 0):
        raise ValueError("count errors must be non-negative.")

    numerator = f - alpha * b
    denominator = beta * f + alpha * b
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    err = np.ones_like(f)

    asym[safe] = numerator[safe] / denominator[safe]
    variance_term = (b * ef) ** 2 + (f * eb) ** 2
    informative = safe & (variance_term > 0.0)
    if np.any(informative):
        den_safe = denominator[informative]
        scale = abs(float(alpha)) * (1.0 + float(beta))
        err[informative] = scale * np.sqrt(variance_term[informative]) / (den_safe * den_safe)

    return asym, err


def slice_to_good_window(
    asymmetry: NDArray[np.float64],
    error: NDArray[np.float64],
    window: BinWindow,
    *,
    common_t0: int,
    bin_width: float,
    t0_time_us: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Slice reduced arrays to a validated inclusive aligned-bin window.

    Bin k is stamped at its centre relative to exact t0, (k + 0.5)·w − t0.
    The bin-centre default preserves the integer-bin axis exactly.
    """
    width, t0 = count_timing(bin_width, common_t0, t0_time_us)
    residual = (common_t0 + 0.5) * width - t0
    selected = asymmetry[window.slice]
    time = (np.arange(selected.size, dtype=float) + window.first - common_t0) * width + residual
    return time, selected, error[window.slice]


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
        length of the arrays is used.

    Returns
    -------
    float
        Finite positive estimated alpha value.

    Raises
    ------
    CorrectionUnavailableError
        No bins are available, either integral is nonpositive, or their ratio
        is not finite and positive. A failed estimate has no default coefficient.
    """
    f, b = validated_count_arrays(forward, backward)
    if f.size == 0:
        raise CorrectionUnavailableError("Automatic alpha requires a nonempty good-bin window.")
    window = good_window(
        {
            key: value
            for key, value in (("first_good_bin", first_good_bin), ("last_good_bin", last_good_bin))
            if value is not None
        },
        f.size,
    )
    fs = float(np.sum(f[window.slice]))
    bs = float(np.sum(b[window.slice]))
    if not np.isfinite(fs) or not np.isfinite(bs) or fs <= 0.0 or bs <= 0.0:
        raise CorrectionUnavailableError("Automatic alpha requires finite positive F/B integrals.")
    alpha = fs / bs
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise CorrectionUnavailableError("Automatic alpha requires a finite positive F/B ratio.")
    return alpha


ALPHA_ESTIMATION_METHODS = ("diamagnetic", "general", "ratio")

# Search bounds for the continuous optimiser, expressed on ln(alpha) so the
# bounds are symmetric about alpha = 1. WiMDA's grid walk clamps trial alpha
# to [0.1, 10]; a wider window costs nothing with a bounded scalar minimiser.
_LN_ALPHA_BOUNDS = (float(np.log(0.01)), float(np.log(100.0)))


@dataclass(frozen=True)
class AlphaEstimate:
    """Result of an alpha estimation.

    ``alpha_error`` is absent for the quick diamagnetic and General estimates.
    ``ratio`` reports first-order propagation of the detector-group sums.
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
    """Weighted asymmetry power Σ(A_i / σ_i)² with first-order Poisson errors.

    This quick TF objective can retain finite-window and amplitude-dependent bias;
    its minimum need not equal the detector efficiency ratio.
    """
    asym = (f - alpha * b) / (f + alpha * b)
    err = 2.0 * alpha * np.sqrt(f * b * (f + b)) / np.square(f + alpha * b)
    return float(np.sum(np.square(asym / err)))


def _general_two_window_alpha(
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    time_us: NDArray[np.float64],
) -> float | None:
    """Balance early/late lifetime-corrected densities at beta = 1.

    For F = α₀ C exp(-t/τ)(1 + P), B = C exp(-t/τ)(1 - P),
    equating A₁ + α B₁ = A₂ + α B₂ yields α = (A₁-A₂)/(B₂-B₁).
    Recovery requires different weighted mean P in the two windows. A ratio
    of linear sums is nonlinear and need not be statistically unbiased.
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

    def window_density(values: NDArray[np.float64], sl: slice) -> tuple[float, float]:
        summands = weight[sl] * values[sl] * corrected[sl]
        normalization = float(np.sum(weight[sl]))
        return (
            float(np.sum(summands) / normalization),
            float(np.sum(np.abs(summands)) / normalization),
        )

    early = slice(0, split)
    late = slice(split, None)
    a1, scale_a1 = window_density(f, early)
    a2, scale_a2 = window_density(f, late)
    b1, scale_b1 = window_density(b, early)
    b2, scale_b2 = window_density(b, late)
    denominator = b2 - b1
    # Subtracting nearly equal densities loses relative precision. The γ_n
    # bound nε/(1−nε) scales with absolute summands, including signed-count
    # cancellation; eight further operations cover weighting, normalisation,
    # and density subtraction. This is not a statistical significance test.
    n_epsilon = (f.size + 8) * np.finfo(np.float64).eps
    rounding = n_epsilon / (1.0 - n_epsilon)
    if abs(a1 - a2) <= rounding * (scale_a1 + scale_a2) or abs(denominator) <= rounding * (
        scale_b1 + scale_b2
    ):
        return None
    alpha = (a1 - a2) / denominator
    if not np.isfinite(alpha) or alpha <= 0.0:
        return None
    return float(alpha)


# The two-window equation can absorb a lifetime mismatch into alpha. Fitting
# ln N against {1, t, ∂ln N/∂ln α} re-frees alpha so the residual rate measures
# the effective-lifetime deviation through s = 1/τ_mu − 1/τ_eff.

#: Sub-windows the gate splits the estimation window into (equal combined
#: counts, so every sub-window carries comparable Poisson error), and the
#: minimum needed for the three-parameter fit to have any degrees of freedom.
_GENERAL_GATE_WINDOWS = 8
_GENERAL_GATE_MIN_WINDOWS = 5

#: Gate tolerances. The effective lifetime of the balanced count must agree with
#: τ_μ to within :data:`_GENERAL_GATE_TAU_TOLERANCE` (a fraction), *and* any
#: larger deviation must be resolved at :data:`_GENERAL_GATE_SIGMA` standard
#: errors before the estimate is rejected. Both clauses are needed: the relative
#: clause alone would fail clean low-statistics data, whose fitted lifetime
#: scatters by more than a per cent; the significance clause alone would reject
#: high-statistics data over a lifetime error far too small to matter.
#:
#: 0.15% / 2.5σ come from a synthetic Poisson study over 60 seeds per operating
#: point (relaxing LF polarization, alpha = 1.37, asymmetry 5–50%, relaxation
#: rate 0.1–3 µs⁻¹, 2·10⁵ to 2·10⁷ counts), the headline cases pinned by
#: ``tests/core/test_alpha_estimation.py``:
#:
#: * a 0.5% lifetime mismatch — which drives the estimate ~10% wrong at 20%
#:   asymmetry and ~70% wrong at 5% — is flagged in 54/60 datasets at 2·10⁷
#:   counts;
#: * a 1% mismatch (~22% wrong) is flagged in 60/60 at 2·10⁷;
#: * clean data is flagged in 0–2/60 across every configuration tested, i.e. a
#:   few-per-cent conservative false alarm — which reports a *quantified* small
#:   deviation the reader can weigh, where a miss is silently wrong.
#:
#: The significance clause does most of the work: on the synthetic grid a 0.20%
#: and a 0.15% tolerance give identical verdicts at every point, because a defect
#: resolvable at 2.5σ is almost always well past either. 0.15% is nevertheless
#: the chosen value — on real continuous-source data a genuinely defective run
#: landed inside that band, which the synthetic grid never sampled, and 0.15%
#: costs nothing where the two agree. 2.5σ rather than 3σ is likewise deliberate:
#: 3σ gives up a sixth of the detections at 2·10⁷ counts (48/60 instead of 54/60)
#: to halve an already-small false-alarm rate.
_GENERAL_GATE_TAU_TOLERANCE = 0.0015
_GENERAL_GATE_SIGMA = 2.5


@dataclass(frozen=True)
class _GeneralFlatness:
    """Flatness diagnostics of the lifetime-corrected balanced count."""

    n_windows: int
    #: Flatness χ² of the sub-window levels about their inverse-variance-weighted
    #: mean — the ``general`` analogue of the diamagnetic objective.
    chi2: float
    #: Effective-lifetime deviation from τ_μ implied by the residual trend once
    #: alpha is re-freed, as a fraction (``tau_eff / tau_mu - 1``), and its
    #: standard error.
    tau_deviation: float
    tau_deviation_error: float


def _equal_count_edges(combined: NDArray[np.float64], n_windows: int) -> NDArray[np.int64]:
    """Sub-window edges placed at equal cumulative combined counts."""
    total = float(np.sum(combined))
    cum = np.cumsum(combined)
    targets = (np.arange(1, n_windows) * total) / n_windows
    edges = np.concatenate(([0], np.searchsorted(cum, targets) + 1, [combined.size]))
    return np.unique(edges)


def _general_flatness(
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    time_us: NDArray[np.float64],
    alpha: float,
) -> _GeneralFlatness | None:
    """Flatness of ``(F/√α + B√α)·exp(t/τ_μ)`` across equal-statistics windows.

    Returns ``None`` when the window cannot be split into at least
    :data:`_GENERAL_GATE_MIN_WINDOWS` usable sub-windows (the gate then makes no
    statement rather than a noise-driven one).

    Each sub-window contributes the same inverse-variance-weighted density the
    solver uses — weights ``exp(-2t/τ_μ)``, the variance scale of a corrected
    count — with its Poisson error propagated from the raw counts. ``ln N`` is
    then fitted against ``{1, t, ∂ln N/∂alpha}``; the ``t`` coefficient is the
    residual rate ``s = 1/τ_μ − 1/τ_eff`` that alpha could *not* absorb, which
    inverts to the effective lifetime.
    """
    root_alpha = float(np.sqrt(alpha))
    combined = np.clip(f, 0.0, None) + np.clip(b, 0.0, None)
    if float(np.sum(combined)) <= 0.0:
        return None
    edges = _equal_count_edges(combined, _GENERAL_GATE_WINDOWS)
    if edges.size - 1 < _GENERAL_GATE_MIN_WINDOWS:
        return None

    weight = np.exp(-2.0 * time_us / MUON_LIFETIME_US)
    corrected = np.exp(time_us / MUON_LIFETIME_US)
    balanced = f / root_alpha + b * root_alpha
    # d/dln(alpha) of the balanced count: (B√α − F/√α)/2.
    d_balanced = 0.5 * (b * root_alpha - f / root_alpha)
    # var(F/√α + B√α) = F/α + Bα under Poisson counts.
    variance = np.clip(f, 0.0, None) / alpha + np.clip(b, 0.0, None) * alpha

    levels: list[float] = []
    errors: list[float] = []
    times: list[float] = []
    derivatives: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        w = weight[lo:hi]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        wc = w * corrected[lo:hi]
        level = float(np.sum(wc * balanced[lo:hi]) / w_sum)
        var = float(np.sum(np.square(wc) * variance[lo:hi])) / (w_sum * w_sum)
        if level <= 0.0 or var <= 0.0:
            continue
        levels.append(level)
        errors.append(float(np.sqrt(var)))
        times.append(float(np.sum(w * time_us[lo:hi]) / w_sum))
        derivatives.append(float(np.sum(wc * d_balanced[lo:hi]) / w_sum) / level)
    if len(levels) < _GENERAL_GATE_MIN_WINDOWS:
        return None

    n = np.asarray(levels, dtype=np.float64)
    sigma = np.asarray(errors, dtype=np.float64)
    # Flatness chi-square about the inverse-variance-weighted mean level.
    inverse_variance = 1.0 / np.square(sigma)
    mean_level = float(np.sum(inverse_variance * n) / np.sum(inverse_variance))
    chi2 = float(np.sum(np.square((n - mean_level) / sigma)))

    # Weighted three-parameter fit of ln N on {1, t, dlnN/dln(alpha)}:
    # sigma(ln N) = sigma(N)/N, and freeing alpha is what stops the two-window
    # condition from cancelling the lifetime term.
    design = np.column_stack(
        [np.ones_like(n), np.asarray(times, dtype=np.float64), np.asarray(derivatives)]
    )
    w_fit = np.square(n / sigma)
    normal = design.T @ (w_fit[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    solution = covariance @ (design.T @ (w_fit * np.log(n)))
    slope = float(solution[1])
    slope_variance = float(covariance[1, 1])
    if not np.isfinite(slope) or not np.isfinite(slope_variance) or slope_variance <= 0.0:
        return None
    slope_error = float(np.sqrt(slope_variance))

    # N ∝ exp(t·(1/τ_μ − 1/τ_eff)) ⇒ τ_eff/τ_μ = 1/(1 − s·τ_μ).
    scaled = slope * MUON_LIFETIME_US
    if scaled >= 1.0:
        return None
    deviation = scaled / (1.0 - scaled)
    deviation_error = slope_error * MUON_LIFETIME_US / (1.0 - scaled) ** 2
    return _GeneralFlatness(
        n_windows=len(levels),
        chi2=chi2,
        tau_deviation=float(deviation),
        tau_deviation_error=float(deviation_error),
    )


def _alpha_window(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    time_us: NDArray[np.float64] | None,
    first_good_bin: int | None,
    last_good_bin: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Slice counts (and optional times) to the good-bin window."""
    arrays = (forward, backward) if time_us is None else (forward, backward, time_us)
    values = validated_count_arrays(*arrays)
    if values[0].size == 0:
        return values[0], values[1], None if time_us is None else values[2]
    window = good_window(
        {
            key: value
            for key, value in (("first_good_bin", first_good_bin), ("last_good_bin", last_good_bin))
            if value is not None
        },
        values[0].size,
    )
    return (
        values[0][window.slice],
        values[1][window.slice],
        (None if time_us is None else values[2][window.slice]),
    )


# The optimising estimators run on internally packed bins so the objective's
# Poisson noise floor does not swamp the alpha-mismatch signal on fine raw
# binning (WiMDA runs on the live display bins, which are already bunched;
# packing deterministically from the data keeps the result independent of
# display settings). Target combined counts per packed
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


def _single_alpha_estimate(
    method: str,
    f: NDArray[np.float64],
    b: NDArray[np.float64],
    t: NDArray[np.float64] | None,
) -> float | None:
    """One point estimate on method-specific prepared counts."""
    if f.size == 0:
        return None
    if method == "ratio":
        bs = float(np.sum(b))
        return float(np.sum(f)) / bs if bs > 0.0 else None
    if method == "diamagnetic":
        result = minimize_scalar(
            lambda ln_alpha: _diamagnetic_objective(float(np.exp(ln_alpha)), f, b),
            bounds=_LN_ALPHA_BOUNDS,
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(np.exp(result.x))
    return _general_two_window_alpha(f, b, t)


def _ratio_alpha_error(
    alpha: float,
    forward: GroupCounts,
    backward: GroupCounts,
    edges: NDArray[np.intp],
) -> float | None:
    """First-order ratio propagation, including within-group shared noise."""
    window = slice(int(edges[0]), int(edges[-1]))
    sum_f = float(np.sum(forward.counts[window]))
    sum_b = float(np.sum(backward.counts[window]))
    if sum_f <= 0.0 or sum_b <= 0.0:
        return None
    var_f = linear_combination_variance(forward, edges)[0]
    var_b = linear_combination_variance(backward, edges)[0]
    return float(alpha * np.sqrt(var_f / sum_f**2 + var_b / sum_b**2))


def estimate_alpha_detailed(
    forward: GroupCounts,
    backward: GroupCounts,
    *,
    method: str = "diamagnetic",
    time_us: NDArray[np.float64] | None = None,
    first_good_bin: int | None = None,
    last_good_bin: int | None = None,
) -> AlphaEstimate:
    """Estimate alpha using a quick balance or the integral ratio.

    ``diamagnetic`` minimises weighted asymmetry power on internally packed,
    positive bins. ``general`` equates early/late lifetime-corrected densities
    at beta = 1 and requires ``time_us`` in microseconds relative to t0.
    Both return ``alpha_error=None``. ``ratio`` uses the raw window sums and
    reports first-order uncertainty, including supplied shared background terms.

    General rejects rounding-level window contrast. Its existing diagnostic
    re-frees alpha across eight sub-windows to check the lifetime assumption:
    a deviation exceeding both 0.15% and 2.5 standard errors sets ``ok=False``.
    This diagnostic uses a Poisson count approximation and does not certify
    identifiability, corrected-data uncertainty, or noisy weak-contrast recovery.
    ``objective_value`` is the flatness chi-square, or weighted asymmetry power
    for Diamagnetic; it is absent for the ratio. See the alpha calibration
    reference for assumptions and diagnostic limits.
    """
    if method not in ALPHA_ESTIMATION_METHODS:
        raise ValueError(
            f"Unknown alpha estimation method {method!r}; "
            f"expected one of {ALPHA_ESTIMATION_METHODS}"
        )
    if method == "general" and time_us is None:
        raise ValueError("The 'general' method requires time_us (bin centres in µs)")
    f, b, t = _alpha_window(forward.counts, backward.counts, time_us, first_good_bin, last_good_bin)
    if method == "diamagnetic":
        # The diamagnetic objective runs on packed, positive-masked bins;
        # ratio and general use the raw window.
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
                "— the General method needs different window polarizations"
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
    flatness: _GeneralFlatness | None = None
    if method == "diamagnetic":
        objective_value = _diamagnetic_objective(alpha, f, b)
    elif method == "general" and t is not None:
        flatness = _general_flatness(f, b, t, alpha)
        if flatness is not None:
            objective_value = flatness.chi2

    alpha_error: float | None = None
    message = ""
    ok = True
    if method == "general":
        if flatness is None:
            message = (
                "Flatness gate not applied: need at least "
                f"{_GENERAL_GATE_MIN_WINDOWS} usable sub-windows and a resolved lifetime fit; "
                "the General estimate is unverified"
            )
        elif (
            abs(flatness.tau_deviation) > _GENERAL_GATE_TAU_TOLERANCE
            and abs(flatness.tau_deviation) > _GENERAL_GATE_SIGMA * flatness.tau_deviation_error
        ):
            ok = False
            message = (
                "Balanced count is not flat after lifetime correction; effective "
                f"lifetime differs from tau_mu by {100.0 * flatness.tau_deviation:+.2f}% "
                f"(+/- {100.0 * flatness.tau_deviation_error:.2f}%) — the 'general' "
                "method is unreliable on this data; consider 'ratio' on a "
                "fully-depolarized window"
            )
    if method == "ratio":
        edges = np.array(
            [
                0 if first_good_bin is None else first_good_bin,
                forward.counts.size if last_good_bin is None else last_good_bin + 1,
            ],
            dtype=np.intp,
        )
        alpha_error = _ratio_alpha_error(alpha, forward, backward, edges)

    return AlphaEstimate(
        alpha=float(alpha),
        alpha_error=alpha_error,
        method=method,
        n_bins_used=n_bins,
        objective_value=objective_value,
        ok=ok,
        message=message,
    )
