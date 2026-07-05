"""Non-exhaustive role-search heuristics for the global-fit wizard (PR 4).

These are the *pure* kernels behind the Low/Balanced effort tiers — the parts
that reason over already-fitted per-dataset estimates without touching Minuit or
any GUI. Keeping them here (rather than inline in ``global_fit_wizard``) makes
them directly unit-testable and keeps ``asymmetry.core`` GUI-free.

Two techniques live here:

* **Homogeneity (Q) pre-tests (technique E).** For each promotable parameter,
  the per-dataset estimates :math:`\\theta_g` with errors :math:`\\sigma_g` are
  combined into a Cochran homogeneity statistic
  :math:`Q = \\sum_g (\\theta_g - \\bar\\theta)^2 / \\sigma_g^2`, which under the
  null "this parameter is global (constant across the series)" is distributed as
  :math:`\\chi^2_{G-1}`. A clearly-inhomogeneous parameter (tiny upper-tail
  p-value) is pre-fixed *local*; a clearly-homogeneous one (large p-value —
  i.e. small lower-tail exceedance) is pre-fixed *global*; the ambiguous middle
  is enumerated by the caller. The two band edges are the effort knob.

* **Wald quadratic surrogate (technique G).** From one all-local joint fit the
  per-dataset estimates + (diagonal) errors give, in closed form, the predicted
  :math:`\\Delta\\chi^2` of collapsing a parameter's per-dataset values onto a
  single shared (GLS-weighted) value. Summed over a globalised subset this is a
  microsecond surrogate for the real fit's IC, used to *rank* subsets so only the
  top-K are actually fitted.

The chi-square tail probabilities use only :mod:`math` (a regularised
upper-incomplete gamma via a continued fraction / series split) so the module
has no SciPy dependency and stays import-light.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "ParameterHomogeneity",
    "chi2_sf",
    "homogeneity_statistic",
    "classify_parameter_homogeneity",
    "wald_globalisation_cost",
    "wald_subset_delta_chi2",
]


# --------------------------------------------------------------------------- #
# Chi-square upper-tail probability (no SciPy)
# --------------------------------------------------------------------------- #


def _lower_gamma_regularised(s: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(s, x) = γ(s, x) / Γ(s)``.

    Series expansion, valid and fast for ``x < s + 1``; the caller routes larger
    ``x`` through the continued fraction for the *upper* tail instead.
    """

    if x <= 0.0:
        return 0.0
    ap = s
    total = 1.0 / s
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + s * math.log(x) - math.lgamma(s))


def _upper_gamma_regularised(s: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(s, x) = Γ(s, x) / Γ(s)``.

    Lentz continued fraction, valid and fast for ``x >= s + 1``.
    """

    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def chi2_sf(statistic: float, dof: int) -> float:
    """Upper-tail survival function ``P(χ²_dof > statistic)``.

    Returns a probability in ``[0, 1]``. ``dof <= 0`` returns ``1.0`` (the null
    is untestable with no degrees of freedom, so nothing is "clearly" anything).
    """

    if dof <= 0:
        return 1.0
    if statistic <= 0.0:
        return 1.0
    if not math.isfinite(statistic):
        return 0.0
    s = 0.5 * dof
    x = 0.5 * statistic
    if x < s + 1.0:
        return max(0.0, min(1.0, 1.0 - _lower_gamma_regularised(s, x)))
    return max(0.0, min(1.0, _upper_gamma_regularised(s, x)))


# --------------------------------------------------------------------------- #
# Homogeneity (Q) statistic + classification (technique E)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParameterHomogeneity:
    """Homogeneity verdict for one promotable parameter.

    ``role`` is ``"global"`` (clearly homogeneous — pre-fix shared),
    ``"local"`` (clearly inhomogeneous — pre-fix free), or ``"ambiguous"``
    (enumerate). ``skipped`` is ``True`` when the test could not be run (too few
    valid estimates, an at-limit estimate, or a non-finite error) — such a
    parameter is *always* treated as ambiguous, never pre-fixed.
    """

    name: str
    role: str
    q_statistic: float
    dof: int
    p_value: float
    skipped: bool
    reason: str


def homogeneity_statistic(
    values: Sequence[float],
    errors: Sequence[float],
) -> tuple[float, int]:
    """Return ``(Q, dof)`` for the Cochran homogeneity statistic.

    ``Q = Σ_g (θ_g − θ̄)² / σ_g²`` with the inverse-variance-weighted mean
    ``θ̄ = Σ θ_g/σ_g² / Σ 1/σ_g²``. ``dof = (#valid points) − 1``. Points whose
    error is non-finite or non-positive are dropped (the caller has already
    flagged such a parameter ambiguous, but we stay defensive).
    """

    weighted_sum = 0.0
    weight_total = 0.0
    valid = 0
    clean: list[tuple[float, float]] = []
    for value, error in zip(values, errors, strict=False):
        if not (math.isfinite(value) and math.isfinite(error)) or error <= 0.0:
            continue
        weight = 1.0 / (error * error)
        weighted_sum += value * weight
        weight_total += weight
        clean.append((value, error))
        valid += 1
    if valid < 2 or weight_total <= 0.0:
        return 0.0, max(0, valid - 1)
    mean = weighted_sum / weight_total
    q = 0.0
    for value, error in clean:
        residual = value - mean
        q += residual * residual / (error * error)
    return q, valid - 1


def classify_parameter_homogeneity(
    name: str,
    values: Sequence[float],
    errors: Sequence[float],
    *,
    at_limit: bool = False,
    p_local_threshold: float,
    p_global_threshold: float,
) -> ParameterHomogeneity:
    """Classify one parameter as global / local / ambiguous from its Q statistic.

    * A non-finite/invalid error, an at-limit estimate, or fewer than two valid
      per-dataset points → ``skipped`` and role ``"ambiguous"`` (never pre-fix a
      parameter the single fits could not pin down — verification-plan item 6).
    * ``p_value < p_local_threshold`` (strong upper-tail evidence of variation) →
      ``"local"``.
    * ``p_value > p_global_threshold`` (no evidence of variation) → ``"global"``.
    * otherwise → ``"ambiguous"``.

    The two thresholds are the effort knob: wide bands (Low) pre-fix more and
    enumerate less; conservative bands (Balanced) pre-fix only the clear tails.
    """

    values = list(values)
    errors = list(errors)
    finite_errors = [error for error in errors if math.isfinite(error) and error > 0.0]
    finite_values = [value for value in values if math.isfinite(value)]
    if at_limit:
        return ParameterHomogeneity(
            name=name,
            role="ambiguous",
            q_statistic=float("nan"),
            dof=0,
            p_value=float("nan"),
            skipped=True,
            reason="single-fit estimate at a parameter limit; not pre-fixed",
        )
    if len(finite_errors) < len(errors) or len(finite_values) < len(values):
        return ParameterHomogeneity(
            name=name,
            role="ambiguous",
            q_statistic=float("nan"),
            dof=0,
            p_value=float("nan"),
            skipped=True,
            reason="invalid single-fit error/estimate; not pre-fixed",
        )
    q, dof = homogeneity_statistic(values, errors)
    if dof < 1:
        return ParameterHomogeneity(
            name=name,
            role="ambiguous",
            q_statistic=q,
            dof=dof,
            p_value=float("nan"),
            skipped=True,
            reason="fewer than two valid per-dataset estimates; not pre-fixed",
        )
    p_value = chi2_sf(q, dof)
    if p_value < p_local_threshold:
        role, reason = (
            "local",
            (
                f"Q={q:.2f} (df={dof}, p={p_value:.3g} < {p_local_threshold:.3g}); "
                "strongly varies across the series"
            ),
        )
    elif p_value > p_global_threshold:
        role, reason = (
            "global",
            (
                f"Q={q:.2f} (df={dof}, p={p_value:.3g} > {p_global_threshold:.3g}); "
                "consistent with a single shared value"
            ),
        )
    else:
        role, reason = (
            "ambiguous",
            (f"Q={q:.2f} (df={dof}, p={p_value:.3g}); inside the ambiguous band"),
        )
    return ParameterHomogeneity(
        name=name,
        role=role,
        q_statistic=q,
        dof=dof,
        p_value=p_value,
        skipped=False,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Wald quadratic surrogate (technique G)
# --------------------------------------------------------------------------- #


def wald_globalisation_cost(
    values: Sequence[float],
    errors: Sequence[float],
) -> float:
    """Predicted Δχ² of collapsing per-dataset ``values`` onto one shared value.

    This is exactly the homogeneity statistic Q (a diagonal-covariance GLS
    collapse): the second-order Taylor expansion of the joint χ² around the
    all-local solution, evaluated at the inverse-variance-weighted common value.
    A large cost means globalising the parameter hurts the fit a lot — it wants
    to stay local. Invalid/degenerate inputs return ``0.0`` (no predicted cost,
    so the surrogate never *invents* a reason to keep something local).
    """

    q, _dof = homogeneity_statistic(values, errors)
    return q if math.isfinite(q) else 0.0


def wald_subset_delta_chi2(
    subset: Sequence[str],
    per_param_estimates: Mapping[str, Sequence[float]],
    per_param_errors: Mapping[str, Sequence[float]],
) -> float:
    """Sum the per-parameter globalisation costs over a globalised ``subset``.

    Diagonal Wald surrogate: cross-parameter correlation is ignored (the v1
    documented limitation — it is why the correlated-pair case is where the
    surrogate/greedy may disagree). The real fit of the top-K subsets is the
    safety net that bounds any error from this approximation.
    """

    total = 0.0
    for name in subset:
        values = per_param_estimates.get(name)
        errors = per_param_errors.get(name)
        if values is None or errors is None:
            continue
        total += wald_globalisation_cost(values, errors)
    return total
