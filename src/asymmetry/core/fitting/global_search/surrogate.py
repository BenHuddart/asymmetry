"""Full-covariance GLS collapse surrogate for global/local role assignments.

:mod:`~asymmetry.core.fitting.global_search.homogeneity` scores one parameter at
a time from its per-run value and error — a *diagonal* Wald surrogate that
ignores every correlation between parameters. This module is the full-covariance
version: given the per-run all-local estimates (values, covariance, χ²) it scores
a whole *subset* of parameters at once, and hands back the warm start the exact
coupled fit needs.

The kernel is the generalised-least-squares collapse. For run ``r`` with
all-local estimate :math:`\\theta_r` and covariance :math:`C_r`, globalising the
subset ``S`` costs

.. math::

    W_r = (C_r[S, S])^{-1}, \\qquad
    \\bar\\theta_S = \\Bigl(\\sum_r W_r\\Bigr)^{-1} \\sum_r W_r \\theta_{r,S},

.. math::

    \\Delta\\chi^2(S) = \\sum_r (\\theta_{r,S} - \\bar\\theta_S)^\\mathsf{T}
                        W_r (\\theta_{r,S} - \\bar\\theta_S),

which is the second-order expansion of the joint χ² around the all-local
solution, minimised over the shared values. Profiling the *other* parameters out
rather than holding them fixed is what the conditional shift does: run ``r``'s
non-subset parameters move to

.. math::

    \\theta_{r,\\neg S} - C_r[\\neg S, S]\\, (C_r[S, S])^{-1}
                          (\\theta_{r,S} - \\bar\\theta_S).

Those are the per-run local start values for the warm coupled fit, and
:math:`\\bar\\theta_S` are the shared ones.

Fallbacks are explicit, never silent-but-unnamed: a run whose covariance is
missing, whose ``S``-block is not finite, or whose ``S``-block has a condition
number above :data:`CONDITION_LIMIT` is scored with the diagonal weight
:math:`\\mathrm{diag}(1/\\sigma^2)` and contributes **no** conditional shift. Such
runs are listed in :attr:`CollapseResult.diagonal_fallback_runs`. A σ that is
non-finite or non-positive means that run carries no information about that
parameter, so its weight is zero.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import SelectionMetric

__all__ = [
    "CONDITION_LIMIT",
    "CollapseResult",
    "RunEstimate",
    "collapse_cost",
    "metric_penalty",
    "rank_assignments",
    "run_estimate_from_fit_result",
    "surrogate_ic",
]


#: Covariance blocks with a 2-norm condition number above this are unusable; the
#: run falls back to the diagonal weight. Matches the plan's 1e12.
CONDITION_LIMIT = 1e12


# --------------------------------------------------------------------------- #
# Per-run all-local estimate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunEstimate:
    """One run's all-local estimate, in a fixed free-parameter order.

    ``covariance`` is a genuine optional: a persisted per-run table drops the
    covariance (``_serialize_fit_result``), and a fit whose HESSE step failed has
    none either. ``None`` means "score this run with the diagonal weight". When
    it *is* present every entry is finite and it is ordered exactly like
    :attr:`names`.

    ``uncertainties`` is the diagonal fallback (σ, not σ²); a non-finite or
    non-positive entry means the run pins nothing down for that parameter and
    carries weight zero.

    ``at_bound`` names parameters resting on a limit in this run. Such a
    parameter is never *proposed* for globalisation by :func:`rank_assignments`
    — its curvature is not the curvature of an interior minimum — though
    :func:`collapse_cost` will still collapse it if a caller asks.
    """

    run_number: int
    names: tuple[str, ...]
    values: NDArray[np.float64]
    covariance: NDArray[np.float64] | None
    uncertainties: NDArray[np.float64]
    at_bound: frozenset[str]
    chi_squared: float
    n_points: int


def _reordered_covariance(result: FitResult, names: Sequence[str]) -> NDArray[np.float64] | None:
    """Return ``result``'s covariance restricted+reordered to ``names``.

    ``None`` when the fit produced no covariance, when it does not cover every
    requested name, or when the extracted block holds a non-finite entry.
    """

    covariance = result.covariance
    if covariance is None:
        return None
    covariance_names = list(result.covariance_parameters)
    positions = {name: index for index, name in enumerate(covariance_names)}
    if any(name not in positions for name in names):
        return None
    index = [positions[name] for name in names]
    block = np.asarray(covariance, dtype=float)[np.ix_(index, index)]
    if not np.all(np.isfinite(block)):
        return None
    return np.ascontiguousarray(block)


def run_estimate_from_fit_result(
    result: FitResult,
    free_names: Sequence[str],
    *,
    run_number: int,
    n_points: int,
    at_bound: Iterable[str] = (),
) -> RunEstimate:
    """Build a :class:`RunEstimate` from one run's all-local :class:`FitResult`.

    ``free_names`` fixes the parameter order for every array on the estimate.
    ``n_points`` is the number of fitted points for this run at the series search
    resolution (the engine's ``dof`` is ``n - k``, not ``n``, so the caller
    passes the count it fitted).
    """

    names = tuple(free_names)
    values = np.array([float(result.parameters[name].value) for name in names], dtype=float)
    uncertainties = np.array(
        [float(result.uncertainties.get(name, math.nan)) for name in names],
        dtype=float,
    )
    return RunEstimate(
        run_number=int(run_number),
        names=names,
        values=values,
        covariance=_reordered_covariance(result, names),
        uncertainties=uncertainties,
        at_bound=frozenset(at_bound),
        chi_squared=float(result.chi_squared),
        n_points=int(n_points),
    )


# --------------------------------------------------------------------------- #
# GLS collapse
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CollapseResult:
    """Outcome of collapsing ``subset`` onto one shared value per parameter.

    ``delta_chi2`` is ``inf`` when the pooled weight :math:`\\sum_r W_r` is itself
    unusable — no run constrains some subset parameter — in which case
    ``shared_values`` is empty and ``conditional_locals_by_run`` holds each run's
    unshifted non-subset values. Any caller comparing ICs sees ``inf`` and drops
    the subset, so the warm start is never consumed.
    """

    subset: tuple[str, ...]
    delta_chi2: float
    shared_values: dict[str, float]
    conditional_locals_by_run: dict[int, dict[str, float]]
    diagonal_fallback_runs: frozenset[int]


def _well_conditioned(matrix: NDArray[np.float64]) -> bool:
    """True when ``matrix`` is finite and invertible to working precision."""

    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = float(np.linalg.cond(matrix))
    return condition <= CONDITION_LIMIT


def _diagonal_weight(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """``diag(1/σ²)``, with weight zero where σ says the run pins nothing down."""

    usable = np.isfinite(sigma) & (sigma > 0.0)
    safe = np.where(usable, sigma, 1.0)
    return np.diag(np.where(usable, 1.0 / (safe * safe), 0.0))


def collapse_cost(estimates: Sequence[RunEstimate], subset: Sequence[str]) -> CollapseResult:
    """GLS collapse of ``subset`` across ``estimates``.

    The returned ``subset`` is ``subset`` de-duplicated in the caller's order;
    :func:`rank_assignments` always emits it in the ``free_names`` order, so
    structure keys built from it are stable.
    """

    runs = list(estimates)
    canonical = tuple(dict.fromkeys(subset))
    shared_names = set(canonical)

    def unshifted_locals(estimate: RunEstimate) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(estimate.names, estimate.values, strict=True)
            if name not in shared_names
        }

    if not canonical:
        return CollapseResult(
            subset=canonical,
            delta_chi2=0.0,
            shared_values={},
            conditional_locals_by_run={
                estimate.run_number: unshifted_locals(estimate) for estimate in runs
            },
            diagonal_fallback_runs=frozenset(),
        )

    size = len(canonical)
    pooled_weight = np.zeros((size, size), dtype=float)
    pooled_weighted_value = np.zeros(size, dtype=float)
    prepared: list[tuple[RunEstimate, list[int], list[int], NDArray[np.float64], bool]] = []
    fallback_runs: set[int] = set()

    for estimate in runs:
        position = {name: index for index, name in enumerate(estimate.names)}
        subset_index = [position[name] for name in canonical]
        rest_index = [
            index for index, name in enumerate(estimate.names) if name not in shared_names
        ]
        block = (
            None
            if estimate.covariance is None
            else estimate.covariance[np.ix_(subset_index, subset_index)]
        )
        if block is not None and _well_conditioned(block):
            weight = np.linalg.inv(block)
            full_covariance = True
        else:
            weight = _diagonal_weight(estimate.uncertainties[subset_index])
            full_covariance = False
            fallback_runs.add(estimate.run_number)
        pooled_weight += weight
        pooled_weighted_value += weight @ estimate.values[subset_index]
        prepared.append((estimate, subset_index, rest_index, weight, full_covariance))

    if not _well_conditioned(pooled_weight):
        return CollapseResult(
            subset=canonical,
            delta_chi2=math.inf,
            shared_values={},
            conditional_locals_by_run={
                estimate.run_number: unshifted_locals(estimate) for estimate in runs
            },
            diagonal_fallback_runs=frozenset(fallback_runs),
        )

    shared = np.linalg.solve(pooled_weight, pooled_weighted_value)

    delta_chi2 = 0.0
    conditional_locals: dict[int, dict[str, float]] = {}
    for estimate, subset_index, rest_index, weight, full_covariance in prepared:
        offset = estimate.values[subset_index] - shared
        delta_chi2 += float(offset @ weight @ offset)
        rest_values = estimate.values[rest_index]
        if full_covariance and rest_index:
            covariance = estimate.covariance
            cross = covariance[np.ix_(rest_index, subset_index)]
            block = covariance[np.ix_(subset_index, subset_index)]
            rest_values = rest_values - cross @ np.linalg.solve(block, offset)
        conditional_locals[estimate.run_number] = {
            estimate.names[index]: float(value)
            for index, value in zip(rest_index, rest_values, strict=True)
        }

    return CollapseResult(
        subset=canonical,
        delta_chi2=float(delta_chi2),
        shared_values={name: float(value) for name, value in zip(canonical, shared, strict=True)},
        conditional_locals_by_run=conditional_locals,
        diagonal_fallback_runs=frozenset(fallback_runs),
    )


# --------------------------------------------------------------------------- #
# Information criterion
# --------------------------------------------------------------------------- #


def metric_penalty(parameter_count: int, *, sample_count: int, metric: SelectionMetric) -> float:
    """The additive IC penalty ``IC - chi2`` for ``k`` params over ``n`` points.

    Mirrors :func:`asymmetry.core.fitting.fit_wizard.compute_information_criteria`
    exactly (AICc falls back to AIC's penalty when ``n <= k + 1``, which is
    precisely when that function reports ``aicc = None``), so a surrogate IC and
    an exact-fit IC are on one scale.
    """

    k = max(int(parameter_count), 0)
    n = max(int(sample_count), 1)
    if metric == SelectionMetric.AIC:
        return 2.0 * k
    if metric == SelectionMetric.BIC:
        return k * math.log(n)
    aic_penalty = 2.0 * k
    if n > k + 1:
        return aic_penalty + 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return aic_penalty


def surrogate_ic(
    estimates: Sequence[RunEstimate],
    subset: Sequence[str],
    metric: SelectionMetric,
) -> float:
    """Predicted IC of globalising ``subset`` across ``estimates``.

    ``Σ_r χ²_r + Δχ²(S) + penalty(k, n)`` with ``k = |S| + (P − |S|)·G`` and
    ``n = Σ_r n_points`` — the same ``k`` the exact path's ``parameter_count``
    carries for that assignment.
    """

    runs = list(estimates)
    canonical = tuple(dict.fromkeys(subset))
    collapse = collapse_cost(runs, canonical)
    chi_squared = math.fsum(float(estimate.chi_squared) for estimate in runs)
    sample_count = sum(int(estimate.n_points) for estimate in runs)
    free_count = len(runs[0].names)
    parameter_count = len(canonical) + (free_count - len(canonical)) * len(runs)
    return (
        chi_squared
        + collapse.delta_chi2
        + metric_penalty(parameter_count, sample_count=sample_count, metric=metric)
    )


def rank_assignments(
    estimates: Sequence[RunEstimate],
    free_names: Sequence[str],
    metric: SelectionMetric,
    *,
    max_enumerated: int = 12,
) -> list[tuple[tuple[str, ...], float]]:
    """Rank globalisation subsets by surrogate IC, cheapest first.

    Parameters at a bound on *any* run are never proposed. With at most
    ``max_enumerated`` remaining candidates every subset is enumerated (the empty
    subset — all-local — included); above that only the single-parameter subsets
    are scored, which is what backward elimination consumes. A caller that also
    wants the all-local reference in the large branch scores ``()`` itself.

    Subsets are emitted in ``free_names`` order and ties broken by size then
    name, so the ranking is deterministic.
    """

    runs = list(estimates)
    bound: set[str] = set()
    for estimate in runs:
        bound |= estimate.at_bound
    eligible = [name for name in free_names if name not in bound]

    if len(eligible) <= max_enumerated:
        subsets = [
            combination
            for size in range(len(eligible) + 1)
            for combination in itertools.combinations(eligible, size)
        ]
    else:
        subsets = [(name,) for name in eligible]

    scored = [(subset, surrogate_ic(runs, subset, metric)) for subset in subsets]
    scored.sort(key=lambda item: (item[1], len(item[0]), item[0]))
    return scored
