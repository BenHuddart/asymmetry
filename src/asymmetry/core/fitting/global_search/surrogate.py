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
    "OrderedCollapse",
    "RunEstimate",
    "collapse_cost",
    "greedy_assignment",
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
    """True when ``matrix`` is finite and invertible to working precision.

    Every matrix reaching here is symmetric — a covariance block, or a sum of
    precision matrices — and a symmetric matrix's 2-norm condition number is
    ``max|λ| / min|λ|``, so the eigenvalues answer the same question
    ``numpy.linalg.cond`` answers by SVD, at rather less than half the cost. This
    runs once per run per subset and again per pooled window, which on a partition
    search is hundreds of thousands of times.
    """

    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return False
    magnitudes = np.abs(np.linalg.eigvalsh(matrix))
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = float(np.max(magnitudes) / np.min(magnitudes))
    return condition <= CONDITION_LIMIT


def _diagonal_weight(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """``diag(1/σ²)``, with weight zero where σ says the run pins nothing down."""

    usable = np.isfinite(sigma) & (sigma > 0.0)
    safe = np.where(usable, sigma, 1.0)
    return np.diag(np.where(usable, 1.0 / (safe * safe), 0.0))


@dataclass(frozen=True)
class _RunWeight:
    """One run's precision block for one subset — the only per-run inversion.

    ``subset_index``/``rest_index`` index into :attr:`RunEstimate.names`;
    ``weight`` is :math:`W_r = (C_r[S,S])^{-1}`, or the diagonal fallback when the
    covariance block is missing or ill-conditioned (``full_covariance`` false).
    ``weighted_value`` is ``W_r θ_{r,S}``, precomputed because every pooled sum
    needs it.

    Nothing here depends on which *window* of the series is being scored, only on
    ``(run, subset)`` — which is why one of these can be memoised and reused
    across every overlapping segment a partition search asks about.
    """

    subset_index: tuple[int, ...]
    rest_index: tuple[int, ...]
    weight: NDArray[np.float64]
    weighted_value: NDArray[np.float64]
    full_covariance: bool


def _run_weight(estimate: RunEstimate, canonical: tuple[str, ...]) -> _RunWeight:
    position = {name: index for index, name in enumerate(estimate.names)}
    shared_names = set(canonical)
    subset_index = tuple(position[name] for name in canonical)
    rest_index = tuple(
        index for index, name in enumerate(estimate.names) if name not in shared_names
    )
    block = (
        None
        if estimate.covariance is None
        else estimate.covariance[np.ix_(subset_index, subset_index)]
    )
    if block is not None and _well_conditioned(block):
        weight = np.linalg.inv(block)
        full_covariance = True
    else:
        weight = _diagonal_weight(estimate.uncertainties[list(subset_index)])
        full_covariance = False
    return _RunWeight(
        subset_index=subset_index,
        rest_index=rest_index,
        weight=weight,
        weighted_value=weight @ estimate.values[list(subset_index)],
        full_covariance=full_covariance,
    )


def _prepare(
    estimates: Sequence[RunEstimate],
    canonical: tuple[str, ...],
) -> tuple[list[_RunWeight], NDArray[np.float64], NDArray[np.float64], frozenset[int]]:
    """Per-run weights plus their pooled sums."""

    size = len(canonical)
    pooled_weight = np.zeros((size, size), dtype=float)
    pooled_weighted_value = np.zeros(size, dtype=float)
    prepared: list[_RunWeight] = []
    fallback_runs: set[int] = set()
    for estimate in estimates:
        weight = _run_weight(estimate, canonical)
        pooled_weight += weight.weight
        pooled_weighted_value += weight.weighted_value
        if not weight.full_covariance:
            fallback_runs.add(estimate.run_number)
        prepared.append(weight)
    return prepared, pooled_weight, pooled_weighted_value, frozenset(fallback_runs)


def _shared_values(
    pooled_weight: NDArray[np.float64], pooled_weighted_value: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """The pooled estimate :math:`\\bar\\theta_S`, or ``None`` when unconstrained."""

    if not _well_conditioned(pooled_weight):
        return None
    return np.linalg.solve(pooled_weight, pooled_weighted_value)


def _delta_chi2(
    estimates: Sequence[RunEstimate],
    canonical: tuple[str, ...],
) -> float:
    """``Δχ²(S)`` alone — the collapse without its warm start.

    Scoring a subset needs only this number, and skipping the conditional shift
    skips one linear solve per run. :func:`collapse_cost` remains the entry point
    when the warm start itself is wanted.
    """

    if not canonical:
        return 0.0
    prepared, pooled_weight, pooled_weighted_value, _ = _prepare(estimates, canonical)
    shared = _shared_values(pooled_weight, pooled_weighted_value)
    if shared is None:
        return math.inf
    total = 0.0
    for estimate, weight in zip(estimates, prepared, strict=True):
        offset = estimate.values[list(weight.subset_index)] - shared
        total += float(offset @ weight.weight @ offset)
    return total


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

    prepared, pooled_weight, pooled_weighted_value, fallback_runs = _prepare(runs, canonical)
    shared = _shared_values(pooled_weight, pooled_weighted_value)

    if shared is None:
        return CollapseResult(
            subset=canonical,
            delta_chi2=math.inf,
            shared_values={},
            conditional_locals_by_run={
                estimate.run_number: unshifted_locals(estimate) for estimate in runs
            },
            diagonal_fallback_runs=fallback_runs,
        )

    delta_chi2 = 0.0
    conditional_locals: dict[int, dict[str, float]] = {}
    for estimate, weight in zip(runs, prepared, strict=True):
        subset_index = list(weight.subset_index)
        rest_index = list(weight.rest_index)
        offset = estimate.values[subset_index] - shared
        delta_chi2 += float(offset @ weight.weight @ offset)
        rest_values = estimate.values[rest_index]
        if weight.full_covariance and rest_index:
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
        diagonal_fallback_runs=fallback_runs,
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
    chi_squared = math.fsum(float(estimate.chi_squared) for estimate in runs)
    sample_count = sum(int(estimate.n_points) for estimate in runs)
    free_count = len(runs[0].names)
    parameter_count = len(canonical) + (free_count - len(canonical)) * len(runs)
    return (
        chi_squared
        + _delta_chi2(runs, canonical)
        + metric_penalty(parameter_count, sample_count=sample_count, metric=metric)
    )


def _eligible_names(estimates: Sequence[RunEstimate], free_names: Sequence[str]) -> list[str]:
    """``free_names`` minus every parameter resting on a bound in any run.

    A bounded parameter's curvature is not the curvature of an interior minimum,
    so the collapse would be modelling numerical noise rather than its spread.
    """

    bound: set[str] = set()
    for estimate in estimates:
        bound |= estimate.at_bound
    return [name for name in free_names if name not in bound]


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

    This is the *exhaustive* ranking: ``2^P`` collapses. A caller that only wants
    the winner should use :func:`greedy_assignment`, which reaches the same answer
    in ``O(P²)`` collapses whenever the subsets score separably.
    """

    runs = list(estimates)
    eligible = _eligible_names(runs, free_names)

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


# --------------------------------------------------------------------------- #
# Windowed surrogate over an ordered series
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _SubsetPrefix:
    """Running sums of one subset's GLS pieces, indexed by run position.

    Entry ``i`` holds the sum over positions ``0 … i-1``, so a window ``[start,
    stop)`` is one subtraction.
    """

    weight: NDArray[np.float64]
    weighted_value: NDArray[np.float64]
    quadratic: NDArray[np.float64]


class OrderedCollapse:
    """Surrogate scores for every contiguous window of an ordered run series.

    :func:`~asymmetry.core.fitting.global_search.partition.tier2_segment_cost`
    asks one question ``O(G²)`` times — "what does globalising ``S`` cost over
    runs ``[start, stop)``?" — and the answer is built from three sums over the
    window,

    .. math::

        A = \\sum_r W_r, \\quad b = \\sum_r W_r \\theta_{r,S}, \\quad
        Q = \\sum_r \\theta_{r,S}^\\mathsf{T} W_r \\theta_{r,S},

    from which :math:`\\Delta\\chi^2(S) = Q - b^\\mathsf{T} A^{-1} b`. That is the
    same number :func:`collapse_cost` reaches as
    :math:`\\sum_r (\\theta_r - \\bar\\theta)^\\mathsf{T} W_r (\\theta_r -
    \\bar\\theta)`, since :math:`\\bar\\theta = A^{-1}b` is what minimises it — but
    written this way each of the three pieces is a *prefix difference*. A subset's
    prefixes are built once and every window then costs a single small solve
    instead of a pass over its runs, which is the difference between a partition
    search over a 29-run series taking a second and taking a minute.

    ``estimates`` is positional: entry ``i`` is the estimate for the ``i``-th run
    of the series order, or ``None`` where this template has none. A window
    containing a gap is not scoreable — :meth:`covers` says so — because the
    surrogate would otherwise silently price a segment it has not seen.
    """

    def __init__(self, estimates: Sequence[RunEstimate | None], names: Sequence[str]) -> None:
        self._estimates = tuple(estimates)
        self._names = tuple(names)
        self._prefix: dict[tuple[str, ...], _SubsetPrefix] = {}

        count = len(self._estimates)
        self._present = np.zeros(count + 1, dtype=int)
        self._chi_squared = np.zeros(count + 1, dtype=float)
        self._sample_count = np.zeros(count + 1, dtype=int)
        self._log_points = np.zeros(count + 1, dtype=float)
        for index, estimate in enumerate(self._estimates):
            present = estimate is not None
            self._present[index + 1] = self._present[index] + int(present)
            self._chi_squared[index + 1] = self._chi_squared[index] + (
                float(estimate.chi_squared) if estimate is not None else 0.0
            )
            self._sample_count[index + 1] = self._sample_count[index] + (
                int(estimate.n_points) if estimate is not None else 0
            )
            self._log_points[index + 1] = self._log_points[index] + (
                math.log(max(int(estimate.n_points), 1)) if estimate is not None else 0.0
            )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def covers(self, start: int, stop: int) -> bool:
        """True when every run of ``[start, stop)`` has an estimate."""

        return int(self._present[stop] - self._present[start]) == stop - start

    def _window(self, start: int, stop: int) -> tuple[RunEstimate, ...]:
        return tuple(estimate for estimate in self._estimates[start:stop] if estimate is not None)

    def _canonical(self, subset: Sequence[str]) -> tuple[str, ...]:
        """``subset`` as a set, ordered by :attr:`names`.

        Δχ² does not depend on the order the subset is written in, so normalising
        here is what makes the prefix cache keyed by the *set*: without it a walk
        that appends its newest parameter last stores ``(a, c, b)`` and ``(a, b,
        c)`` as two entries and rebuilds the same prefixes twice.
        """

        wanted = set(subset)
        return tuple(name for name in self._names if name in wanted)

    def _subset_prefix(self, canonical: tuple[str, ...]) -> _SubsetPrefix:
        cached = self._prefix.get(canonical)
        if cached is not None:
            return cached

        size = len(canonical)
        count = len(self._estimates)
        weight = np.zeros((count + 1, size, size), dtype=float)
        weighted_value = np.zeros((count + 1, size), dtype=float)
        quadratic = np.zeros(count + 1, dtype=float)
        for index, estimate in enumerate(self._estimates):
            weight[index + 1] = weight[index]
            weighted_value[index + 1] = weighted_value[index]
            quadratic[index + 1] = quadratic[index]
            if estimate is None:
                continue
            prepared = _run_weight(estimate, canonical)
            values = estimate.values[list(prepared.subset_index)]
            weight[index + 1] += prepared.weight
            weighted_value[index + 1] += prepared.weighted_value
            quadratic[index + 1] += float(values @ prepared.weighted_value)

        prefix = _SubsetPrefix(weight=weight, weighted_value=weighted_value, quadratic=quadratic)
        self._prefix[canonical] = prefix
        return prefix

    def delta_chi2(self, start: int, stop: int, subset: Sequence[str]) -> float:
        """``Δχ²(S)`` over ``[start, stop)``; ``inf`` when nothing constrains ``S``."""

        canonical = self._canonical(subset)
        if not canonical:
            return 0.0
        prefix = self._subset_prefix(canonical)
        pooled = prefix.weight[stop] - prefix.weight[start]
        if not _well_conditioned(pooled):
            return math.inf
        pooled_value = prefix.weighted_value[stop] - prefix.weighted_value[start]
        quadratic = float(prefix.quadratic[stop] - prefix.quadratic[start])
        return quadratic - float(pooled_value @ np.linalg.solve(pooled, pooled_value))

    def surrogate_ic(
        self, start: int, stop: int, subset: Sequence[str], metric: SelectionMetric
    ) -> float:
        """The window's surrogate IC of globalising ``subset``.

        Identical in definition to :func:`surrogate_ic` restricted to the window's
        estimates.
        """

        canonical = self._canonical(subset)
        run_count = stop - start
        parameter_count = len(canonical) + (len(self._names) - len(canonical)) * run_count
        return (
            float(self._chi_squared[stop] - self._chi_squared[start])
            + self.delta_chi2(start, stop, canonical)
            + metric_penalty(
                parameter_count,
                sample_count=int(self._sample_count[stop] - self._sample_count[start]),
                metric=metric,
            )
        )

    def lower_bound_ic(self, start: int, stop: int, metric: SelectionMetric) -> float:
        """The cheapest IC *any* assignment of this template could reach here.

        ``Δχ²(S) ≥ 0`` and the parameter count is smallest when every free
        parameter is shared, so ``Σ_r χ²_r + penalty(P, n)`` is a floor. A template
        whose floor already loses to another template's scored value cannot win the
        window, and its walk is skipped — the same incumbent bound the role search
        uses across templates, and exact for the same reason.
        """

        return float(self._chi_squared[stop] - self._chi_squared[start]) + metric_penalty(
            len(self._names),
            sample_count=int(self._sample_count[stop] - self._sample_count[start]),
            metric=metric,
        )

    def greedy(
        self,
        start: int,
        stop: int,
        free_names: Sequence[str],
        metric: SelectionMetric,
    ) -> tuple[tuple[str, ...], float]:
        """Forward selection over ``[start, stop)`` — see :func:`greedy_assignment`."""

        remaining = _eligible_names(self._window(start, stop), free_names)
        shared: tuple[str, ...] = ()
        best_ic = self.surrogate_ic(start, stop, shared, metric)

        while remaining:
            candidate_ic, name = min(
                (self.surrogate_ic(start, stop, (*shared, item), metric), item)
                for item in remaining
            )
            if not candidate_ic < best_ic:
                return shared, best_ic
            shared = tuple(item for item in free_names if item in {*shared, name})
            best_ic = candidate_ic
            remaining.remove(name)

        return shared, best_ic


    # ------------------------------------------------------------------ #
    # Partition convention: each local parameter is penalised against the
    # points of *its own run*, each shared parameter against the segment's.
    # ------------------------------------------------------------------ #

    def partition_ic(self, start: int, stop: int, subset: Sequence[str]) -> float:
        """The window's BIC under the partition convention.

        ``Σ_r χ²_r + Δχ²(S) + |S|·ln N_window + (P − |S|)·Σ_r ln n_r``. A local
        parameter is informed by its own run's ``n_r`` points and a shared one
        by the whole window's, so with nothing shared the window costs exactly
        the sum of its runs' own BICs — a segment does not get cheaper, or
        dearer, merely by being scored *together*. Under the joint convention
        (:meth:`surrogate_ic`, ``k·ln N_window``) every local parameter paid an
        extra ``ln G`` for the company it kept, which on a real series charged a
        nine-parameter template ~270 BIC units more than a four-parameter one
        over a 16-run segment for no statistical reason, and pushed the
        partition towards short segments of low-parameter templates.
        """

        canonical = self._canonical(subset)
        shared = len(canonical)
        local = len(self._names) - shared
        sample_count = int(self._sample_count[stop] - self._sample_count[start])
        return (
            float(self._chi_squared[stop] - self._chi_squared[start])
            + self.delta_chi2(start, stop, canonical)
            + shared * math.log(max(sample_count, 1))
            + local * float(self._log_points[stop] - self._log_points[start])
        )

    def lower_bound_partition_ic(self, start: int, stop: int) -> float:
        """The cheapest partition-convention BIC any assignment could reach here.

        ``Δχ²(S) ≥ 0`` and sharing everything is the smallest penalty
        (``Π_r n_r ≥ Σ_r n_r`` for records of two or more points), so
        ``Σ_r χ²_r + P·ln N_window`` is a floor.
        """

        sample_count = int(self._sample_count[stop] - self._sample_count[start])
        return float(self._chi_squared[stop] - self._chi_squared[start]) + len(
            self._names
        ) * math.log(max(sample_count, 1))

    def greedy_partition(
        self, start: int, stop: int, free_names: Sequence[str]
    ) -> tuple[tuple[str, ...], float]:
        """Forward selection over ``[start, stop)`` under :meth:`partition_ic`."""

        remaining = _eligible_names(self._window(start, stop), free_names)
        shared: tuple[str, ...] = ()
        best_ic = self.partition_ic(start, stop, shared)

        while remaining:
            candidate_ic, name = min(
                (self.partition_ic(start, stop, (*shared, item)), item) for item in remaining
            )
            if not candidate_ic < best_ic:
                return shared, best_ic
            shared = tuple(item for item in free_names if item in {*shared, name})
            best_ic = candidate_ic
            remaining.remove(name)

        return shared, best_ic


def greedy_assignment(
    estimates: Sequence[RunEstimate],
    free_names: Sequence[str],
    metric: SelectionMetric,
) -> tuple[tuple[str, ...], float]:
    """The best globalisation subset by forward selection, and its surrogate IC.

    Starting from all-local, each round globalises the single remaining parameter
    that lowers the surrogate IC most and stops when none lowers it. That costs
    ``P(P+1)/2`` collapses against :func:`rank_assignments`' ``2^P``, which is the
    difference between a partition search over an ordered series being seconds or
    minutes: the dynamic program asks for ``O(G²)`` overlapping windows, and every
    window is scored for every template.

    The two agree exactly whenever the subsets score **separably** — diagonal
    per-run covariance and a penalty linear in ``|S|`` (AIC or BIC), where
    ``Δχ²(S) = Σ_{p∈S} Δχ²({p})`` makes each parameter's contribution independent
    of the rest. Correlated blocks can in principle hide a pair that only pays off
    together; the partition search accepts that, because a segment cost is a
    *bound* whose winner tier 3 refits exactly anyway.

    Ties break on the name, so the walk is deterministic. The subset comes back in
    ``free_names`` order, matching :func:`rank_assignments`. A caller scoring many
    windows of one ordered series should hold an :class:`OrderedCollapse` instead,
    which shares each subset's prefix sums across them.
    """

    runs = tuple(estimates)
    return OrderedCollapse(runs, runs[0].names).greedy(0, len(runs), free_names, metric)
