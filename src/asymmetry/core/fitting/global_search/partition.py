"""Exact segmentation of an ordered run series into structural phases.

A temperature or field series can cross a transition, and the fit function that
describes the runs on one side does not describe the runs on the other. This
module answers "where are the breaks?" as a change-point problem: partition the
ordered runs into contiguous *segments*, each with one structure (one template
and one global/local assignment), minimising

    Σ_s IC(best structure of segment s)   +   β · (number of breaks).

The minimisation is an exact dynamic program over "exactly k breaks", so one
pass produces the whole penalty path ``F_0 … F_{K_max}`` rather than a single
answer at one β. The elbow is then read off the marginal gains
``g_k = F_{k−1} − F_k`` against a modified-BIC-style floor
``β_floor = c · ln(N_total)``.

**Breaks are structural.** A break is a change of *structure key* between two
adjacent segments, and the dynamic program admits a partition only when every
pair of adjacent (non-excluded) segments differs in that key — so a parameter
that merely *drifts*, or a sharing pattern that changes, can never be
approximated by a staircase of breaks. The key is what the cost factory names
it: by default the template, or — through ``family_of`` — the template's
*family* (oscillatory, relaxation, multi-rate, Kubo-Toyabe, …), which is what a
physical phase is. Which template within the family, and which parameters it
shares, are priced into the segment cost but never count as a break; the
coupled search decides them per phase.

**Minimum segment length.** A segment shorter than ``min_segment`` is admitted
only at either end of the series. Such an *end stub* is scored at the sum of its
runs' own per-run best cost (no coupled fit), flagged
:attr:`Segment.excluded`, and never merged. An interior run is never excised: a
run that does not fit its neighbours is a per-run gate failure, which the
existing verdicts already report.

Two segment-cost tiers are provided, each an admissible bound for the next:

* :func:`tier1_segment_cost` — free. The per-run per-template IC sums from the
  wizard's phase-1 table. This is the all-local cost of that template, a lower
  bound on every role assignment of it.
* :func:`tier2_segment_cost` — closed form. Adds the full-covariance GLS collapse
  of the best sharing pattern
  (:func:`~asymmetry.core.fitting.global_search.surrogate.greedy_assignment`), so
  a segment is priced with the sharing it can support.

Tier 1 sums per-run ICs (each penalised against that run's own point count);
tier 2 scores the segment as one problem (one penalty against the segment's
total point count). The two are therefore on deliberately different scales —
tier 1 is the cheaper lower bound, tier 2 the number the search runs on.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_search.surrogate import OrderedCollapse, RunEstimate

__all__ = [
    "PartitionConfig",
    "PartitionPath",
    "PartitionSolution",
    "Segment",
    "SegmentCost",
    "partition_series",
    "tier1_segment_cost",
    "tier2_segment_cost",
]


@dataclass(frozen=True)
class PartitionConfig:
    """Knobs for :func:`partition_series`.

    ``floor_coefficient`` is the ``c`` of ``β_floor = c · ln(N_total)``; it is a
    config field, never a GUI control.
    """

    min_segment: int = 3
    floor_coefficient: float = 2.0
    max_breaks: int | None = None


class SegmentCost(Protocol):
    """Cost of describing ``order[start:stop]`` with one structure.

    Returns ``(ic, structure_key)``; ``ic`` is ``inf`` and the key empty when no
    structure describes every run of the segment.
    """

    def __call__(self, start: int, stop: int) -> tuple[float, str]: ...


@dataclass(frozen=True)
class Segment:
    """One contiguous stretch of the ordered series.

    ``start``/``stop`` index into the ``order`` passed to
    :func:`partition_series` (half-open). ``excluded`` marks an end stub shorter
    than ``min_segment``: it gets no coupled fit and is reported as "excluded
    from the global fit: looks like a different phase".
    """

    start: int
    stop: int
    run_numbers: tuple[int, ...]
    structure: str
    ic: float
    excluded: bool


@dataclass(frozen=True)
class PartitionSolution:
    """The best partition with exactly ``breaks`` structural breaks.

    ``total_ic`` is Σ segment IC with no β term; ``gain`` is ``F_{k−1} − F_k``
    (``0.0`` at ``k = 0``) and ``admissible`` says whether it clears
    ``β_floor``.
    """

    breaks: int
    segments: tuple[Segment, ...]
    total_ic: float
    gain: float
    admissible: bool
    boundaries: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PartitionPath:
    """The whole penalty path, with the elbow pre-selected.

    ``solutions[k]`` is the exactly-``k``-break DP solution.
    ``selected_k`` is the largest ``k`` whose gains ``g_1 … g_k`` are *all*
    admissible — a single inadmissible step ends the path, so the selection can
    never jump over a rejected break.
    """

    solutions: tuple[PartitionSolution, ...]
    selected_k: int
    beta_floor: float

    def to_payload(self) -> dict:
        """Plain, JSON-able mirror of this path."""

        return {
            "beta_floor": float(self.beta_floor),
            "selected_k": int(self.selected_k),
            "solutions": [
                {
                    "breaks": int(solution.breaks),
                    "total_ic": float(solution.total_ic),
                    "gain": float(solution.gain),
                    "admissible": bool(solution.admissible),
                    "boundaries": [
                        [float(estimate), float(half_gap)]
                        for estimate, half_gap in solution.boundaries
                    ],
                    "segments": [
                        {
                            "start": int(segment.start),
                            "stop": int(segment.stop),
                            "run_numbers": [int(run) for run in segment.run_numbers],
                            "structure": segment.structure,
                            "ic": float(segment.ic),
                            "excluded": bool(segment.excluded),
                        }
                        for segment in solution.segments
                    ],
                }
                for solution in self.solutions
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> PartitionPath:
        """Rebuild a path from :meth:`to_payload`."""

        solutions = tuple(
            PartitionSolution(
                breaks=int(entry["breaks"]),
                segments=tuple(
                    Segment(
                        start=int(segment["start"]),
                        stop=int(segment["stop"]),
                        run_numbers=tuple(int(run) for run in segment["run_numbers"]),
                        structure=str(segment["structure"]),
                        ic=float(segment["ic"]),
                        excluded=bool(segment["excluded"]),
                    )
                    for segment in entry["segments"]
                ),
                total_ic=float(entry["total_ic"]),
                gain=float(entry["gain"]),
                admissible=bool(entry["admissible"]),
                boundaries=tuple(
                    (float(estimate), float(half_gap)) for estimate, half_gap in entry["boundaries"]
                ),
            )
            for entry in payload["solutions"]
        )
        return cls(
            solutions=solutions,
            selected_k=int(payload["selected_k"]),
            beta_floor=float(payload["beta_floor"]),
        )


# --------------------------------------------------------------------------- #
# Segment cost tiers
# --------------------------------------------------------------------------- #


def _template_keys(table: Mapping[int, Mapping[str, float]], runs: Sequence[int]) -> list[str]:
    """Every template key the table offers for ``runs``, sorted for determinism."""

    keys: set[str] = set()
    for run in runs:
        keys |= set(table.get(run, {}))
    return sorted(keys)


def _template_total(
    table: Mapping[int, Mapping[str, float]], runs: Sequence[int], template: str
) -> float:
    """Σ per-run IC of ``template`` over ``runs``; ``inf`` if any cell is unusable."""

    total = 0.0
    for run in runs:
        value = float(table.get(run, {}).get(template, math.inf))
        if not math.isfinite(value):
            return math.inf
        total += value
    return total


def tier1_segment_cost(
    table: Mapping[int, Mapping[str, float]],
    order: Sequence[int],
    *,
    family_of: Callable[[str], str] | None = None,
) -> SegmentCost:
    """Segment cost from the per-run per-template IC table (the all-local bound).

    ``table[run][template_key]`` is that run's IC for that template; a missing or
    non-finite cell makes the template infeasible on any segment containing the
    run. The segment cost is the cheapest template's sum and the structure key is
    that template's *family* under ``family_of`` (the template key itself when
    no ``family_of`` is given).
    """

    runs = tuple(order)
    structure_of = family_of if family_of is not None else _identity
    cache: dict[tuple[int, int], tuple[float, str]] = {}

    def cost(start: int, stop: int) -> tuple[float, str]:
        key = (start, stop)
        if key not in cache:
            window = runs[start:stop]
            best = (math.inf, "")
            for template in _template_keys(table, window):
                total = _template_total(table, window, template)
                if total < best[0]:
                    best = (total, structure_of(template))
            cache[key] = best
        return cache[key]

    return cost


def _identity(template: str) -> str:
    return template


def tier2_segment_cost(
    table: Mapping[int, Mapping[str, float]],
    order: Sequence[int],
    estimates_by_template: Mapping[str, Mapping[int, RunEstimate]],
    metric: SelectionMetric,
    *,
    family_of: Callable[[str], str] | None = None,
) -> SegmentCost:
    """Tier 1 plus the GLS collapse of the best sharing pattern per template.

    Feasibility still comes from ``table`` (a missing/non-finite cell rules the
    template out for that segment); the *cost* comes from
    :func:`~asymmetry.core.fitting.global_search.surrogate.greedy_assignment` over
    that template's :class:`RunEstimate`\\ s restricted to the segment. When a
    template has no estimates for every run of the segment its tier-1 all-local
    sum is used instead, which is the lower bound tier 1 already provides.

    The structure key is the winning template's family under ``family_of`` (the
    template key when none is given). The sharing pattern the collapse chose is
    *priced* into the cost but is deliberately not part of the key: which
    parameters a phase shares is decided by the coupled search within the
    phase, and a change in it is not a transition.

    **The cost had to come down to be usable.** The dynamic program asks for every
    ``O(G²)`` window and scores every window for every template: on a 29-run series
    with 24 templates and ``P = 9`` that is ~9 000 rankings, and enumerating
    ``2^P`` subsets per ranking took the best part of a minute. Three things carry
    it instead:

    * :func:`~asymmetry.core.fitting.global_search.surrogate.greedy_assignment`'s
      forward selection — ``P(P+1)/2`` collapses, not ``2^P``, and the same subset
      whenever the parameters score separably;
    * :class:`~asymmetry.core.fitting.global_search.surrogate.OrderedCollapse`, one
      per template, which holds each subset's prefix sums so a *window* costs one
      small solve rather than a pass over its runs;
    * an exact per-window bound across templates. ``Σ_r χ²_r + penalty(P, n)`` is
      the cheapest IC any assignment of a template could reach
      (:meth:`~asymmetry.core.fitting.global_search.surrogate.OrderedCollapse.lower_bound_ic`),
      so templates are walked cheapest-floor-first and one whose floor already
      loses to the best scored value is skipped. It cannot win, so the answer is
      unchanged.
    """

    runs = tuple(order)
    structure_of = family_of if family_of is not None else _identity
    cache: dict[tuple[int, int], tuple[float, str]] = {}
    collapse_by_template: dict[str, OrderedCollapse] = {}
    for template, per_run in estimates_by_template.items():
        positional = tuple(per_run.get(run) for run in runs)
        present = [estimate for estimate in positional if estimate is not None]
        if present:
            collapse_by_template[template] = OrderedCollapse(positional, present[0].names)

    def cost(start: int, stop: int) -> tuple[float, str]:
        key = (start, stop)
        if key in cache:
            return cache[key]
        window = runs[start:stop]

        # (floor, tier-1 sum, template): the floor orders the walk and prunes it;
        # a template with no usable estimates here has no floor below its tier-1
        # sum, which is the value it will be scored at anyway.
        candidates: list[tuple[float, float, str]] = []
        for template in _template_keys(table, window):
            tier1_total = _template_total(table, window, template)
            if not math.isfinite(tier1_total):
                continue
            collapse = collapse_by_template.get(template)
            scoreable = collapse is not None and collapse.covers(start, stop)
            floor = collapse.lower_bound_ic(start, stop, metric) if scoreable else tier1_total
            candidates.append((floor, tier1_total, template))
        candidates.sort()

        best = (math.inf, "")
        for floor, tier1_total, template in candidates:
            if floor >= best[0]:
                break
            collapse = collapse_by_template.get(template)
            if collapse is not None and collapse.covers(start, stop):
                shared, value = collapse.greedy(start, stop, collapse.names, metric)
            else:
                shared, value = (), tier1_total
            if value < best[0]:
                best = (value, structure_of(template))
        cache[key] = best
        return best

    return cost


# --------------------------------------------------------------------------- #
# Exact dynamic program
# --------------------------------------------------------------------------- #


def _boundaries(
    segments: Sequence[Segment], axis_values: Mapping[int, float]
) -> tuple[tuple[float, float], ...]:
    """``((x_a + x_b)/2, (x_b − x_a)/2)`` for each adjacent segment pair."""

    estimates: list[tuple[float, float]] = []
    for left, right in zip(segments, segments[1:], strict=False):
        lower = float(axis_values[left.run_numbers[-1]])
        upper = float(axis_values[right.run_numbers[0]])
        estimates.append((0.5 * (lower + upper), 0.5 * (upper - lower)))
    return tuple(estimates)


def partition_series(
    order: Sequence[int],
    axis_values: Mapping[int, float],
    cost: SegmentCost,
    config: PartitionConfig,
    *,
    n_total_points: int,
) -> PartitionPath:
    """Solve the exactly-``k``-breaks DP for ``k = 0 … K_max`` in one pass.

    ``order`` is the run numbers in sweep-axis order (ascending, so a boundary's
    half-gap is non-negative) and ``axis_values`` maps each to its axis value.

    ``K_max = ⌊G / min_segment⌋ − 1``, further capped by ``config.max_breaks``.
    Each ``solutions[k]`` is the exactly-``k`` optimum over partitions whose
    adjacent (non-excluded) segments all differ in structure; the path ends at
    the first ``k`` for which no such partition exists.
    """

    runs = tuple(order)
    total_runs = len(runs)
    minimum = config.min_segment
    max_breaks = max(0, total_runs // minimum - 1)
    if config.max_breaks is not None:
        max_breaks = min(max_breaks, config.max_breaks)

    entry_cache: dict[tuple[int, int], tuple[float, str, bool] | None] = {}

    def entry(start: int, stop: int) -> tuple[float, str, bool] | None:
        """``(ic, structure, excluded)`` for a segment, or ``None`` if not allowed.

        A short segment is allowed only as an end stub, where it is scored as the
        sum of its runs' own single-run best costs — one template per run, no
        coupled fit — because that is exactly what "not part of any phase" means.
        """

        key = (start, stop)
        if key in entry_cache:
            return entry_cache[key]
        length = stop - start
        if length >= minimum:
            ic, structure = cost(start, stop)
            result: tuple[float, str, bool] | None = (ic, structure, False)
        elif start == 0 or stop == total_runs:
            total = 0.0
            keys: list[str] = []
            for index in range(start, stop):
                run_ic, run_structure = cost(index, index + 1)
                total += run_ic
                keys.append(run_structure)
            result = (total, ",".join(dict.fromkeys(keys)), True)
        else:
            result = None
        entry_cache[key] = result
        return result

    infinity = math.inf
    #: DP state per (breaks, stop): the structure key of the *last* segment →
    #: (cost, dropped, split, previous structure). Keeping the last segment's
    #: key in the state is what lets the recursion refuse to place two segments
    #: of one structure side by side — a break is a change of structure, by
    #: construction, not by a post-hoc merge. An end stub obeys the same rule
    #: under the structure its runs' own best templates carry: runs that look
    #: like the neighbouring phase are not "a different phase" and cannot be
    #: excluded. Ties break towards excluding as few runs as possible: a
    #: one-run end stub and a two-run one cost the same whenever the extra run's
    #: own best template differs from the body's the same way, and dropping a
    #: run that the phase describes perfectly well is the worse answer.
    State = tuple[float, int, int, str | None]
    states: list[list[dict[str | None, State]]] = [
        [{} for _ in range(total_runs + 1)] for _ in range(max_breaks + 1)
    ]

    for stop in range(1, total_runs + 1):
        first = entry(0, stop)
        if first is not None:
            ic, structure, excluded = first
            states[0][stop][structure] = (ic, stop if excluded else 0, 0, None)

    for breaks in range(1, max_breaks + 1):
        for stop in range(1, total_runs + 1):
            reached = states[breaks][stop]
            for split in range(1, stop):
                candidate = entry(split, stop)
                if candidate is None:
                    continue
                ic, key, excluded = candidate
                for previous_key, (cost_so_far, dropped_so_far, _, _) in states[breaks - 1][
                    split
                ].items():
                    if previous_key == key:
                        continue
                    total = cost_so_far + ic
                    dropped = dropped_so_far + ((stop - split) if excluded else 0)
                    incumbent = reached.get(key)
                    if incumbent is None or (total, dropped) < (incumbent[0], incumbent[1]):
                        reached[key] = (total, dropped, split, previous_key)

    def best_state(breaks: int) -> tuple[str | None, State] | None:
        candidates = states[breaks][total_runs]
        if not candidates:
            return None
        return min(candidates.items(), key=lambda item: (item[1][0], item[1][1]))

    def reconstruct(breaks: int) -> tuple[Segment, ...]:
        segments: list[Segment] = []
        stop = total_runs
        key, state = best_state(breaks)
        for level in range(breaks, -1, -1):
            _, _, start, previous_key = state
            ic, structure, excluded = entry(start, stop)
            segments.append(
                Segment(
                    start=start,
                    stop=stop,
                    run_numbers=runs[start:stop],
                    structure=structure,
                    ic=ic,
                    excluded=excluded,
                )
            )
            if level > 0:
                state = states[level - 1][start][previous_key]
            stop = start
        segments.reverse()
        return tuple(segments)

    beta_floor = config.floor_coefficient * math.log(max(int(n_total_points), 1))

    solutions: list[PartitionSolution] = []
    previous_total = infinity
    for breaks in range(max_breaks + 1):
        if best_state(breaks) is None:
            break
        segments = reconstruct(breaks)
        total_ic = math.fsum(segment.ic for segment in segments)
        gain = 0.0 if breaks == 0 else previous_total - total_ic
        solutions.append(
            PartitionSolution(
                breaks=breaks,
                segments=segments,
                total_ic=total_ic,
                gain=gain,
                admissible=True if breaks == 0 else gain >= beta_floor,
                boundaries=_boundaries(segments, axis_values),
            )
        )
        previous_total = total_ic

    selected_k = 0
    for index in range(1, len(solutions)):
        if not solutions[index].admissible:
            break
        selected_k = index

    return PartitionPath(solutions=tuple(solutions), selected_k=selected_k, beta_floor=beta_floor)
