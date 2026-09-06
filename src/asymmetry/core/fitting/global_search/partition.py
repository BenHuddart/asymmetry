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

**Breaks are structural.** Adjacent segments with the same structure are merged
after the DP, so a parameter that merely *drifts* can never be approximated by a
staircase of breaks — it has exactly two honest representations, global or
local. Merging can leave the exactly-``k`` solution with fewer than ``k`` breaks;
that partition is then identical to a lower-``k`` one, its gain comes out ``0``,
and the path stops there (see :func:`partition_series`).

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
  (:func:`~asymmetry.core.fitting.global_search.surrogate.rank_assignments`), so
  a break in the *role structure* at a fixed template is visible.

Tier 1 sums per-run ICs (each penalised against that run's own point count);
tier 2 scores the segment as one problem (one penalty against the segment's
total point count). The two are therefore on deliberately different scales —
tier 1 is the cheaper lower bound, tier 2 the number the search runs on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_search.surrogate import (
    RunEstimate,
    rank_assignments,
    surrogate_ic,
)

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
    """The best partition with a given number of DP breaks, after merging.

    ``breaks`` is the count *after* structure merging, so it can be smaller than
    the index this solution sits at in :attr:`PartitionPath.solutions`.
    ``total_ic`` is Σ segment IC with no β term; ``gain`` is ``F_{k−1} − F_k`` on
    those merged totals (``0.0`` at ``k = 0``) and ``admissible`` says whether it
    clears ``β_floor``.
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

    ``solutions[k]`` is the exactly-``k``-break DP solution (after merging).
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
    table: Mapping[int, Mapping[str, float]], order: Sequence[int]
) -> SegmentCost:
    """Segment cost from the per-run per-template IC table (the all-local bound).

    ``table[run][template_key]`` is that run's IC for that template; a missing or
    non-finite cell makes the template infeasible on any segment containing the
    run. The segment cost is the cheapest template's sum and the structure key is
    that template key.
    """

    runs = tuple(order)
    cache: dict[tuple[int, int], tuple[float, str]] = {}

    def cost(start: int, stop: int) -> tuple[float, str]:
        key = (start, stop)
        if key not in cache:
            window = runs[start:stop]
            best = (math.inf, "")
            for template in _template_keys(table, window):
                total = _template_total(table, window, template)
                if total < best[0]:
                    best = (total, template)
            cache[key] = best
        return cache[key]

    return cost


def tier2_segment_cost(
    table: Mapping[int, Mapping[str, float]],
    order: Sequence[int],
    estimates_by_template: Mapping[str, Mapping[int, RunEstimate]],
    metric: SelectionMetric,
) -> SegmentCost:
    """Tier 1 plus the GLS collapse of the best sharing pattern per template.

    Feasibility still comes from ``table`` (a missing/non-finite cell rules the
    template out for that segment); the *cost* comes from
    :func:`~asymmetry.core.fitting.global_search.surrogate.rank_assignments` over
    that template's :class:`RunEstimate`\\ s restricted to the segment. When a
    template has no estimates for every run of the segment its tier-1 all-local
    sum is used instead, which is the lower bound tier 1 already provides.

    The structure key is ``f"{template}|g={','.join(shared) or 'none'}"``, so a
    role change at a fixed template is a different structure and therefore a
    break.
    """

    runs = tuple(order)
    cache: dict[tuple[int, int], tuple[float, str]] = {}

    def cost(start: int, stop: int) -> tuple[float, str]:
        key = (start, stop)
        if key in cache:
            return cache[key]
        window = runs[start:stop]
        best = (math.inf, "")
        for template in _template_keys(table, window):
            tier1_total = _template_total(table, window, template)
            if not math.isfinite(tier1_total):
                continue
            per_run = estimates_by_template.get(template, {})
            estimates = [per_run[run] for run in window if run in per_run]
            if len(estimates) == len(window) and estimates:
                # All-local is scored explicitly: above ``max_enumerated``
                # candidates ``rank_assignments`` returns only the singletons.
                shared: tuple[str, ...] = ()
                value = surrogate_ic(estimates, (), metric)
                for candidate, candidate_ic in rank_assignments(
                    estimates, estimates[0].names, metric
                ):
                    if candidate_ic < value:
                        shared, value = candidate, candidate_ic
            else:
                shared, value = (), tier1_total
            structure = f"{template}|g={','.join(shared) or 'none'}"
            if value < best[0]:
                best = (value, structure)
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


def _merge_equal_structures(
    segments: Sequence[Segment], cost: SegmentCost, runs: Sequence[int]
) -> tuple[Segment, ...]:
    """Fuse adjacent non-excluded segments that share a structure key.

    The fused segment's cost is recomputed with ``cost`` — for tier 2 the best
    sharing pattern over the wider window is not the sum of the two narrower
    ones, and it may even name a different structure, which the next iteration
    then compares against.
    """

    merged: list[Segment] = []
    for segment in segments:
        if (
            merged
            and not merged[-1].excluded
            and not segment.excluded
            and merged[-1].structure == segment.structure
        ):
            previous = merged.pop()
            start, stop = previous.start, segment.stop
            ic, structure = cost(start, stop)
            merged.append(
                Segment(
                    start=start,
                    stop=stop,
                    run_numbers=tuple(runs[start:stop]),
                    structure=structure,
                    ic=ic,
                    excluded=False,
                )
            )
        else:
            merged.append(segment)
    return tuple(merged)


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
    Each ``solutions[k]`` is the exactly-``k`` optimum with adjacent
    equal-structure segments merged; when merging drops the break count the
    solution is identical to a lower-``k`` one and its gain is ``0``, which
    stops the path at that step.
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
    best_cost = [[infinity] * (total_runs + 1) for _ in range(max_breaks + 1)]
    # Ties are broken towards excluding as few runs as possible: a one-run end
    # stub and a two-run one cost the same whenever the extra run's own best
    # template is the body's, and dropping a run that the phase describes
    # perfectly well is the worse answer.
    dropped = [[0] * (total_runs + 1) for _ in range(max_breaks + 1)]
    predecessor: list[list[int | None]] = [[None] * (total_runs + 1) for _ in range(max_breaks + 1)]

    for stop in range(1, total_runs + 1):
        first = entry(0, stop)
        if first is not None:
            best_cost[0][stop] = first[0]
            dropped[0][stop] = stop if first[2] else 0
            predecessor[0][stop] = 0

    for breaks in range(1, max_breaks + 1):
        for stop in range(1, total_runs + 1):
            chosen: int | None = None
            chosen_key = (infinity, 0)
            for split in range(1, stop):
                if predecessor[breaks - 1][split] is None:
                    continue
                candidate = entry(split, stop)
                if candidate is None:
                    continue
                key = (
                    best_cost[breaks - 1][split] + candidate[0],
                    dropped[breaks - 1][split] + ((stop - split) if candidate[2] else 0),
                )
                if chosen is None or key < chosen_key:
                    chosen, chosen_key = split, key
            best_cost[breaks][stop] = chosen_key[0]
            dropped[breaks][stop] = chosen_key[1]
            predecessor[breaks][stop] = chosen

    def reconstruct(breaks: int) -> tuple[Segment, ...]:
        segments: list[Segment] = []
        stop = total_runs
        for level in range(breaks, -1, -1):
            start = predecessor[level][stop]
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
            stop = start
        segments.reverse()
        return tuple(segments)

    beta_floor = config.floor_coefficient * math.log(max(int(n_total_points), 1))

    solutions: list[PartitionSolution] = []
    previous_total = infinity
    for breaks in range(max_breaks + 1):
        if predecessor[breaks][total_runs] is None:
            break
        merged = _merge_equal_structures(reconstruct(breaks), cost, runs)
        total_ic = math.fsum(segment.ic for segment in merged)
        gain = 0.0 if breaks == 0 else previous_total - total_ic
        solutions.append(
            PartitionSolution(
                breaks=len(merged) - 1,
                segments=merged,
                total_ic=total_ic,
                gain=gain,
                admissible=True if breaks == 0 else gain >= beta_floor,
                boundaries=_boundaries(merged, axis_values),
            )
        )
        previous_total = total_ic

    selected_k = 0
    for index in range(1, len(solutions)):
        if not solutions[index].admissible:
            break
        selected_k = index

    return PartitionPath(solutions=tuple(solutions), selected_k=selected_k, beta_floor=beta_floor)
