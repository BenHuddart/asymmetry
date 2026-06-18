"""Detect oscillation-component crossings along an ordered scan (Phase 3b).

When a single crystal is rotated through a transverse-field scan, a multi-component
fit returns the components under stable parameter names (``frequency``,
``frequency_2``, …), but the *physical* assignment can swap when two components'
frequencies approach or cross — the optimiser is free to relabel them. Trending a
derived quantity (e.g. the Knight shift) under the raw names then shows a spurious
discontinuity at the crossing.

This module *detects and flags* those crossings; it does not alter any fitted
result. It computes, between adjacent ordered scan points, the component
assignment that best preserves frequency continuity and reports where that
assignment departs from the raw parameter order (``order_swap``) or where two
frequencies become degenerate (``near_degenerate``).

The frequency-continuity matching here is the foundation a later, opt-in
*realignment* step will build on — that step will additionally be guided by the
fit-function structure (which components share a functional form, their roles and
amplitude relations); it is intentionally out of scope for this module.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from statistics import median

#: A near-degeneracy is flagged when two component frequencies approach within
#: this fraction of the typical within-point component spacing.
_AUTO_TOL_FRACTION = 0.15


@dataclass(frozen=True)
class Component:
    """One oscillation component at a scan point.

    ``amplitude`` and ``damping`` are optional continuity hints used only to break
    ties when frequency continuity alone is ambiguous.
    """

    frequency: float
    amplitude: float | None = None
    damping: float | None = None


@dataclass(frozen=True)
class ScanPoint:
    """The components fitted at one ordered abscissa value (e.g. one angle)."""

    x: float
    components: tuple[Component, ...]


@dataclass(frozen=True)
class CrossingEvent:
    """A flagged crossing/degeneracy between two adjacent scan points.

    ``component_pair`` holds the two raw component indices (stable parameter-name
    order) involved. ``kind`` is ``"order_swap"`` (the continuity-best assignment
    departs from raw order) or ``"near_degenerate"`` (the two frequencies cross or
    come within tolerance). ``min_separation`` is the closest frequency approach
    (MHz) across the transition (0.0 for an actual crossing).
    """

    index_left: int
    x_left: float
    x_right: float
    component_pair: tuple[int, int]
    kind: str
    min_separation: float = field(default=0.0)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def suggest_proximity_tol(points: list[ScanPoint]) -> float:
    """Data-driven near-degeneracy tolerance (MHz).

    A fraction of the median within-point gap between frequency-adjacent
    components — i.e. "closer than ~15% of the typical spacing is degenerate".
    Returns 0.0 when there are too few components to define a spacing (so only
    actual crossings are flagged).
    """
    gaps: list[float] = []
    for point in points:
        freqs = sorted(c.frequency for c in point.components if _finite(c.frequency))
        gaps.extend(b - a for a, b in zip(freqs, freqs[1:]) if b > a)
    if not gaps:
        return 0.0
    return _AUTO_TOL_FRACTION * median(gaps)


def _assignment_cost(
    left: tuple[Component, ...], right: tuple[Component, ...], perm: tuple[int, ...]
) -> tuple[float, float, float]:
    """Lexicographic cost of mapping ``left[i] -> right[perm[i]]``.

    Frequency continuity dominates; amplitude then damping continuity act purely
    as tie-breakers, so no cross-unit scaling is needed.
    """
    freq = amp = damp = 0.0
    for i, j in enumerate(perm):
        a, b = left[i], right[j]
        freq += abs(a.frequency - b.frequency)
        if _finite(a.amplitude) and _finite(b.amplitude):
            amp += abs(a.amplitude - b.amplitude)
        if _finite(a.damping) and _finite(b.damping):
            damp += abs(a.damping - b.damping)
    return freq, amp, damp


def _best_assignment(left: tuple[Component, ...], right: tuple[Component, ...]) -> tuple[int, ...]:
    """Permutation mapping ``left -> right`` that best preserves continuity."""
    n = len(left)
    return min(
        itertools.permutations(range(n)),
        key=lambda perm: _assignment_cost(left, right, perm),
    )


def detect_crossings(
    points: list[ScanPoint], *, proximity_tol: float | None = None
) -> list[CrossingEvent]:
    """Flag component crossings between adjacent ordered scan points.

    ``points`` need not be pre-sorted; they are ordered by ``x`` here (stable).
    ``proximity_tol`` (MHz) controls near-degeneracy flagging; when ``None`` it is
    derived from the data via :func:`suggest_proximity_tol`. A transition is
    skipped when the two points differ in component count or carry a non-finite
    frequency (no reliable matching is possible). Detection only — nothing is
    mutated.
    """
    ordered = sorted((p for p in points if math.isfinite(p.x)), key=lambda p: p.x)
    if len(ordered) < 2:
        return []
    tol = suggest_proximity_tol(ordered) if proximity_tol is None else float(proximity_tol)

    events: list[CrossingEvent] = []
    for k in range(len(ordered) - 1):
        left = ordered[k].components
        right = ordered[k + 1].components
        n = len(left)
        if n == 0 or n != len(right):
            continue
        if not all(_finite(c.frequency) for c in left + right):
            continue

        x_left, x_right = ordered[k].x, ordered[k + 1].x
        perm = _best_assignment(left, right)

        # order_swap: the continuity-best assignment departs from raw order. Report
        # every moved component paired with its destination — this covers 3-cycles
        # and longer cycles, not just simple transpositions. Pairs are deduplicated.
        if perm != tuple(range(n)):
            swapped_pairs = {(min(i, j), max(i, j)) for i, j in enumerate(perm) if i != j}
            for pair in sorted(swapped_pairs):
                events.append(CrossingEvent(k, x_left, x_right, pair, "order_swap"))

        # near_degenerate: a frequency crossing (sign flip of the gap) or an
        # approach within tolerance, evaluated on the raw indices.
        for i, j in itertools.combinations(range(n), 2):
            gap_left = left[i].frequency - left[j].frequency
            gap_right = right[i].frequency - right[j].frequency
            crossed = (gap_left == 0.0 or gap_right == 0.0) or (
                (gap_left > 0.0) != (gap_right > 0.0)
            )
            min_sep = 0.0 if crossed else min(abs(gap_left), abs(gap_right))
            if crossed or min_sep <= tol:
                events.append(CrossingEvent(k, x_left, x_right, (i, j), "near_degenerate", min_sep))
    return events
