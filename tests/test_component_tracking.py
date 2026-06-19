"""Core oscillation-component crossing detection (Phase 3b)."""

from __future__ import annotations

from asymmetry.core.fitting.component_tracking import (
    Component,
    CrossingEvent,
    ScanPoint,
    detect_crossings,
    suggest_proximity_tol,
)


def _point(x: float, *freqs: float, amps: tuple[float, ...] | None = None) -> ScanPoint:
    comps = tuple(Component(f, amplitude=(amps[i] if amps else None)) for i, f in enumerate(freqs))
    return ScanPoint(x, comps)


def _kinds(events: list[CrossingEvent]) -> set[str]:
    return {e.kind for e in events}


def test_clean_separated_scan_has_no_events():
    # Two well-separated components drifting slowly: no swaps, no degeneracy.
    points = [_point(x, 20.0 + 0.1 * x, 100.0 - 0.1 * x) for x in (0, 30, 60, 90)]
    assert detect_crossings(points) == []


def test_genuine_order_swap_is_flagged():
    # The fit swaps labels across a crossing: low freq carries label 1, high label 0.
    points = [
        _point(0.0, 10.0, 20.0),
        _point(30.0, 21.0, 11.0),  # labels swapped relative to continuity
    ]
    events = detect_crossings(points)
    assert "order_swap" in _kinds(events)
    swap = next(e for e in events if e.kind == "order_swap")
    assert swap.component_pair == (0, 1)
    # The frequencies also cross, so the transition is near-degenerate too.
    assert "near_degenerate" in _kinds(events)


def test_near_degeneracy_without_swap_is_flagged_not_swapped():
    # Two components approach but neither cross nor change label order.
    points = [
        _point(0.0, 10.0, 10.5),
        _point(30.0, 10.1, 10.4),
    ]
    events = detect_crossings(points, proximity_tol=1.0)
    assert _kinds(events) == {"near_degenerate"}


def test_three_component_swap():
    # Middle and top components swap; the lowest stays put.
    points = [
        _point(0.0, 5.0, 10.0, 12.0),
        _point(30.0, 5.1, 12.5, 9.5),  # comp1<->comp2 swapped by continuity
    ]
    events = detect_crossings(points)
    swaps = [e for e in events if e.kind == "order_swap"]
    assert any(e.component_pair == (1, 2) for e in swaps)
    assert all(e.component_pair != (0, 1) and e.component_pair != (0, 2) for e in swaps)


def test_three_cycle_swap_is_flagged():
    # A 3-cycle relabelling (comp0→1, comp1→2, comp2→0) is not a transposition;
    # it must still be reported, not silently missed.
    points = [
        _point(0.0, 10.0, 20.0, 30.0),
        _point(30.0, 29.0, 11.0, 21.0),  # continuity-best perm is the cycle (2,0,1)
    ]
    events = detect_crossings(points)
    swaps = {e.component_pair for e in events if e.kind == "order_swap"}
    # The cycle touches all three components, so all three pairs are reported.
    assert swaps == {(0, 1), (0, 2), (1, 2)}


def test_amplitude_breaks_frequency_ties():
    # Frequencies are identical at the right point, so amplitude continuity decides:
    # comp0 (amp 0.5) should match the amp-0.5 right component (the swapped one).
    points = [
        ScanPoint(0.0, (Component(10.0, amplitude=0.5), Component(10.0, amplitude=0.1))),
        ScanPoint(30.0, (Component(10.0, amplitude=0.1), Component(10.0, amplitude=0.5))),
    ]
    events = detect_crossings(points, proximity_tol=0.0)
    assert any(e.kind == "order_swap" and e.component_pair == (0, 1) for e in events)


def test_non_finite_frequency_skips_transition():
    points = [
        _point(0.0, 10.0, 20.0),
        _point(30.0, float("nan"), 11.0),  # a failed component
        _point(60.0, 21.0, 12.0),
    ]
    # Both transitions touch the bad point → no reliable matching → no events.
    assert detect_crossings(points) == []


def test_mismatched_component_count_skips_transition():
    points = [_point(0.0, 10.0, 20.0), _point(30.0, 15.0)]
    assert detect_crossings(points) == []


def test_points_are_sorted_by_x():
    # Out-of-order input is ordered by x; the swap is detected on the real adjacency.
    points = [
        _point(30.0, 21.0, 11.0),
        _point(0.0, 10.0, 20.0),
    ]
    events = detect_crossings(points)
    swap = next(e for e in events if e.kind == "order_swap")
    assert swap.x_left == 0.0 and swap.x_right == 30.0


def test_empty_and_single_point():
    assert detect_crossings([]) == []
    assert detect_crossings([_point(0.0, 10.0, 20.0)]) == []


def test_suggest_proximity_tol():
    # Single component per point → no spacing → 0.0.
    assert suggest_proximity_tol([_point(0.0, 10.0), _point(1.0, 11.0)]) == 0.0
    # Two components ~90 apart → ~15% of the gap.
    tol = suggest_proximity_tol([_point(0.0, 10.0, 100.0), _point(1.0, 10.0, 100.0)])
    assert tol == 90.0 * 0.15
