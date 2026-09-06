"""Tests for the exact structural partition of an ordered run series."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.global_search.partition import (
    PartitionConfig,
    PartitionPath,
    partition_series,
    tier1_segment_cost,
    tier2_segment_cost,
)
from asymmetry.core.fitting.global_search.surrogate import RunEstimate, rank_assignments

METRIC = SelectionMetric.BIC


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _axis(order):
    """Axis value == run number, so a break between r and r+1 sits at r + 0.5."""

    return {run: float(run) for run in order}


def _template_table(order, cheap_by_run, *, cheap=100.0, dear=140.0, templates=("A", "B")):
    return {
        run: {template: cheap if template == cheap_by_run[run] else dear for template in templates}
        for run in order
    }


def _estimate(run, names, values, sigma, *, chi_squared=190.0, n_points=200):
    values = np.asarray(values, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return RunEstimate(
        run_number=run,
        names=tuple(names),
        values=values,
        covariance=np.diag(sigma**2),
        uncertainties=sigma,
        at_bound=frozenset(),
        chi_squared=chi_squared,
        n_points=n_points,
    )


def _feasible_table(order, template="T"):
    return {run: {template: 1000.0} for run in order}


# --------------------------------------------------------------------------- #
# Tier 1: a planted template change
# --------------------------------------------------------------------------- #


def test_a_planted_template_change_puts_the_elbow_at_one_break():
    order = list(range(10))
    table = _template_table(order, {run: "A" if run < 5 else "B" for run in order})

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=2000,
    )

    assert path.selected_k == 1
    zero, one = path.solutions[0], path.solutions[1]
    assert zero.breaks == 0
    assert zero.total_ic == pytest.approx(5 * 100.0 + 5 * 140.0)
    assert one.breaks == 1
    assert one.total_ic == pytest.approx(1000.0)
    assert [segment.structure for segment in one.segments] == ["A", "B"]
    assert one.segments[0].run_numbers == (0, 1, 2, 3, 4)
    assert one.segments[1].run_numbers == (5, 6, 7, 8, 9)
    assert one.gain == pytest.approx(200.0)
    assert one.admissible


def test_the_break_boundary_is_the_midpoint_and_half_gap_of_the_straddling_runs():
    order = list(range(10))
    table = _template_table(order, {run: "A" if run < 5 else "B" for run in order})
    # A wide gap across the transition, narrow spacing elsewhere.
    axis = {run: float(run) for run in range(5)} | {run: 20.0 + run for run in range(5, 10)}

    path = partition_series(
        order,
        axis,
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=2000,
    )

    ((estimate, half_gap),) = path.solutions[1].boundaries
    assert estimate == pytest.approx((4.0 + 25.0) / 2.0)
    assert half_gap == pytest.approx((25.0 - 4.0) / 2.0)


def test_two_planted_transitions_give_two_breaks_and_non_increasing_gains():
    order = list(range(15))
    table = {}
    for run in order:
        if run < 5:
            table[run] = {"A": 100.0, "B": 140.0, "C": 140.0}
        elif run < 10:
            table[run] = {"A": 140.0, "B": 100.0, "C": 140.0}
        else:
            # A shallower third phase, so g_2 < g_1 and the floor can straddle.
            table[run] = {"A": 115.0, "B": 115.0, "C": 100.0}

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=10_000,
    )

    assert path.selected_k == 2
    assert [solution.breaks for solution in path.solutions[:3]] == [0, 1, 2]
    assert path.solutions[1].gain == pytest.approx(200.0)
    assert path.solutions[2].gain == pytest.approx(75.0)
    gains = [solution.gain for solution in path.solutions[1:]]
    assert gains == sorted(gains, reverse=True)
    assert [segment.structure for segment in path.solutions[2].segments] == ["A", "B", "C"]
    assert [segment.run_numbers[0] for segment in path.solutions[2].segments] == [0, 5, 10]


def test_selected_k_stops_at_the_first_gain_below_the_floor():
    order = list(range(15))
    table = {}
    for run in order:
        if run < 5:
            table[run] = {"A": 100.0, "B": 140.0, "C": 140.0}
        elif run < 10:
            table[run] = {"A": 140.0, "B": 100.0, "C": 140.0}
        else:
            table[run] = {"A": 115.0, "B": 115.0, "C": 100.0}
    cost = tier1_segment_cost(table, order)

    # β_floor = 12·ln(10000) ≈ 110.5 sits between g_2 = 75 and g_1 = 200.
    straddling = partition_series(
        order,
        _axis(order),
        cost,
        PartitionConfig(floor_coefficient=12.0),
        n_total_points=10_000,
    )

    assert 75.0 < straddling.beta_floor < 200.0
    assert straddling.solutions[1].admissible
    assert not straddling.solutions[2].admissible
    assert straddling.selected_k == 1

    # A floor of zero admits every non-negative gain.
    permissive = partition_series(
        order,
        _axis(order),
        cost,
        PartitionConfig(floor_coefficient=0.0),
        n_total_points=10_000,
    )
    assert permissive.beta_floor == 0.0
    assert permissive.selected_k >= 2


# --------------------------------------------------------------------------- #
# End stubs and the minimum segment length
# --------------------------------------------------------------------------- #


def test_a_lone_run_at_the_top_end_is_excluded_not_absorbed():
    order = list(range(10))
    table = {
        run: ({"A": 100.0, "B": 300.0} if run < 9 else {"A": 300.0, "B": 50.0}) for run in order
    }

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=2000,
    )

    solution = path.solutions[path.selected_k]
    assert path.selected_k == 1
    assert len(solution.segments) == 2
    body, stub = solution.segments
    assert body.run_numbers == tuple(range(9))
    assert body.structure == "A"
    assert not body.excluded
    assert stub.run_numbers == (9,)
    assert stub.excluded
    assert stub.ic == pytest.approx(50.0)
    assert solution.total_ic == pytest.approx(9 * 100.0 + 50.0)


def test_an_interior_odd_run_is_never_excised():
    order = list(range(11))
    table = {
        run: ({"A": 100.0, "B": 300.0} if run != 5 else {"A": 300.0, "B": 50.0}) for run in order
    }

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=2000,
    )

    for solution in path.solutions:
        for segment in solution.segments:
            if segment.excluded:
                assert segment.start == 0 or segment.stop == len(order)
            else:
                assert segment.stop - segment.start >= 3
        owning = [segment for segment in solution.segments if 5 in segment.run_numbers]
        assert len(owning) == 1
        assert len(owning[0].run_numbers) >= 3


def test_every_non_excluded_segment_respects_min_segment():
    order = list(range(12))
    table = _template_table(order, {run: "A" if run < 6 else "B" for run in order})

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(min_segment=4),
        n_total_points=2400,
    )

    # K_max = 2, but with only two structures a third segment would have to
    # repeat one of them next to itself (a stub of the body's own structure is
    # no different), so the path holds k = 0 and k = 1 only.
    assert len(path.solutions) == 2
    for solution in path.solutions:
        for segment in solution.segments:
            if not segment.excluded:
                assert segment.stop - segment.start >= 4


def test_max_breaks_caps_the_path():
    order = list(range(15))
    table = _template_table(order, {run: "A" if run < 5 else "B" for run in order})

    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(max_breaks=1),
        n_total_points=3000,
    )

    assert len(path.solutions) == 2


# --------------------------------------------------------------------------- #
# Tier 2: role structure
# --------------------------------------------------------------------------- #


def test_a_sharing_change_at_a_fixed_template_is_not_a_break():
    """Which parameters a phase shares is decided within the phase, never a break."""

    order = list(range(12))
    names = ("a", "b")
    sigma = [0.1, 0.1]
    estimates = {}
    for run in order:
        if run < 6:
            # ``a`` is shared here, ``b`` scatters.
            values = [1.0, 0.5 + run]
        else:
            # ... and the roles swap.
            values = [10.0 + run, 7.0]
        estimates[run] = _estimate(run, names, values, sigma)

    cost = tier2_segment_cost(
        _feasible_table(order),
        order,
        {"T": estimates},
        METRIC,
    )
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=12 * 200)

    assert path.selected_k == 0
    # One template is one structure, so no partition with an interior break
    # exists; the only solutions past k = 0 are end stubs, and a stub whose runs
    # carry the body's own structure is never allowed either.
    assert len(path.solutions) == 1
    assert path.solutions[0].segments[0].structure == "T"
    # The sharing is still *priced*: the whole-series segment shares ``b``'s
    # partner nowhere near as cheaply as the two halves would, but that is a
    # role question for the coupled fit, not a transition.
    assert cost(0, 6)[0] + cost(6, 12)[0] < cost(0, 12)[0]


def test_a_smoothly_drifting_global_yields_no_breaks():
    order = list(range(9))
    names = ("a",)
    estimates = {run: _estimate(run, names, [1.0 + 0.05 * run], [0.1]) for run in order}

    cost = tier2_segment_cost(_feasible_table(order), order, {"T": estimates}, METRIC)
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=9 * 200)

    assert path.selected_k == 0
    assert path.solutions[0].segments[0].structure == "T"
    assert path.solutions[0].breaks == 0
    # One template is one structure, so no partition with a break exists.
    assert len(path.solutions) == 1


def test_a_global_that_only_changes_value_is_not_a_break():
    """Rule 1 of the plan: a break is structural, never merely a value change."""

    order = list(range(10))
    names = ("a",)
    estimates = {run: _estimate(run, names, [1.0 if run < 5 else 5.0], [0.1]) for run in order}

    cost = tier2_segment_cost(_feasible_table(order), order, {"T": estimates}, METRIC)
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=10 * 200)

    assert path.selected_k == 0
    # Both halves would carry the same structure, so the exactly-one-break
    # partition does not exist and the path ends at k = 0.
    assert len(path.solutions) == 1


def test_a_template_change_within_one_family_is_not_a_break_but_across_families_is():
    order = list(range(15))
    table = {}
    for run in order:
        if run < 5:
            table[run] = {"osc2": 100.0, "osc1": 130.0, "relax": 160.0}
        elif run < 10:
            table[run] = {"osc2": 130.0, "osc1": 100.0, "relax": 160.0}
        else:
            table[run] = {"osc2": 160.0, "osc1": 160.0, "relax": 100.0}
    family = {"osc2": "oscillatory", "osc1": "oscillatory", "relax": "relaxation"}

    by_template = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=15 * 200,
    )
    by_family = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order, family_of=family.__getitem__),
        PartitionConfig(),
        n_total_points=15 * 200,
    )

    # Template-level structure sees the osc2 → osc1 switch as a break ...
    assert by_template.solutions[2].breaks == 2
    # ... family-level structure does not: the ordered phase is one phase, and
    # the only break is where the family changes.
    assert [segment.structure for segment in by_family.solutions[1].segments] == [
        "oscillatory",
        "relaxation",
    ]
    assert by_family.solutions[1].segments[0].run_numbers == tuple(range(10))
    assert len(by_family.solutions) == 2


def test_tier2_falls_back_to_the_tier1_sum_when_a_template_has_no_estimates():
    order = list(range(6))
    table = {run: {"T": 250.0, "U": 300.0} for run in order}

    cost = tier2_segment_cost(table, order, {}, METRIC)

    ic, structure = cost(0, 6)
    assert ic == pytest.approx(6 * 250.0)
    assert structure == "T"


def test_tier2_marks_a_template_infeasible_when_a_cell_is_missing():
    order = list(range(6))
    table = {run: ({"T": 250.0} if run != 3 else {}) for run in order}

    cost = tier2_segment_cost(table, order, {}, METRIC)

    assert cost(0, 6) == (math.inf, "")
    assert cost(0, 3)[0] == pytest.approx(750.0)


def test_tier1_marks_a_template_infeasible_when_a_cell_is_not_finite():
    order = list(range(6))
    table = {run: {"A": (math.inf if run == 2 else 100.0), "B": 130.0} for run in order}

    cost = tier1_segment_cost(table, order)

    assert cost(0, 6) == (pytest.approx(6 * 130.0), "B")
    assert cost(3, 6) == (pytest.approx(3 * 100.0), "A")


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def test_partition_path_round_trips_through_a_json_payload():
    order = list(range(10))
    table = _template_table(order, {run: "A" if run < 5 else "B" for run in order})
    path = partition_series(
        order,
        _axis(order),
        tier1_segment_cost(table, order),
        PartitionConfig(),
        n_total_points=2000,
    )

    restored = PartitionPath.from_payload(json.loads(json.dumps(path.to_payload())))

    assert restored == path


# --------------------------------------------------------------------------- #
# Tier 2 must stay cheap enough to run on every window
# --------------------------------------------------------------------------- #


def _series_table(n_runs, n_templates, n_params, *, seed=20260906):
    """A per-run table shaped like a real series alphabet.

    Templates differ in fit quality — one describes the data, the rest describe it
    progressively worse, which is what an alphabet built from a union of per-run
    recommendations looks like — and every template's parameters drift smoothly
    along the sweep rather than jumping about at random.
    """

    rng = np.random.default_rng(seed)
    order = list(range(n_runs))
    names = tuple(f"p{index}" for index in range(n_params))
    templates = [f"t{index}" for index in range(n_templates)]

    table = {run: {} for run in order}
    estimates_by_template = {}
    for index, template in enumerate(templates):
        level = 900.0 * (1.0 + 0.15 * index)
        base = rng.normal(size=n_params)
        slope = rng.normal(size=n_params) * 0.02
        per_run = {}
        for run in order:
            values = base + slope * run + rng.normal(scale=0.01, size=n_params)
            sigma = np.abs(rng.normal(loc=0.2, scale=0.05, size=n_params)) + 0.05
            root = rng.normal(size=(n_params, n_params)) * 0.05
            chi_squared = level + rng.normal(scale=10.0)
            table[run][template] = chi_squared + n_params * math.log(1000)
            per_run[run] = RunEstimate(
                run_number=run,
                names=names,
                values=values,
                covariance=np.diag(sigma**2) + root @ root.T,
                uncertainties=sigma,
                at_bound=frozenset(),
                chi_squared=chi_squared,
                n_points=1000,
            )
        estimates_by_template[template] = per_run
    return order, table, estimates_by_template


def test_a_full_tier2_partition_of_a_long_series_stays_affordable():
    """The bound that made tier 2 usable at all.

    The dynamic program scores every one of ``O(G²)`` windows for every template.
    Enumerating ``2^P`` sharing patterns per window took the best part of a minute
    on a real 29-run series, which is not a cost that can sit in front of a user
    on every screening pass. Forward selection, the prefix-sum windowed collapse
    and the per-window template bound bring it to ~0.4 s here — inside the
    standard tier's budget, which is where a guard against that regression is
    worth having. The assertion is deliberately loose by an order of magnitude,
    so it fails on a regression of *kind* rather than on a slow afternoon.
    """

    order, table, estimates = _series_table(30, 20, 8)
    cost = tier2_segment_cost(table, order, estimates, METRIC)

    started = time.perf_counter()
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=30 * 1000)
    elapsed = time.perf_counter() - started

    assert path.solutions
    assert elapsed < 5.0, f"tier-2 partition of a 30-run, 20-template series took {elapsed:.1f}s"


def test_greedy_segment_costs_agree_with_exhaustive_enumeration():
    """Speed, not a different answer.

    Tier 2 used to score each (template, window) by enumerating every sharing
    pattern. This reproduces that reference directly and requires the shipped
    cost — greedy walk, prefix sums, and the template bound that skips a template
    whose floor already loses — to return the same value and the same structure.
    """

    order, table, estimates = _series_table(9, 4, 3, seed=5)
    cost = tier2_segment_cost(table, order, estimates, METRIC)

    for start in range(len(order)):
        for stop in range(start + 1, len(order) + 1):
            window = order[start:stop]
            reference = (math.inf, "")
            for template, per_run in estimates.items():
                subsets = rank_assignments(
                    [per_run[run] for run in window], estimates[template][window[0]].names, METRIC
                )
                best_subset, best_ic = subsets[0]
                if best_ic < reference[0]:
                    reference = (best_ic, template)
            value, structure = cost(start, stop)
            assert value == pytest.approx(reference[0]), (start, stop)
            assert structure == reference[1], (start, stop)
