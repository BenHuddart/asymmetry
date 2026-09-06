"""Tests for the exact structural partition of an ordered run series."""

from __future__ import annotations

import json
import math

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
from asymmetry.core.fitting.global_search.surrogate import RunEstimate

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

    # A floor of zero admits every non-negative gain, including the merged
    # duplicates whose gain is exactly 0.
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

    assert len(path.solutions) == 12 // 4  # K_max = 2, so k = 0, 1, 2
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


def test_tier2_finds_a_role_structure_change_at_a_fixed_template():
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

    assert path.selected_k == 1
    solution = path.solutions[1]
    assert [segment.structure for segment in solution.segments] == ["T|g=a", "T|g=b"]
    assert solution.segments[0].run_numbers == (0, 1, 2, 3, 4, 5)
    assert solution.segments[1].run_numbers == (6, 7, 8, 9, 10, 11)
    ((estimate, half_gap),) = solution.boundaries
    assert (estimate, half_gap) == pytest.approx((5.5, 0.5))


def test_a_smoothly_drifting_global_yields_no_breaks():
    order = list(range(9))
    names = ("a",)
    estimates = {run: _estimate(run, names, [1.0 + 0.05 * run], [0.1]) for run in order}

    cost = tier2_segment_cost(_feasible_table(order), order, {"T": estimates}, METRIC)
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=9 * 200)

    assert path.selected_k == 0
    assert path.solutions[0].segments[0].structure == "T|g=a"
    assert path.solutions[0].breaks == 0


def test_a_global_that_only_changes_value_is_merged_back_to_one_segment():
    """Rule 1 of the plan: a break is structural, never merely a value change."""

    order = list(range(10))
    names = ("a",)
    estimates = {run: _estimate(run, names, [1.0 if run < 5 else 5.0], [0.1]) for run in order}

    cost = tier2_segment_cost(_feasible_table(order), order, {"T": estimates}, METRIC)
    path = partition_series(order, _axis(order), cost, PartitionConfig(), n_total_points=10 * 200)

    assert path.selected_k == 0
    # The exactly-one-break optimum has both halves sharing ``a`` — one
    # structure — so merging collapses it back onto the k = 0 partition and its
    # gain is exactly zero.
    assert path.solutions[1].breaks == 0
    assert path.solutions[1].total_ic == pytest.approx(path.solutions[0].total_ic)
    assert path.solutions[1].gain == pytest.approx(0.0)


def test_tier2_falls_back_to_the_tier1_sum_when_a_template_has_no_estimates():
    order = list(range(6))
    table = {run: {"T": 250.0, "U": 300.0} for run in order}

    cost = tier2_segment_cost(table, order, {}, METRIC)

    ic, structure = cost(0, 6)
    assert ic == pytest.approx(6 * 250.0)
    assert structure == "T|g=none"


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
