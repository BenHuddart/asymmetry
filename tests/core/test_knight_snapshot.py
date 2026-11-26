"""Tests for building a Knight-shift snapshot from trend rows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from asymmetry.core.fitting.knight_analysis import snapshot_from_rows


@dataclass
class _Row:
    run_number: int
    run_label: str
    field: float
    values: dict
    errors: dict
    covariance: dict | None = None
    include_in_trend: bool = True
    fixed_parameters: frozenset[str] = frozenset()


def test_snapshot_from_rows_maps_fields_correctly():
    rows = [
        _Row(
            1, "run1", 7000.0, {"frequency": 95.0}, {"frequency": 0.1}, covariance={"a": {"b": 1.0}}
        ),
        _Row(
            2,
            "run2",
            7050.0,
            {"frequency": 95.5},
            {},
            include_in_trend=False,
            fixed_parameters=frozenset({"frequency"}),
        ),
    ]
    x_values = [10.0, 20.0]

    snapshot = snapshot_from_rows(
        rows,
        x_values=x_values,
        x_key="angle",
        x_label="Angle",
        components=[("frequency", "frequency")],
        source_label="my source",
        batch_id="b1",
        group_id="g1",
    )

    assert snapshot.x_key == "angle"
    assert snapshot.x_label == "Angle"
    assert snapshot.components == (("frequency", "frequency"),)
    assert snapshot.source_label == "my source"
    assert snapshot.batch_id == "b1"
    assert snapshot.group_id == "g1"
    assert len(snapshot.points) == 2

    p0, p1 = snapshot.points
    assert p0.run_number == 1
    assert p0.run_label == "run1"
    assert p0.x == 10.0
    assert p0.field_gauss == 7000.0
    assert p0.values == {"frequency": 95.0}
    assert p0.covariance == {"a": {"b": 1.0}}
    assert p0.include is True

    assert p1.run_number == 2
    assert p1.x == 20.0
    assert p1.include is False
    assert p1.covariance is None
    assert p1.fixed_parameters == frozenset({"frequency"})


def test_snapshot_from_rows_length_mismatch_raises():
    rows = [_Row(1, "run1", 7000.0, {}, {})]
    with pytest.raises(ValueError):
        snapshot_from_rows(rows, x_values=[1.0, 2.0], x_key="angle", x_label="Angle", components=[])


def test_snapshot_from_rows_non_finite_field_and_x_coerced_to_nan():
    rows = [_Row(1, "run1", float("inf"), {}, {})]
    snapshot = snapshot_from_rows(
        rows, x_values=[float("nan")], x_key="angle", x_label="Angle", components=[]
    )
    point = snapshot.points[0]
    assert math.isnan(point.x)
    assert math.isnan(point.field_gauss)
