"""Angle-folding helper (Phase 4)."""

from __future__ import annotations

import math

import pytest

from asymmetry.core.utils.angles import wrap_angle_deg


def test_fold_to_default_180_period():
    assert wrap_angle_deg(10.0) == pytest.approx(10.0)
    assert wrap_angle_deg(190.0) == pytest.approx(10.0)
    assert wrap_angle_deg(-10.0) == pytest.approx(170.0)
    assert wrap_angle_deg(180.0) == pytest.approx(0.0)
    assert wrap_angle_deg(370.0) == pytest.approx(10.0)


def test_fold_to_360_period():
    assert wrap_angle_deg(370.0, period_deg=360.0) == pytest.approx(10.0)
    assert wrap_angle_deg(-30.0, period_deg=360.0) == pytest.approx(330.0)


def test_symmetric_origin():
    # Fold into [-90, 90) with a 180° period.
    assert wrap_angle_deg(100.0, period_deg=180.0, origin_deg=-90.0) == pytest.approx(-80.0)
    assert wrap_angle_deg(-80.0, period_deg=180.0, origin_deg=-90.0) == pytest.approx(-80.0)


def test_non_finite_and_bad_period_pass_through():
    assert math.isnan(wrap_angle_deg(float("nan")))
    assert math.isinf(wrap_angle_deg(float("inf")))
    assert wrap_angle_deg(42.0, period_deg=0.0) == pytest.approx(42.0)
    assert wrap_angle_deg(42.0, period_deg=-180.0) == pytest.approx(42.0)
