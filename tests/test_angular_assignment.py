"""Joint K(θ) fit with per-angle component assignment (core)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.angular_assignment import fit_assigned_angular_curves
from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS


def _axial(theta_deg, k_iso, k_ax):
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    return fn(np.asarray(theta_deg, dtype=float), K_iso=k_iso, K_ax=k_ax)


def test_clean_two_curve_scan_recovers_curves_and_identity():
    angles = np.linspace(-90.0, 90.0, 19)
    a = _axial(angles, 120.0, 40.0)  # well-separated everywhere
    b = _axial(angles, -50.0, 10.0)
    values = np.column_stack([a, b])

    result = fit_assigned_angular_curves(angles, values, model_name="KnightAnisotropy")

    assert result.success and result.converged
    # No crossing → assignment stays the identity at every point.
    assert all(perm == (0, 1) for perm in result.assignment)
    params = [{p.name: p.value for p in c.parameters} for c in result.curves]
    recovered = sorted((round(p["K_iso"], 2), round(p["K_ax"], 2)) for p in params)
    assert recovered == sorted([(120.0, 40.0), (-50.0, 10.0)])


def test_crossing_recovers_continuous_curves_with_swapped_assignment():
    # Two curves crossing once at the magic angle (~54.7°) over 0–90°. The grouped
    # fit emits value-continuous traces but swaps the labels past the crossing
    # (as iminuit relabels the near-degenerate components). The joint fit must
    # un-swap them into the two continuous physical curves and record the swap.
    angles = np.linspace(0.0, 90.0, 19)
    curve_a = _axial(angles, 100.0, 60.0)  # a > b below ~54.7°, a < b above
    curve_b = _axial(angles, 100.0, -20.0)
    past = angles > 54.7356103  # swap labels past the (value-continuous) crossing
    comp0 = np.where(past, curve_b, curve_a)
    comp1 = np.where(past, curve_a, curve_b)
    raw = np.column_stack([comp0, comp1])

    result = fit_assigned_angular_curves(angles, raw, model_name="KnightAnisotropy")

    assert result.success
    # Recovers the two physical curves (continuous), not the scrambled labels.
    params = sorted(
        round(p.value, 1) for c in result.curves for p in c.parameters if p.name == "K_ax"
    )
    assert params == sorted([-20.0, 60.0])
    # The assignment departs from the identity past the crossing (the swap).
    assert any(perm != (0, 1) for perm in result.assignment)


def test_weighting_downweights_noisy_component():
    angles = np.linspace(-90.0, 90.0, 19)
    a = _axial(angles, 80.0, 30.0)
    b = _axial(angles, -40.0, 15.0)
    values = np.column_stack([a, b])
    values[5, 0] += 50.0  # one wild point on curve a...
    errors = np.ones_like(values)
    errors[5, 0] = 1e3  # ...but with a huge error bar

    result = fit_assigned_angular_curves(angles, values, errors, model_name="KnightAnisotropy")
    fitted = {round(p.value, 1) for c in result.curves for p in c.parameters if p.name == "K_iso"}
    assert 80.0 in {round(v) for v in fitted} or any(abs(v - 80.0) < 2.0 for v in fitted)


def test_three_curve_case():
    angles = np.linspace(-90.0, 90.0, 25)
    values = np.column_stack(
        [_axial(angles, 100.0, 50.0), _axial(angles, 0.0, 20.0), _axial(angles, -80.0, 5.0)]
    )
    result = fit_assigned_angular_curves(angles, values, model_name="KnightAnisotropy")
    assert result.success
    assert len(result.curves) == 3
    assert len(result.curve_values) == 3 and len(result.curve_values[0]) == len(angles)


def test_too_few_points_returns_unconverged_empty():
    # Fewer points than the model's free parameters → cannot fit.
    result = fit_assigned_angular_curves([0.0], [[1.0, 2.0]], model_name="KnightAnisotropy")
    assert not result.success
    assert not result.converged
    assert result.curves == []


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown angular model"):
        fit_assigned_angular_curves([0.0, 30.0, 60.0], [[1.0]] * 3, model_name="Bogus")
