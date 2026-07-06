"""Joint K(θ) fit with per-angle component assignment (core)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from asymmetry.core.fitting.angular_assignment import (
    _canonicalize_theta0,
    fit_assigned_angular_curves,
)
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    Parameter,
    ParameterModelFitResult,
    ParameterSet,
)


def _axial(theta_deg, k_iso, k_ax, theta0=0.0):
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    return fn(np.asarray(theta_deg, dtype=float), K_iso=k_iso, K_ax=k_ax, theta0=theta0)


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


# --- theta0 seeding + canonicalization on a real crossing scan --------------


def test_crossing_scan_with_theta0_offset_recovers_canonical_form_for_both_curves():
    # Same crossing scenario as test_crossing_recovers_continuous_curves_with_
    # swapped_assignment, but with a true mount-misalignment offset (theta0 =
    # 12.5 deg) baked into both physical curves. Both curves share K_iso and
    # theta0; only K_ax's sign distinguishes them. Every curve returned by the
    # joint fit must land in the canonical (small-|theta0|) representation:
    # same theta0 (~12.5), opposite-sign K_ax, same K_iso -- never the folded
    # alternative (K_iso + K_ax/2, -K_ax, theta0-90).
    true_theta0 = 12.5
    angles = np.linspace(0.0, 90.0, 19)
    curve_a = _axial(angles, 100.0, 60.0, theta0=true_theta0)
    curve_b = _axial(angles, 100.0, -20.0, theta0=true_theta0)
    crossing_angle = true_theta0 + 54.7356103
    past = angles > crossing_angle
    comp0 = np.where(past, curve_b, curve_a)
    comp1 = np.where(past, curve_a, curve_b)
    raw = np.column_stack([comp0, comp1])

    result = fit_assigned_angular_curves(angles, raw, model_name="KnightAnisotropy")

    assert result.success
    curve_params = [{p.name: p.value for p in c.parameters} for c in result.curves]
    assert len(curve_params) == 2

    for params in curve_params:
        assert params["theta0"] == pytest.approx(true_theta0, abs=0.5)
        assert params["K_iso"] == pytest.approx(100.0, abs=0.5)
        # Never the folded alternative representation (K_iso + K_ax/2).
        assert params["K_iso"] != pytest.approx(110.0, abs=1.0)
        assert params["K_iso"] != pytest.approx(90.0, abs=1.0)

    k_ax_values = sorted(params["K_ax"] for params in curve_params)
    assert k_ax_values == pytest.approx([-20.0, 60.0], abs=0.5)


# --- _canonicalize_theta0() --------------------------------------------------


def _fit_result(
    theta0: float,
    k_iso: float = 1.0,
    k_ax: float = 2.0,
    k_iso_err: float | None = 0.1,
    k_ax_err: float | None = 0.2,
    param_names: tuple[str, str, str] = ("K_iso", "K_ax", "theta0"),
) -> ParameterModelFitResult:
    uncertainties = {}
    if k_iso_err is not None:
        uncertainties[param_names[0]] = k_iso_err
    if k_ax_err is not None:
        uncertainties[param_names[1]] = k_ax_err
    return ParameterModelFitResult(
        success=True,
        parameters=ParameterSet(
            [
                Parameter(param_names[0], k_iso),
                Parameter(param_names[1], k_ax),
                Parameter("theta0", theta0),
            ]
        ),
        uncertainties=uncertainties,
    )


def test_canonicalize_theta0_folds_100_degrees_for_knight_anisotropy():
    fit = _fit_result(theta0=100.0, k_iso=1.0, k_ax=2.0, k_iso_err=0.1, k_ax_err=0.2)

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta0"] == pytest.approx(10.0)
    assert params["K_ax"] == pytest.approx(-2.0)
    assert params["K_iso"] == pytest.approx(2.0)
    assert fit.uncertainties["K_iso"] == pytest.approx(math.hypot(0.1, 0.1))


def test_canonicalize_theta0_folds_minus_100_degrees_for_knight_anisotropy():
    fit = _fit_result(theta0=-100.0, k_iso=1.0, k_ax=2.0, k_iso_err=0.1, k_ax_err=0.2)

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta0"] == pytest.approx(-10.0)
    assert params["K_ax"] == pytest.approx(-2.0)
    assert params["K_iso"] == pytest.approx(2.0)
    assert fit.uncertainties["K_iso"] == pytest.approx(math.hypot(0.1, 0.1))


def test_canonicalize_theta0_already_in_range_is_unchanged():
    fit = _fit_result(theta0=10.0, k_iso=1.0, k_ax=2.0)

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta0"] == pytest.approx(10.0)
    assert params["K_iso"] == pytest.approx(1.0)
    assert params["K_ax"] == pytest.approx(2.0)


def test_canonicalize_theta0_boundary_45_degrees_is_unchanged():
    # (-45, 45] is half-open on the low end: exactly 45 stays put (no flip).
    fit = _fit_result(theta0=45.0, k_iso=1.0, k_ax=2.0)

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta0"] == pytest.approx(45.0)
    assert params["K_iso"] == pytest.approx(1.0)
    assert params["K_ax"] == pytest.approx(2.0)


def test_canonicalize_theta0_angular_cos2_flips_only_k_amp():
    fit = _fit_result(
        theta0=100.0,
        k_iso=5.0,
        k_ax=3.0,
        k_iso_err=0.1,
        k_ax_err=0.2,
        param_names=("K_avg", "K_amp", "theta0"),
    )

    _canonicalize_theta0("AngularCos2", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta0"] == pytest.approx(10.0)
    assert params["K_amp"] == pytest.approx(-3.0)
    assert params["K_avg"] == pytest.approx(5.0)  # untouched
    assert fit.uncertainties["K_avg"] == pytest.approx(0.1)  # untouched


def test_canonicalize_theta0_nan_is_a_no_op():
    fit = _fit_result(theta0=float("nan"), k_iso=1.0, k_ax=2.0)

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert math.isnan(params["theta0"])
    assert params["K_iso"] == pytest.approx(1.0)
    assert params["K_ax"] == pytest.approx(2.0)


def test_canonicalize_theta0_absent_is_a_no_op():
    fit = ParameterModelFitResult(
        success=True,
        parameters=ParameterSet([Parameter("K_iso", 1.0), Parameter("K_ax", 2.0)]),
        uncertainties={"K_iso": 0.1, "K_ax": 0.2},
    )

    _canonicalize_theta0("KnightAnisotropy", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["K_iso"] == pytest.approx(1.0)
    assert params["K_ax"] == pytest.approx(2.0)
