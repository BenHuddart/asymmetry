"""Misalignment-aware Knight-shift K(θ) Fourier model (Phase 2, bed-next-angle study)."""

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
    ParameterCompositeModel,
    ParameterModelFitResult,
    ParameterSet,
    component_names_for_x,
    fit_parameter_model,
)


def test_angle_scope_offers_fourier2_only_for_angle():
    angle = set(component_names_for_x("angle"))
    assert "AngularFourier2" in angle
    assert "AngularFourier2" not in set(component_names_for_x("temperature"))
    assert "AngularFourier2" not in set(component_names_for_x("field"))


def test_angular_fourier2_values_at_key_angles():
    # theta1 = 0, theta2 = 45: at theta1 the second-harmonic term sits at its
    # node (2(theta1-theta2) = -90 deg), and at theta2 + 45 the first-harmonic
    # term sits at *its* node (theta - theta1 = 90 deg) too, so every point
    # but one reduces to a pure amplitude read-off; the remaining point uses
    # the standard cos(45 deg) = root(2)/2 identity.
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    truth = {"K_avg": 50.0, "K_1": 8.0, "theta1": 0.0, "K_amp": 12.0, "theta2": 45.0}
    theta1, theta2 = truth["theta1"], truth["theta2"]
    k_avg, k_1, k_amp = truth["K_avg"], truth["K_1"], truth["K_amp"]
    theta = np.array([theta1, theta1 + 180.0, theta2, theta2 + 45.0])
    y = fn(theta, **truth)

    assert y[0] == pytest.approx(k_avg + k_1)  # theta1: 1st harmonic at +amp, 2nd at node
    assert y[1] == pytest.approx(k_avg - k_1)  # theta1+180: 1st harmonic flips sign
    assert y[2] == pytest.approx(k_avg + k_1 * math.sqrt(2.0) / 2.0 + k_amp)  # theta2
    assert y[3] == pytest.approx(k_avg)  # theta2+45: both harmonics at a node


def test_fit_recovers_fourier2_parameters_on_tilted_data():
    rng = np.random.default_rng(42)
    theta = np.linspace(-90.0, 270.0, 37)
    truth = {"K_avg": 50.0, "K_1": 8.0, "theta1": 20.0, "K_amp": 30.0, "theta2": 10.0}
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    y = fn(theta, **truth) + rng.normal(scale=0.5, size=theta.shape)
    errs = np.full_like(theta, 0.5)

    model = ParameterCompositeModel(["AngularFourier2"])
    params = ParameterSet(
        [
            Parameter("K_avg", 0.0),
            Parameter("K_1", 0.0),
            Parameter("theta1", 0.0, min=-90.0, max=90.0),
            Parameter("K_amp", 1.0),
            Parameter("theta2", 0.0, min=-90.0, max=90.0),
        ]
    )
    result = fit_parameter_model(theta, y, errs, model, params)

    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    for name, truth_value in truth.items():
        sigma = result.uncertainties.get(name)
        assert sigma is not None and sigma > 0.0
        assert fitted[name] == pytest.approx(truth_value, abs=5.0 * sigma)


def test_fit_recovers_zero_k1_on_aligned_data():
    # A perfectly aligned axis: truth K_1 = 0. The fitted K_1 should come back
    # consistent with 0 within a few sigma -- the misalignment signature is
    # absent, so the "nested null" start (K_1 = 0) should hold up.
    rng = np.random.default_rng(7)
    theta = np.linspace(-90.0, 270.0, 37)
    truth = {"K_avg": 40.0, "K_1": 0.0, "theta1": 0.0, "K_amp": 15.0, "theta2": 25.0}
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    y = fn(theta, **truth) + rng.normal(scale=0.5, size=theta.shape)
    errs = np.full_like(theta, 0.5)

    model = ParameterCompositeModel(["AngularFourier2"])
    params = ParameterSet(
        [
            Parameter("K_avg", 0.0),
            Parameter("K_1", 0.0),
            Parameter("theta1", 0.0, min=-90.0, max=90.0),
            Parameter("K_amp", 1.0),
            Parameter("theta2", 0.0, min=-90.0, max=90.0),
        ]
    )
    result = fit_parameter_model(theta, y, errs, model, params)

    assert result.success
    fitted_k1 = next(p.value for p in result.parameters if p.name == "K_1")
    sigma_k1 = result.uncertainties.get("K_1")
    assert sigma_k1 is not None and sigma_k1 > 0.0
    assert abs(fitted_k1) < 3.0 * sigma_k1


# --- canonicalisation (theta1/theta2 folds) -----------------------------------


def test_canonicalize_theta0_fourier2_fold_is_exact_on_dense_grid():
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    truth = {"K_avg": 10.0, "K_1": 6.0, "theta1": 150.0, "K_amp": 4.0, "theta2": 100.0}
    grid = np.linspace(-180.0, 540.0, 361)  # dense, spans well past a full period
    before = fn(grid, **truth)

    fit = ParameterModelFitResult(
        success=True,
        parameters=ParameterSet([Parameter(name, value) for name, value in truth.items()]),
    )

    _canonicalize_theta0("AngularFourier2", fit)

    folded = {p.name: p.value for p in fit.parameters}
    assert -90.0 < folded["theta1"] <= 90.0
    assert -45.0 < folded["theta2"] <= 45.0
    after = fn(
        grid,
        K_avg=folded["K_avg"],
        K_1=folded["K_1"],
        theta1=folded["theta1"],
        K_amp=folded["K_amp"],
        theta2=folded["theta2"],
    )
    np.testing.assert_allclose(before, after, atol=1e-9)


def test_canonicalize_theta0_fourier2_already_in_range_is_unchanged():
    fit = ParameterModelFitResult(
        success=True,
        parameters=ParameterSet(
            [
                Parameter("K_avg", 10.0),
                Parameter("K_1", 6.0),
                Parameter("theta1", 30.0),
                Parameter("K_amp", 4.0),
                Parameter("theta2", 15.0),
            ]
        ),
    )

    _canonicalize_theta0("AngularFourier2", fit)

    params = {p.name: p.value for p in fit.parameters}
    assert params["theta1"] == pytest.approx(30.0)
    assert params["K_1"] == pytest.approx(6.0)
    assert params["theta2"] == pytest.approx(15.0)
    assert params["K_amp"] == pytest.approx(4.0)
    assert params["K_avg"] == pytest.approx(10.0)


def test_canonicalize_theta0_fourier2_transforms_covariance_for_both_folds():
    theta1_val, theta2_val = 150.0, 100.0
    fit = ParameterModelFitResult(
        success=True,
        parameters=ParameterSet(
            [
                Parameter("K_avg", 10.0),
                Parameter("K_1", 6.0),
                Parameter("theta1", theta1_val),
                Parameter("K_amp", 4.0),
                Parameter("theta2", theta2_val),
            ]
        ),
        uncertainties={"K_avg": 0.3, "K_1": 0.4, "theta1": 5.0, "K_amp": 0.5, "theta2": 5.0},
    )
    names = ["K_avg", "K_1", "theta1", "K_amp", "theta2"]
    sigma = [
        [0.09, 0.01, 0.0, 0.0, 0.0],
        [0.01, 0.16, 0.0, 0.02, 0.0],
        [0.0, 0.0, 25.0, 0.0, 0.0],
        [0.0, 0.02, 0.0, 0.25, 0.01],
        [0.0, 0.0, 0.0, 0.01, 25.0],
    ]
    fit.covariance = (names, [row[:] for row in sigma])

    _canonicalize_theta0("AngularFourier2", fit)

    params = {p.name: p.value for p in fit.parameters}
    # Both folds fire: theta1 = 150 -> -30 (K_1 flips), theta2 = 100 -> 10
    # (K_amp flips); K_avg untouched by either fold.
    assert params["theta1"] == pytest.approx(-30.0)
    assert params["K_1"] == pytest.approx(-6.0)
    assert params["theta2"] == pytest.approx(10.0)
    assert params["K_amp"] == pytest.approx(-4.0)
    assert params["K_avg"] == pytest.approx(10.0)

    jac = np.diag([1.0, -1.0, 1.0, -1.0, 1.0])
    expected = jac @ np.array(sigma) @ jac.T
    result_names, result_matrix = fit.covariance
    assert result_names == names
    np.testing.assert_allclose(np.array(result_matrix), expected)
    for i, name in enumerate(names):
        assert fit.uncertainties[name] == pytest.approx(math.sqrt(expected[i, i]))


# --- joint fit (fit_assigned_angular_curves) ----------------------------------


def test_fit_assigned_angular_curves_fourier2_joint_fit_recovers_curves():
    fn = PARAMETER_MODEL_COMPONENTS["AngularFourier2"].function
    angles = np.linspace(-90.0, 270.0, 9)
    curve_a = fn(angles, K_avg=80.0, K_1=6.0, theta1=15.0, K_amp=25.0, theta2=5.0)
    curve_b = fn(angles, K_avg=-30.0, K_1=-4.0, theta1=40.0, K_amp=10.0, theta2=-20.0)
    values = np.column_stack([curve_a, curve_b])

    result = fit_assigned_angular_curves(angles, values, model_name="AngularFourier2")

    assert result.success
    assert len(result.curves) == 2
    # Well-separated curves everywhere -> no crossing, identity assignment.
    assert all(perm == (0, 1) for perm in result.assignment)


def test_fit_assigned_angular_curves_fourier2_too_few_points_returns_empty():
    # 4 angles < 5 free parameters -> the existing n_points < n_params guard
    # rejects the scan cleanly instead of attempting an underdetermined fit.
    angles = [0.0, 30.0, 60.0, 90.0]
    values = [[1.0, 2.0]] * 4

    result = fit_assigned_angular_curves(angles, values, model_name="AngularFourier2")

    assert not result.success
    assert not result.converged
    assert result.curves == []
    assert result.message
