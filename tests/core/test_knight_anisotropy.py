"""Angle-dependent Knight-shift K(θ) basis models (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    Parameter,
    ParameterCompositeModel,
    ParameterSet,
    component_names_for_x,
    fit_parameter_model,
)


def test_angle_scope_offers_anisotropy_models_only_for_angle():
    angle = set(component_names_for_x("angle"))
    assert {"KnightAnisotropy", "AngularCos2"} <= angle
    assert "Linear" in angle  # common models still available
    assert "OrderParameter" not in angle  # temperature-only stays hidden
    # The anisotropy models are NOT offered for non-angle axes.
    assert "KnightAnisotropy" not in set(component_names_for_x("temperature"))
    assert "AngularCos2" not in set(component_names_for_x("field"))


def test_knight_anisotropy_values_at_key_angles():
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    theta = np.array([0.0, 90.0, 54.7356103])  # 0, perpendicular, magic angle
    y = fn(theta, K_iso=100.0, K_ax=30.0)
    assert y[0] == pytest.approx(100.0 + 30.0)  # (3-1)/2 = 1
    assert y[1] == pytest.approx(100.0 - 15.0)  # (0-1)/2 = -1/2
    assert y[2] == pytest.approx(100.0, abs=1e-3)  # 3cos²θ-1 = 0 at the magic angle


def test_angular_cos2_values():
    fn = PARAMETER_MODEL_COMPONENTS["AngularCos2"].function
    y = fn(np.array([20.0, 110.0]), K_avg=5.0, K_amp=2.0, theta0=20.0)
    assert y[0] == pytest.approx(5.0 + 2.0)  # at theta0 → +amp
    assert y[1] == pytest.approx(5.0 - 2.0)  # 90° away → −amp


def test_fit_recovers_anisotropy_parameters():
    theta = np.linspace(-90.0, 90.0, 19)
    truth = {"K_iso": 120.0, "K_ax": 40.0}
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    y = fn(theta, **truth)
    errs = np.full_like(y, 0.5)

    model = ParameterCompositeModel(["KnightAnisotropy"])
    params = ParameterSet([Parameter("K_iso", 100.0), Parameter("K_ax", 10.0)])
    result = fit_parameter_model(theta, y, errs, model, params)

    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["K_iso"] == pytest.approx(120.0, abs=1e-2)
    assert fitted["K_ax"] == pytest.approx(40.0, abs=1e-2)


# --- theta0 (mount/zero misalignment) ---------------------------------------


def test_knight_anisotropy_theta0_default_matches_old_two_arg_form():
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    theta = np.linspace(-90.0, 90.0, 13)
    with_default = fn(theta, K_iso=100.0, K_ax=30.0, theta0=0.0)
    two_arg = fn(theta, K_iso=100.0, K_ax=30.0)
    np.testing.assert_allclose(with_default, two_arg)


def test_knight_anisotropy_theta0_shifts_the_extremum():
    fn = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"].function
    theta0 = 17.0
    delta = 8.0
    # K(theta) is symmetric about its extremum, and its extremum value is
    # K_iso + K_ax (theta = theta0, where cos^2 = 1).
    left = fn(np.array([theta0 - delta]), K_iso=100.0, K_ax=30.0, theta0=theta0)[0]
    right = fn(np.array([theta0 + delta]), K_iso=100.0, K_ax=30.0, theta0=theta0)[0]
    peak = fn(np.array([theta0]), K_iso=100.0, K_ax=30.0, theta0=theta0)[0]
    assert left == pytest.approx(right)
    assert peak == pytest.approx(100.0 + 30.0)


def test_knight_anisotropy_registry_lists_theta0_param():
    definition = PARAMETER_MODEL_COMPONENTS["KnightAnisotropy"]
    assert definition.param_names == ["K_iso", "K_ax", "theta0"]
