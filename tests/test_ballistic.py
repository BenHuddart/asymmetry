"""Tests for ballistic LF relaxation models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning
from scipy.special import j0

from asymmetry.core.fitting.ballistic import (
    autocorrelation_nD,
    lambda_ball,
    lambda_total,
    spectral_density,
)
from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    component_names_for_x,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.widgets.component_info_dialog import build_component_info_html


def test_autocorrelation_shape_and_t0_value() -> None:
    t = np.linspace(0.0, 10.0, 101)
    for n in (1, 2, 3):
        s = autocorrelation_nD(t, D_hop=1.5, n=n)
        assert s.shape == t.shape
        assert np.isclose(s[0], 1.0)
        assert np.all(np.isfinite(s))


def test_autocorrelation_matches_reference_form() -> None:
    t = np.linspace(0.0, 8.0, 200)
    d_hop = 1.2
    for n in (1, 2, 3):
        expected = np.power(j0(2.0 * d_hop * t), 2 * n)
        got = autocorrelation_nD(t, D_hop=d_hop, n=n)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)


def test_autocorrelation_dimension_dependence() -> None:
    t = np.array([0.5, 1.5, 3.0])
    s1 = autocorrelation_nD(t, D_hop=1.0, n=1)
    s2 = autocorrelation_nD(t, D_hop=1.0, n=2)
    s3 = autocorrelation_nD(t, D_hop=1.0, n=3)
    assert np.all(s1[1:] > s2[1:])
    assert np.all(s2[1:] > s3[1:])


def test_autocorrelation_zero_hopping_is_constant() -> None:
    t = np.linspace(0.0, 20.0, 50)
    s = autocorrelation_nD(t, D_hop=0.0, n=2)
    np.testing.assert_allclose(s, np.ones_like(t), rtol=0.0, atol=0.0)


def test_autocorrelation_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n must be one of"):
        autocorrelation_nD(np.array([0.0, 1.0]), D_hop=1.0, n=4)
    with pytest.raises(ValueError, match="D_hop must be >= 0"):
        autocorrelation_nD(np.array([0.0, 1.0]), D_hop=-0.1, n=2)
    with pytest.raises(ValueError, match="t must be >= 0"):
        autocorrelation_nD(np.array([-0.1, 1.0]), D_hop=1.0, n=2)


def test_spectral_density_1d_matches_log_reference_at_low_frequency() -> None:
    d_hop = 3.0
    for omega in (1.0, 2.0, 5.0):
        numeric = spectral_density(omega=omega, D_hop=d_hop, n=1)
        approx = (0.318 / d_hop) * np.log((16.0 * d_hop) / omega)
        assert numeric > 0.0
        assert np.isclose(numeric, approx, rtol=0.04)


def test_spectral_density_zero_frequency_behavior_by_dimension() -> None:
    assert np.isinf(spectral_density(omega=0.0, D_hop=3.0, n=1))
    assert np.isclose(spectral_density(omega=0.0, D_hop=3.0, n=2), 0.3008842008, rtol=2e-4)
    assert np.isclose(spectral_density(omega=0.0, D_hop=3.0, n=3), 0.2354756565, rtol=2e-4)


def test_spectral_density_converges_without_integration_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        values = [
            spectral_density(omega=0.0, D_hop=3.0, n=2),
            spectral_density(omega=0.0, D_hop=3.0, n=3),
            spectral_density(omega=1.0, D_hop=3.0, n=2),
            spectral_density(omega=1.0, D_hop=3.0, n=3),
        ]

    assert np.all(np.isfinite(values))
    assert all(value >= 0.0 for value in values)
    assert not caught


def test_spectral_density_low_frequency_nd_uses_warning_free_zero_frequency_limit() -> None:
    j0_2d = spectral_density(omega=0.0, D_hop=3.0, n=2)
    j0_3d = spectral_density(omega=0.0, D_hop=3.0, n=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        values = [
            spectral_density(omega=0.01, D_hop=3.0, n=2),
            spectral_density(omega=0.03, D_hop=10.0, n=2),
            spectral_density(omega=0.01, D_hop=3.0, n=3),
        ]

    assert not caught
    assert np.isclose(values[0], j0_2d, rtol=0.0, atol=1e-12)
    assert np.isclose(values[1], j0_2d / (10.0 / 3.0), rtol=0.0, atol=1e-12)
    assert np.isclose(values[2], j0_3d, rtol=0.0, atol=1e-12)


def test_ballistic_lambda_remains_finite_without_integration_warnings_in_stiff_range() -> None:
    b = np.array([1.0, 10.0, 50.0, 200.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        lam_2d = lambda_ball(b, C=0.8, D_hop=0.05, n=2)
        lam_3d = lambda_ball(b, C=0.8, D_hop=0.05, n=3)

    assert not caught
    assert np.all(np.isfinite(lam_2d))
    assert np.all(np.isfinite(lam_3d))
    assert np.all(lam_2d >= 0.0)
    assert np.all(lam_3d >= 0.0)


def test_lambda_ball_vectorized_and_c_scaling() -> None:
    b = np.array([0.01, 0.05, 0.1, 0.5])
    lam1 = lambda_ball(b, C=0.5, D_hop=3.0, n=1)
    lam2 = lambda_ball(b, C=1.0, D_hop=3.0, n=1)
    assert lam1.shape == b.shape
    np.testing.assert_allclose(lam2, 4.0 * lam1, rtol=1e-8, atol=1e-12)


def test_lambda_ball_1d_decreases_over_low_field_window() -> None:
    b = np.array([0.01, 0.05, 0.1, 0.5])
    lam = lambda_ball(b, C=0.8, D_hop=3.0, n=1)
    assert np.all(np.diff(lam) < 0.0)


def test_lambda_total_adds_offset() -> None:
    b = np.array([0.01, 0.05, 0.1])
    lam_ballistic = lambda_ball(b, C=0.8, D_hop=3.0, n=2)
    lam_total = lambda_total(b, C=0.8, D_hop=3.0, n=2, lambda_0D=0.07)
    np.testing.assert_allclose(lam_total, lam_ballistic + 0.07, rtol=1e-10, atol=1e-12)


def test_lambda_zero_hopping_limit_is_zero_for_positive_field() -> None:
    b = np.array([0.1, 1.0, 10.0])
    lam = lambda_ball(b, C=0.8, D_hop=0.0, n=2)
    np.testing.assert_allclose(lam, np.zeros_like(b), rtol=0.0, atol=0.0)


def test_lambda_invalid_inputs() -> None:
    b = np.array([0.1, 1.0])
    with pytest.raises(ValueError, match="n must be one of"):
        lambda_ball(b, C=1.0, D_hop=1.0, n=0)
    with pytest.raises(ValueError, match="D_hop must be >= 0"):
        lambda_ball(b, C=1.0, D_hop=-1.0, n=2)


def test_ballistic_components_registered_for_field_scope() -> None:
    names = component_names_for_x("field")
    assert "BallisticLF_1D" in names
    assert "BallisticLF_2D" in names
    assert "BallisticLF_3D" in names

    names_temp = component_names_for_x("temperature")
    names_run = component_names_for_x("run")
    assert "BallisticLF_2D" not in names_temp
    assert "BallisticLF_2D" not in names_run


def test_ballistic_component_metadata_and_info_html() -> None:
    comp = PARAMETER_MODEL_COMPONENTS["BallisticLF_1D"]
    assert comp.param_names == ["A", "D_hop"]
    assert comp.param_defaults["D_hop"] == 1.0
    assert "logarithmic" in get_component_applicability("BallisticLF_1D")
    assert "Ballistic hopping rate" in get_param_info("D_hop").description

    b = np.array([0.01, 0.05, 0.1])
    y = comp.function(b, A=0.8, D_hop=3.0)
    assert y.shape == b.shape
    assert np.all(np.isfinite(y))
    assert np.all(y >= 0.0)

    html = build_component_info_html(comp, render_latex_images=False)
    assert "D_hop" in html
    assert "Ballistic hopping rate" in html
    assert "Applicability" in html