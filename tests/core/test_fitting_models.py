"""Tests for built-in fit model functions and registry."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss

from asymmetry.core.fitting.models import (
    MODELS,
    abragam,
    exponential_relaxation,
    gaussian_relaxation,
    oscillatory,
    static_gkt_zf,
    stretched_exponential,
)


def test_model_functions_return_finite_arrays() -> None:
    t = np.linspace(0.0, 5.0, 50)

    y1 = exponential_relaxation(t, A0=25.0, Lambda=0.5)
    y2 = gaussian_relaxation(t, A0=25.0, sigma=0.3)
    y3 = oscillatory(t, A0=25.0, frequency=1.0, Lambda=0.2)
    y4 = stretched_exponential(t, A0=25.0, Lambda=0.4, beta=0.8)
    y5 = static_gkt_zf(t, A0=25.0, Delta=0.2)

    for y in (y1, y2, y3, y4, y5):
        assert y.shape == t.shape
        assert np.all(np.isfinite(y))


def test_registry_contains_expected_models_and_defaults() -> None:
    expected = {
        "ExponentialRelaxation",
        "GaussianRelaxation",
        "Oscillatory",
        "StretchedExponential",
        "StaticGKT_ZF",
    }
    assert expected.issubset(set(MODELS))

    osc = MODELS["Oscillatory"]
    assert "frequency" in osc.param_names
    assert osc.param_defaults["A0"] == 25.0


@pytest.mark.parametrize("delta", [0.1, 0.5, 2.0])
@pytest.mark.parametrize("u", [1e-3, 2e-3, 4e-3])
def test_static_gkt_short_time_coefficient_is_one(delta: float, u: float) -> None:
    value = static_gkt_zf(np.array([u / delta]), A0=1.0, Delta=delta)[0]
    assert (1 - value) / u**2 == pytest.approx(1 - 5 * u**2 / 12, abs=1e-8, rel=0)


@pytest.mark.parametrize("u", [0.02, 0.05])
def test_gaussian_zero_field_match_and_fourth_order_difference(u: float) -> None:
    value = static_gkt_zf(np.array([u]), A0=1.0, Delta=1.0)[0]
    assert (value - np.exp(-(u**2))) / u**4 == pytest.approx(-1 / 12, rel=1e-2)
    half_width = gaussian_relaxation(np.array([u]), A0=1.0, sigma=1 / np.sqrt(2))[0]
    assert (1 - half_width) / u**2 == pytest.approx(0.5 - u**2 / 8, abs=2e-7, rel=0)


def test_static_gkt_and_gaussian_differ_at_finite_time() -> None:
    dip_time = np.sqrt(3)
    dip = static_gkt_zf(np.array([dip_time]), A0=1.0, Delta=1.0)[0]
    assert dip == pytest.approx(1 / 3 - 4 * np.exp(-1.5) / 3)
    gaussian_dip = gaussian_relaxation(np.array([dip_time]), A0=1.0, sigma=1.0)[0]
    assert gaussian_dip == pytest.approx(np.exp(-3))
    assert abs(static_gkt_zf(np.array([3.0]), 1.0, 1.0)[0] - np.exp(-9)) > 0.25


@pytest.mark.parametrize("u", [0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
def test_static_gkt_is_an_isotropic_gaussian_field_average(u: float) -> None:
    nodes, weights = hermgauss(40)
    x, y, z = np.meshgrid(np.sqrt(2) * nodes, np.sqrt(2) * nodes, np.sqrt(2) * nodes, indexing="ij")
    radius_squared = x**2 + y**2 + z**2
    parallel_fraction = z**2 / radius_squared
    polarisation = parallel_fraction + (1 - parallel_fraction) * np.cos(np.sqrt(radius_squared) * u)
    probability = np.einsum("i,j,k->ijk", weights, weights, weights) / np.pi**1.5
    expected = np.sum(probability * polarisation)
    assert static_gkt_zf(np.array([u]), A0=1.0, Delta=1.0)[0] == pytest.approx(
        expected, abs=1e-10, rel=0
    )


def test_transverse_field_envelope_uses_delta_over_root_two() -> None:
    nodes, weights = hermgauss(40)
    time = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    delta = 0.7
    expected = np.sum(
        weights[:, None] * np.cos(np.sqrt(2) * delta * nodes[:, None] * time), axis=0
    ) / np.sqrt(np.pi)
    np.testing.assert_allclose(
        gaussian_relaxation(time, 1.0, delta / np.sqrt(2)), expected, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(abragam(time, 1.0, delta, nu=0.0), expected, atol=1e-10, rtol=0)
