"""Tests for superconducting sigma(T) penetration-depth models."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.fitting import ParameterCompositeModel
from asymmetry.core.fitting.parameter_models import component_names_for_x, fit_parameter_model
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.sc import bcs, constants, gaps, kernel, models


def test_delta_bcs_limits_and_monotonicity() -> None:
    t = np.linspace(0.0, 1.2, 200)
    d = bcs.delta_bcs(t)

    assert np.isclose(d[0], 1.0)
    assert np.isclose(d[-1], 0.0)
    assert np.all(d[(t >= 0.0) & (t <= 1.0)] <= 1.0)
    assert np.all(d[(t >= 0.0) & (t <= 1.0)] >= 0.0)

    # On (0, 1) the reduced gap should decrease with temperature.
    mid = d[(t > 0.0) & (t < 1.0)]
    assert np.all(np.diff(mid) <= 1e-10)


def test_gap_ratio_mev_conversion_round_trip() -> None:
    tc = 10.0
    ratio = bcs.gap_ratio_from_mev(gap_mev=1.5, tc=tc)
    ratio2 = bcs.resolve_gap_ratio(tc=tc, gap_mev=1.5)
    assert np.isclose(ratio, ratio2)


def test_gap_functions_reference_values() -> None:
    phi = np.array([0.0, np.pi / 4.0, np.pi / 2.0])

    g_s = gaps.isotropic_s(phi)
    g_d = gaps.d_wave(phi)
    g_p = gaps.p_wave_axial_2d(phi)

    np.testing.assert_allclose(g_s, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(g_d, [1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(g_p, [1.0, np.sqrt(2.0) / 2.0, 0.0], atol=1e-12)


def test_anisotropic_s_nodes_depend_on_a_parameter() -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 2001)

    g_nodeless = gaps.anisotropic_s_cos4(phi, a_anis=0.6)
    g_nodal = gaps.anisotropic_s_cos4(phi, a_anis=1.2)

    assert np.min(g_nodeless) > 0.0
    assert np.min(np.abs(g_nodal)) < 5e-3


def test_energy_integral_zero_gap_limit() -> None:
    vals = kernel.energy_integral(np.array([0.0, 0.0]), t_reduced=0.2, n_energy=80)
    np.testing.assert_allclose(vals, [-0.5, -0.5], atol=2e-3)


def test_superfluid_density_limit_values() -> None:
    tc = 20.0
    temp_arr = np.array([0.0, 2.0, 5.0, 10.0, 19.0, 20.0, 22.0])

    rho_s = models.rho_s_wave(temp_arr, Tc=tc)
    rho_d = models.rho_d_wave(temp_arr, Tc=tc)

    assert np.isclose(rho_s[0], 1.0)
    assert np.isclose(rho_d[0], 1.0)
    assert rho_s[-1] == 0.0
    assert rho_d[-1] == 0.0
    assert np.all((rho_s >= 0.0) & (rho_s <= 1.0))
    assert np.all((rho_d >= 0.0) & (rho_d <= 1.0))


def test_d_wave_has_stronger_low_t_variation_than_s_wave() -> None:
    tc = 30.0
    t_low = np.array([0.0, 1.5, 3.0])

    rho_s = models.rho_s_wave(t_low, Tc=tc)
    rho_d = models.rho_d_wave(t_low, Tc=tc)

    drop_s = rho_s[0] - rho_s[-1]
    drop_d = rho_d[0] - rho_d[-1]

    assert drop_d > drop_s


def test_sigma_models_respect_zero_temperature_and_tc_limits() -> None:
    tc = 25.0
    temp = np.array([0.0, 5.0, 15.0, 25.0])

    sigma = models.sc_s_wave(temp, sigma_0=1.5, Tc=tc, gap_ratio=1.8)
    assert np.isclose(sigma[0], 1.5, atol=2e-2)
    assert sigma[-1] < 1e-6


def test_two_gap_weight_limits_reduce_to_single_gap() -> None:
    tc = 22.0
    temp = np.linspace(0.0, tc, 15)

    single_1 = models.sc_s_wave(temp, sigma_0=1.0, Tc=tc, gap_ratio=1.4)
    single_2 = models.sc_s_wave(temp, sigma_0=1.0, Tc=tc, gap_ratio=2.2)

    mix_1 = models.sc_two_gap_ss(
        temp,
        sigma_0=1.0,
        Tc=tc,
        gap_ratio_1=1.4,
        gap_ratio_2=2.2,
        weight=1.0,
    )
    mix_2 = models.sc_two_gap_ss(
        temp,
        sigma_0=1.0,
        Tc=tc,
        gap_ratio_1=1.4,
        gap_ratio_2=2.2,
        weight=0.0,
    )

    np.testing.assert_allclose(single_1, mix_1, atol=1e-6)
    np.testing.assert_allclose(single_2, mix_2, atol=1e-6)


def test_quadrature_model_is_bounded_by_sigma_nm() -> None:
    tc = 18.0
    temp = np.linspace(0.0, tc, 20)

    sigma_nm = 0.15
    sigma_q = models.sc_s_wave_q(temp, sigma_sc=1.2, sigma_nm=sigma_nm, Tc=tc)
    assert np.all(sigma_q >= sigma_nm)


def test_brandt_conversions_are_consistent() -> None:
    sigma = np.array([0.2, 0.5, 1.0])
    lam = constants.sigma_to_lambda_nm(sigma)
    sigma_back = constants.lambda_nm_to_sigma_us(lam)
    np.testing.assert_allclose(sigma_back, sigma, rtol=1e-12, atol=1e-12)


def test_sc_components_are_registered_for_temperature_scope() -> None:
    names = component_names_for_x("temperature")
    for required in [
        "SC_SWave",
        "SC_DWave",
        "SC_AnisotropicS_Cos4",
        "SC_NonmonotonicD",
        "SC_PWaveAxial",
        "SC_TwoGap_SS",
        "SC_TwoGap_SD",
        "SC_SWave_Q",
    ]:
        assert required in names


def test_parameter_composite_sc_component_callable() -> None:
    model = ParameterCompositeModel(["SC_SWave"])
    temps = np.array([0.0, 2.0, 5.0, 10.0, 15.0])

    values = model.function(temps, sigma_0=1.2, Tc=20.0, gap_ratio=1.8, sigma_bg=0.05)
    assert values.shape == temps.shape
    assert np.all(np.isfinite(values))
    assert values[0] >= values[-1]


def test_fit_parameter_model_recovers_sigma0_for_sc_swave() -> None:
    rng = np.random.default_rng(2026)

    temps = np.linspace(1.0, 22.0, 15)
    sigma_true = models.sc_s_wave(temps, sigma_0=1.3, Tc=24.0, gap_ratio=1.76, sigma_bg=0.02)
    noise = rng.normal(0.0, 0.01, size=temps.shape)
    sigma_data = sigma_true + noise
    sigma_err = np.full_like(temps, 0.01)

    model = ParameterCompositeModel(["SC_SWave"])
    params = ParameterSet(
        [
            Parameter("sigma_0", value=1.0, min=0.0, max=3.0),
            Parameter("Tc", value=24.0, fixed=True),
            Parameter("gap_ratio", value=1.76, fixed=True),
            Parameter("sigma_bg", value=0.02, fixed=True),
        ]
    )

    result = fit_parameter_model(temps, sigma_data, sigma_err, model, params)

    if not result.success and "iminuit import error" in result.message:
        pytest.skip("iminuit backend unavailable in this environment")

    assert result.success
    fitted_sigma0 = next(p.value for p in result.parameters if p.name == "sigma_0")
    assert abs(fitted_sigma0 - 1.3) < 0.08
