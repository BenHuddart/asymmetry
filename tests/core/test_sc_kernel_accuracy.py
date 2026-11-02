"""Independent quadrature, Bessel-series and asymptotic checks of SC numerics."""

from functools import lru_cache

import numpy as np
import pytest
from scipy.integrate import quad, quad_vec
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import expit, k1e, roots_legendre, zeta

from asymmetry.core.fitting import Parameter, ParameterCompositeModel, ParameterSet
from asymmetry.core.fitting.parameter_models import fit_parameter_model
from asymmetry.core.fitting.sc import gaps, kernel, models


def _reference_integral(delta, temperature):
    # q = sqrt(E²-Delta²) removes the density-of-states endpoint singularity.
    def integrand(q):
        occupation = expit(-np.hypot(q, delta) / temperature)
        return -occupation * (1 - occupation) / temperature

    return quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]


def _reference_s_wave(temperatures, tc, gap_ratio):
    values = []
    for temperature in temperatures:
        reduced = temperature / tc
        if reduced <= 0:
            values.append(1.0)
        elif reduced >= 1:
            values.append(0.0)
        else:
            # Carrington-Manzano interpolation, independent of the model helper.
            gap = gap_ratio * np.tanh(1.82 * (1.018 * (1 / reduced - 1)) ** 0.51)
            values.append(1 + 2 * _reference_integral(gap, reduced))
    return np.array(values)


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [0.01, 0.1, 1.0, 3.0, 10.0])
def test_thermal_integral_matches_transformed_quad(temperature, ratio):
    expected = _reference_integral(ratio * temperature, temperature)
    assert kernel.thermal_integral(ratio) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0])
def test_thermal_integral_matches_bessel_series(ratio):
    orders = np.arange(1, 201)
    # Expanding f(1-f) into exponentials and integrating gives d*n*K1(n*d).
    expected = -ratio * np.sum(
        (-1.0) ** (orders + 1) * orders * k1e(orders * ratio) * np.exp(-orders * ratio)
    )
    assert kernel.thermal_integral(ratio) == pytest.approx(expected, abs=1e-14)


def test_zero_gap_and_ginzburg_landau_limits():
    assert kernel.thermal_integral(0.0) == pytest.approx(-0.5, abs=1e-15)
    d = 0.01
    density = 1 + 2 * kernel.thermal_integral(d)
    assert density / d**2 == pytest.approx(7 * zeta(3) / (4 * np.pi**2), rel=1e-4)


@pytest.mark.parametrize("ratio", [10.0, 20.0])
def test_s_wave_low_temperature_asymptote(ratio):
    density = 1 + 2 * kernel.thermal_integral(ratio)
    leading = np.sqrt(2 * np.pi * ratio) * np.exp(-ratio)
    assert (1 - density) / leading == pytest.approx(
        1 + 3 / (8 * ratio) - 15 / (128 * ratio**2), abs=2 / ratio**3
    )


def test_default_s_wave_curve_matches_independent_integration():
    temperature = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    np.testing.assert_allclose(
        models.rho_s_wave(temperature, Tc=1.0, gap_ratio=1.764),
        _reference_s_wave(temperature, 1.0, 1.764),
        rtol=0,
        atol=1e-10,
    )


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_thermal_integral_and_density_are_scale_free(scale):
    deltas, temperatures = np.array([0.01, 0.1, 1.0, 3.0]), np.array([0.1, 0.5, 1.0, 0.5])
    np.testing.assert_allclose(
        kernel.thermal_integral(scale * deltas / (scale * temperatures)),
        kernel.thermal_integral(deltas / temperatures),
        rtol=0,
        atol=5e-16,
    )
    np.testing.assert_allclose(
        models.rho_s_wave(temperatures * scale, Tc=2 * scale),
        models.rho_s_wave(temperatures, Tc=2.0),
        rtol=0,
        atol=1e-15,
    )


def test_thermal_integral_shapes_symmetry_and_bounds():
    arguments = np.linspace(-80, 80, 801).reshape(3, 267)
    values = kernel.thermal_integral(arguments)
    assert values.shape == arguments.shape
    assert np.all(values >= -0.5 - 1e-15)
    assert np.all(values <= 0)
    np.testing.assert_array_equal(values, kernel.thermal_integral(-arguments))
    assert kernel.thermal_integral(np.empty((0, 2))).shape == (0, 2)


def test_s_wave_gap_law_recovers_parameters():
    temperature = np.linspace(0.5, 12.0, 24)
    values = 1.3 * _reference_s_wave(temperature, 10.0, 1.764) + 0.1
    model = ParameterCompositeModel(["SC_SWave"])
    parameters = ParameterSet(
        [
            Parameter("sigma_0", 1.2, min=0),
            Parameter("Tc", 9.8, min=0.1),
            Parameter("gap_ratio", 1.8, min=0.1),
            Parameter("sigma_bg", 0.12, min=0),
        ]
    )
    result = fit_parameter_model(temperature, values, np.full_like(values, 0.01), model, parameters)
    assert result.success
    assert result.parameters["Tc"].value == pytest.approx(10.0, abs=1e-3)
    assert result.parameters["gap_ratio"].value == pytest.approx(1.764, abs=5e-4)


@lru_cache(maxsize=1)
def _reference_thermal_spline():
    # Independent adaptive integration, sampled densely for the angular oracle.
    d = np.linspace(0, 80, 40001)

    def integrand(q):
        f = expit(-np.hypot(q, d))
        return -f * (1 - f)

    values, _ = quad_vec(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-13, norm="max")
    return CubicSpline(d, values)


def _reference_thermal(d):
    argument = np.abs(d)
    return np.where(argument <= 80, _reference_thermal_spline()(np.minimum(argument, 80)), 0.0)


def test_independent_angular_oracle_kernel_is_accurate():
    d = np.r_[np.linspace(0, 8, 41) + 0.00073, [10.123, 20.234, 40.345, 79.876]]
    expected = np.array([_reference_integral(value, 1.0) for value in d])
    np.testing.assert_allclose(_reference_thermal(d), expected, rtol=0, atol=1e-12)


def _reference_surface_density(temperature, ratio, shape, gap, sphere):
    reduced_gap = (
        np.tanh(1.82 * (1.018 * (1 / temperature - 1)) ** 0.51)
        if shape is None
        else np.tanh(np.pi / ratio * np.sqrt(shape * (1 / temperature - 1)))
    )
    scale = ratio * reduced_gap / temperature
    if sphere:
        u, w_u = roots_legendre(2048)
        x, w_phi = roots_legendre(2048)
        phi = np.pi * (x + 1)
        values = _reference_thermal(scale * gap(u[:, None], phi[None, :]))
        return 1 + 2 * np.sum(w_u[:, None] * w_phi[None, :] * values) / 4
    grid = np.linspace(0, 2 * np.pi, 1025)
    g = gap(grid)
    nodes = list(grid[np.abs(g) < 1e-10])
    for a, b, ga, gb in zip(grid[:-1], grid[1:], g[:-1], g[1:], strict=True):
        if ga * gb < 0 and min(abs(ga), abs(gb)) > 1e-10:
            nodes.append(brentq(gap, a, b, xtol=1e-14))
    integral = quad(
        lambda phi: float(_reference_thermal(scale * gap(phi))),
        0,
        2 * np.pi,
        points=nodes,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=200,
    )[0]
    return 1 + integral / np.pi


_ANGULAR_CASES = [
    (models.rho_s_wave, {}, 1.764, None, lambda p: np.ones_like(p), False),
    (models.rho_d_wave, {}, 2.14, 4 / 3, lambda p: np.cos(2 * p), False),
    (
        models.rho_anisotropic_s_cos4,
        {"a_anis": 0.2},
        1.764,
        None,
        lambda p: 1 + 0.2 * np.cos(4 * p),
        False,
    ),
    (
        models.rho_anisotropic_s_cos4,
        {"a_anis": 1.5},
        1.764,
        None,
        lambda p: 1 + 1.5 * np.cos(4 * p),
        False,
    ),
    (
        models.rho_nonmonotonic_d,
        {},
        2.14,
        4 / 3,
        lambda p: 0.8 * np.cos(2 * p) + 0.2 * np.cos(6 * p),
        False,
    ),
    (models.rho_p_wave_axial, {}, 2.0, None, lambda p: np.cos(p), False),
    (models.rho_extended_s, {}, 2.14, 4 / 3, lambda p: np.cos(2 * p), False),
    (
        models.rho_s_plus_g,
        {},
        2.77,
        2.0,
        lambda u, p: (1 - (1 - u * u) ** 2 * np.cos(4 * p)) / 2,
        True,
    ),
    (models.rho_p_wave_polar_3d, {}, 2.0, None, lambda u, p: np.sqrt(1 - u * u), True),
]


@pytest.mark.parametrize(
    "model,kwargs,ratio,shape,gap,sphere",
    _ANGULAR_CASES,
    ids=[
        "s",
        "d",
        "cos4-nodeless",
        "cos4-nodal",
        "nonmonotonic",
        "axial",
        "extended",
        "s+g",
        "polar",
    ],
)
def test_fermi_surface_average_matches_independent_reference(
    model, kwargs, ratio, shape, gap, sphere
):
    temperatures = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9])
    expected = np.array(
        [_reference_surface_density(t, ratio, shape, gap, sphere) for t in temperatures]
    )
    np.testing.assert_allclose(model(temperatures, Tc=1.0, **kwargs), expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize("temperature", [0.005, 0.01, 0.02])
def test_d_wave_is_linear_at_low_temperature(temperature):
    ratio = 2.14
    reduced_gap = np.tanh(np.pi / ratio * np.sqrt((4 / 3) * (1 / temperature - 1)))
    expected = 1 - 2 * np.log(2) * temperature / (ratio * reduced_gap)
    assert models.rho_d_wave(temperature, Tc=1.0) == pytest.approx(
        expected, abs=0.5 * temperature**3
    )


def test_unresolvable_temperature_raises():
    with pytest.raises(ValueError, match="d_wave.*T/Tc"):
        models.rho_d_wave(np.array([1e-4]), Tc=1.0)


@pytest.mark.parametrize(
    "model,kwargs,ratio,shape,gap,sphere",
    _ANGULAR_CASES,
    ids=[
        "s",
        "d",
        "cos4-nodeless",
        "cos4-nodal",
        "nonmonotonic",
        "axial",
        "extended",
        "s+g",
        "polar",
    ],
)
def test_converged_density_bounds_and_minimum_temperature(model, kwargs, ratio, shape, gap, sphere):
    values = model(np.array([0.0, 0.005, 0.02, 0.1, 0.9, 0.99, 1.0, 1.1]), Tc=1.0, **kwargs)
    assert np.all(values >= -1e-14)
    assert np.all(values <= 1 + 1e-14)
    assert values[0] == 1
    np.testing.assert_array_equal(values[-2:], [0.0, 0.0])


def test_d_wave_gap_law_recovers_parameters():
    temperature = np.linspace(0.5, 12.0, 24)
    density = np.array(
        [
            _reference_surface_density(t / 10, 2.14, 4 / 3, lambda p: np.cos(2 * p), False)
            if t < 10
            else 0
            for t in temperature
        ]
    )
    values = 1.3 * density + 0.1
    model = ParameterCompositeModel(["SC_DWave"])
    parameters = ParameterSet(
        [
            Parameter("sigma_0", 1.2, min=0),
            Parameter("Tc", 9.8, min=0.1),
            Parameter("gap_ratio", 2.1, min=0.1),
            Parameter("sigma_bg", 0.12, min=0),
        ]
    )
    result = fit_parameter_model(temperature, values, np.full_like(values, 0.01), model, parameters)
    assert result.success
    assert result.parameters["Tc"].value == pytest.approx(10.0, abs=1e-3)
    assert result.parameters["gap_ratio"].value == pytest.approx(2.14, abs=5e-4)


def test_gap_form_symmetries_hold():
    rng = np.random.default_rng(20261031)
    phi = rng.uniform(0, 2 * np.pi, 100)
    theta = rng.uniform(0, np.pi, 100)
    params = {
        "anisotropic_s_cos4": {"a_anis": 1.5},
        "anisotropic_s_cos2": {"a_anis": 0.2},
        "anisotropic_d_with_harmonics": {"b_harm": 0.2},
        "nonmonotonic_d_wave": {"beta_nm": 0.8},
        "extended_s": {"signed": True},
    }
    for name, gap in gaps.GAP_FORMS.items():
        kwargs = params.get(name, {})
        args = (theta, phi) if gap.surface == "sphere" else (phi,)
        base = np.abs(gap.function(*args, **kwargs))
        shifted = phi + gap.period if gap.period is not None else np.zeros_like(phi)
        shifted_args = (theta, shifted) if gap.surface == "sphere" else (shifted,)
        np.testing.assert_allclose(
            np.abs(gap.function(*shifted_args, **kwargs)), base, atol=1e-14, rtol=1e-13
        )
        if gap.equator_mirror:
            np.testing.assert_allclose(
                np.abs(gap.function(np.pi - theta, phi, **kwargs)), base, atol=1e-14, rtol=1e-13
            )


def test_angular_cap_raises_instead_of_returning_unconverged_density(monkeypatch):
    monkeypatch.setattr(kernel, "_MAX_ANGULAR_NODES", 32)
    with pytest.raises(ValueError, match="d_wave.*T/Tc=.*node cap"):
        models.rho_d_wave(0.005, Tc=1.0)


@pytest.mark.timeout(5)
def test_nonfinite_gap_output_raises_instead_of_stalling_refinement():
    gap = gaps.GapForm("invalid", lambda phi: np.full_like(phi, np.nan), "cylinder", np.pi, False)
    with pytest.raises(ValueError, match="surface average must be finite"):
        kernel.superfluid_density([0.1], tc=1.0, gap_ratio=1.0, gap=gap)


def test_isotropic_gap_retains_exact_low_temperature_limit():
    assert models.rho_s_wave(1e-4, Tc=1.0) == 1.0


@pytest.mark.parametrize("kwargs", [{"gap_ratio": np.nan}, {"a_anis": np.inf}])
def test_nonfinite_gap_parameters_raise(kwargs):
    with pytest.raises(ValueError, match="parameters must be finite"):
        models.rho_anisotropic_s_cos4([0.1], Tc=1.0, **kwargs)


def test_angular_thermal_interpolation_matches_independent_integral():
    d = np.r_[np.linspace(0, 80, 32001) + 0.00137, -0.123, 81.0, 1000.0]
    np.testing.assert_allclose(
        kernel._angular_thermal_integral(d), _reference_thermal(d), rtol=0, atol=1e-11
    )
