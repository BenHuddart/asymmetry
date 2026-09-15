"""Tests for the FermiStep transition-step trend model.

The published forms the model must reproduce are transcribed as numerical
oracles: Mantid's ``SmoothTransition`` (Framework/CurveFitting/src/Functions/
SmoothTransition.cpp — the code, whose rst formula has a sign error), the
Khasanov et al. PRB 102, 094504 (2020) wTF amplitude + background form, and
the tanh form. See docs/porting/fermi-step-transition/.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from asymmetry.core.fitting.component_docs import get_component_applicability
from asymmetry.core.fitting.parameter_models import (
    PARAMETER_MODEL_COMPONENTS,
    ParameterCompositeModel,
    _fermi_step,
    component_names_for_x,
    fit_parameter_model,
    suggest_model_seeds,
    suggest_trend_seeds,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info


def test_fermi_step_plateaus_midpoint_and_width() -> None:
    a1, a2, tc, dt = 0.11, 0.22, 50.0, 0.5
    assert _fermi_step(np.array([0.0]), a1, a2, tc, dt)[0] == pytest.approx(a1)
    assert _fermi_step(np.array([200.0]), a1, a2, tc, dt)[0] == pytest.approx(a2)
    assert _fermi_step(np.array([tc]), a1, a2, tc, dt)[0] == pytest.approx((a1 + a2) / 2)
    # The 10 % and 90 % levels sit 2 ln 9 · dT apart.
    half_span = np.log(9.0) * dt
    lo, hi = _fermi_step(np.array([tc - half_span, tc + half_span]), 0.0, 1.0, tc, dt)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.9)


def test_fermi_step_matches_mantid_smooth_transition_code() -> None:
    a1, a2, midpoint, growth_rate = 3.0, -1.5, 120.0, 7.0
    x = np.linspace(60.0, 180.0, 41)
    mantid = a2 + (a1 - a2) / (np.exp((x - midpoint) / growth_rate) + 1)
    np.testing.assert_allclose(_fermi_step(x, a1, a2, midpoint, growth_rate), mantid, rtol=1e-12)


def test_fermi_step_matches_khasanov_wtf_form() -> None:
    a_s, a_bg, t_n, dt_n = 0.111, 0.112, 50.0, 0.5
    temperature = np.linspace(45.0, 55.0, 51)
    khasanov = a_s / (1.0 + np.exp((t_n - temperature) / dt_n)) + a_bg
    np.testing.assert_allclose(
        _fermi_step(temperature, a_bg, a_s + a_bg, t_n, dt_n), khasanov, rtol=1e-12
    )


def test_fermi_step_matches_tanh_form() -> None:
    amplitude, k, t0, offset = 0.6, 0.25, 274.0, 27.2
    temperature = np.linspace(200.0, 350.0, 61)
    tanh_form = amplitude * np.tanh(k * (temperature - t0)) + offset
    np.testing.assert_allclose(
        _fermi_step(temperature, offset - amplitude, offset + amplitude, t0, 1.0 / (2.0 * k)),
        tanh_form,
        rtol=1e-12,
    )


def test_fermi_step_narrow_width_stays_finite_without_overflow() -> None:
    temperature = np.linspace(0.0, 100.0, 201)  # includes T == Tc exactly
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = _fermi_step(temperature, 1.0, 0.0, 50.0, get_param_info("dT").default_min)
    assert np.all(np.isfinite(values))
    assert values[0] == 1.0 and values[-1] == 0.0


def test_fermi_step_registered_on_temperature_axis_under_critical_behaviour() -> None:
    assert "FermiStep" in component_names_for_x("temperature")
    assert "FermiStep" not in component_names_for_x("field")
    definition = PARAMETER_MODEL_COMPONENTS["FermiStep"]
    assert definition.category == "Critical behaviour"
    model = ParameterCompositeModel(["FermiStep"])
    assert model.param_names == ["A1", "A2", "Tc", "dT"]
    assert model.formula_string() == "A2 + (A1 - A2)/(exp((T - Tc)/dT) + 1)"


def test_fermi_step_parameter_metadata() -> None:
    # Plateaus are signed (a falling step has A2 < A1); the width divides T - Tc
    # so its floor must be strictly positive.
    assert get_param_info("A1").default_min is None
    assert get_param_info("A2").default_min is None
    assert get_param_info("dT").default_min > 0.0
    assert get_param_info("dT").unit == "K"
    assert get_param_info("dT").unicode == "ΔT"
    assert get_param_info("A2_2").latex == r"$A_{2,2}$"


def test_fermi_step_applicability_text() -> None:
    text = get_component_applicability("FermiStep")
    assert "volume fraction" in text
    assert "OrderParameter" in text


def _synthetic_wtf_step() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature = np.arange(40.0, 61.0, 1.0)
    rng = np.random.default_rng(7)
    values = _fermi_step(temperature, 0.112, 0.223, 50.3, 0.8)
    values = values + rng.normal(0.0, 0.002, temperature.size)
    return temperature, values, np.full_like(values, 0.002)


def test_suggest_trend_seeds_fermi_step_rising_step() -> None:
    temperature, values, _ = _synthetic_wtf_step()
    seeds = suggest_trend_seeds(ParameterCompositeModel(["FermiStep"]), temperature, values)
    assert seeds["A1"] == pytest.approx(0.112, abs=0.005)
    assert seeds["A2"] == pytest.approx(0.223, abs=0.005)
    assert seeds["Tc"] == pytest.approx(50.3, abs=1.0)
    assert 0.2 < seeds["dT"] < 3.0


def test_suggest_trend_seeds_fermi_step_falling_unsorted_with_nan() -> None:
    temperature = np.array([9.0, 2.0, np.nan, 6.0, 4.0, 7.0, 5.0, 3.0, 8.0, 1.0])
    values = _fermi_step(temperature, 12.0, 4.0, 5.5, 0.4)
    seeds = suggest_trend_seeds(ParameterCompositeModel(["FermiStep"]), temperature, values)
    assert seeds["A1"] > seeds["A2"]
    assert seeds["Tc"] == pytest.approx(5.5, abs=0.5)
    assert seeds["dT"] > 0.0


def test_suggest_trend_seeds_fermi_step_flat_trace_seeds_plateaus_only() -> None:
    temperature = np.linspace(1.0, 10.0, 10)
    seeds = suggest_trend_seeds(
        ParameterCompositeModel(["FermiStep"]), temperature, np.full(10, 0.2)
    )
    assert seeds == {"A1": 0.2, "A2": 0.2}


def test_suggest_trend_seeds_fermi_step_suffixed_for_repeated_component() -> None:
    temperature, values, _ = _synthetic_wtf_step()
    model = ParameterCompositeModel(["FermiStep", "FermiStep"])
    seeds = suggest_trend_seeds(model, temperature, values)
    assert {"Tc_1", "Tc_2", "dT_1", "dT_2"} <= set(seeds)
    # The generic seeder merges the same trend seeds.
    assert suggest_model_seeds(model, temperature, values)["Tc_1"] == seeds["Tc_1"]


def test_fermi_step_fit_from_seeds_recovers_parameters() -> None:
    temperature, values, errors = _synthetic_wtf_step()
    model = ParameterCompositeModel(["FermiStep"])
    seeds = suggest_trend_seeds(model, temperature, values)
    params = ParameterSet(
        [
            Parameter(name, value=seeds[name], min=get_param_info(name).default_min)
            if get_param_info(name).default_min is not None
            else Parameter(name, value=seeds[name])
            for name in model.param_names
        ]
    )
    result = fit_parameter_model(temperature, values, errors, model, params)
    assert result.success
    fitted = {p.name: p.value for p in result.parameters}
    assert fitted["A1"] == pytest.approx(0.112, abs=0.003)
    assert fitted["A2"] == pytest.approx(0.223, abs=0.003)
    assert fitted["Tc"] == pytest.approx(50.3, abs=0.2)
    assert fitted["dT"] == pytest.approx(0.8, abs=0.2)
