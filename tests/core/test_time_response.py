"""Independent count-ratio targets for ordinary double-pulse predictions."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.utils.constants import MUON_LIFETIME_US


def test_serialized_model_fits_independent_causal_counts():
    time = np.linspace(-0.15, 2, 250)
    forward = np.zeros_like(time)
    backward = np.zeros_like(time)
    alpha, beta = 1.13, 0.83
    for birth in [-0.162, 0.162]:
        live = time >= birth
        age = time[live] - birth
        population = 0.5 * np.exp(-age / MUON_LIFETIME_US)
        polarization = 0.19 * np.exp(-0.7 * age) + 0.015
        forward[live] += population * (1 + polarization)
        backward[live] += population / alpha * (1 - beta * polarization)
    values = 100 * (forward - alpha * backward) / (beta * forward + alpha * backward)
    model = CompositeModel.from_dict(
        {
            "component_names": ["Exponential", "Constant"],
            "time_response": {"mode": "double_delta", "separation_us": 0.324},
        }
    )
    parameters = ParameterSet(
        [
            Parameter("A_1", 17, min=0, max=30),
            Parameter("Lambda", 0.5, min=0, max=3),
            Parameter("A_bg", 1, min=-5, max=5),
        ]
    )
    data = MuonDataset(time, values, np.full_like(time, 0.05))
    result = FitEngine().fit(data, model.function, parameters)
    assert result.success
    assert result.parameters["A_1"].value == pytest.approx(19, abs=1e-4)
    assert result.parameters["Lambda"].value == pytest.approx(0.7, abs=1e-5)
    assert np.max(np.abs(result.residuals)) < 5e-5


@pytest.mark.parametrize("beta", [1.0, 0.72])
@pytest.mark.parametrize("scale", [1, 100])
def test_complete_product_origin_and_constant_match_count_ratio(beta, scale):
    from asymmetry.core.fitting.time_response import TimeResponse

    time = np.array([-0.162, -0.05, 0.161, 0.162, 0.3, 0.91, 4.0])
    model = CompositeModel.from_expression("Exponential * Oscillatory + Constant")
    model = model.with_time_response(TimeResponse("double_delta", 0.324))
    parameters = dict(
        A_1=0.21 * scale, Lambda=0.7, frequency=1.3, phase=np.deg2rad(17), A_bg=0.013 * scale
    )
    f, b = np.zeros_like(time), np.zeros_like(time)
    for birth in [-0.162, 0.162]:
        live = time >= birth
        age = time[live] - birth
        population = np.exp(-age / MUON_LIFETIME_US) / 2
        p = 0.21 * np.exp(-0.7 * age) * np.cos(2 * np.pi * 1.3 * age + np.deg2rad(17)) + 0.013
        f[live] += population * (1 + p)
        b[live] += population / 1.1 * (1 - beta * p)
    expected = scale * (f - 1.1 * b) / (beta * f + 1.1 * b)
    np.testing.assert_allclose(model.function(time, **parameters), expected, atol=1e-13)
    assert model.param_names == ["A_1", "Lambda", "frequency", "phase", "A_bg"]


def test_causal_callable_never_sees_negative_age_and_onset_includes_zero():
    from asymmetry.core.fitting.time_response import TimeResponse

    seen = []

    def source(age, amplitude):
        assert np.all(age >= 0)
        seen.extend(age)
        return amplitude * np.exp(-np.sqrt(age))

    response = TimeResponse("double_delta", 0.4)
    time = np.array([-0.2, -0.1, 0.2, 0.3])
    out = response.compose(source)(time, amplitude=0.2)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.2
    assert seen.count(0) == 2
    with pytest.raises(ValueError, match="no source population.*fit window"):
        response.evaluate(source, np.array([-0.201]), amplitude=0.2)


def test_off_zero_separation_late_underflow_and_inert_settings():
    from asymmetry.core.fitting.time_response import TimeResponse

    def source(t, amplitude):
        return amplitude + t

    time = np.array([0.0, 0.1, 2.0])
    np.testing.assert_array_equal(
        TimeResponse("double_delta", 0).evaluate(source, time, amplitude=3), time + 3
    )
    off = TimeResponse("off", float("nan"))
    assert off.to_dict() == {"mode": "off"}
    assert off.compose(source) is source
    np.testing.assert_array_equal(off.evaluate(source, np.array([-1.0]), amplitude=3), [2])
    response = TimeResponse("double_delta", 0.324)
    np.testing.assert_allclose(
        response.evaluate(lambda age: np.ones_like(age) * 7, np.array([1e6])), [7], atol=1e-14
    )


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_active_input_boundary(value):
    from asymmetry.core.fitting.time_response import TimeResponse

    with pytest.raises(ValueError, match="finite nonnegative"):
        TimeResponse("double_delta", value)


def test_active_payload_requires_supplied_separation():
    with pytest.raises(ValueError, match="Supply a fixed"):
        CompositeModel.from_dict(
            {"component_names": ["Exponential"], "time_response": {"mode": "double_delta"}}
        )


def test_additive_components_and_fraction_groups_use_same_response():
    from asymmetry.core.fitting.time_response import TimeResponse

    for expression in [
        "Exponential + Gaussian + Constant",
        "(Exponential + Gaussian){frac} + Constant",
    ]:
        model = CompositeModel.from_expression(expression).with_time_response(
            TimeResponse("double_delta", 0.324)
        )
        time = np.linspace(-0.15, 3, 30)
        values = model.param_defaults
        curves = model.evaluate_components(time, **values)
        np.testing.assert_allclose(sum(y for _, y in curves), model.function(time, **values))


def test_point_model_does_not_integrate_onset_straddling_packed_bin():
    from scipy.integrate import quad

    from asymmetry.core.fitting.time_response import TimeResponse

    def source(age):
        return 0.21 * np.cos(2 * np.pi * 1.3 * age + 0.17)

    def counts(t, sign):
        return sum(
            0.5 * np.exp(-(t - birth) / MUON_LIFETIME_US) * (1 + sign * source(t - birth))
            for birth in [-0.162, 0.162]
            if t >= birth
        )

    f = quad(lambda t: counts(t, 1), 0.1, 0.2, points=[0.162])[0]
    b = quad(lambda t: counts(t, -1), 0.1, 0.2, points=[0.162])[0]
    point = TimeResponse("double_delta", 0.324).evaluate(source, np.array([0.15]))[0]
    # A point prediction and a ratio of integrated counts are different observations.
    assert abs(point - (f - b) / (f + b)) > 0.05


def test_global_recovery_and_selected_observations():
    from asymmetry.core.fitting.time_response import TimeResponse

    model = CompositeModel(["Exponential"], time_response=TimeResponse("double_delta", 0.324))
    time = np.linspace(-0.15, 4, 180)
    datasets = []
    initial = {}
    for run, amplitude in [(1, 20), (2, 16)]:
        values = []
        for t in time:
            ages = np.array([t + 0.162, t - 0.162])
            ages = ages[ages >= 0]
            w = np.exp(-ages / MUON_LIFETIME_US)
            values.append(amplitude * np.sum(w * np.exp(-0.7 * ages)) / np.sum(w))
        datasets.append(MuonDataset(time, values, np.full_like(time, 0.1), {"run_number": run}))
        initial[run] = ParameterSet(
            [Parameter("A_1", amplitude * 0.9), Parameter("Lambda", 0.6, min=0, max=3)]
        )
    result = FitEngine().global_fit(
        datasets,
        model.function,
        global_params=["Lambda"],
        local_params=["A_1"],
        initial_params=initial,
        t_min=-0.1,
        t_max=3,
    )
    results, globals_ = result
    assert all(r.success for r in results.values())
    assert globals_["Lambda"].value == pytest.approx(0.7, abs=2e-4)
