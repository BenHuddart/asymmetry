"""Independent count-ratio targets for ordinary double-pulse predictions."""

from functools import partial

import numpy as np
import pytest

from asymmetry.core.computation import COMPUTATION_REVISION
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.data.reduction_recipe import EffectiveReductionRecipe
from asymmetry.core.fitting.asymmetry_global import fit_global
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.input_attribution import saved_fit_matches
from asymmetry.core.fitting.model_comparison import saved_fit_curves
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.rrf_offset import rrf_offset_model
from asymmetry.core.representation.base import FitSlot
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


def test_fit_window_tolerance_preserves_exact_causal_support():
    from asymmetry.core.fitting.time_response import TimeResponse

    first_arrival = -0.162
    time = np.array([np.nextafter(first_arrival, -np.inf), first_arrival, 0.0, 1.0])
    data = MuonDataset(time, np.full(4, 3.0), np.ones(4))
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.324))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    engine = FitEngine()

    fit = engine.fit(data, model.function, parameters, t_min=first_arrival)
    assert fit.success
    np.testing.assert_array_equal(fit.residual_time, time[1:])
    assert fit.input_attribution["min"] == first_arrival
    slot = FitSlot(
        model=model.to_dict(),
        result=fit_result_summary(fit),
        fit_range={"min": first_arrival, "max": None},
    )
    assert saved_fit_matches(slot, data)
    curves = saved_fit_curves(slot, data)
    assert curves.fit[0][0] == first_arrival
    np.testing.assert_array_equal(curves.residuals[0], time[1:])
    with pytest.raises(ValueError, match="before the first arrival"):
        engine.fit(data, model.function, parameters, t_min=time[0])
    with pytest.raises(ValueError, match="before the first arrival"):
        engine.fit(data.time_range(time[0]), model.function, parameters)


def test_array_and_preselected_plot_fits_share_causal_window_selection():
    from asymmetry.core.fitting.time_response import TimeResponse

    first_arrival = -0.162
    time = np.array([np.nextafter(first_arrival, -np.inf), first_arrival, 0.0, 1.0])
    data = MuonDataset(time, np.full(4, 3.0), np.ones(4))
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.324))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    engine = FitEngine()
    bounds = (first_arrival, time[-1])
    array_fit = engine.fit_arrays(
        time,
        data.asymmetry,
        data.error,
        model.function,
        parameters,
        t_min=bounds[0],
        t_max=bounds[1],
    )
    crop = data.time_range(*bounds)
    crop.metadata["fit_range"] = bounds
    plot_fit = engine.fit(crop, model.function, parameters)

    assert array_fit.success and plot_fit.success
    np.testing.assert_array_equal(array_fit.residual_time, time[1:])
    np.testing.assert_array_equal(plot_fit.residual_time, time[1:])


def test_saved_curve_can_draw_a_first_arrival_retained_beyond_the_upper_bound():
    from asymmetry.core.fitting.time_response import TimeResponse

    first_arrival = -0.162
    upper = np.nextafter(first_arrival, -np.inf)
    lower = np.nextafter(upper, -np.inf)
    data = MuonDataset(np.array([-1.0, first_arrival, 0.0]), np.full(3, 3.0), np.ones(3))
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.324))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    fit = FitEngine().fit(data, model.function, parameters, t_min=lower, t_max=upper)
    slot = FitSlot(
        model=model.to_dict(),
        result=fit_result_summary(fit),
        fit_range={"min": lower, "max": upper},
    )

    assert saved_fit_matches(slot, data)
    curves = saved_fit_curves(slot, data)
    np.testing.assert_array_equal(curves.fit[0], first_arrival)
    np.testing.assert_array_equal(curves.residuals[0], [first_arrival])


def test_saved_residual_evaluates_at_the_second_birth_discontinuity():
    from asymmetry.core.fitting.time_response import TimeResponse

    time = np.array([np.nextafter(0.162, -np.inf), 0.3, 0.4, 0.5])
    model = CompositeModel(["Exponential"], time_response=TimeResponse("double_delta", 0.324))
    values = dict(zip(model.param_names, (10.0, 5.0), strict=True))
    parameters = ParameterSet(
        [Parameter(name, value, fixed=True) for name, value in values.items()]
    )
    data = MuonDataset(time, model.function(time, **values), np.ones(time.size))
    fit = FitEngine().fit(data, model.function, parameters, t_min=0.162)
    slot = FitSlot(
        model=model.to_dict(), result=fit_result_summary(fit), fit_range={"min": 0.162, "max": None}
    )

    assert saved_fit_matches(slot, data)
    _, residuals = saved_fit_curves(slot, data).residuals
    np.testing.assert_array_equal(residuals, np.zeros(time.size))


@pytest.mark.parametrize("revision", [COMPUTATION_REVISION, "21"])
@pytest.mark.parametrize("open_lower", [False, True])
def test_saved_curve_uses_verified_centres_when_display_bounds_round_before_birth(
    revision, open_lower
):
    from asymmetry.core.fitting.time_response import TimeResponse

    birth = -0.1626
    time = np.array([birth, -0.1625756, -0.1625512, 1.0])
    data = MuonDataset(time, np.full(time.size, 3.0), np.ones(time.size))
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.3252))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    result = FitEngine().fit(
        data, model.function, parameters, t_min=np.nextafter(birth, np.inf), t_max=time[2]
    )
    window = (
        {"min": None, "max": np.nextafter(birth, -np.inf)}
        if open_lower
        else {"min": -0.163, "max": -0.163}
    )
    slot = FitSlot(
        model=model.to_dict(),
        result=fit_result_summary(result) | {"computation_revision": revision},
        fit_range=window,
    )

    curves = saved_fit_curves(slot, data)
    if revision != COMPUTATION_REVISION:
        assert not saved_fit_matches(slot, data)
        assert curves is None
        return
    assert saved_fit_matches(slot, data)
    assert (curves.fit[0][0], curves.fit[0][-1]) == (birth, time[2])
    np.testing.assert_array_equal(curves.residuals[0], time[:3])
    np.testing.assert_array_equal(curves.residuals[1], np.zeros(3))


@pytest.mark.parametrize("with_recipe", [False, True])
def test_factor_one_rebin_retains_the_literal_causal_crop(with_recipe):
    from asymmetry.core.fitting.time_response import TimeResponse

    birth = -0.162
    time = np.array([birth - 0.75e-12, birth - 0.5e-12, birth, 0.0, 1.0])
    recipe = EffectiveReductionRecipe.from_dict({"version": 1}) if with_recipe else None
    data = MuonDataset(time, np.full(time.size, 3.0), np.ones(time.size), effective_recipe=recipe)
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.324))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    crop = data.time_range(birth)

    copied = crop.merge_curve_bins(1)
    result = FitEngine().fit(copied, model.function, parameters)
    np.testing.assert_array_equal(result.residual_time, time[2:])
    merged = crop.merge_curve_bins(2)
    assert merged.time[0] < birth
    with pytest.raises(ValueError, match="before the first arrival"):
        FitEngine().fit(merged, model.function, parameters)


def test_partial_model_keeps_the_exact_first_arrival_guard():
    from asymmetry.core.fitting.time_response import TimeResponse

    birth = -0.162
    time = np.array([np.nextafter(birth, -np.inf), birth, 0.0, 1.0])
    data = MuonDataset(time, np.full(time.size, 3.0), np.ones(time.size))
    model = CompositeModel(["Constant"], time_response=TimeResponse("double_delta", 0.324))
    parameters = ParameterSet([Parameter(model.param_names[0], 3.0, fixed=True)])
    wrapped = partial(model.function)

    fit = FitEngine().fit(data, wrapped, parameters, t_min=birth)
    np.testing.assert_array_equal(fit.residual_time, time[1:])
    with pytest.raises(ValueError, match="before the first arrival"):
        FitEngine().fit(data, wrapped, parameters, t_min=time[0])


@pytest.mark.parametrize(
    "route", ["engine_global", "public_global", "preselected", "recipe", "compose", "rrf"]
)
def test_window_preserves_causal_support_across_inference_routes(route):
    from asymmetry.core.fitting.time_response import TimeResponse

    birth = -0.162
    time = np.array([np.nextafter(birth, -np.inf), birth, 0.0, 1.0])
    data = MuonDataset(time, np.full(time.size, 3.0), np.ones(time.size))
    response = TimeResponse("double_delta", 0.324)
    model = CompositeModel(
        ["Oscillatory" if route == "rrf" else "Constant"], time_response=response
    )
    parameters = ParameterSet([Parameter(name, 1.0, fixed=True) for name in model.param_names])
    engine = FitEngine()
    global_names = list(model.param_names)
    if route == "engine_global":
        members, _ = engine.global_fit(
            [data], model.function, global_names, [], {data.run_number: parameters}, t_min=birth
        )
        fit = members[data.run_number]
    elif route == "public_global":
        fit = fit_global(
            [data],
            model.function,
            global_params=global_names,
            local_params=[],
            initial_params=parameters,
            t_min=birth,
        ).dataset_results[0]
    elif route in ("preselected", "recipe"):
        if route == "recipe":
            data.effective_recipe = EffectiveReductionRecipe.from_dict({"version": 1})
        crop = data.time_range(birth).time_range(t_max=1.0)
        fit = engine.fit(crop, model.function, parameters)
    elif route == "compose":

        def source(time, **kwargs):
            return np.full_like(time, 2.0)

        fit = engine.fit(data, response.compose(source), parameters, t_min=birth)
    else:
        fit = engine.fit(data, rrf_offset_model(model, 2.0), parameters, t_min=birth)

    assert fit.success
    np.testing.assert_array_equal(fit.residual_time, time[1:])
    assert fit.input_attribution["min"] == birth


@pytest.mark.parametrize(
    "payload",
    [{}, [], "double_delta", {"mode": []}, {"mode": "double_delta", "separation_us": None}],
)
def test_malformed_authored_response_has_a_scoped_diagnosis(payload):
    from asymmetry.core.fitting.time_response import TimeResponse

    with pytest.raises(ValueError, match="[Tt]ime response|[Ss]upply.*separation"):
        TimeResponse.from_dict(payload)


def test_human_result_description_records_response_without_changing_expression():
    from asymmetry.core.fitting.time_response import TimeResponse

    off = CompositeModel(["Exponential", "Constant"])
    active = off.with_time_response(TimeResponse("double_delta", 0.324))
    assert "d=0.324 µs" in active.formula_string()
    assert "midpoint" in active.to_model_definition().description
    assert active.component_expression_string() == off.component_expression_string()
