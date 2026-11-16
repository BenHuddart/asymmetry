"""Tests for :mod:`asymmetry.core.workflow.trend_fit`."""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.workflow.series import TrendTable
from asymmetry.core.workflow.trend_fit import fit_trend

#: An EuO-like precession frequency: y0 (1 - T/Tc)^beta.
_Y0, _TC, _BETA = 30.0, 70.0, 0.35
_TEMPERATURES = np.arange(5.0, 70.0, 5.0)
#: A run near Tc whose fit the series flagged, carrying a wild value.
_FLAGGED_RUN = "199"


def _frequency(temperature: float) -> float:
    return _Y0 * (1.0 - temperature / _TC) ** _BETA


def _trend() -> TrendTable:
    rows = [
        {
            "key": str(100 + index),
            "x": float(temperature),
            "frequency": _frequency(temperature),
            "frequency_err": 0.05,
            "params_at_bound": [],
            "flags": [],
        }
        for index, temperature in enumerate(_TEMPERATURES)
    ]
    rows.append(
        {
            "key": _FLAGGED_RUN,
            "x": 68.0,
            "frequency": 25.0,
            "frequency_err": 0.05,
            "params_at_bound": [],
            "flags": ["spurious_reseeded"],
        }
    )
    return TrendTable(
        order_key="temperature",
        columns=["key", "x", "frequency", "frequency_err", "flags"],
        rows=rows,
    )


def test_a_flagged_row_enters_and_is_named() -> None:
    fit = fit_trend(_trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0})

    assert _FLAGGED_RUN in fit.keys
    assert fit.flagged == [{"key": _FLAGGED_RUN, "flags": ["spurious_reseeded"]}]
    assert fit.excluded == []
    # The wild value pulls the fit off the curve the clean points trace ...
    assert fit.reduced_chi_squared > 100.0


def test_excluding_the_flagged_row_recovers_tc_and_beta() -> None:
    fit = fit_trend(
        _trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0}, exclude=[_FLAGGED_RUN]
    )

    assert fit.success
    assert fit.excluded == [{"key": _FLAGGED_RUN, "reason": "excluded"}]
    assert fit.flagged == []
    assert fit.to_dict()["n_points"] == len(_TEMPERATURES)
    assert fit.parameters["Tc"] == pytest.approx(_TC, rel=1e-3)
    assert fit.parameters["beta"] == pytest.approx(_BETA, rel=1e-2)
    assert fit.parameters["y0"] == pytest.approx(_Y0, rel=1e-3)
    assert "alpha" in fit.fixed
    assert "alpha" not in fit.uncertainties


def test_excluding_runs_and_windowing_x_are_both_reported() -> None:
    fit = fit_trend(
        _trend(),
        "frequency",
        "OrderParameter",
        fixed={"alpha": 1.0},
        exclude=["100", _FLAGGED_RUN],
        x_max=50.0,
    )

    reasons = {entry["key"]: entry["reason"] for entry in fit.excluded}
    assert reasons["100"] == "excluded"
    assert reasons["112"] == "outside the x range"
    assert all(_TEMPERATURES[int(key) - 100] <= 50.0 for key in fit.keys)
    assert fit.success


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"param": "Lambda"}, "no column 'Lambda'"),
        ({"param": "frequency_err"}, "no column 'frequency_err'"),
        ({"expression": "NoSuchModel"}, "NoSuchModel"),
        ({"fixed": {"gamma": 1.0}}, "gamma is not a parameter of 'OrderParameter'"),
        ({"initial": {"Tn": 60.0}}, "Tn is not a parameter of 'OrderParameter'"),
        ({"exclude": ["5"]}, "Not in the trend: 5 \\(it holds 100, "),
        ({"x_min": 60.0}, "point\\(s\\) are left to fit"),
    ],
)
def test_bad_requests_are_refused_naming_the_problem(kwargs, match) -> None:
    arguments = {"param": "frequency", "expression": "OrderParameter"} | kwargs
    param = arguments.pop("param")
    expression = arguments.pop("expression")
    with pytest.raises(ValueError, match=match):
        fit_trend(_trend(), param, expression, **arguments)


def test_a_turning_point_is_found_only_when_both_ends_clear_it() -> None:
    from asymmetry.core.workflow.trend_fit import turning_point

    x = np.array([5.0, 40.0, 60.0, 100.0, 200.0])
    errors = np.full(5, 0.01)
    # A minimum at 40 K both ends rise well clear of.
    assert turning_point(x, np.array([0.10, 0.02, 0.05, 0.4, 2.2]), errors) == 40.0
    # Monotonic: none.
    assert turning_point(x, np.array([0.01, 0.02, 0.05, 0.4, 2.2]), errors) is None
    # An interior dip inside the errors: none.
    assert turning_point(x, np.array([0.03, 0.02, 0.05, 0.4, 2.2]), errors) is None


def test_the_fit_reports_its_span_and_units() -> None:
    fit = fit_trend(
        _trend(), "frequency", "OrderParameter", fixed={"alpha": 1.0}, exclude=[_FLAGGED_RUN]
    )
    assert fit.x_fitted == (float(_TEMPERATURES[0]), float(_TEMPERATURES[-1]))
    assert fit.units["Tc"] == "K"
    assert fit.turning_point is None


def _stored_fit(**changes) -> dict:
    """A stored Linear trend fit as ``trend --model`` writes it."""
    return {
        "param": "Lambda",
        "expression": "Linear",
        "success": True,
        "parameters": {"m": 0.5, "b": 0.1},
        "uncertainties": {"m": 0.02, "b": 0.01},
        "reduced_chi_squared": 4.0,
        "params_at_bound": [],
    } | changes


def test_a_fit_trend_row_carries_the_scaled_error_and_the_fits_state() -> None:
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    fits = {
        "hot": _stored_fit(),
        "cold": _stored_fit(success=False, reduced_chi_squared=50.0),
        "warm": _stored_fit(params_at_bound=["m"], uncertainties={"b": 0.01}),
    }
    axis = ScanAxis("temperature", {"hot": 300.0, "cold": 200.0, "warm": 250.0})

    trend = fit_trend_table(fits, "m", axis)

    assert trend.columns == ["key", "x", "m", "m_err", "flags"]
    rows = {row["key"]: row for row in trend.rows}
    assert [row["key"] for row in trend.rows] == ["cold", "warm", "hot"]
    # sqrt(chi2_red) = 2 scales a converged fit's error; a failed fit's is left alone.
    assert rows["hot"]["m_err"] == pytest.approx(0.04)
    assert rows["cold"]["m_err"] == pytest.approx(0.02)
    assert rows["cold"]["flags"] == ["failed"]
    # A parameter the law held has no error, so the row enters no law.
    assert rows["warm"]["m_err"] is None
    assert rows["warm"]["flags"] == ["bound_pinned"]


def test_a_fit_trend_of_a_parameter_the_law_lacks_is_refused() -> None:
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    with pytest.raises(ValueError, match="The trend fit of a has no parameter 'Ea'"):
        fit_trend_table({"a": _stored_fit()}, "Ea", ScanAxis("temperature", {"a": 1.0}))


@pytest.mark.parametrize("fixed", [{}, {"m": 2.0}])
def test_generalized_redfield_fit_recovers_positive_physical_parameters(fixed) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.fitting.seeding import seed_trend_parameters

    model = ParameterCompositeModel.from_expression("Redfield")
    x = np.linspace(0, 12000, 41)
    y = model.function(x, D=28.0, nu=140.0, m=2.0)
    rows = [
        {
            "key": str(i),
            "x": float(b),
            "Lambda": float(rate),
            "Lambda_err": 0.02,
            "params_at_bound": [],
            "flags": [],
        }
        for i, (b, rate) in enumerate(zip(x, y, strict=True))
    ]
    trend = TrendTable(
        order_key="field", columns=["key", "x", "Lambda", "Lambda_err", "flags"], rows=rows
    )
    seeds = seed_trend_parameters(model, x, y)
    assert seeds["m"].min > 0
    assert seeds["m"].value == 2.0
    assert (
        seed_trend_parameters(ParameterCompositeModel.from_expression("Linear"), x, y)["m"].min
        == -np.inf
    )
    fit = fit_trend(trend, "Lambda", "Redfield", fixed=fixed)
    assert fit.success
    assert fit.parameters["D"] == pytest.approx(28.0, rel=0.01)
    assert fit.parameters["nu"] == pytest.approx(140.0, rel=0.01)
    assert fit.parameters["m"] == pytest.approx(2.0, rel=0.01)


@pytest.fixture
def gaussian_scan():
    """Actual Gaussian fits to a synthetic single-crystal Brandt width scan."""
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.models import brandt_field_width_sigma
    from asymmetry.core.workflow.recipe import FitRecipe
    from asymmetry.core.workflow.series import ScanAxis, fit_series

    fields = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 2000.0])
    widths = brandt_field_width_sigma(fields, 220.0, 2.0)
    widths /= GaussianWidth.GAUSSIAN_COMPONENT.value
    time = np.linspace(0.0, 5.0, 301)
    datasets = {
        run: MuonDataset(
            time=time,
            asymmetry=0.2 * np.exp(-np.square(width * time)),
            error=np.full_like(time, 0.0005),
            metadata={"run_number": run, "field": field},
        )
        for run, (field, width) in enumerate(zip(fields, widths, strict=True), 1)
    }
    recipe = FitRecipe.from_expression("Gaussian", dataset=datasets[1])
    outcome = fit_series(
        datasets, recipe, axis=ScanAxis("field", dict(enumerate(fields, 1))), name="gaussian"
    )
    assert all(entry["success"] for entry in outcome.results)
    return outcome


def test_known_gaussian_series_refuses_second_moment_fit(gaussian_scan):
    with pytest.raises(ValueError, match="Gaussian.*second-moment.*sqrt"):
        fit_trend(gaussian_scan.trend, "sigma", "SC_Brandt_VortexLattice")


def test_cli_refuses_known_gaussian_series(tmp_path, capsys, gaussian_scan):
    from asymmetry.cli import main
    from asymmetry.core.workflow.workdir import WorkDir

    workdir = WorkDir(tmp_path / "work")
    workdir.bind(tmp_path, None)
    workdir.write_series(
        "gaussian", {"kind": "series", "trend_fits": {}, **gaussian_scan.to_dict()}
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "trend",
                str(tmp_path),
                "--workdir",
                str(workdir.root),
                "--series",
                "gaussian",
                "--model",
                "SC_Brandt_VortexLattice",
                "--param",
                "sigma",
                "--json",
            ]
        )
    assert error.value.code == 1
    assert "second-moment" in capsys.readouterr().err


def test_explicit_conversion_recovers_lambda_and_scales_errors(gaussian_scan, monkeypatch):
    import asymmetry.core.workflow.trend_fit as trend_module
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.sc import GaussianWidth

    original = trend_module.fit_parameter_model
    captured = {}

    def capture(x, y, errors, *args, **kwargs):
        captured.update(y=y, errors=errors)
        return original(x, y, errors, *args, **kwargs)

    monkeypatch.setattr(trend_module, "fit_parameter_model", capture)
    fit = fit_trend(
        gaussian_scan.trend,
        "sigma",
        "SC_Brandt_VortexLattice",
        y_transform=AxisTransform.custom("sqrt(2)*x"),
        initial={"lambda_ab": 220},
        fixed={"Bc2": 2, "sigma_bg": 0},
    )
    assert fit.success
    assert fit.parameters["lambda_ab"] == pytest.approx(220, rel=1e-5)
    factor = GaussianWidth.GAUSSIAN_COMPONENT.value
    np.testing.assert_allclose(
        captured["y"], [r["sigma"] * factor for r in gaussian_scan.trend.rows]
    )
    np.testing.assert_allclose(
        captured["errors"], [r["sigma_err"] * factor for r in gaussian_scan.trend.rows]
    )
    assert fit.input_width_context["transform"]["expression"] == "sqrt(2)*x"


def test_old_unknown_table_remains_usable_and_load_does_not_fit(
    gaussian_scan, tmp_path, monkeypatch
):
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.workflow.workdir import WorkDir

    legacy = gaussian_scan.trend.to_dict()
    for row in legacy["rows"]:
        row.pop("parameter_sources")
    table = TrendTable(**legacy)
    fit = fit_trend(table, "sigma", "SC_Brandt_VortexLattice", fixed={"Bc2": 2, "sigma_bg": 0})
    assert fit.parameters["lambda_ab"] == pytest.approx(
        220 * GaussianWidth.GAUSSIAN_COMPONENT.value**0.5, rel=1e-5
    )
    workdir = WorkDir(tmp_path / "work")
    payload = {"kind": "series", "trend_fits": {}, **gaussian_scan.to_dict(), "trend": legacy}
    workdir.write_series("legacy", payload)
    monkeypatch.setattr(
        "asymmetry.core.fitting.parameter_models.fit_parameter_model",
        lambda *a, **k: pytest.fail("load must not fit"),
    )
    assert workdir.read_series("legacy")["trend"] == legacy


def test_cli_explicit_conversion_is_saved_without_changing_source_table(
    tmp_path, capsys, gaussian_scan, monkeypatch
):
    import json

    from asymmetry.cli import main, plots
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.workflow.workdir import WorkDir

    displayed = {}
    original_frame = plots._trend_frame

    def capture_frame(y, errors, flagged):
        displayed.update(values=y, errors=errors)
        return original_frame(y, errors, flagged)

    monkeypatch.setattr(plots, "_trend_frame", capture_frame)
    workdir = WorkDir(tmp_path / "work")
    workdir.bind(tmp_path, None)
    workdir.write_series(
        "gaussian", {"kind": "series", "trend_fits": {}, **gaussian_scan.to_dict()}
    )
    main(
        [
            "trend",
            str(tmp_path),
            "--workdir",
            str(workdir.root),
            "--series",
            "gaussian",
            "--model",
            "SC_Brandt_VortexLattice",
            "--param",
            "sigma",
            "--y-transform",
            "sqrt(2)*x",
            "--fix",
            "Bc2=2",
            "--fix",
            "sigma_bg=0",
            "--initial",
            "lambda_ab=220",
            "--plot",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["fit"]["parameters"]["lambda_ab"] == pytest.approx(220, rel=1e-5)
    stored = workdir.read_series("gaussian")
    assert stored["trend"] == gaussian_scan.trend.to_dict()
    assert (
        stored["trend_fits"]["sigma:SC_Brandt_VortexLattice"]["input_width_context"]["transform"][
            "expression"
        ]
        == "sqrt(2)*x"
    )

    factor = GaussianWidth.GAUSSIAN_COMPONENT.value
    np.testing.assert_allclose(
        displayed["values"], [r["sigma"] * factor for r in gaussian_scan.trend.rows]
    )
    np.testing.assert_allclose(
        displayed["errors"], [r["sigma_err"] * factor for r in gaussian_scan.trend.rows]
    )
    assert workdir.trend_plot_path("gaussian", "sigma").is_file()


def test_temperature_law_outputs_inherit_only_selected_widths():
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.models import sc_s_wave
    from asymmetry.core.fitting.sc.trend_width import WidthSource
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    temperatures = np.linspace(1, 25, 12)
    values = sc_s_wave(temperatures, 1.2, 20, 1.764, 0.1)
    source = WidthSource("Gaussian", 1, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,)).to_dict()
    table = TrendTable(
        "temperature",
        ["key", "x", "width", "width_err", "flags"],
        [
            {
                "key": str(i),
                "x": t,
                "width": y,
                "width_err": 0.01,
                "parameter_sources": {"width": source},
                "params_at_bound": [],
                "flags": [],
            }
            for i, (t, y) in enumerate(zip(temperatures, values, strict=True))
        ],
    )
    fits = [
        fit_trend(
            table,
            "width",
            "SC_SWave",
            fixed={"Tc": 20, "gap_ratio": 1.764, "sigma_bg": bg},
            y_transform=transform,
        )
        for bg, transform in [
            (0.1, AxisTransform.identity()),
            (0.1 * GaussianWidth.GAUSSIAN_COMPONENT.value, AxisTransform.custom("sqrt(2)*x")),
        ]
    ]
    assert fits[0].success and fits[1].success
    assert fits[0].parameters["sigma_0"] == pytest.approx(1.2, rel=1e-4)
    assert fits[1].parameters["sigma_0"] == pytest.approx(
        1.2 * GaussianWidth.GAUSSIAN_COMPONENT.value, rel=1e-4
    )
    assert fits[0].parameter_sources["sigma_0"]["widths"] == ["GAUSSIAN_COMPONENT"]
    assert fits[1].parameter_sources["sigma_0"]["widths"] == ["SECOND_MOMENT"]
    assert fits[0].parameter_sources["Tc"]["widths"] == []
    selected = table.to_dict()
    for row in selected["rows"]:
        row["parameter_sources"]["width"] = {**source, "widths": ["SECOND_MOMENT"]}
    selected["rows"][0]["parameter_sources"]["width"] = source
    selected["rows"][0]["x"] = float("nan")
    fitted = fit_trend(
        TrendTable(**selected),
        "width",
        "SC_SWave",
        fixed={"Tc": 20, "gap_ratio": 1.764, "sigma_bg": 0.1},
    )
    assert fitted.parameter_sources["sigma_0"]["widths"] == ["SECOND_MOMENT"]

    derived = fit_trend_table(
        {"a": fits[0].to_dict(), "b": fits[1].to_dict()},
        "sigma_0",
        ScanAxis("field", {"a": 0.01, "b": 0.02}),
    )
    assert [r["parameter_sources"]["sigma_0"]["widths"] for r in derived.rows] == [
        ["GAUSSIAN_COMPONENT"],
        ["SECOND_MOMENT"],
    ]
    assert ParameterCompositeModel.from_expression("SC_SWave").param_names


def test_cross_group_width_check_uses_selected_rows_and_round_trips_context():
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import (
        CrossGroupFitResult,
        ParameterCompositeModel,
        ParameterGroupData,
        global_fit_parameter_model,
    )
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.models import brandt_field_width_sigma
    from asymmetry.core.fitting.sc.trend_width import TrendWidthContext, WidthSource

    fields = np.linspace(100.0, 10000.0, 10)
    values = brandt_field_width_sigma(fields, 220, 2)
    source = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    groups = [
        ParameterGroupData(
            str(i),
            str(i),
            fields,
            values,
            np.full_like(values, 0.01),
            i,
            width_context=TrendWidthContext(
                "width", (source,) * len(fields), AxisTransform.custom("sqrt(2)*x")
            ),
        )
        for i in range(2)
    ]
    model = ParameterCompositeModel.from_expression("SC_Brandt_VortexLattice")
    result = global_fit_parameter_model(
        groups, model, ["lambda_ab"], [], {"Bc2": 2, "sigma_bg": 0}, {"lambda_ab": 220}
    )
    assert result.success
    assert result.global_parameters["lambda_ab"].value == pytest.approx(220, rel=1e-5)
    assert (
        CrossGroupFitResult.from_dict(result.to_dict()).input_width_contexts
        == result.input_width_contexts
    )
    assert (
        ParameterGroupData.from_dict(groups[0].to_dict()).width_context == groups[0].width_context
    )
    groups[1].width_context = TrendWidthContext("width", (None,) * 9 + (source,))
    with pytest.raises(ValueError, match="Gaussian.*second-moment"):
        global_fit_parameter_model(groups, model, ["lambda_ab"], [], {"Bc2": 2, "sigma_bg": 0})
    # A mismatched row outside the authored windows does not poison a different fit.
    result = global_fit_parameter_model(
        groups, model, ["lambda_ab"], [], {"Bc2": 2, "sigma_bg": 0}, windows=[(100.0, 8000.0)]
    )
    assert result.success
    assert len(result.input_width_contexts["1"]["sources"]) == 8


def test_nonlinear_known_width_origin_survives_temperature_fit_and_derived_trend():
    from asymmetry.core.fitting.axis_transforms import AxisTransform
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.fitting.sc import GaussianWidth
    from asymmetry.core.fitting.sc.trend_width import (
        TrendWidthContext,
        WidthSource,
        fitted_width_sources,
    )
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    source = WidthSource("Gaussian", 0, "sigma", (GaussianWidth.GAUSSIAN_COMPONENT,))
    context = TrendWidthContext("sigma", (source,) * 5, AxisTransform.custom("x*x"))
    sources = fitted_width_sources(
        ParameterCompositeModel.from_expression("SC_SWave"), context.to_dict()
    )
    assert sources["sigma_0"]["widths"] == ["GAUSSIAN_COMPONENT"]
    assert sources["sigma_0"]["unverified_transforms"] == ["x*x"]
    stored = {
        "parameters": {"sigma_0": 1.0},
        "uncertainties": {"sigma_0": 0.01},
        "params_at_bound": [],
        "success": True,
        "reduced_chi_squared": 1.0,
        "parameter_sources": sources,
    }
    names = [str(i) for i in range(5)]
    trend = fit_trend_table(
        dict.fromkeys(names, stored),
        "sigma_0",
        ScanAxis("field", {name: 0.01 * (i + 1) for i, name in enumerate(names)}),
    )
    with pytest.raises(ValueError, match="unverified transform.*second-moment"):
        fit_trend(
            trend,
            "sigma_0",
            "SC_Brandt_VortexLattice",
            y_transform=AxisTransform.custom("sqrt(2)*x"),
        )


@pytest.mark.parametrize("model,param", [("Constant", "c"), ("Linear", "b")])
def test_declared_same_width_scalar_outputs_keep_gaussian_convention(gaussian_scan, model, param):
    from asymmetry.core.workflow.series import ScanAxis
    from asymmetry.core.workflow.trend_fit import fit_trend_table

    fitted = fit_trend(gaussian_scan.trend, "sigma", model)
    assert fitted.success
    assert fitted.parameter_sources[param]["widths"] == ["GAUSSIAN_COMPONENT"]
    if model == "Linear":
        assert fitted.parameter_sources["m"]["widths"] == []
    names = [str(i) for i in range(5)]
    trend = fit_trend_table(
        dict.fromkeys(names, fitted.to_dict()),
        param,
        ScanAxis("field", {name: 0.01 * (i + 1) for i, name in enumerate(names)}),
    )
    with pytest.raises(ValueError, match="Gaussian.*second-moment"):
        fit_trend(trend, param, "SC_Brandt_VortexLattice")
