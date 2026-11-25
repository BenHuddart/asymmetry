"""Real gleplot ranges include observations, errors and independently scaled fills."""

import re
import shutil
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.fitting.axis_transforms import AxisTransform
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.utils.gle_bounds import GleBounds
from asymmetry.gui.utils.gle_figure import gle_figure
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from tests.gui._trend_panel import select_params

glp = pytest.importorskip("gleplot")
pytestmark = pytest.mark.gui


def _limits(script, axis):
    match = re.search(rf"^\s*{axis}axis min (\S+) max (\S+)", script, re.MULTILINE)
    assert match is not None, script
    return tuple(map(float, match.groups()))


def _fit(name, scale=1.0, *, negative=False):
    model = ParameterCompositeModel.from_expression(
        "Linear - Constant" if negative else "Linear + Constant"
    )
    parameters = ParameterSet(
        [
            Parameter("m", value=0.05 * scale),
            Parameter("b", value=0.1 * scale),
            Parameter("c", value=(0.5 if negative else 0.05) * scale),
        ]
    )
    result = ParameterModelFitResult(success=True, parameters=parameters)
    return ParameterModelFit(
        parameter_name=name,
        x_key="temperature",
        active=True,
        ranges=[
            ModelFitRange(x_min=1.0, x_max=3.0, model=model, parameters=parameters, result=result)
        ],
    )


@pytest.fixture
def trend(qapp):
    panel = FitParametersPanel()
    panel.load_representation_series(
        [
            (
                "one",
                "One",
                [
                    {
                        "run_number": i,
                        "temperature": float(i),
                        "values": {"sigma": v, "Lambda": v * 1000},
                        "errors": {"sigma": 0.01, "Lambda": 10.0},
                    }
                    for i, v in enumerate([0.2, 0.4, 0.3], start=1)
                ],
            )
        ]
    )
    select_params(panel, ["sigma"])
    panel._model_fits = {"sigma": _fit("sigma"), "Lambda": _fit("Lambda", 1000)}
    yield panel
    panel.close()


@pytest.mark.parametrize("model_scale", [1.0, 10.0])
@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("error", [0.0, 0.01, np.nan])
@pytest.mark.parametrize("surface", ["global", "trend"])
def test_observations_and_complete_errors_contribute_to_export_bounds(
    qapp, tmp_path, trend, surface, error, components, model_scale
):
    panel = GlobalParameterFitWindow() if surface == "global" else trend
    if surface == "global":
        panel._groups = [
            ParameterGroupData(
                "one",
                "One",
                np.array([1.0, 2.0, 3.0]),
                np.array([0.2, 0.4, 0.3]),
                np.full(3, error),
                20.0,
            )
        ]
        fit = _fit("sigma", model_scale)
        panel._model = fit.ranges[0].model
        panel._result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            global_parameters=fit.ranges[0].parameters,
        )
        snapshot = replace(panel._gle_snapshot(), show_components=components)
    else:
        panel._model_fits["sigma"] = _fit("sigma", model_scale)
        for row in panel._rows:
            row.errors["sigma"] = error
        snapshot = replace(panel._gle_snapshot(), show_components=components)
    script = tmp_path / "observations.gle"
    try:
        build = snapshot.build_fits if surface == "global" else snapshot.build
        build(glp, script, tmp_path)
        lo, hi = _limits(script.read_text(), "y")
        assert hi > 0.4 + (error if np.isfinite(error) else 0)
        assert lo < 0.2 - (error if np.isfinite(error) else 0)
        assert hi > 0.3 * model_scale
        if components:
            assert lo <= 0
    finally:
        if surface == "global":
            panel.close()


@pytest.mark.parametrize("mode", ["Subplots", "Shared axes"])
def test_trend_component_stack_and_negative_total_extend_observation_range(trend, tmp_path, mode):
    trend._model_fits["sigma"] = _fit("sigma", 3.0, negative=True)
    snapshot = replace(
        trend._gle_snapshot(),
        selected_params=["sigma", "Lambda"],
        show_components=True,
        plot_mode=mode,
    )
    script = tmp_path / "negative.gle"
    snapshot.build(glp, script, tmp_path)
    lo, hi = _limits(script.read_text(), "y")
    assert lo < -1.05  # Full total model, including the negative constant.
    assert hi > 0.75  # Nonnegative stack extends above the observations.


def test_negative_observations_are_visible_below_component_baseline(trend, tmp_path):
    trend._rows[1].values["sigma"] = -0.4
    script = tmp_path / "negative.gle"
    replace(trend._gle_snapshot(), show_components=True).build(glp, script, tmp_path)
    assert _limits(script.read_text(), "y")[0] < -0.41


@pytest.mark.parametrize("asymmetric", [False, True])
def test_dual_axis_stacks_keep_units_and_fill_geometry_independent(trend, tmp_path, asymmetric):
    select_params(trend, ["sigma", "Lambda"])
    if asymmetric:
        trend._rows[1].values["sigma"] = -0.4
    script = tmp_path / "dual.gle"
    replace(
        trend._gle_snapshot(),
        show_components=True,
        plot_mode="Shared axes",
        selected_params=["sigma", "Lambda"],
    ).build(glp, script, tmp_path)
    content = script.read_text()
    lo, hi = _limits(content, "y")
    lo2, hi2 = _limits(content, "y2")
    assert hi < 1.0
    assert hi2 > 410
    assert hi2 < 1000
    if asymmetric:
        assert lo < -0.41
        assert (lo2, hi2) == pytest.approx((-20.5, 430.5))
    else:
        assert (lo2, hi2) == pytest.approx((lo * 1000, hi * 1000))
    # Fill sidecars are render coordinates, while edges and scientific sidecars
    # stay in their own units; independently invert the affine coordinate map.
    secondary_fills = list(tmp_path.glob("*component_lambda*fill*.dat"))
    assert secondary_fills
    all_bounds = []
    for path in secondary_fills:
        table = np.loadtxt(path, skiprows=1)
        all_bounds.append(lo2 + (table[:, 1:] - lo) * (hi2 - lo2) / (hi - lo))
    assert min(arr.min() for arr in all_bounds) == pytest.approx(0, abs=5e-4)
    assert max(arr.max() for arr in all_bounds) == pytest.approx(300, abs=5e-4)
    raw = np.loadtxt(script.with_suffix(".dat"), comments="!")
    assert np.any(raw == 400)
    secondary_fit = next(tmp_path.glob("*Lambda*.fit"))
    assert np.loadtxt(secondary_fit, comments="!")[:, 1].max() == pytest.approx(300, abs=5e-4)


def test_transformed_observation_values_and_errors_set_bounds(trend, tmp_path):
    trend._y_transforms["sigma"] = AxisTransform.custom("1000*x")
    script = tmp_path / "transformed.gle"
    trend._gle_snapshot().build(glp, script, tmp_path)
    lo, hi = _limits(script.read_text(), "y")
    assert 0 < lo < 190
    assert hi > 410
    assert hi < 1000


def test_log_bounds_ignore_ineligible_points_and_keep_positive_error_extent(tmp_path):
    figure = gle_figure(glp)
    ax = figure.add_subplot(111)
    bounds = GleBounds(log_x=True, log_y=frozenset({"y", "y2"}))
    bounds.include([-100, 1, 2, 3], [1000, -200, 0.2, 0.4], [0.1, 0.1, 0.3, 0.01])
    bounds.include([1, 2], [200, 400], [1, 10], yaxis="y2")
    bounds.include([1.5, 3], [0.1, 0.3])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yscale("log", axis="y2")
    bounds.apply(ax)
    assert 0 < ax.get_ylim()[0] < 0.1
    assert 0.5 < ax.get_ylim()[1] < 1
    assert 0 < ax.get_xlim()[0] <= 1
    assert ax.get_xlim()[1] < 4
    assert ax.get_ylim(axis="y2")[1] > 410
    figure.savefig(str(tmp_path / "log.gle"))


@pytest.mark.parametrize("explicit", [(None, 0.25), (-0.5, None), (0.5, -0.5)])
def test_authored_axis_endpoints_survive_range_resolution(tmp_path, explicit):
    figure = gle_figure(glp)
    ax = figure.add_subplot(111)
    ax.set_xlim(*explicit)
    ax.set_ylim(*explicit)
    ax.set_ylim(*explicit, axis="y2")
    bounds = GleBounds()
    bounds.include([1, 2, 3], [0.2, 0.4, 0.3], [0.01] * 3)
    bounds.include([1, 2], [200, 400], [1, 10], yaxis="y2")
    bounds.apply(ax)
    for resolved in (ax.get_xlim(), ax.get_ylim(), ax.get_ylim(axis="y2")):
        for original, actual in zip(explicit, resolved, strict=True):
            if original is not None:
                assert actual == original
    figure.savefig(str(tmp_path / "explicit.gle"))


@pytest.mark.parametrize("surface", ["global", "trend"])
def test_log_export_ranges_use_only_eligible_displayed_points(qapp, trend, tmp_path, surface):
    script = tmp_path / "log_export.gle"
    if surface == "global":
        panel = GlobalParameterFitWindow()
        panel._groups = [
            ParameterGroupData(
                "one",
                "One",
                np.array([-100.0, 1.0, 2.0, 3.0]),
                np.array([1000.0, -200.0, 0.2, 0.4]),
                np.array([0.1, 0.1, 0.3, 0.01]),
                20.0,
            )
        ]
        fit = _fit("sigma")
        panel._model = fit.ranges[0].model
        panel._result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            global_parameters=fit.ranges[0].parameters,
        )
        try:
            replace(panel._gle_snapshot(), fit_log_x=True, fit_log_y=True).build_fits(
                glp, script, tmp_path
            )
        finally:
            panel.close()
    else:
        trend._rows[0].temperature = -100.0
        trend._rows[0].values["sigma"] = 1000.0
        replace(trend._gle_snapshot(), log_x=True, log_y=frozenset({"sigma"})).build(
            glp, script, tmp_path
        )
    lo, hi = _limits(script.read_text(), "y")
    assert 0 < lo < 0.2
    assert 0.41 < hi < 1
    assert _limits(script.read_text(), "x")[0] > 0


def test_dual_log_axes_preserve_independent_ranges(trend, tmp_path):
    script = tmp_path / "dual_log.gle"
    snapshot = replace(
        trend._gle_snapshot(),
        selected_params=["sigma", "Lambda"],
        plot_mode="Shared axes",
        log_y=frozenset({"sigma", "Lambda"}),
    )
    snapshot.build(glp, script, tmp_path)
    lo, hi = _limits(script.read_text(), "y")
    lo2, hi2 = _limits(script.read_text(), "y2")
    assert 0 < lo < 0.19 and 0.41 < hi < 1
    assert (lo2, hi2) == pytest.approx((lo * 1000, hi * 1000))
    assert re.search(r"yaxis .* log", script.read_text())
    assert re.search(r"y2axis .* log", script.read_text())


@pytest.mark.skipif(shutil.which("gle") is None, reason="GLE compiler not installed")
@pytest.mark.parametrize("secondary", [False, True])
def test_native_trend_components_render_with_complete_observations(
    qapp, trend, tmp_path, secondary
):
    from PySide6.QtCore import QSize
    from PySide6.QtPdf import QPdfDocument

    from asymmetry.gui.utils.export import compile_gle

    script = tmp_path / "trend.gle"
    params = ["sigma", "Lambda"] if secondary else ["sigma"]
    replace(
        trend._gle_snapshot(), selected_params=params, plot_mode="Shared axes", show_components=True
    ).build(glp, script, tmp_path)
    compile_gle(shutil.which("gle"), script, ("pdf", "eps"), cwd=tmp_path, cairo=True)
    document = QPdfDocument()
    document.load(str(script.with_suffix(".pdf")))
    assert document.pageCount() == 1
    image = document.render(0, QSize(1200, 900))
    assert not image.isNull()
    assert image.save(str(tmp_path / "trend.png"))
    assert _limits(script.read_text(), "y")[1] > 0.41
    assert script.with_suffix(".eps").stat().st_size > 0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_singleton_logarithmic_padding_does_not_depend_on_units(scale):
    ax = gle_figure(glp).add_subplot(111)
    bounds = GleBounds(log_y=frozenset({"y"}))
    bounds.include([1.0], [scale])
    bounds.apply(ax)
    assert np.array(ax.get_ylim()) / scale == pytest.approx(10 ** np.array([-0.05, 0.05]))


@pytest.mark.parametrize("explicit", [(None, None), (10.0, None), (None, -10.0), (5.0, -5.0)])
def test_secondary_components_render_when_primary_transform_has_no_valid_values(
    trend, tmp_path, explicit, monkeypatch
):
    trend._y_transforms["sigma"] = AxisTransform.custom("sqrt(-x)")
    trend._model_fits.pop("sigma")
    axes = []
    add_subplot = glp.Figure.add_subplot

    def subplot(figure, *args, **kwargs):
        axis = add_subplot(figure, *args, **kwargs)
        axis.set_ylim(*explicit)
        axes.append(axis)
        return axis

    monkeypatch.setattr(glp.Figure, "add_subplot", subplot)
    script = tmp_path / "empty_primary.gle"
    replace(
        trend._gle_snapshot(),
        selected_params=["sigma", "Lambda"],
        plot_mode="Shared axes",
        show_components=True,
    ).build(glp, script, tmp_path)
    lo, hi = axes[0].get_ylim()
    lo2, hi2 = axes[0].get_ylim(axis="y2")
    for authored, actual in zip(explicit, (lo, hi), strict=True):
        if authored is not None:
            assert actual == authored
    assert np.isfinite([lo, hi, lo2, hi2]).all()
    fills = list(tmp_path.glob("*component_lambda*fill*.dat"))
    assert fills
    mapped = [np.loadtxt(path, skiprows=1)[:, 1:] for path in fills]
    fractions = [(arr - lo) / (hi - lo) for arr in mapped]
    # GLE's render sidecars round coordinates; compare physical axis positions
    # to 0.005% of the range independently of either axis's units or offset.
    assert min(arr.min() for arr in fractions) == pytest.approx((0 - lo2) / (hi2 - lo2), abs=5e-5)
    assert max(arr.max() for arr in fractions) == pytest.approx((300 - lo2) / (hi2 - lo2), abs=5e-5)


@pytest.mark.parametrize("surface", ["global", "trend"])
def test_components_without_logarithmically_eligible_x_have_no_fills(
    qapp, trend, tmp_path, surface
):
    script = tmp_path / "no_positive_x.gle"
    if surface == "trend":
        for row in trend._rows:
            row.temperature *= -1
        for fit in trend._model_fits.values():
            fit.ranges[0].x_min, fit.ranges[0].x_max = -3.0, -1.0
        replace(
            trend._gle_snapshot(),
            selected_params=["sigma", "Lambda"],
            plot_mode="Shared axes",
            show_components=True,
            log_x=True,
        ).build(glp, script, tmp_path)
    else:
        panel = GlobalParameterFitWindow()
        panel._groups = [
            ParameterGroupData(
                "one",
                "One",
                np.array([-3.0, -2.0, -1.0]),
                np.array([0.2, 0.4, 0.3]),
                np.full(3, 0.01),
                20.0,
            )
        ]
        fit = _fit("sigma")
        panel._model = fit.ranges[0].model
        panel._result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            global_parameters=fit.ranges[0].parameters,
        )
        try:
            replace(panel._gle_snapshot(), show_components=True, fit_log_x=True).build_fits(
                glp, script, tmp_path
            )
        finally:
            panel.close()
    assert "fill d" not in script.read_text()
    assert (
        script.with_suffix(".dat").exists() if surface == "trend" else list(tmp_path.glob("*.dat"))
    )


@pytest.mark.parametrize("mode", ["Subplots", "Shared axes"])
@pytest.mark.parametrize("ranges", [1, 2])
def test_independent_component_stacks_preserve_overlap_alpha(trend, tmp_path, mode, ranges):
    select_params(trend, ["sigma", "Lambda"])
    if ranges == 2:
        fit = trend._model_fits["sigma"]
        fit.ranges.append(replace(fit.ranges[0], x_min=1.5, x_max=2.5))
    snapshot = replace(
        trend._gle_snapshot(),
        show_components=True,
        plot_mode=mode,
        selected_params=["sigma", "Lambda"],
    )
    script = tmp_path / "stacks.gle"
    result = snapshot.build(glp, script, tmp_path)
    overlaps = mode == "Shared axes" or ranges == 2
    assert result.cairo is overlaps
    assert ("rgba255" in script.read_text()) is overlaps
