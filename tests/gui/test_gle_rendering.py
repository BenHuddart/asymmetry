"""Real-library and native-renderer contracts for scientific GLE exports."""

import shutil
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtPdf import QPdfDocument
from PySide6.QtWidgets import QFileDialog

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, PhaseDecoration
from asymmetry.gui.panels.plot_panel import PlotPanel
from asymmetry.gui.utils import gle_export
from asymmetry.gui.utils.gle_figure import gle_figure, save_gle_figure
from asymmetry.gui.utils.parameter_gle import ParameterGleSnapshot
from asymmetry.gui.utils.plot_gle import PlotGleSnapshot
from asymmetry.gui.windows.gle_preview import GlePlotPreview, close_gle_previews
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from tests._qt_helpers import wait_for

glp = pytest.importorskip("gleplot")
pytestmark = pytest.mark.gui


@pytest.mark.parametrize("error", [0.0, 0.01])
def test_time_export_keeps_observation_markers_without_connecting_lines(qapp, tmp_path, error):
    panel = PlotPanel()
    time = np.linspace(0, 4, 9)
    data = MuonDataset(time, np.exp(-time), np.full_like(time, error))
    try:
        panel.plot_dataset(data)
        script = tmp_path / "observations.gle"
        panel._gle_snapshot(panel._collect_export_payloads()).build(glp, script, tmp_path)
        content = script.read_text()
        assert "marker FCIRCLE" in content
        assert " line " not in content
        assert " smooth" not in content
        table = np.loadtxt(next(tmp_path.glob("*.dat")), comments="!")
        np.testing.assert_allclose(table[:, 1], data.asymmetry)
        np.testing.assert_allclose(table[:, 2], data.error)
    finally:
        panel.close()


@pytest.mark.skipif(shutil.which("gle") is None, reason="GLE compiler not installed")
@pytest.mark.parametrize("output_format", ["pdf", "eps"])
@pytest.mark.parametrize("surface", ["frequency", "phase", "components"])
def test_filled_export_compiles_and_displays_vector_preview(
    qapp, tmp_path, monkeypatch, output_format, surface
):
    destination = tmp_path / "filled.gleplot"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(destination), ""))
    monkeypatch.setattr(gle_export, "get_gle_executable", lambda: shutil.which("gle"))
    compiler = gle_export.compile_gle
    compilations = []

    def compile(*args, **kwargs):
        result = compiler(*args, **kwargs)
        compilations.append(result)
        return result

    monkeypatch.setattr(gle_export, "compile_gle", compile)
    if surface == "frequency":
        panel = PlotPanel(domain="frequency")
        frequency = np.linspace(0, 100, 32)
        data = MuonDataset(
            frequency,
            np.exp(-(((frequency - 40) / 10) ** 2)),
            np.full_like(frequency, 0.02),
            metadata={"plot_domain": "frequency", "run_number": 1},
        )
        panel.plot_dataset(data)
        snapshot = panel._gle_snapshot(panel._collect_export_payloads())
    elif surface == "components":
        panel = GlobalParameterFitWindow()
        panel._groups = [
            ParameterGroupData(
                "one",
                "One",
                np.array([1.0, 2.0, 3.0]),
                np.array([0.2, 0.4, 0.3]),
                np.array([0.01, 0.01, 0.01]),
                20.0,
            )
        ]
        panel._model = ParameterCompositeModel.from_expression("Linear + Constant")
        panel._result = CrossGroupFitResult(
            success=True,
            chi_squared=1.0,
            reduced_chi_squared=1.0,
            global_parameters=ParameterSet(
                [
                    Parameter("m", value=0.05),
                    Parameter("b", value=0.1),
                    Parameter("c", value=0.05),
                ]
            ),
        )
        panel._parameter_name = "Lambda"
        snapshot = replace(panel._gle_snapshot(), show_components=True)
    else:
        panel = FitParametersPanel()
        panel.load_representation_series(
            [
                (
                    "phase",
                    "Phase I",
                    [
                        {
                            "run_number": i,
                            "temperature": float(i),
                            "values": {"sigma": v},
                            "errors": {"sigma": 0.01},
                        }
                        for i, v in enumerate([0.2, 0.4, 0.3], start=1)
                    ],
                )
            ],
            phase_by_id={
                "phase": PhaseDecoration(
                    "blue", "blue", 1, "Phase I", "temperature", (1.0, 2.5), None, (2.5, 0.2)
                )
            },
        )
        snapshot = panel._gle_snapshot()
    try:
        gle_export.run_gle_export(
            panel,
            dialog_title="Export",
            default_name="filled.gleplot",
            output_format=output_format,
            prepare=lambda: snapshot.build_fits if surface == "components" else snapshot.build,
        )
        preview = panel.findChild(GlePlotPreview)
        assert preview is not None
        wait_for(lambda: not preview.tasks.active_count, qapp)
        assert not preview.status.text().startswith("Export failed"), preview.status.text()
        assert preview.document.pageCount() == 1
        assert len(compilations) == 1
        assert "-cairo" in compilations[0].args
        assert "PostScript fonts not supported" not in compilations[0].stdout
        script = destination / "filled.gle"
        assert "rgba255" in script.read_text()
        if surface == "components":
            image = preview.document.render(0, QSize(900, 700))
            assert not image.isNull()
            assert all(
                image.pixelColor(x, y).alpha() == 0 or image.pixelColor(x, y).lightness() > 245
                for y in (0, image.height() - 1)
                for x in range(image.width())
            ), "Title or axis label clipped at the page boundary"
        assert script.with_suffix("." + output_format).stat().st_size > 0
        pdf = QPdfDocument()
        pdf.load(str(script.with_suffix(".pdf")))
        assert pdf.pageCount() == 1
    finally:
        close_gle_previews(panel)
        panel.close()


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("secondary_axis", [False, True])
def test_phase_guides_cover_fit_limits_without_mixing_y_axis_units(
    tmp_path, logarithmic, secondary_axis
):
    figure = gle_figure(glp)
    axis = figure.add_subplot(111)
    axis.plot([1, 2, 3], [0.2, 0.4, 0.3])
    # The fitted curve extends far beyond the observations on the primary axis.
    axis.plot([1, 2, 3], [0.1, 100.0, 0.1])
    axis.set_ylim(0.1, 100.0)
    if logarithmic:
        axis.set_yscale("log")
    if secondary_axis:
        axis.plot([1, 2, 3], [1e6, 2e6, 3e6], yaxis="y2")
        axis.set_ylim(1e6, 3e6, axis="y2")
    phase = PhaseDecoration("blue", "blue", 1, "I", "temperature", (1.5, 2.5), None, (2.5, 0.2))
    ParameterGleSnapshot._add_gle_phase_band(axis, phase)
    script = tmp_path / "phase.gle"
    figure.savefig(str(script))
    assert axis.get_ylim() == (0.1, 100.0)
    if secondary_axis:
        assert axis.get_ylim(axis="y2") == (1e6, 3e6)
    content = script.read_text()
    assert "rgba255" in content
    # Read the guide's plotted coordinates independently of gleplot's model.
    spans = []
    for path in tmp_path.glob("*.dat"):
        table = np.loadtxt(path, skiprows=1)
        if table.ndim == 2 and table.shape[1] == 3:
            spans.append(table)
    assert len(spans) == 2
    for table in spans:
        bounds = np.sort(table[:, 1:], axis=1)
        np.testing.assert_allclose(bounds[:, 0], 0.1)
        np.testing.assert_allclose(bounds[:, 1], 100.0)


@pytest.mark.parametrize("name", ["Lambda", "sigma", "Delta"])
def test_microsecond_parameter_labels_do_not_draw_literal_group_braces(tmp_path, name):
    figure = gle_figure(glp)
    axis = figure.add_subplot(111)
    axis.plot([1, 2], [0.2, 0.4])
    axis.set_ylabel(get_param_info(name).gle_label())
    script = tmp_path / "units.gle"
    figure.savefig(str(script))
    content = script.read_text()
    assert r"\char{123}" not in content
    assert r"\char{125}" not in content


@pytest.mark.parametrize("projection", ["x", "y", "z"])
def test_projection_export_preserves_subscripts_and_chi_squared_annotation(tmp_path, projection):
    key = f"P_{projection}"
    payload = {
        "data": {"t": [0.0, 1.0], "y": [0.2, 0.3], "err": [0.01, 0.01]},
        "label": "one",
        "annotations": [{"x": 0.5, "y": 0.5, "text": "χ² = 1.2"}],
    }
    snapshot = PlotGleSnapshot(axes=((key, [payload], ""),), y_limits={key: (0.0, 1.0)})
    script = tmp_path / "projection.gle"
    snapshot.build(glp, script, tmp_path)
    content = script.read_text()
    assert f"a_{{0}} P_{{{projection}}}(t) (%)" in content
    assert r"a\_0" not in content
    assert "chi^{2} = 1.2" in content
    assert r"\char{94}" not in content


@pytest.mark.parametrize("translucent", [False, True])
def test_export_font_preserves_default_and_literal_label_punctuation(tmp_path, translucent):
    from gleplot.cairo_support import is_cairo_safe_font, text_glyph_warning

    figure = gle_figure(glp)
    if not translucent:
        assert figure.style.font == glp.GlobalConfig.get_style().font
    axis = figure.add_subplot(111)
    axis.plot([1, 2], [0.2, 0.4])
    if translucent:
        axis.fill_between([1, 2], [0.1, 0.3], [0.3, 0.5], color="blue", alpha=0.25)
    axis.set_title("A_B {C} <D> |F|")
    script = tmp_path / "punctuation.gle"
    cairo = save_gle_figure(figure, script)
    assert cairo is translucent
    if cairo:
        assert is_cairo_safe_font(figure.style.font)
    else:
        assert figure.style.font == glp.GlobalConfig.get_style().font
    assert text_glyph_warning(figure.style.font, script.read_text()) is None
