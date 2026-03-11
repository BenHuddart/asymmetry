"""Tests for FitParametersPanel GLE export helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    ParameterCompositeModel,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow
from asymmetry.gui.panels.fit_parameters_panel import (
    _format_gle_label,
    _format_gle_legend_label,
    _format_plot_label,
    _format_plot_legend_label,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """Ensure a QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def panel(qapp: QApplication) -> FitParametersPanel:
    """Create a panel with predictable test data."""
    w = FitParametersPanel()
    w._rows = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"A0": 0.22, "Lambda": 0.12},
            errors={"A0": 0.02, "Lambda": 0.01},
        ),
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        ),
    ]
    w._varying_params = ["A0", "Lambda"]
    w._inferred_x_key = "field"
    w._x_combo.setCurrentText("Auto")
    return w


def test_gle_columns_for_param(panel: FitParametersPanel) -> None:
    assert panel._gle_columns_for_param("A0") == (4, 5)
    assert panel._gle_columns_for_param("Lambda") == (6, 7)
    assert panel._gle_columns_for_param("missing") is None


def test_write_gle_data_file_contains_column_map_and_sorted_rows(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
    panel._global_params = ParameterSet([Parameter("Lambda", value=0.5)])

    out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(out)
    text = out.read_text(encoding="utf-8")

    assert "! Column map:" in text
    assert "!   Lambda (μs⁻¹) = 0.5" in text
    assert "!   c 1 = Run" in text
    assert "!   c 2 = B_field(G)" in text
    assert "!   c 3 = Temperature(K)" in text
    assert "!   c 4 = A0 (%)" in text
    assert "!   c 5 = err_A0 (%)" in text
    assert "!   c 6 = Lambda (μs⁻¹)" in text
    assert "!   c 7 = err_Lambda (μs⁻¹)" in text

    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_lines) == 2

    # Rows should be sorted by inferred x-axis (field): 100 before 200.
    assert float(data_lines[0].split()[0]) == pytest.approx(1.0)
    assert float(data_lines[0].split()[1]) == pytest.approx(100.0)
    assert float(data_lines[1].split()[0]) == pytest.approx(2.0)
    assert float(data_lines[1].split()[1]) == pytest.approx(200.0)


def test_write_gle_data_file_includes_combined_run_mapping_comments(tmp_path: Path, qapp: QApplication) -> None:
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=-1,
            run_label="3039 + 3040",
            field=150.0,
            temperature=10.0,
            values={"A0": 0.21},
            errors={"A0": 0.01},
            combined_from=[3039, 3040],
        )
    ]
    panel._varying_params = ["A0"]
    panel._inferred_x_key = "run"
    panel._x_combo.setCurrentText("Run")

    out = tmp_path / "fit_parameters.dat"
    panel._write_gle_data_file(out)
    text = out.read_text(encoding="utf-8")

    assert "! Combined run mapping:" in text
    assert "!   -1 = 3039 + 3040" in text

    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("!")]
    assert len(data_lines) == 1
    assert float(data_lines[0].split()[0]) == pytest.approx(-1.0)
    assert float(data_lines[0].split()[1]) == pytest.approx(150.0)
    assert float(data_lines[0].split()[2]) == pytest.approx(10.0)


def test_export_csv_headers_include_units(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel._refresh_table()
    out = tmp_path / "fit_parameters.csv"

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out), "CSV files (*.csv)"),
    )

    panel._export_csv()

    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "Run,B (G),T (K),A0 (%),err_A0 (%),Lambda (μs⁻¹),err_Lambda (μs⁻¹)"


def test_refresh_table_uses_run_label_for_combined_rows(qapp: QApplication) -> None:
    panel = FitParametersPanel()
    panel._rows = [
        _FitRow(
            run_number=-1,
            run_label="3039 + 3040",
            field=150.0,
            temperature=10.0,
            values={"A0": 0.21},
            errors={"A0": 0.01},
            combined_from=[3039, 3040],
        )
    ]
    panel._varying_params = ["A0"]
    panel._refresh_table()

    run_item = panel._table.item(0, 0)
    assert run_item is not None
    assert run_item.text() == "3039 + 3040"


def test_background_labels_use_subscript_formatting() -> None:
    assert _format_plot_label("A_bg") == "$A_{bg}$ (%)"
    assert _format_plot_legend_label("A_bg") == "$A_{bg}$"
    assert _format_gle_label("A_bg") == "{\\it{A}}_{bg} (%)"
    assert _format_gle_legend_label("A_bg") == "{\\it{A}}_{bg}"


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.line_calls: list[dict[str, object]] = []
        self.xscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.yscale_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.ylabel_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def errorbar_from_file(self, *args, **kwargs) -> None:
        self.calls.append({"args": args, "kwargs": kwargs})

    def line_from_file(self, *args, **kwargs) -> None:
        self.line_calls.append({"args": args, "kwargs": kwargs})

    def set_xlabel(self, *_args, **_kwargs) -> None:
        return

    def set_ylabel(self, *_args, **_kwargs) -> None:
        self.ylabel_calls.append((_args, _kwargs))
        return

    def set_xscale(self, *_args, **_kwargs) -> None:
        self.xscale_calls.append((_args, _kwargs))
        return

    def set_yscale(self, *_args, **_kwargs) -> None:
        self.yscale_calls.append((_args, _kwargs))
        return

    def legend(self, *_args, **_kwargs) -> None:
        return


class _FakeFigure:
    def __init__(self, axis: _FakeAxis) -> None:
        self._axis = axis
        self.saved_paths: list[str] = []

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str) -> None:
        self.saved_paths.append(path)
        Path(path).write_text("! fake gle", encoding="utf-8")


def test_generate_gle_plot_uses_errorbar_from_file(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot.gle"

    axis = _FakeAxis()
    fig = _FakeFigure(axis)

    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: "gle")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert axis.calls, "Expected at least one errorbar_from_file call"
    first = axis.calls[0]
    assert first["args"][0] == data_path.name
    assert first["kwargs"]["x_col"] == 2
    assert first["kwargs"]["y_col"] == 4
    assert first["kwargs"]["yerr_col"] == 5
    assert str(gle_path) in fig.saved_paths


def test_generate_gle_plot_warns_for_old_gleplot(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot.gle"

    warnings: list[str] = []
    fake_glp = SimpleNamespace(Axes=type("OldAxes", (), {}))

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **_kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert warnings
    assert "gleplot" in warnings[0]


def test_effective_x_key_mapping(panel: FitParametersPanel) -> None:
    panel._x_combo.setCurrentText("𝐵 (G)")
    assert panel._effective_x_key() == "field"

    panel._x_combo.setCurrentText("𝑇 (K)")
    assert panel._effective_x_key() == "temperature"

    panel._x_combo.setCurrentText("Run")
    assert panel._effective_x_key() == "run"

    panel._x_combo.setCurrentText("Auto")
    panel._inferred_x_key = "temperature"
    assert panel._effective_x_key() == "temperature"


def test_generate_gle_plot_subplots_mode_uses_black_series(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot_subplots.gle"

    ax1 = _FakeAxis()
    ax2 = _FakeAxis()
    fig = _FakeFigure(ax1)

    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        subplots=lambda **_kwargs: (fig, [ax1, ax2]),
        figure=lambda **_kwargs: fig,
    )

    panel._plot_mode_combo.setCurrentText("Subplots")
    panel._log_x_check.setChecked(True)
    panel._log_y_check.setChecked(True)
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(QMessageBox, "exec", lambda _self: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert ax1.calls and ax2.calls
    assert ax1.calls[0]["kwargs"]["color"] == "black"
    assert ax2.calls[0]["kwargs"]["color"] == "black"
    assert ax1.xscale_calls and ax1.yscale_calls
    assert ax2.xscale_calls and ax2.yscale_calls


def test_generate_gle_plot_dual_axis_assigns_y_and_y2(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    gle_path = tmp_path / "plot_dual.gle"

    axis = _FakeAxis()
    fig = _FakeFigure(axis)
    fake_glp = SimpleNamespace(
        Axes=type("FakeAxes", (), {"errorbar_from_file": staticmethod(lambda *a, **k: None)}),
        figure=lambda **_kwargs: fig,
    )

    panel._plot_mode_combo.setCurrentText("Single Axes")
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["A0", "Lambda"])

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr("shutil.which", lambda _name: "gle")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(panel, "_show_gle_preview", lambda *_a, **_k: None)

    panel._generate_gle_plot(gle_path, data_path, "pdf")

    assert len(axis.calls) == 2
    assert axis.calls[0]["kwargs"]["x_col"] == 2
    assert axis.calls[1]["kwargs"]["x_col"] == 2
    assert axis.calls[0]["kwargs"]["yaxis"] == "y"
    assert axis.calls[1]["kwargs"]["yaxis"] == "y2"


def test_write_fit_files_restored_fit_without_bounds(panel: FitParametersPanel, tmp_path: Path) -> None:
    """Loaded fits with missing range bounds should still export .fit curve files."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )

    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=None,
                    x_max=None,
                    model=model,
                    parameters=params,
                    result=result,
                )
            ],
        )
    }

    fit_map = panel._write_fit_files(tmp_path / "fit_parameters.gle", "field", ["A0"])
    assert fit_map

    out_path = next(iter(fit_map.values()))
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert not any("line_style" in ln for ln in lines)
    assert not any("line_color" in ln for ln in lines)
    assert not any("columns:" in ln for ln in lines)
    data_lines = [ln for ln in lines if ln and not ln.startswith("!")]
    assert data_lines, "Expected numeric x,y rows in exported .fit file"


def test_export_gle_writes_fit_files_for_active_unselected_fit_param(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export should still emit .fit sidecars for active fits not in current Y selection."""
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )
    panel._model_fits = {
        "A0": ParameterModelFit(
            parameter_name="A0",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(
                    x_min=100.0,
                    x_max=200.0,
                    model=model,
                    parameters=params,
                    result=result,
                )
            ],
        )
    }

    out_gle = tmp_path / "fit_parameters.gle"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out_gle), "GLE files (*.gle)"),
    )

    # Simulate user selecting a different parameter than the one with active fit.
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["Lambda"])

    captured: dict[str, object] = {}

    def _fake_generate(gle_path, data_path, output_format, fit_file_map=None):
        captured["gle_path"] = gle_path
        captured["data_path"] = data_path
        captured["output_format"] = output_format
        captured["fit_file_map"] = fit_file_map or {}

    monkeypatch.setattr(panel, "_generate_gle_plot", _fake_generate)

    panel._export_gle()

    fit_map = captured.get("fit_file_map")
    assert isinstance(fit_map, dict)
    assert fit_map, "Expected active fit sidecar map to be passed to GLE generator"
    fit_file = next(iter(fit_map.values()))
    assert Path(fit_file).exists()


def test_add_gle_model_overlay_uses_formatted_labels_without_hash_one(panel: FitParametersPanel, tmp_path: Path) -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", value=0.21)])
    result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"c": 0.01},
        message="Fit successful",
    )
    panel._model_fits = {
        "A_bg": ParameterModelFit(
            parameter_name="A_bg",
            x_key="field",
            active=True,
            ranges=[
                ModelFitRange(x_min=100.0, x_max=150.0, model=model, parameters=params, result=result),
                ModelFitRange(x_min=150.0, x_max=200.0, model=model, parameters=params, result=result),
            ],
        )
    }
    fit_map = {
        ("A_bg", 0): tmp_path / "a.fit",
        ("A_bg", 1): tmp_path / "b.fit",
    }
    axis = _FakeAxis()

    panel._add_gle_model_overlay(axis, "A_bg", color="red", include_labels=True, fit_file_map=fit_map)

    assert len(axis.line_calls) == 2
    assert axis.line_calls[0]["kwargs"]["label"] == "fit {\\it{A}}_{bg}"
    assert axis.line_calls[1]["kwargs"]["label"] == "fit {\\it{A}}_{bg} #2"
    assert axis.line_calls[0]["kwargs"]["color"] != axis.line_calls[1]["kwargs"]["color"]
