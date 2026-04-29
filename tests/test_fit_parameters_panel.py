"""Tests for FitParametersPanel GLE export helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # type: ignore
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    ParameterModelFitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.export_paths import resolve_gle_export_paths
from asymmetry.gui.panels.fit_parameters_panel import (
    FitParametersPanel,
    _FitRow,
    _format_gle_label,
    _format_gle_legend_label,
    _format_plot_label,
    _format_plot_legend_label,
    _GroupFitData,
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


def test_write_gle_data_file_includes_combined_run_mapping_comments(
    tmp_path: Path, qapp: QApplication
) -> None:
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


def test_delete_group_fits_removes_group_and_emits_run_numbers(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = FitParametersPanel()
    rows = [
        _FitRow(
            run_number=101,
            run_label="101",
            field=100.0,
            temperature=5.0,
            values={"A0": 0.2},
            errors={"A0": 0.01},
        ),
        _FitRow(
            run_number=102,
            run_label="102",
            field=120.0,
            temperature=5.0,
            values={"A0": 0.19},
            errors={"A0": 0.01},
        ),
    ]
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Group 1",
            rows=rows,
            global_params=None,
            varying_params=["A0"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g2": _GroupFitData(
            group_id="g2",
            group_name="Group 2",
            rows=list(rows),
            global_params=None,
            varying_params=["A0"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g1"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g1"], emit=False)

    emitted: list[tuple[str, object]] = []
    panel.delete_group_fits_requested.connect(
        lambda gid, run_numbers: emitted.append((gid, run_numbers))
    )

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_a, **_k: QMessageBox.StandardButton.Ok,
    )

    panel._delete_group_fits("g1")

    assert "g1" not in panel._group_fit_results
    assert panel._active_group_id == "g2"
    assert emitted == [("g1", [101, 102])]


def test_background_labels_use_subscript_formatting() -> None:
    assert _format_plot_label("A_bg") == "$A_{bg}$ (%)"
    assert _format_plot_legend_label("A_bg") == "$A_{bg}$"
    assert _format_gle_label("A_bg") == "{\\it A}_{bg} (%)"
    assert _format_gle_legend_label("A_bg") == "{\\it A}_{bg}"


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
        self.saved_kwargs: list[dict[str, object]] = []

    def add_subplot(self, *_args, **_kwargs) -> _FakeAxis:
        return self._axis

    def savefig(self, path: str, **kwargs) -> None:
        self.saved_kwargs.append(dict(kwargs))
        output_path = Path(path)
        if kwargs.get("folder"):
            output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(str(output_path))
        output_path.write_text("! fake gle", encoding="utf-8")


def test_generate_gle_plot_uses_errorbar_from_file(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    requested_gle_path = tmp_path / "plot.gle"
    gle_path, _ = resolve_gle_export_paths(requested_gle_path, folder=True)

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

    panel._generate_gle_plot(requested_gle_path, gle_path, data_path, "pdf")

    assert axis.calls, "Expected at least one errorbar_from_file call"
    first = axis.calls[0]
    assert first["args"][0] == data_path.name
    assert first["kwargs"]["x_col"] == 2
    assert first["kwargs"]["y_col"] == 4
    assert first["kwargs"]["yerr_col"] == 5
    assert str(gle_path) in fig.saved_paths
    assert fig.saved_kwargs[-1]["folder"] is True


def test_generate_gle_plot_warns_for_old_gleplot(
    panel: FitParametersPanel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path = tmp_path / "fit_parameters.dat"
    data_path.write_text("1 2 3\n", encoding="utf-8")
    requested_gle_path = tmp_path / "plot.gle"
    gle_path, _ = resolve_gle_export_paths(requested_gle_path, folder=True)

    warnings: list[str] = []
    fake_glp = SimpleNamespace(Axes=type("OldAxes", (), {}))

    monkeypatch.setitem(sys.modules, "gleplot", fake_glp)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **_kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )

    panel._generate_gle_plot(requested_gle_path, gle_path, data_path, "pdf")

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
    requested_gle_path = tmp_path / "plot_subplots.gle"
    gle_path, _ = resolve_gle_export_paths(requested_gle_path, folder=True)

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

    panel._generate_gle_plot(requested_gle_path, gle_path, data_path, "pdf")

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
    requested_gle_path = tmp_path / "plot_dual.gle"
    gle_path, _ = resolve_gle_export_paths(requested_gle_path, folder=True)

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

    panel._generate_gle_plot(requested_gle_path, gle_path, data_path, "pdf")

    assert len(axis.calls) == 2
    assert axis.calls[0]["kwargs"]["x_col"] == 2
    assert axis.calls[1]["kwargs"]["x_col"] == 2
    assert axis.calls[0]["kwargs"]["yaxis"] == "y"
    assert axis.calls[1]["kwargs"]["yaxis"] == "y2"


def test_write_fit_files_restored_fit_without_bounds(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
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
    resolved_gle, _ = resolve_gle_export_paths(out_gle, folder=True)
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out_gle), "GLE files (*.gle)"),
    )

    # Simulate user selecting a different parameter than the one with active fit.
    monkeypatch.setattr(panel, "_selected_y_parameters", lambda: ["Lambda"])

    captured: dict[str, object] = {}

    def _fake_generate(requested_gle_path, gle_path, data_path, output_format, fit_file_map=None):
        captured["requested_gle_path"] = requested_gle_path
        captured["gle_path"] = gle_path
        captured["data_path"] = data_path
        captured["output_format"] = output_format
        captured["fit_file_map"] = fit_file_map or {}

    monkeypatch.setattr(panel, "_generate_gle_plot", _fake_generate)

    panel._export_gle()

    assert captured["requested_gle_path"] == out_gle
    assert captured["gle_path"] == resolved_gle
    assert Path(captured["data_path"]).parent == resolved_gle.parent
    fit_map = captured.get("fit_file_map")
    assert isinstance(fit_map, dict)
    assert fit_map, "Expected active fit sidecar map to be passed to GLE generator"
    fit_file = next(iter(fit_map.values()))
    assert Path(fit_file).exists()
    assert Path(fit_file).parent == resolved_gle.parent


def test_add_gle_model_overlay_uses_formatted_labels_without_hash_one(
    panel: FitParametersPanel, tmp_path: Path
) -> None:
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
                ModelFitRange(
                    x_min=100.0, x_max=150.0, model=model, parameters=params, result=result
                ),
                ModelFitRange(
                    x_min=150.0, x_max=200.0, model=model, parameters=params, result=result
                ),
            ],
        )
    }
    fit_map = {
        ("A_bg", 0): tmp_path / "a.fit",
        ("A_bg", 1): tmp_path / "b.fit",
    }
    axis = _FakeAxis()

    panel._add_gle_model_overlay(
        axis, "A_bg", color="red", include_labels=True, fit_file_map=fit_map
    )

    assert len(axis.line_calls) == 2
    assert axis.line_calls[0]["kwargs"]["label"] == "fit {\\it A}_{bg}"
    assert axis.line_calls[1]["kwargs"]["label"] == "fit {\\it A}_{bg} #2"
    assert axis.line_calls[0]["kwargs"]["color"] != axis.line_calls[1]["kwargs"]["color"]


def test_build_inherited_cross_group_config_uses_best_successful_fit(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["Linear"])

    best_params = ParameterSet(
        [
            Parameter("m", value=0.002, min=0.0, max=1.0, fixed=True),
            Parameter("b", value=0.21, min=-1.0, max=1.0, fixed=False),
        ]
    )
    best_result = ParameterModelFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.4,
        parameters=best_params,
        uncertainties={"b": 0.01},
        message="ok",
    )
    best_range = ModelFitRange(
        x_min=100.0,
        x_max=300.0,
        model=model,
        parameters=best_params,
        result=best_result,
    )

    other_params = ParameterSet(
        [
            Parameter("m", value=0.01, min=0.0, max=1.0, fixed=False),
            Parameter("b", value=0.5, min=-1.0, max=1.0, fixed=False),
        ]
    )
    other_result = ParameterModelFitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=2.0,
        parameters=other_params,
        uncertainties={"m": 0.1, "b": 0.1},
        message="ok",
    )
    other_range = ModelFitRange(
        x_min=90.0,
        x_max=310.0,
        model=model,
        parameters=other_params,
        result=other_result,
    )

    best_group = _GroupFitData(
        group_id="g_best",
        group_name="Best",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[best_range],
                active=True,
            )
        },
        plot_annotations=[],
    )
    other_group = _GroupFitData(
        group_id="g_other",
        group_name="Other",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[other_range],
                active=True,
            )
        },
        plot_annotations=[],
    )

    config = panel._build_inherited_cross_group_config(
        "Lambda",
        "field",
        [other_group, best_group],
    )

    assert config is not None
    assert config["source_group_id"] == "g_best"
    assert config["source_group_name"] == "Best"
    assert config["fit_x_min"] == pytest.approx(100.0)
    assert config["fit_x_max"] == pytest.approx(300.0)

    rows = config.get("parameter_rows")
    assert isinstance(rows, list)
    row_by_name = {str(row["name"]): row for row in rows if isinstance(row, dict)}
    assert row_by_name["m"]["initial"] == pytest.approx(0.002)
    assert row_by_name["m"]["type"] == "Fixed"
    assert row_by_name["b"]["initial"] == pytest.approx(0.21)
    assert row_by_name["b"]["type"] == "Global"


def test_build_inherited_cross_group_config_returns_none_without_success(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["Linear"])
    failed_params = ParameterSet([Parameter("m", value=0.1), Parameter("b", value=0.2)])
    failed_range = ModelFitRange(
        x_min=0.0,
        x_max=1.0,
        model=model,
        parameters=failed_params,
        result=ParameterModelFitResult(success=False, message="failed"),
    )
    failed_group = _GroupFitData(
        group_id="g0",
        group_name="G0",
        rows=[],
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={
            "Lambda": ParameterModelFit(
                parameter_name="Lambda",
                x_key="field",
                ranges=[failed_range],
                active=True,
            )
        },
        plot_annotations=[],
    )

    config = panel._build_inherited_cross_group_config(
        "Lambda",
        "field",
        [failed_group],
    )
    assert config is None


def test_group_switch_persists_active_group_model_fits(panel: FitParametersPanel) -> None:
    fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)

    group_a = _GroupFitData(
        group_id="g_a",
        group_name="Group A",
        rows=list(panel._rows),
        global_params=None,
        varying_params=list(panel._varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    group_b = _GroupFitData(
        group_id="g_b",
        group_name="Group B",
        rows=list(panel._rows),
        global_params=None,
        varying_params=list(panel._varying_params),
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )

    panel._group_fit_results = {"g_a": group_a, "g_b": group_b}
    panel._active_group_id = "g_a"
    panel._model_fits = {"Lambda": fit}
    panel._rebuild_group_buttons()

    panel._set_selected_group_ids(["g_b"], emit=False)
    panel._apply_group_selection_to_view()

    assert "Lambda" in panel._group_fit_results["g_a"].model_fits


def test_get_state_syncs_active_group_before_serializing(panel: FitParametersPanel) -> None:
    fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._active_group_id = "g_a"
    panel._model_fits = {"Lambda": fit}

    state = panel.get_state()
    groups_state = state.get("group_fit_results", {})
    assert isinstance(groups_state, dict)
    g_a_state = groups_state.get("g_a", {})
    assert isinstance(g_a_state, dict)
    model_fits_state = g_a_state.get("model_fits", {})
    assert isinstance(model_fits_state, dict)
    assert "Lambda" in model_fits_state


def test_multi_group_view_excludes_asymmetry_global_parameters(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"A0": 0.30, "Lambda": 0.15},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]

    global_params = ParameterSet([Parameter("A0", value=0.25)])

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=global_params,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=global_params,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)

    panel._apply_group_selection_to_view(sync_active=False)

    assert "Lambda" in panel._varying_params
    assert "A0" not in panel._varying_params


def test_group_global_params_roundtrip_in_panel_state(panel: FitParametersPanel) -> None:
    global_params = ParameterSet([Parameter("A0", value=0.25, min=0.0, max=1.0, fixed=False)])
    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=global_params,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        )
    }
    panel._active_group_id = "g_a"
    panel._global_params = global_params

    state = panel.get_state()

    restored = FitParametersPanel()
    restored.restore_state(state)

    assert "g_a" in restored._group_fit_results
    restored_global = restored._group_fit_results["g_a"].global_params
    assert restored_global is not None
    assert "A0" in restored_global
    assert restored_global["A0"].value == pytest.approx(0.25)


def test_new_asymmetry_fit_overwrites_existing_group_model_fits(panel: FitParametersPanel) -> None:
    existing_fit = ParameterModelFit(parameter_name="Lambda", x_key="field", ranges=[], active=True)
    panel._group_fit_results = {
        "g1": _GroupFitData(
            group_id="g1",
            group_name="Group 1",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={"Lambda": existing_fit},
            plot_annotations=[{"parameter": "Lambda"}],
        )
    }
    panel._active_group_id = "g1"
    panel._model_fits = {"Lambda": existing_fit}
    panel._plot_annotations = [{"parameter": "Lambda"}]

    result = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.24), Parameter("Lambda", value=0.11)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_by_run = {
        101: SimpleNamespace(
            metadata={
                "run_label": "101",
                "field": 150.0,
                "temperature": 12.0,
            }
        )
    }

    panel.set_fit_results(
        {101: (result, (np.array([0.0, 1.0]), np.array([0.24, 0.24])))},
        datasets_by_run,
        global_params=None,
        group_id="g1",
        group_name="Group 1",
    )

    assert panel._group_fit_results["g1"].model_fits == {}
    assert panel._group_fit_results["g1"].plot_annotations == []
    assert panel._model_fits == {}
    assert panel._plot_annotations == []


def test_multi_group_view_does_not_sync_merged_rows_back_into_active_group(
    panel: FitParametersPanel,
) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"A0": 0.20, "Lambda": 0.10},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=11.0,
            values={"A0": 0.30, "Lambda": 0.15},
            errors={"A0": 0.01, "Lambda": 0.01},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)

    panel._apply_group_selection_to_view(sync_active=False)
    assert panel._active_group_id == "g_a"
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 1

    # Regression guard: merged rows must never be written into Group A.
    panel._sync_active_group_state()
    assert [row.run_number for row in panel._group_fit_results["g_a"].rows] == [1]


def test_group_button_styles_distinguish_single_active_from_multi_highlight(
    panel: FitParametersPanel,
) -> None:
    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=list(panel._rows),
            global_params=None,
            varying_params=list(panel._varying_params),
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._rebuild_group_buttons()

    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    assert "2px solid #1f6feb" in panel._group_button_map["g_a"].styleSheet()

    panel._set_selected_group_ids(["g_a", "g_b"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)
    assert "2px solid #1f6feb" in panel._group_button_map["g_a"].styleSheet()
    assert "1px solid #d29922" not in panel._group_button_map["g_a"].styleSheet()
    assert "1px solid #d29922" in panel._group_button_map["g_b"].styleSheet()


def test_shift_click_highlights_group_without_changing_active_selection(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"Lambda": 0.10},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"Lambda": 0.14},
            errors={"Lambda": 0.01},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="Group A",
            rows=rows_a,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="Group B",
            rows=rows_b,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QApplication.keyboardModifiers",
        lambda: Qt.KeyboardModifier.ShiftModifier,
    )

    panel._group_button_map["g_b"].click()

    assert panel._active_group_id == "g_a"
    assert panel._group_button_map["g_a"].isChecked()
    assert panel._group_button_map["g_b"].isChecked()
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 1


def test_cross_group_fit_success_updates_each_group_model_fit(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=100.0,
            temperature=10.0,
            values={"Lambda": 0.10},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=2,
            run_label="2",
            field=200.0,
            temperature=10.0,
            values={"Lambda": 0.14},
            errors={"Lambda": 0.01},
        )
    ]
    group_a = _GroupFitData(
        group_id="g_a",
        group_name="Group A",
        rows=rows_a,
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    group_b = _GroupFitData(
        group_id="g_b",
        group_name="Group B",
        rows=rows_b,
        global_params=None,
        varying_params=["Lambda"],
        inferred_x_key="field",
        model_fits={},
        plot_annotations=[],
    )
    panel._group_fit_results = {"g_a": group_a, "g_b": group_b}

    model = ParameterCompositeModel(["Linear"])
    global_params = ParameterSet([Parameter("m", value=0.002, min=0.0, max=1.0, fixed=False)])
    local_params = {
        "g_a": ParameterSet([Parameter("b", value=0.18, min=-1.0, max=1.0, fixed=False)]),
        "g_b": ParameterSet([Parameter("b", value=0.23, min=-1.0, max=1.0, fixed=False)]),
    }
    fixed_params = ParameterSet()
    fit_result = CrossGroupFitResult(
        success=True,
        chi_squared=1.2,
        reduced_chi_squared=0.6,
        global_parameters=global_params,
        local_parameters=local_params,
        fixed_parameters=fixed_params,
        global_uncertainties={"m": 1e-4},
        local_uncertainties={"g_a": {"b": 0.01}, "g_b": {"b": 0.02}},
        message="Fit successful",
    )

    output = SimpleNamespace(
        fit_result=fit_result,
        model=model,
        x_key="field",
        fit_x_min=90.0,
        fit_x_max=250.0,
        groups=[
            ParameterGroupData(
                "g_a", "Group A", np.array([100.0]), np.array([0.1]), np.array([0.01]), 100.0
            ),
            ParameterGroupData(
                "g_b", "Group B", np.array([200.0]), np.array([0.14]), np.array([0.01]), 200.0
            ),
        ],
    )

    panel._apply_cross_group_fit_to_groups(
        parameter_name="Lambda",
        x_key="field",
        selected_groups=[group_a, group_b],
        output=output,
    )

    fit_a = panel._group_fit_results["g_a"].model_fits.get("Lambda")
    fit_b = panel._group_fit_results["g_b"].model_fits.get("Lambda")
    assert fit_a is not None
    assert fit_b is not None
    assert fit_a.ranges[0].result is not None
    assert fit_b.ranges[0].result is not None
    assert fit_a.ranges[0].result.parameters["b"].value == pytest.approx(0.18)
    assert fit_b.ranges[0].result.parameters["b"].value == pytest.approx(0.23)
    assert fit_a.ranges[0].result.parameters["m"].value == pytest.approx(0.002)
    assert fit_b.ranges[0].result.parameters["m"].value == pytest.approx(0.002)


def test_build_cross_group_group_model_fit_defaults_shape_factor_a_to_fixed(
    panel: FitParametersPanel,
) -> None:
    model = ParameterCompositeModel(["SC_PWaveAxial"])
    fit_result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=0.5,
        global_parameters=ParameterSet([Parameter("sigma_0", value=1.0)]),
        local_parameters={"g1": ParameterSet([Parameter("Tc", value=20.0)])},
        fixed_parameters=ParameterSet(
            [
                Parameter("gap_ratio", value=2.0, fixed=True),
                Parameter("sigma_bg", value=0.0, fixed=True),
            ]
        ),
    )

    fit = panel._build_cross_group_group_model_fit(
        parameter_name="Lambda",
        x_key="temperature",
        group_id="g1",
        model=model,
        fit_result=fit_result,
        fit_x_min=1.0,
        fit_x_max=30.0,
    )

    shape_factor = fit.ranges[0].parameters["shape_factor_a"]
    assert shape_factor.value == 0.0
    assert shape_factor.fixed is True


def test_set_fit_results_copies_global_params_per_group(panel: FitParametersPanel) -> None:
    shared_global = ParameterSet([Parameter("A0", value=0.25, min=0.0, max=1.0, fixed=False)])

    result_g1 = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.24), Parameter("Lambda", value=0.11)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_g1 = {
        101: SimpleNamespace(metadata={"run_label": "101", "field": 150.0, "temperature": 12.0})
    }
    panel.set_fit_results(
        {101: (result_g1, (np.array([0.0, 1.0]), np.array([0.24, 0.24])))},
        datasets_g1,
        global_params=shared_global,
        group_id="g1",
        group_name="Group 1",
    )

    # Simulate reuse/mutation of the same ParameterSet object by later fits.
    shared_global["A0"].value = 0.42

    result_g2 = FitResult(
        success=True,
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A0", value=0.30), Parameter("Lambda", value=0.12)]),
        uncertainties={"A0": 0.01, "Lambda": 0.01},
    )
    datasets_g2 = {
        202: SimpleNamespace(metadata={"run_label": "202", "field": 300.0, "temperature": 12.0})
    }
    panel.set_fit_results(
        {202: (result_g2, (np.array([0.0, 1.0]), np.array([0.30, 0.30])))},
        datasets_g2,
        global_params=shared_global,
        group_id="g2",
        group_name="Group 2",
    )

    g1_params = panel._group_fit_results["g1"].global_params
    g2_params = panel._group_fit_results["g2"].global_params
    assert g1_params is not None
    assert g2_params is not None
    assert g1_params["A0"].value == pytest.approx(0.25)
    assert g2_params["A0"].value == pytest.approx(0.42)


def test_click_group_switch_does_not_copy_previous_group_rows(
    panel: FitParametersPanel, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_a = [
        _FitRow(
            run_number=11,
            run_label="11",
            field=101.0,
            temperature=0.1,
            values={"Lambda": 0.11},
            errors={"Lambda": 0.01},
        )
    ]
    rows_b = [
        _FitRow(
            run_number=22,
            run_label="22",
            field=202.0,
            temperature=2.0,
            values={"Lambda": 0.22},
            errors={"Lambda": 0.02},
        )
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="T = 0.1 K",
            rows=rows_a,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="T = 2 K",
            rows=rows_b,
            global_params=None,
            varying_params=["Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rows = list(rows_a)
    panel._varying_params = ["Lambda"]
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)

    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QApplication.keyboardModifiers",
        lambda: Qt.KeyboardModifier.NoModifier,
    )

    panel._group_button_map["g_b"].click()

    # Active view now shows group B's own rows.
    assert panel._active_group_id == "g_b"
    assert len(panel._rows) == 1
    assert panel._rows[0].run_number == 22

    # Group B storage remains intact and was not overwritten by group A rows.
    assert len(panel._group_fit_results["g_b"].rows) == 1
    assert panel._group_fit_results["g_b"].rows[0].run_number == 22


def test_group_switch_keeps_selected_y_parameter_when_available(panel: FitParametersPanel) -> None:
    rows_a = [
        _FitRow(
            run_number=11,
            run_label="11",
            field=101.0,
            temperature=0.1,
            values={"Lambda": 0.11, "A0": 0.21},
            errors={"Lambda": 0.01, "A0": 0.01},
        ),
        _FitRow(
            run_number=12,
            run_label="12",
            field=102.0,
            temperature=0.1,
            values={"Lambda": 0.12, "A0": 0.22},
            errors={"Lambda": 0.01, "A0": 0.01},
        ),
    ]
    rows_b = [
        _FitRow(
            run_number=21,
            run_label="21",
            field=201.0,
            temperature=2.0,
            values={"Lambda": 0.21, "A0": 0.31},
            errors={"Lambda": 0.02, "A0": 0.02},
        ),
        _FitRow(
            run_number=22,
            run_label="22",
            field=202.0,
            temperature=2.0,
            values={"Lambda": 0.22, "A0": 0.32},
            errors={"Lambda": 0.02, "A0": 0.02},
        ),
    ]

    panel._group_fit_results = {
        "g_a": _GroupFitData(
            group_id="g_a",
            group_name="T = 0.1 K",
            rows=rows_a,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
        "g_b": _GroupFitData(
            group_id="g_b",
            group_name="T = 2 K",
            rows=rows_b,
            global_params=None,
            varying_params=["A0", "Lambda"],
            inferred_x_key="field",
            model_fits={},
            plot_annotations=[],
        ),
    }
    panel._active_group_id = "g_a"
    panel._rebuild_group_buttons()
    panel._set_selected_group_ids(["g_a"], emit=False)
    panel._apply_group_selection_to_view(sync_active=False)

    # Select Lambda in group A.
    lambda_row = panel._varying_params.index("Lambda")
    lambda_item = panel._y_selector_table.item(lambda_row, 0)
    assert lambda_item is not None
    panel._y_selector_table.clearSelection()
    lambda_item.setSelected(True)
    assert panel._selected_y_parameters() == ["Lambda"]

    # Switch to group B; Lambda exists there too and should stay selected.
    panel._set_selected_group_ids(["g_b"], emit=False)
    panel._active_group_id = "g_b"
    panel._apply_group_selection_to_view()
    assert panel._selected_y_parameters() == ["Lambda"]


def test_group_variable_value_for_rows_uses_complementary_axis(panel: FitParametersPanel) -> None:
    rows = [
        _FitRow(
            run_number=1,
            run_label="1",
            field=20.0,
            temperature=20.10,
            values={"Lambda": 0.1},
            errors={"Lambda": 0.01},
        ),
        _FitRow(
            run_number=2,
            run_label="2",
            field=20.0,
            temperature=20.13,
            values={"Lambda": 0.1},
            errors={"Lambda": 0.01},
        ),
    ]

    gv_field_fit = panel._group_variable_value_for_rows(rows, "field")
    gv_temp_fit = panel._group_variable_value_for_rows(rows, "temperature")

    # For field-based fits, group variable should be temperature.
    assert gv_field_fit == pytest.approx((20.10 + 20.13) / 2.0)
    # For temperature-based fits, group variable should be field.
    assert gv_temp_fit == pytest.approx(20.0)
