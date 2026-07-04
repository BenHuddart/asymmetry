"""Tests for GlobalParameterFitWindow robustness on restored states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # type: ignore

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
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from tests._qt_helpers import wait_for


def _wait_fit_curves(window: GlobalParameterFitWindow, timeout_s: float = 10.0) -> None:
    """Block until the off-thread cross-group fit-curve recompute lands."""
    wait_for(
        lambda: not window._fit_curve_compute_active,
        QApplication.instance(),
        timeout_s=timeout_s,
    )


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_refresh_plot_tolerates_missing_model_parameter_in_restored_result(
    qapp: QApplication,
) -> None:
    window = GlobalParameterFitWindow()

    # DiffusionLF_2D requires D_perp among others; omit it from result params to
    # emulate older saved projects that predate this parameter.
    model = ParameterCompositeModel(["DiffusionLF_2D"])

    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0, 300.0], dtype=float),
            y=np.array([0.15, 0.12, 0.10], dtype=float),
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]

    local_params = ParameterSet(
        [
            Parameter("A", value=0.20),
            Parameter("D_2D", value=0.04),
            # Intentionally missing: D_perp
        ]
    )

    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet(),
        local_parameters={"g0": local_params},
        fixed_parameters=ParameterSet(),
    )

    # Should not raise even though the restored result omits a now-required
    # model parameter — the off-thread compute fills model defaults.
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )
    _wait_fit_curves(window)
    assert not window._fit_curve_compute_active


def test_set_results_computes_fit_curves_off_thread_behind_overlay(
    qapp: QApplication,
) -> None:
    """The cross-group fit curves are recomputed off-thread under the overlay."""
    window = GlobalParameterFitWindow()
    model = ParameterCompositeModel(["Linear"])
    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0, 30.0], dtype=float),
            y=np.array([0.2, 0.15, 0.1], dtype=float),
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=0.1,
        )
    ]
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("m", value=-0.005), Parameter("b", value=0.25)]),
        local_parameters={},
        fixed_parameters=ParameterSet(),
    )

    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )
    # Mid-flight: the overlay covers the plot and the compute is active.
    assert window._fit_curve_compute_active
    assert window._fit_overlay is not None and not window._fit_overlay.isHidden()

    _wait_fit_curves(window)

    # Landed: overlay cleared, the curve cache warm for the current flag, and
    # the red model curve drawn on the fit plot.
    assert not window._fit_curve_compute_active
    assert window._fit_overlay.isHidden()
    assert window._curve_cache.get(False) is not None
    assert window._left_figure is not None and window._left_figure.axes
    red_lines = [
        line
        for ax in window._left_figure.axes
        for line in ax.get_lines()
        if line.get_color() == "red"
    ]
    assert red_lines, "the cross-group fit curve should be drawn after the compute"


def test_close_with_pending_fit_curve_compute_shuts_down(qapp: QApplication) -> None:
    """Closing mid-recompute tears the worker down within the bounded wait."""
    window = GlobalParameterFitWindow()
    model = ParameterCompositeModel(["Linear"])
    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0, 30.0], dtype=float),
            y=np.array([0.2, 0.15, 0.1], dtype=float),
            yerr=np.array([0.01, 0.01, 0.01], dtype=float),
            group_variable_value=0.1,
        )
    ]
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("m", value=1.0), Parameter("b", value=0.0)]),
        local_parameters={},
        fixed_parameters=ParameterSet(),
    )
    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )
    assert window._fit_curve_compute_active

    window.close()
    assert window._tasks.active_count == 0


def test_local_parameter_plot_uses_complementary_group_axis_label(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    model = ParameterCompositeModel(["Linear"])

    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="T=0.1K",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=0.1,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="T=20K",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]

    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("b", value=0.1)]),
            "g1": ParameterSet([Parameter("b", value=0.2)]),
        },
        local_uncertainties={
            "g0": {"b": 0.01},
            "g1": {"b": 0.01},
        },
    )

    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )

    assert window._local_figure is not None
    assert window._local_figure.axes
    assert window._local_figure.axes[-1].get_xlabel() == "$T$ (K)"


def test_sample_group_fit_curve_uses_log_spacing_for_complementary_field_axis(
    qapp: QApplication,
) -> None:
    window = GlobalParameterFitWindow()
    model = ParameterCompositeModel(["Constant"])

    group = ParameterGroupData(
        group_id="g0",
        group_name="T=5K",
        x=np.array([100.0, 1000.0, 10000.0], dtype=float),
        y=np.array([0.2, 0.2, 0.2], dtype=float),
        yerr=np.array([0.01, 0.01, 0.01], dtype=float),
        group_variable_value=5.0,
    )

    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("c", value=0.2)]),
        local_parameters={},
        fixed_parameters=ParameterSet(),
    )

    window.set_results(
        parameter_name="Lambda",
        x_key="temperature",
        groups=[group],
        model=model,
        result=result,
    )

    sampled = window._sample_group_fit_curve(group)

    assert sampled is not None
    xs, ys = sampled
    np.testing.assert_allclose(xs[[0, -1]], [100.0, 10000.0])
    np.testing.assert_allclose(ys, np.full(xs.shape, 0.2))
    ratios = xs[1:] / xs[:-1]
    np.testing.assert_allclose(ratios, np.full_like(ratios, ratios[0]))


def test_local_parameter_subplots_do_not_show_titles(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._local_plot_mode_combo.setCurrentText("Subplots")
    model = ParameterCompositeModel(["Linear"])

    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=0.1,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]

    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1), Parameter("nu", value=1.0)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2), Parameter("nu", value=2.0)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01, "nu": 0.1},
            "g1": {"Lambda": 0.01, "nu": 0.1},
        },
    )

    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )

    assert window._local_figure is not None
    assert window._local_figure.axes
    assert all(ax.get_title() == "" for ax in window._local_figure.axes)


def test_parameter_label_uses_mathtext_symbols_and_units(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    assert window._parameter_label("Lambda") == r"$\lambda$ (µs⁻¹)"
    assert window._parameter_label("nu") == r"$\nu$ (MHz)"
    assert window._parameter_label("A0") == r"$A_0$ (%)"


def test_export_actions_use_separate_defaults(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    calls: list[dict[str, object]] = []

    def _capture(**kwargs):
        calls.append(kwargs)

    window._export_plot_gle = _capture  # type: ignore[method-assign]

    window._export_fit_subplot_gle()
    window._export_local_parameters_gle()

    assert len(calls) == 2
    assert calls[0]["default_name"] == "global_parameter_fit_subplots.gleplot"
    assert calls[1]["default_name"] == "global_parameter_fit_local_parameters.gleplot"
    assert calls[0]["output_format"] == "pdf"
    assert calls[1]["output_format"] == "pdf"


def test_export_button_labels_match_intent(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    assert window._export_btn.text() == "Export fits to GLE"
    assert window._export_local_btn.text() == "Export plot(s) to GLE"
    assert not window._fit_share_x_check.isChecked()


def test_show_components_disables_log_y(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._fit_log_y_check.setChecked(True)
    assert window._fit_log_y_check.isChecked()

    window._show_components_check.setChecked(True)

    assert not window._fit_log_y_check.isChecked()
    assert not window._fit_log_y_check.isEnabled()

    window._show_components_check.setChecked(False)
    assert window._fit_log_y_check.isEnabled()


def test_window_state_round_trip_controls_and_annotations(qapp: QApplication) -> None:
    source = GlobalParameterFitWindow()
    source._show_components_check.setChecked(False)
    source._fit_log_x_check.setChecked(True)
    source._fit_log_y_check.setChecked(True)
    source._fit_share_x_check.setChecked(True)
    source._fit_subplot_aspect_spin.setValue(2.35)
    source._local_log_x_check.setChecked(True)
    source._local_param_log_y = {"nu": True}
    source._local_selected_y_names = ["nu"]
    source._local_plot_mode_combo.setCurrentText("Subplots")
    source._local_model_fits["nu"] = ParameterModelFit(
        parameter_name="nu",
        x_key="temperature",
        ranges=[
            ModelFitRange(
                x_min=1.0,
                x_max=10.0,
                model=ParameterCompositeModel.from_expression("Linear + ( Arrhenius * Constant )"),
                parameters=ParameterSet(
                    [
                        Parameter("m", value=0.1),
                        Parameter("b", value=1.0),
                        Parameter("a", value=0.4),
                        Parameter("Ea", value=2.5),
                        Parameter("c", value=0.05),
                    ]
                ),
                result=ParameterModelFitResult(
                    success=True,
                    chi_squared=1.0,
                    reduced_chi_squared=0.5,
                    parameters=ParameterSet(
                        [
                            Parameter("m", value=0.1),
                            Parameter("b", value=1.0),
                            Parameter("a", value=0.4),
                            Parameter("Ea", value=2.5),
                            Parameter("c", value=0.05),
                        ]
                    ),
                    uncertainties={"m": 0.01, "b": 0.02, "a": 0.03, "Ea": 0.04, "c": 0.01},
                    message="Fit successful",
                ),
            )
        ],
        active=True,
    )
    source._plot_annotations = [
        {
            "x": 1.0,
            "y": 2.0,
            "text": "A",
            "axis_tag": "g0",
            "is_group_label": True,
            "artist": object(),
        }
    ]
    source._local_plot_annotations = [
        {"x": 3.0, "y": 4.0, "text": "B", "axis_tag": "main", "artist": object()}
    ]
    source._suppressed_group_label_tags = {"g2"}

    state = source.get_state()

    restored = GlobalParameterFitWindow()
    restored.restore_state(state)

    assert restored._show_components_check.isChecked() is False
    assert restored._fit_log_x_check.isChecked() is True
    assert restored._fit_log_y_check.isChecked() is True
    assert restored._fit_share_x_check.isChecked() is True
    assert restored._fit_subplot_aspect_spin.value() == pytest.approx(2.35)
    assert restored._local_log_x_check.isChecked() is True
    assert restored._local_param_log_y.get("nu") is True
    assert restored._local_selected_y_names == ["nu"]
    assert "nu" in restored._local_model_fits
    assert restored._local_plot_mode_combo.currentText() == "Subplots"
    assert restored._suppressed_group_label_tags == {"g2"}
    restored_model = restored._local_model_fits["nu"].ranges[0].model
    assert restored_model.component_names == ["Linear", "Arrhenius", "Constant"]
    assert restored_model.operators == ["+", "*"]
    assert restored_model.open_parentheses == [0, 1, 0]
    assert restored_model.close_parentheses == [0, 0, 1]
    assert len(restored._plot_annotations) == 1
    assert len(restored._local_plot_annotations) == 1
    assert restored._plot_annotations[0].get("artist") is None
    assert restored._local_plot_annotations[0].get("artist") is None


def test_export_plot_gle_saves_and_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)
    window._parameter_name = "Lambda"
    window._fit_gle_format_combo.setCurrentText("EPS")

    out_path = tmp_path / "export_test.gle"
    resolved_gle_path, _ = resolve_gle_export_paths(out_path, folder=True)
    monkeypatch.setattr(
        "asymmetry.gui.windows.global_parameter_fit_window.QFileDialog.getSaveFileName",
        lambda *_a, **_k: (str(out_path), "GLE files (*.gle)"),
    )

    compile_calls: list[tuple[Path, str]] = []
    save_kwargs: list[dict[str, object]] = []

    def _fake_compile(path: Path, fmt: str) -> None:
        compile_calls.append((path, fmt))

    window._compile_and_preview_gle = _fake_compile  # type: ignore[method-assign]

    class _FakeFigure:
        def savefig(self, path: str, **kwargs) -> None:
            save_kwargs.append(dict(kwargs))
            output_path = Path(path)
            if kwargs.get("folder"):
                output_path, export_dir = resolve_gle_export_paths(output_path, folder=True)
                export_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("! fake gle", encoding="utf-8")

    class _FakeGlp:
        @staticmethod
        def figure(**_kwargs):
            return _FakeFigure()

    monkeypatch.setattr(
        "asymmetry.gui.windows.global_parameter_fit_window.importlib.import_module",
        lambda _name: _FakeGlp(),
    )

    def _builder(glp, _gle_path):
        return glp.figure(figsize=(7.0, 4.5))

    window._export_plot_gle(
        title="Export Test",
        default_name="test.gle",
        builder=_builder,
        output_format="eps",
    )

    assert resolved_gle_path.exists()
    assert "folder" not in save_kwargs[-1]
    assert compile_calls == [(resolved_gle_path, "eps")]


def test_export_plot_gle_requires_result(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._result = None
    window._parameter_name = None

    infos: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **_kwargs: infos.append(str(args[2]) if len(args) > 2 else ""),
    )

    window._export_plot_gle(
        title="Export Test",
        default_name="test.gle",
        builder=lambda _glp, _gle_path: None,
        output_format="pdf",
    )

    assert infos
    assert "Run a cross-group fit first" in infos[0]


def test_compile_and_preview_gle_runs_in_export_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    gle_path = tmp_path / "export_bundle.gleplot" / "export_bundle.gle"
    gle_path.parent.mkdir(parents=True, exist_ok=True)
    gle_path.write_text("! fake gle", encoding="utf-8")

    subprocess_kwargs: list[dict[str, object]] = []
    infos: list[str] = []
    previews: list[Path] = []

    monkeypatch.setattr("shutil.which", lambda _name: "gle")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: subprocess_kwargs.append(dict(kwargs)) or None,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **_kwargs: infos.append("ok"))
    monkeypatch.setattr(window, "_show_gle_preview", lambda path: previews.append(path))

    window._compile_and_preview_gle(gle_path, "pdf")

    assert subprocess_kwargs
    assert subprocess_kwargs[0]["cwd"] == str(gle_path.parent)
    assert infos == ["ok"]
    assert previews == [gle_path.with_suffix(".pdf")]


class _LabelAxis:
    def __init__(self) -> None:
        self.xlabel: str | None = None
        self.ylabel: str | None = None
        self.title: str | None = None
        self.text_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.fill_between_calls: int = 0
        self.fill_between_data_names: list[str] = []
        self.fill_between_colors: list[str] = []
        self.plot_data_names: list[str] = []
        self.xscale: str | None = None
        self.yscale: str | None = None

    def errorbar_from_file(self, *_args, **_kwargs) -> None:
        return

    def line_from_file(self, *_args, **_kwargs) -> None:
        return

    def errorbar(self, *_args, **_kwargs) -> None:
        return

    def plot(self, *_args, **_kwargs) -> None:
        data_name = _kwargs.get("data_name")
        if isinstance(data_name, str):
            self.plot_data_names.append(data_name)
        return

    def fill_between(self, *_args, **_kwargs) -> None:
        self.fill_between_calls += 1
        data_name = _kwargs.get("data_name")
        if isinstance(data_name, str):
            self.fill_between_data_names.append(data_name)
        color = _kwargs.get("color")
        if isinstance(color, str):
            self.fill_between_colors.append(color)
        return

    def text(self, *args, **kwargs) -> None:
        self.text_calls.append((args, kwargs))
        return

    def set_xlabel(self, value: str, **_kwargs) -> None:
        self.xlabel = value

    def set_ylabel(self, value: str, **_kwargs) -> None:
        self.ylabel = value

    def set_title(self, value: str) -> None:
        self.title = value

    def set_xscale(self, *_args, **_kwargs) -> None:
        if _args:
            self.xscale = str(_args[0])
        return

    def set_yscale(self, *_args, **_kwargs) -> None:
        if _args:
            self.yscale = str(_args[0])
        return

    def get_xlim(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def get_ylim(self) -> tuple[float, float]:
        return (0.0, 1.0)


class _LabelFigure:
    def __init__(self, axes: list[_LabelAxis]) -> None:
        self._axes = axes
        self.subplots_adjust_kwargs: dict[str, float] | None = None

    def add_subplot(self, *_args, **_kwargs):
        return self._axes[0]

    def subplots_adjust(self, **kwargs) -> None:
        self.subplots_adjust_kwargs = {k: float(v) for k, v in kwargs.items()}


def test_fit_subplot_gle_builder_uses_gle_labels(tmp_path: Path, qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def subplots(**_kwargs):
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out.gle")

    assert ax.xlabel is not None
    assert ax.ylabel is not None
    assert "$" not in ax.xlabel
    assert "$" not in ax.ylabel
    assert "{\\it B}" in ax.xlabel
    assert "\\lambda" in ax.ylabel


def test_fit_subplot_export_embeds_model_and_global_params(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=4.2,
        reduced_chi_squared=1.05,
        global_parameters=ParameterSet([Parameter("c", value=0.137)]),
        global_uncertainties={"c": 0.004},
        n_points=8,
    )

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def subplots(**_kwargs):
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out.gle")

    data_path = tmp_path / "out_G0_data.dat"
    fit_path = tmp_path / "out_G0_fit.fit"
    assert data_path.exists()
    data_text = data_path.read_text(encoding="utf-8")
    # Comprehensive metadata: model formula, global parameter table + value,
    # and fit quality must all be embedded in the exported data file.
    assert "! model:" in data_text
    assert "! Global parameter table:" in data_text
    assert "0.137" in data_text
    assert "! chi_squared: 4.2" in data_text
    assert "! reduced_chi_squared: 1.05" in data_text
    # The model curve file carries the same provenance header.
    if fit_path.exists():
        fit_text = fit_path.read_text(encoding="utf-8")
        assert "! model:" in fit_text
        assert "! Global parameter table:" in fit_text


def test_fit_subplot_gle_builder_respects_aspect_ratio_control(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._fit_subplot_aspect_spin.setValue(2.8)
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    ax = _LabelAxis()
    fig = _LabelFigure([ax])
    subplots_kwargs: dict[str, object] = {}

    class _FakeGlp:
        @staticmethod
        def subplots(**kwargs):
            subplots_kwargs.update(kwargs)
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out_width.gle")
    assert "figsize" in subplots_kwargs
    figsize = subplots_kwargs["figsize"]
    assert isinstance(figsize, tuple)
    assert len(figsize) == 2
    assert float(figsize[0]) == pytest.approx(2.8 * 3.1)


def test_fit_subplot_gle_builder_honors_one_to_one_aspect(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._fit_subplot_aspect_spin.setValue(1.0)
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    ax = _LabelAxis()
    fig = _LabelFigure([ax])
    subplots_kwargs: dict[str, object] = {}

    class _FakeGlp:
        @staticmethod
        def subplots(**kwargs):
            subplots_kwargs.update(kwargs)
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out_aspect_1_1.gle")
    figsize = subplots_kwargs["figsize"]
    assert isinstance(figsize, tuple)
    assert float(figsize[0]) == pytest.approx(3.1)


def test_fit_subplot_layout_expands_side_margins_for_square_aspect(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    square = window._fit_subplot_layout_params(1.0)
    wide = window._fit_subplot_layout_params(3.0)

    assert square["left"] > wide["left"]
    assert square["right"] < wide["right"]
    assert square["left"] >= 0.19
    assert square["bottom"] == pytest.approx(0.065)


def test_fit_subplot_gle_builder_applies_dynamic_layout_params(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._fit_subplot_aspect_spin.setValue(1.0)
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def subplots(**_kwargs):
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out_dynamic_layout.gle")
    assert fig.subplots_adjust_kwargs is not None
    expected = window._fit_subplot_layout_params(1.0)
    assert fig.subplots_adjust_kwargs["left"] == pytest.approx(expected["left"])
    assert fig.subplots_adjust_kwargs["right"] == pytest.approx(expected["right"])


def test_restore_state_converts_legacy_fit_width_to_aspect(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window.restore_state({"fit_plot_width": 9.3})
    assert window._fit_subplot_aspect_spin.value() == pytest.approx(3.0)


def test_fit_subplot_without_shared_x_uses_group_titles(tmp_path: Path, qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._fit_share_x_check.setChecked(False)
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="Group A",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="Group B",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.16, 0.11], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=200.0,
        ),
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)
    window._plot_annotations = [
        {
            "x": 0.1,
            "y": 0.9,
            "text": "Group A",
            "axis_tag": "g0",
            "is_group_label": True,
            "artist": None,
        }
    ]

    ax0 = _LabelAxis()
    ax1 = _LabelAxis()
    fig = _LabelFigure([ax0, ax1])

    class _FakeGlp:
        @staticmethod
        def subplots(**_kwargs):
            return fig, [ax0, ax1]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out_unshared.gle")

    assert ax0.title == "Group A"
    assert ax1.title == "Group B"
    assert not ax0.text_calls
    assert not ax1.text_calls


def test_fit_subplot_gle_builder_draws_components_when_enabled(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._parameter_name = "Lambda"
    window._x_key = "field"
    window._show_components_check.setChecked(True)
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([100.0, 200.0], dtype=float),
            y=np.array([0.15, 0.12], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=100.0,
        )
    ]
    window._model = ParameterCompositeModel(["Constant"])
    window._result = CrossGroupFitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def subplots(**_kwargs):
            return fig, [ax]

    window._build_fit_subplot_gle_figure(_FakeGlp(), tmp_path / "out_components.gle")
    assert ax.fill_between_calls > 0
    assert ax.fill_between_data_names
    assert all(name.startswith("component_") for name in ax.fill_between_data_names)
    assert any(name.endswith("_fill") for name in ax.fill_between_data_names)
    assert any(name.endswith("_edge") for name in ax.plot_data_names)
    assert ax.fill_between_colors
    assert ax.fill_between_colors[0] == "lightblue"


def test_local_plot_gle_builder_uses_gle_labels(tmp_path: Path, qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._x_key = "field"
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=5.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01},
            "g1": {"Lambda": 0.01},
        },
    )
    window._local_plot_mode_combo.setCurrentText("Single Axes")

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def figure(**_kwargs):
            return fig

    window._build_local_parameter_gle_figure(_FakeGlp(), tmp_path / "out_local.gle")

    assert ax.xlabel is not None
    assert ax.ylabel is not None
    assert "$" not in ax.xlabel
    assert "$" not in ax.ylabel
    assert "{\\it T}" in ax.xlabel
    assert "\\lambda" in ax.ylabel


def test_local_plot_gle_subplots_share_x_axis(tmp_path: Path, qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._x_key = "field"
    window._local_plot_mode_combo.setCurrentText("Subplots")
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=5.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1), Parameter("nu", value=1.0)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2), Parameter("nu", value=2.0)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01, "nu": 0.1},
            "g1": {"Lambda": 0.01, "nu": 0.1},
        },
    )

    ax0 = _LabelAxis()
    ax1 = _LabelAxis()
    fig = _LabelFigure([ax0, ax1])
    subplots_kwargs: dict[str, object] = {}

    class _FakeGlp:
        @staticmethod
        def subplots(**kwargs):
            subplots_kwargs.update(kwargs)
            return fig, [ax0, ax1]

    window._build_local_parameter_gle_figure(_FakeGlp(), tmp_path / "out_local_subplots.gle")
    assert subplots_kwargs.get("sharex") is True
    assert subplots_kwargs.get("ncols") == 1


def test_local_export_data_file_includes_all_local_and_global_with_units(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._x_key = "field"
    window._local_plot_mode_combo.setCurrentText("Single Axes")
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=1.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=10.0,
        ),
    ]
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        global_parameters=ParameterSet([Parameter("Lambda", value=0.12)]),
        global_uncertainties={"Lambda": 0.005},
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1), Parameter("nu", value=1.0)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2), Parameter("nu", value=2.0)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.0, "nu": 0.1},
            "g1": {"Lambda": 0.02, "nu": 0.2},
        },
    )

    # Select only nu for plotting; data file should still include Lambda local columns too.
    window._rebuild_local_y_controls(["Lambda", "nu"], preferred_selected=["nu"])

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def figure(**_kwargs):
            return fig

    gle_path = tmp_path / "out_local.gle"
    window._build_local_parameter_gle_figure(_FakeGlp(), gle_path)

    data_path = tmp_path / "out_local_local_parameters.dat"
    assert data_path.exists()
    text = data_path.read_text(encoding="utf-8")
    assert "! Global parameter table:" in text
    assert "!   Parameter" in text
    assert "Lambda (µs⁻¹)" in text
    assert "err_Lambda (µs⁻¹)" in text
    assert "nu (MHz)" in text
    assert "err_nu (MHz)" in text
    assert "! Column map:" in text
    assert "!   c 1 = T (K)" in text

    # First data row should preserve zero local uncertainty (not NaN).
    data_rows = [line for line in text.splitlines() if line and not line.startswith("!")]
    assert data_rows
    first = data_rows[0].split()
    # columns: x, Lambda, err_Lambda, nu, err_nu, global_Lambda, err_global_Lambda
    assert first[2] == "0"


def test_local_export_writes_fit_file_with_descriptive_header(
    tmp_path: Path, qapp: QApplication
) -> None:
    window = GlobalParameterFitWindow()
    window._x_key = "field"
    window._local_plot_mode_combo.setCurrentText("Single Axes")
    window._groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=1.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=10.0,
        ),
    ]
    window._result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01},
            "g1": {"Lambda": 0.01},
        },
    )

    fit_model = ParameterCompositeModel(["Linear"])
    fit_result = ParameterModelFitResult(
        success=True,
        parameters=ParameterSet([Parameter("m", value=0.08), Parameter("b", value=0.012)]),
    )
    window._local_model_fits["Lambda"] = ParameterModelFit(
        parameter_name="Lambda",
        x_key="temperature",
        ranges=[
            ModelFitRange(
                x_min=1.0,
                x_max=10.0,
                model=fit_model,
                parameters=ParameterSet([Parameter("m", value=0.08), Parameter("b", value=0.012)]),
                result=fit_result,
            )
        ],
        active=True,
    )

    ax = _LabelAxis()
    fig = _LabelFigure([ax])

    class _FakeGlp:
        @staticmethod
        def figure(**_kwargs):
            return fig

    gle_path = tmp_path / "out_local_fit.gle"
    window._build_local_parameter_gle_figure(_FakeGlp(), gle_path)

    fit_path = tmp_path / "out_local_fit_local_lambda.fit"
    assert fit_path.exists()
    fit_text = fit_path.read_text(encoding="utf-8")
    assert "! Local parameter model fit curve" in fit_text
    assert "! parameter:" in fit_text
    assert "! columns: x y" in fit_text


def test_restore_state_local_selected_y_applies_on_refresh(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    model = ParameterCompositeModel(["Linear"])
    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=0.1,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=20.0,
        ),
    ]
    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1), Parameter("nu", value=1.0)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2), Parameter("nu", value=2.0)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01, "nu": 0.1},
            "g1": {"Lambda": 0.01, "nu": 0.1},
        },
    )

    window.restore_state({"local_selected_y": ["nu"]})
    window.set_results(
        parameter_name="Lambda", x_key="field", groups=groups, model=model, result=result
    )
    selected = window._selected_local_y_parameters()
    assert selected == ["nu"]


def test_local_parameter_plot_supports_log_axes(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._local_log_x_check.setChecked(True)
    model = ParameterCompositeModel(["Linear"])

    groups = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.2, 0.15], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=1.0,
        ),
        ParameterGroupData(
            group_id="g1",
            group_name="G1",
            x=np.array([10.0, 20.0], dtype=float),
            y=np.array([0.22, 0.16], dtype=float),
            yerr=np.array([0.01, 0.01], dtype=float),
            group_variable_value=10.0,
        ),
    ]

    result = CrossGroupFitResult(
        success=True,
        chi_squared=1.0,
        reduced_chi_squared=1.0,
        local_parameters={
            "g0": ParameterSet([Parameter("Lambda", value=0.1)]),
            "g1": ParameterSet([Parameter("Lambda", value=0.2)]),
        },
        local_uncertainties={
            "g0": {"Lambda": 0.01},
            "g1": {"Lambda": 0.01},
        },
    )

    window.set_results(
        parameter_name="Lambda",
        x_key="field",
        groups=groups,
        model=model,
        result=result,
    )

    # Per-parameter log-y control should drive y scaling.
    controls = window._local_y_controls.get("Lambda", {})
    checkbox = controls.get("log") if isinstance(controls, dict) else None
    assert checkbox is not None
    checkbox.setChecked(True)
    window._refresh_local_parameter_plots()

    assert window._local_figure is not None
    assert window._local_figure.axes
    ax = window._local_figure.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"


def test_local_y_selector_uses_non_latex_labels(qapp: QApplication) -> None:
    window = GlobalParameterFitWindow()
    window._rebuild_local_y_controls(["Lambda", "nu"], preferred_selected=["Lambda"])
    item = window._local_y_selector_table.item(0, 0)
    assert item is not None
    assert "$" not in item.text()
