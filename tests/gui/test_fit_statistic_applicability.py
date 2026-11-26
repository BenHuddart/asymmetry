"""Fit statistic and applicability labels across the live GUI and saved results."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.cross_group import run_cross_group_fit_from_config
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameter_models import (
    ErrorMode,
    ModelFitRange,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFit,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.trend_result import ColumnErrorSource
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.representation.global_fit_study import GlobalFitStudy
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupDialogOutput, CrossGroupFitDialog
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel, _FitRow, _GroupFitData
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from asymmetry.gui.windows.global_parameter_fit_window import GlobalParameterFitWindow
from docs.screenshots.scenarios._base import auto_dismiss_modals
from tests._qt_helpers import wait_for
from tests.gui.test_model_fit_dialog import _make_dialog, _wait_for_fit

pytestmark = pytest.mark.gui


def test_use_cross_group_fit_projects_its_disjoint_windows_and_group_statistic(qapp, monkeypatch):
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    y = np.array([1.0, 100.0, 3.0, 3.0, 100.0])
    sigma = np.array([0.1, 10.0, 1.0, 1.0, np.nan])
    windows = [(0.0, 0.25), (0.75, 3.0)]
    panel = FitParametersPanel()
    groups = []
    for index, gid in enumerate(("a", "b")):
        rows = [
            _FitRow(
                run_number=i + 1,
                run_label=str(i + 1),
                field=xi,
                temperature=float(index),
                values={"Lambda": yi + index},
                errors={"Lambda": ei},
            )
            for i, (xi, yi, ei) in enumerate(zip(x, y, sigma, strict=True))
        ]
        group = _GroupFitData(gid, gid, rows, None, ["Lambda"], "field", {}, [])
        groups.append(group)
    panel._group_fit_results = {group.group_id: group for group in groups}

    def use_fit(dialog):
        dialog._apply_existing_config(
            {
                "model": ParameterCompositeModel(["Constant"]).to_dict(),
                "parameter_rows": [{"name": "c", "type": "Global", "initial": 0.0}],
                "error_mode": ErrorMode.COLUMN_STABILIZED.value,
                "windows": [list(window) for window in windows],
            }
        )
        dialog._run_fit(0)
        _wait_for_fit(dialog)
        assert dialog._result.success
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        return dialog.result()

    monkeypatch.setattr(CrossGroupFitDialog, "exec", use_fit)
    payload = panel._run_cross_group_model_fit("Lambda", groups, x_key_override="field")
    joint = payload["fit_result"]
    contributions = []
    for group in groups:
        gid = group.group_id
        fit = panel._group_fit_results[gid].model_fits["Lambda"]
        projected = fit.ranges[0]
        assert projected.windows == windows
        assert projected.result.n_points == 3
        assert projected.result.parameters["c"].value == joint.global_parameters["c"].value
        assert projected.result.dof is None
        assert np.isnan(projected.result.reduced_chi_squared)
        assert projected.result.chi_squared == pytest.approx(joint.per_group_chi_squared[gid])
        contributions.append(projected.result.chi_squared)
    assert sum(contributions) == pytest.approx(joint.chi_squared)

    dialog = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=sigma,
        existing_fit=panel._group_fit_results["a"].model_fits["Lambda"],
    )
    dialog._show_residuals_check.setChecked(True)
    wait_for(
        lambda: dialog._preview._ranges[0].residuals is not None,
        qapp,
    )
    xs, values = dialog._preview._ranges[0].residuals
    np.testing.assert_array_equal(xs, x[[0, 2, 3]])
    assert sum(values**2) == pytest.approx(contributions[0])
    dialog.close()
    panel.close()


@pytest.mark.parametrize("mode, value", [(ErrorMode.ABSOLUTE, 0.42), (ErrorMode.PERCENT, 21.0)])
def test_trend_member_refit_reuses_the_recorded_error_setting(qapp, mode, value):
    panel = FitParametersPanel()
    x = np.arange(3.0)
    y = np.array([1.0, 3.0, 4.0])
    sigma = np.full_like(x, value) if mode is ErrorMode.ABSOLUTE else y * value / 100
    model = ParameterCompositeModel(["Constant"])
    parameters = ParameterSet([Parameter("c", 0.0)])
    initial = fit_parameter_model(x, y, None, model, parameters, error_mode=mode, error_value=value)
    panel._model_fits["Lambda"] = ParameterModelFit(
        "Lambda", "field", [ModelFitRange(0.0, 2.0, model, parameters, initial)]
    )
    panel._rows = [
        _FitRow(i + 1, str(i + 1), xi, 0.0, {"Lambda": yi}, {})
        for i, (xi, yi) in enumerate(zip(x, y, strict=True))
    ]
    panel.refit_active_model_fits()
    wait_for(lambda: not panel._refit_in_progress, qapp)
    refit = panel._model_fits["Lambda"].ranges[0].result
    assert refit is not initial
    assert refit.error_value == value
    assert refit.uncertainties["c"] == pytest.approx(1 / np.sqrt(np.sum(sigma**-2)))
    panel.close()


def test_global_window_local_trend_drops_missing_column_errors(qapp, monkeypatch):
    x = np.arange(4.0)
    y = [1.0, 3.0, 3.0, 100.0]
    errors = [0.1, 1.0, 1.0, np.nan]
    model = ParameterCompositeModel(["Constant"])
    groups = [
        ParameterGroupData(str(i), str(i), x, x, np.ones_like(x), xi) for i, xi in enumerate(x)
    ]
    result = run_cross_group_fit_from_config(
        groups, {"model": model.to_dict(), "parameter_rows": [{"name": "c", "type": "Local"}]}
    ).result
    for group, yi, error in zip(groups, y, errors, strict=True):
        result.local_parameters[group.group_id]["c"].value = yi
        result.local_uncertainties[group.group_id]["c"] = error
    window = GlobalParameterFitWindow()
    window.set_results(
        parameter_name="Lambda", x_key="field", groups=groups, model=model, result=result
    )
    window._local_model_fits["c"] = ParameterModelFit(
        "c", "temperature", [ModelFitRange(0.0, 3.0, model, ParameterSet([Parameter("c", 0.0)]))]
    )

    def fit_local(dialog):
        dialog._run_fit(0)
        _wait_for_fit(dialog)
        result = dialog.get_model_fit().ranges[0].result
        assert result.success
        assert result.n_points == 3
        assert result.parameters["c"].value == pytest.approx(106 / 102, abs=1e-4)
        assert result.column_error_source is ColumnErrorSource.SUPPLIED
        assert result.assess_quality() is not None
        dialog.close()
        return dialog.result()

    monkeypatch.setattr(ModelFitDialog, "exec", fit_local)
    window._open_local_model_fit_dialog("c")
    window.close()


@pytest.mark.parametrize("cross_group", [False, True])
def test_stabilised_column_selector_and_residuals(qapp, cross_group, tmp_path):
    x = np.arange(4.0)
    y = np.array([1.0, 3.0, 3.0, 100.0])
    sigma = np.array([0.1, 1.0, 1.0, np.nan])
    model = ParameterCompositeModel(["Constant"])
    common = {"parameter_name": "Lambda", "x_key": "field"}
    if cross_group:
        groups = [
            ParameterGroupData(name, name, x, y + 10 * index, sigma, 0.0)
            for index, name in enumerate(("a", "b"))
        ]
        config = {
            "model": model.to_dict(),
            "parameter_rows": [{"name": "c", "type": "Local", "initial": 0.0}],
        }
        dialog = CrossGroupFitDialog(**common, groups=groups, existing_config=config)
    else:
        fit = ParameterModelFit(
            **common,
            ranges=[ModelFitRange(0.0, 3.0, model, ParameterSet([Parameter("c", 0.0)]))],
        )
        data = {"x_values": x, "y_values": y, "y_errors": sigma}
        dialog = ModelFitDialog(**common, **data, existing_fit=fit)
    picker = dialog._error_mode_combo
    assert dialog._error_mode() is ErrorMode.COLUMN
    picker.setCurrentText("Column (stabilised errors)")
    dialog._show_residuals_check.setChecked(True)
    dialog._run_fit(0)
    _wait_for_fit(dialog)
    result = dialog._result_for_range(0)
    assert result.success
    assert result.column_error_floor_applied
    assert "floor changes the weights" in dialog._quality_label.text()

    def plotted():
        axes = dialog._preview._figure.get_axes()
        return (
            [line for line in axes[-1].get_lines() if line.get_marker() == "o"]
            if len(axes) == 2
            else []
        )

    expected = (y[:3] - 5 / 3) / np.array([0.5, 1.0, 1.0])
    wait_for(lambda: plotted() and np.allclose(plotted()[0].get_ydata(), expected), qapp)
    line = plotted()[0]
    np.testing.assert_array_equal(line.get_xdata(), x[:3])
    np.testing.assert_allclose(line.get_ydata(), expected, atol=1e-4)
    chi_squared = result.per_group_chi_squared["a"] if cross_group else result.chi_squared
    assert np.sum(line.get_ydata() ** 2) == pytest.approx(chi_squared)

    project = tmp_path / "weights.asymproj"
    if cross_group:
        config = dialog._collect_config()
        reopened = CrossGroupFitDialog(**common, groups=groups, existing_config=config)
    else:
        panel = FitParametersPanel()
        panel._model_fits["Lambda"] = dialog.get_model_fit()
        save_project(
            {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "datasets": [],
                "fit_parameters_state": panel.get_state(),
            },
            project,
        )
        panel.restore_state(load_project(project)["fit_parameters_state"])
        reopened = ModelFitDialog(**common, **data, existing_fit=panel._model_fits["Lambda"])
        assert reopened._result_for_range(0).column_error_floor_applied
        panel.close()
    assert reopened._error_mode_combo.currentText() == "Column (stabilised errors)"
    reopened.close()
    dialog.close()


@pytest.mark.parametrize(
    ("mode", "value"),
    [(ErrorMode.COLUMN_STABILIZED, None), (ErrorMode.ABSOLUTE, 0.42), (ErrorMode.PERCENT, 17.0)],
)
def test_results_residuals_use_the_captured_objective(qapp, mode, value) -> None:
    x = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    y = np.array([1.0, 100.0, 3.0, 3.0, 100.0])
    sigma = np.array([0.1, 10.0, 1.0, 1.0, 10.0])
    xerr = np.array([0.4, 0.1, 0.3, 0.2, 0.5])
    groups = [
        ParameterGroupData(str(i), str(i), x, y, sigma, float(i), xerr=xerr) for i in range(2)
    ]
    model = ParameterCompositeModel(["Linear"])
    config = {
        "model": model.to_dict(),
        "windows": [[0.0, 0.25], [0.75, 2.25]],
        "use_x_errors": True,
        "error_mode": mode.value,
        "error_value": value,
    }
    fit = run_cross_group_fit_from_config(groups, config).result
    assert fit.success
    assert fit.uses_effective_variance
    study = GlobalFitStudy(
        "study",
        "trend",
        "Lambda",
        "field",
        "Field",
        "temperature",
        "Temperature",
        "",
        "",
        groups=groups,
        model=model,
        result=fit,
        config=config,
    )
    window = GlobalParameterFitWindow()
    window.set_study(GlobalFitStudy.from_dict(study.to_dict()), stale=False)
    wait_for(lambda: not window._fit_curve_compute_active, qapp)
    window._fit_residuals_check.setChecked(True)

    mask = [0, 2, 3]
    if mode is ErrorMode.COLUMN_STABILIZED:
        effective = np.array([0.5, 1.0, 1.0])
    elif mode is ErrorMode.ABSOLUTE:
        effective = np.full(3, value)
    else:
        effective = y[mask] * value / 100
    slope = fit.global_parameters["m"].value
    intercept = fit.global_parameters["b"].value
    expected = (y[mask] - intercept - slope * x[mask]) / np.sqrt(
        effective**2 + (slope * xerr[mask]) ** 2
    )
    for group, axis in zip(groups, window._left_figure.axes, strict=True):
        line = axis.containers[0].markerline
        np.testing.assert_array_equal(line.get_xdata(), x[mask])
        np.testing.assert_allclose(line.get_ydata(), expected)
        assert sum(line.get_ydata() ** 2) == pytest.approx(
            fit.per_group_chi_squared[group.group_id]
        )
    window.close()


def test_quality_verdict_shown_for_column_mode_fit(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        column_error_source=ColumnErrorSource.SUPPLIED,
        n_points=12,
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "Quality of fit" in text
    assert "good" in text
    assert "target band" in text
    assert dlg._quality_label.toolTip()


def test_quality_verdict_suppressed_for_scatter_mode(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=10.0,
        reduced_chi_squared=1.0,
        parameters=fit_range.parameters,
        error_mode="scatter",
        n_points=12,
    )
    dlg._select_range(0)
    text = dlg._quality_label.text()
    assert "No χ² quality verdict" in text

    fit_range.result = None
    dlg._select_range(0)
    assert dlg._quality_label.text() == ""


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_missing_column_errors_show_no_verdict(qapp: QApplication, scale: float, tmp_path) -> None:
    model = ParameterCompositeModel(["Constant"])
    params = ParameterSet([Parameter("c", 0.0)])
    dialog = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=np.arange(12.0),
        y_values=np.tile([scale, -scale], 6),
        y_errors=None,
        existing_fit=ParameterModelFit(
            parameter_name="Lambda",
            x_key="field",
            ranges=[ModelFitRange(0.0, 11.0, model, params)],
            active=True,
        ),
    )

    dialog._run_fit(0)
    _wait_for_fit(dialog)

    assert dialog._fit.ranges[0].result.effective_error_mode == ErrorMode.NONE.value
    assert "No χ² quality verdict" in dialog._quality_label.text()
    assert "target band" not in dialog._quality_label.text()
    assert dialog._range_cards[0]._view.status_chip_html == ""

    panel = FitParametersPanel()
    restored = FitParametersPanel()
    try:
        panel._model_fits["Lambda"] = dialog._fit
        project = tmp_path / "trend.asymp"
        state = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "fit_parameters_state": panel.get_state(),
        }
        save_project(state, project, backup=False)
        json.loads(project.read_text(), parse_constant=lambda token: pytest.fail(token))
        restored.restore_state(load_project(project)["fit_parameters_state"])
        fit = restored._model_fits["Lambda"].ranges[0].result
        assert fit.error_mode == ErrorMode.COLUMN.value
        assert fit.effective_error_mode == ErrorMode.NONE.value
        assert fit.quality_reason == dialog._fit.ranges[0].result.quality_reason
        written = restored._write_fit_files(tmp_path / "trend.gle", "field", ["Lambda"])
        text = written[("Lambda", 0)].read_text()
        assert f"! chi_squared: {12 * scale**2:g}" in text
        assert "! requested_error_mode: column" in text
        assert "! effective_y_error_mode: none" in text
        assert "! quality_reason: No χ² quality verdict" in text
    finally:
        panel.close()
        restored.close()
        dialog.close()


def test_window_bounds_edit_invalidates_stale_result(qapp: QApplication) -> None:
    from asymmetry.core.fitting.parameter_models import ParameterModelFitResult

    dlg = _make_dialog(qapp)
    fit_range = dlg._fit.ranges[0]
    fit_range.windows = [(1.0, 5.0)]
    fit_range.result = ParameterModelFitResult(
        success=True,
        chi_squared=9.0,
        reduced_chi_squared=0.9,
        parameters=fit_range.parameters,
        error_mode="column",
        column_error_source=ColumnErrorSource.SUPPLIED,
        n_points=12,
    )
    dlg._rebuild_ranges_ui()
    dlg._select_range(0)
    assert "Quality of fit" in dlg._quality_label.text()

    # Editing an interval's max (single-window range collapses to a plain range).
    dlg._on_region_interval_edited(0, 1, 4.0)

    assert fit_range.result is None
    assert fit_range.x_min == 1.0
    assert fit_range.x_max == 4.0
    assert dlg._quality_label.text() == ""
    assert "not yet run" in dlg._chi2_label.text().lower()
    # The card now carries the range's status; a cleared result reads "not_run".
    assert dlg._range_cards[0]._view.status == "not_run"


@pytest.mark.parametrize("historical", [None, "missing", "floored"])
def test_single_fit_reopens_and_exports_applicable_statistics(
    qapp, tmp_path, monkeypatch, historical
):
    dataset = MuonDataset(
        time=np.array([0.0]),
        asymmetry=np.array([5.0]),
        error=np.array([0.1]),
        metadata={"run_number": 101},
    )
    model = CompositeModel(["Constant"])
    result = FitEngine().fit_arrays(
        dataset.time,
        dataset.asymmetry,
        dataset.error,
        model.function,
        ParameterSet([Parameter(model.param_names[0], 0.5, min=0, max=1)]),
    )
    assert result.success
    project = tmp_path / "single-point.asymp"
    report = tmp_path / "fit-report.txt"
    windows = []
    with auto_dismiss_modals():
        try:
            window = MainWindow()
            windows.append(window)
            window._on_fit()
            window._data_browser.add_dataset(dataset)
            window._current_dataset = dataset
            window._fit_panel.set_dataset(dataset)
            window._fit_panel._single_tab._set_composite_model(model)
            window._record_single_fit_slot(result)
            [(_, live)] = window._collect_latest_fit_records()
            assert live["ndof"] == 0
            if historical:
                stale_html = "<b>poor fit; target band</b>"
                live.update(
                    reduced_chi_squared=live["chi_squared"],
                    quality={"dof": 1, "verdict": "poor", "band_low": 0.1, "band_high": 4.0},
                    quality_reason=None,
                    result_html=stale_html,
                )
                if historical == "floored":
                    live["ndof"] = 1
                else:
                    del live["ndof"]
                rep = window._project_model.representation(
                    101, window._active_representation_type()
                )
                rep.fit.ui_state["result_html"] = stale_html
            state = window.collect_project_state()
            if historical:
                state["schema_version"] = 41
                state.pop("view_modes_state")
            save_project(state, project, backup=False)
            json.loads(project.read_text(), parse_constant=lambda token: pytest.fail(token))
            restored = MainWindow()
            windows.append(restored)
            restored.restore_project_state(load_project(project), str(project))
            wait_for(lambda: bool(restored._collect_latest_fit_records()), qapp)
            [(_, record)] = restored._collect_latest_fit_records()
            assert record["ndof"] == (None if historical else 0)
            assert record["chi_squared"] == pytest.approx(1600.0, abs=1e-3)
            assert record["reduced_chi_squared"] is None
            assert record["quality"] is None
            if historical:
                assert record["historical_reduced_chi_squared"] == record["chi_squared"]
                restored_html = restored._single_fit_restore_payload(dataset)["result_html"]
                assert "unavailable" in restored_html
                assert "poor" not in restored_html

            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(report), ""))
            restored._on_export_fit_report()
            text = report.read_text()
            assert "chi^2 = 1600" in text
            assert "chi^2/nu = unavailable" in text
            if historical:
                assert "Residual degrees of freedom are unknown" in text
            else:
                assert "(nu=0)" in text
                assert "requires positive residual degrees of freedom" in text
            assert "target" not in text and "[poor" not in text
        finally:
            for window in reversed(windows):
                window._dirty = False
                window.close()
                window.deleteLater()


def test_completed_trend_results_reopen_without_a_synthetic_reduced_statistic(qapp, tmp_path):
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 5.0])
    model = ParameterCompositeModel(["Linear"])
    parameters = ParameterSet([Parameter("m", 2.0), Parameter("b", 1.0)])
    single = ParameterModelFit(
        parameter_name="Lambda",
        x_key="field",
        ranges=[ModelFitRange(1.0, 2.0, model, parameters)],
        active=True,
    )
    dialog = ModelFitDialog(
        parameter_name="Lambda",
        x_key="field",
        x_values=x,
        y_values=y,
        y_errors=None,
        existing_fit=single,
    )
    dialog._run_fit(0)
    _wait_for_fit(dialog)
    fit = dialog.get_model_fit()
    assert fit.ranges[0].result.success
    assert fit.ranges[0].result.dof == 0
    assert "reduced chi2 = unavailable" in dialog._chi2_label.text()

    groups = [
        ParameterGroupData(name, name, x, y, None, value) for name, value in [("a", 2), ("b", 3)]
    ]
    cross = run_cross_group_fit_from_config(
        groups,
        {
            "model": model.to_dict(),
            "parameter_rows": [
                {"name": name, "initial": parameters[name].value, "type": "Local"}
                for name in parameters.names
            ],
        },
    )
    assert cross.result.success and cross.result.dof == 0
    output = CrossGroupDialogOutput(
        cross.model, cross.result, cross.fitted_groups, "field", 1.0, 2.0, cross.config
    )
    windows = []
    with auto_dismiss_modals():
        try:
            window = MainWindow()
            windows.append(window)
            panel = window._fit_parameters_panel
            panel._model_fits["Lambda"] = fit
            panel.model_fit_completed.emit("Lambda", "field", fit)
            panel.cross_group_fit_completed.emit("Lambda", groups, output)
            project = tmp_path / "trend-series.asymp"
            save_project(window.collect_project_state(), project, backup=False)

            restored = MainWindow()
            windows.append(restored)
            restored.restore_project_state(load_project(project), str(project))
            wait_for(lambda: len(restored._project_model.batches) >= 3, qapp)
            series = [
                batch
                for key, batch in restored._project_model.batches.items()
                if key.startswith("modelfit-")
            ]
            assert len(series) == 3
            for batch in series:
                for record in batch.results_by_run.values():
                    assert record["chi_squared"] == pytest.approx(0.0, abs=1e-10)
                    assert record["ndof"] == 0
                    assert record["reduced_chi_squared"] is None
                    assert (
                        "requires positive residual degrees of freedom" in record["quality_reason"]
                    )
                    assert record["error_mode"] == ErrorMode.COLUMN.value
                    assert "chi2_r" not in record["parameters"]
                    if batch.batch_id.startswith("modelfit-single-"):
                        assert record["effective_y_error_mode"] == ErrorMode.NONE.value
                    else:
                        assert record["column_error_sources"] == {
                            "a": "assumed_unit",
                            "b": "assumed_unit",
                        }
        finally:
            dialog.close()
            for window in reversed(windows):
                window._dirty = False
                window.close()
                window.deleteLater()
