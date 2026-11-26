"""Fit statistic and applicability labels across the live GUI and saved results."""

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

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
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.trend_result import ColumnErrorSource
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.cross_group_fit_dialog import CrossGroupDialogOutput
from asymmetry.gui.panels.fit_parameters_panel import FitParametersPanel
from asymmetry.gui.panels.model_fit_dialog import ModelFitDialog
from docs.screenshots.scenarios._base import auto_dismiss_modals
from tests._qt_helpers import wait_for
from tests.gui.test_model_fit_dialog import _make_dialog, _wait_for_fit

pytestmark = pytest.mark.gui


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
