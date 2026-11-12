"""Actual plot and scan adapters retain captured history and export context."""

import json

import numpy as np
import pytest

from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run
from tests.core.test_workflow_reduction import _continuous_run

pytestmark = pytest.mark.gui


def test_plot_payload_and_text_writer_use_captured_recipe(qapp, tmp_path):
    from asymmetry.gui.panels.plot_panel import PlotPanel

    dataset = reduce_run(_continuous_run(0), ReductionSettings(alpha=1.0, alpha_source="user"))
    panel = PlotPanel()
    panel._current_datasets = [dataset]
    payload = panel.get_current_plot_export_data()[0]
    assert payload["effective_recipe"] is not None
    captured = json.dumps(payload["effective_recipe"], sort_keys=True)
    dataset.run.grouping["alpha"] = 3.0
    payload["waterfall_offset"] = 2.0
    for bake in (True, False):
        path = tmp_path / f"{bake}.dat"
        panel._write_data_file(path, payload, x_range=(0.3, 0.5), bake_waterfall_offset=bake)
        lines = path.read_text().splitlines()
        exported = json.loads(
            next(line.split(": ", 1)[1] for line in lines if line.startswith("! effective_recipe:"))
        )
        assert exported["curve_operations"][-1]["applied_value_offset"] == (2.0 if bake else 0.0)
        arrays = np.loadtxt(path, comments="!")
        mask = (payload["data"]["t"] >= 0.3) & (payload["data"]["t"] <= 0.5)
        np.testing.assert_allclose(
            arrays[:, 1], payload["data"]["y"][mask] + (2.0 if bake else 0.0), rtol=1e-8
        )
    assert json.dumps(payload["effective_recipe"], sort_keys=True) == captured
    panel.close()


def test_gui_scan_series_restores_captured_coordinates_and_recipes(qapp):
    from asymmetry.core.representation.series import FitSeries
    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.scan.observations import display_scan
    from tests.gui.test_scan_construction_tasks import scan_datasets

    window = MainWindow()
    datasets = scan_datasets()
    for dataset in datasets:
        window._data_browser.add_dataset(dataset)
    window._plot_workspace.set_active_view("integral_scan")
    window._fit_panel.set_datasets(datasets)
    window._on_scan_requested()
    series = window._project_model.batch(window._alc_scan_series_id)
    window._sync_alc_series_extra()
    restored = FitSeries.from_dict(json.loads(json.dumps(series.to_dict())))
    assert restored.extra["effective_recipe"] == series.extra["effective_recipe"]
    assert all(
        result["effective_recipe"]["units"] == "percent"
        for result in restored.results_by_run.values()
    )
    coordinates = {run: result["coordinates"] for run, result in restored.results_by_run.items()}
    for dataset in datasets:
        dataset.run.metadata["field"] = 999.0
    points = window._alc_points_from_series(restored)
    assert all(point["field"] == coordinates[point["run"]]["field"] for point in points)
    assert all(point["effective_recipe"]["units"] == "fraction" for point in points)
    window._alc_scan_points = points
    shown = display_scan(window, "run")
    assert shown.effective_recipe.to_dict()["units"] == "percent"
    assert all(point.effective_recipe.to_dict()["units"] == "percent" for point in shown.points)
    window._clear_dirty()
    window.close()
