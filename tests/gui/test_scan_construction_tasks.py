"""Actual scan action, estimator identity, and cancellable construction lifecycle."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.preparation import histogram_observation
from tests.core.test_scan_estimators import two_bin_run

pytestmark = pytest.mark.gui


def scan_datasets():
    return [
        MuonDataset(
            np.arange(2.0),
            np.zeros(2),
            np.ones(2),
            run=run,
            metadata=dict(run.metadata),
            histogram_observation=histogram_observation(run),
        )
        for run in (two_bin_run(13001), two_bin_run(13002))
    ]


def test_actual_gui_action_records_count_integral_percent(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    window._plot_workspace.set_active_view("integral_scan")
    window._fit_panel.set_datasets(scan_datasets())
    window._alc_fit_panel._rf_difference_check.setChecked(True)
    window._on_scan_requested()
    assert [p["value"] for p in window._alc_scan_points] == pytest.approx(
        [0.015841584158415828] * 2
    )
    series = window._project_model.batch(window._alc_scan_series_id)
    assert series.results_by_run[13001]["parameters"][window._SCAN_QUANTITY] == pytest.approx(
        1.5841584158415828
    )
    window._sync_alc_series_extra()
    assert series.extra["estimator"] == "integral"
    window.close()


@pytest.mark.parametrize(
    "action",
    [
        "success",
        "cancel",
        "error",
        "selection",
        "field",
        "window",
        "mode",
        "project",
        "close",
        "supersede",
    ],
)
def test_scan_heartbeat_and_atomic_publication(qapp, monkeypatch, action):
    import threading

    from PySide6.QtCore import QThread, QTimer

    import asymmetry.core.transform.integral as integral
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    window._plot_workspace.set_active_view("integral_scan")
    datasets = scan_datasets()
    for dataset in datasets:
        window._data_browser.add_dataset(dataset)
    window._plot_panel.set_fit_range(0, 1)
    window._data_browser._table.selectAll()
    window._on_new_series_from_selection()
    crops = window._fit_panel.batch_datasets()
    assert crops[0] is not datasets[0]
    assert crops[0].run is datasets[0].run
    window._on_scan_requested()
    previous = window._alc_scan_series_id
    original = integral.reduce_period_integral
    started, released = threading.Event(), threading.Event()
    calls = []

    def slow(*args, **kwargs):
        calls.append(QThread.currentThread() != window.thread())
        if len(calls) == 1:
            started.set()
            assert released.wait(5), "GUI heartbeat did not release the worker"
            if action == "error":
                raise ValueError("authored scan failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(integral, "reduce_period_integral", slow)
    ticks = []
    timer = QTimer()
    timer.setInterval(5)

    def heartbeat():
        ticks.append(1)
        if not started.is_set() or len(ticks) < 3:
            return
        timer.stop()
        if action == "cancel":
            window._scan_controller.jobs[0].cancel()
        elif action == "selection":
            window._fit_panel.set_datasets(scan_datasets())
        elif action == "field":
            window._data_browser._table.item(0, 3).setText("200")
        elif action == "window":
            window._plot_panel.set_fit_range(0, 0.5)
        elif action == "mode":
            window._alc_fit_panel._rf_difference_check.setChecked(True)
        elif action == "project":
            window._clear_all_state()
        elif action == "close":
            window._clear_dirty()
            # Close joins the worker; release after the observed GUI heartbeats.
            threading.Timer(0.1, released.set).start()
            window.close()
        released.set()
        if action == "supersede":
            window._on_scan_requested()

    timer.timeout.connect(heartbeat)
    timer.start()
    window._on_scan_requested()
    timer.stop()
    assert len(ticks) >= 3
    assert all(calls)
    assert not window._scan_controller.jobs
    if action in {"success", "supersede"}:
        assert window._alc_scan_series_id != previous
    elif action not in {"project", "close"}:
        assert window._alc_scan_series_id == previous
    if action == "success":
        assert len(calls) == 2  # one correction/reduction per selected run
    if action == "error":
        assert "authored scan failure" in window._log_panel.to_plain_text()
    window._clear_dirty()
    window.close()


def test_actual_browser_field_edit_rebuilds_at_current_coordinate(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    datasets = scan_datasets()
    for dataset in datasets:
        window._data_browser.add_dataset(dataset)
    window._plot_workspace.set_active_view("integral_scan")
    window._plot_panel.set_fit_range(0, 1)
    window._data_browser._table.selectAll()
    window._on_new_series_from_selection()
    crop = window._fit_panel.batch_datasets()[0]
    assert crop is not datasets[0]
    window._data_browser._table.item(0, 3).setText("200")
    assert crop.metadata["field"] == 100
    assert crop.run.metadata["field"] == 200
    window._on_scan_requested()
    points = {point["run"]: point for point in window._alc_scan_points}
    assert points[13001]["field"] == 200
    assert points[13001]["value"] == pytest.approx(0.015841584158415828)
    window._clear_dirty()
    window.close()


def test_gui_curve_only_request_explains_unavailable_replay(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    curves = [
        MuonDataset(np.arange(2.0), np.ones(2), np.ones(2), metadata={"run_number": n})
        for n in (1, 2)
    ]
    window._plot_workspace.set_active_view("integral_scan")
    window._fit_panel.set_datasets(curves)
    window._on_scan_requested()
    assert not window._alc_scan_points
    assert "source run" in window._log_panel.to_plain_text()
    window.close()
