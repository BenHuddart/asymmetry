"""Actual scan action, estimator identity, and cancellable construction lifecycle."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform.preparation import histogram_observation
from tests.core.test_scan_estimators import two_bin_run
from tests.gui.scan_fit_helpers import load_scan_selection

pytestmark = pytest.mark.gui


def scan_datasets(*run_numbers):
    return [
        MuonDataset(
            np.arange(2.0),
            np.zeros(2),
            np.ones(2),
            run=run,
            metadata=dict(run.metadata),
            histogram_observation=histogram_observation(run),
        )
        for run in map(two_bin_run, run_numbers or (13001, 13002))
    ]


def test_actual_gui_action_records_count_integral_percent(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    window._plot_workspace.set_active_view("integral_scan")
    load_scan_selection(window, *scan_datasets())
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
    ("action", "rf_mode"),
    [
        ("success", False),
        ("cancel", False),
        ("error", False),
        ("selection", False),
        ("selection", True),
        ("selection_restored", False),
        ("selection_unchanged", False),
        ("field", False),
        ("window", False),
        ("mode", False),
        ("project", False),
        ("close", False),
        ("supersede", False),
    ],
)
def test_scan_heartbeat_and_atomic_publication(qapp, monkeypatch, action, rf_mode):
    import threading

    from PySide6.QtCore import QThread, QTimer

    import asymmetry.core.transform.integral as integral
    from asymmetry.core.representation import RepresentationType
    from asymmetry.core.representation.series import FitSeries
    from asymmetry.gui.mainwindow import MainWindow

    window = MainWindow()
    window._plot_workspace.set_active_view("integral_scan")
    datasets = scan_datasets(13001, 13002, 13003)
    for dataset in datasets:
        window._data_browser.add_dataset(dataset)
    selection = [13001, 13002]
    batch = FitSeries(
        "batch-held",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=selection.copy(),
        canonical_model="Constant",
    )
    window._project_model.add_batch(batch)
    window._on_series_open_requested(batch.batch_id)
    window._plot_panel.set_fit_range(0, 1)
    window._data_browser.select_runs(selection)
    window._alc_fit_panel._rf_difference_check.setChecked(rf_mode)
    window._on_scan_requested()
    previous = window._project_model.batch(window._alc_scan_series_id)
    view = window._alc_scan_view
    view._add_region()
    analysis = view.analysis_state()
    assert analysis["regions"]
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
        elif action in {"selection", "selection_restored"}:
            window._data_browser.select_runs([13002, 13003])
            if action == "selection_restored":
                window._data_browser.select_runs(selection)
        elif action == "selection_unchanged":
            window._data_browser.select_runs(selection)
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
    if action in {"success", "supersede", "selection_unchanged"}:
        assert window._alc_scan_series_id != previous.batch_id
    elif action not in {"project", "close"}:
        assert window._alc_scan_series_id == previous.batch_id
        assert window._project_model.batches == {batch.batch_id: batch, previous.batch_id: previous}
        assert view.analysis_state() == analysis
    if action in {"selection", "selection_restored"}:
        assert len(calls) == 1  # the changed selection cancels the remaining reduction
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
    window._plot_workspace.set_active_view("integral_scan")
    window._plot_panel.set_fit_range(0, 1)
    load_scan_selection(window, *datasets)
    window._data_browser._table.item(0, 3).setText("200")
    assert datasets[0].histogram_observation.source.metadata["field"] == 100
    assert datasets[0].run.metadata["field"] == 200
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
    load_scan_selection(window, *curves)
    window._on_scan_requested()
    assert not window._alc_scan_points
    assert "source run" in window._log_panel.to_plain_text()
    window.close()
