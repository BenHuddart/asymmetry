"""Calibration workers and immutable promotion tables at the GUI boundary."""

import threading

import pytest
from PySide6.QtWidgets import QCheckBox, QMessageBox

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.transform.deadtime import DeadtimeCalibration
from asymmetry.gui.windows.grouping import deadtime_section
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow
from tests._qt_helpers import wait_for
from tests.gui.test_grouping_preview_pane import _histogram_dataset

pytestmark = pytest.mark.gui


def test_calibration_runs_off_the_gui_thread(qapp, monkeypatch):
    dataset = _histogram_dataset(grouping_extra={"good_frames": 1000})
    dialog = GroupingDialog([dataset])
    threads = []
    gui_thread = threading.get_ident()

    def calibrate(histograms, *, cancel_callback, **kwargs):
        threads.append(threading.get_ident())
        assert not cancel_callback()
        return DeadtimeCalibration((-0.02, 0.01), (0.001, 0.002), "calibrate", dataset.run_number)

    monkeypatch.setattr(deadtime_section, "fit_deadtime_calibration", calibrate)
    section = dialog._deadtime_section
    section._on_calibrate_clicked()
    wait_for(lambda: section.isEnabled(), qapp)
    assert threads and threads[0] != gui_thread
    assert section._display_values() == [-0.02, 0.01]
    assert section.get_policy().source_run == dataset.run_number
    dialog._clear_dirty()
    dialog.close()


def test_calibration_exposes_missing_frame_error(qapp, monkeypatch):
    dialog = GroupingDialog([_histogram_dataset()])
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, text: messages.append(text))
    dialog._deadtime_section._on_calibrate_clicked()
    wait_for(lambda: bool(messages), qapp)
    assert "good frames must be finite and > 0" in messages[0]
    dialog._clear_dirty()
    dialog.close()


def test_promote_uses_captured_total_and_reports_increment(qapp):
    dataset = _histogram_dataset()
    window = MultiGroupFitWindow()
    window.set_dataset(dataset)
    tab = window._single_fit_tab
    calibration = DeadtimeCalibration((0.0112, 0.01), (None, None), "count_fit", dataset.run_number)
    result = FitResult(
        objective=Objective.GAUSSIAN,
        success=True,
        parameters=ParameterSet([Parameter("DT0", 0.0012)]),
        deadtime_calibration=calibration,
    )
    tab._store_count_deadtime(result, 1)
    dataset.run.grouping["dead_time_us"] = [0.3, 0.3]
    window._on_promote_deadtime()
    assert dataset.run.grouping["dead_time_us"] == [0.0112, 0.01]
    assert tab._results_card.content_html() == (
        "Group 1 detectors: DT₀ +0.0012 µs added to the applied correction. Re-reduce the run to apply."
    )
    assert all(box.text() != "accumulate" for box in window.findChildren(QCheckBox))
    window.close()


def test_unfitted_estimate_keeps_an_incomplete_request(qapp):
    dialog = GroupingDialog([_histogram_dataset()])
    section = dialog._deadtime_section
    section._mode_buttons["estimate"].click()
    payload = dialog.get_grouping_result()
    assert payload["deadtime_correction"]
    assert payload["deadtime_mode"] == "estimate"
    assert payload["dead_time_us"] == []
    assert not section._tasks.active_count
    wait_for(lambda: not dialog._preview_pane.corrections_complete, qapp)
    assert not dialog._apply_btn.isEnabled()
    dialog._clear_dirty()
    dialog.close()
