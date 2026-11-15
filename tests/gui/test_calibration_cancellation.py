"""Actual numerical work and queued calibration terminals obey request ownership."""

from copy import deepcopy
from threading import Event

import pytest
from PySide6.QtCore import QTimer

from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.gui.windows.grouping import alpha_section
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from docs.screenshots.data.calibration import calibration_control
from tests.gui.test_alpha_section import configured
from tests.gui.test_grouping_dialog import _wait_for_t0_detection, _wait_until

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("stage", ["quick", "general", "preparation", "solve"])
def test_stop_during_actual_work_exits_at_work_boundary(qapp, monkeypatch, stage):
    from asymmetry.core.fitting import count_domain, count_inputs
    from asymmetry.core.transform import asymmetry

    method = {"quick": "diamagnetic", "general": "general"}.get(stage, "tf_calibration_fit")
    section, _, _ = configured(qapp, method)
    owner, seam = {
        "quick": (asymmetry, "_diamagnetic_objective"),
        "general": (asymmetry, "_equal_count_edges"),
        "preparation": (count_inputs, "count_time_selection"),
        "solve": (count_domain, "_cost_value"),
    }[stage]
    original = vars(owner)[seam]
    entered, release = Event(), Event()
    calls, inversions, published, beats = [], [], [], []
    inverse = asymmetry.np.linalg.inv

    def hold_actual_work(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(10), "Work barrier was not released"
        return value

    def observe_inverse(*args, **kwargs):
        inversions.append(1)
        return inverse(*args, **kwargs)

    monkeypatch.setattr(owner, seam, hold_actual_work)
    if stage == "general":
        monkeypatch.setattr(asymmetry.np.linalg, "inv", observe_inverse)
    section.result_ready.connect(published.append)
    timer = QTimer()
    timer.timeout.connect(lambda: beats.append(1))
    timer.start(1)
    try:
        section._on_estimate()
        _wait_until(entered.is_set)
        worker = section._worker
        _wait_until(lambda: len(beats) >= 2)
        assert not release.is_set() and section._tasks.active_count == 1
        section._stop_btn.click()
        assert worker.is_cancelled()
        release.set()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert len(calls) <= (14 if stage == "solve" else 1)
        assert inversions == []
        assert published == [] and section.pending() == {}
        assert section._estimate_btn.isEnabled() and not section._stop_btn.isEnabled()
        monkeypatch.setattr(owner, seam, original)
        if stage == "general":
            section._method_combo.setCurrentIndex(section._method_combo.findData("ratio"))
        section._on_estimate()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert len(published) == 1 and section.pending()["single"].alpha is not None
    finally:
        release.set()
        timer.stop()
        section.shutdown()


@pytest.mark.parametrize("terminal", ["success", "error", "cancel"])
@pytest.mark.parametrize("new_state", ["working", "complete"])
def test_queued_old_terminal_cannot_change_new_dialog_request(
    qapp, monkeypatch, terminal, new_state
):
    data = calibration_control()
    dialog = GroupingDialog([data])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    section._method_combo.setCurrentIndex(section._method_combo.findData("ratio"))
    original = alpha_section.run_calibration
    entered_a, entered_b, release_a, release_b = Event(), Event(), Event(), Event()
    requests, published = [], []
    section.result_ready.connect(published.append)

    def held_terminal(worker, request):
        reply = original(worker, request)
        requests.append(request.generation)
        if len(requests) == 1:
            entered_a.set()
            assert release_a.wait(10)
            if terminal == "error":
                raise ValueError("owned late error")
            if terminal == "cancel":
                raise FitCancelledError("owned late cancellation")
        else:
            entered_b.set()
            if new_state == "working":
                assert release_b.wait(10)
        return reply

    monkeypatch.setattr(alpha_section, "run_calibration", held_terminal)
    try:
        section._on_estimate()
        _wait_until(entered_a.is_set)
        section._stop_btn.click()
        section._on_estimate()
        _wait_until(entered_b.is_set)
        if new_state == "complete":
            _wait_until(lambda: section._worker is None)
        worker_b = section._worker
        draft_b = deepcopy(dialog._draft.to_dict())
        pending_b = dict(section._pending)
        values_b = (dialog._alpha_spin.value(), dialog._beta_section.value())
        controls_b = (
            section._estimate_btn.isEnabled(),
            section._stop_btn.isEnabled(),
            dialog._apply_btn.isEnabled(),
            section._result_label.text(),
        )
        release_a.set()
        _wait_until(lambda: section._tasks.active_count == (1 if new_state == "working" else 0))
        assert section._worker is worker_b
        assert dialog._draft.to_dict() == draft_b
        assert section._pending == pending_b
        assert (dialog._alpha_spin.value(), dialog._beta_section.value()) == values_b
        assert (
            section._estimate_btn.isEnabled(),
            section._stop_btn.isEnabled(),
            dialog._apply_btn.isEnabled(),
            section._result_label.text(),
        ) == controls_b
        release_b.set()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert len(published) == 1
        assert section.pending()["single"].alpha is not None
        dialog._preview_pane.flush()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        dialog._on_apply()
        assert (
            dialog.get_profile_result()["profile"].alpha_policy.value == published[0].outcome.alpha
        )
    finally:
        release_a.set()
        release_b.set()
        dialog._clear_dirty()
        dialog.close()


@pytest.mark.parametrize("terminal", ["success", "error", "cancel"])
@pytest.mark.parametrize("teardown", ["close", "reject", "escape", "parent"])
def test_queued_terminal_after_dialog_teardown_never_publishes(
    qapp, monkeypatch, terminal, teardown
):
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    from asymmetry.gui.tasks import drain_orphan_threads

    parent = QWidget()
    dialog = GroupingDialog([calibration_control()], parent=parent)
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    section._method_combo.setCurrentIndex(section._method_combo.findData("ratio"))
    original = alpha_section.run_calibration
    entered, release, queued = Event(), Event(), Event()
    published = []
    section.result_ready.connect(published.append)

    def hold_computed(worker, request):
        result = original(worker, request)
        entered.set()
        assert release.wait(10)
        if terminal == "error":
            raise ValueError("owned queued failure")
        if terminal == "cancel":
            raise FitCancelledError("owned queued cancellation")
        return result

    monkeypatch.setattr(alpha_section, "run_calibration", hold_computed)
    try:
        section._on_estimate()
        _wait_until(entered.is_set)
        signal = {
            "success": section._worker.finished,
            "error": section._worker.error,
            "cancel": section._worker.cancelled,
        }[terminal]
        signal.connect(lambda *args: queued.set(), Qt.ConnectionType.DirectConnection)
        release.set()
        assert queued.wait(10), "Worker terminal was not queued"
        # No event-loop turn separates the terminal signal from teardown.
        dialog._clear_dirty()
        if teardown == "close":
            dialog.close()
        elif teardown == "reject":
            dialog.reject()
        elif teardown == "escape":
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        else:
            parent.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if isValid(section):
            _wait_until(lambda: section._tasks.active_count == 0)
            assert section.pending() == {}
        drain_orphan_threads()
        qapp.processEvents()
        assert published == []
    finally:
        release.set()
        if isValid(dialog):
            dialog._clear_dirty()
            dialog.close()
        if isValid(parent):
            parent.close()
            parent.deleteLater()


@pytest.mark.parametrize("changed", ["counts", "variance", "clock", "metadata"])
def test_actual_reference_replacement_discards_queued_result(qapp, monkeypatch, changed):
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.project.profiles import BackgroundPolicy
    from tests.core.test_reference_timing import timing_runs

    sample, reference = timing_runs()
    sample.grouping["background_run"] = {
        "run_number": 2,
        "scale_request": {"mode": "manual", "coefficient": 1.0},
    }
    empty = np.empty(0)
    source = MuonDataset(empty, empty, empty, run=sample)
    ref = MuonDataset(empty, empty, empty, run=reference)
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload

    profile = profile_from_payload(
        sample.grouping, "reference", profile_fingerprint_for_run(sample)
    )
    profile.background_policy = BackgroundPolicy(
        mode="reference_run", details={"background_run": sample.grouping["background_run"]}
    )
    section = alpha_section.AlphaSectionWidget()
    section.configure(
        datasets=[source],
        selected_run_number=sample.run_number,
        context_provider=lambda: (
            lambda slot, ds: (
                profile,
                alpha_section.CalibrationReference(sample.grouping["background_run"], (ref,)),
            )
        ),
    )
    section._method_combo.setCurrentIndex(section._method_combo.findData("ratio"))
    entered, release = Event(), Event()
    original = alpha_section.run_calibration
    captured, published = [], []
    section.result_ready.connect(published.append)

    def computed(worker, request):
        result = original(worker, request)
        captured.append(result)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(alpha_section, "run_calibration", computed)
    try:
        section._on_estimate()
        _wait_until(entered.is_set)
        assert captured[0].outcome.alpha is not None
        if changed == "counts":
            reference.histograms[0].counts = reference.histograms[0].counts.copy()
        elif changed == "variance":
            reference.histograms[0].count_variance = reference.histograms[0].counts.copy()
        elif changed == "clock":
            reference.histograms[0].t0_time_us = 0.056
        else:
            reference.metadata["field"] = 2.0
        release.set()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert published == [] and section.pending() == {}
    finally:
        release.set()
        section.shutdown()
