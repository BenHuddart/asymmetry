"""Tests for the shared WizardWindowBase skeleton.

Pins the TaskRunner-driven analysis lifecycle: the request-id staleness
guard, result population on completion, and closeEvent -> TaskRunner.shutdown
cancelling any live worker (AGENTS thread-shutdown invariant).
"""

from __future__ import annotations

import os
import threading
import time

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.usefixtures("qapp")]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QWidget

from asymmetry.gui.tasks import TaskWorker
from asymmetry.gui.windows.wizard_base import WizardWindowBase


def _wait_until(predicate, timeout_ms: int = 30_000) -> None:
    """Run a real nested event loop until *predicate* is true.

    Mirrors tests/test_gui_tasks.py: inter-thread queued signals need the
    loop to be entered, not just poked with processEvents().
    """
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(timeout_ms)
    loop.exec()
    check.stop()
    guard.stop()
    assert predicate(), "timed out waiting for background task"


class _FakeWizard(WizardWindowBase):
    """Minimal concrete subclass exercising every abstract hook."""

    def __init__(self, parent=None, *, task_fn=None):
        self.populated_results: list[object] = []
        self.reset_calls = 0
        self._task_fn = task_fn or (lambda worker: {"value": 42})
        super().__init__(parent)

    def _build_tabs(self) -> None:
        # Self-contained: must not touch state set after super().__init__().
        self._tabs.addTab(QWidget(), "Only")

    def _create_worker_task(self, request_id: int):
        fn = self._task_fn
        return lambda worker: fn(worker)

    def _populate_results(self, result: object) -> None:
        self.populated_results.append(result)

    def _analysis_signature(self) -> dict:
        return {"n": len(self.populated_results)}

    def _reset_result_state(self) -> None:
        self.reset_calls += 1


def test_construction_builds_expected_members():
    window = _FakeWizard()
    try:
        assert window._tasks.active_count == 0
        assert window._tabs.count() == 1
        assert window._analysis_request_id == 0
        assert window._analysis_in_progress is False
        assert window._cached_signature is None
        assert window._cached_log_text == ""
        assert window._current_worker is None
        assert window._progress_bar.isVisible() is False
        assert window._progress_label.isVisible() is False
    finally:
        window.close()


def test_run_analysis_completes_and_populates_results():
    window = _FakeWizard()
    try:
        window._run_analysis()
        assert window._analysis_in_progress is True

        _wait_until(lambda: bool(window.populated_results) and window._tasks.active_count == 0)

        assert window.populated_results == [{"value": 42}]
        assert window._analysis_in_progress is False
        assert window._current_worker is None
        assert window._cached_signature == {"n": 0}
        assert window.reset_calls == 1
    finally:
        window.close()


def test_stale_request_id_is_ignored():
    """A finished callback whose request_id no longer matches must be dropped."""
    release = threading.Event()

    def slow_task(worker: TaskWorker):
        release.wait(10.0)
        return {"value": "first"}

    window = _FakeWizard(task_fn=slow_task)
    try:
        window._run_analysis()
        first_request_id = window._analysis_request_id
        assert window._tasks.active_count == 1

        # Bump the request id (as a second _run_analysis would) without
        # waiting for the first to finish, then let the stale task complete.
        window._analysis_request_id += 1
        release.set()

        _wait_until(lambda: window._tasks.active_count == 0)
        # Give the queued finished-slot delivery a moment to run and confirm
        # it did NOT populate results for the stale request.
        time.sleep(0.05)
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        assert window.populated_results == []
        assert window._analysis_request_id != first_request_id
    finally:
        window.close()


def test_close_event_shuts_down_task_runner():
    """closeEvent cancels a live worker via TaskRunner.shutdown (decision #2).

    The task polls worker.is_cancelled() cooperatively, so shutdown()'s
    cancel-then-wait succeeds quickly instead of timing out and handing the
    thread to the orphan reaper -- exercising the actual cancel-on-close path,
    not just the slow fallback.
    """
    started = threading.Event()

    def cooperative_task(worker: TaskWorker):
        started.set()
        while not worker.is_cancelled():
            time.sleep(0.005)
        return None

    window = _FakeWizard(task_fn=cooperative_task)
    window._run_analysis()
    assert started.wait(10.0)
    assert window._tasks.active_count == 1

    window.close()

    assert window._tasks.active_count == 0
