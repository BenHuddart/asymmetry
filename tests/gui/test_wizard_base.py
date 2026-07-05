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
from PySide6.QtWidgets import QTabWidget, QWidget

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
        # Record where the tab widget sits in the central layout at hook time:
        # the real subclasses insert banners relative to it and append nav rows
        # beneath it, so it must already be attached when this hook runs.
        self.tabs_index_during_build = self._central_layout.indexOf(self._tabs)
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
        assert isinstance(window._tabs, QTabWidget)
        assert window._tabs.count() == 1
        # Default _build_central attaches the tabs beneath the chrome
        # ([heading, status, controls, tabs] -> index 3) BEFORE _build_tabs
        # runs, so subclass hooks can position widgets relative to it.
        assert window.tabs_index_during_build == 3
        assert window._central_layout.indexOf(window._tabs) == 3
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


def test_stale_error_clears_busy():
    """A stale ERROR callback must still clear busy (regression: soft-lock).

    Mirrors _handle_finished: if the context changes mid-run (the id is bumped
    while busy stays True, as set_analysis_context does on its in-progress
    branch) and the now-stale worker then raises, the window must not stay
    soft-locked with _analysis_in_progress stuck True.
    """
    release = threading.Event()

    def slow_error_task(worker: TaskWorker):
        release.wait(10.0)
        raise RuntimeError("boom")

    window = _FakeWizard(task_fn=slow_error_task)
    try:
        window._run_analysis()
        assert window._analysis_in_progress is True

        # Context changed mid-run: bump the id but leave busy True, then let the
        # stale task raise.
        window._analysis_request_id += 1
        release.set()

        _wait_until(lambda: window._tasks.active_count == 0)
        time.sleep(0.05)
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()

        assert window._analysis_in_progress is False
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


class _CentralOverrideWizard(WizardWindowBase):
    """Minimal subclass overriding _build_central with a plain QWidget.

    Deliberately does NOT implement _build_tabs: on the override path the
    base must never call it, so its absence must be safe.
    """

    def __init__(self, parent=None, *, task_fn=None):
        self.populated_results: list[object] = []
        self.reset_calls = 0
        self._task_fn = task_fn or (lambda worker: {"value": "central"})
        super().__init__(parent)

    def _build_central(self) -> QWidget:
        self._content = QWidget()
        return self._content

    def _create_worker_task(self, request_id: int):
        fn = self._task_fn
        return lambda worker: fn(worker)

    def _populate_results(self, result: object) -> None:
        self.populated_results.append(result)

    def _analysis_signature(self) -> dict:
        return {"n": len(self.populated_results)}

    def _reset_result_state(self) -> None:
        self.reset_calls += 1


def test_build_central_override_constructs_without_tabs():
    """Override path: no QTabWidget exists anywhere, _tabs stays None."""
    window = _CentralOverrideWizard()
    try:
        assert window._tabs is None
        assert window.findChildren(QTabWidget) == []
        # __init__ attached the returned content widget beneath the chrome
        # ([heading, status, controls, content] -> index 3).
        assert window._central_layout.indexOf(window._content) == 3
        assert window._content.parentWidget() is window.centralWidget()
    finally:
        window.close()


def test_build_central_override_runs_analysis_lifecycle():
    """The analysis mechanism is content-region-agnostic."""
    window = _CentralOverrideWizard()
    try:
        window._run_analysis()
        assert window._analysis_in_progress is True
        assert window._analysis_request_id == 1

        _wait_until(lambda: bool(window.populated_results) and window._tasks.active_count == 0)

        assert window.populated_results == [{"value": "central"}]
        assert window._analysis_in_progress is False
        assert window._current_worker is None
        assert window._cached_signature == {"n": 0}
        assert window.reset_calls == 1
    finally:
        window.close()


def test_default_path_still_requires_build_tabs():
    """Without a _build_central override, _build_tabs stays abstract."""

    class _NoBuildTabs(WizardWindowBase):
        def _create_worker_task(self, request_id: int):
            return lambda worker: None

        def _populate_results(self, result: object) -> None:
            pass

        def _analysis_signature(self) -> dict:
            return {}

        def _reset_result_state(self) -> None:
            pass

    with pytest.raises(NotImplementedError):
        _NoBuildTabs()
