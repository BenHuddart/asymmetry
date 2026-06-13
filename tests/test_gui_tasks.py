"""TaskRunner / TaskWorker lifecycle tests.

These pin the threading contract every background feature relies on:
exactly one terminal signal per task, results marshalled to the GUI thread,
cooperative cancellation (generic and engine-specific exception types), and
a shutdown that never leaves live threads behind.
"""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtCore import QEventLoop, QThread, QTimer

from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker

pytestmark = pytest.mark.usefixtures("qapp")


def _wait_until(predicate, timeout_ms: int = 10_000) -> None:
    """Run a real nested event loop until *predicate* is true.

    Inter-thread queued signals need the loop to be entered, not just poked
    with processEvents().
    """
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    check.stop()
    assert predicate(), "timed out waiting for background task"


def test_finished_result_marshalled_to_gui_thread():
    runner = TaskRunner()
    results: list[object] = []
    task_thread: list[QThread] = []

    def fn(worker: TaskWorker):
        task_thread.append(QThread.currentThread())
        return {"answer": 42}

    runner.start(fn, on_finished=results.append)
    _wait_until(lambda: bool(results) and runner.active_count == 0)

    assert results == [{"answer": 42}]
    assert task_thread[0] is not QThread.currentThread()
    runner.shutdown()


def test_error_emits_message_not_finished():
    runner = TaskRunner()
    errors: list[str] = []
    results: list[object] = []

    def fn(worker: TaskWorker):
        raise ValueError("bad alpha")

    runner.start(fn, on_finished=results.append, on_error=errors.append)
    _wait_until(lambda: bool(errors) and runner.active_count == 0)

    assert errors == ["bad alpha"]
    assert results == []
    runner.shutdown()


def test_cooperative_cancel_generic_and_custom_exception():
    class EngineCancelledError(Exception):
        pass

    for raise_type in (TaskCancelledError, EngineCancelledError):
        runner = TaskRunner()
        cancelled: list[bool] = []
        started = threading.Event()

        def fn(worker: TaskWorker, _raise=raise_type):
            started.set()
            while not worker.is_cancelled():
                time.sleep(0.005)
            raise _raise()

        worker = runner.start(
            fn,
            on_cancelled=lambda: cancelled.append(True),
            cancel_exceptions=(EngineCancelledError,),
        )
        assert started.wait(10.0)
        worker.cancel()
        _wait_until(lambda: bool(cancelled) and runner.active_count == 0)
        assert cancelled == [True]
        runner.shutdown()


def test_progress_signals_arrive_in_order():
    runner = TaskRunner()
    seen: list[tuple[int, int, str]] = []
    done: list[object] = []

    def fn(worker: TaskWorker):
        for i in range(5):
            worker.progress.emit(i + 1, 5, f"step {i + 1}")
        return None

    runner.start(
        fn,
        on_progress=lambda c, t, m: seen.append((c, t, m)),
        on_finished=done.append,
    )
    _wait_until(lambda: bool(done) and runner.active_count == 0)

    assert seen == [(i + 1, 5, f"step {i + 1}") for i in range(5)]
    runner.shutdown()


def test_plain_callable_callbacks_run_on_gui_thread():
    """Lambdas/closures passed as callbacks must be relayed to the GUI thread.

    A signal connected directly to a bare callable has no receiver QObject and
    is delivered in the emitting (worker) thread — the classic cross-thread
    widget-access bug. TaskRunner's relay must prevent it by construction.
    """
    runner = TaskRunner()
    gui_thread = QThread.currentThread()
    callback_threads: list[QThread] = []
    done: list[object] = []

    def fn(worker: TaskWorker):
        worker.progress.emit(1, 1, "tick")
        return "ok"

    runner.start(
        fn,
        on_progress=lambda *_: callback_threads.append(QThread.currentThread()),
        on_finished=lambda result: (
            callback_threads.append(QThread.currentThread()),
            done.append(result),
        ),
    )
    _wait_until(lambda: bool(done) and runner.active_count == 0)

    assert done == ["ok"]
    assert callback_threads and all(t is gui_thread for t in callback_threads)
    runner.shutdown()


def test_shutdown_cancels_running_task():
    runner = TaskRunner()
    started = threading.Event()
    outcomes: list[str] = []

    def fn(worker: TaskWorker):
        started.set()
        while not worker.is_cancelled():
            time.sleep(0.005)
        raise TaskCancelledError()

    runner.start(fn, on_cancelled=lambda: outcomes.append("cancelled"))
    assert started.wait(10.0)
    runner.shutdown(timeout_ms=10_000)

    assert runner.active_count == 0
    # The cancelled signal may or may not be delivered after shutdown
    # (the receiver loop is still alive here, so normally it is) — the hard
    # requirement is that shutdown returned with no live threads.


def test_two_tasks_run_concurrently_and_both_finish():
    runner = TaskRunner()
    results: list[str] = []
    barrier = threading.Barrier(2, timeout=10.0)

    def make(tag: str):
        def fn(worker: TaskWorker):
            # Both tasks must be inside their threads at once to pass.
            barrier.wait()
            return tag

        return fn

    runner.start(make("a"), on_finished=results.append)
    runner.start(make("b"), on_finished=results.append)
    _wait_until(lambda: len(results) == 2 and runner.active_count == 0)

    assert sorted(results) == ["a", "b"]
    runner.shutdown()


def test_shutdown_keeps_unjoinable_thread_alive_until_it_finishes():
    """A worker that outlasts the shutdown wait is parked, not dropped.

    Dropping the last Python reference to a still-running QThread lets GC
    destroy the C++ object mid-run, which qFatal-aborts the process — the very
    failure the unparent/keep-alive path exists to prevent. The reaper must
    hold the thread until it genuinely finishes, then prune it.
    """
    from asymmetry.gui import tasks as tasks_mod

    runner = TaskRunner()
    started = threading.Event()
    release = threading.Event()

    def fn(worker: TaskWorker):
        started.set()
        # Ignores cancellation; only stops when the test releases it. Models a
        # worker wedged between cancel polls when shutdown's wait times out.
        release.wait(10.0)
        return "done"

    runner.start(fn)
    assert started.wait(10.0)

    reaper_before = tasks_mod._orphan_reaper
    parked_before = len(reaper_before._threads) if reaper_before is not None else 0

    # Short timeout so wait() expires while the worker is still running.
    runner.shutdown(timeout_ms=200)
    assert runner.active_count == 0

    reaper = tasks_mod._orphan_reaper
    assert reaper is not None
    # The running thread is held by the reaper (strong ref), not dropped.
    assert len(reaper._threads) == parked_before + 1

    # Let it finish; the reaper prunes it on the GUI thread via finished.
    release.set()
    _wait_until(lambda: len(reaper._threads) == parked_before)
