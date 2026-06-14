"""Shared background-task infrastructure for the GUI.

Engineering invariant: long-running core work (file loading, fitting, MaxEnt,
run combination, spectra) never runs on the GUI thread. Run it through
:class:`TaskRunner`, which owns the QThread/worker lifecycle so call sites
don't repeat it — and don't get it subtly wrong.

The lifecycle rules encoded here are load-bearing, especially on Windows:

- Strong references to every live ``(QThread, TaskWorker)`` pair are held
  until the thread finishes. A garbage-collected live ``QThread`` aborts the
  process.
- Cancellation is cooperative only (a polled flag) — never
  ``QThread.terminate()``.
- Results cross back to the GUI thread as plain Python objects via queued
  signal connections. Workers must never touch ``QWidget`` or matplotlib.
- On window close, :meth:`TaskRunner.shutdown` cancels, quits and waits;
  a timed-out wait degrades to a leaked (unparented) thread instead of the
  qFatal that destroying a running parented ``QThread`` triggers.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QCoreApplication, QObject, QThread, Qt, Signal


class TaskCancelledError(Exception):
    """Raise from a task function to signal cooperative cancellation."""


class _OrphanThreadReaper(QObject):
    """Process-level keep-alive for worker threads that timed out on a bounded wait.

    Destroying a *running* QThread qFatal-aborts the process, so a thread we
    could not join must be kept alive until it genuinely finishes — and the
    keep-alive must outlive any owning TaskRunner/window (closeEvent destroys
    both). This reaper lives on the GUI thread for the life of the process, so
    ``finished`` (emitted in the worker thread) is delivered to :meth:`_reap`
    as a queued connection — pruning never races the worker thread. A thread
    that never finishes simply stays referenced here, the deliberate trade for
    not aborting.
    """

    def __init__(self) -> None:
        super().__init__()
        self._threads: list[tuple[QThread, object]] = []

    def adopt(self, thread: QThread, worker: object | None) -> None:
        self._threads.append((thread, worker))
        thread.finished.connect(self._reap)

    def _reap(self) -> None:
        # sender() can return None in queued cross-thread connections on some
        # PySide6 builds, leaving stale entries in _threads. Scan all entries
        # and prune any thread that is no longer running instead: Qt sets
        # running=False before emitting finished(), so isRunning() is reliable
        # here even when called from a queued slot.
        #
        # Guard against RuntimeError: if _on_thread_finished fired first and
        # called deleteLater(), the event loop may have destroyed the C++ object
        # before this queued slot ran. Skip those entries — nothing left to do.
        remaining = []
        for t, w in self._threads:
            try:
                running = t.isRunning()
            except RuntimeError:
                continue
            if running:
                remaining.append((t, w))
            else:
                t.deleteLater()
        self._threads = remaining


_orphan_reaper: _OrphanThreadReaper | None = None


def retire_thread(thread: QThread, worker: object | None = None) -> None:
    """Keep a still-running QThread alive until it finishes, then delete it.

    Call after a bounded ``wait()`` times out and the caller must drop its own
    reference (window teardown, or reusing a one-thread slot). The reference
    held by the reaper is process-level, so neither GC nor parent destruction
    can free the C++ thread mid-run. Call from the GUI thread.
    """
    global _orphan_reaper
    if _orphan_reaper is None:
        _orphan_reaper = _OrphanThreadReaper()
        app = QCoreApplication.instance()
        if app is not None:
            # Re-parent to the application so the reaper outlives every window
            # but is still destroyed at process exit; keeps GUI-thread affinity.
            _orphan_reaper.setParent(app)
    _orphan_reaper.adopt(thread, worker)


class TaskWorker(QObject):
    """Runs one callable off the GUI thread and emits exactly one terminal signal.

    The callable receives this worker as its only argument and may use
    ``worker.progress.emit(current, total, message)`` and
    ``worker.is_cancelled()`` — the same shapes the core engines accept as
    ``progress_callback`` / ``cancel_callback``.
    """

    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        fn: Callable[[TaskWorker], object],
        *,
        cancel_exceptions: tuple[type[BaseException], ...] = (),
    ) -> None:
        super().__init__()
        self._fn = fn
        self._cancel_requested = False
        self._cancel_exceptions: tuple[type[BaseException], ...] = (
            TaskCancelledError,
            *cancel_exceptions,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation (safe from any thread)."""
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        try:
            result = self._fn(self)
        except self._cancel_exceptions:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)


class _TaskRelay(QObject):
    """GUI-thread relay between a worker's signals and plain-Python callbacks.

    A signal connected to a bare function, lambda or ``functools.partial``
    has no receiver QObject, so Qt delivers it DIRECTLY — the callback runs
    on the worker thread. Routing through this relay's bound methods (it
    lives on the GUI thread) makes every delivery queued, so caller
    callbacks may safely touch widgets.
    """

    def __init__(
        self,
        parent: QObject,
        *,
        on_finished: Callable[[object], None] | None,
        on_error: Callable[[str], None] | None,
        on_cancelled: Callable[[], None] | None,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> None:
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_error = on_error
        self._on_cancelled = on_cancelled
        self._on_progress = on_progress

    def finished(self, result: object) -> None:
        if self._on_finished is not None:
            self._on_finished(result)

    def error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)

    def cancelled(self) -> None:
        if self._on_cancelled is not None:
            self._on_cancelled()

    def progress(self, current: int, total: int, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(current, total, message)


class TaskRunner(QObject):
    """Owns background tasks for one window; create it parented to the window.

    Typical use::

        worker = self._tasks.start(
            lambda w: maxent(run, config,
                             progress_callback=w.progress.emit,
                             cancel_callback=w.is_cancelled),
            on_finished=self._on_done,
            on_error=self._on_error,
            on_cancelled=self._on_cancelled,
            cancel_exceptions=(MaxEntCancelledError,),
        )
        ...
        worker.cancel()  # e.g. from a Stop button

    Call :meth:`shutdown` from the window's ``closeEvent``.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._live: list[tuple[QThread, TaskWorker]] = []

    def start(
        self,
        fn: Callable[[TaskWorker], object],
        *,
        on_finished: Callable[[object], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        cancel_exceptions: tuple[type[BaseException], ...] = (),
    ) -> TaskWorker:
        """Start *fn* on a fresh worker thread and return its worker handle."""
        thread = QThread(self)
        worker = TaskWorker(fn, cancel_exceptions=cancel_exceptions)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # All caller callbacks go through a GUI-thread relay: connecting a
        # plain callable directly would run it on the worker thread.
        relay = _TaskRelay(
            self,
            on_finished=on_finished,
            on_error=on_error,
            on_cancelled=on_cancelled,
            on_progress=on_progress,
        )
        worker.progress.connect(relay.progress)
        worker.finished.connect(relay.finished)
        worker.error.connect(relay.error)
        worker.cancelled.connect(relay.cancelled)

        for terminal in (worker.finished, worker.error, worker.cancelled):
            terminal.connect(thread.quit)
            # Queued so deleteLater() is posted to the worker thread's event loop
            # and runs after emit() returns, not synchronously during it. A direct
            # connection would call deleteLater() inline while two concurrent workers
            # are both inside emit(), causing concurrent PySide6 signal-state access
            # and a segfault. The event loop processes this before quit() exits it.
            terminal.connect(worker.deleteLater, Qt.ConnectionType.QueuedConnection)
        thread.finished.connect(relay.deleteLater)
        # Bound-method slot on a GUI-thread QObject => queued connection, so
        # bookkeeping never mutates _live from the worker thread.
        thread.finished.connect(self._on_thread_finished)

        self._live.append((thread, worker))
        thread.start()
        return worker

    def _on_thread_finished(self) -> None:
        thread = self.sender()
        self._live = [(t, w) for (t, w) in self._live if t is not thread]
        if isinstance(thread, QThread):
            thread.deleteLater()

    @property
    def active_count(self) -> int:
        return len(self._live)

    def shutdown(self, timeout_ms: int = 10_000) -> None:
        """Cancel all tasks and wait for their threads (call from closeEvent)."""
        for _thread, worker in list(self._live):
            try:
                worker.cancel()
            except RuntimeError:
                # The worker's C++ object can already be gone if its terminal
                # signal fired but the queued bookkeeping hasn't run yet.
                pass
        for thread, worker in list(self._live):
            thread.quit()
            if not thread.wait(timeout_ms):
                # Still inside the task (e.g. a long numpy call between cancel
                # polls). Unparent it from this runner and hand it to the
                # process-level keep-alive: clearing _live below drops our
                # only other reference, and GC-ing a running QThread aborts.
                #
                # Disconnect our own _on_thread_finished slot before handing
                # off: the reaper's _reap will own cleanup, and leaving this
                # connection would cause _on_thread_finished to call
                # deleteLater() independently — potentially destroying the C++
                # object before _reap gets to call isRunning() on it.
                try:
                    thread.finished.disconnect(self._on_thread_finished)
                except RuntimeError:
                    pass  # already disconnected (signal already fired)
                thread.setParent(None)
                retire_thread(thread, worker)
        self._live.clear()
