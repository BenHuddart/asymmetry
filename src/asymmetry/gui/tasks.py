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

from PySide6.QtCore import QObject, QThread, Signal


class TaskCancelledError(Exception):
    """Raise from a task function to signal cooperative cancellation."""


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

        if on_progress is not None:
            worker.progress.connect(on_progress)
        if on_finished is not None:
            worker.finished.connect(on_finished)
        if on_error is not None:
            worker.error.connect(on_error)
        if on_cancelled is not None:
            worker.cancelled.connect(on_cancelled)

        for terminal in (worker.finished, worker.error, worker.cancelled):
            terminal.connect(thread.quit)
            # Processed by the worker thread's event loop just before it quits;
            # after the loop is gone a deleteLater would never run.
            terminal.connect(worker.deleteLater)
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
        for thread, _worker in list(self._live):
            thread.quit()
            if not thread.wait(timeout_ms):
                # Still inside the task (e.g. a long numpy call between cancel
                # polls): unparent so window destruction leaks the thread
                # instead of qFatal-aborting the process.
                thread.setParent(None)
                thread.finished.connect(thread.deleteLater)
        self._live.clear()
