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

from PySide6.QtCore import QCoreApplication, QObject, Qt, QThread, Signal


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

    def join_running(self, timeout_ms: int) -> int:
        """Cancel + bounded-join every still-running adopted thread.

        Returns the count that were still running on entry. **Test-teardown
        hygiene only** — see :func:`drain_orphan_threads`. Production keeps
        orphaned threads running on purpose (destroying a running ``QThread``
        aborts the process), so the app never calls this; a test fixture does,
        so a leaked worker finishes during teardown instead of contending with
        the next test. The ``cancel()`` helps workers that poll
        ``is_cancelled()``; the rest are simply waited out (bounded).
        """
        running = 0
        for thread, worker in list(self._threads):
            try:
                if not thread.isRunning():
                    continue
            except RuntimeError:
                continue
            running += 1
            try:
                if worker is not None:
                    worker.cancel()
            except (RuntimeError, AttributeError):
                pass
            try:
                # quit() before wait(): the worker QThread runs an event loop, so
                # it only finishes once quit() breaks it. quit() is safe to call
                # cross-thread; without it wait() would block until the timeout
                # because the finished->quit relay is queued to this (blocked)
                # GUI thread. Mirrors TaskRunner.shutdown().
                thread.quit()
                thread.wait(timeout_ms)
            except RuntimeError:
                pass
        return running


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


def drain_orphan_threads(timeout_ms: int = 5000) -> int:
    """Cancel + bounded-join any orphaned worker threads the reaper holds.

    Test-support hook for the autouse Qt-cleanup fixture. A worker whose
    ``TaskRunner`` was destroyed without ``shutdown()`` (a panel GC'd at the end
    of a test), or one that timed out on ``shutdown``'s bounded wait, is handed
    to the process-level :data:`_orphan_reaper`, which deliberately keeps it
    **running** until it finishes on its own (destroying a running ``QThread``
    aborts the process). Left alone, that lingering compute bleeds CPU into the
    next test and makes GUI tests flaky under the sharded, 2-core CI runners.
    This joins them at teardown instead.

    Cheap no-op when nothing has been orphaned (the common case). Returns the
    number of threads that were still running. Not called by the app itself —
    production keeps the keep-alive semantics unchanged.
    """
    if _orphan_reaper is None:
        return 0
    return _orphan_reaper.join_running(timeout_ms)


def _retire_running_threads(live: list[tuple[QThread, TaskWorker]]) -> None:
    """Safety net for a :class:`TaskRunner` destroyed without ``shutdown()``.

    A runner created with ``TaskRunner(parent)`` owns its worker ``QThread``\\ s
    as QObject children (``QThread(self)``). When the parent is destroyed via
    C++ destruction rather than ``close()``/``done()`` — e.g. a parented *child*
    dialog reaped through its parent's destruction, or a test's teardown
    ``deleteLater``-ing a window — the runner's ``closeEvent``/``done`` never
    run, so ``shutdown()`` is never called, and a still-running child
    ``QThread`` would be destroyed with the runner. Destroying a running
    ``QThread`` qFatal-aborts the process.

    ``destroyed`` is emitted from ``~QObject`` *before* children are deleted, so
    a slot connected to ``TaskRunner.destroyed`` can still reach the live
    threads here. Each still-running thread is unparented from the dying runner
    (so ``~QObject`` no longer owns it) and handed to the process-level reaper —
    the exact hand-off :meth:`TaskRunner.shutdown` uses for a timed-out wait.

    The caller passes the runner's ``_live`` list (never the runner itself), so
    this touches no half-destroyed wrapper and cannot form a GC cycle that
    delays collecting a parentless runner. It calls nothing that pumps the event
    loop, so ``~QObject``'s auto-disconnect of the runner's own connections
    (right after this returns) still races nothing.
    """
    for thread, worker in list(live):
        try:
            running = thread.isRunning()
        except RuntimeError:
            # C++ side already gone (mid-teardown); nothing to retire.
            continue
        if not running:
            continue
        try:
            worker.cancel()
        except (RuntimeError, AttributeError):
            pass
        try:
            thread.setParent(None)
        except RuntimeError:
            continue
        retire_thread(thread, worker)
    live.clear()


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
        # Safety net for destruction WITHOUT shutdown() (a parented child dialog
        # reaped through its parent, or a test deleteLater-ing a window): the
        # slot captures the _live *list object* — never self — so it neither
        # touches the half-destroyed wrapper nor forms a GC cycle that would
        # delay collecting a parentless runner. This REQUIRES _live to be
        # mutated in place forever (see _on_thread_finished); reassigning it
        # would silently orphan this net. See _retire_running_threads.
        _live = self._live
        self.destroyed.connect(lambda: _retire_running_threads(_live))

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
        # Mutate _live IN PLACE (never rebind): the destroyed-safety-net slot in
        # __init__ captures this exact list object, so reassigning self._live
        # would silently orphan it and let a running thread be destroyed with us.
        self._live[:] = [(t, w) for (t, w) in self._live if t is not thread]
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
        # Flush pending queued MetaCall events that target this runner (e.g.
        # the cross-thread _on_thread_finished delivery posted by thread.finished).
        # Without this drain, if the caller drops the last reference immediately
        # after shutdown() the GC destroys the C++ receiver while those events
        # are still queued; the next QEventLoop turn tries to invoke a deleted
        # slot and segfaults.  Calling sendPostedEvents(self, 0) while the
        # runner is still alive processes them safely before it can be collected.
        app = QCoreApplication.instance()
        if app is not None:
            app.sendPostedEvents(self, 0)
