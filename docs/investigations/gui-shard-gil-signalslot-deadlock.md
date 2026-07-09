# Investigation: GUI-shard hang — GIL ↔ Qt signal-slot mutex deadlock

**Status: ROOT CAUSE FOUND AND FIXED (2026-07-09).** The intermittent hang that
wedged a GUI test shard (all xdist workers idle 25+ min, never timing out) is an
**ABBA deadlock between the CPython GIL and Qt's global signal-slot connection
mutex**, triggered by `TaskRunner` destroying its Python-subclassed worker
`QObject` on the worker thread. Not a Qt/macOS defect.

## The stack (live capture)

Captured with `lldb` attached to a hung xdist worker of
`python tools/harness.py validate` (`~/asymmetry-crash-capture/2026-07-09-gui-shard-deadlock-gil-vs-signalslot-mutex.txt`):

- **Main (GUI) thread** — holds the GIL, executing `QLabel.setText` →
  lazy `QWidgetTextControl` construction → `QObject::connectImpl` → blocked in
  `QBasicMutex::lockInternal`, i.e. **waiting for Qt's connection-mutex pool
  lock**.
- **A worker `QThread`** — inside `QThread::exec()`, processing a posted
  `DeferredDelete`: `QObject::~QObject` → `QThreadWrapper::disconnectNotify`
  → shiboken `Sbk_GetPyOverride` → `PyGILState_Ensure`, i.e. **waiting for the
  GIL** while `~QObject` holds the connection lock(s).

Each thread holds what the other wants. Neither can progress.

The `QThreadWrapper::disconnectNotify` frame is the tell: the object being
destroyed is the **`TaskWorker`** (a Python `QObject` subclass), and as
`~QObject` tears down the `thread.started → worker.run` connection it calls
`disconnectNotify` on the *sender* of that connection — the `QThread` — whose
shiboken override lookup needs the GIL.

## Why unbounded

`pytest-timeout` runs with `--timeout-method=thread`: a *Python* watchdog thread
that also needs the GIL. The GUI thread holds the GIL forever, so the watchdog
never fires and the hang is unbounded — the shard stalls until the job-level
`timeout-minutes` kills the runner. (`conftest.py`'s `faulthandler`-based
watchdog, a *C* thread, is what let us dump the stack.)

## Why intermittent

Qt hashes connection objects into a small **pool** of mutexes
(`QBasicMutexPool`). The deadlock needs the GUI thread's `connectImpl` and the
worker's `~QObject` teardown to land on the *same* pooled mutex — a hash
collision, gated further by the two events overlapping in time. Hence the flake:
common enough to wedge CI occasionally, rare enough to pass most runs. A
standalone stress repro (many short-lived workers + a GUI-thread `QLabel.setText`
loop) hung roughly 1 run in 6.

## The trigger in this codebase

`TaskRunner.start` did `worker.moveToThread(thread)` (worker affinity → worker
thread) and connected each terminal signal to `worker.deleteLater` as a
**`QueuedConnection`**. `deleteLater` posts `DeferredDelete` to the object's own
thread → the worker's `~QObject` ran on the **worker thread**. That queued
connection was originally added to dodge a *different* segfault (two concurrent
workers calling `deleteLater` inline during `emit`); it inadvertently created
the deadlock condition.

## Fix (`gui/tasks.py`)

Never destroy a Python `QObject` on a live worker thread. `TaskWorker` captures
its home (GUI) thread at construction and, in `run`'s `finally`, moves its own
affinity back there:

```python
self._home_thread = QThread.currentThread()   # __init__, runs on the GUI thread
...
finally:
    self.moveToThread(self._home_thread)        # run(), on the worker thread — legal
```

`run` executes via a *direct* `started` connection, before `QThread::exec()`, so
the move completes before the thread's event loop even starts. The worker is
then `deleteLater`'d from `_on_thread_finished` (and, for the shutdown-wait path,
from `shutdown`) — both on the GUI thread — so `~QObject` always runs on the GUI
thread, holding the GIL it needs. Routing deletion through the GUI event loop
also serialises it, so the original concurrent-`emit` segfault cannot recur
either. The `worker.deleteLater` `QueuedConnection` is removed.

The orphan/reaper path is unchanged in spirit: a worker that outlives
`shutdown`'s bounded wait is still handed to the process-level reaper running,
and is reclaimed later on the GUI thread once it finishes — a benign leak in the
rare timeout case, never the deadlock.

## Regression coverage

`tests/gui/test_gui_tasks.py::test_worker_destroyed_on_gui_thread` connects a
bare probe to the worker's `destroyed` signal (invoked directly on the
destroying thread) and asserts it fires on the GUI thread. It is deterministic —
it does not depend on the intermittent race — and fails on the pre-fix code
(`[False]`) while passing after (`[True]`).

## Validation

- Standalone stress repro: 10/10 clean after the fix (was ~1/6 hanging).
- No `QObject::moveToThread: Current thread is not the object's thread` warning
  anywhere (which would mean the move silently no-op'd and the fix were inert).
- `tests/gui/test_gui_tasks.py` green 8× serially; a worker-heavy GUI file set
  (185 tests) green 3×; the full GUI subset (`-m gui -n auto`, 2755 tests) green
  twice with no hang.
