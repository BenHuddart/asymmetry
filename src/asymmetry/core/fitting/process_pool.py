"""Shared spawn-safe process-pool helper.

A single place that opens a :class:`~concurrent.futures.ProcessPoolExecutor` with
the ``spawn`` start method. Spawn is required for frozen (PyInstaller) builds and
sidesteps fork-related issues; an environment that cannot start workers (a
restricted sandbox) yields ``None`` so callers fall back to sequential execution
instead of crashing. Both the grouped-fit solver and the global-fit wizard use
this rather than each re-implementing the create/guard dance.

Spawn safety
------------
Under ``spawn``, every worker starts a fresh interpreter that **re-imports the
parent's** ``__main__`` **module** before it can unpickle and run anything. A
plain script whose analysis sits at module level — no
``if __name__ == "__main__":`` guard — therefore re-runs its whole body inside
each worker, which either duplicates the analysis N times over or trips
multiprocessing's own bootstrap ``RuntimeError`` when the re-import in turn tries
to start processes. :func:`spawn_pool_unsafe_reason` detects that (and the
nested case, where a worker's own re-import would spawn further workers), so
:func:`open_spawn_pool` can degrade to serial execution with an actionable
warning rather than crashing. Hosts with no ``__main__`` file at all — an
interactive session, ``python -c ...``, a pytest-xdist worker — are *safe*:
multiprocessing skips the re-import entirely there, so they keep their
parallelism. ``max_workers <= 1`` never starts a worker at all, so it is always
a safe escape hatch.

``max_workers`` does not bound total CPU
----------------------------------------
It bounds *fits in flight*, not threads. A run with ``max_workers=1`` can still
saturate several cores, and measurably does: the superconducting line-shape
kernel forms its spectrum as one complex matrix product
(``weights @ exp(...)`` in :mod:`asymmetry.core.fitting.sc.lineshape`), which
NumPy dispatches to a **multi-threaded BLAS** — so every vortex-lattice
evaluation, of which a fit makes thousands, fans out across the machine from
inside a single process. Measured on a 1500-point transverse-field record with
``max_workers=1``: 153 s wall for 2697 s of CPU, ~17x, essentially all of it in
the two vortex-lattice candidates.

This is not the pool, and no pool setting reaches it. The threading is decided
by the BLAS at load time, so the knob is the environment, before the
interpreter starts::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m your.analysis

On that same record pinning took CPU/wall from ~17 to 1.0 **and wall-clock from
153 s to 55 s** — the oversubscription was costing nearly three times the run,
not buying anything. Bounding it in-process would need either a thread-pool control
dependency the project does not carry or a rewrite of the line-shape kernel to
stop materialising the (n_field x n_time) matrix; both are out of scope here and
neither is a property of the pool, so the *parent* process is documented rather
than patched.

The **workers**, however, are ours to configure: :func:`open_spawn_pool` starts
them with an initializer that pins the BLAS thread-count variables to 1 (see
:mod:`asymmetry._worker_env`). Inside a pool the oversubscription is worst —
``max_workers`` processes each fanning a matrix product across every core — and
no caller can reach the workers' environment from outside. The pin is applied
only to variables the caller has **not** already set, so setting any of them (in
the shell, before the interpreter starts) remains both the parent-side knob
above and the opt-out here: an explicit ``OMP_NUM_THREADS=4`` is honoured in the
workers too.

A pool only when it pays
------------------------
Starting a pool is not free: every spawn worker is a fresh interpreter that
imports NumPy, SciPy, iminuit and the fitting package before its first fit.
:func:`map_when_a_pool_pays` therefore fits in-process until the measured
per-item cost shows a pool would finish the rest sooner, so a batch of a few
cheap fits never starts a process at all.
"""

from __future__ import annotations

import ast
import math
import multiprocessing as mp
import sys
import time
import warnings
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache

from asymmetry._worker_env import blas_thread_pins, pin_worker_blas_threads
from asymmetry.core.fitting.engine import FitCancelledError

#: Wall-clock seconds for a spawn pool's workers to become ready: each imports the
#: fitting stack from scratch (1.4–2.3 s measured on an Apple-silicon laptop, and
#: ten times that on a host saturated by other processes).
POOL_STARTUP_S = 2.0


class SpawnUnsafeWarning(UserWarning):
    """Warned when parallel work degrades to serial for spawn-safety reasons."""


_SPAWN_SAFETY_WARNED = False


def reset_spawn_safety_warning() -> None:
    """Re-arm the once-per-process spawn-safety warning (tests use this)."""
    global _SPAWN_SAFETY_WARNED
    _SPAWN_SAFETY_WARNED = False


@lru_cache(maxsize=8)
def main_module_has_spawn_guard(path: str) -> bool:
    """Does the ``__main__`` script at *path* carry an ``if __name__`` guard?

    Answered by parsing the source, because that is the only way to know whether
    a spawn worker's re-import of this module will re-run the caller's analysis.
    Anything we cannot positively rule out counts as guarded: an unreadable or
    unparseable ``__main__`` (a frozen build, a zipapp, a generated entry point)
    must keep its parallelism, so the check only ever *demotes* a file whose
    source it has actually read and found unguarded.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return True

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for child in ast.walk(node.test):
            if isinstance(child, ast.Name) and child.id == "__name__":
                return True
    return False


def spawn_pool_unsafe_reason() -> str | None:
    """Why spawn workers cannot be started safely here, or ``None`` if they can.

    The returned text is user-facing: it names the problem and the fix.
    """
    if getattr(mp.current_process(), "_inheriting", False):
        return (
            "this process is a spawn worker re-importing __main__, so starting "
            "further workers here would cascade"
        )

    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", None)
    if main_path is None:
        # No file means nothing for the worker to re-import (an interactive
        # session, ``python -c ...``, a stdin script, a pytest-xdist worker):
        # multiprocessing skips the ``__main__`` re-import entirely, so the
        # workers are safe as long as the submitted callables are importable.
        return None
    if not main_module_has_spawn_guard(str(main_path)):
        return (
            f"the entry-point script {main_path} has no "
            '`if __name__ == "__main__":` guard, so every spawn worker would '
            "re-run it from the top"
        )
    return None


def open_spawn_pool(max_workers: int) -> ProcessPoolExecutor | None:
    """Return a spawn-context process pool, or ``None`` when one cannot start.

    ``None`` signals the caller to run sequentially (identical results, no
    parallelism). It is returned for the environmental failures a constrained or
    frozen host raises at pool construction, for ``max_workers <= 1`` (the
    caller asked for no parallelism, so no worker process is started — the
    guaranteed escape hatch), and for the spawn-safety cases
    :func:`spawn_pool_unsafe_reason` detects, which additionally warn once per
    process with :class:`SpawnUnsafeWarning`.

    Workers start with the BLAS thread-count pin described in the module
    docstring; a caller that has set any of those variables itself gets no
    initializer and keeps its own configuration.
    """
    if int(max_workers) <= 1:
        return None

    reason = spawn_pool_unsafe_reason()
    if reason is not None:
        global _SPAWN_SAFETY_WARNED
        if not _SPAWN_SAFETY_WARNED:
            _SPAWN_SAFETY_WARNED = True
            warnings.warn(
                f"Running serially instead of in parallel: {reason}. "
                'Wrap the entry point in `if __name__ == "__main__":` (or pass '
                "max_workers=1 to make the serial run explicit) to silence this.",
                SpawnUnsafeWarning,
                stacklevel=2,
            )
        return None

    pins = blas_thread_pins()
    worker_setup: dict[str, object] = (
        {"initializer": pin_worker_blas_threads, "initargs": (pins,)} if pins else {}
    )
    try:
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            **worker_setup,  # type: ignore[arg-type]
        )
    except (OSError, PermissionError, ValueError):
        return None


def terminate_spawn_pool(pool: ProcessPoolExecutor) -> None:
    """Tear a spawn pool down *now*, without waiting for in-flight tasks.

    Drops queued work (``cancel_futures=True``, ``wait=False``) and then
    force-kills the worker processes, reaping each so a cancelled run leaves no
    orphaned spawn workers *and* no zombies (a bare ``kill()`` without a
    following ``join()`` leaves the killed child unreaped until someone
    ``waitpid``s it). Use this on the cancellation path where a blocking
    ``shutdown(wait=True)`` would stall the UI for one in-flight fit's duration;
    the normal completion path still calls plain :meth:`shutdown`.

    Best-effort on the private ``_processes`` map — a missing/renamed attribute
    (or a non-real pool, e.g. a test fake) degrades to the plain non-blocking
    shutdown. ``_processes`` is snapshotted before shutdown because shutdown may
    clear it.
    """
    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for proc in processes:
        try:
            proc.kill()
        except (OSError, ValueError, AttributeError):
            pass
    for proc in processes:
        try:
            # Reap the killed child so it does not linger as a zombie. The
            # executor's own wind-down may race us to it, so tolerate an
            # already-reaped process. The timeout only binds when the child is
            # slow to die after SIGKILL (a starved host); join returns as soon
            # as the process is reaped, typically milliseconds.
            proc.join(timeout=5.0)
        except (ChildProcessError, OSError, ValueError, AttributeError):
            pass


def map_when_a_pool_pays(
    worker: Callable[..., tuple[Hashable, object]],
    payloads: Sequence[object],
    *,
    workers: int,
    cancel_callback: Callable[[], bool] | None,
) -> dict[Hashable, object]:
    """Run ``worker(payload)`` over *payloads*, return ``{key: result}`` from its pairs.

    Payloads run in-process until a pool would finish the rest sooner. After ``n``
    payloads in ``elapsed`` seconds, the ``r`` left take ``m * r`` serially
    (``m = elapsed / n``) against ``POOL_STARTUP_S + m * ceil(r / workers)`` on
    ``workers`` processes; once the second is smaller, the rest go to a spawn pool.
    Whatever a pool does not return — none could start, or one broke — runs
    in-process too, so the result never depends on where a payload ran.

    *worker* is module-level so it survives ``spawn``. In-process it is called
    with ``cancel_callback=``; across the boundary cancellation is coarse — a
    requested cancel stops collecting and tears the pool down at once.
    """
    results: dict[Hashable, object] = {}

    def run_here(payload: object) -> None:
        if cancel_callback is not None and cancel_callback():
            raise FitCancelledError("Fit cancelled.")
        key, result = worker(payload, cancel_callback=cancel_callback)
        results[key] = result

    pending = deque(payloads)
    started = time.perf_counter()
    while pending:
        if results:
            per_item = (time.perf_counter() - started) / len(results)
            left = len(pending)
            if per_item * (left - math.ceil(left / workers)) > POOL_STARTUP_S:
                break
        run_here(pending.popleft())
    leftover = list(pending)
    executor = open_spawn_pool(min(workers, len(leftover))) if leftover else None
    if executor is not None:
        futures = {executor.submit(worker, payload): payload for payload in leftover}
        try:
            for future in as_completed(futures):
                if cancel_callback is not None and cancel_callback():
                    terminate_spawn_pool(executor)
                    raise FitCancelledError("Fit cancelled.")
                key, result = future.result()
                results[key] = result
                del futures[future]
        except BrokenExecutor:
            # A worker died for an environmental reason (a failed fit returns
            # success=False without raising); what it left runs in-process below.
            pass
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        leftover = list(futures.values())
    for payload in leftover:
        run_here(payload)
    return results
