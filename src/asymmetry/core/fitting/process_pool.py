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
neither is a property of the pool, so this is documented rather than patched.
"""

from __future__ import annotations

import ast
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache


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

    try:
        return ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
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
