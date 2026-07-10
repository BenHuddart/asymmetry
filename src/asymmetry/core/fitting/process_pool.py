"""Shared spawn-safe process-pool helper.

A single place that opens a :class:`~concurrent.futures.ProcessPoolExecutor` with
the ``spawn`` start method. Spawn is required for frozen (PyInstaller) builds and
sidesteps fork-related issues; an environment that cannot start workers (a
restricted sandbox) yields ``None`` so callers fall back to sequential execution
instead of crashing. Both the grouped-fit solver and the global-fit wizard use
this rather than each re-implementing the create/guard dance.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def open_spawn_pool(max_workers: int) -> ProcessPoolExecutor | None:
    """Return a spawn-context process pool, or ``None`` when one cannot start.

    ``None`` signals the caller to run sequentially (identical results, no
    parallelism); it is returned for the environmental failures a constrained or
    frozen host raises at pool construction.
    """
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
