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
