r"""Garbage-collector settling for a data-heavy, long-lived GUI session.

A loaded session holds a few hundred thousand collector-tracked objects
(datasets, metadata dicts, matplotlib artists, Qt wrappers). CPython's
generational collector re-scans *all* of them on every full (generation-2)
collection, which the run-switch allocations trigger every couple of
switches: a 35–110 ms pause on the GUI thread, landing at random inside a
selection change and read by the user as a stutter.

:func:`settle_memory` runs one full collection and then :func:`gc.freeze`\ s
everything still alive, moving it to the permanent generation the collector
never scans again; subsequent full collections cost ~0 ms. It is called
after every bulk data arrival or departure (project open, file loads,
clear-all) via ``MainWindow._schedule_memory_settle``; each call first
unfreezes, so objects frozen in an earlier settle that have since died are
reclaimed rather than leaking. Reference counting is unaffected — frozen
objects are still freed the moment their last reference goes — only the
cycle collector's scanning is skipped.

Set ``ASYMMETRY_GC_FREEZE=0`` to disable the settle entirely.
"""

from __future__ import annotations

import gc
import os

__all__ = ["settle_memory", "gc_freeze_enabled"]

_ENV_VAR = "ASYMMETRY_GC_FREEZE"


def gc_freeze_enabled() -> bool:
    """Return whether :func:`settle_memory` freezes the surviving objects."""
    return os.environ.get(_ENV_VAR, "1").strip().lower() not in ("0", "false", "no", "off")


def settle_memory() -> dict[str, int] | None:
    """Collect cyclic garbage now and freeze the survivors out of future scans.

    Returns ``{"collected": n, "frozen": m}`` for perf logging, or ``None``
    when disabled.
    """
    if not gc_freeze_enabled():
        return None
    gc.unfreeze()
    collected = int(gc.collect())
    gc.freeze()
    return {"collected": collected, "frozen": int(gc.get_freeze_count())}
