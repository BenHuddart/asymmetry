"""GUI-layer memo cache for the two uncached time-domain reductions.

Dataset switching and view toggles re-run two reductions on every paint:

* ``build_grouped_time_domain_datasets`` (Groups / Raw-counts view), and
* the ``bunch > 1`` counts-first re-reduction (2--4x per render).

Both are pure functions of a run's histograms plus its grouping recipe, so a
switch back to a run/view already computed can reuse the result instead of
recomputing (~1 s reduce / ~0.4 s grouped build at 128x1M). :class:`ReductionCache`
is a byte-budgeted LRU keyed by run identity + a caller-supplied recipe key that
must embed everything the wrapped function reads, so a grouping edit never serves
stale data (see the wiring in ``mainwindow.py`` for the exact keys).

Design notes:

* **Keying is correctness by construction.** The caller builds the key from the
  grouping digest plus every field the reduction reads that the digest omits
  (alpha, forward/backward group, deadtime mode, per-detector t0 overrides,
  good_frames, period selection, the display bunch multiplier). A changed key is
  a fresh entry; a stale key can never hit.
* **Lifetime.** ``Run`` is an unhashable dataclass, so entries key on ``id(run)``
  and register a :func:`weakref.finalize` on first insertion. When the run is
  garbage collected its entries are dropped, which also prevents an ``id`` reused
  by a later object from aliasing a dead run's entries.
* **Aliasing.** The cache returns the *stored* object; callers that hand the
  result to code that may mutate it in place must copy on the way out. The two
  ``mainwindow.py`` wrappers do (a memcpy is negligible against the reduction).
* **Threading.** GUI-thread only. All current call sites run on the GUI thread;
  the cache takes no locks. Do not call it from a worker thread.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from asymmetry.core.utils.perf import perf_logging_enabled

logger = logging.getLogger("asymmetry.perf")

T = TypeVar("T")

#: Default LRU byte budget across all runs and kinds (256 MiB).
_DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024

#: Cap on live entries per ``(run, kind)``. A user toggling between two views or
#: two bunch factors ping-pongs between two entries; a third is browsing, where
#: the global LRU handles reuse.
_PER_RUN_KIND_CAP = 2


@dataclass
class _Entry:
    """One cached reduction result and its bookkeeping."""

    run_id: int
    kind: str
    value: object
    nbytes: int


class ReductionCache:
    """Byte-budgeted LRU cache for GUI-thread time-domain reductions.

    One instance is owned by ``MainWindow``. Entries are keyed by
    ``(id(run), kind, key)`` where ``key`` is a caller-built recipe tuple.
    """

    def __init__(self, budget_bytes: int = _DEFAULT_BUDGET_BYTES) -> None:
        self._budget_bytes = int(budget_bytes)
        #: Ordered oldest-first; move-to-end marks most-recently-used.
        self._entries: OrderedDict[tuple[int, str, tuple], _Entry] = OrderedDict()
        self._total_bytes = 0
        #: One finalize handle per live run id, so we register only once.
        self._finalizers: dict[int, weakref.finalize] = {}
        self.hits = 0
        self.misses = 0

    # ── public API ────────────────────────────────────────────────────

    def get_or_compute(
        self,
        run: object,
        kind: str,
        key: tuple,
        compute: Callable[[], T | None],
        nbytes: Callable[[T], int],
    ) -> T | None:
        """Return the cached result for ``(run, kind, key)`` or compute it.

        On a hit the entry is moved to most-recently-used and returned. On a
        miss ``compute()`` runs; a non-``None`` result is stored (subject to the
        budget) and returned. ``compute()`` returning ``None`` is *not* cached
        (the reductions return ``None`` for no-data / not-applicable inputs),
        and ``nbytes`` is never called on ``None``.

        A single result larger than the whole budget is returned uncached.
        """
        entry_key = (id(run), kind, key)
        cached = self._entries.get(entry_key)
        if cached is not None:
            self._entries.move_to_end(entry_key)
            self.hits += 1
            self._log(kind, hit=True)
            return cached.value  # type: ignore[return-value]

        self.misses += 1
        self._log(kind, hit=False)
        value = compute()
        if value is None:
            return None

        size = int(nbytes(value))
        # An oversized single entry would evict everything and still not fit;
        # skip the cache and hand it back uncached.
        if size > self._budget_bytes:
            return value

        self._register_finalizer(run)
        entry = _Entry(run_id=id(run), kind=kind, value=value, nbytes=size)
        self._entries[entry_key] = entry
        self._total_bytes += size
        self._enforce_per_run_kind_cap(id(run), kind)
        self._evict_to_budget()
        return value

    def invalidate_run(self, run: object) -> None:
        """Drop every entry for ``run`` (belt-and-braces on wholesale edits).

        The key-embedded digest already makes grouping edits self-invalidating;
        this removes dead entries early rather than waiting for LRU pressure.
        """
        self._drop_run(id(run))

    def clear(self) -> None:
        """Drop all entries and finalizers."""
        self._entries.clear()
        self._total_bytes = 0
        for finalizer in self._finalizers.values():
            finalizer.detach()
        self._finalizers.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # ── internals ─────────────────────────────────────────────────────

    def _register_finalizer(self, run: object) -> None:
        run_id = id(run)
        if run_id in self._finalizers:
            return
        # ``weakref.finalize`` fires when the run is garbage collected, dropping
        # its entries and guarding against a later object reusing this id.
        self._finalizers[run_id] = weakref.finalize(run, self._drop_run, run_id)

    def _drop_run(self, run_id: int) -> None:
        dead = [k for k in self._entries if k[0] == run_id]
        for k in dead:
            self._total_bytes -= self._entries.pop(k).nbytes
        finalizer = self._finalizers.pop(run_id, None)
        if finalizer is not None:
            finalizer.detach()

    def _enforce_per_run_kind_cap(self, run_id: int, kind: str) -> None:
        matching = [k for k in self._entries if k[0] == run_id and k[1] == kind]
        # OrderedDict preserves insertion/LRU order, so the front entries are the
        # least-recently-used for this (run, kind).
        while len(matching) > _PER_RUN_KIND_CAP:
            victim = matching.pop(0)
            self._total_bytes -= self._entries.pop(victim).nbytes

    def _evict_to_budget(self) -> None:
        while self._total_bytes > self._budget_bytes and self._entries:
            _key, entry = self._entries.popitem(last=False)
            self._total_bytes -= entry.nbytes
            self._drop_finalizer_if_orphaned(entry.run_id)

    def _drop_finalizer_if_orphaned(self, run_id: int) -> None:
        if any(k[0] == run_id for k in self._entries):
            return
        finalizer = self._finalizers.pop(run_id, None)
        if finalizer is not None:
            finalizer.detach()

    def _log(self, kind: str, *, hit: bool) -> None:
        if perf_logging_enabled():
            logger.info(
                "PERF gui.reduction_cache: %s kind=%s entries=%d bytes=%d",
                "hit" if hit else "miss",
                kind,
                len(self._entries),
                self._total_bytes,
            )


__all__ = ["ReductionCache"]
