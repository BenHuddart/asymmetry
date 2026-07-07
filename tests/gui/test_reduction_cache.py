"""Unit + wiring tests for the GUI reduction cache (audit D1).

Two layers:

* ``TestReductionCacheClass`` exercises the pure :class:`ReductionCache` — hit/miss
  semantics, the byte budget, the oversized-entry bypass, the ``None``-not-cached
  contract, weakref lifetime, the per-``(run, kind)`` cap, and invalidation.
* ``TestWiredCallSites`` drives the two ``MainWindow`` wrappers and asserts that the
  underlying reduction runs exactly once per distinct recipe, and re-runs when a
  key component changes — including the three confirmed digest gaps (alpha,
  per-detector t0 overrides, good_frames) plus the additional keyed-but-undigested
  fields (included_groups, deadtime_mode, period_mode). Copy-on-handout is checked
  so a mutated result cannot poison the next hit.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from asymmetry.core.fourier import fourier_grouping_digest
from asymmetry.core.transform import EFFECTIVE_DETECTOR_T0_KEY
from asymmetry.gui.utils.reduction_cache import ReductionCache

# The wired tests construct a MainWindow; mark the whole module GUI so CI routes
# it to the GUI shards (the pure cache-class tests are cheap and ride along).
pytestmark = pytest.mark.gui


class _FakeRun:
    """A weakly-referenceable stand-in for a Run (``object()`` is not)."""


def _arr(n_floats: int) -> np.ndarray:
    return np.ones(n_floats, dtype=np.float64)


def _nbytes(a: np.ndarray) -> int:
    return int(a.nbytes)


class TestReductionCacheClass:
    def test_same_key_computes_once_changed_key_recomputes(self) -> None:
        cache = ReductionCache()
        run = _FakeRun()
        calls = {"n": 0}

        def compute() -> np.ndarray:
            calls["n"] += 1
            return _arr(4)

        cache.get_or_compute(run, "k", (1,), compute, _nbytes)
        cache.get_or_compute(run, "k", (1,), compute, _nbytes)
        assert calls["n"] == 1
        assert cache.hits == 1 and cache.misses == 1

        cache.get_or_compute(run, "k", (2,), compute, _nbytes)
        assert calls["n"] == 2
        assert cache.misses == 2

    def test_byte_budget_evicts_lru(self) -> None:
        # Three distinct runs (so the per-(run,kind) cap never fires); each entry
        # is 400 bytes, budget 1000 → the third insert evicts the LRU first.
        cache = ReductionCache(budget_bytes=1000)
        runs = [_FakeRun() for _ in range(3)]
        for run in runs:
            cache.get_or_compute(run, "k", (0,), lambda: _arr(50), _nbytes)
        assert len(cache) == 2
        # runs[0] is the least-recently-used and must be gone.
        recomputed = {"n": 0}

        def compute() -> np.ndarray:
            recomputed["n"] += 1
            return _arr(50)

        cache.get_or_compute(runs[0], "k", (0,), compute, _nbytes)
        assert recomputed["n"] == 1  # it was evicted, so this is a miss

    def test_oversized_single_entry_bypasses_cache(self) -> None:
        cache = ReductionCache(budget_bytes=100)
        run = _FakeRun()
        result = cache.get_or_compute(run, "k", (0,), lambda: _arr(1000), _nbytes)
        assert result is not None and result.size == 1000
        assert len(cache) == 0  # not stored

    def test_none_result_not_cached_and_nbytes_not_called(self) -> None:
        cache = ReductionCache()
        run = _FakeRun()

        def boom(_value: object) -> int:  # pragma: no cover - must never run
            raise AssertionError("nbytes called on a None result")

        assert cache.get_or_compute(run, "k", (0,), lambda: None, boom) is None
        assert len(cache) == 0

    def test_weakref_lifetime_drops_entries_on_gc(self) -> None:
        cache = ReductionCache()
        run = _FakeRun()
        cache.get_or_compute(run, "k", (0,), lambda: _arr(4), _nbytes)
        assert len(cache) == 1
        del run
        gc.collect()
        assert len(cache) == 0

    def test_per_run_kind_cap_of_two(self) -> None:
        cache = ReductionCache()  # generous budget
        run = _FakeRun()
        for i in range(3):
            cache.get_or_compute(run, "k", (i,), lambda: _arr(4), _nbytes)
        # A third entry for the same (run, kind) evicts the oldest → cap 2.
        assert len(cache) == 2
        # A different kind on the same run is tracked separately.
        cache.get_or_compute(run, "other", (0,), lambda: _arr(4), _nbytes)
        assert len(cache) == 3

    def test_invalidate_run_and_clear(self) -> None:
        cache = ReductionCache()
        run_a, run_b = _FakeRun(), _FakeRun()
        cache.get_or_compute(run_a, "k", (0,), lambda: _arr(4), _nbytes)
        cache.get_or_compute(run_b, "k", (0,), lambda: _arr(4), _nbytes)
        cache.invalidate_run(run_a)
        assert len(cache) == 1  # only run_b remains
        cache.clear()
        assert len(cache) == 0


# ── Wired call-site tests (require a MainWindow) ───────────────────────────

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore  # noqa: E402
from PySide6.QtWidgets import QApplication  # type: ignore  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402

BIN_WIDTH_US = 0.016


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    return MainWindow()


def _multi_group_dataset(
    n_groups: int = 2,
    *,
    n: int = 400,
    grouping_extra: dict | None = None,
    run_number: int = 4242,
) -> MuonDataset:
    """A run with one histogram per group and a fixed-binning grouping."""
    rng = np.random.default_rng(7)
    histograms = [
        Histogram(counts=rng.poisson(500.0, n).astype(float), bin_width=BIN_WIDTH_US, t0_bin=0)
        for _ in range(n_groups)
    ]
    grouping = {
        "groups": {gid: [gid] for gid in range(1, n_groups + 1)},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": n - 1,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if grouping_extra:
        grouping.update(grouping_extra)
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number},
        grouping=grouping,
    )
    return MuonDataset(
        time=np.arange(n, dtype=float) * BIN_WIDTH_US,
        asymmetry=np.zeros(n),
        error=np.ones(n),
        metadata={"run_number": run_number},
        run=run,
    )


def _count_builds(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch the grouped-count build to count invocations."""
    original = mw_module.build_grouped_time_domain_datasets
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mw_module, "build_grouped_time_domain_datasets", counting)
    return calls


def _count_reduces(mw: MainWindow) -> dict[str, int]:
    """Wrap the reduction chokepoint on the instance to count invocations."""
    original = mw._reduce_grouped_histograms_to_asymmetry
    calls = {"n": 0}

    def counting(**kwargs):
        calls["n"] += 1
        return original(**kwargs)

    mw._reduce_grouped_histograms_to_asymmetry = counting  # type: ignore[method-assign]
    return calls


class TestWiredCallSites:
    # ── grouped time-domain build ─────────────────────────────────────

    def test_grouped_td_reuses_then_rebuilds_on_digest_change(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _count_builds(monkeypatch)
        dataset = _multi_group_dataset()

        first = mainwindow._grouped_time_domain_display_datasets(dataset)
        second = mainwindow._grouped_time_domain_display_datasets(dataset)
        assert calls["n"] == 1  # second served from cache
        assert len(first) == len(second) == 2

        # A digest-covered edit (good-bin window) forces a rebuild.
        dataset.run.grouping["first_good_bin"] = 5
        mainwindow._grouped_time_domain_display_datasets(dataset)
        assert calls["n"] == 2

    def test_grouped_td_recomputes_on_included_groups_change(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _count_builds(monkeypatch)
        dataset = _multi_group_dataset(n_groups=3)
        run = dataset.run
        run.grouping["included_groups"] = {1: True, 2: True, 3: True}

        digest_before = fourier_grouping_digest(run)
        first = mainwindow._grouped_time_domain_display_datasets(dataset)
        assert len(first) == 3

        run.grouping["included_groups"] = {1: True, 2: True, 3: False}
        # included_groups is not in the digest — proving the key must carry it.
        assert fourier_grouping_digest(run) == digest_before
        second = mainwindow._grouped_time_domain_display_datasets(dataset)
        assert calls["n"] == 2  # recomputed
        assert len(second) == 2

    def test_grouped_td_recomputes_on_good_frames_change(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _count_builds(monkeypatch)
        dataset = _multi_group_dataset(grouping_extra={"good_frames": 1000.0})
        run = dataset.run
        digest_before = fourier_grouping_digest(run)

        mainwindow._grouped_time_domain_display_datasets(dataset)
        run.grouping["good_frames"] = 2000.0
        assert fourier_grouping_digest(run) == digest_before  # digest-invisible
        mainwindow._grouped_time_domain_display_datasets(dataset)
        assert calls["n"] == 2

    def test_grouped_td_copies_on_handout(self, mainwindow: MainWindow) -> None:
        dataset = _multi_group_dataset()
        first = mainwindow._grouped_time_domain_display_datasets(dataset)
        baseline = float(first[0].asymmetry[0])
        first[0].asymmetry[0] += 1234.0  # mutate a handed-out result
        second = mainwindow._grouped_time_domain_display_datasets(dataset)
        assert float(second[0].asymmetry[0]) == pytest.approx(baseline)

    # ── counts-first rebunch ──────────────────────────────────────────

    def test_rebunch_reuses_then_recomputes_on_digest_change(self, mainwindow: MainWindow) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 1  # cached

        # Different display factor is a different recipe.
        mainwindow._counts_first_rebunched_arrays(dataset, 4)
        assert calls["n"] == 2

        # A digest-covered edit re-reduces.
        dataset.run.grouping["last_good_bin"] = 300
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 3

    def test_rebunch_recomputes_on_alpha_change(self, mainwindow: MainWindow) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()
        run = dataset.run
        digest_before = fourier_grouping_digest(run)

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        run.grouping["alpha"] = 1.7
        assert fourier_grouping_digest(run) == digest_before  # alpha not digested
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2

    def test_rebunch_recomputes_on_detector_t0_override_change(
        self, mainwindow: MainWindow
    ) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()
        run = dataset.run
        run.grouping[EFFECTIVE_DETECTOR_T0_KEY] = [0, 0]
        digest_before = fourier_grouping_digest(run)

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        run.grouping[EFFECTIVE_DETECTOR_T0_KEY] = [1, 1]
        assert fourier_grouping_digest(run) == digest_before  # overrides not digested
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2

    def test_rebunch_recomputes_on_good_frames_change(self, mainwindow: MainWindow) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset(grouping_extra={"good_frames": 1000.0})
        run = dataset.run
        digest_before = fourier_grouping_digest(run)

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        run.grouping["good_frames"] = 2000.0
        assert fourier_grouping_digest(run) == digest_before  # good_frames not digested
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2

    def test_rebunch_recomputes_on_deadtime_mode_change(self, mainwindow: MainWindow) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()
        run = dataset.run
        digest_before = fourier_grouping_digest(run)

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        run.grouping["deadtime_mode"] = "file"
        assert fourier_grouping_digest(run) == digest_before  # mode not digested
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2

    def test_rebunch_recomputes_on_period_mode_change(self, mainwindow: MainWindow) -> None:
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()
        run = dataset.run
        digest_before = fourier_grouping_digest(run)

        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        run.grouping["period_mode"] = "Green"
        assert fourier_grouping_digest(run) == digest_before  # period mode not digested
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2

    def test_rebunch_copies_on_handout(self, mainwindow: MainWindow) -> None:
        dataset = _multi_group_dataset()
        first = mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert first is not None
        baseline = float(first[1][0])
        first[1][0] += 999.0  # mutate the handed-out asymmetry
        second = mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert second is not None
        assert float(second[1][0]) == pytest.approx(baseline)

    def test_combined_dataset_replacement_computes_fresh(self, mainwindow: MainWindow) -> None:
        """``existing_dataset.run = new_run`` (co-add/combine) keys on the new run."""
        calls = _count_reduces(mainwindow)
        dataset = _multi_group_dataset()
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 1

        # Simulate _store_combined_reduction swapping in a new Run in place.
        replacement = _multi_group_dataset(run_number=9999).run
        dataset.run = replacement
        gc.collect()
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2  # new run identity → fresh compute
        mainwindow._counts_first_rebunched_arrays(dataset, 2)
        assert calls["n"] == 2  # and the new run now caches

    def test_apply_grouping_settings_invalidates_run(self, mainwindow: MainWindow) -> None:
        """Group-label / metadata freshness leans on the belt-and-braces hook.

        ``group_names`` is not in the digest, so a rename is only guaranteed to
        refresh cached grouped-count datasets because the wholesale grouping
        rewrite calls ``invalidate_run``. Pin that it actually does.
        """
        dataset = _multi_group_dataset()
        seen: list[object] = []
        original = mainwindow._reduction_cache.invalidate_run

        def spy(run: object) -> None:
            seen.append(run)
            original(run)

        mainwindow._reduction_cache.invalidate_run = spy  # type: ignore[method-assign]
        mainwindow._apply_grouping_settings_to_dataset(dataset, dict(dataset.run.grouping))
        assert dataset.run in seen
