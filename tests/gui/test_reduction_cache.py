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


# ── User-action-level invocation-count regressions (audit D2) ──────────────
#
# ``TestWiredCallSites`` above pins the cache-wrapper behaviour by calling the
# ``_grouped_time_domain_display_datasets``/``_counts_first_rebunched_arrays``
# wrappers directly. The tests below instead drive the real user-facing paths
# — data-browser row selection (``select_runs``, which mirrors a mouse click:
# it goes through the same ``_on_selection_changed`` → ``dataset_selected``/
# ``selection_changed`` signal chain), the Domain toolbar buttons, and the
# View → Diagnostics → Raw counts action — and assert the *end-to-end*
# payoff the audit measured: dataset ping-pong and view toggling used to
# recompute a reduction on every dataset switch, and a single render could
# invoke the underlying provider 2-4 times.


class TestUserActionInvocationCounts:
    def test_dataset_ping_pong_in_groups_view_computes_each_run_once(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting A, B, A in the Groups view builds A and B once each.

        The groups Domain button is disabled until a grouped-capable dataset
        is current, so run A is selected first (enabling it) and then the
        button click renders A in the Groups view for the first time — that
        click *is* the first of the three ping-pong renders. Selecting B
        renders B (new run -> cache miss); reselecting A hits the still-live
        per-run cache entry instead of rebuilding.
        """
        run_a = _multi_group_dataset(run_number=5101)
        run_b = _multi_group_dataset(run_number=5102)
        mainwindow._data_browser.add_dataset(run_a)
        mainwindow._data_browser.add_dataset(run_b)

        mainwindow._data_browser.select_runs({5101})
        assert mainwindow._domain_buttons_by_token["groups"].isEnabled()

        calls = _count_builds(monkeypatch)

        mainwindow._domain_buttons_by_token["groups"].click()  # render A (1st)
        assert mainwindow._plot_workspace.active_view() == "groups"
        mainwindow._data_browser.select_runs({5102})  # render B
        mainwindow._data_browser.select_runs({5101})  # render A again (cache hit)

        assert calls["n"] == 2

    def test_single_render_with_bunch_factor_reduces_exactly_once(
        self, mainwindow: MainWindow
    ) -> None:
        """One render at bunch factor > 1 reduces once, though it is probed
        repeatedly.

        The main-toolbar bunch spinbox bakes its value into
        ``run.grouping['bunching_factor']`` and re-reduces through
        ``_apply_grouping_settings_to_dataset`` for any dataset with a real
        grouping (see ``_apply_bunch_factor_to_context``), so it never
        reaches the counts-first-rebunch provider wired here. The provider
        (``_counts_first_rebunched_arrays``, installed as the plot panel's
        ``counts_rebunch_provider``) is instead driven by the panel's own
        (normally hidden) bunch spinbox, exactly as
        ``tests/gui/test_plot_bunch_counts_first.py`` exercises it:
        ``plot_panel.set_bunch_factor(factor, emit_signal=False)``.

        The seam counted is ``_reduce_grouped_histograms_to_asymmetry`` — the
        chokepoint shared by every reduction path in this module (see its
        docstring and ``_count_reduces`` above) — rather than counting calls
        to ``get_analysis_dataset``/the rebunch provider directly, so the
        assertion holds regardless of how many internal call sites
        (plotting, RRF display, fit-range seeding) probe the analysis
        dataset during one render. A plain ``calls["n"] == 1`` would pass
        vacuously if the render happened to only probe once even without a
        cache, so the cache's own hit/miss counters are asserted too: exactly
        one miss (the single real compute) plus several hits confirms the
        render actually exercised — and deduplicated — more than one call
        into the provider (measured at 4 probes for one render of this
        fixture, matching the audit's "2-4 calls per render" finding).

        The bunch factor is pre-armed on the panel *before* the dataset is
        ever selected, and the counter/snapshot are installed just before
        that first selection. This matters for two reasons: (1)
        ``set_bunch_factor(..., emit_signal=False)`` itself redraws
        immediately (it calls ``_redraw_current_view()``, which no-ops here
        because the panel has no current dataset yet), so arming it first
        avoids a stray extra render; and (2) the panel's per-window fit-range
        seed (``_fit_x_min``/``_fit_x_max``) is populated on the *first* plot
        of a session and then stays sticky, and one of the probe call sites
        (``_raw_fit_seed_range``) only fires while it is still unset — so
        this is the one render in a window's life where that seam's full
        multiplicity is guaranteed to show up. A run already selected once at
        bunch factor 1 and then switched to bunch factor 3 would still prove
        the dedup (one miss), but could under-count the probes this test
        wants to demonstrate collapsing.
        """
        dataset = _multi_group_dataset(run_number=5201, n=1200)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._plot_panel.set_bunch_factor(3, emit_signal=False)  # pre-armed; no-op redraw

        calls = _count_reduces(mainwindow)
        misses_before = mainwindow._reduction_cache.misses
        hits_before = mainwindow._reduction_cache.hits

        mainwindow._data_browser.select_runs({5201})  # the one render
        assert mainwindow._plot_workspace.active_view() == "fb_asymmetry"

        assert calls["n"] == 1
        assert mainwindow._reduction_cache.misses - misses_before == 1
        assert mainwindow._reduction_cache.hits - hits_before >= 1

    def test_grouping_edit_invalidates_cached_groups_view_build(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A digest-covered grouping edit forces the next selection to rebuild.

        Mirrors ``test_grouped_td_reuses_then_rebuilds_on_digest_change``
        above but through the real selection path: the run's grouping is
        mutated directly (as the grouping dialog would leave it after
        ``_apply_grouping_settings_to_dataset`` writes ``first_good_bin``),
        then the same run is reselected. ``select_runs`` on an already-
        selected run still fires ``_on_selection_changed`` (it calls it
        unconditionally after restoring the selection, not only on a Qt-
        detected change), so "reselect" reliably triggers a fresh render.
        """
        dataset = _multi_group_dataset(run_number=5301)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._data_browser.select_runs({5301})

        calls = _count_builds(monkeypatch)
        mainwindow._domain_buttons_by_token["groups"].click()
        assert calls["n"] == 1

        dataset.run.grouping["first_good_bin"] = 5
        mainwindow._data_browser.select_runs({5301})  # reselect after the edit

        assert calls["n"] == 2

    def test_view_toggle_ping_pong_caches_both_kinds(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Groups -> Raw counts -> Groups builds each representation once.

        Groups (lifetime-corrected) and Raw counts (uncorrected) key to two
        distinct cache entries for the same ``(run, "grouped_td")`` bucket;
        the per-``(run, kind)`` cap is 2 (``TestReductionCacheClass
        .test_per_run_kind_cap_of_two``), so both stay resident and the
        third leg — back to Groups — is a hit rather than an eviction-forced
        rebuild.
        """
        dataset = _multi_group_dataset(run_number=5401)
        mainwindow._data_browser.add_dataset(dataset)
        mainwindow._data_browser.select_runs({5401})

        calls = _count_builds(monkeypatch)

        mainwindow._domain_buttons_by_token["groups"].click()
        assert mainwindow._plot_workspace.active_view() == "groups"
        assert calls["n"] == 1

        mainwindow._raw_counts_action.trigger()
        assert mainwindow._plot_workspace.active_view() == "raw_counts"
        assert calls["n"] == 2

        mainwindow._raw_counts_action.trigger()
        assert mainwindow._plot_workspace.active_view() == "groups"
        assert calls["n"] == 2  # cache hit, not a third build
