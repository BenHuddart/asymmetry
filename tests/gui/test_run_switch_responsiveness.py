"""Run-switch cost pins: what a data-browser selection change may compute.

Switching the active run on fine-resolution data (~90k bins) used to stall
the GUI for 150–300 ms: matplotlib ``errorbar`` construction, four grouped
count-domain rebuilds of the same crop, a hidden grouped-fit window rebuilt
on every switch, the leaving run's fit form saved and restored, and periodic
full garbage collections. These tests pin each fix with an invocation count
driven through the real selection pipeline (``DataBrowserPanel.select_runs``
→ ``selection_changed`` → ``dataset_selected``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

import asymmetry.gui.mainwindow as mw_module  # noqa: E402
import asymmetry.gui.panels.fit.global_tab as global_tab_module  # noqa: E402
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run  # noqa: E402
from asymmetry.core.representation import RepresentationType  # noqa: E402
from asymmetry.gui.mainwindow import MainWindow  # noqa: E402

pytestmark = [pytest.mark.gui]


@pytest.fixture
def mainwindow(qapp: QApplication) -> MainWindow:
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    qapp.processEvents()
    yield window
    window.close()
    window.deleteLater()


def _grouped_dataset(run_number: int, n_bins: int = 64) -> MuonDataset:
    """A two-detector run with grouping, so grouped count domains can be built."""
    t = np.arange(n_bins, dtype=float) * 0.05
    counts_f = 1000.0 * np.exp(-t / 2.2) * (1.0 + 0.2 * np.cos(2.0 * t)) + 5.0
    counts_b = 900.0 * np.exp(-t / 2.2) * (1.0 - 0.2 * np.cos(2.0 * t)) + 5.0
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=counts_f, bin_width=0.05),
            Histogram(counts=counts_b, bin_width=0.05),
        ],
        metadata={"run_number": run_number, "field": 100.0, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "F", 2: "B"},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n_bins - 1,
            "bunching_factor": 1,
            "deadtime_correction": False,
        },
    )
    asym = 100.0 * (counts_f - counts_b) / (counts_f + counts_b)
    return MuonDataset(
        time=t,
        asymmetry=asym,
        error=np.full_like(t, 0.5),
        metadata={"run_number": run_number, "field": 100.0, "temperature": 5.0},
        run=run,
    )


def _load_two_runs(mainwindow: MainWindow) -> tuple[MuonDataset, MuonDataset]:
    a, b = _grouped_dataset(9101), _grouped_dataset(9102)
    mainwindow._data_browser.add_dataset(a)
    mainwindow._data_browser.add_dataset(b)
    mainwindow._data_browser.select_runs({9101})
    QApplication.processEvents()
    return a, b


# ── fit-range crop memo ────────────────────────────────────────────────────


class TestFitDatasetMemo:
    def test_same_source_yields_the_same_crop_object(self, mainwindow: MainWindow) -> None:
        a, _b = _load_two_runs(mainwindow)
        first = mainwindow._get_fit_dataset(a)
        assert mainwindow._get_fit_dataset(a) is first
        assert mainwindow._get_fit_dataset(a) is first

    def test_reassigned_arrays_invalidate_the_crop(self, mainwindow: MainWindow) -> None:
        a, _b = _load_two_runs(mainwindow)
        first = mainwindow._get_fit_dataset(a)
        a.asymmetry = a.asymmetry * 1.1  # a re-reduction reassigns, never mutates in place
        second = mainwindow._get_fit_dataset(a)
        assert second is not first
        np.testing.assert_allclose(second.asymmetry, first.asymmetry * 1.1)

    def test_fit_range_change_invalidates_the_crop(self, mainwindow: MainWindow) -> None:
        a, _b = _load_two_runs(mainwindow)
        first = mainwindow._get_fit_dataset(a)
        mainwindow._plot_panel.set_fit_range(0.5, 2.0)
        second = mainwindow._get_fit_dataset(a)
        assert second is not first
        assert second.time.min() >= 0.5 and second.time.max() <= 2.0

    def test_bunch_factor_change_invalidates_the_crop(self, mainwindow: MainWindow) -> None:
        a, _b = _load_two_runs(mainwindow)
        mainwindow._plot_panel.set_fit_range(0.0, 1.0)  # a genuine (copied) crop
        first = mainwindow._get_fit_dataset(a)
        mainwindow._plot_panel.set_bunch_factor(2, emit_signal=False)
        second = mainwindow._get_fit_dataset(a)
        assert second is not first
        assert second.n_points < first.n_points

    def test_metadata_edits_reach_the_cached_crop(self, mainwindow: MainWindow) -> None:
        a, _b = _load_two_runs(mainwindow)
        mainwindow._plot_panel.set_fit_range(0.0, 1.0)
        first = mainwindow._get_fit_dataset(a)
        assert first is not a
        a.metadata["field"] = 4321.0  # field overrides are edited in place
        assert mainwindow._get_fit_dataset(a) is first
        assert first.metadata["field"] == pytest.approx(4321.0)

    def test_no_fit_range_entry_never_pins_the_source(self, mainwindow: MainWindow) -> None:
        # Nothing plotted yet, so no fit range is seeded: the "crop" is the
        # source itself and the memo must not hold it strongly.
        assert mainwindow._plot_panel.get_fit_range() == (None, None)
        source = _grouped_dataset(9198)
        assert mainwindow._get_fit_dataset(source) is source
        _key, cached = mainwindow._fit_dataset_memo[id(source)]
        assert cached is None

    def test_dead_source_drops_its_entry(self, mainwindow: MainWindow) -> None:
        import gc

        _load_two_runs(mainwindow)
        mainwindow._plot_panel.set_fit_range(0.0, 1.0)
        source = _grouped_dataset(9199)
        crop = mainwindow._get_fit_dataset(source)
        assert crop is not source
        key = id(source)
        assert key in mainwindow._fit_dataset_memo
        del source
        gc.collect()
        assert key not in mainwindow._fit_dataset_memo


# ── what one switch computes ───────────────────────────────────────────────


class _CountingGroupedWindow(QWidget):
    """Stand-in for MultiGroupFitWindow that records every binding."""

    def __init__(self) -> None:
        super().__init__()
        self.datasets: list[object] = []
        self.members: list[list[object]] = []
        self._title = "Multi-Group Fit"

    def set_dataset(self, dataset) -> None:
        self.datasets.append(dataset)
        if dataset is not None:
            self._title = f"Multi-Group Fit — {dataset.run_number}"

    def set_member_datasets(self, datasets) -> None:
        self.members.append(list(datasets))

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        return

    def set_fit_range_display(self, x_min, x_max) -> None:
        return

    def dock_title(self) -> str:
        return self._title


def _install_counting_window(mainwindow: MainWindow) -> _CountingGroupedWindow:
    stub = _CountingGroupedWindow()
    mainwindow._multi_group_fit_window = stub
    mainwindow._fit_stack.addWidget(stub)
    return stub


class TestRunSwitchWork:
    def test_switch_binds_the_fit_panel_exactly_once(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _a, _b = _load_two_runs(mainwindow)
        bound: list[object] = []
        original = mainwindow._fit_panel.set_dataset
        monkeypatch.setattr(
            mainwindow._fit_panel,
            "set_dataset",
            lambda dataset: (bound.append(dataset), original(dataset))[1],
        )

        mainwindow._data_browser.select_runs({9102})

        # The leaving run is not re-bound first: one bind, to the new run.
        assert [getattr(d, "run_number", None) for d in bound] == [9102]

    def test_switch_builds_grouped_count_domains_for_the_crop_at_most_once(
        self, mainwindow: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _a, _b = _load_two_runs(mainwindow)
        calls: list[int] = []
        original = global_tab_module.build_grouped_time_domain_groups

        def _counting(dataset, **kwargs):
            calls.append(int(dataset.run_number))
            return original(dataset, **kwargs)

        monkeypatch.setattr(global_tab_module, "build_grouped_time_domain_groups", _counting)

        mainwindow._data_browser.select_runs({9102})

        # Before the crop memo + same-object short-circuits this was four
        # rebuilds of the same run per switch (two visible-surface binds plus
        # the hidden window's two tabs).
        assert calls.count(9102) <= 1
        assert 9101 not in calls

    def test_switch_does_not_bind_the_hidden_grouped_window(self, mainwindow: MainWindow) -> None:
        _a, _b = _load_two_runs(mainwindow)
        stub = _install_counting_window(mainwindow)
        assert not mainwindow._should_launch_multi_group_fit_window()

        mainwindow._data_browser.select_runs({9102})
        mainwindow._data_browser.select_runs({9101})

        assert stub.datasets == []
        assert stub.members == []

    def test_entering_groups_view_binds_the_window_and_its_members(
        self, mainwindow: MainWindow
    ) -> None:
        a, b = _load_two_runs(mainwindow)
        stub = _install_counting_window(mainwindow)
        mainwindow._data_browser.select_runs({9102})
        assert stub.datasets == []

        mainwindow._plot_workspace.set_available_views(["fb_asymmetry", "groups"])
        mainwindow._plot_workspace.set_active_view("groups")
        assert mainwindow._should_launch_multi_group_fit_window()

        assert {d.run_number for d in stub.datasets if d is not None} == {9102}
        assert stub.members and [d.run_number for d in stub.members[-1]] == [9102]
        # While the window is the visible surface it tracks the selection.
        mainwindow._data_browser.select_runs({9101})
        assert [d.run_number for d in stub.datasets if d is not None][-1] == 9101


# ── in-flight spectrum recomputes ──────────────────────────────────────────


class TestRecomputeWaiters:
    def test_await_fires_immediately_when_nothing_is_in_flight(
        self, mainwindow: MainWindow
    ) -> None:
        fired: list[int] = []
        mainwindow._await_frequency_recomputes(
            [(1, RepresentationType.FREQ_FFT)], lambda: fired.append(1)
        )
        assert fired == [1]

    def test_ensure_waits_for_an_in_flight_run_instead_of_reporting_ready(
        self, mainwindow: MainWindow
    ) -> None:
        rep = RepresentationType.FREQ_FFT
        key = (4242, rep)  # no such run loaded: nothing to compute once it lands
        mainwindow._frequency_recompute_inflight.add(key)
        ready: list[int] = []

        mainwindow._ensure_frequency_spectra_for_runs_async([4242], rep, lambda: ready.append(1))

        assert ready == []  # parked behind the in-flight recompute
        mainwindow._frequency_recompute_inflight.discard(key)
        mainwindow._notify_frequency_recompute_done(key)
        assert ready == [1]
        assert mainwindow._frequency_recompute_waiters == []

    def test_superseded_recompute_result_is_dropped_and_waiters_released(
        self, mainwindow: MainWindow
    ) -> None:
        rep = RepresentationType.FREQ_FFT
        key = (4243, rep)
        cached: list[object] = []
        representation = SimpleNamespace(
            recipe={"fourier_config": {"padding": 8}},
            cache_datasets=lambda spectra: cached.append(spectra),
        )
        mainwindow._frequency_recompute_inflight.add(key)
        released: list[int] = []
        mainwindow._await_frequency_recomputes([key], lambda: released.append(1))

        mainwindow._on_frequency_recompute_finished(
            4243,
            rep,
            representation,
            [object()],
            0.0,
            recipe_at_start={"fourier_config": {"padding": 4}},  # recipe changed meanwhile
        )

        assert cached == []
        assert key not in mainwindow._frequency_recompute_inflight
        assert released == [1]
