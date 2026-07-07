"""Threading contract of the Data Browser's interactive combine actions (E3).

The interactive run-arithmetic actions (co-add, reference subtract, signed
subtract) run ``combine_runs`` + ``reduce_combined_run`` on the panel's
background ``TaskRunner`` while the GUI thread spins a nested event loop —
the window stays responsive (events are processed mid-combine) but the
handlers keep completed-when-returned semantics, so the table-state
invariants and the programmatic ``add_combined_dataset`` path (``.asymp``
restore, which depends on completed-when-returned) are unchanged.
Behavioural coverage of the combine results themselves lives in
tests/gui/test_data_browser_combine.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer

import asymmetry.core.data.combine as combine_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.panels.data_browser import DataBrowserPanel

pytestmark = [pytest.mark.gui, pytest.mark.usefixtures("qapp")]


def _dataset(
    rn: int, *, frames: float = 1000.0, expected: float = 300.0, seed: int = 0
) -> MuonDataset:
    rng = np.random.default_rng(seed)
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 12,
        "last_good_bin": 90,
        "good_frames": frames,
        "t0_bin": 10,
    }
    hs = [
        Histogram(
            counts=rng.poisson(np.full(100, expected * (1.0 if d == 0 else 0.97))).astype(float),
            bin_width=0.016,
            t0_bin=10,
            good_bin_start=10,
            good_bin_end=90,
        )
        for d in range(2)
    ]
    run = Run(
        run_number=rn,
        histograms=hs,
        metadata={"run_number": rn, "title": "S", "temperature": 5.0, "field": 100.0},
        grouping=grouping,
        source_file=f"/tmp/run_{rn}.nxs",
    )
    return MuonDataset(
        time=np.arange(100.0),
        asymmetry=np.zeros(100),
        error=np.ones(100),
        metadata=dict(run.metadata),
        run=run,
    )


def _coadd_panel() -> DataBrowserPanel:
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(401, frames=1000, seed=1))
    panel.add_dataset(_dataset(402, frames=1000, seed=2))
    panel.select_runs({401, 402})
    return panel


def test_interactive_coadd_runs_off_gui_thread_and_stays_responsive(monkeypatch):
    panel = _coadd_panel()
    real_combine = combine_module.combine_runs
    combine_threads: list[QThread] = []
    pumped: list[int] = []
    gate = threading.Event()

    def spy(*args, **kwargs):
        combine_threads.append(QThread.currentThread())
        # Hold the worker until the GUI-thread timer below has fired, proving
        # the event loop kept spinning while the combine was in flight.
        gate.wait(10)
        return real_combine(*args, **kwargs)

    monkeypatch.setattr(combine_module, "combine_runs", spy)

    def _mid_combine_event() -> None:
        pumped.append(1)
        gate.set()

    QTimer.singleShot(0, _mid_combine_event)
    panel._coadd_selected()

    # The event loop processed GUI events while the combine computed: the
    # click no longer freezes the window.
    assert pumped, "the GUI event loop did not run during the combine"
    # The heavy compute happened off the GUI thread.
    assert combine_threads, "combine_runs was never invoked"
    gui_thread = QThread.currentThread()
    assert all(t is not gui_thread for t in combine_threads)

    # Completed when returned: combined row present, sources consumed.
    crn = next(iter(panel._combined_datasets))
    assert panel._combined_datasets[crn] == [401, 402]
    assert 401 not in panel._datasets
    assert 402 not in panel._datasets
    assert set(panel._get_selected_run_numbers()) == {crn}
    combined = panel.get_dataset(crn)
    assert combined.run is not None and combined.run.histograms
    panel.shutdown_workers()


def test_double_fire_yields_exactly_one_combine(monkeypatch):
    panel = _coadd_panel()
    real_combine = combine_module.combine_runs
    calls: list[int] = []
    gate = threading.Event()

    def spy(*args, **kwargs):
        calls.append(1)
        gate.wait(10)
        return real_combine(*args, **kwargs)

    monkeypatch.setattr(combine_module, "combine_runs", spy)

    def _second_fire() -> None:
        # Delivered inside the first combine's nested loop: the in-flight
        # guard must swallow it without starting a second worker.
        panel._coadd_selected()
        gate.set()

    QTimer.singleShot(0, _second_fire)
    panel._coadd_selected()

    assert len(calls) == 1
    assert len(panel._combined_datasets) == 1
    panel.shutdown_workers()


def test_add_combined_dataset_restore_path_stays_synchronous(monkeypatch):
    panel = DataBrowserPanel()
    panel.add_dataset(_dataset(401, frames=1000, seed=1))
    panel.add_dataset(_dataset(402, frames=1000, seed=2))
    real_combine = combine_module.combine_runs
    combine_threads: list[QThread] = []

    def spy(*args, **kwargs):
        combine_threads.append(QThread.currentThread())
        return real_combine(*args, **kwargs)

    monkeypatch.setattr(combine_module, "combine_runs", spy)

    crn = panel.add_combined_dataset([401, 402], sign=1)

    # Completed-when-returned on the GUI thread itself: no worker, no event
    # pumping — project restore's chunked runner depends on this.
    assert crn is not None
    assert crn in panel._combined_datasets
    assert panel.get_dataset(crn) is not None
    assert combine_threads and all(t is QThread.currentThread() for t in combine_threads)
    assert panel._tasks.active_count == 0
    panel.shutdown_workers()


def test_shutdown_mid_flight_combine_does_not_crash(monkeypatch):
    panel = _coadd_panel()
    real_combine = combine_module.combine_runs

    def slow_combine(*args, **kwargs):
        time.sleep(0.3)
        return real_combine(*args, **kwargs)

    monkeypatch.setattr(combine_module, "combine_runs", slow_combine)

    # The main window's closeEvent path, delivered while the combine is in
    # flight (inside the nested loop): shutdown must be reachable and safe.
    QTimer.singleShot(20, panel.shutdown_workers)
    panel._coadd_selected()

    assert panel._tasks.active_count == 0
    assert panel._combine_worker is None


def test_combine_error_surfaces_warning_and_clears_guard(monkeypatch):
    panel = _coadd_panel()
    warnings: list[tuple] = []
    monkeypatch.setattr(
        "asymmetry.gui.panels.data_browser.QMessageBox.warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    def failing_combine(*args, **kwargs):
        raise combine_module.CombineError("bin widths differ")

    monkeypatch.setattr(combine_module, "combine_runs", failing_combine)

    panel._coadd_selected()

    assert warnings, "the worker's CombineError never reached QMessageBox.warning"
    assert not panel._combined_datasets
    # Sources untouched, guard cleared: the next attempt may start.
    assert 401 in panel._datasets and 402 in panel._datasets
    assert panel._combine_worker is None
    panel.shutdown_workers()
