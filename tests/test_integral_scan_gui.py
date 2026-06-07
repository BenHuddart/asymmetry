"""GUI tests for integral-scan (ALC) mode (G2).

Covers the fit-panel toggle that switches the batch action into integral-scan
mode and the main-window handler that integrates each run over the fit-range
window and records the result as a model-less ("computed") ``FitSeries`` the
trend panel renders.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.representation import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fit_panel import FitPanel


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


def _ds(run_number: int, fwd: float, bwd: float, field: float) -> MuonDataset:
    n = 4
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, fwd), bin_width=0.01),
            Histogram(counts=np.full(n, bwd), bin_width=0.01),
        ],
        metadata={"run_number": run_number, "field": field},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "bunching_factor": 1,
        },
    )
    t = np.arange(n) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros(n),
        error=np.full(n, 0.01),
        metadata={"run_number": run_number, "field": field},
        run=run,
    )


# --- fit-panel toggle --------------------------------------------------------


def test_scan_mode_toggle_relabels_and_emits(qapp: QApplication):
    panel = FitPanel()
    tab = panel._global_tab

    emitted: list[bool] = []
    panel.scan_requested.connect(lambda: emitted.append(True))

    assert not tab.is_scan_mode()
    tab._scan_mode_check.setChecked(True)
    assert tab.is_scan_mode()
    assert tab._fit_btn.text() == "Build Integral Scan"

    # In scan mode the batch action emits scan_requested instead of fitting.
    tab._run_global_fit()
    assert emitted == [True]

    tab._scan_mode_check.setChecked(False)
    assert tab._fit_btn.text() == "Run Batch Fit"


def test_scan_mode_state_round_trips(qapp: QApplication):
    panel = FitPanel()
    panel._global_tab._scan_mode_check.setChecked(True)
    state = panel._global_tab.get_state()
    assert state.get("scan_mode") is True

    other = FitPanel()
    other._global_tab.restore_state(state)
    assert other._global_tab.is_scan_mode()


def test_scan_mode_label_survives_update_mode_ui(qapp: QApplication):
    # The button label must not silently revert to "Run Batch Fit" while scan
    # mode is still on (e.g. after a selection change re-runs _update_mode_ui).
    panel = FitPanel()
    tab = panel._global_tab
    tab._scan_mode_check.setChecked(True)
    assert tab._fit_btn.text() == "Build Integral Scan"
    tab.set_datasets([])  # triggers _update_mode_ui
    assert tab._fit_btn.text() == "Build Integral Scan"
    assert tab.is_scan_mode()


# --- main-window scan build --------------------------------------------------


def test_on_scan_requested_builds_scan_series(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    monkeypatch.setattr(
        mw, "_active_representation_type", lambda: RepresentationType.TIME_FB_ASYMMETRY
    )

    datasets = [
        _ds(11, fwd=110.0, bwd=90.0, field=100.0),  # A = (440-360)/800 = 0.1
        _ds(12, fwd=120.0, bwd=80.0, field=200.0),  # A = (480-320)/800 = 0.2
        _ds(13, fwd=130.0, bwd=70.0, field=300.0),  # A = (520-280)/800 = 0.3
    ]
    mw._fit_panel.set_datasets(datasets)

    mw._on_scan_requested()

    scans = [s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-")]
    assert len(scans) == 1
    series = scans[0]
    assert series.canonical_model is None
    assert set(series.member_run_numbers) == {11, 12, 13}

    by_run = {
        rn: summary["parameters"]["Integral asymmetry"]
        for rn, summary in series.results_by_run.items()
    }
    assert by_run[11] == pytest.approx(0.1)
    assert by_run[12] == pytest.approx(0.2)
    assert by_run[13] == pytest.approx(0.3)

    # The trend panel picked the scan up via the pull-based refresh.
    assert series.batch_id in mw._fit_parameters_panel._group_fit_results


def test_deleting_scan_series_does_not_clear_run_fits(mainwindow: MainWindow, monkeypatch):
    # A computed scan series owns no per-run FitSlots, so deleting it must not
    # clear the fit overlays of runs it shares with a real fit.
    mw = mainwindow
    monkeypatch.setattr(
        mw, "_active_representation_type", lambda: RepresentationType.TIME_FB_ASYMMETRY
    )
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0)])
    mw._on_scan_requested()
    scan = next(s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-"))

    cleared: list[object] = []
    monkeypatch.setattr(mw._fit_panel, "clear_fits_for_runs", lambda runs: cleared.append(runs))
    monkeypatch.setattr(mw._plot_panel, "clear_fits_for_runs", lambda runs: cleared.append(runs))

    mw._on_series_delete_requested(scan.batch_id)

    assert cleared == []  # no per-run fit state touched
    assert mw._project_model.batch(scan.batch_id) is None  # series removed


def test_on_scan_requested_needs_two_runs(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    monkeypatch.setattr(
        mw, "_active_representation_type", lambda: RepresentationType.TIME_FB_ASYMMETRY
    )
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0)])

    mw._on_scan_requested()

    scans = [s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-")]
    assert scans == []


def test_on_scan_requested_rejects_non_fb_representation(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    monkeypatch.setattr(mw, "_active_representation_type", lambda: RepresentationType.FREQ_FFT)
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0)])

    mw._on_scan_requested()

    scans = [s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-")]
    assert scans == []
