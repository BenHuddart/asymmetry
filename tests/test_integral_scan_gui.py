"""GUI tests for ALC mode (integral-asymmetry field scan).

ALC mode is a main-window toolbar toggle (enabled only for the F-B asymmetry
representation) that swaps the Fit and Parameters docks for bespoke ALC widgets:
the build panel and the scan view. Building integrates each selected run's
asymmetry over the fit-range window (percent units) and records a model-less
``FitSeries`` rendered in the scan view.
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
from asymmetry.gui.panels.alc_panel import ALCFitPanel, ALCScanView

_FB = RepresentationType.TIME_FB_ASYMMETRY


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


def _enter_alc(mw: MainWindow, monkeypatch) -> None:
    monkeypatch.setattr(mw, "_active_representation_type", lambda: _FB)
    mw._alc_mode_action.setChecked(True)


# --- toggle + dock swap ------------------------------------------------------


def test_alc_toggle_swaps_docks(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    assert mw._alc_mode is True
    assert mw._fit_stack.currentWidget() is mw._alc_fit_panel
    assert mw._parameters_stack.currentWidget() is mw._alc_scan_view

    mw._alc_mode_action.setChecked(False)
    assert mw._alc_mode is False
    assert mw._parameters_stack.currentWidget() is mw._fit_parameters_panel
    assert mw._fit_stack.currentWidget() is mw._fit_panel


def test_alc_toggle_guarded_to_fb_representation(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    monkeypatch.setattr(mw, "_active_representation_type", lambda: RepresentationType.FREQ_FFT)
    mw._alc_mode_action.setChecked(True)
    assert mw._alc_mode is False
    assert mw._alc_mode_action.isChecked() is False  # guard reverted it


def test_alc_action_starts_disabled(mainwindow: MainWindow):
    # The toolbar action is disabled until the F-B asymmetry view is active.
    assert mainwindow._alc_mode_action.isEnabled() is False


# --- build + render ----------------------------------------------------------


def test_alc_build_creates_percent_scan_and_renders(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets(
        [
            _ds(11, 110.0, 90.0, 100.0),  # A = 0.1 -> 10.0 %
            _ds(12, 120.0, 80.0, 200.0),  # A = 0.2 -> 20.0 %
            _ds(13, 130.0, 70.0, 300.0),  # A = 0.3 -> 30.0 %
        ]
    )

    mw._alc_fit_panel.build_requested.emit()  # exercises the wiring

    series = next(s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-"))
    assert series.is_computed is True
    by_run = {
        rn: summary["parameters"][mw._SCAN_QUANTITY]
        for rn, summary in series.results_by_run.items()
    }
    assert by_run[11] == pytest.approx(10.0)
    assert by_run[12] == pytest.approx(20.0)
    assert by_run[13] == pytest.approx(30.0)

    # The bespoke scan view rendered the three points.
    assert mw._alc_scan_view._table.rowCount() == 3


def test_alc_build_needs_two_runs(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0)])
    mw._on_scan_requested()
    scans = [s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-")]
    assert scans == []


def test_deleting_scan_series_does_not_clear_run_fits(mainwindow: MainWindow, monkeypatch):
    # A computed scan series owns no per-run FitSlots, so deleting it must not
    # clear the fit overlays of runs it shares with a real fit.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0)])
    mw._on_scan_requested()
    scan = next(s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-"))

    cleared: list[object] = []
    monkeypatch.setattr(mw._fit_panel, "clear_fits_for_runs", lambda runs: cleared.append(runs))
    monkeypatch.setattr(mw._plot_panel, "clear_fits_for_runs", lambda runs: cleared.append(runs))

    mw._on_series_delete_requested(scan.batch_id)
    assert cleared == []
    assert mw._project_model.batch(scan.batch_id) is None


# --- scan view widget --------------------------------------------------------


def test_alc_fit_panel_range_spinboxes(qapp: QApplication):
    panel = ALCFitPanel()
    # Spinboxes start disabled (no plot fit-range yet).
    assert panel._min_spin.isEnabled() is False

    # Plot -> panel: set_fit_range_display fills + enables the spinboxes.
    panel.set_fit_range_display(0.2, 8.0)
    assert panel._min_spin.isEnabled() is True
    assert panel._min_spin.value() == pytest.approx(0.2)
    assert panel._max_spin.value() == pytest.approx(8.0)

    # Panel -> plot: editing a spinbox emits the committed range.
    emitted: list[tuple[float, float]] = []
    panel.fit_range_edit_committed.connect(lambda lo, hi: emitted.append((lo, hi)))
    panel._max_spin.setValue(6.0)
    panel._on_spin_committed()
    assert emitted[-1] == pytest.approx((0.2, 6.0))

    # A cleared range disables the spinboxes again.
    panel.set_fit_range_display(None, None)
    assert panel._max_spin.isEnabled() is False


def test_alc_fit_panel_committed_range_drives_plot(mainwindow: MainWindow, monkeypatch):
    # The ALC panel's committed range is pushed to the active plot, mirroring
    # the regular fit-range spinbox path.
    mw = mainwindow
    pushed: list[tuple[float, float]] = []
    monkeypatch.setattr(mw._plot_panel, "set_fit_range", lambda lo, hi: pushed.append((lo, hi)))
    mw._alc_fit_panel.fit_range_edit_committed.emit(0.3, 7.5)
    assert pushed[-1] == pytest.approx((0.3, 7.5))


def test_alc_scan_view_show_and_clear(qapp: QApplication):
    view = ALCScanView()
    view.show_scan(
        np.array([0.0, 50.0, 100.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.1, 0.1]),
        [1, 2, 3],
        x_label="B (G)",
        y_label="Integral asymmetry (%)",
    )
    assert view._table.rowCount() == 3
    view.clear()
    assert view._table.rowCount() == 0
