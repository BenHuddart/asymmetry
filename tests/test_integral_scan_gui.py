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
from PySide6.QtWidgets import QApplication, QTableWidgetItem  # type: ignore

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


def _ds(
    run_number: int, fwd: float, bwd: float, field: float, temperature: float = 10.0
) -> MuonDataset:
    n = 4
    meta = {"run_number": run_number, "field": field, "temperature": temperature}
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=np.full(n, fwd), bin_width=0.01),
            Histogram(counts=np.full(n, bwd), bin_width=0.01),
        ],
        metadata=dict(meta),
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
        metadata=dict(meta),
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


def test_alc_x_axis_selector_reorders(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets(
        [
            _ds(11, 110.0, 90.0, field=100.0, temperature=5.0),
            _ds(12, 120.0, 80.0, field=200.0, temperature=15.0),
            _ds(13, 130.0, 70.0, field=300.0, temperature=25.0),
        ]
    )
    mw._on_scan_requested()
    view = mw._alc_scan_view
    assert view.x_key() == "field"
    assert view._table.rowCount() == 3

    # Switching the x-axis to temperature re-renders the same 3 points.
    view._x_combo.setCurrentIndex(1)  # "T (K)" -> triggers options_changed
    assert view.x_key() == "temperature"
    assert view._table.rowCount() == 3


def test_alc_derivative_toggle(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets(
        [
            _ds(11, 110.0, 90.0, 100.0),
            _ds(12, 120.0, 80.0, 200.0),
            _ds(13, 130.0, 70.0, 300.0),
        ]
    )
    mw._on_scan_requested()
    view = mw._alc_scan_view
    assert view._table.rowCount() == 3

    # dA/dB derivative has one fewer point (midpoints between adjacent runs).
    view._derivative_check.setChecked(True)
    assert view.derivative_enabled() is True
    assert view._table.rowCount() == 2


def test_alc_rebuild_replaces_scan_series(mainwindow: MainWindow, monkeypatch):
    # Iterating the window (rebuilding) must replace the scan series, not
    # accumulate a new one each time.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets([_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0)])
    mw._on_scan_requested()
    mw._on_scan_requested()
    mw._on_scan_requested()
    scans = [s for s in mw._project_model.batches.values() if s.batch_id.startswith("scan-")]
    assert len(scans) == 1
    assert scans[0].order_key == "run"


def test_alc_derivative_label_follows_x_axis(qapp: QApplication):
    view = ALCScanView()
    assert view._derivative_check.text() == "dA/dB"
    view._x_combo.setCurrentIndex(1)  # T (K)
    assert view._derivative_check.text() == "dA/dT"
    view._x_combo.setCurrentIndex(2)  # Run
    assert view._derivative_check.text() == "dA/d(run)"


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


def _set_regions(view, regions: list[tuple[float, float]]) -> None:
    view._regions_table.setRowCount(0)
    for lo, hi in regions:
        r = view._regions_table.rowCount()
        view._regions_table.insertRow(r)
        view._regions_table.setItem(r, 0, QTableWidgetItem(str(lo)))
        view._regions_table.setItem(r, 1, QTableWidgetItem(str(hi)))


def _synthetic_alc_points(mw: MainWindow) -> None:
    # Linear baseline (x100 -> 0.01*B + 2 %) plus a dip at B = 150.
    mw._alc_scan_points = [
        {
            "run": 100 + i,
            "value": 0.0001 * f + 0.02 + (-0.005 if f == 150 else 0.0),
            "error": 1e-4,
            "field": float(f),
            "temperature": 10.0,
        }
        for i, f in enumerate([0, 50, 100, 150, 200, 250, 300])
    ]


def test_alc_baseline_regions_parsing(qapp: QApplication):
    view = ALCScanView()
    _set_regions(view, [(0.0, 100.0), (200.0, 300.0), (50.0, 10.0)])  # last is inverted
    assert view.baseline_regions() == [(0.0, 100.0), (200.0, 300.0)]


def test_alc_baseline_fit_produces_corrected(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    _synthetic_alc_points(mw)
    view = mw._alc_scan_view
    _set_regions(view, [(0.0, 100.0), (200.0, 300.0)])  # non-resonant edges

    mw._on_fit_baseline()

    assert mw._alc_corrected_scan is not None
    corrected = mw._alc_corrected_scan
    flat = (corrected.x <= 100.0) | (corrected.x >= 200.0)
    assert np.allclose(corrected.value[flat], 0.0, atol=0.05)  # baseline removed
    # The dip at B=150 survives (clearly negative after subtraction).
    dip = corrected.value[np.argmin(np.abs(corrected.x - 150.0))]
    assert dip < -0.2


def test_alc_baseline_requires_regions(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    _synthetic_alc_points(mw)
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    mw._on_fit_baseline()  # no regions set
    assert mw._alc_corrected_scan is None


def _gauss(x, amp, b0, w):
    return amp * np.exp(-0.5 * ((x - b0) / w) ** 2)


def test_alc_peak_specs_and_add_remove(qapp: QApplication):
    view = ALCScanView()
    view._add_peak("Gaussian")
    view._add_peak("Lorentzian")
    specs = view.peak_specs()
    assert [s["component"] for s in specs] == ["GaussianLCR", "LorentzianLCR"]
    view._peaks_table.setCurrentCell(0, 0)
    view._remove_peak()
    assert len(view.peak_specs()) == 1


def test_alc_peak_fit_recovers_resonance(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    fields = np.linspace(0.0, 300.0, 31)
    # A corrected scan = a single Gaussian dip at B0 = 150 G (-5 %, width 20 G).
    mw._alc_scan_points = [
        {
            "run": 100 + i,
            "value": float(_gauss(f, -0.05, 150.0, 20.0)),
            "error": 1e-4,
            "field": float(f),
            "temperature": 10.0,
        }
        for i, f in enumerate(fields)
    ]
    mw._render_alc_scan()  # populate the view's plot state
    mw._alc_corrected_scan = mw._alc_display_scan("field")
    mw._alc_baseline_curve = np.zeros(fields.size)

    view = mw._alc_scan_view
    view._add_peak("Gaussian")  # default guess B0 = mid-range = 150
    view._peaks_table.item(0, 1).setText("140")  # offset so it's a real fit

    mw._on_fit_peaks()

    fitted_b0 = float(view._peaks_table.item(0, 1).text())
    assert abs(fitted_b0 - 150.0) < 5.0
    assert view._peaks_results.text()  # summary populated
    assert view._fit_curve is not None  # overlay drawn


def test_alc_peaks_require_baseline(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    mw._alc_corrected_scan = None
    mw._alc_scan_view._add_peak("Gaussian")
    mw._on_fit_peaks()
    assert mw._alc_scan_view._fit_curve is None


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
