"""GUI tests for the integral-scan representation (ALC field scan).

Selecting "Integral scan" routes the central workspace to the dedicated scan
panel (scan plot + integration-window time strip) and swaps the Fit and
Parameters docks to the build panel and the Baseline/Peaks/RF analysis
section. Building integrates each selected run's asymmetry over the fit-range
window (percent units) and records a model-less ``FitSeries`` rendered on the
central scan canvas; clicking a scan point excludes/restores that run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

pyside6 = pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # type: ignore
from PySide6.QtWidgets import QApplication, QTableWidgetItem  # type: ignore

import asymmetry.gui.mainwindow as mw_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.representation import RepresentationType
from asymmetry.core.representation.series import FitSeries
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
    """Enter the integral-scan representation (the old ALC mode toggle)."""
    mw._plot_workspace.set_active_view("integral_scan")


# --- view + dock swap --------------------------------------------------------


def test_alc_view_swaps_docks(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    assert mw._alc_mode is True
    assert mw._fit_stack.currentWidget() is mw._alc_fit_panel
    # The Parameters dock shows the scan's analysis section (Baseline/Peaks/RF);
    # the scan plot itself occupies the central workspace panel.
    assert mw._parameters_stack.currentWidget() is mw._alc_scan_view.analysis_widget()
    assert mw._plot_workspace.current_panel() is mw._integral_scan_panel

    mw._plot_workspace.set_active_view("fb_asymmetry")
    assert mw._alc_mode is False
    assert mw._parameters_stack.currentWidget() is mw._fit_parameters_panel
    assert mw._fit_stack.currentWidget() is mw._fit_panel
    assert mw._plot_workspace.current_panel() is mw._plot_panel


def test_alc_view_reachable_from_frequency(mainwindow: MainWindow):
    # As a representation, the integral scan is one click away from any view.
    mw = mainwindow
    mw._plot_workspace.set_active_view("frequency")
    mw._plot_workspace.set_active_view("integral_scan")
    assert mw._alc_mode is True
    assert mw._fit_stack.currentWidget() is mw._alc_fit_panel


def test_integral_scan_button_present_and_enabled(mainwindow: MainWindow):
    # The integral scan is a Time-cluster representation, always available
    # (it needs only the F-B reduction); the two-run rule is enforced at
    # build time, not on the button.
    assert mainwindow._plot_workspace.active_view() == "fb_asymmetry"
    btn = mainwindow._domain_buttons_by_token["integral_scan"]
    assert btn.isEnabled()
    assert not btn.isChecked()


def test_integral_scan_button_checked_state_tracks_view(mainwindow: MainWindow):
    # Switching views drives the checked state through the real signal path.
    mw = mainwindow
    btn = mw._domain_buttons_by_token["integral_scan"]
    mw._plot_workspace.set_active_view("integral_scan")
    assert btn.isChecked()
    mw._plot_workspace.set_active_view("frequency")
    assert not btn.isChecked()
    mw._plot_workspace.set_active_view("integral_scan")
    assert btn.isChecked()


def test_integral_scan_is_remembered_as_time_view(mainwindow: MainWindow, monkeypatch):
    # The integral scan is a primary time view: leaving for the frequency
    # domain and coming "back to time" lands on it again (unlike the
    # raw-counts / reconstruction diagnostics).
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._plot_workspace.set_active_view("frequency")
    assert mw._alc_mode is False
    mw._plot_workspace.set_active_domain("time")
    assert mw._plot_workspace.active_view() == "integral_scan"
    assert mw._alc_mode is True


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
    assert mw._alc_scan_view.point_count() == 3


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
    assert view.point_count() == 3

    # Switching the x-axis to temperature re-renders the same 3 points.
    view._x_combo.setCurrentIndex(1)  # "T (K)" -> triggers options_changed
    assert view.x_key() == "temperature"
    assert view.point_count() == 3


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
    assert view.point_count() == 3

    # dA/dB derivative has one fewer point (midpoints between adjacent runs).
    view._derivative_check.setChecked(True)
    assert view.derivative_enabled() is True
    assert view.point_count() == 2


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


def test_alc_baseline_combo_offers_higher_order_models(qapp: QApplication):
    # Cubic alone tops out too low for a steep 0–3 T muonium-repolarisation
    # envelope (corpus MED #6); the combo must expose degree 4/5/6 baselines.
    view = ALCScanView()
    combo = view._baseline_model_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    for name in ("Linear", "Constant", "Cubic", "Quartic", "Quintic", "Sextic"):
        assert name in items, name
    # The selected text round-trips through baseline_model() (the value passed
    # straight to fit_scan_baseline + persisted in the ALC series state).
    combo.setCurrentIndex(combo.findText("Sextic"))
    assert view.baseline_model() == "Sextic"


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
    # The last row's edges are reversed; a region is its two edges regardless of
    # order, so it normalises to (10, 50) rather than being silently dropped.
    _set_regions(view, [(0.0, 100.0), (200.0, 300.0), (50.0, 10.0)])
    assert view.baseline_regions() == [(0.0, 100.0), (200.0, 300.0), (10.0, 50.0)]


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


# --- staleness / invalidation (review fixes #1-#5) ---------------------------


def test_alc_region_edit_invalidates_corrected_scan(mainwindow: MainWindow, monkeypatch):
    # #1: editing a baseline region after a fit must discard the corrected scan,
    # so a later peak fit can't run against a baseline that no longer matches.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    _synthetic_alc_points(mw)
    mw._render_alc_scan()
    view = mw._alc_scan_view
    _set_regions(view, [(0.0, 100.0), (200.0, 300.0)])
    mw._on_fit_baseline()
    assert mw._alc_corrected_scan is not None

    view._regions_table.item(0, 1).setText("120")  # move a region edge
    assert mw._alc_corrected_scan is None
    assert mw._alc_baseline_curve is None
    assert view._baseline_curve is None  # red overlay dropped too


def test_alc_peak_specs_rejects_invalid_row(qapp: QApplication):
    # #2: an unparseable peak row raises (naming the row) instead of being
    # silently skipped, which would misalign fitted values written back by row.
    view = ALCScanView()
    _seed_view_scan(view)
    view._add_peak("Gaussian")
    view._peaks_table.item(0, 1).setText("not-a-number")
    with pytest.raises(ValueError, match="Peak 1"):
        view.peak_specs()


def test_alc_fit_peaks_aborts_on_invalid_row(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    _synthetic_alc_points(mw)
    mw._render_alc_scan()
    mw._alc_corrected_scan = mw._alc_display_scan("field")
    monkeypatch.setattr(mw_module.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    view = mw._alc_scan_view
    view._add_peak("Gaussian")
    view._peaks_table.item(0, 2).setText("")  # blank width
    mw._on_fit_peaks()
    assert view._fit_curve is None  # no fit ran


def test_alc_clear_empties_analysis_tables(qapp: QApplication):
    # #3: when the scan goes away, its regions/peaks/results go with it.
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(0.0, 50.0)])
    view._add_peak("Gaussian")
    assert view._regions_table.rowCount() == 1
    assert view._peaks_table.rowCount() == 1
    view.clear()
    assert view._regions_table.rowCount() == 0
    assert view._peaks_table.rowCount() == 0


def test_alc_inverted_region_normalises(qapp: QApplication):
    # #4: dragging an edge past the other reverses the cells; the region is its
    # two edges regardless of order, not silently dropped.
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(150.0, 100.0)])
    assert view.baseline_regions() == [(100.0, 150.0)]


def test_alc_peak_edit_clears_stale_fit_overlay(qapp: QApplication):
    # #5: editing/dragging a peak after a fit drops the now-stale total-fit
    # overlay (the marker would otherwise diverge from the drawn curve).
    view = ALCScanView()
    _seed_view_scan(view)
    view._add_peak("Gaussian")
    view._fit_curve = np.zeros(4)  # simulate an existing fit overlay
    view._peaks_table.item(0, 1).setText("123")  # move the peak centre
    assert view._fit_curve is None


# --- persistence (G4) --------------------------------------------------------


def test_alc_scan_view_analysis_state_round_trip(qapp: QApplication):
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(10.0, 50.0), (80.0, 120.0)])
    view._add_peak("Lorentzian")
    view._peaks_table.item(0, 1).setText("123")
    state = view.analysis_state()
    assert state["regions"] == [[10.0, 50.0], [80.0, 120.0]]
    assert state["peaks"][0][0] == "Lorentzian"

    view2 = ALCScanView()
    view2.restore_analysis_state(state)
    assert view2.baseline_regions() == [(10.0, 50.0), (80.0, 120.0)]
    specs = view2.peak_specs()
    assert specs[0]["component"] == "LorentzianLCR"
    assert specs[0]["B0"] == pytest.approx(123.0)
    assert view2.baseline_model() == view.baseline_model()
    assert view2.x_key() == view.x_key()


def test_alc_persistence_round_trip(mainwindow: MainWindow, monkeypatch):
    # Build a scan with a dip, fit baseline + peaks, then save the analysis onto
    # the series and restore it into a fresh window — scan + fits come back.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    datasets = [
        _ds(11, 110.0, 90.0, 100.0),
        _ds(12, 110.0, 90.0, 150.0),
        _ds(13, 105.0, 95.0, 200.0),  # dip (A = 0.05)
        _ds(14, 110.0, 90.0, 250.0),
        _ds(15, 110.0, 90.0, 300.0),
    ]
    mw._fit_panel.set_datasets(datasets)
    mw._on_scan_requested()
    view = mw._alc_scan_view
    _set_regions(view, [(50.0, 175.0), (225.0, 350.0)])
    mw._on_fit_baseline()
    view._add_peak("Gaussian")  # default guess B0 = mid-range = 200 (the dip)
    mw._on_fit_peaks()
    assert mw._alc_corrected_scan is not None
    assert view._fit_curve is not None

    # Save: the analysis is stamped onto the scan series' `extra`.
    mw._sync_alc_series_extra()
    saved = mw._project_model.batch(mw._alc_scan_series_id).to_dict()
    assert saved["extra"]["kind"] == "alc_scan"
    assert saved["extra"]["regions"] == [[50.0, 175.0], [225.0, 350.0]]
    assert saved["extra"]["baseline_fitted"] is True
    assert saved["extra"]["peaks_fitted"] is True
    assert saved["extra"]["mode_active"] is True  # ALC mode was on at save

    # Load into a fresh window: scan points + analysis reconstructed, fits re-run.
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    mw2 = MainWindow()
    for ds in datasets:
        mw2._data_browser.add_dataset(ds)
    mw2._project_model.add_batch(FitSeries.from_dict(saved))
    # The fresh window opens in the F-B view, so the saved mode_active flag
    # resumes the integral-scan representation on restore.
    mw2._restore_alc_scan()

    assert {p["run"] for p in mw2._alc_scan_points} == {11, 12, 13, 14, 15}
    assert mw2._alc_scan_view.baseline_regions() == [(50.0, 175.0), (225.0, 350.0)]
    assert len(mw2._alc_scan_view.peak_specs()) == 1
    assert mw2._alc_corrected_scan is not None  # baseline re-fit on load
    assert mw2._alc_scan_view._fit_curve is not None  # peak overlay re-fit
    assert mw2._alc_mode is True  # integral-scan view resumed
    assert mw2._plot_workspace.active_view() == "integral_scan"


def test_alc_analysis_state_is_json_serialisable(qapp: QApplication):
    import json

    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(10.0, 50.0)])
    view._add_peak("Gaussian")
    json.dumps(view.analysis_state())  # must not raise (e.g. no numpy scalars)


def test_alc_analysis_state_persists_valid_regions_only(qapp: QApplication):
    # A degenerate (inverted/zero-width) region must not be persisted as a
    # phantom region the fit would silently ignore.
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(10.0, 50.0), (90.0, 90.0)])  # second is zero-width
    assert view.analysis_state()["regions"] == [[10.0, 50.0]]


def test_alc_restore_tolerates_malformed_extra(mainwindow: MainWindow):
    # A computed series with a garbage `extra` must not raise out of restore.
    mw = mainwindow
    mw._data_browser.add_dataset(_ds(11, 110.0, 90.0, 100.0))
    mw._data_browser.add_dataset(_ds(12, 120.0, 80.0, 200.0))
    series = FitSeries(
        "scan-9",
        _FB,
        results_by_run={
            11: {"parameters": {mw._SCAN_QUANTITY: 10.0}},
            12: {"parameters": {mw._SCAN_QUANTITY: 20.0}},
        },
        extra={
            "kind": "alc_scan",
            "regions": [123, None, [1.0, 2.0]],  # garbage + one good
            "peaks": [["Gaussian", "x", 1, 1], ["NotAShape", 1, 1, 1]],  # both bad
        },
    )
    mw._project_model.add_batch(series)
    mw._restore_alc_scan()  # must not raise
    assert mw._alc_scan_view.baseline_regions() == [(1.0, 2.0)]  # only the good row
    assert mw._alc_scan_view.peak_specs() == []  # both peaks skipped


def test_alc_data_table_dialog(mainwindow: MainWindow, monkeypatch):
    # The per-point table now lives in a separate dialog (to free plot space).
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets(
        [_ds(11, 110.0, 90.0, 100.0), _ds(12, 120.0, 80.0, 200.0), _ds(13, 130.0, 70.0, 300.0)]
    )
    mw._on_scan_requested()
    view = mw._alc_scan_view
    assert view.point_count() == 3
    view._on_show_data_table()
    assert view._data_table is not None
    assert view._data_table.rowCount() == 3


def test_alc_baseline_combo_offers_cubic(qapp: QApplication):
    # Issue 1: the WiMDA/Mantid-prescribed cubic ALC background is selectable.
    view = ALCScanView()
    options = [
        view._baseline_model_combo.itemText(i) for i in range(view._baseline_model_combo.count())
    ]
    assert "Cubic" in options
    view._baseline_model_combo.setCurrentText("Cubic")
    assert view.baseline_model() == "Cubic"


def test_alc_analysis_sections_live_in_scroll_area(qapp: QApplication):
    # Issue 2: the fitted-parameter (Baseline/Peaks) sections are wrapped in a
    # scroll area so they stay reachable instead of falling below the fold when
    # the plot grabs the dock height.
    from PySide6.QtWidgets import QScrollArea

    view = ALCScanView()
    assert isinstance(view._analysis_scroll, QScrollArea)
    assert view._analysis_scroll.widgetResizable()
    # Both parameter tables are descendants of the scroll area's widget.
    content = view._analysis_scroll.widget()
    assert content.isAncestorOf(view._regions_table)
    assert content.isAncestorOf(view._peaks_table)


def _seed_view_scan(view: ALCScanView) -> None:
    view.show_scan(
        np.array([0.0, 100.0, 200.0, 300.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.full(4, 0.1),
        [1, 2, 3, 4],
        x_label="B (G)",
        y_label="A (%)",
    )


def test_alc_drag_handles_lists_region_edges_and_peaks(qapp: QApplication):
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(10.0, 90.0)])
    view._add_peak("Gaussian")  # default B0 = mid-range = 150
    handles = view._drag_handles()  # (x, (kind, row, col))
    assert sorted({key[0] for _x, key in handles}) == ["peak", "region"]
    assert sorted(x for x, key in handles if key[0] == "region") == [10.0, 90.0]


def test_alc_region_edge_drag_updates_table(qapp: QApplication):
    view = ALCScanView()
    _seed_view_scan(view)
    _set_regions(view, [(0.0, 100.0)])
    view._drag = ("region", 0, 1)  # grab the end edge
    view._on_canvas_motion(SimpleNamespace(inaxes=view._ax, xdata=150.0))
    assert view.baseline_regions() == [(0.0, 150.0)]
    view._on_canvas_release(SimpleNamespace())
    assert view._drag is None


def test_alc_peak_centre_drag_updates_b0(qapp: QApplication):
    view = ALCScanView()
    _seed_view_scan(view)
    view._add_peak("Gaussian")
    view._drag = ("peak", 0, 1)
    view._on_canvas_motion(SimpleNamespace(inaxes=view._ax, xdata=175.0))
    assert view.peak_specs()[0]["B0"] == pytest.approx(175.0)


def test_alc_x_axis_change_clears_analysis(mainwindow: MainWindow, monkeypatch):
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
    _set_regions(view, [(0.0, 50.0)])
    view._add_peak("Gaussian")
    assert view.baseline_regions() and view.peak_specs()

    view._x_combo.setCurrentIndex(1)  # field -> temperature clears the analysis
    assert view.baseline_regions() == []
    assert view.peak_specs() == []


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
    assert view.point_count() == 3
    view.clear()
    assert view.point_count() == 0


def test_trend_panel_scan_quantity_ylabel_is_drawable(qapp: QApplication):
    """Regression: the scan's free-text quantity name ('Integral asymmetry
    (%)') was mathtext-wrapped by the trend panel's label formatter, and
    matplotlib raised ValueError at draw time whenever the trend panel
    refreshed with the scan series loaded (e.g. on entering the
    integral-scan view)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from asymmetry.gui.panels.fit_parameters_panel import _format_plot_label

    label = _format_plot_label("Integral asymmetry (%)")
    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.set_ylabel(label)
    canvas.draw()  # raised ValueError (mathtext ParseException) before the fix


# --- central panel, exclusions, provenance, and the time strip ----------------


def test_alc_point_toggle_excludes_and_restores(mainwindow: MainWindow, monkeypatch):
    # Clicking a scan point (the point_toggled signal) excludes that run: it is
    # rendered greyed, skipped by the fit-facing scan, and shown in the
    # provenance line; a second toggle restores it.
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

    mw._alc_scan_view.point_toggled.emit(12)
    assert mw._alc_excluded_runs == {12}
    # Fits never see the excluded run; the renderer still shows it (greyed).
    assert mw._alc_display_scan("field").run_numbers == [11, 13]
    full = mw._alc_display_scan("field", include_excluded=True)
    assert full.run_numbers == [11, 12, 13]
    assert mw._alc_scan_view._last_plot["excluded"].tolist() == [False, True, False]
    assert "1 excluded" in mw._alc_scan_view._provenance_label.text()

    mw._alc_scan_view.point_toggled.emit(12)
    assert mw._alc_excluded_runs == set()
    assert mw._alc_scan_view._last_plot["excluded"].tolist() == [False, False, False]


def test_alc_exclusions_persist_and_survive_rebuild(mainwindow: MainWindow, monkeypatch):
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    datasets = [
        _ds(11, 110.0, 90.0, 100.0),
        _ds(12, 120.0, 80.0, 200.0),
        _ds(13, 130.0, 70.0, 300.0),
    ]
    mw._fit_panel.set_datasets(datasets)
    mw._on_scan_requested()
    mw._on_alc_point_toggled(12)

    # Rebuild over the same selection: the exclusion is keyed by run and stays.
    mw._on_scan_requested()
    assert mw._alc_excluded_runs == {12}
    assert mw._alc_scan_view._last_plot["excluded"].tolist() == [False, True, False]

    # Persist onto the series and restore into a fresh window.
    mw._sync_alc_series_extra()
    saved = mw._project_model.batch(mw._alc_scan_series_id).to_dict()
    assert saved["extra"]["excluded_runs"] == [12]
    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    mw2 = MainWindow()
    for ds in datasets:
        mw2._data_browser.add_dataset(ds)
    mw2._project_model.add_batch(FitSeries.from_dict(saved))
    mw2._restore_alc_scan()
    assert mw2._alc_excluded_runs == {12}
    assert mw2._alc_scan_view._last_plot["excluded"].tolist() == [False, True, False]

    # Rebuild over a selection that no longer contains the run: pruned.
    mw._fit_panel.set_datasets([datasets[0], datasets[2]])
    mw._on_scan_requested()
    assert mw._alc_excluded_runs == set()


def test_alc_baseline_fit_skips_excluded_points(mainwindow: MainWindow, monkeypatch):
    # The dip run is excluded, so a flat baseline over the remaining points
    # leaves an (essentially) zero-corrected scan of 4 points.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._fit_panel.set_datasets(
        [
            _ds(11, 110.0, 90.0, 100.0),
            _ds(12, 110.0, 90.0, 150.0),
            _ds(13, 105.0, 95.0, 200.0),  # dip (A = 0.05)
            _ds(14, 110.0, 90.0, 250.0),
            _ds(15, 110.0, 90.0, 300.0),
        ]
    )
    mw._on_scan_requested()
    mw._on_alc_point_toggled(13)
    _set_regions(mw._alc_scan_view, [(50.0, 350.0)])
    mw._on_fit_baseline()
    corrected = mw._alc_corrected_scan
    assert corrected is not None
    assert corrected.run_numbers == [11, 12, 14, 15]
    assert np.allclose(corrected.value, 0.0, atol=1e-9)
    # The overlay carries its own x (the included points), shorter than the
    # rendered scan (which still shows the greyed excluded point).
    bx, _by = mw._alc_scan_view._baseline_curve
    assert bx.size == 4
    assert mw._alc_scan_view._last_plot["x"].size == 5


def test_alc_canvas_click_on_point_emits_toggle(qapp: QApplication):
    # A stationary click on a data point emits point_toggled for its run; a
    # click on empty axes space does not.
    from types import SimpleNamespace

    view = ALCScanView()
    view.resize(600, 500)
    view.show_scan(
        np.array([100.0, 200.0, 300.0]),
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 1.0, 1.0]),
        [11, 12, 13],
        x_label="B (G)",
        y_label="A (%)",
    )
    view._canvas.draw()
    toggled: list[int] = []
    view.point_toggled.connect(toggled.append)

    px, py = view._ax.transData.transform((200.0, 20.0))
    event = SimpleNamespace(inaxes=view._ax, button=1, x=px, y=py, xdata=200.0, ydata=20.0)
    view._on_canvas_press(event)
    view._on_canvas_release(event)
    assert toggled == [12]

    # Far from any point: no toggle.
    px2, py2 = view._ax.transData.transform((150.0, 28.0))
    event2 = SimpleNamespace(inaxes=view._ax, button=1, x=px2, y=py2, xdata=150.0, ydata=28.0)
    view._on_canvas_press(event2)
    view._on_canvas_release(event2)
    assert toggled == [12]

    # A drag that starts on a point is not a click: no toggle.
    view._on_canvas_press(event)
    moved = SimpleNamespace(inaxes=view._ax, button=1, x=px + 20.0, y=py, xdata=210.0, ydata=20.0)
    view._on_canvas_motion(moved)
    view._on_canvas_release(moved)
    assert toggled == [12]

    # A right/middle-button release mid-gesture must not complete the toggle.
    view._on_canvas_press(event)
    right_release = SimpleNamespace(inaxes=view._ax, button=3, x=px, y=py, xdata=200.0, ydata=20.0)
    view._on_canvas_release(right_release)
    assert toggled == [12]
    view._on_canvas_release(event)  # the left release still completes it
    assert toggled == [12, 12]


def test_alc_canvas_click_disarmed_for_derivative_view(qapp: QApplication):
    # The derivative view marks its points non-toggleable (no greyed marker
    # exists there to restore an exclusion), so a click must not exclude.
    from types import SimpleNamespace

    view = ALCScanView()
    view.resize(600, 500)
    view.show_scan(
        np.array([150.0, 250.0]),
        np.array([0.1, 0.1]),
        np.array([0.01, 0.01]),
        [12, 13],
        x_label="B (G)",
        y_label="dA/dx (%/G)",
        toggleable=False,
    )
    view._canvas.draw()
    toggled: list[int] = []
    view.point_toggled.connect(toggled.append)
    px, py = view._ax.transData.transform((150.0, 0.1))
    event = SimpleNamespace(inaxes=view._ax, button=1, x=px, y=py, xdata=150.0, ydata=0.1)
    view._on_canvas_press(event)
    view._on_canvas_release(event)
    assert toggled == []


def test_alc_canvas_click_ignores_nonfinite_points(qapp: QApplication):
    # A NaN point must not win the nearest-point test for every click.
    from types import SimpleNamespace

    view = ALCScanView()
    view.resize(600, 500)
    view.show_scan(
        np.array([100.0, 200.0, 300.0]),
        np.array([10.0, np.nan, 30.0]),
        np.array([1.0, 1.0, 1.0]),
        [11, 12, 13],
        x_label="B (G)",
        y_label="A (%)",
    )
    view._canvas.draw()
    toggled: list[int] = []
    view.point_toggled.connect(toggled.append)
    # Click far from every finite point: nothing toggles (previously the NaN
    # point's NaN distance passed the tolerance test for any click).
    px, py = view._ax.transData.transform((150.0, 25.0))
    event = SimpleNamespace(inaxes=view._ax, button=1, x=px, y=py, xdata=150.0, ydata=25.0)
    view._on_canvas_press(event)
    view._on_canvas_release(event)
    assert toggled == []


def test_alc_build_drops_surface_in_provenance(mainwindow: MainWindow, monkeypatch):
    # A run that cannot be integrated (no grouping) is dropped at build time
    # and surfaces in the provenance line, not just the log.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    good1 = _ds(11, 110.0, 90.0, 100.0)
    good2 = _ds(12, 120.0, 80.0, 200.0)
    bad = _ds(13, 130.0, 70.0, 300.0)
    bad.run.grouping = None
    mw._fit_panel.set_datasets([good1, good2, bad])
    mw._on_scan_requested()
    text = mw._alc_scan_view._provenance_label.text()
    assert "2 runs in scan" in text
    assert "1 dropped at build" in text
    assert "13" in mw._alc_scan_view._provenance_label.toolTip()


def test_integral_strip_syncs_with_fit_range(mainwindow: MainWindow, monkeypatch):
    # The strip mirrors the canonical fit-range (the time plot panel) in both
    # directions: panel -> strip echo, and strip drag -> panel + spinboxes.
    mw = mainwindow
    _enter_alc(mw, monkeypatch)
    mw._plot_panel.set_fit_range(0.5, 7.0)
    strip = mw._integral_time_strip
    assert strip.window() == (0.5, 7.0)

    strip.window_edited.emit(1.0, 5.0)
    assert mw._plot_panel.get_fit_range() == (1.0, 5.0)
    assert mw._alc_fit_panel._min_spin.value() == pytest.approx(1.0)
    assert mw._alc_fit_panel._max_spin.value() == pytest.approx(5.0)
    assert strip.window() == (1.0, 5.0)


def test_integral_strip_tracks_selection_dataset(mainwindow: MainWindow, monkeypatch):
    # Entering the scan view (and selection changes while in it) feed the strip
    # the current run's spectrum for the window preview.
    mw = mainwindow
    ds = _ds(11, 110.0, 90.0, 100.0)
    mw._data_browser.add_dataset(ds)
    mw._current_dataset = ds
    _enter_alc(mw, monkeypatch)
    strip = mw._integral_time_strip
    assert strip._time is not None
    assert strip._time.size == ds.time.size
    assert "11" in strip._window_label.text() or strip._run_label != ""


def test_integral_strip_drag_normalises_and_emits(qapp: QApplication):
    from types import SimpleNamespace

    from asymmetry.gui.panels.alc_panel import IntegralTimeStrip

    strip = IntegralTimeStrip()
    strip.resize(600, 200)
    t = np.linspace(0.0, 10.0, 200)
    strip.show_dataset(t, np.zeros_like(t))
    strip.set_window(2.0, 8.0)
    strip._canvas.draw()
    committed: list[tuple[float, float]] = []
    strip.window_edited.connect(lambda a, b: committed.append((a, b)))

    # Grab the max edge and drag it past the min edge: the committed window is
    # normalised (min <= max).
    px, _py = strip._ax.transData.transform((8.0, 0.0))
    press = SimpleNamespace(inaxes=strip._ax, button=1, x=px, xdata=8.0)
    strip._on_press(press)
    assert strip._drag_edge == 1
    move_px, _ = strip._ax.transData.transform((1.0, 0.0))
    move = SimpleNamespace(inaxes=strip._ax, button=1, x=move_px, xdata=1.0)
    strip._on_motion(move)
    strip._on_release(move)
    assert committed == [(1.0, 2.0)]
    assert strip.window() == (1.0, 2.0)


def test_workspace_routes_integral_scan_to_scan_panel(mainwindow: MainWindow):
    mw = mainwindow
    ws = mw._plot_workspace
    assert ws.scan_panel() is mw._integral_scan_panel
    ws.set_active_view("integral_scan")
    assert ws.current_panel() is mw._integral_scan_panel
    assert ws.active_domain() == "time"  # the scan is a time-domain view
    ws.set_active_view("fb_asymmetry")
    assert ws.current_panel() is mw._plot_panel
