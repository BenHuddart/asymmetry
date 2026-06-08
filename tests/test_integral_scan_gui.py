"""GUI tests for ALC mode (integral-asymmetry field scan).

ALC mode is a main-window toolbar toggle (enabled only for the F-B asymmetry
representation) that swaps the Fit and Parameters docks for bespoke ALC widgets:
the build panel and the scan view. Building integrates each selected run's
asymmetry over the fit-range window (percent units) and records a model-less
``FitSeries`` rendered in the scan view.
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


def test_alc_action_enabled_in_default_fb_view(mainwindow: MainWindow):
    # The app opens in the F-B asymmetry view, where ALC mode applies, so the
    # toggle is enabled at startup — no view-switch workaround needed.
    assert mainwindow._plot_workspace.active_view() == "fb_asymmetry"
    assert mainwindow._alc_mode_action.isEnabled() is True


def test_alc_enabled_predicate_tracks_representation(mainwindow: MainWindow, monkeypatch):
    # Enabled iff the active representation is F-B asymmetry, regardless of how
    # many runs are selected (the fixture has none). The two-run requirement is
    # enforced at build time, not on the toggle.
    mw = mainwindow
    cases = {
        RepresentationType.TIME_FB_ASYMMETRY: True,
        RepresentationType.TIME_GROUPS: False,
        RepresentationType.FREQ_FFT: False,
        RepresentationType.FREQ_MAXENT: False,
    }
    for rep, expected in cases.items():
        monkeypatch.setattr(mw, "_active_representation_type", lambda rep=rep: rep)
        mw._refresh_alc_mode_enabled()
        assert mw._alc_mode_action.isEnabled() is expected
        expected_tip = (
            mw_module._ALC_TOOLTIP_ENABLED if expected else mw_module._ALC_TOOLTIP_DISABLED
        )
        assert mw._alc_mode_action.toolTip() == expected_tip


def test_alc_toggle_enabled_state_follows_view_change(mainwindow: MainWindow):
    # Switching views drives the enabled flag through the real signal path.
    mw = mainwindow
    mw._plot_workspace.set_active_view("fb_asymmetry")
    assert mw._alc_mode_action.isEnabled() is True
    mw._plot_workspace.set_active_view("frequency")
    assert mw._alc_mode_action.isEnabled() is False
    mw._plot_workspace.set_active_view("fb_asymmetry")
    assert mw._alc_mode_action.isEnabled() is True


def test_alc_disabled_button_shows_tooltip_via_event_filter(mainwindow: MainWindow, monkeypatch):
    # Qt suppresses tooltips on disabled widgets; the event filter renders the
    # "switch to F-B view" hint itself when the button is disabled, and defers to
    # default handling when it is enabled.
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent

    mw = mainwindow
    if mw._alc_mode_button is None:
        pytest.skip("toolbar provides no widget for the ALC action")

    monkeypatch.setattr(mw, "_active_representation_type", lambda: RepresentationType.FREQ_FFT)
    mw._refresh_alc_mode_enabled()
    assert mw._alc_mode_action.isEnabled() is False
    disabled_event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))
    assert mw.eventFilter(mw._alc_mode_button, disabled_event) is True

    monkeypatch.setattr(mw, "_active_representation_type", lambda: _FB)
    mw._refresh_alc_mode_enabled()
    assert mw._alc_mode_action.isEnabled() is True
    enabled_event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))
    assert mw.eventFilter(mw._alc_mode_button, enabled_event) is False


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
    # Make ALC mode resumable (FB active) so the restored scan becomes visible.
    mw2._alc_mode_action.setEnabled(True)
    monkeypatch.setattr(mw2, "_active_representation_type", lambda: _FB)
    mw2._restore_alc_scan()

    assert {p["run"] for p in mw2._alc_scan_points} == {11, 12, 13, 14, 15}
    assert mw2._alc_scan_view.baseline_regions() == [(50.0, 175.0), (225.0, 350.0)]
    assert len(mw2._alc_scan_view.peak_specs()) == 1
    assert mw2._alc_corrected_scan is not None  # baseline re-fit on load
    assert mw2._alc_scan_view._fit_curve is not None  # peak overlay re-fit
    assert mw2._alc_mode is True  # ALC mode resumed


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
