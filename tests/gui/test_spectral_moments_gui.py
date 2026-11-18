"""GUI + persistence tests for the spectral-moments feature.

Covers eligibility gating, the on-plot window overlay, send-to-trend recording a
computed ``FitSeries`` (replace-on-resend), the ``.asymp`` round-trip of the
series + recipe, and live-setting persistence without a schema bump. See
``docs/porting/spectral-moments/verification-plan.md`` (C9–C14).
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core import computation
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.project_model import ProjectModel
from asymmetry.gui.mainwindow import MainWindow

pytestmark = pytest.mark.gui


def _spectrum(run: int, *, display: str = "phase_corrected") -> MuonDataset:
    f = np.linspace(20.0, 60.0, 1001)
    amp = np.exp(-0.5 * ((f - 40.0) / 4.0) ** 2) + 0.3 * np.exp(-0.5 * ((f - 48.0) / 5.0) ** 2)
    return MuonDataset(
        time=f,
        asymmetry=amp,
        error=0.01 * np.ones_like(f),
        metadata={
            "fourier_display": display,
            "field": 3000.0,
            "temperature": 5.0,
            "run_number": run,
            "run_label": f"R{run}",
        },
    )


class _FakeBrowser:
    """Minimal data-browser stand-in exposing the selected datasets."""

    def __init__(self, datasets):
        self._datasets = list(datasets)

    def get_selected_datasets(self):
        return list(self._datasets)


def _activate_fft_spectrum(window: MainWindow, runs, *, display="phase_corrected"):
    cache = window._frequency_cache(RepresentationType.FREQ_FFT)
    datasets = [_spectrum(r, display=display) for r in runs]
    for ds in datasets:
        cache[int(ds.run_number)] = [ds]
    window._frequency_plot_panel.plot_dataset(datasets[0])
    window._data_browser = _FakeBrowser(datasets)
    window._refresh_spectral_moments()
    _wait_moments(window)
    return datasets


def _wait_moments(window):
    import time

    from PySide6.QtCore import QCoreApplication

    for _ in range(500):
        if window._moment_readout._worker is None and window._moment_batch._worker is None:
            return
        QCoreApplication.processEvents()
        time.sleep(0.01)
    raise AssertionError("Spectral moment workers did not settle")


@pytest.fixture
def window(qapp):

    win = MainWindow()
    yield win
    win.close()


# ── C9: eligibility guard ───────────────────────────────────────────────────


def test_eligible_for_phase_corrected_fft(window):
    _activate_fft_spectrum(window, [1], display="phase_corrected")
    assert window._fourier_panel.moments_widget.is_eligible() is True


@pytest.mark.parametrize("mode", ["power", "magnitude", "phase_spectrum", "burg", "correlation"])
def test_ineligible_modes_grey_out_with_reason(window, mode):
    _activate_fft_spectrum(window, [1], display=mode)
    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is False
    assert widget.toolTip()  # explanatory tooltip present
    assert not window._frequency_plot_panel._moments_overlay_visible


def test_inactive_representation_widget_greyed(window):
    _activate_fft_spectrum(window, [1])
    # FFT is active, so the MaxEnt widget must be greyed.
    assert window._maxent_panel.moments_widget.is_eligible() is False


# ── C14: window visible on the plot ─────────────────────────────────────────


def test_eligible_spectrum_draws_window_overlay(window):
    _activate_fft_spectrum(window, [1])
    panel = window._frequency_plot_panel
    assert panel._moments_overlay_visible is True
    assert len(panel._moments_span_artists) == 3  # span + two handles


def test_ineligible_clears_overlay(window):
    _activate_fft_spectrum(window, [1])
    assert window._frequency_plot_panel._moments_overlay_visible is True
    _activate_fft_spectrum(window, [1], display="power")
    assert window._frequency_plot_panel._moments_overlay_visible is False


# ── C10/C11: send to trend records a computed series; resend replaces ────────


def _moments_batches(window):
    return [b for bid, b in window._project_model.batches.items() if bid.startswith("moments-")]


def test_send_to_trend_records_computed_series(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    batches = _moments_batches(window)
    assert len(batches) == 1
    series = batches[0]
    assert series.canonical_model is None  # computed series
    assert series.is_computed is True
    assert sorted(series.member_run_numbers) == [1, 2]
    assert series.rep_type == RepresentationType.FREQ_FFT
    row = series.results_by_run[1]
    assert row["success"] is True
    assert "B_rms_mean" in row["parameters"] and "beta" in row["parameters"]
    assert series.extra.get("moments_recipe", {}).get("unit") == "gauss"


def test_resending_same_selection_replaces(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    first_ids = [b.batch_id for b in _moments_batches(window)]
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    second_ids = [b.batch_id for b in _moments_batches(window)]
    assert first_ids == second_ids
    assert len(second_ids) == 1


def test_different_selection_is_a_new_series(window):
    datasets = _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    # Narrow the selection → a different member set → a different series.
    window._data_browser = _FakeBrowser([datasets[0]])
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    assert len(_moments_batches(window)) == 2


# ── C12: .asymp round-trip of the series + recipe ───────────────────────────


def test_moments_series_round_trips_through_project_model(window):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.1)
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    original = _moments_batches(window)[0]

    restored_model = ProjectModel.from_dict(window._project_model.to_dict())
    restored = restored_model.batches[original.batch_id]
    assert sorted(restored.member_run_numbers) == [1, 2]
    assert restored.canonical_model is None
    assert restored.results_by_run[1]["parameters"]["B_ave"] == pytest.approx(
        original.results_by_run[1]["parameters"]["B_ave"]
    )
    recipe = restored.extra.get("moments_recipe", {})
    assert recipe.get("cutoff_fraction") == pytest.approx(0.1)
    assert recipe.get("unit") == "gauss"


# ── C13: live settings persist without a schema bump ────────────────────────


def test_live_settings_persist_in_fourier_state(window):
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.2)
    widget.set_range_mhz(25.0, 55.0)
    state = window._fourier_panel.get_state()
    assert "moments" in state
    assert state["moments"]["cutoff_fraction"] == pytest.approx(0.2)
    # Restore into a fresh panel tolerates and applies it.
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    fresh = FourierPanel()
    fresh.restore_state(state)
    assert fresh.moments_widget.cutoff_fraction() == pytest.approx(0.2)
    assert fresh.moments_widget.range_mhz() == (25.0, 55.0)


def test_resending_in_a_different_unit_replaces_not_duplicates(window):
    from asymmetry.core.fourier.units import FieldUnit

    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    first = [b.batch_id for b in _moments_batches(window)]
    # Same selection + window, different display unit → still one series.
    widget._unit = FieldUnit.MHZ
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    second = [b.batch_id for b in _moments_batches(window)]
    assert first == second
    assert len(second) == 1


def test_non_overlapping_spectrum_resets_window_to_full():
    from asymmetry.gui.panels.spectral_moments_widget import SpectralMomentsWidget

    w = SpectralMomentsWidget()
    w.set_spectrum_bounds(20.0, 60.0)
    w.set_range_mhz(40.0, 55.0)  # a window inside run A
    # Switch to a run whose spectrum does not overlap [40,55].
    w.set_spectrum_bounds(100.0, 200.0)
    lo, hi = w.range_mhz()
    assert lo < hi  # not inverted
    assert (lo, hi) == (100.0, 200.0)  # fell back to the new full extent


def test_spectral_moments_info_affordance():
    """The dense readout carries a single Info affordance (mirrors the FFT panel)."""
    from asymmetry.gui.panels.spectral_moments_widget import (
        SpectralMomentsWidget,
        _build_moments_info_html,
    )

    w = SpectralMomentsWidget()
    assert w._info_btn.text() == "Info"

    html_text = _build_moments_info_html()
    # Every readout row is explained, and the unit-invariance note is present.
    for name in ("B_pk", "B_ave", "B_rms (vs mean)", "Skewness α", "Asymmetry β"):
        assert name in html_text
    assert "dimensionless" in html_text

    dialog = w._show_info() or w._info_dialog
    assert dialog is not None
    dialog.deleteLater()


def test_restore_state_tolerates_absent_moments(qapp):
    from asymmetry.gui.panels.fourier_panel import FourierPanel

    panel = FourierPanel()
    panel.restore_state({})  # no "moments" key — must not raise
    assert panel.moments_widget.cutoff_fraction() == 0.0


# ── apodisation caveat (integrity: filtered widths are not the physics) ──────


def test_apodised_spectrum_shows_moments_caveat(window):
    datasets = _activate_fft_spectrum(window, [61], display="phase_corrected")
    datasets[0].metadata["fourier_window"] = "lorentzian"
    datasets[0].metadata["fourier_filter_time_constant_us"] = 1.8
    window._refresh_spectral_moments()

    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is True
    assert not widget._caveat_label.isHidden()
    assert "lorentzian" in widget._caveat_label.text()
    assert "1.8" in widget._caveat_label.text()
    assert "broadening" in widget._caveat_label.text()


def test_unapodised_spectrum_shows_no_caveat(window):
    datasets = _activate_fft_spectrum(window, [62], display="phase_corrected")
    datasets[0].metadata["fourier_window"] = "none"
    window._refresh_spectral_moments()

    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is True
    assert widget._caveat_label.isHidden()


def test_legacy_spectrum_without_window_metadata_shows_no_caveat(window):
    _activate_fft_spectrum(window, [63], display="phase_corrected")
    window._refresh_spectral_moments()

    widget = window._fourier_panel.moments_widget
    assert widget._caveat_label.isHidden()


# ── C1: mid-drag recomputes are coalesced and bootstrap-free; release ───────
# restores the full bootstrap. See gui-responsiveness audit item C1: dragging a
# moments window/cutoff handle used to run a 256-resample bootstrap per
# mouse-move event; it now runs the cheap point-estimate path, coalesced behind
# a 30 ms single-shot timer, with exactly one full bootstrap on release.


def _wrap_spectrum_moments_counter(monkeypatch) -> list:
    """Patch mainwindow's bound ``spectrum_moments`` to record every call's
    ``uncertainty`` kwarg while still delegating to the real implementation, so
    ``_compute_and_show_moments`` still gets a real ``SpectrumMoments`` back
    (``window_peak_amplitude``, the readout attrs, etc. all need to resolve).
    """
    import asymmetry.gui.moment_tasks as mainwindow_module

    calls: list[str] = []
    original = mainwindow_module.spectrum_moments

    def counting(*args, **kwargs):
        calls.append(kwargs.get("uncertainty"))
        return original(*args, **kwargs)

    monkeypatch.setattr(mainwindow_module, "spectrum_moments", counting)
    return calls


def test_drag_burst_defers_and_coalesces_to_one_cheap_recompute(window, monkeypatch):
    _activate_fft_spectrum(window, [1])
    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is True

    calls = _wrap_spectrum_moments_counter(monkeypatch)

    # A burst of 20 rapid drag events (mouse-move cadence): every one updates the
    # widget's live window (latest-wins) but must not itself trigger a recompute
    # — recomputes are coalesced behind the drag timer.
    for i in range(20):
        window._frequency_plot_panel.moments_window_changed.emit(25.0 + 0.1 * i, 55.0)

    assert calls == []  # zero recomputes fired mid-burst, bootstrap or otherwise
    assert widget.range_mhz() == pytest.approx((25.0 + 0.1 * 19, 55.0))  # live, latest wins

    # The coalescing timer would fire once after ~30 ms of quiet; dispatch it
    # directly rather than waiting on real Qt timing (flaky under xdist).
    window._dispatch_pending_moments_drag()

    assert calls == ["none"]  # exactly one recompute, and it is the cheap path
    # Live but uncertainty-free: value updates, no "±" error term while dragging.
    assert "±" not in widget._value_labels["b_ave"].text()
    assert widget._value_labels["b_ave"].text() != "—"


def test_real_timer_fires_the_coalesced_cheap_recompute(window, monkeypatch):
    """A real timer delivers one cheap result for the latest range in a burst."""
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtTest import QSignalSpy

    from asymmetry.core.fourier.units import FieldUnit, convert

    dataset = _activate_fft_spectrum(window, [1])[0]
    widget = window._fourier_panel.moments_widget
    widget.clear_readout()
    assert widget._value_labels["b_ave"].text() == "—"
    calls = _wrap_spectrum_moments_counter(monkeypatch)
    timer = window._moments_drag_timer
    timeout = QSignalSpy(timer.timeout)
    assert timeout.isValid()

    for lo in np.linspace(25.0, 30.0, 20):
        window._frequency_plot_panel.moments_window_changed.emit(float(lo), 50.0)
    assert timer.isActive()
    assert timeout.count() == 0
    assert calls == []

    assert timeout.wait(5_000), "The moments drag timer did not deliver its timeout"
    QCoreApplication.processEvents()

    assert not timer.isActive()
    assert timeout.count() == 1
    assert calls == ["none"]
    assert window._frequency_plot_panel._moments_window_mhz == (30.0, 50.0)
    selected = (dataset.time >= 30.0) & (dataset.time <= 50.0)
    axis = convert(dataset.time[selected], FieldUnit.MHZ, widget.unit())
    expected_mean = float(np.average(axis, weights=dataset.asymmetry[selected]))
    assert widget._value_labels["b_ave"].text() == f"{expected_mean:.4g} G"
    assert "±" not in widget._value_labels["b_ave"].text()


def test_drag_release_runs_one_bootstrap_and_restores_uncertainty(window, monkeypatch):
    _activate_fft_spectrum(window, [1])
    widget = window._fourier_panel.moments_widget
    assert widget.is_eligible() is True

    calls = _wrap_spectrum_moments_counter(monkeypatch)

    for i in range(20):
        window._frequency_plot_panel.moments_window_changed.emit(25.0 + 0.1 * i, 55.0)
    window._dispatch_pending_moments_drag()
    assert calls == ["none"]
    assert "±" not in widget._value_labels["b_ave"].text()

    # Release: cancels any still-pending cheap tick and runs one full bootstrap.
    window._frequency_plot_panel.moments_drag_finished.emit()
    _wait_moments(window)

    assert calls == ["none", "none", "bootstrap"]
    assert not window._moments_drag_timer.isActive()
    assert "±" in widget._value_labels["b_ave"].text()  # uncertainty is back


def _fake_moments_event(panel, data_x, *, button=1):
    """A minimal matplotlib-event stand-in for the moments-handle press/motion/
    release path. Mirrors ``tests/gui/test_trend_preview.py``'s ``_fake_event``:
    projects the intended data-space *x* through the real ``transData`` so
    ``_detect_moments_handle_hit``'s pixel-space ``nearest_handle`` hit test
    resolves correctly under the offscreen backend.
    """

    class _E:
        pass

    ax = panel._ax
    e = _E()
    e.button = button
    e.x = float(ax.transData.transform((data_x, 0.0))[0])
    e.y = float(ax.transData.transform((0.0, 0.0))[1])
    e.xdata = float(data_x)
    e.ydata = 0.0
    e.inaxes = ax
    return e


def test_plot_panel_emits_drag_finished_only_after_an_actual_move(window):
    """Exercises the real press/motion/release path in ``plot_panel.py`` (not
    just the signal fired directly): a genuine drag of the window's min handle
    emits ``moments_drag_finished`` exactly once on release, while a stationary
    click (press + release, no motion) on the same handle does not — mirroring
    the existing fit-range-handle ``was_drag`` gate right above it.

    The widget's default window is the spectrum's full extent (20-60 MHz),
    which coincides with the plot's generic fit-range handles (also defaulted
    to the axis view limits). This used to require nudging the moments window
    off that default so the click would unambiguously hit a moments handle
    rather than the fit-range handle sitting at the same x; that workaround is
    gone now that ``_detect_handle_hit`` excludes the frequency panel's
    fit-range handles entirely (see ``_on_canvas_button_press`` priority and
    ``test_default_window_click_grabs_moments_handle_not_fit_range`` below,
    which pins the default-window case directly).
    """
    _activate_fft_spectrum(window, [1])
    panel = window._frequency_plot_panel
    panel._canvas.draw()  # force a real draw so transData is valid offscreen
    lo, hi = panel._moments_window_display()

    finished_events: list[None] = []
    panel.moments_drag_finished.connect(lambda: finished_events.append(None))

    # A stationary click on the min handle: press then release at the same spot.
    panel._on_canvas_button_press(_fake_moments_event(panel, lo))
    assert panel._active_moments_handle == "min"
    panel._on_canvas_button_release(_fake_moments_event(panel, lo))
    assert finished_events == []  # no move: not a drag, no recompute

    # A genuine drag: press, move, release.
    panel._on_canvas_button_press(_fake_moments_event(panel, lo))
    assert panel._active_moments_handle == "min"
    panel._on_canvas_motion_notify(_fake_moments_event(panel, lo + 1.0))
    panel._on_canvas_button_release(_fake_moments_event(panel, lo + 1.0))
    assert finished_events == [None]  # exactly one, on release
    assert panel._active_moments_handle is None


def test_default_window_click_grabs_moments_handle_not_fit_range(window):
    """Regression for the hit-test priority bug found during the moments-drag
    work (PR #218): the moments widget's default window equals the spectrum's
    full extent, which coincides with the frequency panel's generic fit-range
    handles (also seeded to the full extent the moment a spectrum is plotted —
    see ``plot_dataset``). ``_on_canvas_button_press`` used to check fit-range
    handles first, so a click on the *visible* moments handle at its default
    position actually grabbed the *invisible* (never drawn, never draggable —
    see ``test_frequency_domain_fitting.py``'s "no draggable selector" test)
    fit-range handle instead.

    Drags the min handle at the untouched default window (no (30, 50) nudge)
    and asserts the moments drag fires, not a fit-range drag.
    """
    _activate_fft_spectrum(window, [1])
    panel = window._frequency_plot_panel
    panel._canvas.draw()  # force a real draw so transData is valid offscreen

    # Confirm the premise: the frequency panel's fit-range state really does
    # coincide with the moments window's default (full-extent) position.
    assert panel._fit_x_min is not None and panel._fit_x_max is not None
    lo, hi = panel._moments_window_display()
    assert (panel._fit_x_min, panel._fit_x_max) == pytest.approx((lo, hi))

    panel._on_canvas_button_press(_fake_moments_event(panel, lo))
    assert panel._active_moments_handle == "min"
    assert panel._active_fit_handle is None

    panel._on_canvas_motion_notify(_fake_moments_event(panel, lo + 1.0))
    panel._on_canvas_button_release(_fake_moments_event(panel, lo + 1.0))
    assert panel._active_moments_handle is None
    # The moments window moved; the (inert, invisible) fit-range state did not.
    assert panel._moments_window_display() != (lo, hi)
    assert (panel._fit_x_min, panel._fit_x_max) == pytest.approx((lo, hi))


def test_approximate_errors_survive_moment_project_trend_and_exports(window, monkeypatch, tmp_path):
    _activate_fft_spectrum(window, [1, 2])
    widget = window._fourier_panel.moments_widget
    assert not widget._uncertainty_label.isHidden()
    assert widget._uncertainty_label.text() == "Approximate errors"
    assert "Spectral correlations" in widget._uncertainty_label.toolTip()
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    original = _moments_batches(window)[0]
    restored = ProjectModel.from_dict(window._project_model.to_dict()).batches[original.batch_id]
    assert restored.results_by_run[1]["warnings"] == original.results_by_run[1]["warnings"]
    rows = window._build_series_rows(restored)
    panel = window._fit_parameters_panel
    panel.load_representation_series(
        [(restored.batch_id, "Moments", rows)], select_id=restored.batch_id
    )
    assert panel._rows[0].warnings == original.results_by_run[1]["warnings"]
    assert "Recorded warnings" in panel._trend_provenance_label.text()
    state = panel.get_state()
    panel.restore_state(state)
    assert panel._rows[0].warnings == original.results_by_run[1]["warnings"]
    tsv = tmp_path / "moments.tsv"
    monkeypatch.setattr(
        "asymmetry.gui.panels.fit_parameters_panel.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tsv), ""),
    )
    panel._export_tsv()
    assert "Heuristic approximate errors" in tsv.read_text()
    gle = tmp_path / "moments.dat"
    panel._gle_snapshot()._write_gle_data_file(gle)
    assert "Heuristic approximate errors" in gle.read_text()
    widget.clear_readout()
    assert widget._uncertainty_label.isHidden()


def _await_event(event):
    import time

    from PySide6.QtCore import QCoreApplication

    for _ in range(500):
        if event.is_set():
            return
        QCoreApplication.processEvents()
        time.sleep(0.01)
    raise AssertionError("Worker did not reach the requested boundary")


def test_full_readout_runs_off_thread_and_latest_edit_wins(window, monkeypatch):
    import threading

    from PySide6.QtCore import QTimer

    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1])
    entered, release, heartbeat = (threading.Event() for _ in range(3))
    calls = []
    original = tasks.spectrum_moments

    def gated(*args, **kwargs):
        if kwargs["uncertainty"] == "bootstrap":
            calls.append((threading.get_ident(), kwargs["cutoff_fraction"]))
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", gated)
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.1)
    window._on_moments_settings_changed(widget)
    _await_event(entered)
    try:
        assert widget._points_label.text() == "Calculating approximate errors…"
        for _ in range(5):
            window._refresh_spectral_moments()
        QTimer.singleShot(0, window, heartbeat.set)
        _await_event(heartbeat)
        for cutoff in (0.12, 0.14, 0.16, 0.18):
            widget.set_cutoff_fraction(cutoff)
            window._on_moments_settings_changed(widget)
        assert "±" not in widget._value_labels["b_ave"].text()
        assert len(calls) == 1
    finally:
        release.set()
    _wait_moments(window)
    assert [cutoff for _, cutoff in calls] == [0.1, 0.18]
    assert all(thread_id != threading.get_ident() for thread_id, _ in calls)
    assert "±" in widget._value_labels["b_ave"].text()
    assert window._moment_readout._wanted.recipe.cutoff_fraction == 0.18


def test_obsolete_spectrum_preparation_cannot_start_moment_batch(window, monkeypatch):
    _activate_fft_spectrum(window, [1, 2])
    callbacks = []
    monkeypatch.setattr(
        window,
        "_ensure_frequency_spectra_for_runs_async",
        lambda *args, on_ready: callbacks.append(on_ready),
    )
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    widget.set_cutoff_fraction(0.2)
    window._on_moments_settings_changed(widget)
    callbacks.pop()()
    assert window._moment_batch._worker is None
    assert _moments_batches(window) == []


def test_batch_cancel_between_runs_never_publishes_partial_series(window, monkeypatch):
    import threading

    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1, 2])
    entered, release = threading.Event(), threading.Event()
    calls = []
    original = tasks.moments_trend_row

    def gated(*args, **kwargs):
        calls.append(kwargs["run_number"])
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "moments_trend_row", gated)
    window._on_moments_send_to_trend(window._fourier_panel.moments_widget)
    _await_event(entered)
    try:
        window._on_moments_send_to_trend(window._fourier_panel.moments_widget)
        widget = window._fourier_panel.moments_widget
        widget.set_cutoff_fraction(0.2)
        window._on_moments_settings_changed(widget)
        assert _moments_batches(window) == []
    finally:
        release.set()
    _wait_moments(window)
    assert calls == [1]
    assert _moments_batches(window) == []


def test_batch_uses_each_spectrums_recorded_padding(window):
    from asymmetry.core.fourier.moments import spectrum_moments
    from asymmetry.core.fourier.units import FieldUnit, convert

    datasets = _activate_fft_spectrum(window, [1, 2])
    datasets[0].metadata["fourier_padding"] = 1
    datasets[1].metadata["fourier_padding"] = 4
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    rows = _moments_batches(window)[0].results_by_run
    for dataset in datasets:
        expected = spectrum_moments(
            convert(dataset.time, FieldUnit.MHZ, widget.unit()),
            dataset.asymmetry,
            errors=dataset.error,
            x_range=None,
            cutoff_fraction=widget.cutoff_fraction(),
            error_oversampling=dataset.metadata["fourier_padding"],
        )
        assert rows[int(dataset.run_number)]["uncertainties"]["B_ave"] == expected.b_ave_err
    assert rows[2]["moments_recipe"]["seed"] == 0
    assert rows[2]["moments_recipe"]["error_oversampling"] == 4
    assert rows[1]["moments_recipe"]["error_oversampling"] == 1
    assert rows[2]["uncertainties"]["B_ave"] == 2 * rows[1]["uncertainties"]["B_ave"]


@pytest.mark.parametrize("action", ["source", "reset", "close"])
def test_obsolete_full_readout_does_not_publish_after_lifecycle_change(window, monkeypatch, action):
    import threading
    import time

    import asymmetry.gui.moment_tasks as tasks

    browser = window._data_browser
    _activate_fft_spectrum(window, [1])
    entered = threading.Event()
    original = tasks.spectrum_moments
    published = []

    def cancellable(*args, **kwargs):
        if kwargs["uncertainty"] == "bootstrap":
            entered.set()
            deadline = time.monotonic() + 5
            while not kwargs["cancel_callback"]() and time.monotonic() < deadline:
                time.sleep(0.005)
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", cancellable)
    widget = window._fourier_panel.moments_widget
    widget.set_cutoff_fraction(0.2)
    window._on_moments_settings_changed(widget)
    _await_event(entered)
    monkeypatch.setattr(window, "_show_moment_result", lambda *args: published.append(args))
    if action == "source":
        window._frequency_plot_panel.plot_dataset(_spectrum(2, display="power"))
        window._refresh_spectral_moments()
    elif action == "reset":
        window._data_browser = browser
        window._clear_all_state()
    else:
        window._shutdown_workers()
    _wait_moments(window)
    assert published == []


def test_repeated_refresh_reuses_full_readout_and_retains_current_errors(window, monkeypatch):
    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1])
    widget = window._fourier_panel.moments_widget
    calls = []
    original = tasks.spectrum_moments

    def counted(*args, **kwargs):
        calls.append(kwargs["uncertainty"])
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", counted)
    before = widget._value_labels["b_ave"].text()
    assert "±" in before
    for _ in range(5):
        window._refresh_spectral_moments()
    _wait_moments(window)
    assert calls == []
    assert widget._value_labels["b_ave"].text() == before


def test_readout_snapshot_detaches_arrays_and_recipe(window):
    from asymmetry.gui.moment_tasks import MomentInput, MomentRecipe

    dataset = _activate_fft_spectrum(window, [1])[0]
    widget = window._fourier_panel.moments_widget
    request = MomentInput.capture(
        dataset,
        (dataset.time, dataset.asymmetry, dataset.error),
        MomentRecipe.from_widget(widget, RepresentationType.FREQ_FFT),
    )
    before = request.compute()
    dataset.asymmetry[:] = 0
    dataset.error[:] = 100
    widget.set_cutoff_fraction(0.5)
    assert not request.frequency.flags.writeable
    assert not request.amplitude.flags.writeable
    assert not request.errors.flags.writeable
    assert request.compute() == before
    assert request.recipe.seed == 0
    assert request.recipe.mode == RepresentationType.FREQ_FFT.value
    assert request.source[0] == id(dataset)
    assert not request.matches(dataset, request.recipe)


def test_selected_run_extraction_is_off_thread_and_duplicate_send_is_single_flight(
    window, monkeypatch
):
    import threading

    import asymmetry.gui.moment_tasks as tasks

    _activate_fft_spectrum(window, [1, 2])
    calls = []
    original = tasks.spectrum_moments

    def counted(*args, **kwargs):
        calls.append((threading.get_ident(), kwargs["uncertainty"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(tasks, "spectrum_moments", counted)
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    assert len(_moments_batches(window)) == 1
    assert len(calls) == 2
    assert all(thread != threading.get_ident() and mode == "bootstrap" for thread, mode in calls)


def test_batch_retains_skipped_ineligible_run_diagnostic(window):
    datasets = _activate_fft_spectrum(window, [1, 2, 3])
    datasets[1].metadata["fourier_display"] = "power"
    datasets[2].time = np.array([])
    datasets[2].asymmetry = np.array([])
    notes = []
    window._set_fourier_status = notes.append
    window._on_moments_send_to_trend(window._fourier_panel.moments_widget)
    _wait_moments(window)
    assert _moments_batches(window)[0].member_run_numbers == [1]
    assert notes[-1] == (
        "Spectral moments sent to trend: 1 run(s). Skipped 2 without an eligible spectrum."
    )


def test_malformed_error_shape_retains_uncertainty_free_readout(window):
    dataset = _activate_fft_spectrum(window, [1])[0]
    dataset.error = np.array([0.01])
    window._refresh_spectral_moments()
    _wait_moments(window)
    widget = window._fourier_panel.moments_widget
    assert widget._value_labels["b_ave"].text() != "—"
    assert "±" not in widget._value_labels["b_ave"].text()
    assert widget._uncertainty_label.isHidden()


@pytest.mark.parametrize("historical", ["5", "6", None])
def test_fresh_moment_rows_record_current_revision_without_relabelling_history(window, historical):
    _activate_fft_spectrum(window, [1])
    widget = window._fourier_panel.moments_widget
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    recorded = _moments_batches(window)[0]
    assert recorded.results_by_run[1]["computation_revision"] == computation.COMPUTATION_REVISION
    recorded.results_by_run[1]["computation_revision"] = historical
    reopened = ProjectModel.from_dict(window._project_model.to_dict())
    assert (
        reopened.batches[recorded.batch_id].results_by_run[1]["computation_revision"] == historical
    )
    widget.set_cutoff_fraction(0.2)
    window._on_moments_send_to_trend(widget)
    _wait_moments(window)
    assert recorded.results_by_run[1]["computation_revision"] == historical
    fresh = [batch for batch in _moments_batches(window) if batch is not recorded]
    assert len(fresh) == 1
    assert fresh[0].results_by_run[1]["computation_revision"] == computation.COMPUTATION_REVISION
