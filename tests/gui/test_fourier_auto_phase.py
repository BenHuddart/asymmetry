"""Automatic phase controls and asynchronous representation publication."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fourier.phase_fit import (
    PhaseFitCancelled,
    PhaseFitRejectedError,
    PhaseRejection,
)
from asymmetry.core.representation.base import RepresentationType
from asymmetry.core.representation.frequency import FrequencyFFT
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fourier_panel import FourierPanel
from asymmetry.gui.widgets.axis_limits import FloatLimitField
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui
FFT = RepresentationType.FREQ_FFT
REPORT = {
    "fourier_phase_fit": {
        "method": "sampled_common_phase_v1",
        "groups": {
            "1": {
                "phase_radians": 0.5,
                "standard_error_radians": 0.01,
                "maximum_change_radians": 0.02,
            }
        },
    }
}


@pytest.fixture
def panel(qapp):
    settings = QSettings("AsymmetryTest", "fourier_auto_phase")
    settings.clear()
    widget = FourierPanel(settings=settings)
    yield widget
    widget.close()


def test_auto_phase_controls_round_trip_and_group_policy(panel):
    assert panel._auto_phase_section.isHidden()
    panel.restore_state(
        {
            "display": "phaseOptReal",
            "phase_fit_min_mhz": 1.2,
            "phase_fit_max_mhz": 3.4,
            "phase_fit_tolerance_degrees": 7.0,
        }
    )
    assert panel._phase_opt_real_radio.text() == "Auto Phase"
    assert not panel._auto_phase_section.isHidden()
    for field in (
        panel._phase_fit_min_edit,
        panel._phase_fit_max_edit,
        panel._phase_fit_tolerance_edit,
    ):
        assert isinstance(field, FloatLimitField)
    state = panel.get_state()
    assert state["display"] == "phaseOptReal"
    assert state["phase_fit_min_mhz"] == 1.2
    assert state["phase_fit_max_mhz"] == 3.4
    assert state["phase_fit_tolerance_degrees"] == 7.0
    panel.set_group_definitions({1: "Forward", 2: "Backward"})
    assert panel._phase_table.isEnabled()
    assert panel._phase_table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not panel._phase_table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable
    assert not panel._phase_spin.isEnabled()
    for section in (panel._conditioning_section, panel._diamag_section, panel._exclusions_section):
        assert not section.isEnabled()
    panel._signal_fb_radio.setChecked(True)
    assert not panel._phase_table.isEnabled()
    panel.restore_state({"display": "phaseOptReal"})
    assert panel.get_state()["phase_fit_min_mhz"] is None
    assert panel._phase_fit_min_edit.placeholderText() == "Auto"
    panel._phase_fit_tolerance_edit.setValue(100)
    assert panel.get_state()["phase_fit_tolerance_degrees"] == 45.0


def test_auto_phase_edits_are_debounced_and_restore_is_quiet(panel):
    calls = []
    panel.settings_changed.connect(lambda: calls.append(True))
    panel.restore_state({"display": "phaseOptReal"})
    assert not panel._settings_debounce.isActive()
    panel._phase_fit_min_edit.setText("1.0")
    panel._phase_fit_max_edit.setText("3.0")
    panel._phase_fit_tolerance_edit.setText("6")
    assert calls == []
    assert panel._settings_debounce.isActive()
    panel._settings_debounce.timeout.emit()
    assert calls == [True]
    panel._phase_fit_min_edit.clear()
    assert panel.get_state()["phase_fit_min_mhz"] is None


def test_phase_summary_exposes_conditional_estimates(panel):
    panel.set_phase_fit_summary(REPORT)
    text = panel._phase_fit_summary.text()
    assert "Group 1: phase" in text
    assert "conditional SE" in text
    assert "largest processing change" in text
    assert panel._phase_fit_summary.toolTip() == text
    panel.set_phase_fit_summary({})
    assert panel._phase_fit_summary.text() == ""


@pytest.fixture
def window(qapp):
    win = MainWindow()
    counts = np.full(64, 1000.0)
    run = Run(
        run_number=731,
        histograms=[Histogram(counts=counts, bin_width=0.04, t0_bin=0)],
        grouping={
            "groups": {1: [1]},
            "group_names": {1: "Forward"},
            "first_good_bin": 0,
            "last_good_bin": 63,
            "deadtime_correction": False,
        },
    )
    dataset = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.zeros(2),
        error=np.ones(2),
        metadata={"run_number": run.run_number},
        run=run,
    )
    win._data_browser.add_dataset(dataset)
    win._current_dataset = dataset
    win._sync_fourier_panel_for_dataset(dataset)
    win._fourier_panel.restore_state({"display": "phaseOptReal"})
    yield win
    win.close()


def _spectrum():
    return MuonDataset(
        time=np.array([1.0, 2.0, 3.0]),
        asymmetry=np.array([0.0, 2.0, 0.0]),
        error=np.ones(3),
        metadata={"fourier_display": "phaseOptReal"},
    )


def test_explicit_compute_snapshots_and_publishes_metadata(window, qapp, monkeypatch):
    gui_thread = threading.get_ident()
    original = window._project_model.ensure_dataset(731).ensure(FFT)
    calls = []
    publication_threads = []
    cache = original.cache_datasets

    def publish(spectra):
        publication_threads.append(threading.get_ident())
        cache(spectra)

    def compute(rep, run, *, context=None):
        assert rep is not original
        assert threading.get_ident() != gui_thread
        assert callable(context)
        assert original.result_metadata == {}
        rep.result_metadata = REPORT
        calls.append(rep.recipe["fourier_config"])
        return [_spectrum()]

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    monkeypatch.setattr(original, "cache_datasets", publish)
    window._fourier_panel._phase_fit_min_edit.setValue(1.1)
    window._fourier_panel._phase_fit_max_edit.setValue(3.2)
    window._fourier_panel._phase_fit_tolerance_edit.setValue(6.0)
    window._on_compute_fourier()
    wait_for(lambda: not window._fourier_compute_active, qapp)
    assert len(calls) == 1
    assert calls[0]["phase_fit_tolerance_degrees"] == 6.0
    assert calls[0]["phase_fit_min_mhz"] == 1.1
    assert calls[0]["phase_fit_max_mhz"] == 3.2
    assert publication_threads == [gui_thread]
    assert original.result_metadata == REPORT
    assert len(window._cached_frequency_spectra(731, FFT)) == 1
    assert not window._frequency_recompute_inflight


def test_lazy_auto_recipe_and_view_seed_do_not_compute(window, monkeypatch):
    def compute(*args, **kwargs):
        pytest.fail("Auto Phase must require explicit Compute FFT")

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    assert window._seed_fft_recipe_for_view(731) is None
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "phaseOptReal"}}
    assert window._frequency_recompute_target(731, FFT) is None
    assert window._frequency_recompute_target(731, FFT, explicit=True) is not None
    assert window._ensure_frequency_spectra_for_run(731, FFT) == []
    rep.cache_datasets([_spectrum()])
    window._pending_recipe_recompute.add((731, FFT))
    assert window._frequency_spectra_from_cache(731, FFT) == [rep.primary]


@pytest.mark.parametrize("cancelled", [False, True])
def test_unsuccessful_compute_cleans_up_and_keeps_explanation(window, qapp, monkeypatch, cancelled):
    def compute(*args, **kwargs):
        if cancelled:
            raise PhaseFitCancelled()
        raise PhaseFitRejectedError(
            PhaseRejection(
                "uncertain", "Group 1: phase uncertainty exceeds tolerance; widen fit band"
            )
        )

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    window._on_compute_fourier()
    wait_for(lambda: not window._fourier_compute_active, qapp)
    rep = window._project_model.representation(731, FFT)
    assert rep.primary is None
    assert window._cached_frequency_spectra(731, FFT) == []
    assert not window._frequency_recompute_inflight
    assert window._frequency_overlay.isHidden()
    status = window._fourier_panel._action_footer._status_label.text()
    assert ("cancelled" if cancelled else "Group 1: phase uncertainty") in status


def test_saved_auto_spectrum_without_method_marker_is_stale(window):
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    config = window._candidate_fourier_config(window._current_dataset)
    rep.recipe = {"fourier_config": config.to_dict()}
    rep.cache_datasets([_spectrum()])
    window._frequency_cache(FFT)[731] = [rep.primary]
    stale, reason = window._fourier_staleness_for_current_run()
    assert stale
    assert "phase estimation method changed" in reason
    rep.result_metadata = REPORT
    assert window._fourier_staleness_for_current_run() == (False, "")


def test_explicit_retry_retains_permission_after_inflight_completion(window, qapp, monkeypatch):
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "phaseOptReal"}}
    calls = []
    ready = []

    def compute(snapshot, run, *, context=None):
        calls.append(run.run_number)
        return [_spectrum()]

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    key = (731, FFT)
    window._frequency_recompute_inflight.add(key)
    window._ensure_frequency_spectra_for_runs_async(
        [731], FFT, lambda: ready.append(True), explicit=True
    )
    assert ready == []
    window._frequency_recompute_inflight.discard(key)
    window._notify_frequency_recompute_done(key)
    wait_for(lambda: bool(ready), qapp)
    assert calls == [731]
    assert ready == [True]


def test_cancel_requested_on_worker_reaches_fit_and_cleans_up(window, qapp, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    start = window._tasks.start
    handles = []

    def capture(*args, **kwargs):
        worker = start(*args, **kwargs)
        handles.append(worker)
        return worker

    def compute(snapshot, run, *, context=None):
        entered.set()
        assert release.wait(timeout=5)
        assert context()
        raise PhaseFitCancelled()

    monkeypatch.setattr(window._tasks, "start", capture)
    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    window._on_compute_fourier()
    try:
        wait_for(entered.is_set, qapp)
        handles[0].cancel()
    finally:
        release.set()
    wait_for(lambda: not window._fourier_compute_active, qapp)
    assert not window._frequency_recompute_inflight
    assert window._frequency_overlay.isHidden()
    assert window._cached_frequency_spectra(731, FFT) == []


def test_legacy_auto_snapshot_without_recipe_is_stale(window):
    window._frequency_cache(FFT)[731] = [_spectrum()]
    stale, reason = window._fourier_staleness_for_current_run()
    assert stale
    assert reason == "phase estimation method changed"


def test_batch_rejection_fails_only_the_refused_run(window, qapp, monkeypatch):
    from dataclasses import replace

    first = window._current_dataset
    second_run = replace(first.run, run_number=732)
    second = MuonDataset(
        time=first.time,
        asymmetry=first.asymmetry,
        error=first.error,
        metadata={"run_number": 732},
        run=second_run,
    )
    window._data_browser.add_dataset(second)
    for number in (731, 732):
        rep = window._project_model.ensure_dataset(number).ensure(FFT)
        rep.recipe = {"fourier_config": {"display": "phaseOptReal"}}
    calls = []
    ready = []

    def compute(snapshot, run, *, context=None):
        calls.append(run.run_number)
        snapshot.result_metadata = REPORT
        if run.run_number == 732:
            raise PhaseFitRejectedError(PhaseRejection("uncertain", "Group 1: widen fit band"))
        return [_spectrum()]

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    window._ensure_frequency_spectra_for_runs_async(
        [731, 732], FFT, lambda: ready.append(True), explicit=True
    )
    wait_for(lambda: bool(ready), qapp)
    assert calls == [731, 732]
    accepted = window._project_model.representation(731, FFT)
    assert accepted.result_metadata == REPORT
    assert window._cached_frequency_spectra(731, FFT) == [accepted.primary]
    refused = window._project_model.representation(732, FFT)
    assert refused.primary is None
    assert refused.result_metadata == {}
    assert window._cached_frequency_spectra(732, FFT) == []
    assert window._lazy_recompute_failures == {(732, FFT)}
    status = window._fourier_panel._action_footer._status_label.text()
    assert "run 732: Group 1: widen fit band" in status


@pytest.mark.parametrize("dispatch", ["single", "batch"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_superseded_failure_releases_queued_explicit_compute(
    window, qapp, monkeypatch, dispatch, cancelled
):
    old_entered, old_release = threading.Event(), threading.Event()
    new_entered, new_release = threading.Event(), threading.Event()
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "(Power)^1/2"}}
    old_ready = []
    statuses = []
    set_status = window._set_fourier_status

    def status(message, **kwargs):
        statuses.append(message)
        set_status(message, **kwargs)

    def compute(snapshot, run, *, context=None):
        if snapshot.fourier_config()["display"] == "(Power)^1/2":
            old_entered.set()
            assert old_release.wait(timeout=5)
            if cancelled:
                raise PhaseFitCancelled()
            raise RuntimeError("obsolete worker failure")
        new_entered.set()
        assert new_release.wait(timeout=5)
        snapshot.result_metadata = REPORT
        return [_spectrum()]

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    monkeypatch.setattr(window, "_set_fourier_status", status)
    if dispatch == "single":
        window._start_async_frequency_recompute(731, FFT, (rep, window._current_dataset.run))
    else:
        window._ensure_frequency_spectra_for_runs_async([731], FFT, lambda: old_ready.append(True))
    try:
        wait_for(old_entered.is_set, qapp)
        window._on_compute_fourier()
        assert window._fourier_compute_active
        old_release.set()
        wait_for(new_entered.is_set, qapp)
        assert window._fourier_compute_active
        assert (731, FFT) in window._frequency_recompute_inflight
        assert (731, FFT) not in window._lazy_recompute_failures
        assert not window._frequency_overlay.isHidden()
        assert old_ready == []
        assert not any("obsolete" in message or "cancelled" in message for message in statuses)
    finally:
        old_release.set()
        new_release.set()
    wait_for(lambda: not window._fourier_compute_active, qapp)
    assert rep.result_metadata == REPORT
    assert window._cached_frequency_spectra(731, FFT) == [rep.primary]
    if dispatch == "batch":
        wait_for(lambda: bool(old_ready), qapp)
        assert old_ready == [True]


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("replace_representation", [False, True])
def test_superseded_batch_failure_preserves_new_result(
    window, qapp, monkeypatch, cancelled, replace_representation
):
    entered, release = threading.Event(), threading.Event()
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "(Power)^1/2"}}
    old_ready = []

    def compute(snapshot, run, *, context=None):
        entered.set()
        assert release.wait(timeout=5)
        if cancelled:
            raise PhaseFitCancelled()
        raise RuntimeError("obsolete worker failure")

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    window._ensure_frequency_spectra_for_runs_async([731], FFT, lambda: old_ready.append(True))
    try:
        wait_for(entered.is_set, qapp)
        if replace_representation:
            window._project_model = type(window._project_model)()
            current = window._project_model.ensure_dataset(731).ensure(FFT)
        else:
            current = rep
        current.recipe = {"fourier_config": {"display": "phaseOptReal"}}
        spectrum = _spectrum()
        current.cache_datasets([spectrum])
        current.result_metadata = REPORT
        window._frequency_cache(FFT)[731] = [spectrum]
        window._set_fourier_status("New result remains current.", success=True)
    finally:
        release.set()
    wait_for(lambda: not window._frequency_recompute_inflight, qapp)
    assert old_ready == [True]
    assert current.primary is spectrum
    assert current.result_metadata == REPORT
    assert window._cached_frequency_spectra(731, FFT) == [spectrum]
    assert (731, FFT) not in window._lazy_recompute_failures
    assert (
        "New result remains current." in window._fourier_panel._action_footer._status_label.text()
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_superseded_single_failure_resyncs_displayed_run(window, qapp, monkeypatch, cancelled):
    entered, release = threading.Event(), threading.Event()
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "(Power)^1/2"}}
    synced = []

    def compute(snapshot, run, *, context=None):
        entered.set()
        assert release.wait(timeout=5)
        if cancelled:
            raise PhaseFitCancelled()
        raise RuntimeError("obsolete worker failure")

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    monkeypatch.setattr(window, "_sync_frequency_plot_for_run", synced.append)
    window._start_async_frequency_recompute(731, FFT, (rep, window._current_dataset.run))
    try:
        wait_for(entered.is_set, qapp)
        window._frequency_display_key = (731, FFT)
        rep.recipe = {"fourier_config": {"display": "Cos"}}
    finally:
        release.set()
    wait_for(lambda: not window._frequency_recompute_inflight, qapp)
    wait_for(lambda: bool(synced), qapp)
    assert synced == [731]
    assert (731, FFT) not in window._lazy_recompute_failures


@pytest.mark.parametrize("dispatch", ["single", "batch"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_retired_project_failure_preserves_new_inflight_tracking(
    window, qapp, monkeypatch, dispatch, cancelled
):
    entered, release = threading.Event(), threading.Event()
    rep = window._project_model.ensure_dataset(731).ensure(FFT)
    rep.recipe = {"fourier_config": {"display": "(Power)^1/2"}}
    old_ready = []

    def compute(snapshot, run, *, context=None):
        entered.set()
        assert release.wait(timeout=5)
        if cancelled:
            raise PhaseFitCancelled()
        raise RuntimeError("obsolete project failure")

    monkeypatch.setattr(FrequencyFFT, "compute", compute)
    if dispatch == "single":
        window._start_async_frequency_recompute(731, FFT, (rep, window._current_dataset.run))
    else:
        window._ensure_frequency_spectra_for_runs_async([731], FFT, lambda: old_ready.append(True))
    old_inflight = window._frequency_recompute_inflight
    try:
        wait_for(entered.is_set, qapp)
        window._project_model = type(window._project_model)()
        window._frequency_recompute_inflight = {(731, FFT)}
        new_ready = []
        window._frequency_recompute_waiters = [({(731, FFT)}, lambda: new_ready.append(True))]
        window._fourier_compute_active = True
        window._frequency_overlay.show_message("New project computation")
        window._set_fourier_status("New project computation")
    finally:
        release.set()
    try:
        wait_for(lambda: not old_inflight, qapp)
        assert window._frequency_recompute_inflight == {(731, FFT)}
        assert window._fourier_compute_active
        assert old_ready == new_ready == []
        assert not window._frequency_overlay.isHidden()
        assert (
            "New project computation" in window._fourier_panel._action_footer._status_label.text()
        )
    finally:
        window._frequency_recompute_inflight.clear()
        window._frequency_recompute_waiters.clear()
        window._fourier_compute_active = False
