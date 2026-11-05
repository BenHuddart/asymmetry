"""Raw timing workers cannot publish a selection or manual value they did not resolve."""

import threading

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.windows.grouping import dialog as dialog_module
from asymmetry.gui.windows.grouping import timing_section as timing_module
from tests.core.test_selected_timing_policy import timing_run
from tests.gui.test_grouping_preview_pane import _wait_until

pytestmark = pytest.mark.gui


def timing_dialog(monkeypatch):
    run = timing_run()
    dataset = MuonDataset(
        time=np.arange(40) * 0.1, asymmetry=np.zeros(40), error=np.ones(40), run=run
    )
    entered, release = threading.Event(), threading.Event()
    original = timing_module.find_t0_for_run

    def search(histograms, metadata=None, **options):
        entered.set()
        assert release.wait(10)
        return original(histograms, metadata, **options)

    monkeypatch.setattr(timing_module, "find_t0_for_run", search)
    dialog = dialog_module.GroupingDialog([dataset])
    dialog._set_t0_mode_combo("auto_detect")
    dialog._on_t0_mode_changed()
    _wait_until(entered.is_set)
    return dialog, release


def test_completed_scan_accepts_current_pair_and_ignores_unused_failure(qapp, monkeypatch):
    dialog, release = timing_dialog(monkeypatch)
    try:
        assert not dialog._apply_btn.isEnabled()
        dialog._backward_combo.setCurrentIndex(dialog._backward_combo.findData(3))
        release.set()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._t0_spin.value() == 12
        assert "Detected: bin 12" in dialog._t0_detected_label.text()
        dialog._backward_combo.setCurrentIndex(dialog._backward_combo.findData(2))
        _wait_until(lambda: "unavailable" in dialog._preview_pane._status.text().lower())
        assert not dialog._apply_btn.isEnabled()
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()


def test_pending_scan_cannot_overwrite_manual_edit(qapp, monkeypatch):
    dialog, release = timing_dialog(monkeypatch)
    try:
        dialog._set_t0_mode_combo("manual")
        dialog._on_t0_mode_changed()
        dialog._t0_spin.setValue(11)
        release.set()
        _wait_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)
        assert dialog._t0_spin.value() == 11
        assert dialog._current_t0_mode() == "manual"
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()


def test_stale_success_cannot_supply_new_unknown_source(qapp, monkeypatch):
    dialog, release = timing_dialog(monkeypatch)
    try:
        dialog._backward_combo.setCurrentIndex(dialog._backward_combo.findData(3))
        dialog._run.metadata.clear()
        dialog._refresh_preview()
        release.set()
        _wait_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)
        _wait_until(lambda: "unavailable" in dialog._preview_pane._status.text().lower())
        assert not dialog._apply_btn.isEnabled()
        assert not dialog._t0_search_cache[dialog._t0_cache_key()].ok
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()


def make_dialog(run, *, profiles=()):
    dataset = MuonDataset(
        time=np.arange(40) * 0.1, asymmetry=np.zeros(40), error=np.ones(40), run=run
    )
    return dialog_module.GroupingDialog([dataset], profiles=list(profiles))


def test_missing_fallback_controls_and_manual_offset_use_the_effective_base(qapp):
    from asymmetry.core.project.profiles import (
        T0Policy,
        profile_fingerprint_for_run,
        profile_from_payload,
    )

    run = timing_run(missing=True)
    run.grouping["backward_group"] = 3
    profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
    profile.t0_policy = T0Policy(mode="manual", offset_bins=2)
    dialog = make_dialog(run, profiles=[profile])
    try:
        _wait_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._t0_spin.value() == 14
        assert dialog._current_t0_policy().offset_bins == 2
        dialog._set_t0_mode_combo("from_file")
        dialog._on_t0_mode_changed()
        assert dialog._t0_spin.value() == 12
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_combined_preview_reuses_both_raw_period_searches(qapp, monkeypatch):
    run = timing_run()
    green = timing_run()
    run.grouping.update(
        backward_group=3,
        period_histograms=[run.histograms, green.histograms],
        period_t0_sources=["file", "file"],
    )
    scans = []
    original = timing_module.find_t0_for_run

    def search(histograms, metadata=None, **options):
        scans.append(threading.current_thread())
        return original(histograms, metadata, **options)

    monkeypatch.setattr(timing_module, "find_t0_for_run", search)
    dialog = make_dialog(run)
    try:
        dialog._set_t0_mode_combo("auto_detect")
        dialog._on_t0_mode_changed()
        dialog._set_period_mode("green_minus_red")
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert len(scans) == 2
        assert all(thread is not threading.current_thread() for thread in scans)
        dialog._bunch_spin.setValue(2)
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert len(scans) == 2
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_completed_raw_cache_is_reused_after_core_period_cloning(qapp):
    from asymmetry.core.project.profiles import (
        T0Policy,
        profile_fingerprint_for_run,
        profile_from_payload,
        resolve_effective_grouping,
    )

    run = timing_run()
    run.grouping.update(
        backward_group=3,
        period_histograms=[run.histograms, run.histograms],
        period_t0_sources=["file", "file"],
    )
    dialog = make_dialog(run)
    try:
        _wait_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)
        profile = profile_from_payload(run.grouping, "timing", profile_fingerprint_for_run(run))
        profile.t0_policy, profile.period_mode = T0Policy(mode="auto_detect"), "red"
        resolved = resolve_effective_grouping(profile, run, timing_search=dialog._cached_t0_search)
        assert resolved["t0_bin"] == 12
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_form_seeding_never_runs_automatic_alpha_on_gui_thread(qapp, monkeypatch):
    from asymmetry.core.project import profiles
    from tests.gui.test_automatic_grouping_policies import automatic_dialog

    original = profiles._estimate_run_alpha
    threads = []

    def estimate(*args, **kwargs):
        threads.append(threading.current_thread())
        assert threading.current_thread() is not threading.main_thread()
        return original(*args, **kwargs)

    monkeypatch.setattr(profiles, "_estimate_run_alpha", estimate)
    dialog, _ = automatic_dialog()
    try:
        dialog._preview_pane.flush()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert threads
        assert dialog._draft.alpha_policy.mode == "per_run_estimate"
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_stale_scan_error_cannot_overwrite_manual_edit(qapp, monkeypatch):
    run = timing_run()
    entered, release = threading.Event(), threading.Event()

    def search(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        raise ValueError("synthetic search failure")

    monkeypatch.setattr(timing_module, "find_t0_for_run", search)
    dialog = make_dialog(run)
    try:
        _wait_until(entered.is_set)
        dialog._set_t0_mode_combo("manual")
        dialog._on_t0_mode_changed()
        dialog._t0_spin.setValue(11)
        release.set()
        _wait_until(lambda: dialog._t0_search_cache.get(dialog._t0_cache_key()) is not None)
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._t0_spin.value() == 11
        assert not dialog._t0_search_cache[dialog._t0_cache_key()].ok
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()
