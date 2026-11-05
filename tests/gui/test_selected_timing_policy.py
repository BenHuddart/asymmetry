"""Raw timing workers cannot publish a selection or manual value they did not resolve."""

import threading

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.windows.grouping import dialog as dialog_module
from tests.core.test_selected_timing_policy import timing_run
from tests.gui.test_grouping_preview_pane import _wait_until

pytestmark = pytest.mark.gui


def timing_dialog(monkeypatch):
    run = timing_run()
    dataset = MuonDataset(
        time=np.arange(40) * 0.1, asymmetry=np.zeros(40), error=np.ones(40), run=run
    )
    entered, release = threading.Event(), threading.Event()
    original = dialog_module.find_t0_for_run

    def search(histograms, metadata=None, **options):
        entered.set()
        assert release.wait(10)
        return original(histograms, metadata, **options)

    monkeypatch.setattr(dialog_module, "find_t0_for_run", search)
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
