"""Tests for the FFT out-of-date indicator.

Panel level: the debounced settings_changed signal must coalesce rapid edits
(radio double-fires, typing) behind a single-shot debounce timer, and must
never fire from a programmatic restore (restore_state, set_group_definitions,
set_group_phases, set_group_enabled, restore_group_phase_state); the stale
banner is driven by set_stale()/is_stale().

MainWindow level: a computed spectrum starts in sync; panel edits, grouping
changes (via the recipe's grouping digest), and time-window changes flag it;
a recompute clears the flag; inert edits and pre-digest legacy recipes do not
flag.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.representation.base import RepresentationType
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.panels.fourier_panel import FourierPanel
from tests._qt_helpers import wait_for

pytestmark = pytest.mark.gui


def _scratch_panel() -> FourierPanel:
    """A FourierPanel over a scratch QSettings scope (isolated from the app's)."""
    settings = QSettings("AsymmetryTest", "fourier_staleness_test")
    settings.clear()
    return FourierPanel(settings=settings)


# ── settings_changed: debounce ──────────────────────────────────────────


def test_padding_change_emits_settings_changed_once_after_debounce(qapp) -> None:
    panel = _scratch_panel()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel._padding_spin.setValue(panel._padding_spin.value() + 1)

    # The debounce timer is scheduled, not fired synchronously.
    assert panel._settings_debounce.isActive()
    assert calls == []

    # Drive the timeout directly rather than sleeping on the real timer.
    panel._settings_debounce.timeout.emit()
    assert len(calls) == 1


def test_rapid_edits_coalesce_to_one_emit(qapp) -> None:
    """Multiple edits before the timer fires still produce exactly one emit."""
    panel = _scratch_panel()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel._padding_spin.setValue(panel._padding_spin.value() + 1)
    panel._filter_start_edit.setText("1.0")
    panel._phase_spin.setText("10")
    assert panel._settings_debounce.isActive()
    assert calls == []

    panel._settings_debounce.timeout.emit()
    assert len(calls) == 1


# ── settings_changed: suppression during programmatic restores ─────────


def test_restore_state_does_not_emit(qapp) -> None:
    panel = _scratch_panel()
    saved = panel.get_state()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel.restore_state(saved)

    assert not panel._settings_debounce.isActive()
    assert calls == []


def test_set_group_definitions_does_not_emit(qapp) -> None:
    panel = _scratch_panel()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel.set_group_definitions({1: "Fwd", 2: "Bwd"}, {1: 0.0, 2: 90.0}, {1: True, 2: True})

    assert not panel._settings_debounce.isActive()
    assert calls == []


def test_set_group_phases_does_not_emit(qapp) -> None:
    panel = _scratch_panel()
    panel.set_group_definitions({1: "Fwd", 2: "Bwd"})
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel.set_group_phases({1: 12.5}, auto_filled=True)

    assert not panel._settings_debounce.isActive()
    assert calls == []


def test_set_group_enabled_does_not_emit(qapp) -> None:
    panel = _scratch_panel()
    panel.set_group_definitions({1: "Fwd", 2: "Bwd"})
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel.set_group_enabled({1: False})

    assert not panel._settings_debounce.isActive()
    assert calls == []


def test_restore_group_phase_state_does_not_emit(qapp) -> None:
    panel = _scratch_panel()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel.restore_group_phase_state(
        {"group_enabled_table": {1: True}, "group_phase_table": {1: 5.0}},
        {1: "Fwd"},
    )

    assert not panel._settings_debounce.isActive()
    assert calls == []


# ── phase-table item edits: guarded vs. real user edits ─────────────────


def test_phase_table_rebuild_does_not_emit_but_user_edit_does(qapp) -> None:
    """A programmatic table rebuild must not schedule an emit; a simulated
    user edit to a phase cell must."""
    panel = _scratch_panel()
    panel.set_group_definitions({1: "Fwd", 2: "Bwd"})

    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    # Rebuild again (programmatic) — no schedule.
    panel.set_group_definitions({1: "Fwd", 2: "Bwd"}, {1: 45.0})
    assert not panel._settings_debounce.isActive()
    assert calls == []

    # Simulate a user typing a new phase into row 0's phase cell (column 2).
    item = panel._phase_table.item(0, 2)
    assert item is not None
    item.setText("99.0")

    assert panel._settings_debounce.isActive()
    panel._settings_debounce.timeout.emit()
    assert len(calls) == 1


def test_apply_psi_harmonics_preset_is_a_user_action_and_emits(qapp) -> None:
    """The PSI-harmonics preset button is a user action; unlike the restore
    helpers it must schedule an emit (task explicitly calls this out)."""
    panel = _scratch_panel()
    calls: list[None] = []
    panel.settings_changed.connect(lambda: calls.append(None))

    panel._apply_psi_harmonics_preset()

    assert panel._settings_debounce.isActive()
    panel._settings_debounce.timeout.emit()
    assert len(calls) == 1


# ── stale banner ─────────────────────────────────────────────────────────


def test_banner_hidden_by_default(qapp) -> None:
    panel = _scratch_panel()
    assert panel.is_stale() is False


def test_set_stale_shows_exact_composed_text(qapp) -> None:
    panel = _scratch_panel()
    panel.set_stale("grouping changed")

    assert panel.is_stale() is True
    assert (
        panel._stale_banner.text()
        == "Spectrum out of date — grouping changed. Compute FFT to refresh."
    )


def test_set_stale_none_hides_banner(qapp) -> None:
    panel = _scratch_panel()
    panel.set_stale("grouping changed")
    assert panel.is_stale() is True

    panel.set_stale(None)
    assert panel.is_stale() is False


def test_set_stale_empty_string_hides_banner(qapp) -> None:
    panel = _scratch_panel()
    panel.set_stale("grouping changed")
    panel.set_stale("")
    assert panel.is_stale() is False


# ── overlay-mismatch banner ───────────────────────────────────────────────


def test_overlay_mismatch_banner_hidden_by_default(qapp) -> None:
    panel = _scratch_panel()
    assert panel.is_overlay_mismatched() is False


def test_set_overlay_mismatch_shows_and_hides_banner(qapp) -> None:
    panel = _scratch_panel()
    panel.set_overlay_mismatch(True)

    assert panel.is_overlay_mismatched() is True
    assert (
        panel._overlay_mismatch_banner.text()
        == "Overlaid spectra use different settings — Compute FFT to unify."
    )

    panel.set_overlay_mismatch(False)
    assert panel.is_overlay_mismatched() is False


def test_stale_and_overlay_mismatch_banners_are_independent(qapp) -> None:
    """The two banners track different conditions and can show together."""
    panel = _scratch_panel()
    panel.set_stale("grouping changed")
    panel.set_overlay_mismatch(True)

    assert panel.is_stale() is True
    assert panel.is_overlay_mismatched() is True

    panel.set_stale(None)
    assert panel.is_stale() is False
    assert panel.is_overlay_mismatched() is True


# ── MainWindow integration: staleness evaluation end-to-end ──────────────


def _tf_run(*, n: int = 512, bin_width: float = 0.04) -> Run:
    """A two-detector transverse-field run with a plain F/B grouping."""
    rng = np.random.default_rng(11)
    time_axis = np.arange(n, dtype=float) * bin_width
    histograms: list[Histogram] = []
    for sign in (+1.0, -1.0):
        signal = 1.0 + sign * 0.2 * np.cos(2.0 * np.pi * 2.7 * time_axis)
        counts = 4000.0 * np.exp(-time_axis / 2.1969811) * signal
        counts = rng.poisson(np.clip(counts, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=bin_width, t0_bin=0))
    return Run(
        run_number=77,
        histograms=histograms,
        metadata={"field": 200.0, "temperature": 5.0},
        grouping={
            "groups": {1: [1], 2: [2]},
            "group_names": {1: "Fwd", 2: "Bwd"},
            "first_good_bin": 0,
            "last_good_bin": n - 1,
            "deadtime_correction": False,
        },
    )


@pytest.fixture
def window():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    win = MainWindow()
    yield win
    win.close()


def _computed_window(window: MainWindow) -> tuple[MainWindow, MuonDataset]:
    """A window whose current run has one explicitly computed FFT spectrum.

    Drives the real, unified Compute FFT handler (selection-scoped since the
    one-button consolidation): the run is loaded into the Data Browser (the
    batch recompute resolves target runs through it), the handler stamps the
    recipe + digest and computes off-thread, and the loop is spun until the
    completion lands — exercising recipe recording, digest stamping, and the
    staleness refresh through the same path a button click takes.
    """
    from PySide6.QtWidgets import QApplication

    run = _tf_run()
    dataset = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.zeros(2),
        error=np.zeros(2),
        metadata={"run_number": run.run_number, "field": 200.0},
        run=run,
    )
    if window._data_browser.get_dataset(run.run_number) is None:
        window._data_browser.add_dataset(dataset)
    window._current_dataset = dataset
    window._sync_fourier_panel_for_dataset(dataset)
    window._on_compute_fourier()
    wait_for(lambda: not window._fourier_compute_active, QApplication.instance(), timeout_s=15.0)
    assert window._cached_frequency_spectra(run.run_number, RepresentationType.FREQ_FFT)
    return window, dataset


def test_computed_spectrum_starts_in_sync(window) -> None:
    window, dataset = _computed_window(window)
    recipe = window._project_model.representation(
        int(dataset.run_number), RepresentationType.FREQ_FFT
    ).recipe
    assert recipe.get("grouping_digest")
    assert window._fourier_panel.is_stale() is False


def test_panel_edit_flags_stale_and_recompute_clears(window) -> None:
    window, dataset = _computed_window(window)

    window._fourier_panel._padding_spin.setValue(8)  # != the panel default
    window._fourier_panel._settings_debounce.timeout.emit()
    assert window._fourier_panel.is_stale() is True
    assert "zero-pad factor" in window._fourier_panel._stale_banner.text()

    # Recompute with the edited settings: in sync again by construction.
    _computed_window(window)
    assert window._fourier_panel.is_stale() is False


def test_inert_edit_does_not_flag(window) -> None:
    """A filter τ edit while apodisation is None cannot change the spectrum."""
    window, _dataset = _computed_window(window)

    window._fourier_panel._filter_time_constant_edit.setText("3.5")
    window._fourier_panel._settings_debounce.timeout.emit()
    assert window._fourier_panel.is_stale() is False


def test_grouping_change_flags_stale(window) -> None:
    window, dataset = _computed_window(window)

    dataset.run.grouping["groups"] = {1: [1, 2], 2: [2]}
    window._refresh_fourier_staleness()
    assert window._fourier_panel.is_stale() is True
    assert "grouping changed" in window._fourier_panel._stale_banner.text()


def test_forward_backward_swap_without_background_does_not_flag(window) -> None:
    """A polarization-axis style forward/backward rewrite is inert to the FFT
    unless a list-routed background is in play."""
    window, dataset = _computed_window(window)

    dataset.run.grouping["forward_group"] = 2
    dataset.run.grouping["backward_group"] = 1
    window._refresh_fourier_staleness()
    assert window._fourier_panel.is_stale() is False


def test_time_window_change_flags_stale(window) -> None:
    window, dataset = _computed_window(window)

    # Set the range without the emit cascade: with no plotted selection in this
    # bare harness, _update_selected_datasets would re-clear the plot (and the
    # range) again. Production reaches the same refresh via fit_range_changed →
    # _on_fit_range_changed.
    window._plot_panel._set_fit_range(0.0, 5.0, emit_signal=False, redraw=False)
    window._refresh_fourier_staleness()
    assert window._fourier_panel.is_stale() is True
    assert "time window" in window._fourier_panel._stale_banner.text()


def test_legacy_recipe_without_digest_skips_grouping_check(window) -> None:
    """Projects saved before the digest existed cannot false-flag on grouping."""
    window, dataset = _computed_window(window)
    recipe = window._project_model.representation(
        int(dataset.run_number), RepresentationType.FREQ_FFT
    ).recipe
    recipe.pop("grouping_digest", None)

    dataset.run.grouping["groups"] = {1: [1, 2], 2: [2]}
    window._refresh_fourier_staleness()
    assert window._fourier_panel.is_stale() is False


def test_run_switch_reevaluates_banner(window) -> None:
    """The banner is per-run state: switching to a run with no spectrum clears it."""
    window, _dataset = _computed_window(window)
    window._fourier_panel._padding_spin.setValue(8)  # != the panel default
    window._fourier_panel._settings_debounce.timeout.emit()
    assert window._fourier_panel.is_stale() is True

    window._current_dataset = None
    window._sync_fourier_panel_for_dataset(None)
    assert window._fourier_panel.is_stale() is False
