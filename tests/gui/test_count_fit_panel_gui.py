"""GUI wiring for the count-domain fit-target selector in the Multi-Group window.

These tests exercise the routing only — the count-fit numerics are covered by
``tests/test_count_domain_fits.py``. The detector-balance α is recovered here
even with the default model because it is fixed by the forward/backward count
ratio, independent of the oscillation model.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.simulate import (
    build_builtin_template,
    simulate_double_pulse_run,
    simulate_run,
)
from asymmetry.gui.panels.fit_panel import FitParameterTable
from asymmetry.gui.windows.multi_group_fit_window import MultiGroupFitWindow


def _tf(t, A=20.0, f=1.5, phi=0.0):  # noqa: N803 (A is the conventional asymmetry symbol)
    return A * np.cos(2.0 * np.pi * f * np.asarray(t, dtype=float) + phi)


@pytest.fixture
def fb_dataset() -> MuonDataset:
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.5, "phi": 0.3}, total_events=40e6, alpha=1.25, seed=1
    )
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def test_target_selector_pushes_mode_to_both_tabs(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # Forward + Backward (free α)
    assert window._single_fit_tab._count_fit_mode == "fb"
    assert window._batch_fit_tab._count_fit_mode == "fb"
    # The Poisson/Gaussian cost now applies to every grouped target, including
    # the lifetime-corrected fgAll ("all") fit, so the selector stays enabled.
    assert window._cost_combo.isEnabled()
    window._target_combo.setCurrentIndex(0)
    assert window._single_fit_tab._count_fit_mode == "all"
    assert window._cost_combo.isEnabled()


def test_advanced_fit_target_controls_collapsed_by_default(qapp, fb_dataset):
    # The advanced count-fit options and calibration promotes are folded into
    # two sections collapsed by default, so they don't push the model table down.
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    assert not window._count_options_section.isExpanded()
    assert not window._calibration_section.isExpanded()


def test_side_row_visible_only_for_single_group_target(qapp, fb_dataset):
    # Forward/Backward side only means anything for the single-group target, so
    # its row is shown only then (rather than always present but greyed out).
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    # Default target = All groups → Side hidden.
    assert window._side_combo.isHidden()
    # Single group → Side shown.
    window._target_combo.setCurrentIndex(2)
    assert not window._side_combo.isHidden()
    # Forward + Backward → hidden again.
    window._target_combo.setCurrentIndex(1)
    assert window._side_combo.isHidden()


def test_fb_count_fit_runs_and_recovers_alpha(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb

    captured = []
    window.count_fit_completed.connect(lambda dataset, payload: captured.append(payload["result"]))
    window._single_fit_tab._run_count_domain_fit()
    assert window._single_fit_tab.wait_for_fit()

    assert len(captured) == 1
    result = captured[0]
    assert result.success
    alpha = result.group_results[1].parameters["alpha"].value
    assert alpha == pytest.approx(1.25, abs=0.05)


def test_count_fit_uses_separate_worker_handle(qapp, fb_dataset, monkeypatch):
    """A count-domain fit owns its own worker handle, distinct from _fit_worker.

    Every fit now runs on the shared TaskRunner, but the count-domain and the
    global/grouped paths keep SEPARATE live handles (_count_fit_worker vs
    _fit_worker) so the shared Stop button cancels exactly the running fit.
    """
    import asymmetry.gui.panels.fit.global_tab as fit_panel_mod
    from asymmetry.core.fitting.engine import FitCancelledError

    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb
    tab = window._single_fit_tab

    idle = threading.Event()  # never set; used only as an interruptible sleep

    def blocking_fb(*args, cancel_callback=None, **kwargs):
        while cancel_callback is None or not cancel_callback():
            idle.wait(0.005)
        raise FitCancelledError("cancelled")

    monkeypatch.setattr(fit_panel_mod, "fit_fb_alpha", blocking_fb)

    tab._run_count_domain_fit()
    # The count fit owns its own handle; the global/grouped slot stays empty.
    assert tab._count_fit_worker is not None
    assert tab._fit_worker is None

    tab._on_stop_fit()
    assert tab.wait_for_fit()
    assert tab._count_fit_worker is None


def test_single_count_fit_runs_and_emits(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # Single group
    window._side_combo.setCurrentIndex(0)  # Forward
    assert window._single_fit_tab._count_single_side == "forward"
    assert window._side_combo.isEnabled()

    captured = []
    window.count_fit_completed.connect(lambda dataset, payload: captured.append(payload["result"]))
    window._single_fit_tab._run_count_domain_fit()
    assert window._single_fit_tab.wait_for_fit()
    assert len(captured) == 1
    assert "N0" in captured[0].parameters.names


def test_all_groups_mode_leaves_existing_path(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    # Default is All groups; the count-domain routing must not intercept it.
    assert window._single_fit_tab._count_fit_mode == "all"


def test_phase2_controls_push_to_tabs(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb → count mode enables the controls
    assert window._t0_check.isEnabled()

    window._exclude_min.setValue(2.0)
    window._exclude_max.setValue(3.0)
    window._t0_check.setChecked(True)
    window._baseline_check.setChecked(True)

    tab = window._single_fit_tab
    assert tab._count_exclude == (2.0, 3.0)
    assert tab._count_fit_t0 is True
    assert tab._count_baseline is True

    # An inverted window disables the exclude (max ≤ min → None).
    window._exclude_max.setValue(1.0)
    assert tab._count_exclude is None


def test_count_controls_disabled_for_all_groups(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(0)  # All groups
    assert not window._t0_check.isEnabled()
    assert not window._exclude_min.isEnabled()
    assert not window._deadtime_check.isEnabled()
    assert not window._promote_btn.isEnabled()


def test_phase3_controls_push_to_tabs(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    window._deadtime_check.setChecked(True)
    window._dpsep_spin.setValue(0.324)
    tab = window._single_fit_tab
    assert tab._count_deadtime is True
    assert tab._count_dpsep == pytest.approx(0.324)


def test_colliding_model_name_reports_error(qapp, fb_dataset):
    """A model parameter named like a reserved count-fit slot is rejected loudly."""

    class _Model:
        param_names = ["A", "dpsep", "phase"]  # 'dpsep' collides with the reserved name

    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb
    tab = window._single_fit_tab
    with pytest.raises(ValueError, match="collide with reserved count-fit names"):
        tab._count_fit_seed_params(fb_dataset, _Model(), mode="fb")


def test_dpsep_fit_toggle_routes_to_tab(qapp, fb_dataset):
    """The dpsep 'fit' checkbox flips the tab into scan-refinement mode."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single → count mode enables dpsep
    window._dpsep_spin.setValue(0.324)
    window._dpsep_fit_check.setChecked(True)
    tab = window._single_fit_tab
    assert window._dpsep_fit_check.isEnabled()
    assert tab._count_dpsep_fit is True
    # A scan-mode seed yields a FREE dpsep bracketing the instrument value.
    params = tab._count_fit_seed_params(fb_dataset, tab._grouped_fit_model(), mode="single")
    dpsep = params["dpsep"]
    assert not dpsep.fixed
    assert dpsep.min < 0.324 < dpsep.max


def test_promote_button_without_fit_shows_hint(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    # No deadtime fit run yet → promoting reports a hint rather than mutating.
    window._on_promote_deadtime()
    grouping = fb_dataset.run.grouping
    assert "Run a deadtime count fit" in window._single_fit_tab._result_text.toPlainText()
    assert grouping.get("deadtime_correction") in (None, False)


def test_promote_after_deadtime_fit_writes_grouping(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    window._deadtime_check.setChecked(True)
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()
    assert tab._last_count_dt0 is not None
    window._on_promote_deadtime()
    grouping = fb_dataset.run.grouping
    assert grouping.get("deadtime_correction") is True
    assert any(v != 0.0 for v in grouping.get("dead_time_us", []))


def test_fb_double_pulse_target_runs_via_dpsep_control(qapp):
    """The dpsep control routes to the F+B target and the fit recovers α."""
    template = build_builtin_template("ideal_continuous_fb")
    run = simulate_double_pulse_run(
        template, _tf, {"A": 20.0, "f": 1.0, "phi": 0.0},
        total_events=20e6, dpsep_us=0.324, alpha=1.2, seed=4,
    )  # fmt: skip
    ds = MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )
    window = MultiGroupFitWindow()
    window.set_dataset(ds)
    window._target_combo.setCurrentIndex(1)  # fb
    window._cost_combo.setCurrentIndex(1)  # Gaussian √N
    window._dpsep_spin.setValue(0.324)
    assert window._dpsep_spin.isEnabled()  # dpsep control active for the F+B target
    assert window._single_fit_tab._count_dpsep == pytest.approx(0.324)

    captured = []
    window.count_fit_completed.connect(lambda dataset, payload: captured.append(payload["result"]))
    window._single_fit_tab._run_count_domain_fit()
    assert window._single_fit_tab.wait_for_fit()

    assert len(captured) == 1 and captured[0].success
    assert captured[0].group_results[1].parameters["alpha"].value == pytest.approx(1.2, abs=0.05)


def test_fb_fit_emits_overlay_for_both_banks(qapp, fb_dataset):
    """A finished F+B fit emits overlay curves keyed by both bank group ids."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb

    overlays = []
    window.count_fit_completed.connect(
        lambda dataset, payload: overlays.append(payload["overlays"])
    )
    window._single_fit_tab._run_count_domain_fit()
    assert window._single_fit_tab.wait_for_fit()

    assert len(overlays) == 1
    forward, backward = window._single_fit_tab._count_fb_groups(fb_dataset)
    assert set(overlays[0]) == {forward, backward}
    for _gid, (time, corrected) in overlays[0].items():
        assert time.size == corrected.size and time.size > 0
        assert np.all(np.isfinite(corrected))


def test_single_fit_emits_overlay_for_target_group(qapp, fb_dataset):
    """A finished single-histogram fit emits one overlay keyed by the target group."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    window._side_combo.setCurrentIndex(0)  # Forward

    overlays = []
    window.count_fit_completed.connect(
        lambda dataset, payload: overlays.append(payload["overlays"])
    )
    window._single_fit_tab._run_count_domain_fit()
    assert window._single_fit_tab.wait_for_fit()

    forward, _backward = window._single_fit_tab._count_fb_groups(fb_dataset)
    assert len(overlays) == 1
    assert set(overlays[0]) == {forward}


def test_promote_alpha_after_fb_fit_writes_grouping(qapp, fb_dataset):
    """F7: a finished F+B fit's α promotes into the grouping with count_fit provenance."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()
    assert tab._last_count_alpha is not None

    promoted = []
    window.count_grouping_promoted.connect(lambda dataset: promoted.append(dataset))
    window._promote_alpha_btn.click()

    grouping = fb_dataset.run.grouping
    assert grouping["alpha"] == pytest.approx(1.25, abs=0.05)
    assert grouping["alpha_method"] == "count_fit"
    assert grouping["alpha_reference_run"] == fb_dataset.run_number
    assert len(promoted) == 1


def test_switching_run_clears_captured_calibrations(qapp, fb_dataset):
    """A captured calibration must not survive a run switch (wrong-run promote)."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()
    assert tab._last_count_alpha is not None

    other = MuonDataset(
        time=np.array([]),
        asymmetry=np.array([]),
        error=np.array([]),
        metadata={},
        run=fb_dataset.run,  # any different dataset object
    )
    window.set_dataset(other)
    assert tab._last_count_alpha is None
    # Promoting now reports a hint rather than writing run A's α into run B.
    window._promote_alpha_btn.click()
    assert "Forward + Backward" in tab._result_text.toPlainText()


def test_promote_alpha_without_fb_fit_shows_hint(qapp, fb_dataset):
    """Promoting α without a forward/backward fit reports a hint, not a mutation."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single — no α produced
    window._promote_alpha_btn.click()
    assert "Forward + Backward" in window._single_fit_tab._result_text.toPlainText()
    assert "alpha_method" not in fb_dataset.run.grouping


def test_promote_t0_after_fit_writes_t0_bin_and_discloses_residual(qapp, fb_dataset):
    """F5: a fitted t₀ offset promotes to t0_bin; the residual is disclosed."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    window._t0_check.setChecked(True)
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()
    assert tab._last_count_t0_us is not None

    window._promote_t0_btn.click()
    grouping = fb_dataset.run.grouping
    assert grouping["t0_method"] == "count_fit"
    text = tab._result_text.toPlainText()
    assert "residual" in text.lower()
    assert "run-wide" in text


def test_promote_background_after_fb_fit_sets_fixed_mode(qapp, fb_dataset):
    """N3: fitted flat backgrounds promote to grouping fixed mode."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()
    assert tab._last_count_bg is not None

    window._promote_bg_btn.click()
    grouping = fb_dataset.run.grouping
    assert grouping["background_mode"] == "fixed"
    assert grouping["background_method"] == "count_fit"
    assert len(grouping["background_fixed_values"]) == 2


def test_background_active_note_visibility_tracks_grouping(qapp, fb_dataset):
    """N3 guard: the count-fit note appears only when grouping background is on."""
    # isHidden() reflects the explicit visibility flag without a shown ancestor
    # (isVisible() is always False until the top-level window is shown).
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    assert window._bg_active_note.isHidden()

    fb_dataset.run.grouping["background_correction"] = True
    fb_dataset.run.grouping["background_mode"] = "tail_fit"
    window.set_dataset(fb_dataset)
    assert not window._bg_active_note.isHidden()


def test_skip_window_label_is_relabelled(qapp):
    """F8: the count-fit exclude control is relabelled to encode its semantics."""
    window = MultiGroupFitWindow()
    # "Skip" (not the bare "Exclude") still encodes the hard-skip semantics that
    # distinguish this from the MaxEnt de-weight window; the tooltip has the rest.
    assert window._exclude_label.text() == "Skip (µs)"
    assert "exclude" in window._exclude_label.toolTip().lower()


def test_count_skip_window_round_trips_through_state(qapp):
    """NEW-R1: the count-fit skip window persists and restores via window state."""
    window = MultiGroupFitWindow()
    window._exclude_min.setValue(2.5)
    window._exclude_max.setValue(4.0)
    state = window.get_state()
    assert state["count_skip_window"] == [2.5, 4.0]

    restored = MultiGroupFitWindow()
    restored.restore_state(state)
    assert restored._exclude_min.value() == pytest.approx(2.5)
    assert restored._exclude_max.value() == pytest.approx(4.0)
    # The restored window pushes the skip window down to its tabs.
    assert restored._single_fit_tab._count_exclude == (2.5, 4.0)


def test_promote_uses_dedicated_signal_not_a_fit_none(qapp, fb_dataset):
    """Promote emits count_grouping_promoted, never a fit-shaped None on the fit signal."""
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(2)  # single
    window._deadtime_check.setChecked(True)
    tab = window._single_fit_tab
    tab._run_count_domain_fit()
    assert tab.wait_for_fit()

    fit_payloads = []
    promoted = []
    window.count_fit_completed.connect(lambda dataset, result: fit_payloads.append(result))
    window.count_grouping_promoted.connect(lambda dataset: promoted.append(dataset))
    window._on_promote_deadtime()

    # The fit-result signal never fires for a promote (so no None sentinel leaks).
    assert fit_payloads == []
    assert len(promoted) == 1
    assert promoted[0] is fb_dataset


# ── FB-parity for the Single grouped surface (carry-forward / share / snapshot) ──


def _grouped_run_dataset(run_number: int) -> MuonDataset:
    """A grouped (F-B) dataset stamped with a distinct run number."""
    template = build_builtin_template("ideal_pulsed_fb")
    run = simulate_run(
        template, _tf, {"A": 20.0, "f": 1.5, "phi": 0.3}, total_events=4e6, alpha=1.25, seed=1
    )
    run.run_number = run_number
    return MuonDataset(
        time=np.array([]), asymmetry=np.array([]), error=np.array([]), metadata={}, run=run
    )


def _set_single_model(window: MultiGroupFitWindow, names: list[str]) -> None:
    window._single_fit_tab._set_composite_model(
        CompositeModel(names, operators=["+"] * (len(names) - 1))
    )


def _single_model_names(window: MultiGroupFitWindow) -> list[str]:
    return list(window._single_fit_tab._composite_model.component_names)


def test_grouped_single_carries_function_forward_and_drops_result(qapp):
    # Moving to an unseen run keeps the current function but drops the previous
    # run's result (it belongs to the run it was computed on).
    win = MultiGroupFitWindow()
    a = _grouped_run_dataset(701)
    b = _grouped_run_dataset(702)
    win.set_dataset(a)
    _set_single_model(win, ["Gaussian", "Constant"])
    win._single_fit_tab._result_text.setHtml("<b>chi2 = 1.0</b>")

    win.set_dataset(b)  # unseen run, no recorded fit, no stored form
    assert _single_model_names(win) == ["Gaussian", "Constant"]  # function carried
    assert win._single_fit_tab._result_text.toPlainText().strip() == ""  # result dropped


def test_grouped_single_per_run_form_round_trips(qapp):
    # Each run remembers its own in-progress function; switching away and back
    # restores it rather than carrying the other run's function.
    win = MultiGroupFitWindow()
    a = _grouped_run_dataset(703)
    b = _grouped_run_dataset(704)
    win.set_dataset(a)
    _set_single_model(win, ["Gaussian", "Constant"])
    win.set_dataset(b)
    _set_single_model(win, ["Exponential", "Constant"])

    win.set_dataset(a)
    assert _single_model_names(win) == ["Gaussian", "Constant"]
    win.set_dataset(b)
    assert _single_model_names(win) == ["Exponential", "Constant"]


def test_grouped_single_form_survives_tab_switch(qapp):
    # An in-progress (unfit) function survives a Single↔Batch round trip.
    win = MultiGroupFitWindow()
    win.set_dataset(_grouped_run_dataset(705))
    _set_single_model(win, ["Gaussian", "Constant"])
    win._tabs.setCurrentIndex(1)  # → Batch (snapshots the Single form)
    _set_single_model(win, ["Exponential", "Constant"])  # disturb it
    win._tabs.setCurrentIndex(0)  # → Single (restores the snapshot)
    assert _single_model_names(win) == ["Gaussian", "Constant"]


def test_grouped_single_share_with_group_copies_to_peers(qapp):
    # Share-with-Group copies the function into each peer run's stored form, so
    # selecting a peer inherits it.
    win = MultiGroupFitWindow()
    a = _grouped_run_dataset(706)
    b = _grouped_run_dataset(707)
    win.set_dataset(a)
    _set_single_model(win, ["Gaussian", "Constant"])

    assert win.share_single_grouped_function_state(706, [707]) == 1
    win.set_dataset(b)
    assert _single_model_names(win) == ["Gaussian", "Constant"]


def _single_physics_rows(tab) -> dict[str, int]:
    table = tab._group_model_table
    return {
        table.item(r, FitParameterTable.COL_NAME).data(Qt.ItemDataRole.UserRole): r
        for r in range(table.rowCount())
    }


def test_grouped_single_multi_oscillatory_seeds_all_field_params(qapp):
    # A model with more than one oscillatory component seeds *every* field param
    # from the run's applied field, not just the first (the rest used to keep the
    # 100 G component default).
    win = MultiGroupFitWindow()
    ds = _grouped_run_dataset(730)
    ds.run.metadata["field"] = 250.0
    win.set_dataset(ds)
    tab = win._single_fit_tab
    tab._set_composite_model(
        CompositeModel(["OscillatoryField", "OscillatoryField", "Constant"], operators=["+", "+"])
    )
    table = tab._group_model_table
    rows = _single_physics_rows(tab)
    for fname in ("field_1", "field_2"):
        assert float(table.item(rows[fname], FitParameterTable.COL_VALUE).text()) == pytest.approx(
            250.0
        )


def test_grouped_single_oscillatory_phase_fixed_at_zero(qapp):
    # The individual-groups fit holds every oscillation phase fixed at zero by
    # default; the phase lives in the per-group relative_phase nuisances.
    win = MultiGroupFitWindow()
    ds = _grouped_run_dataset(731)
    ds.run.metadata["field"] = 180.0
    win.set_dataset(ds)
    tab = win._single_fit_tab
    tab._set_composite_model(
        CompositeModel(["OscillatoryField", "OscillatoryField", "Constant"], operators=["+", "+"])
    )
    table = tab._group_model_table
    rows = _single_physics_rows(tab)
    for pname in ("phase_1", "phase_2"):
        assert float(table.item(rows[pname], FitParameterTable.COL_VALUE).text()) == pytest.approx(
            0.0
        )
        checkbox = table.cellWidget(rows[pname], FitParameterTable.COL_FIX).findChild(QCheckBox)
        assert checkbox is not None and checkbox.isChecked()


def test_grouped_single_share_reseeds_field_params_per_peer(qapp):
    # Sharing a grouped Single function to a peer at a *different* applied field
    # must re-seed that peer's field-specific parameters (e.g. B_L) from the
    # peer's own dataset — in BOTH the grouped-fit "parameters" list and the
    # per-group "group_model_parameters" list — rather than leaving the source
    # run's field value. Mirrors the FB single-fit re-seeding.
    win = MultiGroupFitWindow()
    source = _grouped_run_dataset(714)
    source.run.metadata["field"] = 50.0
    peer = _grouped_run_dataset(715)
    peer.run.metadata["field"] = 300.0

    # Craft a source state carrying B_L at the source field (50 G) in both
    # parameter lists, plus a non-field parameter that must stay put.
    win._single_grouped_state_by_run[714] = {
        "model_name": "Composite",
        "result_html": "<b>source fit</b>",
        "parameters": [
            {"name": "B_L", "value": 50.0},
            {"name": "A", "value": 0.2},
        ],
        "group_model_parameters": [
            {"name": "B_L", "value": 50.0},
            {"name": "Lambda", "value": 1.0},
        ],
    }

    datasets_by_run = {714: source, 715: peer}
    assert win.share_single_grouped_function_state(714, [715], datasets_by_run=datasets_by_run) == 1

    shared = win._single_grouped_state_by_run[715]
    params = {p["name"]: p["value"] for p in shared["parameters"]}
    group_params = {p["name"]: p["value"] for p in shared["group_model_parameters"]}
    # B_L re-seeded from the peer's 300 G field in both lists.
    assert params["B_L"] == pytest.approx(300.0)
    assert group_params["B_L"] == pytest.approx(300.0)
    # Non-field parameters keep the shared source value.
    assert params["A"] == pytest.approx(0.2)
    assert group_params["Lambda"] == pytest.approx(1.0)


def test_grouped_single_share_does_not_propagate_result_to_peers(qapp):
    # Sharing must copy the function but NOT the source run's fit result — an
    # unfit peer should never display a fit it did not perform.
    win = MultiGroupFitWindow()
    a = _grouped_run_dataset(708)
    b = _grouped_run_dataset(709)
    win.set_dataset(a)
    _set_single_model(win, ["Gaussian", "Constant"])
    win._single_fit_tab._result_text.setHtml("<b>chi2 = 1.0 CONVERGED</b>")

    win.share_single_grouped_function_state(708, [709])
    win.set_dataset(b)
    assert _single_model_names(win) == ["Gaussian", "Constant"]  # function shared
    assert "chi2" not in win._single_fit_tab._result_text.toPlainText()  # result not shared


def test_grouped_single_state_cleared_on_project_reset(qapp):
    # The per-run form store must not bleed across projects: clearing drops it,
    # so a reused run number starts clean instead of inheriting a stale function.
    win = MultiGroupFitWindow()
    win.set_dataset(_grouped_run_dataset(710))
    _set_single_model(win, ["Gaussian", "Constant"])
    win.set_dataset(_grouped_run_dataset(711))  # stores 710's form
    assert 710 in win._single_grouped_state_by_run

    win.clear_grouped_single_state()
    assert win._single_grouped_state_by_run == {}
    assert win._active_single_grouped_run is None


def test_grouped_single_state_pruned_on_run_removal(qapp):
    # Removing/refitting a run forgets its stored grouped form.
    win = MultiGroupFitWindow()
    win.set_dataset(_grouped_run_dataset(712))
    _set_single_model(win, ["Gaussian", "Constant"])
    win.set_dataset(_grouped_run_dataset(713))  # stores 712's form
    assert 712 in win._single_grouped_state_by_run

    win.prune_grouped_single_state([712])
    assert 712 not in win._single_grouped_state_by_run
