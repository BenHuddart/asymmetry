"""GUI wiring for the count-domain fit-target selector in the Multi-Group window.

These tests exercise the routing only — the count-fit numerics are covered by
``tests/test_count_domain_fits.py``. The detector-balance α is recovered here
even with the default model because it is fixed by the forward/backward count
ratio, independent of the oscillation model.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.simulate import (
    build_builtin_template,
    simulate_double_pulse_run,
    simulate_run,
)
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
    # Cost selector is enabled for count modes, disabled for All groups.
    assert window._cost_combo.isEnabled()
    window._target_combo.setCurrentIndex(0)
    assert window._single_fit_tab._count_fit_mode == "all"
    assert not window._cost_combo.isEnabled()


def test_fb_count_fit_runs_and_recovers_alpha(qapp, fb_dataset):
    window = MultiGroupFitWindow()
    window.set_dataset(fb_dataset)
    window._target_combo.setCurrentIndex(1)  # fb

    captured = []
    window.count_fit_completed.connect(lambda dataset, payload: captured.append(payload["result"]))
    window._single_fit_tab._run_count_domain_fit()

    assert len(captured) == 1
    result = captured[0]
    assert result.success
    alpha = result.group_results[1].parameters["alpha"].value
    assert alpha == pytest.approx(1.25, abs=0.05)


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
    assert tab._last_count_alpha is not None

    promoted = []
    window.count_grouping_promoted.connect(lambda dataset: promoted.append(dataset))
    window._promote_alpha_btn.click()

    grouping = fb_dataset.run.grouping
    assert grouping["alpha"] == pytest.approx(1.25, abs=0.05)
    assert grouping["alpha_method"] == "count_fit"
    assert grouping["alpha_reference_run"] == fb_dataset.run_number
    assert len(promoted) == 1


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
    assert window._exclude_label.text() == "Skip window (μs)"


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

    fit_payloads = []
    promoted = []
    window.count_fit_completed.connect(lambda dataset, result: fit_payloads.append(result))
    window.count_grouping_promoted.connect(lambda dataset: promoted.append(dataset))
    window._on_promote_deadtime()

    # The fit-result signal never fires for a promote (so no None sentinel leaks).
    assert fit_payloads == []
    assert len(promoted) == 1
    assert promoted[0] is fb_dataset
