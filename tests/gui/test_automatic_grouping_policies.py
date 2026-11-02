"""Current automatic previews gate Apply and preserve the selected profile policy."""

import threading

import numpy as np
import pytest

from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    BetaPolicy,
    DeadtimePolicy,
    profile_fingerprint_for_run,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import load_project, save_project
from asymmetry.core.transform.grouping import independent_pair_indices
from asymmetry.core.transform.rebin import binning_spec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping import preview_pane as preview_module
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_dialog import _vector_dataset_with_ratio
from tests.gui.test_grouping_preview_pane import _histogram_dataset, _wait_until
from tests.project.test_automatic_grouping_policies import automatic_profile, policy_run

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("calibrated", [False, True])
def test_vector_pz_anchor_reaches_profile_and_preview(qapp, calibrated):
    dataset = _vector_dataset_with_ratio(5201, ratio=2.0)
    dialog = GroupingDialog([dataset])
    try:
        spin = dialog._vector_alpha_spins["P_z"]
        if calibrated:
            dialog._apply_calibrated_policy(
                "P_z",
                spin,
                AlphaPolicy(mode="calibrated", value=2.0, method="diamagnetic", source_run=5201),
            )
        else:
            spin.setValue(2.0)
        dialog._refresh_preview()
        dialog._preview_pane.flush()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._preview_pane._last_result.alpha == 2.0
        assert resolve_effective_grouping(dialog.draft_profile, dataset.run)["alpha"] == 2.0
        if calibrated:
            assert dialog.draft_profile.alpha_policy.source_run == 5201
    finally:
        dialog._clear_dirty()
        dialog.close()


def automatic_dialog(levels=(10, 30, 50)):
    dataset = _histogram_dataset()
    dataset.run = policy_run(levels)
    dataset.run.run_number = dataset.run_number
    profile = automatic_profile()
    profile.fingerprint = profile_fingerprint_for_run(dataset.run)
    dialog = GroupingDialog([dataset], profiles=[profile])
    return dialog, dataset


def test_unity_beta_calibration_keeps_provenance_through_form_and_project(qapp, tmp_path):
    dataset = _histogram_dataset()
    profile = automatic_profile()
    profile.groups = {1: [1], 2: [2]}
    profile.fingerprint = profile_fingerprint_for_run(dataset.run)
    beta = BetaPolicy(mode="calibrated", value=1.0, error=0.02, method="count_fit", source_run=999)
    profile.beta_policy = beta
    dialog = GroupingDialog([dataset], profiles=[profile])
    try:
        dialog._bunch_spin.setValue(2)
        assert dialog.draft_profile.beta_policy == beta
        project = tmp_path / "unity.asymp"
        state = MainWindow().collect_project_state()
        state["grouping_profiles"] = [dialog.draft_profile.to_dict()]
        save_project(state, project)
        saved = load_project(project)["grouping_profiles"][0]
        assert saved["beta_reference_run"] == 999
        assert saved["beta_error"] == 0.02
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_fresh_inheritance_resolves_its_loaded_reference(qapp, monkeypatch, tmp_path):
    sample = _histogram_dataset(forward=np.full(6, 40.0), backward=np.full(6, 60.0))
    reference = _histogram_dataset(
        run_number=999, forward=np.full(6, 10.0), backward=np.full(6, 5.0)
    )
    for dataset in (sample, reference):
        dataset.run.grouping["good_frames"] = 100
    profile = automatic_profile()
    profile.groups = {1: [1], 2: [2]}
    profile.fingerprint = profile_fingerprint_for_run(sample.run)
    profile.background_policy = BackgroundPolicy(
        mode="reference_run", details={"background_run": {"run_number": 999}}
    )
    window = MainWindow()
    window._data_browser.add_dataset(reference)
    window._grouping_profiles = [profile]
    source = tmp_path / "sample.nxs"
    source.touch()
    monkeypatch.setattr(
        type(window), "_load_paths_with_progress", lambda self, paths: {str(source): sample}
    )
    window._load_files([str(source)])
    assert "correction_error" not in sample.metadata
    assert sample.run.grouping["alpha"] == pytest.approx(30 / 55)


@pytest.mark.parametrize("manual_deadtime", [False, True])
def test_green_preview_and_applied_reduction_use_current_exposure_and_coefficients(
    qapp, manual_deadtime
):
    sample = _histogram_dataset(forward=np.full(6, 20.0), backward=np.full(6, 30.0))
    green = _histogram_dataset(forward=np.full(6, 40.0), backward=np.full(6, 60.0))
    reference = _histogram_dataset(
        run_number=999, forward=np.full(6, 10.0), backward=np.full(6, 5.0)
    )
    reference.run.grouping["good_frames"] = 100
    sample.run.grouping.update(
        period_histograms=[sample.run.histograms, green.run.histograms],
        good_frames=100,
        period_good_frames=[100, 200],
        period_dead_time_us=[[0.1, 0.2], [0.1, 0.2]],
    )
    profile = automatic_profile()
    profile.groups = {1: [1], 2: [2]}
    profile.fingerprint = profile_fingerprint_for_run(sample.run)
    profile.period_mode = "green"
    profile.background_policy = BackgroundPolicy(
        mode="reference_run", details={"background_run": {"run_number": 999}}
    )
    if manual_deadtime:
        profile.deadtime_policy = DeadtimePolicy(mode="manual", values=[0.01, 0.02])
    dialog = GroupingDialog(
        [sample, reference], profiles=[profile], selected_run_number=sample.run_number
    )
    window = MainWindow()
    window._data_browser.add_dataset(reference)
    try:
        pane = dialog._preview_pane
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        if not manual_deadtime:
            assert pane._last_result.alpha == pytest.approx(0.4)
        effective = resolve_effective_grouping(
            dialog.draft_profile,
            sample.run,
            reference_resolver=window._resolve_background_reference,
        )
        np.testing.assert_allclose(effective["alpha"], pane._last_result.alpha)
        applied, _ = window._apply_grouping_settings_to_dataset(sample, effective)
        assert applied
        np.testing.assert_allclose(sample.asymmetry, pane._last_result.asymmetry, atol=1e-12)
        np.testing.assert_allclose(sample.asymmetry, 0, atol=1e-12)
        assert sample.run.grouping["good_frames"] == 200
        if manual_deadtime:
            assert sample.run.grouping["dead_time_us"] == [0.01, 0.02]
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_group_edit_keeps_automatic_policy_through_preview_and_apply(qapp):
    dialog, dataset = automatic_dialog()
    try:
        pane = dialog._preview_pane
        pane.flush()
        _wait_until(lambda: pane._last_result is not None)
        assert pane._last_result.alpha == pytest.approx(0.2)
        dialog._group_table.item(0, 3).setText("1,2")
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        assert pane._last_result.alpha == pytest.approx(0.8)
        assert dialog._apply_btn.isEnabled()
        dialog._on_apply()
        applied = dialog.get_profile_result()["profile"]
        assert applied.alpha_policy.mode == "per_run_estimate"
        effective = resolve_effective_grouping(applied, dataset.run)
        forward, backward = independent_pair_indices(effective, n_histograms=3)
        reduced = reduce_grouped_asymmetry(
            histograms=dataset.run.histograms,
            grouping=effective,
            binning=binning_spec(effective),
            forward_idx=forward,
            backward_idx=backward,
            alpha=effective["alpha"],
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )
        np.testing.assert_allclose(reduced.asymmetry, pane._last_result.asymmetry)
        np.testing.assert_allclose(reduced.time, pane._last_result.time)
    finally:
        dialog._clear_dirty()
        dialog.close()


@pytest.mark.parametrize("obsolete_members", ["1,2", "3"])
def test_pending_automatic_result_cannot_enable_apply(qapp, monkeypatch, obsolete_members):
    dialog, dataset = automatic_dialog()
    entered, release = threading.Event(), threading.Event()
    real_resolve = preview_module.resolve_effective_grouping
    accepted = []

    def held_resolve(profile, run, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release preview worker"
        return real_resolve(profile, run, **kwargs)

    try:
        pane = dialog._preview_pane
        pane.preview_changed.connect(
            lambda: accepted.append(pane._last_result.alpha) if pane.corrections_complete else None
        )
        pane.flush()
        _wait_until(lambda: pane._last_result is not None)
        monkeypatch.setattr(preview_module, "resolve_effective_grouping", held_resolve)
        dialog._group_table.item(0, 3).setText(obsolete_members)
        pane.flush()
        _wait_until(entered.is_set)
        assert not dialog._apply_btn.isEnabled()
        dialog._group_table.item(0, 3).setText("2")
        pane.flush()
        assert not pane.corrections_complete
        release.set()
        _wait_until(lambda: pane.corrections_complete)
        assert pane._last_result.alpha == pytest.approx(0.6)
        assert accepted == pytest.approx([0.2, 0.6])
        assert dialog._apply_btn.isEnabled()
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()


@pytest.mark.parametrize("mode", ["fixed", "calibrated"])
def test_group_edits_preserve_stored_alpha_in_the_form(qapp, mode):
    dialog, dataset = automatic_dialog()
    try:
        policy = AlphaPolicy(mode=mode, value=1.7)
        if mode == "calibrated":
            policy = AlphaPolicy(mode=mode, value=1.7, error=0.02, method="general", source_run=999)
        dialog._draft.alpha_policy = policy
        dialog._suppress_dirty = True
        dialog._reload_controls_from_seed()
        dialog._suppress_dirty = False
        dialog._group_table.item(0, 3).setText("1,2")
        pane = dialog._preview_pane
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        assert pane._last_result.alpha == 1.7
        assert dialog.draft_profile.alpha_policy == policy
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_automatic_failure_can_be_repaired_and_manual_unity_remains_fixed(qapp):
    dialog, dataset = automatic_dialog((10, 30, 0))
    raw = [hist.counts.copy() for hist in dataset.run.histograms]
    try:
        pane = dialog._preview_pane
        pane.flush()
        _wait_until(lambda: pane._status.text().startswith("Preview unavailable"))
        assert "alpha" in pane._status.text()
        assert not dialog._apply_btn.isEnabled()
        assert dialog.draft_profile.alpha_policy.mode == "per_run_estimate"
        dialog._group_table.item(1, 3).setText("2")
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        assert pane._last_result.alpha == pytest.approx(1 / 3)
        dialog._alpha_spin.setValue(1.1)
        dialog._alpha_spin.setValue(1.0)
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        assert pane._last_result.alpha == 1.0
        assert dialog.draft_profile.alpha_policy == AlphaPolicy(value=1.0)
        for hist, counts in zip(dataset.run.histograms, raw):
            np.testing.assert_array_equal(hist.counts, counts)
        dialog._on_apply()
        assert dialog.result() == dialog.DialogCode.Accepted
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_failed_inherited_policy_keeps_raw_data_and_profile_on_save_reopen(
    qapp, monkeypatch, tmp_path
):
    dataset = _histogram_dataset()
    dataset.run.histograms[1].counts[:] = 0
    raw = [hist.counts.copy() for hist in dataset.run.histograms]
    source = tmp_path / "synthetic.nxs"
    source.touch()
    dataset.run.source_file = str(source)
    profile = automatic_profile()
    profile.groups = {1: [1], 2: [2]}
    profile.fingerprint = profile_fingerprint_for_run(dataset.run)
    window = MainWindow()
    window._grouping_profiles = [profile]
    monkeypatch.setattr(
        type(window), "_load_paths_with_progress", lambda self, paths: {str(source): dataset}
    )
    monkeypatch.setattr(type(window), "_load_file", lambda self, path: dataset)
    window._load_files([str(source)])
    restored = window._data_browser.get_dataset(dataset.run_number)
    assert restored.metadata["correction_error"]
    project = tmp_path / "automatic.asymp"
    save_project(window.collect_project_state(), project)
    window.restore_project_state(load_project(project), str(project))
    restored = window._data_browser.get_dataset(dataset.run_number)
    assert "alpha" in restored.metadata["correction_error"]
    assert window._grouping_profiles[0].alpha_policy.mode == "per_run_estimate"
    for hist, counts in zip(restored.run.histograms, raw):
        np.testing.assert_array_equal(hist.counts, counts)
