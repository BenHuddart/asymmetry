"""Existing form capture retains frozen calibration until a manual edit."""

import pytest

from asymmetry.core.project.calibration import CalibrationRecord, apply_calibration
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.grouping.profile_bridge import profile_from_form_payload
from tests.project.test_tf_calibration_records import candidate

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("manual_beta", [False, True])
def test_form_capture_preserves_history_and_authored_beta(manual_beta):
    profile = GroupingProfile("TF", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("frozen", candidate()))
    payload = {"groups": {1: [1], 2: [2]}, "alpha": 1.2, "beta": 1.1}
    if not manual_beta:
        payload.update(beta_method="tf_calibration_fit", beta_reference_run=123, beta_error=0.3)
    captured = profile_from_form_payload(
        payload,
        name=profile.name,
        fingerprint=profile.fingerprint,
        alpha_policy=profile.alpha_policy,
        calibration_source=profile,
    )
    restored = GroupingProfile.from_dict(captured.to_dict())
    assert restored.alpha_policy.record is restored.calibration_records[0]
    if manual_beta:
        assert restored.beta_policy.method == ""
        assert restored.beta_policy.value == 1.1
    else:
        assert restored.beta_policy.record is restored.alpha_policy.record


def test_vector_form_capture_keeps_unedited_base_calibration():
    from dataclasses import replace

    from asymmetry.core.project.profiles import resolve_effective_grouping
    from asymmetry.core.transform.projections import projection_alphas
    from tests.core.test_tf_calibration_fit import _raw_tf

    result = candidate()
    result = replace(
        result,
        fit=replace(
            result.fit,
            parameters=(
                replace(result.fit.parameters[0], value=1.23456789),
                result.fit.parameters[1],
            ),
        ),
    )
    profile = GroupingProfile("vector", ProfileFingerprint("test", 2))
    apply_calibration(profile, CalibrationRecord("vector-base", result))
    payload = {
        "groups": {1: [1], 2: [2]},
        "alpha": 1.234568,
        "alpha_method": "tf_calibration_fit",
        "alpha_reference_run": 123,
        "alpha_error": 0.2,
        "projections": [{"label": "pair", "forward_group": 1, "backward_group": 2}],
    }
    captured = profile_from_form_payload(
        payload, name=profile.name, fingerprint=profile.fingerprint, calibration_source=profile
    )
    assert captured.alpha_policy is profile.alpha_policy
    assert captured.beta_policy is profile.beta_policy
    restored = GroupingProfile.from_dict(captured.to_dict())
    assert restored.alpha_policy.value == 1.23456789
    assert restored.alpha_policy.record is restored.beta_policy.record
    grouping = resolve_effective_grouping(restored, _raw_tf().run)
    assert projection_alphas(grouping)["pair"] == 1.23456789


@pytest.mark.parametrize("method", ["tf_calibration_fit", "ratio", "diamagnetic"])
def test_staged_dialog_result_applies_and_reopens_without_refitting(qapp, tmp_path, method):
    from asymmetry.core.project.calibration import QuickCalibratedCoefficient
    from asymmetry.core.project.schema import load_project, save_project
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_until

    data = calibration_control()
    data.run.grouping["beta"] = 1.3
    dialog = GroupingDialog([data])
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection

    _wait_for_t0_detection(dialog)
    before = dialog._draft.to_dict()
    section = dialog._alpha_section
    section._method_combo.setCurrentIndex(section._method_combo.findData(method))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    outcome = section.pending()["single"]
    assert dialog._draft.to_dict() == before
    assert dialog._beta_section.value() == 1.3
    dialog._preview_pane.flush()
    _wait_until(lambda: dialog._preview_pane.corrections_complete)
    dialog._on_apply()
    profile = dialog.get_profile_result()["profile"]
    assert profile.alpha_policy.value == outcome.alpha
    assert profile.beta_policy.value == 1
    if outcome.record:
        assert profile.alpha_policy.record is profile.beta_policy.record
        assert profile.alpha_policy.record == outcome.record
    else:
        assert isinstance(profile.alpha_policy, QuickCalibratedCoefficient)
        assert profile.alpha_policy.result == outcome.result
    path = tmp_path / "calibrated.asymp"
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION

    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "grouping_profiles": [profile.to_dict()],
        },
        path,
    )
    restored = GroupingProfile.from_dict(load_project(path)["grouping_profiles"][0])
    assert restored.to_dict() == profile.to_dict()
    reopened = GroupingDialog([data], profiles=[restored])
    assert reopened._alpha_section._tasks.active_count == 0
    assert reopened._alpha_section.pending() == {}
    assert reopened._draft.alpha_policy.value == outcome.alpha
    reopened._clear_dirty()
    reopened.close()


def test_vector_records_keep_each_pair_with_beta_fixed(qapp, tmp_path):
    from asymmetry.core.project.profiles import resolve_effective_grouping
    from asymmetry.core.transform.projections import projection_alphas
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_until

    data = calibration_control(vector=True)
    dialog = GroupingDialog([data])
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection

    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    assert not section._fit_beta.isEnabled()
    dialog._estimate_all_alpha()
    _wait_until(lambda: section._tasks.active_count == 0)
    outcomes = section.pending()
    assert set(outcomes) == {"P_x", "P_y", "P_z"}
    dialog._commit_calibrations()
    dialog._sync_draft_from_form()
    restored = GroupingProfile.from_dict(dialog._draft.to_dict())
    grouping = resolve_effective_grouping(restored, data.run)
    assert grouping["beta"] == 1
    for label, outcome in outcomes.items():
        policy = restored.projection_alpha_policies[label]
        assert policy.record == outcome.record
        assert (
            policy.result.request.forward_group,
            policy.result.request.backward_group,
        ) == dialog._vector_axis_pairs[label]
        assert policy.result.beta_fixed
        assert projection_alphas(grouping)[label] == policy.value
    dialog._clear_dirty()
    dialog.close()


def test_override_apply_keeps_record_on_override_and_leaves_profile_alone(qapp):
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection, _wait_until

    data = calibration_control()
    data.metadata["grouping_overrides"] = True
    dialog = GroupingDialog([data], overridden_run_numbers={data.run_number})
    _wait_for_t0_detection(dialog)
    assert dialog._editing_target() == data.run_number
    before = dialog._draft.to_dict()
    section = dialog._alpha_section
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    expected = section.pending()["single"].record
    dialog._preview_pane.flush()
    _wait_until(lambda: dialog._preview_pane.corrections_complete)
    dialog._on_apply()
    result = dialog.get_profile_result()
    assert result["profile"].to_dict() == before
    payload = result["override_edits"][data.run_number]
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload

    restored = profile_from_payload(payload, "override", profile_fingerprint_for_run(data.run))
    assert restored.alpha_policy.record == expected
    assert restored.beta_policy.record is restored.alpha_policy.record


def test_mainwindow_override_application_preserves_canonical_record_and_exact_values(
    qapp, tmp_path
):
    from asymmetry.core.project.calibration import (
        apply_calibration_outcome,
        profile_calibration_payload,
    )
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload
    from asymmetry.core.workflow.calibration import calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings
    from asymmetry.gui.mainwindow import MainWindow
    from docs.screenshots.data.calibration import calibration_control

    data = calibration_control()
    outcome = calibrate_run(data, ReductionSettings())
    profile = profile_from_payload(
        data.run.grouping, "source", profile_fingerprint_for_run(data.run)
    )
    apply_calibration_outcome(profile, outcome)
    payload = dict(data.run.grouping, alpha=round(outcome.alpha, 6))
    payload.update(profile_calibration_payload(profile, {}))
    window = MainWindow()
    assert window._apply_grouping_settings_to_dataset(data, payload)[0]
    stored = window._extract_grouping_overrides(data)
    restored = profile_from_payload(stored, "restored", profile.fingerprint)
    assert restored.alpha_policy.record == outcome.record
    assert data.run.grouping["alpha"] == outcome.alpha
    from asymmetry.core.project.schema import load_project, save_project

    data.metadata["grouping_overrides"] = True
    window._data_browser.add_dataset(data)
    path = tmp_path / "override.asymp"
    save_project(window.collect_project_state(), path)
    entries = load_project(path)["datasets"]
    saved = profile_from_payload(entries[0]["grouping_overrides"], "saved", profile.fingerprint)
    assert saved.alpha_policy.record == outcome.record
    window._clear_all_state()
    window.close()


def test_manual_override_edit_drops_quick_policy_but_keeps_authored_value(qapp):
    from asymmetry.core.project.profiles import profile_fingerprint_for_run, profile_from_payload
    from asymmetry.gui.windows.grouping.dialog import GroupingDialog
    from docs.screenshots.data.calibration import calibration_control
    from tests.gui.test_grouping_dialog import _wait_for_t0_detection, _wait_until

    data = calibration_control()
    dialog = GroupingDialog([data], overridden_run_numbers=[data.run_number])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    section._method_combo.setCurrentIndex(section._method_combo.findData("diamagnetic"))
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    dialog._commit_calibrations()
    dialog._alpha_spin.setValue(1.7)
    dialog._sync_draft_from_form()
    payload = dialog._override_drafts[data.run_number]
    restored = profile_from_payload(payload, "manual", profile_fingerprint_for_run(data.run))
    assert restored.alpha_policy.mode == "fixed"
    assert restored.alpha_policy.value == 1.7
    dialog._clear_dirty()
    dialog.close()
