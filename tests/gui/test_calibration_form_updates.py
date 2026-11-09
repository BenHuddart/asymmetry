"""Completed calibration populates the draft form; Apply commits the saved owner."""

from copy import deepcopy

import pytest

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from docs.screenshots.data.calibration import calibration_control
from tests.gui.test_grouping_dialog import _wait_for_t0_detection, _wait_until

pytestmark = pytest.mark.gui


@pytest.mark.parametrize(
    "method,fit_beta",
    [
        ("tf_calibration_fit", False),
        ("tf_calibration_fit", True),
        ("diamagnetic", False),
        ("general", False),
        ("ratio", False),
    ],
)
def test_completed_calibration_updates_spins_and_method_before_apply(qapp, method, fit_beta):
    data = calibration_control(beta=0.8 if fit_beta else 1, relaxing=method == "general")
    data.run.grouping["beta"] = 1.3
    before = deepcopy(data.run.grouping)
    dialog = GroupingDialog([data])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    section._method_combo.setCurrentIndex(section._method_combo.findData(method))
    section._fit_beta.setChecked(fit_beta)
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    outcome = section.pending()["single"]
    assert dialog._alpha_spin.value() == pytest.approx(outcome.alpha, abs=0.5e-6)
    assert dialog._beta_section.value() == pytest.approx(
        outcome.selected.beta if outcome.record else 1, abs=0.5e-6
    )
    assert section._method_combo.currentText() in dialog._alpha_provenance_label.text()
    if fit_beta:
        assert "TF calibration fit" in dialog._beta_section._provenance_label.text()
    else:
        assert dialog._beta_section._provenance_label.text() == "fixed"
    assert data.run.grouping == before
    # A value-preserving edit retains the method; a changed coefficient stays manual.
    dialog._alpha_spin.setValue(dialog._alpha_spin.value())
    assert "manual" not in dialog._alpha_provenance_label.text()
    dialog._alpha_spin.setValue(1.7)
    assert dialog._alpha_provenance_label.text() == "manual"
    dialog._beta_section.set_value(1.1)
    assert dialog._beta_section._provenance_label.text() == "manual"
    dialog._beta_section.set_value(1.0)
    assert dialog._beta_section._provenance_label.text() == "manual"
    dialog._clear_dirty()
    dialog.close()


def test_vector_results_populate_each_pair_without_applying_to_the_run(qapp):
    data = calibration_control(vector=True)
    before = deepcopy(data.run.grouping)
    dialog = GroupingDialog([data])
    _wait_for_t0_detection(dialog)
    dialog._estimate_all_alpha()
    _wait_until(lambda: dialog._alpha_section._tasks.active_count == 0)
    outcomes = dialog._alpha_section.pending()
    assert set(outcomes) == set(dialog._vector_alpha_spins)
    for label, outcome in outcomes.items():
        assert dialog._vector_alpha_spins[label].value() == pytest.approx(outcome.alpha, abs=0.5e-6)
    assert data.run.grouping == before
    dialog._clear_dirty()
    dialog.close()


def test_filled_joint_values_keep_canonical_methods_after_apply_and_reopen(qapp, tmp_path):
    from asymmetry.core.project.profiles import GroupingProfile
    from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project

    data = calibration_control(beta=0.8)
    dialog = GroupingDialog([data])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    section._fit_beta.setChecked(True)
    section._on_estimate()
    _wait_until(lambda: section._tasks.active_count == 0)
    record = section.pending()["single"].record
    dialog._preview_pane.flush()
    _wait_until(lambda: dialog._preview_pane.corrections_complete)
    dialog._on_apply()
    profile = dialog.get_profile_result()["profile"]
    assert profile.alpha_policy.record is profile.beta_policy.record
    assert profile.alpha_policy.record == record
    path = tmp_path / "filled.asymp"
    save_project(
        {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "datasets": [],
            "grouping_profiles": [profile.to_dict()],
        },
        path,
    )
    restored = GroupingProfile.from_dict(load_project(path)["grouping_profiles"][0])
    reopened = GroupingDialog([data], profiles=[restored])
    assert "TF calibration fit" in reopened._alpha_provenance_label.text()
    assert reopened._beta_section._provenance_label.text() == "TF calibration fit"
    assert reopened._alpha_section._tasks.active_count == 0
    reopened._clear_dirty()
    reopened.close()


@pytest.mark.parametrize("finish", ["failure", "stop"])
@pytest.mark.parametrize("failed_slot", ["P_y", "P_x"])
def test_vector_batch_does_not_publish_partial_results(qapp, monkeypatch, finish, failed_slot):
    from threading import Event

    from asymmetry.gui.windows.grouping import alpha_section as module

    data = calibration_control(vector=True)
    dialog = GroupingDialog([data])
    _wait_for_t0_detection(dialog)
    section = dialog._alpha_section
    before = dialog._draft.to_dict()
    spins = {label: spin.value() for label, spin in dialog._vector_alpha_spins.items()}
    entered, release = Event(), Event()
    original = module.run_calibration
    published = []
    section.result_ready.connect(published.append)

    def held(worker, request):
        if request.slot == failed_slot:
            entered.set()
            assert release.wait(5)
            raise ValueError("owned second-pair failure")
        return original(worker, request)

    monkeypatch.setattr(module, "run_calibration", held)
    try:
        dialog._estimate_all_alpha()
        _wait_until(entered.is_set)
        during = list(published)
        pending_during = section.pending()
        if finish == "stop":
            section._stop_btn.click()
        release.set()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert pending_during == {}
        assert during == []
        assert published == []
        assert section.pending() == {}
        assert dialog._draft.to_dict() == before
        assert {label: spin.value() for label, spin in dialog._vector_alpha_spins.items()} == spins
        dialog._preview_pane.flush()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._apply_btn.isEnabled()
        dialog._on_apply()
        assert dialog.get_profile_result()["profile"].calibration_records == ()
        monkeypatch.setattr(module, "run_calibration", original)
        dialog._estimate_all_alpha()
        _wait_until(lambda: section._tasks.active_count == 0)
        assert len(published) == 3
        assert set(section.pending()) == set(dialog._vector_alpha_spins)
        assert len(dialog._draft.calibration_records) == 3
    finally:
        release.set()
        dialog._clear_dirty()
        dialog.close()
