"""Core membership errors remain editable through the GUI/project adapters."""

import numpy as np
import pytest

from asymmetry.gui.mainwindow import MainWindow
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_preview_pane import _histogram_dataset, _wait_until

pytestmark = pytest.mark.gui


@pytest.fixture
def window(qapp):
    return MainWindow()


@pytest.mark.parametrize("members", [[0], [True], [1.5], ["bad"], [1, 1], [1, 4]])
def test_invalid_authored_members_remain_editable_and_unavailable(qapp, members):
    dataset = _histogram_dataset(grouping_extra={"groups": {1: members, 2: [2]}})
    dialog = GroupingDialog([dataset])
    try:
        pane = dialog._preview_pane
        pane.flush()
        _wait_until(lambda: not pane.corrections_complete)
        assert not pane.corrections_complete
        assert not dialog._apply_btn.isEnabled()
        assert dataset.run.grouping["groups"][1] == members
        dialog._group_table.item(0, 3).setText("1")
        pane.flush()
        _wait_until(lambda: pane.corrections_complete)
        assert dialog._apply_btn.isEnabled()
    finally:
        dialog._clear_dirty()
        dialog.close()


@pytest.mark.parametrize("members", [[], [0], [1, 1], [1, 4], ["bad"]])
def test_project_restore_retains_invalid_request_and_raw_data(
    window, monkeypatch, tmp_path, members
):
    dataset = _histogram_dataset()
    source = tmp_path / "synthetic.nxs"
    source.touch()
    monkeypatch.setattr(type(window), "_load_file", lambda self, path: dataset)
    state = window.collect_project_state()
    requested = dict(dataset.run.grouping, groups={1: members, 2: [2]}, alpha=1.7)
    state["datasets"] = [
        {
            "run_number": dataset.run_number,
            "source_file": str(source),
            "grouping_overrides": requested,
        }
    ]
    window.restore_project_state(state, str(tmp_path / "saved.asymp"))
    restored = window._data_browser.get_dataset(dataset.run_number)
    assert restored is not None
    assert restored.run.grouping["groups"][1] == members
    assert restored.run.grouping["alpha"] == 1.7
    assert restored.metadata["correction_error"]
    assert len(restored.run.histograms) == 2
    saved = window.collect_project_state()["datasets"][0]["grouping_overrides"]
    assert saved["groups"][1] == members


def test_payload_matches_preset_requires_migrated_detector_t0_pair_entries(
    qapp,
) -> None:
    """Project migration preserves preset membership without runtime pair decoding."""
    from asymmetry.core.instrument import get_instrument_layout
    from asymmetry.gui.windows.grouping.profile_bridge import payload_matches_preset

    layout = get_instrument_layout("GPS")
    preset = layout.presets["Longitudinal"]
    payload = {
        "groups": {1: [[1, 100]], 2: [[2, 100]]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": preset.forward_group,
        "backward_group": preset.backward_group,
    }
    from asymmetry.core.project.schema import migrate_to_current

    assert not payload_matches_preset(payload, layout, "Longitudinal")
    payload = migrate_to_current(
        {"schema_version": 29, "datasets": [{"grouping_overrides": payload}]}
    )["datasets"][0]["grouping_overrides"]
    assert payload_matches_preset(payload, layout, "Longitudinal")


@pytest.mark.parametrize("excluded", [[2, 2], [True], [1.5], ["1"]])
def test_structured_exclusions_remain_invalid_until_explicitly_edited(qapp, monkeypatch, excluded):
    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.dialog.QMessageBox.warning", lambda *args: None
    )
    dataset = _histogram_dataset(grouping_extra={"excluded_detectors": excluded})
    dialog = GroupingDialog([dataset])
    try:
        dialog._preview_pane.flush()
        _wait_until(lambda: not dialog._preview_pane.corrections_complete)
        assert dialog._current_excluded_detectors() == excluded
        assert not dialog._apply_btn.isEnabled()
        dialog._on_find_t0()
        dialog._exclude_edit.setText("")
        dialog._exclude_edit.setModified(True)
        dialog._refresh_preview()
        dialog._preview_pane.flush()
        _wait_until(lambda: dialog._preview_pane.corrections_complete)
        assert dialog._apply_btn.isEnabled()
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_invalid_members_with_timing_and_legacy_manual_policy_remain_editable(qapp):
    from asymmetry.core.project.profiles import (
        GroupingProfile,
        heal_t0_policies,
        profile_fingerprint_for_run,
    )

    dataset = _histogram_dataset(
        grouping_extra={"groups": {1: [0], 2: [2]}, "detector_t0_bins": [0, 0]}
    )
    profile = GroupingProfile.from_dict(
        {
            "name": "Legacy",
            "fingerprint": profile_fingerprint_for_run(dataset.run).to_dict(),
            "groups": {1: [0], 2: [2]},
            "t0_policy": {"mode": "manual", "value": 3},
        }
    )
    heal_t0_policies([profile], {dataset.run_number: dataset.run})
    assert profile.t0_policy.legacy_value == 3
    dialog = GroupingDialog([dataset], profiles=[profile])
    try:
        assert dialog._t0_spin.value() == 3
        dialog._preview_pane.flush()
        _wait_until(lambda: not dialog._preview_pane.corrections_complete)
    finally:
        dialog._clear_dirty()
        dialog.close()


def test_invalid_grouping_keeps_the_all_detector_t0_base(qapp):
    from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint, T0Policy

    dataset = _histogram_dataset(forward=np.full(30, 20.0), backward=np.full(30, 10.0))
    dataset.run.histograms[0].t0_bin = 1
    dataset.run.histograms[1].t0_bin = 9
    profile = GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("TESTINST", 2),
        groups={1: [0], 2: [2]},
        t0_policy=T0Policy(mode="manual", offset_bins=2),
    )
    dialog = GroupingDialog([dataset], profiles=[profile])
    try:
        assert dialog._file_common_t0_for_preview_run() == 9
        assert dialog._common_t0_for_preview_run([3, 12]) == 12
        assert dialog._current_t0_policy().offset_bins == 2
    finally:
        dialog._clear_dirty()
        dialog.close()
