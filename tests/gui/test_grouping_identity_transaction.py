"""Identity confirmation and explicit detector edits remain one cancellable UI transaction."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import InstrumentSelection, RecordedInstrumentChoice
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog
from asymmetry.gui.windows.grouping_dialog import GroupingDialog
from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

pytestmark = pytest.mark.gui


@pytest.fixture
def editor(qapp):
    grouping = {
        "groups": {1: [1, 3], 2: [2, 4]},
        "group_names": {1: "Saved positive", 2: "Saved negative"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 0.93,
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": 7,
    }
    run = Run(
        run_number=17,
        histograms=[Histogram(np.full(8, 100.0), 0.01, 0, 0, 7) for _ in range(8)],
        metadata={"facility": "PSI", "instrument": "FLAME"},
        grouping=deepcopy(grouping),
        instrument_choice=RecordedInstrumentChoice(
            InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
        ),
    )
    dataset = MuonDataset(
        np.arange(8) * 0.01, np.zeros(8), np.ones(8), metadata={"run_number": 17}, run=run
    )
    profile = GroupingProfile(
        "Saved GPS",
        ProfileFingerprint("psi.gps", 8, "bin_combined"),
        groups=deepcopy(grouping["groups"]),
        group_names=deepcopy(grouping["group_names"]),
    )
    dialog = GroupingDialog([dataset], profiles=[profile], assigned_profiles={17: profile.name})
    yield dialog, run, profile
    dialog.done(QDialog.Rejected)


@pytest.mark.parametrize("apply_preset", [False, True])
def test_layout_choice_preserves_groups_and_explicit_preset_is_a_separate_authored_edit(
    editor, monkeypatch, apply_preset
):
    dialog, run, profile = editor
    before = deepcopy((run.grouping, run.instrument_choice, profile.to_dict()))

    def layout_exec(layout):
        layout._instrument_combo.setCurrentIndex(layout._instrument_combo.findData("FLAME"))
        assert layout.get_result()["groups"] == before[0]["groups"]
        if apply_preset:
            layout._preset_combo.setCurrentText("Longitudinal")
            layout._on_apply_preset()
        layout.accept()
        return QDialog.Accepted

    monkeypatch.setattr(DetectorLayoutDialog, "exec", layout_exec)
    monkeypatch.setattr(InstrumentIdentityDialog, "exec", lambda self: QDialog.Accepted)
    dialog._on_detector_layout()
    result = dialog.get_profile_result()
    assert result["instrument_choice"].selection.profile_key == "psi.flame"
    assert 17 in result["newly_released"]
    assert 17 not in dialog.get_grouping_result()["run_numbers"]
    assert result["profile"].groups == before[2]["groups"]
    request = result["override_edits"][17]
    expected = {1: [1], 2: [2]} if apply_preset else before[0]["groups"]
    assert request["groups"] == expected
    if apply_preset:
        assert (request["forward_group"], request["backward_group"]) == (2, 1)
    assert (run.grouping, run.instrument_choice, profile.to_dict()) == before


def test_cancel_identity_confirmation_discards_layout_and_preset_edits(editor, monkeypatch):
    dialog, run, profile = editor
    before = deepcopy((dialog._current_grouping_payload(), run.grouping, profile.to_dict()))

    def layout_exec(layout):
        layout._instrument_combo.setCurrentIndex(layout._instrument_combo.findData("FLAME"))
        layout._preset_combo.setCurrentText("Longitudinal")
        layout._on_apply_preset()
        return QDialog.Accepted

    monkeypatch.setattr(DetectorLayoutDialog, "exec", layout_exec)
    monkeypatch.setattr(InstrumentIdentityDialog, "exec", lambda self: QDialog.Rejected)
    dialog._on_detector_layout()
    assert dialog._instrument_edit is None
    assert (dialog._current_grouping_payload(), run.grouping, profile.to_dict()) == before
