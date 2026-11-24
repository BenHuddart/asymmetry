"""Identity confirmation and explicit detector edits remain one cancellable UI transaction."""

from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QMessageBox

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
    accept_instrument_choice,
)
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


def test_confirmed_identity_with_incompatible_map_keeps_generic_drawing_and_no_presets(
    editor, monkeypatch
):
    dialog, run, _ = editor
    before = deepcopy(run.grouping)

    def layout_exec(layout):
        layout._instrument_combo.setCurrentIndex(layout._instrument_combo.findData("HiFi"))
        return QDialog.Accepted

    monkeypatch.setattr(DetectorLayoutDialog, "exec", layout_exec)
    monkeypatch.setattr(InstrumentIdentityDialog, "exec", lambda self: QDialog.Accepted)
    dialog._on_detector_layout()
    assert dialog._instrument_edit.choice.selection.profile_key == "isis.hifi"
    assert dialog._current_instrument_layout().name == "Generic"
    assert dialog._preset_combo.count() == 0
    assert dialog._current_grouping_payload()["groups"] == before["groups"]
    assert run.grouping == before


@pytest.mark.parametrize("entry", ["layout", "identity"])
@pytest.mark.parametrize("finish", ["cancel", "apply"])
def test_pending_identity_release_labels_and_edits_only_the_selected_run(
    editor, monkeypatch, finish, entry
):
    dialog, run, profile = editor
    other = deepcopy(dialog._datasets[0])
    other.run.run_number = 18
    other.run.metadata["run_number"] = other.metadata["run_number"] = 18
    other.run.instrument_choice = accept_instrument_choice(
        other.run, other.run.instrument_choice.selection
    )
    dialog._datasets.append(other)
    dialog._assigned_profiles[18] = profile.name
    dialog._refresh_scope_panel()
    before = deepcopy((run.grouping, other.run.grouping, profile.to_dict()))
    dialog._sync_draft_from_form()
    shared_draft = deepcopy(dialog._draft.to_dict())

    def layout_exec(layout):
        layout._instrument_combo.setCurrentIndex(layout._instrument_combo.findData("HiFi"))
        return QDialog.Accepted

    def identity_exec(identity):
        if entry == "identity":
            index = next(
                i
                for i in range(identity._choices.count())
                if isinstance(identity._choices.itemData(i), InstrumentSelection)
                and identity._choices.itemData(i).profile_key == "isis.hifi"
            )
            identity._choices.setCurrentIndex(index)
        return QDialog.Accepted

    monkeypatch.setattr(DetectorLayoutDialog, "exec", layout_exec)
    monkeypatch.setattr(InstrumentIdentityDialog, "exec", identity_exec)
    if entry == "identity":
        dialog._on_instrument_identity()
    else:
        dialog._on_detector_layout()
    assert dialog._editing_target() == 17
    assert dialog._editing_strip.text() == "Editing override for run 17 — this run only"
    assert dialog._scope_panel.released_run_numbers() == {17}
    assert dialog._scope_panel.runs_following(profile.name) == {18}
    dialog._alpha_spin.setValue(0.81)
    result = dialog.get_profile_result()
    assert result["override_edits"][17]["alpha"] == pytest.approx(0.81)
    assert result["newly_released"] == {17}
    assert result["profile"].to_dict() == shared_draft
    assert dialog.get_grouping_result()["run_numbers"] == [18]
    assert (run.grouping, other.run.grouping, profile.to_dict()) == before
    if finish == "apply":
        dialog.show()
        for _ in range(500):
            if dialog._apply_btn.isEnabled():
                break
            QTest.qWait(10)
        assert dialog._apply_btn.isEnabled(), dialog._preview_pane._status.text()
        dialog._apply_btn.click()
        assert dialog.result() == QDialog.Accepted
        assert dialog.get_profile_result()["override_edits"][17]["alpha"] == pytest.approx(0.81)
    else:
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Discard)
        dialog.reject()
        assert dialog.result() == QDialog.Rejected
    assert run.instrument_choice.selection.profile_key == "psi.gps"
    assert (run.grouping, other.run.grouping, profile.to_dict()) == before


@pytest.mark.parametrize("accepted", [False, True])
def test_identity_button_stages_only_the_selected_run(editor, monkeypatch, accepted):
    from PySide6.QtWidgets import QPushButton

    dialog, run, profile = editor
    before = deepcopy((run.grouping, run.instrument_choice, profile.to_dict()))
    dialog._sync_draft_from_form()
    request = deepcopy(dialog._current_grouping_payload())
    opened = []

    def review(identity):
        opened.append(identity)
        index = next(
            i
            for i in range(identity._choices.count())
            if isinstance(identity._choices.itemData(i), InstrumentSelection)
            and identity._choices.itemData(i).profile_key == "psi.flame"
        )
        identity._choices.setCurrentIndex(index)
        return QDialog.Accepted if accepted else QDialog.Rejected

    monkeypatch.setattr(InstrumentIdentityDialog, "exec", review)
    button = next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Instrument identity…"
    )
    button.click()
    assert len(opened) == 1
    if accepted:
        result = dialog.get_profile_result()
        assert result["instrument_choice"].selection.profile_key == "psi.flame"
        assert result["instrument_choice_run_number"] == 17
        assert result["override_edits"][17]["groups"] == request["groups"]
        assert result["override_edits"][17]["alpha"] == request["alpha"]
        assert result["newly_released"] == {17}
        assert dialog._editing_strip.text() == "Editing override for run 17 — this run only"
    else:
        assert dialog._instrument_edit is None
        assert dialog._current_grouping_payload() == request
    assert (run.grouping, run.instrument_choice, profile.to_dict()) == before
