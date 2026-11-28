"""Atomic detector-layout edits and their existing calibration authorities."""

from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QMessageBox

from asymmetry.core.instrument import derive_projection_pairs
from asymmetry.core.instruments.resolution import RecordedInstrumentChoice, choice_for_layout
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BetaPolicy,
    GroupingProfile,
    profile_fingerprint_for_run,
    profile_grouping_request,
)
from asymmetry.core.transform.grouping import format_detector_list
from asymmetry.gui.windows.grouping.channel_entries import shift_channel_entries
from asymmetry.gui.windows.grouping.profile_bridge import payload_matches_preset


@dataclass(frozen=True)
class InstrumentEdit:
    """An accepted identity draft and the profile settings authored before that change."""

    choice: RecordedInstrumentChoice
    profile: GroupingProfile
    run_number: int


def _pair_members(groups, pair, excluded):
    """Ordered effective acquired IDs, independent of the UI's group labels."""
    return tuple(frozenset(groups.get(gid, ())) - excluded for gid in pair)


def prepare_readout_change(dialog, payload):
    """Stage explicit readouts without silently transferring a fitted balance."""
    edited = {label: float(spin.value()) for label, spin in dialog._vector_alpha_spins.items()}
    instrument = dialog._current_instrument_layout()
    target = dialog._editing_target()
    owner = dialog._draft if target == "profile" else dialog._override_profiles[int(target)]
    if not instrument.readout_sets:
        if "readout_set" in payload:
            owner.extra["readout_set"] = payload["readout_set"]
        elif payload.get("grouping_preset"):
            owner.extra.pop("readout_set", None)
        return edited
    before = {gid: shift_channel_entries(ids, 1) for gid, ids in dialog._groups.items()}
    after = payload["groups"]
    excluded_before = set(dialog._current_excluded_detectors() or ())
    excluded_after = set(payload.get("excluded_detectors", excluded_before))
    old_pairs = dialog._vector_axis_pairs
    new_pairs = derive_projection_pairs(
        after, payload.get("group_names"), payload.get("projections")
    )
    changed = {
        label
        for label in old_pairs
        if label not in new_pairs
        or _pair_members(before, old_pairs[label], excluded_before)
        != _pair_members(after, new_pairs[label], excluded_after)
    }
    old_pair = (
        int(dialog._forward_combo.currentData() or 1),
        int(dialog._backward_combo.currentData() or 2),
    )
    new_pair = (
        int(payload.get("forward_group", old_pair[0])),
        int(payload.get("backward_group", old_pair[1])),
    )
    base_changed = _pair_members(before, old_pair, excluded_before) != _pair_members(
        after, new_pair, excluded_after
    )
    alpha_bound = (
        owner.alpha_policy.mode == "calibrated" or "single" in dialog._alpha_estimate_state
    )
    beta_bound = owner.beta_policy.mode == "calibrated" or dialog._beta_estimate_state is not None
    captured = bool(
        changed.intersection(owner.projection_alpha_policies)
        or changed.intersection(dialog._alpha_estimate_state)
        or base_changed
        and (alpha_bound or beta_bound)
    )
    preset = payload.get("grouping_preset")
    is_preset = bool(preset and payload_matches_preset(payload, instrument, preset))
    if captured and not is_preset:
        QMessageBox.warning(
            dialog,
            "Readout set",
            "Changing calibrated readouts requires recalibration. Clear the active calibration first, or apply a preset.",
        )
        return None
    for label in changed:
        owner.projection_alpha_policies.pop(label, None)
        dialog._alpha_estimate_state.pop(label, None)
        if is_preset:
            edited.pop(label, None)
    if base_changed and alpha_bound:
        owner.alpha_policy = AlphaPolicy(value=1.0)
        with QSignalBlocker(dialog._alpha_spin):
            dialog._alpha_spin.setValue(1.0)
        dialog._alpha_estimate_state.pop("single", None)
        dialog._alpha_estimate_state.pop("P_z", None)
        dialog._alpha_correction_digest = None
        dialog._set_alpha_method("")
    if base_changed and beta_bound:
        owner.beta_policy = BetaPolicy(value=1.0)
        with QSignalBlocker(dialog._beta_section):
            dialog._beta_section.set_value(1.0)
        dialog._beta_estimate_state = None
        dialog._beta_method = ""
        dialog._beta_correction_digest = None
        dialog._beta_section.set_provenance("manual")
    if "readout_set" in payload:
        owner.extra["readout_set"] = payload["readout_set"]
    else:
        owner.extra.pop("readout_set", None)
    return edited


def confirm_instrument_choice(self, selection=None) -> bool:
    """Stage a reviewed identity while preserving the selected run's authored settings."""
    from asymmetry.gui.windows.instrument_identity_dialog import InstrumentIdentityDialog

    if self._instrument_edit is not None and (
        self._instrument_edit.run_number != int(self._run.run_number)
    ):
        QMessageBox.information(
            self,
            "Pending instrument identity",
            f"An instrument identity change is pending for run {self._instrument_edit.run_number}. "
            "Apply or cancel the Grouping window before reviewing another run's identity.",
        )
        return False
    self._sync_draft_from_form()
    run = self._run
    if (
        self._instrument_edit is not None
        and int(run.run_number) == self._instrument_edit.run_number
    ):
        run = replace(run, instrument_choice=self._instrument_edit.choice)
    confirmation = InstrumentIdentityDialog(
        run,
        following_profile=None if self._run_is_overridden(int(run.run_number)) else self._draft,
        selection=selection,
        parent=self,
    )
    if confirmation.exec() != confirmation.DialogCode.Accepted:
        return False
    before = self._instrument_edit.profile if self._instrument_edit is not None else self._draft
    self._instrument_edit = InstrumentEdit(
        confirmation.get_choice(),
        GroupingProfile.from_dict(before.to_dict()),
        int(self._reference_dataset.run_number),
    )
    selected_run = replace(self._run, instrument_choice=self._instrument_edit.choice)
    if not self._fingerprint.matches(profile_fingerprint_for_run(selected_run)):
        number = self._instrument_edit.run_number
        if not self._run_is_overridden(number):
            self._override_profiles[number] = GroupingProfile.from_dict(before.to_dict())
            self._override_drafts[number] = profile_grouping_request(before, self._run)
            self._scope_panel.set_released(number, True)
    return True


def edit_instrument_identity(self) -> None:
    """Review identity as a cancellable edit of the selected grouping run."""
    if not confirm_instrument_choice(self):
        return
    self._mark_dirty()
    self._rebuild_preset_combo()
    self._refresh_editing_strip()
    self._refresh_preview()


def edit_detector_layout(self) -> None:
    """Open the interactive detector layout editor as a sub-dialog."""
    from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

    n_histo = len(self._run.histograms) if self._run is not None else 0
    grouping = self._run.grouping if self._run is not None else {}
    instrument = self._current_instrument_layout()

    forward_gid = int(grouping.get("forward_group", self._forward_combo.currentData() or 1))
    backward_gid = int(grouping.get("backward_group", self._backward_combo.currentData() or 2))

    # Convert internal 0-based indices to 1-based for the layout editor
    groups_1based = {gid: shift_channel_entries(idxs, 1) for gid, idxs in self._groups.items()}

    current_exclusion = self._current_excluded_detectors() or []
    target = self._editing_target()
    owner = self._draft if target == "profile" else self._override_profiles[int(target)]
    dlg = DetectorLayoutDialog(
        instrument=instrument,
        groups=groups_1based,
        group_names=dict(self._group_names),
        initial_preset_name=self._grouping_preset_name,
        forward_group=forward_gid,
        backward_group=backward_gid,
        excluded_detectors=current_exclusion,
        projections=self._projection_specs,
        field_direction=self._reference_field_direction(),
        n_histograms=n_histo,
        readout_set=owner.extra.get("readout_set"),
        parent=self,
    )
    if dlg.exec() != DetectorLayoutDialog.DialogCode.Accepted:
        return

    result = dlg.get_result()
    key = result["instrument"]
    if key != instrument.name:
        chosen = choice_for_layout(self._run, key)
        if not confirm_instrument_choice(self, chosen.selection):
            return
    edited_alpha = prepare_readout_change(self, result)
    if edited_alpha is None:
        return
    needed_repair = self._file_timing_needs_membership_repair()
    self._exclude_edit.setText(format_detector_list(result.get("excluded_detectors", [])))
    self._exclude_edit.setModified(True)
    result_projections = result.get("projections")
    if result_projections:
        merged: list[dict[str, Any]] = []
        for proj in result_projections:
            proj = dict(proj)
            label = str(proj.get("label"))
            if label in edited_alpha:
                proj["alpha"] = edited_alpha[label]
            merged.append(proj)
        self._projection_specs = merged
    else:
        self._projection_specs = None

    # Write back: convert 1-based IDs back to 0-based internal indices
    new_groups_0based: dict[int, list[int]] = {}
    for gid, det_ids in result["groups"].items():
        new_groups_0based[gid] = sorted(max(0, d - 1) for d in det_ids)

    # Keep any previously defined groups that are not in the result
    # (preserves groups beyond the 8 shown in the editor)
    self._groups = new_groups_0based
    previous_included = dict(self._included_groups)
    self._included_groups = {
        int(gid): bool(previous_included.get(int(gid), True)) for gid in self._groups
    }
    self._group_names = result.get("group_names", {})
    preset_name = result.get("grouping_preset")
    # Preset matching is recomputed from the authored detector groups.
    if preset_name:
        self._grouping_preset_name = str(preset_name)
    # Update forward/backward combos
    new_fwd = result.get("forward_group", forward_gid)
    new_bwd = result.get("backward_group", backward_gid)
    new_fwd, new_bwd = self._analysis_pair_for_reference(int(new_fwd), int(new_bwd))
    self._refresh_group_combo_items(forward_gid=int(new_fwd), backward_gid=int(new_bwd))

    self._populate_group_table()
    self._update_vector_mode_controls()
    self._mark_dirty()
    # The editor may have switched instrument, so the preset list too.
    self._rebuild_preset_combo()
    self._refresh_editing_strip()
    self._refresh_timing_after_membership_edit(needed_repair)
    self._refresh_preview()
