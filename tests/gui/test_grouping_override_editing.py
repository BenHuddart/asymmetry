"""Tests for the unified selection-driven editing model in the grouping window.

The scope-panel selection is the selector: the run selected there is previewed
and edited, and the editing target follows the run's status. Selecting an
inheriting run edits the profile draft; selecting an overridden run edits that
run's own override draft (seeded once from its stored payload). Override drafts
accumulate — switching selection never prompts — and Apply commits everything
dirty: the profile plus each edited override. The only guard is closing the
window with uncommitted changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset(run_number: int, *, instrument: str = "MuSR", alpha: float = 1.0) -> MuonDataset:
    h1 = Histogram(counts=np.full(4, 100.0), bin_width=0.01)
    h2 = Histogram(counts=np.full(4, 50.0), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": f"{instrument} {run_number}"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": alpha,
            "first_good_bin": 0,
            "last_good_bin": 3,
            "instrument": instrument,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dialog(overridden: list[int], *, selected: int | None = None) -> GroupingDialog:
    ds_a = _dataset(1, alpha=1.0)
    ds_b = _dataset(2, alpha=2.0)
    return GroupingDialog(
        [ds_a, ds_b],
        overridden_run_numbers=overridden,
        selected_run_number=selected,
    )


def _select_run(dialog: GroupingDialog, run_number: int) -> None:
    """Select *run_number* in the scope panel (the window's selector)."""
    dialog._scope_panel.set_current_run(int(run_number))


def test_selecting_overridden_run_switches_editing_target(qapp: QApplication) -> None:
    """Selecting an overridden run makes it the editing target (no profile edit)."""
    dialog = _dialog(overridden=[2], selected=1)
    assert dialog._editing_target() == "profile"

    _select_run(dialog, 2)

    assert dialog._editing_target() == 2
    # The editing-target strip reflects the override (warning-tinted text).
    assert "override for run 2" in dialog._editing_strip.text()
    assert "this run only" in dialog._editing_strip.text()
    # No modal state: the profile + instrument switchers stay enabled.
    assert dialog._profile_combo.isEnabled() is True
    assert dialog._instrument_combo.isEnabled() is True


def test_override_form_seeds_from_run_grouping(qapp: QApplication) -> None:
    """The override form seeds from the run's own grouping payload (its alpha)."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_run(dialog, 2)
    # Run 2 was built with alpha=2.0; the form reflects the override, not the profile.
    assert dialog._alpha_spin.value() == pytest.approx(2.0)


def test_edits_do_not_leak_into_profile_draft(qapp: QApplication) -> None:
    """Editing an overridden run leaves the profile draft's settings untouched."""
    dialog = _dialog(overridden=[2], selected=1)
    alpha_before = dialog._draft.alpha_policy.value

    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)
    dialog._sync_draft_from_form()

    # The override edit (alpha=3.5) went to the override draft, not the profile.
    assert dialog._override_drafts[2]["alpha"] == pytest.approx(3.5)
    assert dialog._draft.alpha_policy.value == pytest.approx(alpha_before)
    assert dialog._draft.alpha_policy.value != pytest.approx(3.5)


def test_apply_writes_only_the_overridden_run(qapp: QApplication) -> None:
    """Apply returns override_edits for the edited run and no broadcast to it."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)
    dialog._on_apply()

    grouping_result = dialog.get_grouping_result()
    profile_result = dialog.get_profile_result()

    # Only run 1 inherits; run 2 (overridden) is not broadcast to.
    assert grouping_result["run_numbers"] == [1]
    assert set(profile_result["override_edits"].keys()) == {2}
    assert profile_result["override_edits"][2]["alpha"] == pytest.approx(3.5)
    assert 1 not in profile_result["override_edits"]


def test_switching_selection_never_prompts_and_accumulates(qapp: QApplication) -> None:
    """Switching selection keeps each target's draft — no prompt, edits accumulate."""
    dialog = _dialog(overridden=[1, 2], selected=1)

    _select_run(dialog, 1)
    dialog._alpha_spin.setValue(1.25)
    # Switching to run 2 must not prompt (monkeypatch would blow up if it did).
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(2.75)
    # Back to run 1: its in-progress override draft is restored, not the file.
    _select_run(dialog, 1)
    assert dialog._alpha_spin.value() == pytest.approx(1.25)
    _select_run(dialog, 2)
    assert dialog._alpha_spin.value() == pytest.approx(2.75)


def test_apply_commits_all_dirty_overrides(qapp: QApplication) -> None:
    """Apply commits every dirty override alongside the profile in one pass."""
    dialog = _dialog(overridden=[1, 2], selected=1)
    _select_run(dialog, 1)
    dialog._alpha_spin.setValue(1.25)
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(2.75)
    dialog._on_apply()

    edits = dialog.get_profile_result()["override_edits"]
    assert set(edits.keys()) == {1, 2}
    assert edits[1]["alpha"] == pytest.approx(1.25)
    assert edits[2]["alpha"] == pytest.approx(2.75)


def test_apply_label_shows_override_blast_radius(qapp: QApplication) -> None:
    """The Apply button names the pending override count when overrides are dirty."""
    dialog = _dialog(overridden=[1, 2], selected=1)
    assert dialog._apply_btn.text() == "Apply"
    _select_run(dialog, 1)
    dialog._alpha_spin.setValue(1.25)
    assert dialog._apply_btn.text() == "Apply (profile + 1 override)"
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(2.75)
    assert dialog._apply_btn.text() == "Apply (profile + 2 overrides)"


def test_editing_strip_accent_for_inheriting_run(qapp: QApplication) -> None:
    """Selecting an inheriting run shows the accent profile strip."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_run(dialog, 1)
    assert dialog._editing_target() == "profile"
    assert "Editing profile" in dialog._editing_strip.text()
    assert "applies to" in dialog._editing_strip.text()


def test_close_guard_lists_profile_and_overrides(qapp: QApplication, monkeypatch) -> None:
    """The close guard names the profile and every dirty override that would be lost."""
    dialog = _dialog(overridden=[2], selected=1)
    # Dirty the profile (inheriting run) and an override.
    _select_run(dialog, 1)
    dialog._alpha_spin.setValue(1.9)
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)

    captured = {}

    def _question(_parent, _title, text, *_a, **_k):
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    dialog.reject()  # consults the guard; cancel keeps the window open
    assert "profile" in captured["text"]
    assert "run 2" in captured["text"] or "runs 2" in captured["text"]


def test_close_guard_clean_when_nothing_dirty(qapp: QApplication) -> None:
    """No prompt when neither the profile nor any override has edits."""
    dialog = _dialog(overridden=[2], selected=1)
    # Guard returns True immediately without a QMessageBox.
    assert dialog._guard_discard() is True


def test_release_selected_run_flips_to_fresh_override_draft(qapp: QApplication) -> None:
    """Releasing the selected run flips its target to an override seeded from effect."""
    dialog = _dialog(overridden=[], selected=1)
    _select_run(dialog, 1)
    assert dialog._editing_target() == "profile"

    # Release run 1 through the scope panel, then fire the change signal.
    dialog._scope_panel.set_released(1, True)
    dialog._on_scope_changed()

    assert dialog._editing_target() == 1
    # The override draft is seeded from run 1's current effective settings.
    assert dialog._alpha_spin.value() == pytest.approx(1.0)


def test_reattach_dirty_override_confirms_then_discards(qapp: QApplication, monkeypatch) -> None:
    """Reattaching a run with a dirty override draft confirms, then drops it."""
    dialog = _dialog(overridden=[2], selected=2)
    _select_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)
    assert 2 in dialog._override_dirty_runs

    # Cancel the discard: the reattach is undone, run 2 stays overridden + dirty.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    dialog._scope_panel.set_released(2, False)
    dialog._on_scope_changed()
    assert 2 in dialog._scope_panel.released_run_numbers()
    assert 2 in dialog._override_dirty_runs

    # Accept the discard: the override draft is dropped, run 2 inherits again.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    dialog._scope_panel.set_released(2, False)
    dialog._on_scope_changed()
    assert 2 not in dialog._scope_panel.released_run_numbers()
    assert 2 not in dialog._override_drafts


def test_open_directly_on_overridden_run_edits_that_run(qapp: QApplication) -> None:
    """Opening on an already-overridden run edits that run's override from the start."""
    dialog = _dialog(overridden=[2], selected=2)
    assert dialog._editing_target() == 2
    assert dialog._alpha_spin.value() == pytest.approx(2.0)


def test_profile_change_on_overridden_run_selects_inheriting_first(qapp: QApplication) -> None:
    """Changing the profile while an overridden run is selected first picks inheriting."""
    dialog = _dialog(overridden=[2], selected=2)
    assert dialog._editing_target() == 2

    # The rule (invoked by the profile/instrument combos) switches selection to
    # the inheriting run 1 first, so a profile edit has a valid target.
    assert dialog._select_inheriting_run_before_profile_change() is True
    assert dialog._editing_target() == "profile"
    assert dialog._current_run == 1


def test_profile_change_blocked_when_all_released(qapp: QApplication) -> None:
    """With every run released there is no inheriting target, so the rule refuses."""
    dialog = _dialog(overridden=[1, 2], selected=2)
    assert dialog._select_inheriting_run_before_profile_change() is False


# --------------------------------------------------------------------------- #
# Multi-profile assignment model (schema v17, M2)
# --------------------------------------------------------------------------- #

from asymmetry.core.project.profiles import (  # noqa: E402
    profile_fingerprint_for_run,
    profile_from_payload,
)


def _two_profile_dialog(
    assigned: dict[int, str] | None = None,
    *,
    overridden: list[int] | None = None,
    selected: int | None = 1,
) -> GroupingDialog:
    """A dialog over two runs with stored profiles 'Sample A' (default) + 'Sample B'."""
    ds_a = _dataset(1, alpha=1.0)
    ds_b = _dataset(2, alpha=2.0)
    fingerprint = profile_fingerprint_for_run(ds_a.run)
    prof_a = profile_from_payload(dict(ds_a.run.grouping), "Sample A", fingerprint, active=True)
    prof_b = profile_from_payload(dict(ds_b.run.grouping), "Sample B", fingerprint, active=False)
    return GroupingDialog(
        [ds_a, ds_b],
        profiles=[prof_a, prof_b],
        overridden_run_numbers=list(overridden or []),
        assigned_profiles=dict(assigned or {}),
        selected_run_number=selected,
    )


def test_runs_without_assignment_follow_the_default_profile(qapp: QApplication) -> None:
    """Unassigned runs resolve to the default profile; Apply covers them all."""
    dialog = _two_profile_dialog()
    assert dialog._draft_name == "Sample A"  # editor opens on the default
    result = dialog.get_grouping_result()
    assert set(result["run_numbers"]) == {1, 2}
    profile_result = dialog.get_profile_result()
    assert profile_result["assignments"] == {1: "Sample A", 2: "Sample A"}
    assert profile_result["newly_assigned"] == {}
    assert profile_result["default_profile"] == "Sample A"


def test_run_assigned_elsewhere_is_excluded_from_profile_apply(qapp: QApplication) -> None:
    """A run assigned to another profile is not an Apply target of this one."""
    dialog = _two_profile_dialog({2: "Sample B"})
    result = dialog.get_grouping_result()
    assert set(result["run_numbers"]) == {1}
    profile_result = dialog.get_profile_result()
    assert profile_result["assignments"] == {1: "Sample A", 2: "Sample B"}
    # The editing strip counts only the followers of the edited profile.
    assert "applies to 1 run" in dialog._editing_strip.text()


def test_stale_assignment_falls_back_to_default(qapp: QApplication) -> None:
    """An assignment naming a missing profile resolves to the default."""
    dialog = _two_profile_dialog({2: "Ghost"})
    profile_result = dialog.get_profile_result()
    assert profile_result["assignments"][2] == "Sample A"


def test_assign_runs_moves_between_profiles(qapp: QApplication) -> None:
    """Assigning a run re-scopes Apply and reports the change."""
    dialog = _two_profile_dialog()
    dialog._scope_panel.assign_runs([2], "Sample B")

    assert set(dialog.get_grouping_result()["run_numbers"]) == {1}
    profile_result = dialog.get_profile_result()
    assert profile_result["newly_assigned"] == {2: "Sample B"}
    assert profile_result["assignments"][2] == "Sample B"


def test_assignments_survive_switching_the_edited_profile(qapp: QApplication) -> None:
    """Assign to B, then switch the combo to edit B: the assignment holds."""
    dialog = _two_profile_dialog()
    dialog._scope_panel.assign_runs([2], "Sample B")
    dialog._draft_dirty = False  # the scope change armed the guard; disarm to switch

    assert dialog._load_stored_profile_into_draft("Sample B")
    assert dialog._draft_name == "Sample B"
    # The session assignment survived the repopulate; Apply now targets run 2.
    assert set(dialog.get_grouping_result()["run_numbers"]) == {2}
    assert dialog.get_profile_result()["newly_assigned"] == {2: "Sample B"}


def test_preview_cue_names_the_governing_profile(qapp: QApplication) -> None:
    """Previewing a run that follows another profile is cued in the strip."""
    dialog = _two_profile_dialog({2: "Sample B"})
    dialog._scope_panel.set_current_run(2)
    assert "preview run follows Sample B" in dialog._editing_strip.text()


def test_default_checkbox_moves_the_default(qapp: QApplication) -> None:
    """Checking 'Default for new runs' on another profile moves the star."""
    dialog = _two_profile_dialog()
    assert dialog._default_check.isChecked()  # editing the default profile
    assert not dialog._default_check.isEnabled()  # locked while default

    dialog._load_stored_profile_into_draft("Sample B")
    assert not dialog._default_check.isChecked()
    assert dialog._default_check.isEnabled()

    dialog._default_check.setChecked(True)
    profile_result = dialog.get_profile_result()
    assert profile_result["default_profile"] == "Sample B"
    # The combo star follows the pending default.
    labels = [dialog._profile_combo.itemText(i) for i in range(dialog._profile_combo.count())]
    assert "★ Sample B" in labels


def test_rename_is_first_class_and_moves_references(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renaming a stored profile records renamed_from and rewrites the maps."""
    from PySide6.QtWidgets import QInputDialog

    dialog = _two_profile_dialog()
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Sample A2", True)))
    dialog._on_rename_profile()

    assert dialog._draft_name == "Sample A2"
    profile_result = dialog.get_profile_result()
    assert profile_result["renamed_from"] == "Sample A"
    assert profile_result["profile"].name == "Sample A2"
    # Every run that followed the old name now names the new one.
    assert profile_result["assignments"] == {1: "Sample A2", 2: "Sample A2"}
    assert profile_result["newly_assigned"] == {}  # a rename is not a reassignment
    assert profile_result["default_profile"] == "Sample A2"


def test_rename_rejects_an_existing_name(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    dialog = _two_profile_dialog()
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Sample B", True)))
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    dialog._on_rename_profile()

    assert warnings and "already exists" in warnings[0]
    assert dialog._draft_name == "Sample A"
    assert dialog._renamed_from is None


def test_delete_profile_reassigns_runs_and_switches_draft(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the edited profile moves its runs and the default to the target."""
    from PySide6.QtWidgets import QInputDialog

    dialog = _two_profile_dialog()
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("Sample B", True)))
    dialog._on_delete_profile()

    assert dialog._draft_name == "Sample B"
    profile_result = dialog.get_profile_result()
    assert profile_result["deleted_profiles"] == ["Sample A"]
    assert profile_result["assignments"] == {1: "Sample B", 2: "Sample B"}
    assert profile_result["newly_assigned"] == {1: "Sample B", 2: "Sample B"}
    assert profile_result["default_profile"] == "Sample B"
    # The deleted profile is gone from the selector.
    labels = [dialog._profile_combo.itemText(i) for i in range(dialog._profile_combo.count())]
    assert all("Sample A" != label.removeprefix("★ ") for label in labels)


def test_delete_refuses_the_last_profile(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    ds_a = _dataset(1, alpha=1.0)
    fingerprint = profile_fingerprint_for_run(ds_a.run)
    prof_a = profile_from_payload(dict(ds_a.run.grouping), "Only", fingerprint, active=True)
    dialog = GroupingDialog([ds_a], profiles=[prof_a], selected_run_number=1)

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    dialog._on_delete_profile()

    assert warnings and "only profile" in warnings[0]
    assert dialog.get_profile_result()["deleted_profiles"] == []


def test_released_run_chip_names_its_base_profile(qapp: QApplication) -> None:
    """A released run based on another profile shows that base in its chip."""
    dialog = _two_profile_dialog({2: "Sample B"}, overridden=[2])
    panel = dialog._scope_panel
    assert panel.runs_following("Sample A") == {1}
    # Chip text for run 2 names the base profile it would reattach to.
    texts = [panel._list.item(i).text() for i in range(panel._list.count())]
    assert any("override · Sample B" in t for t in texts)
