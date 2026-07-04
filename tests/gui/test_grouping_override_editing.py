"""Tests for override-editing mode in the grouping window (M3).

An overridden run's grouping was previously uneditable — it could only be
created (Release) or dropped (Reattach). This adds an explicit override-editing
mode: selecting an overridden run as the preview run seeds the form from that
run's own grouping, keeps edits off the profile draft, and applies to that run
alone.
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


def _select_preview_run(dialog: GroupingDialog, run_number: int) -> None:
    idx = dialog._reference_combo.findData(int(run_number))
    dialog._reference_combo.setCurrentIndex(idx)


def test_enter_override_mode_via_preview_switch(qapp: QApplication) -> None:
    """Selecting an overridden run as preview enters override-editing mode."""
    dialog = _dialog(overridden=[2], selected=1)
    assert dialog._override_mode is False

    _select_preview_run(dialog, 2)

    assert dialog._override_mode is True
    assert dialog._override_run_number == 2
    assert dialog._override_banner.isHidden() is False
    assert "run 2" in dialog._override_banner.text()
    # Profile + instrument switchers are disabled; scope panel stays present.
    assert dialog._profile_combo.isEnabled() is False
    assert dialog._instrument_combo.isEnabled() is False


def test_override_form_seeds_from_run_grouping(qapp: QApplication) -> None:
    """The override form seeds from the run's own grouping payload (its alpha)."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    # Run 2 was built with alpha=2.0; the form reflects the override, not the profile.
    assert dialog._alpha_spin.value() == pytest.approx(2.0)


def test_edits_do_not_leak_into_profile_draft(qapp: QApplication) -> None:
    """Editing in override mode leaves the profile draft untouched."""
    dialog = _dialog(overridden=[2], selected=1)
    draft_before = dialog._draft.to_dict()

    _select_preview_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)
    dialog._sync_draft_from_form()

    # The profile draft is unchanged; the override payload captured the edit.
    assert dialog._draft.to_dict() == draft_before
    assert dialog._override_payload is not None
    assert dialog._override_payload["alpha"] == pytest.approx(3.5)


def test_apply_writes_only_the_overridden_run(qapp: QApplication) -> None:
    """Apply in override mode returns override_edits for that run and no broadcast."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)
    dialog._on_apply()

    grouping_result = dialog.get_grouping_result()
    profile_result = dialog.get_profile_result()

    # No inheriting run is broadcast to; the override is carried separately.
    assert grouping_result["run_numbers"] == []
    assert set(profile_result["override_edits"].keys()) == {2}
    assert profile_result["override_edits"][2]["alpha"] == pytest.approx(3.5)
    # Run 1 (inheriting) is untouched: not present in any apply target.
    assert 1 not in profile_result["override_edits"]


def test_exit_mode_via_switch_back_to_inheriting_run(qapp: QApplication) -> None:
    """Switching preview back to an inheriting run exits override mode."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    assert dialog._override_mode is True

    _select_preview_run(dialog, 1)
    assert dialog._override_mode is False
    assert dialog._override_banner.isHidden() is True
    assert dialog._profile_combo.isEnabled() is True


def test_dirty_override_guards_on_exit(qapp: QApplication, monkeypatch) -> None:
    """A dirty override prompts before a preview switch takes it out of the mode."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)  # marks the override draft dirty
    assert dialog._override_draft_dirty is True

    # Cancel the discard prompt: the switch is aborted, still in override mode.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    _select_preview_run(dialog, 1)
    assert dialog._override_mode is True
    assert int(dialog._reference_combo.currentData()) == 2

    # Accept the discard: the switch proceeds and leaves the mode.
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
    )
    _select_preview_run(dialog, 1)
    assert dialog._override_mode is False


def test_dirty_override_guards_on_close(qapp: QApplication, monkeypatch) -> None:
    """The close guard covers a dirty override draft."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    dialog._alpha_spin.setValue(3.5)

    calls = {"n": 0}

    def _question(*_a, **_k):
        calls["n"] += 1
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    # reject() consults the guard; cancelling keeps the override-editing state.
    dialog.reject()
    assert calls["n"] == 1
    assert dialog._override_mode is True
    assert dialog._override_draft_dirty is True


def test_enter_override_mode_via_scope_edit(qapp: QApplication) -> None:
    """The scope panel's Edit… affordance selects the run and enters the mode."""
    dialog = _dialog(overridden=[2], selected=1)
    assert dialog._override_mode is False

    # Drive the scope panel's Edit… request directly (as clicking would).
    dialog._scope_panel.edit_requested.emit(2)

    assert dialog._override_mode is True
    assert dialog._override_run_number == 2
    assert int(dialog._reference_combo.currentData()) == 2


def test_reattach_from_within_override_mode_exits(qapp: QApplication) -> None:
    """Reattaching the run being override-edited drops the override and exits."""
    dialog = _dialog(overridden=[2], selected=1)
    _select_preview_run(dialog, 2)
    assert dialog._override_mode is True

    # Reattach run 2 through the scope panel state, then fire the change signal.
    dialog._scope_panel._released[2] = False
    dialog._scope_panel._rebuild()
    dialog._on_scope_changed()

    assert dialog._override_mode is False
    assert 2 not in dialog._scope_panel.released_run_numbers()


def test_open_directly_on_overridden_run_enters_mode(qapp: QApplication) -> None:
    """Opening the dialog on an already-overridden run enters override mode."""
    dialog = _dialog(overridden=[2], selected=2)
    assert dialog._override_mode is True
    assert dialog._override_run_number == 2
