"""Tests for the ProjectionChipBar widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from asymmetry.core.instrument import PROJECTION_TINTS, TRANSVERSE_PROJECTION_TINTS
from asymmetry.gui.widgets.projection_chip_bar import ProjectionChipBar, short_projection_label

_VECTOR = [
    {"label": "P_x", "tint": PROJECTION_TINTS["P_x"]},
    {"label": "P_y", "tint": PROJECTION_TINTS["P_y"]},
    {"label": "P_z", "tint": PROJECTION_TINTS["P_z"]},
]
_TRANSVERSE = [
    {"label": label, "tint": tint} for label, tint in TRANSVERSE_PROJECTION_TINTS.items()
]


def _capture(bar: ProjectionChipBar) -> list[list[str]]:
    events: list[list[str]] = []
    bar.selection_changed.connect(lambda labels: events.append(list(labels)))
    return events


class TestProjectionChipBar:
    def test_three_projections_all_selected_in_order(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        assert not bar.isHidden()
        assert bar.selected_labels() == ["P_x", "P_y", "P_z"]

    def test_single_projection_hides_bar(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections([{"label": "P_z"}])
        assert bar.isHidden()

    def test_two_projections_show_bar(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR[:2])
        # Shown once at least two projections exist (visibility honoured even
        # without a parent show()).
        assert not bar.isHidden()

    def test_toggling_chip_emits_remaining_selection(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        events = _capture(bar)
        bar._chips["P_y"].setChecked(False)
        assert events[-1] == ["P_x", "P_z"]
        assert bar.selected_labels() == ["P_x", "P_z"]

    def test_floor_of_one_last_chip_will_not_release(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        bar.set_selected(["P_y"])
        events = _capture(bar)
        bar._chips["P_y"].setChecked(False)
        # The release is vetoed: still selected, no empty-selection event.
        assert bar.selected_labels() == ["P_y"]
        assert events == []

    def test_all_action_selects_everything(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        bar.set_selected(["P_x"])
        events = _capture(bar)
        bar._on_all_clicked()
        assert bar.selected_labels() == ["P_x", "P_y", "P_z"]
        assert events[-1] == ["P_x", "P_y", "P_z"]

    def test_all_action_disabled_when_all_selected(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        assert bar._all_btn.isEnabled() is False
        bar.set_selected(["P_x"])
        assert bar._all_btn.isEnabled() is True

    def test_set_selected_floor_when_empty(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        bar.set_selected([])
        # Empty request snaps to the first projection rather than zero.
        assert bar.selected_labels() == ["P_x"]

    def test_selection_preserved_across_rebuild(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        bar.set_selected(["P_x", "P_z"])
        bar.set_projections(_VECTOR)
        assert bar.selected_labels() == ["P_x", "P_z"]

    def test_chip_carries_its_tint(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        assert PROJECTION_TINTS["P_x"] in bar._chips["P_x"].styleSheet()

    def test_set_projections_selected_argument(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR, ["P_y"])
        assert bar.selected_labels() == ["P_y"]

    def test_unchanged_projection_set_keeps_chip_widgets(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        chip_before = bar._chips["P_x"]
        # Same label/tint set, different selection: must not tear down chips
        # (a click round-trips through here and must not delete the chip pressed).
        bar.set_projections(_VECTOR, ["P_x"])
        assert bar._chips["P_x"] is chip_before
        assert bar.selected_labels() == ["P_x"]

    def test_changed_projection_set_rebuilds_chips(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        chip_before = bar._chips["P_x"]
        bar.set_projections(_VECTOR[:2])
        assert "P_z" not in bar._chips
        assert bar._chips["P_x"] is not chip_before

    def test_bar_minimum_is_the_folded_button_not_the_chips(self, qapp):
        """The chips never set the toolbar's minimum width: the bar folds instead."""
        bar = ProjectionChipBar()
        bar.set_projections(_TRANSVERSE)
        assert bar.minimumSizeHint().width() < bar.sizeHint().width()
        many = [{"label": f"Label-{n}-Pair"} for n in range(8)]
        wide = ProjectionChipBar()
        wide.set_projections(many)
        assert wide.sizeHint().width() > bar.sizeHint().width()
        assert wide.minimumSizeHint().width() < wide.sizeHint().width() / 4

    def test_presentation_steps_from_full_to_short_to_folded(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_TRANSVERSE)
        bar.show()

        bar.resize(bar.sizeHint().width(), bar.sizeHint().height())
        qapp.processEvents()
        assert bar.presentation() == "full"
        assert bar._chips["Top-Bottom"].text() == "Top-Bottom"

        bar.resize(bar.sizeHint().width() - 1, bar.sizeHint().height())
        qapp.processEvents()
        assert bar.presentation() == "short"
        assert bar._chips["Top-Bottom"].text() == "T–B"
        assert bar._chips["Top-Bottom"].toolTip() == "Top-Bottom"

        bar.resize(bar.minimumSizeHint().width(), bar.sizeHint().height())
        qapp.processEvents()
        assert bar.presentation() == "folded"
        assert all(chip.isHidden() for chip in bar._chips.values())
        assert bar._all_btn.isHidden()
        assert not bar._fold_btn.isHidden()

    def test_folded_button_summarises_the_selection(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_TRANSVERSE)
        assert bar._fold_btn.text() == "All 3 ▾"
        bar.set_selected(["Fwd-Back"])
        assert bar._fold_btn.text() == "F–B ▾"
        assert TRANSVERSE_PROJECTION_TINTS["Fwd-Back"] in bar._fold_btn.styleSheet()
        bar.set_selected(["Fwd-Back", "Left-Right"])
        assert bar._fold_btn.text() == "2 of 3 ▾"

    def test_fold_menu_toggles_through_the_chips(self, qapp):
        """Menu entries drive the chips, so the floor of one still holds."""
        bar = ProjectionChipBar()
        bar.set_projections(_TRANSVERSE, ["Top-Bottom"])
        events = _capture(bar)
        bar._rebuild_fold_menu()
        actions = {a.text(): a for a in bar._fold_menu.actions()}

        actions["Top-Bottom"].trigger()  # the last selected projection
        assert bar.selected_labels() == ["Top-Bottom"]
        assert events == []

        actions["Left-Right"].trigger()
        assert bar.selected_labels() == ["Top-Bottom", "Left-Right"]
        assert events[-1] == ["Top-Bottom", "Left-Right"]

        bar._rebuild_fold_menu()
        actions = {a.text(): a for a in bar._fold_menu.actions()}
        actions["Show all"].trigger()
        assert bar.selected_labels() == ["Top-Bottom", "Fwd-Back", "Left-Right"]

    def test_long_unhyphenated_label_does_not_widen_the_folded_button(self, qapp):
        """A custom label is clipped on the fold button; the menu keeps it whole."""
        bar = ProjectionChipBar()
        bar.set_projections([{"label": "UpstreamComposite"}, {"label": "Downstream"}])
        bounded = bar.minimumSizeHint().width()
        long_label = "UpstreamCompositeAsymmetryProjection" * 4
        bar.set_projections([{"label": long_label}, {"label": "Downstream"}], [long_label])
        # Both names clip to the same text, so the label's length costs nothing.
        assert bar.minimumSizeHint().width() == bounded
        assert bar._fold_btn.text() == "Upstrea… ▾"
        bar._rebuild_fold_menu()
        assert long_label in {action.text() for action in bar._fold_menu.actions()}

    def test_short_label_abbreviates_detector_pairs_only(self):
        assert short_projection_label("Top-Bottom") == "T–B"
        assert short_projection_label("Fwd-Back") == "F–B"
        assert short_projection_label("P_x") == "P_x"


class TestFrameSwitch:
    """The Lab | Rotating switch of the transverse pair (rotating-frame-projection D1)."""

    def test_switch_needs_the_transverse_pair(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_TRANSVERSE)
        assert not bar.frame_available()
        assert bar._frame_switch.isHidden()
        bar.set_projections(_VECTOR)
        assert bar.frame_available()
        assert not bar._frame_switch.isHidden()

    def test_rotating_relabels_the_pair_and_keeps_chip_identity(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR, ["P_x", "P_z"])
        frames: list[bool] = []
        bar.frame_changed.connect(frames.append)
        bar._rotating_btn.click()
        assert frames == [True]
        assert bar.is_rotating()
        assert [chip.text() for chip in bar._chips.values()] == ["P′_x", "P′_y", "P_z"]
        assert bar.selected_labels() == ["P_x", "P_z"]
        assert PROJECTION_TINTS["P_x"] in bar._chips["P_x"].styleSheet()
        bar._rotating_btn.click()  # already rotating: no second event
        assert frames == [True]
        bar._lab_btn.click()
        assert frames == [True, False]
        assert bar._chips["P_x"].text() == "P_x"

    def test_switch_shortens_then_folds_into_the_menu(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        bar.show()
        bar.resize(bar.sizeHint().width(), bar.sizeHint().height())
        qapp.processEvents()
        assert (bar._lab_btn.text(), bar._rotating_btn.text()) == ("Lab", "Rotating")

        bar.resize(bar.sizeHint().width() - 1, bar.sizeHint().height())
        qapp.processEvents()
        assert bar.presentation() == "short"
        assert (bar._lab_btn.text(), bar._rotating_btn.text()) == ("Lab", "Rot")
        assert bar._rotating_btn.toolTip() == "Rotating frame"

        bar.resize(bar.minimumSizeHint().width(), bar.sizeHint().height())
        qapp.processEvents()
        assert bar.presentation() == "folded"
        assert bar._frame_switch.isHidden()
        frames: list[bool] = []
        bar.frame_changed.connect(frames.append)
        bar._rebuild_fold_menu()
        texts = [action.text() for action in bar._fold_menu.actions()]
        assert texts[:3] == ["Frame", "Lab", "Rotating"]
        assert texts.index("Projections shown") > 3
        next(a for a in bar._fold_menu.actions() if a.text() == "Rotating").trigger()
        assert frames == [True]
        assert bar._fold_btn.text() == "Rot · All 3 ▾"
        bar._rebuild_fold_menu()
        assert "P′_x" in [action.text() for action in bar._fold_menu.actions()]

    def test_rotating_does_not_widen_the_folded_bar(self, qapp):
        bar = ProjectionChipBar()
        bar.set_projections(_VECTOR)
        lab = bar.minimumSizeHint().width()
        bar.set_rotating(True)
        assert bar.minimumSizeHint().width() == lab
        bar.set_selected(["P_y"])
        assert bar._fold_btn.text() == "Rot · P′_y ▾"

    def test_set_rotating_is_silent_and_survives_runs_without_the_pair(self, qapp):
        bar = ProjectionChipBar()
        frames: list[bool] = []
        bar.frame_changed.connect(frames.append)
        bar.set_rotating(True)
        assert frames == []
        assert bar.rotating_chosen() and not bar.is_rotating()
        bar.set_projections(_VECTOR)
        assert bar.is_rotating()
