"""Tests for the ProjectionChipBar widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from asymmetry.core.instrument import PROJECTION_TINTS
from asymmetry.gui.widgets.projection_chip_bar import ProjectionChipBar

_VECTOR = [
    {"label": "P_x", "tint": PROJECTION_TINTS["P_x"]},
    {"label": "P_y", "tint": PROJECTION_TINTS["P_y"]},
    {"label": "P_z", "tint": PROJECTION_TINTS["P_z"]},
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
