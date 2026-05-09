"""Tests for the DetectorLayoutDialog."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import INSTRUMENT_NAMES, get_instrument_layout
from asymmetry.gui.windows.detector_layout_dialog import _MAX_GROUPS, DetectorLayoutDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _default_groups() -> dict[int, list[int]]:
    return {1: list(range(1, 33)), 2: list(range(33, 65))}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_for_hifi(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        assert dlg is not None

    def test_creates_for_musr(self, qapp):
        layout = get_instrument_layout("MuSR")
        groups = {1: list(range(1, 33)), 2: list(range(33, 65))}
        dlg = DetectorLayoutDialog(layout, groups=groups)
        assert dlg is not None

    def test_creates_for_emu(self, qapp):
        layout = get_instrument_layout("EMU")
        groups = {1: list(range(1, 49)), 2: list(range(49, 97))}
        dlg = DetectorLayoutDialog(layout, groups=groups)
        assert dlg is not None

    def test_creates_for_flame(self, qapp):
        layout = get_instrument_layout("FLAME")
        groups = {1: [1], 2: [2]}
        dlg = DetectorLayoutDialog(layout, groups=groups)
        assert dlg is not None

    def test_creates_with_empty_groups(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert dlg is not None

    def test_creates_with_group_names(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(
            layout,
            groups=_default_groups(),
            group_names={1: "Forward", 2: "Backward"},
        )
        assert dlg._group_names == {1: "Forward", 2: "Backward"}

    def test_eight_group_buttons_exist(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert len(dlg._group_buttons) == _MAX_GROUPS

    def test_eight_name_edits_exist(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert len(dlg._group_name_edits) == _MAX_GROUPS

    def test_group_button_styles_scale_geometry(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})

        dlg._on_ui_scale_changed(1.0, 1.1)

        assert dlg._group_buttons[1].width() == 84
        assert dlg._group_name_edits[1].width() == 121
        assert "border-radius: 15px;" in dlg._group_buttons[1].styleSheet()

    def test_instrument_combo_has_all_names(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        combo_texts = [
            dlg._instrument_combo.itemText(i) for i in range(dlg._instrument_combo.count())
        ]
        for name in INSTRUMENT_NAMES:
            assert name in combo_texts

    def test_preset_combo_populated_for_hifi(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        preset_texts = [dlg._preset_combo.itemText(i) for i in range(dlg._preset_combo.count())]
        assert "Longitudinal" in preset_texts

    def test_instrument_combo_shows_current_instrument(self, qapp):
        for name in INSTRUMENT_NAMES:
            layout = get_instrument_layout(name)
            groups = {}
            dlg = DetectorLayoutDialog(layout, groups=groups)
            assert dlg._instrument_combo.currentText() == name


# ---------------------------------------------------------------------------
# Group name edits prefilled from constructor
# ---------------------------------------------------------------------------


class TestGroupNamesPrefill:
    def test_name_edits_prefilled(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(
            layout,
            groups=_default_groups(),
            group_names={1: "Fwd", 2: "Bwd"},
        )
        assert dlg._group_name_edits[1].text() == "Fwd"
        assert dlg._group_name_edits[2].text() == "Bwd"

    def test_unused_group_name_edits_are_empty(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(
            layout,
            groups=_default_groups(),
            group_names={1: "Fwd"},
        )
        # Slot 3 through 8 should be empty
        for gid in range(3, _MAX_GROUPS + 1):
            assert dlg._group_name_edits[gid].text() == ""


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------


class TestPresetApplication:
    def test_apply_longitudinal_preset_hifi(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        # Select Longitudinal and apply
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        result = dlg.get_result()
        assert set(result["groups"][1]) == set(range(1, 33))
        assert set(result["groups"][2]) == set(range(33, 65))

    def test_apply_longitudinal_sets_group_names(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        result = dlg.get_result()
        assert result["group_names"].get(1) == "Forward"
        assert result["group_names"].get(2) == "Backward"

    def test_apply_longitudinal_sets_forward_backward(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        result = dlg.get_result()
        assert result["forward_group"] == 1
        assert result["backward_group"] == 2

    def test_apply_emu_vector_polarization(self, qapp):
        layout = get_instrument_layout("EMU")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Vector Polarization")
        dlg._on_apply_preset()
        result = dlg.get_result()
        # Six groups
        assert len(result["groups"]) == 6
        names = set(result["group_names"].values())
        assert "Pz Forward" in names
        assert "Pz Backward" in names
        assert "Py Top" in names
        assert "Py Bottom" in names
        assert "Px Left" in names
        assert "Px Right" in names

    def test_apply_flame_transverse(self, qapp):
        layout = get_instrument_layout("FLAME")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Transverse")
        dlg._on_apply_preset()
        result = dlg.get_result()
        assert set(result["groups"][1]) == {3, 5, 6}
        assert set(result["groups"][2]) == {4, 7, 8}
        assert result["group_names"][1] == "Right"
        assert result["group_names"][2] == "Left"

    def test_apply_preset_populates_name_edits(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        assert dlg._group_name_edits[1].text() == "Forward"
        assert dlg._group_name_edits[2].text() == "Backward"

    def test_apply_preset_clears_unused_slot_names(self, qapp):
        layout = get_instrument_layout("HiFi")
        # Start with group 3 named something
        dlg = DetectorLayoutDialog(layout, groups={}, group_names={3: "Old name"})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        # After applying Longitudinal (only 2 groups), slot 3 should be cleared
        assert dlg._group_name_edits[3].text() == ""

    def test_instrument_override_changes_presets(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        # Switch instrument to EMU
        dlg._instrument_combo.setCurrentText("EMU")
        preset_texts = [dlg._preset_combo.itemText(i) for i in range(dlg._preset_combo.count())]
        assert "Vector Polarization" in preset_texts
        assert "Longitudinal" in preset_texts

    def test_preset_status_shows_applied_preset(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        assert dlg._preset_status_label.text() == "(Current: Longitudinal)"

    def test_preset_status_resets_to_custom_after_edit(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()

        dlg._active_group = 1
        dlg._on_detector_toggled(1, False)
        assert dlg._preset_status_label.text() == "(Current: Custom)"

    def test_initial_preset_name_restores_status(self, qapp):
        layout = get_instrument_layout("HiFi")
        groups = {1: list(range(1, 33)), 2: list(range(33, 65))}
        dlg = DetectorLayoutDialog(
            layout,
            groups=groups,
            group_names={1: "Forward", 2: "Backward"},
            initial_preset_name="Longitudinal",
            forward_group=1,
            backward_group=2,
        )
        assert dlg._preset_status_label.text() == "(Current: Longitudinal)"


# ---------------------------------------------------------------------------
# get_result contract
# ---------------------------------------------------------------------------


class TestGetResult:
    def test_result_keys_present(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        result = dlg.get_result()
        assert "groups" in result
        assert "group_names" in result
        assert "forward_group" in result
        assert "backward_group" in result
        assert "instrument" in result
        assert "grouping_preset" in result

    def test_result_instrument_name(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert dlg.get_result()["instrument"] == "HiFi"

    def test_result_only_non_empty_groups(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [1, 2], 2: []})
        result = dlg.get_result()
        assert 2 not in result["groups"]
        assert 1 in result["groups"]

    def test_result_groups_sorted_detector_ids(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [3, 1, 2]})
        result = dlg.get_result()
        assert result["groups"][1] == [1, 2, 3]

    def test_result_group_names_from_edits(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [1, 2]})
        dlg._group_name_edits[1].setText("My Group")
        result = dlg.get_result()
        assert result["group_names"].get(1) == "My Group"

    def test_result_group_names_empty_edit_excluded(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [1, 2]})
        dlg._group_name_edits[1].setText("")
        result = dlg.get_result()
        assert 1 not in result["group_names"]

    def test_forward_backward_preserved_from_constructor(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(
            layout,
            groups=_default_groups(),
            forward_group=2,
            backward_group=1,
        )
        result = dlg.get_result()
        assert result["forward_group"] == 2
        assert result["backward_group"] == 1


# ---------------------------------------------------------------------------
# Detector toggle
# ---------------------------------------------------------------------------


class TestDetectorToggle:
    def test_on_detector_toggled_adds_to_active_group(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._active_group = 1
        dlg._on_detector_toggled(5, True)
        assert 5 in dlg._groups.get(1, set())

    def test_on_detector_toggled_removes_from_active_group(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [5]})
        dlg._active_group = 1
        dlg._on_detector_toggled(5, False)
        assert 5 not in dlg._groups.get(1, set())

    def test_toggle_in_keeps_detector_in_other_groups(self, qapp):
        """Adding a detector to group 2 does not remove it from group 1."""
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={1: [5], 2: []})
        dlg._active_group = 2
        dlg._on_detector_toggled(5, True)
        assert 5 in dlg._groups.get(1, set())
        assert 5 in dlg._groups.get(2, set())
