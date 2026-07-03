"""Tests for the DetectorLayoutDialog."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import (
    INSTRUMENT_NAMES,
    get_instrument_layout,
    instrument_choices_for,
    instrument_display_name,
)
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
        # The dropdown shows display names, with the GPS variants collapsed to a
        # single "GPS" entry.
        for display, _key in instrument_choices_for(None):
            assert display in combo_texts
        assert combo_texts.count("GPS") == 1

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
            # Combo shows the display name; itemData carries the registry key.
            assert dlg._instrument_combo.currentText() == instrument_display_name(name)
            assert dlg._instrument_combo.currentData() == name

    def test_gps_root_variant_shown_as_gps(self, qapp):
        # An 11-detector GPS ROOT run shows a single "GPS" entry mapping to the
        # GPS-RD layout.
        layout = get_instrument_layout("GPS-RD")
        dlg = DetectorLayoutDialog(layout, groups={})
        combo_texts = [
            dlg._instrument_combo.itemText(i) for i in range(dlg._instrument_combo.count())
        ]
        assert combo_texts.count("GPS") == 1
        assert dlg._instrument_combo.currentText() == "GPS"
        assert dlg._instrument_combo.currentData() == "GPS-RD"
        assert dlg.get_result()["instrument"] == "GPS-RD"


# ---------------------------------------------------------------------------
# Transverse-field grouping nudge (B8a)
# ---------------------------------------------------------------------------


class TestTransverseFieldNudge:
    """A TF run on a longitudinal preset should nudge toward the spin-rotated one."""

    _RECOMMENDED = "Spin-rotated (B+U/F+D)"

    def test_tf_on_longitudinal_shows_hint_and_preselects(self, qapp):
        layout = get_instrument_layout("GPS")
        dlg = DetectorLayoutDialog(
            layout,
            groups={1: [1], 2: [2]},
            initial_preset_name="Longitudinal",
            field_direction="Transverse",
        )
        assert dlg._tf_hint_label.isVisibleTo(dlg)
        assert self._RECOMMENDED in dlg._tf_hint_label.text()
        # Recommended preset pre-selected so applying it is one click.
        assert dlg._preset_combo.currentText() == self._RECOMMENDED

    def test_no_field_direction_does_not_nudge(self, qapp):
        layout = get_instrument_layout("GPS")
        dlg = DetectorLayoutDialog(
            layout, groups={1: [1], 2: [2]}, initial_preset_name="Longitudinal"
        )
        assert not dlg._tf_hint_label.isVisibleTo(dlg)

    def test_longitudinal_run_does_not_nudge(self, qapp):
        layout = get_instrument_layout("GPS")
        dlg = DetectorLayoutDialog(
            layout,
            groups={1: [1], 2: [2]},
            initial_preset_name="Longitudinal",
            field_direction="Longitudinal",
        )
        assert not dlg._tf_hint_label.isVisibleTo(dlg)

    def test_hint_clears_after_applying_recommended_preset(self, qapp):
        layout = get_instrument_layout("GPS")
        dlg = DetectorLayoutDialog(
            layout,
            groups={1: [1], 2: [2]},
            initial_preset_name="Longitudinal",
            field_direction="Transverse",
        )
        assert dlg._tf_hint_label.isVisibleTo(dlg)
        # The recommended preset is pre-selected; applying it dismisses the nudge.
        dlg._on_apply_preset()
        assert dlg._applied_preset_name == self._RECOMMENDED
        assert not dlg._tf_hint_label.isVisibleTo(dlg)


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

    def test_apply_emu_vector_polarization_emits_projections(self, qapp):
        layout = get_instrument_layout("EMU")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Vector Polarization")
        dlg._on_apply_preset()
        result = dlg.get_result()
        projections = result["projections"]
        assert [p["label"] for p in projections] == ["P_x", "P_y", "P_z"]
        by_label = {p["label"]: p for p in projections}
        assert (by_label["P_z"]["forward_group"], by_label["P_z"]["backward_group"]) == (1, 2)
        assert all(p["tint"] for p in projections)

    def test_longitudinal_preset_emits_no_projections(self, qapp):
        layout = get_instrument_layout("EMU")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        assert dlg.get_result()["projections"] == []

    def test_seeded_projections_survive_open_without_reapply(self, qapp):
        layout = get_instrument_layout("EMU")
        seeded = [{"label": "P_x", "forward_group": 5, "backward_group": 6}]
        dlg = DetectorLayoutDialog(layout, groups={}, projections=seeded)
        assert dlg.get_result()["projections"] == seeded

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


# ---------------------------------------------------------------------------
# Group button member-count labels (problem 5a)
# ---------------------------------------------------------------------------


class TestGroupButtonCounts:
    def test_button_label_shows_member_count(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        assert dlg._group_buttons[1].text() == "Group 1 (32)"
        assert dlg._group_buttons[2].text() == "Group 2 (32)"

    def test_button_label_uses_group_name_when_set(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(
            layout, groups=_default_groups(), group_names={1: "Forward", 2: "Backward"}
        )
        assert dlg._group_buttons[1].text() == "Forward (32)"
        assert dlg._group_buttons[2].text() == "Backward (32)"

    def test_empty_group_button_has_no_count_suffix(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert dlg._group_buttons[3].text() == "Group 3"

    def test_button_label_updates_after_detector_toggle(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._active_group = 1
        dlg._on_detector_toggled(5, True)
        assert dlg._group_buttons[1].text() == "Group 1 (1)"

    def test_transverse_vector_preset_matches_audited_example(self, qapp):
        """HiFi's Transverse (Vector) preset gives each split group 18 members
        (verbatim example from the audit: "Top-Bottom Top (18)")."""
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Transverse (Vector)")
        dlg._on_apply_preset()
        assert dlg._group_buttons[3].text() == "Top-Bottom Top (18)"

    def test_button_label_updates_after_apply_preset(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        dlg._preset_combo.setCurrentText("Longitudinal")
        dlg._on_apply_preset()
        assert dlg._group_buttons[1].text() == "Forward (32)"
        assert dlg._group_buttons[2].text() == "Backward (32)"

    def test_button_label_updates_after_clear_all(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        for members in dlg._groups.values():
            members.clear()
        dlg._sync_schematic()
        dlg._on_group_definition_changed()
        assert dlg._group_buttons[1].text() == "Group 1"


# ---------------------------------------------------------------------------
# Group-row hover highlight (problem 5b)
# ---------------------------------------------------------------------------


class TestGroupRowHoverHighlight:
    def test_hover_enter_highlights_schematic_group(self, qapp):
        from PySide6.QtCore import QEvent

        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        row = dlg._group_rows[1]
        dlg.eventFilter(row, QEvent(QEvent.Type.Enter))
        assert dlg._schematic._highlighted_groups == {1}

    def test_hover_leave_clears_highlight(self, qapp):
        from PySide6.QtCore import QEvent

        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups=_default_groups())
        row = dlg._group_rows[1]
        dlg.eventFilter(row, QEvent(QEvent.Type.Enter))
        dlg.eventFilter(row, QEvent(QEvent.Type.Leave))
        assert dlg._schematic._highlighted_groups == set()

    def test_all_eight_group_rows_exist(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert len(dlg._group_rows) == _MAX_GROUPS


# ---------------------------------------------------------------------------
# "Clear excluded" button (problem 5c)
# ---------------------------------------------------------------------------


class TestClearExcluded:
    def test_clear_excluded_button_exists(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={})
        assert dlg._clear_excluded_btn is not None

    def test_clear_excluded_empties_exclusion_set(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={}, excluded_detectors=[1, 2, 3])
        assert dlg._schematic.get_excluded_detectors() == {1, 2, 3}
        dlg._on_clear_excluded()
        assert dlg._schematic.get_excluded_detectors() == set()

    def test_clear_excluded_button_click_clears_via_ui(self, qapp):
        layout = get_instrument_layout("HiFi")
        dlg = DetectorLayoutDialog(layout, groups={}, excluded_detectors=[7])
        dlg._clear_excluded_btn.click()
        assert dlg._schematic.get_excluded_detectors() == set()
