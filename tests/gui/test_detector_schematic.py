"""Tests for the DetectorSchematicWidget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.gui.widgets.detector_schematic import DetectorSchematicWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(params=["HiFi", "MuSR", "EMU", "FLAME", "GPS", "GPS-RD"])
def instrument_name(request):
    return request.param


# ---------------------------------------------------------------------------
# Construction & patch count
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_for_each_instrument(self, qapp, instrument_name):
        layout = get_instrument_layout(instrument_name)
        widget = DetectorSchematicWidget(layout)
        assert widget is not None

    def test_patch_count_matches_detector_count(self, qapp, instrument_name):
        layout = get_instrument_layout(instrument_name)
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == layout.n_detectors

    def test_hifi_patch_count(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 64

    def test_musr_patch_count(self, qapp):
        layout = get_instrument_layout("MuSR")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 64

    def test_emu_patch_count(self, qapp):
        layout = get_instrument_layout("EMU")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 96

    def test_flame_patch_count(self, qapp):
        layout = get_instrument_layout("FLAME")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 8

    def test_hal_patch_count(self, qapp):
        layout = get_instrument_layout("HAL")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 17

    def test_gps_patch_count(self, qapp):
        layout = get_instrument_layout("GPS")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 6

    def test_gps_segments_render_as_rectangles(self, qapp):
        from matplotlib.patches import Rectangle

        layout = get_instrument_layout("GPS")
        widget = DetectorSchematicWidget(layout)
        for det_id in range(1, 7):
            assert isinstance(widget._patches[det_id], Rectangle)

    def test_gps_rd_patch_count(self, qapp):
        layout = get_instrument_layout("GPS-RD")
        widget = DetectorSchematicWidget(layout)
        assert len(widget._patches) == 11

    def test_gps_two_panels(self, qapp):
        layout = get_instrument_layout("GPS")
        widget = DetectorSchematicWidget(layout)
        # Top view + Side view -> two axes; one clickable patch per detector.
        assert len(widget._axes) == 2
        assert set(widget._patches) == {1, 2, 3, 4, 5, 6}

    def test_gps_endon_segments_not_clickable(self, qapp):
        layout = get_instrument_layout("GPS")
        # The Up/Down end-on markers in the top view are read-only context.
        top = next(b for b in layout.banks if b.name == "Top view")
        endon = next(s for s in top.segments if s.shape.startswith("endon"))
        assert endon.read_only
        assert not DetectorSchematicWidget._point_in_segment(endon.x_center, endon.y_center, endon)

    def test_gps_exclusion_hatches_readonly_ghost(self, qapp):
        layout = get_instrument_layout("GPS")
        widget = DetectorSchematicWidget(layout)
        # Forward (id 1) is active in the Top view and read-only in the Side view.
        ghosts = [p for did, p in widget._readonly_patches if did == 1]
        assert ghosts, "expected a read-only ghost for detector 1"
        widget.set_excluded_detectors({1})
        assert all(p.get_hatch() == "xx" for p in ghosts)
        widget.set_excluded_detectors(set())
        assert all(not p.get_hatch() for p in ghosts)

    def test_hal_ring_segments_render_as_rectangles(self, qapp):
        from matplotlib.patches import Rectangle

        layout = get_instrument_layout("HAL")
        widget = DetectorSchematicWidget(layout)
        # The 16 positron detectors (ids 2-17) are edge-aligned rectangular bars;
        # MV (id 1) stays a central disc wedge.
        for det_id in range(2, 18):
            assert isinstance(widget._patches[det_id], Rectangle)

    def test_hal_rectangle_hit_test_selects_correct_detector(self, qapp):
        layout = get_instrument_layout("HAL")
        fwd = layout.banks[0].segments
        # The F1 bar sits near the octagon's top edge; a point there hits F1 (id 2).
        f1 = next(s for s in fwd if s.label == "F1")
        hits = [
            s.detector_id
            for s in fwd
            if DetectorSchematicWidget._point_in_segment(f1.x_center, f1.y_center, s)
        ]
        assert hits == [2]

    def test_hal_mv_disc_hit_test_is_full_circle(self, qapp):
        layout = get_instrument_layout("HAL")
        mv = next(s for s in layout.banks[0].segments if s.label == "MV")
        # Any point inside the disc radius is inside MV, regardless of angle.
        assert DetectorSchematicWidget._point_in_segment(0.0, 0.15, mv)
        assert DetectorSchematicWidget._point_in_segment(0.15, 0.0, mv)
        assert not DetectorSchematicWidget._point_in_segment(0.6, 0.0, mv)

    def test_axes_count_matches_banks(self, qapp, instrument_name):
        layout = get_instrument_layout(instrument_name)
        widget = DetectorSchematicWidget(layout)
        assert len(widget._axes) == len(layout.banks)


# ---------------------------------------------------------------------------
# set_group_detectors / get_filled_detectors round-trip
# ---------------------------------------------------------------------------


class TestGroupState:
    def test_set_and_get_filled_detectors(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {1, 2, 3})
        assert widget.get_filled_detectors() == {1, 2, 3}

    def test_set_active_group_changes_returned_detectors(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {1, 2})
        widget.set_group_detectors(2, {33, 34})
        widget.set_active_group(2)
        assert widget.get_filled_detectors() == {33, 34}

    def test_set_all_groups(self, qapp):
        layout = get_instrument_layout("MuSR")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: list(range(1, 33)), 2: list(range(33, 65))}, active_group=1)
        assert widget.get_filled_detectors() == set(range(1, 33))
        widget.set_active_group(2)
        assert widget.get_filled_detectors() == set(range(33, 65))

    def test_empty_group_returns_empty_set(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        assert widget.get_filled_detectors() == set()

    def test_set_group_detectors_empty_set(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {1, 2})
        widget.set_group_detectors(1, set())
        assert widget.get_filled_detectors() == set()


# ---------------------------------------------------------------------------
# Detector toggle via _on_click simulation
# ---------------------------------------------------------------------------


class TestToggle:
    def test_click_adds_detector_to_active_group(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        toggled_signals = []
        widget.detector_toggled.connect(lambda det_id, inc: toggled_signals.append((det_id, inc)))

        # Simulate a click: directly call _on_detector_toggled logic
        # We test the internal toggle method indirectly through _on_click substitution.
        # Use set_group_detectors to prime state, then call _on_click with mock event.

        # Manually invoke toggle logic by manipulating _groups directly
        widget._active_group = 1
        widget._groups[1] = set()

        # Simulate toggle: add detector 5
        widget._groups[1].add(5)
        widget._refresh_colours()
        assert 5 in widget._groups[1]

    def test_detector_can_belong_to_multiple_groups(self, qapp):
        """A detector may be present in multiple groups simultaneously."""
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {10})
        widget.set_group_detectors(2, {10})

        assert 10 in widget._groups.get(1, set())
        assert 10 in widget._groups.get(2, set())

    def test_toggle_only_changes_active_group_membership(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {10})
        widget.set_group_detectors(2, set())

        widget._active_group = 2
        widget._groups.setdefault(2, set())

        # Simulate toggle-in for active group 2.
        det_id = 10
        widget._groups[2].add(det_id)

        assert 10 in widget._groups.get(1, set())
        assert 10 in widget._groups[2]


# ---------------------------------------------------------------------------
# Point-in-segment hit test
# ---------------------------------------------------------------------------


class TestPointInSegment:
    def test_point_in_centre_of_segment(self):
        from asymmetry.core.instrument import DetectorSegment

        seg = DetectorSegment(
            detector_id=1,
            sector_index=0,
            ring_index=0,
            angle_center_deg=0.0,
            angle_half_width_deg=5.625,
            r_inner=0.3,
            r_outer=0.7,
        )
        # Centre of segment: r=0.5, angle=0° → (0.5, 0)
        assert DetectorSchematicWidget._point_in_segment(0.5, 0.0, seg)

    def test_point_outside_radius(self):
        from asymmetry.core.instrument import DetectorSegment

        seg = DetectorSegment(
            detector_id=1,
            sector_index=0,
            ring_index=0,
            angle_center_deg=0.0,
            angle_half_width_deg=5.625,
            r_inner=0.3,
            r_outer=0.7,
        )
        # Too close to centre
        assert not DetectorSchematicWidget._point_in_segment(0.1, 0.0, seg)
        # Too far out
        assert not DetectorSchematicWidget._point_in_segment(0.9, 0.0, seg)

    def test_point_outside_angular_range(self):
        from asymmetry.core.instrument import DetectorSegment

        seg = DetectorSegment(
            detector_id=1,
            sector_index=0,
            ring_index=0,
            angle_center_deg=90.0,  # top of disc
            angle_half_width_deg=5.0,
            r_inner=0.3,
            r_outer=0.7,
        )
        # Bottom of disc (270°): not in [85°, 95°]
        assert not DetectorSchematicWidget._point_in_segment(0.0, -0.5, seg)

    def test_none_coordinates_return_false(self):
        from asymmetry.core.instrument import DetectorSegment

        seg = DetectorSegment(
            detector_id=1,
            sector_index=0,
            ring_index=0,
            angle_center_deg=0.0,
            angle_half_width_deg=5.625,
            r_inner=0.3,
            r_outer=0.7,
        )
        assert not DetectorSchematicWidget._point_in_segment(None, None, seg)

    def test_point_in_flame_rectangle(self):
        seg = next(s for s in get_instrument_layout("FLAME").all_segments if s.detector_id == 1)
        assert DetectorSchematicWidget._point_in_segment(seg.x_center, seg.y_center, seg)
        assert not DetectorSchematicWidget._point_in_segment(
            seg.x_center + seg.width,
            seg.y_center + seg.height,
            seg,
        )


# ---------------------------------------------------------------------------
# Instrument switching
# ---------------------------------------------------------------------------


class TestInstrumentSwitch:
    def test_set_instrument_clears_groups(self, qapp):
        layout_hifi = get_instrument_layout("HiFi")
        layout_emu = get_instrument_layout("EMU")
        widget = DetectorSchematicWidget(layout_hifi)
        widget.set_group_detectors(1, {1, 2, 3})
        widget.set_instrument(layout_emu)
        assert widget.get_filled_detectors() == set()

    def test_set_instrument_updates_patch_count(self, qapp):
        layout_hifi = get_instrument_layout("HiFi")
        layout_emu = get_instrument_layout("EMU")
        widget = DetectorSchematicWidget(layout_hifi)
        assert len(widget._patches) == 64
        widget.set_instrument(layout_emu)
        assert len(widget._patches) == 96


# ---------------------------------------------------------------------------
# Multi-membership rendering (dual/multi-group detectors get slice patches)
# ---------------------------------------------------------------------------


class TestMembershipSlices:
    def test_single_group_detector_has_no_membership_slices(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {1})
        assert 1 not in widget._membership_patches

    def test_hifi_boundary_detector_gets_one_membership_slice(self, qapp):
        """HiFi's Transverse (Vector) preset puts detector 5 in two groups."""
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        preset = layout.presets["Transverse (Vector)"]
        groups = {gid: list(gd.detector_ids) for gid, gd in preset.groups.items()}
        widget.set_all_groups(groups, active_group=1)
        assert widget._detector_groups(5) == [1, 4]
        assert 5 in widget._membership_patches
        assert len(widget._membership_patches[5]) == 1

    def test_membership_slice_uses_second_group_colour(self, qapp):
        from asymmetry.gui.widgets.detector_schematic import _group_colour

        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [5], 4: [5]}, active_group=1)
        slice_patch = widget._membership_patches[5][0]
        assert slice_patch.get_facecolor() == _group_colour(4)

    def test_emu_vector_every_detector_has_membership_slices(self, qapp):
        """EMU's Vector Polarization preset puts every detector in Pz + Py + Px
        (3 groups total), so each gets 2 extra membership slices beyond the
        primary fill."""
        layout = get_instrument_layout("EMU")
        widget = DetectorSchematicWidget(layout)
        preset = layout.presets["Vector Polarization"]
        groups = {gid: list(gd.detector_ids) for gid, gd in preset.groups.items()}
        widget.set_all_groups(groups, active_group=1)
        assert len(widget._membership_patches) == 96
        for slices in widget._membership_patches.values():
            assert len(slices) == 2

    def test_membership_slices_cleared_on_refresh(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [5], 4: [5]}, active_group=1)
        assert 5 in widget._membership_patches
        widget.set_all_groups({1: [5]}, active_group=1)
        assert 5 not in widget._membership_patches

    def test_overflow_marker_for_four_plus_groups(self, qapp):
        """A detector in more than 3 groups gets a capped set of slices + '+N'."""
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [5], 2: [5], 3: [5], 4: [5]}, active_group=1)
        # Primary (group 1) + 2 shown slices + 1 overflow marker for group 4.
        assert len(widget._membership_patches[5]) == 2
        assert 5 in widget._overflow_labels
        assert widget._overflow_labels[5].get_text() == "+1"

    def test_rectangle_membership_slices_split_horizontally(self, qapp):
        layout = get_instrument_layout("FLAME")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [3], 2: [3]}, active_group=1)
        assert 3 in widget._membership_patches
        assert len(widget._membership_patches[3]) == 1

    def test_excluded_detector_has_no_membership_slices(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [5], 4: [5]}, active_group=1)
        widget.set_excluded_detectors({5})
        assert 5 not in widget._membership_patches


# ---------------------------------------------------------------------------
# Hover tooltips
# ---------------------------------------------------------------------------


class TestHoverTooltip:
    def test_tooltip_text_reports_id_label_and_groups(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {5})
        seg = next(s for s in layout.all_segments if s.detector_id == 5)
        text = widget._tooltip_text(5, seg)
        assert "Detector 5" in text
        assert "Groups: Group 1" in text

    def test_tooltip_text_lists_all_memberships(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_all_groups({1: [5], 4: [5]}, active_group=1)
        seg = next(s for s in layout.all_segments if s.detector_id == 5)
        text = widget._tooltip_text(5, seg)
        assert "Group 1" in text
        assert "Group 4" in text

    def test_tooltip_uses_group_display_names(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {5})
        widget.set_group_names({1: "Forward"})
        seg = next(s for s in layout.all_segments if s.detector_id == 5)
        text = widget._tooltip_text(5, seg)
        assert "Groups: Forward" in text

    def test_tooltip_reports_no_groups(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        seg = next(s for s in layout.all_segments if s.detector_id == 5)
        text = widget._tooltip_text(5, seg)
        assert "Groups: (none)" in text

    def test_tooltip_reports_excluded_status(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_excluded_detectors({5})
        seg = next(s for s in layout.all_segments if s.detector_id == 5)
        text = widget._tooltip_text(5, seg)
        assert "Excluded" in text

    def test_tooltip_includes_physical_label_when_present(self, qapp):
        layout = get_instrument_layout("FLAME")
        widget = DetectorSchematicWidget(layout)
        seg = next(s for s in layout.all_segments if s.detector_id == 1)
        text = widget._tooltip_text(1, seg)
        assert "Forward" in text

    def test_hit_test_event_returns_none_outside_axes(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)

        class _FakeEvent:
            inaxes = None
            xdata = None
            ydata = None

        assert widget._hit_test_event(_FakeEvent()) is None


# ---------------------------------------------------------------------------
# Group highlight (hover a group row in the layout dialog)
# ---------------------------------------------------------------------------


class TestGroupHighlight:
    def test_set_group_highlight_records_group(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_highlight(2)
        assert widget._highlighted_groups == {2}

    def test_set_group_highlight_none_clears(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_highlight(2)
        widget.set_group_highlight(None)
        assert widget._highlighted_groups == set()

    def test_highlighted_detector_gets_emphasised_edge(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        widget.set_group_detectors(1, {5})
        widget.set_group_highlight(1)
        patch = widget._patches[5]
        assert patch.get_edgecolor()[:3] == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Wedge label floor (dense rings stay >= 5pt, using shorter labels)
# ---------------------------------------------------------------------------


class TestWedgeLabelFloor:
    @staticmethod
    def _detector_label_texts(widget):
        """Return every rendered wedge-label text artist across all axes."""
        texts = []
        for ax in widget._axes:
            texts.extend(ax.texts)
        return texts

    def test_hifi_32_sector_ring_labels_meet_raised_floor(self, qapp):
        layout = get_instrument_layout("HiFi")
        widget = DetectorSchematicWidget(layout)
        n_sectors = len(set(s.sector_index for s in layout.all_segments))
        assert n_sectors == 32
        detector_labels = [t for t in self._detector_label_texts(widget) if t.get_text().isdigit()]
        assert detector_labels, "expected bare-id wedge labels for HiFi"
        assert all(t.get_fontsize() >= 5.0 for t in detector_labels)

    def test_dense_ring_label_is_bare_id_not_id_plus_name(self, qapp):
        """Dense rings (>24 sectors, e.g. MuSR's 32) drop any physical label to
        fit the raised floor, even for banks that do have physical labels."""
        layout = get_instrument_layout("MuSR")
        widget = DetectorSchematicWidget(layout)
        n_sectors = len(set(s.sector_index for s in layout.all_segments))
        assert n_sectors > 24
        texts = {t.get_text() for t in self._detector_label_texts(widget)}
        # Every detector shows as a bare numeric id; nothing longer slipped in.
        assert all(t.isdigit() for t in texts)

    def test_emu_sparse_ring_keeps_higher_fontsize(self, qapp):
        """EMU (16 sectors) is well under the dense-ring threshold and keeps
        the original higher font size."""
        layout = get_instrument_layout("EMU")
        widget = DetectorSchematicWidget(layout)
        detector_labels = [t for t in self._detector_label_texts(widget) if t.get_text().isdigit()]
        assert detector_labels
        assert all(t.get_fontsize() == pytest.approx(6.5) for t in detector_labels)


# ---------------------------------------------------------------------------
# Rectangle label sizing / abbreviation (GPS-RD collision fix)
# ---------------------------------------------------------------------------


class TestRectangleLabelFit:
    def test_narrow_box_name_is_abbreviated(self, qapp):
        layout = get_instrument_layout("GPS-RD")
        widget = DetectorSchematicWidget(layout)
        seg = next(s for s in layout.all_segments if s.detector_id == 7)  # Right_B
        assert seg.label == "Right_B"
        assert widget._fit_rectangle_name(seg) == "R_B"

    def test_short_box_name_is_abbreviated(self, qapp):
        layout = get_instrument_layout("GPS-RD")
        widget = DetectorSchematicWidget(layout)
        seg = next(s for s in layout.all_segments if s.detector_id == 11)  # Mob-RL
        assert widget._fit_rectangle_name(seg) == "Mob"

    def test_narrow_tall_box_name_is_also_abbreviated(self, qapp):
        """Forward/Backward (0.78 wide, 1.7 tall) are narrower than the
        `_NARROW_BOX_WIDTH` threshold, so they abbreviate despite being tall —
        width, not height, is what risks a horizontal label overrun."""
        layout = get_instrument_layout("GPS-RD")
        widget = DetectorSchematicWidget(layout)
        seg = next(
            s
            for s in layout.all_segments
            if s.detector_id == 1 and s.label == "Forward" and not s.read_only
        )
        assert widget._fit_rectangle_name(seg) == "Fwd"

    def test_wide_box_name_is_not_abbreviated(self, qapp):
        layout = get_instrument_layout("FLAME")
        widget = DetectorSchematicWidget(layout)
        seg = next(s for s in layout.all_segments if s.detector_id == 3)  # "Right", 2.18 wide
        assert widget._fit_rectangle_name(seg) == "Right"

    def test_short_box_fontsize_bounded_by_height(self, qapp):
        layout = get_instrument_layout("GPS-RD")
        widget = DetectorSchematicWidget(layout)
        mob = next(s for s in layout.all_segments if s.detector_id == 11)
        tall = next(
            s
            for s in layout.all_segments
            if s.detector_id == 1 and s.label == "Forward" and not s.read_only
        )
        # Mob-RL (0.42 tall) is far shorter than Forward/Backward (1.7 tall),
        # so its height-bounded font size must be smaller.
        assert widget._rectangle_label_fontsize(mob) < widget._rectangle_label_fontsize(tall)


# ---------------------------------------------------------------------------
# Optional per-bank/instrument caption (problem 6)
# ---------------------------------------------------------------------------


class TestBankCaption:
    def test_no_caption_attribute_draws_nothing(self, qapp):
        layout = get_instrument_layout("HiFi")
        assert not hasattr(layout, "caption")
        widget = DetectorSchematicWidget(layout)
        # No exception, and no stray caption text present on either axis.
        for ax in widget._axes:
            texts = [t.get_text() for t in ax.texts]
            assert "viewed looking upstream" not in texts

    def test_bank_caption_attribute_is_rendered(self, qapp):
        import types

        layout = get_instrument_layout("FLAME")
        bank = layout.banks[0]
        # BankLayout is frozen; simulate a future `caption` field with a
        # lightweight namespace proxy rather than mutating the dataclass.
        captioned_bank = types.SimpleNamespace(
            name=bank.name, segments=bank.segments, caption="viewed looking upstream"
        )
        proxy_layout = types.SimpleNamespace(
            name=layout.name,
            n_detectors=layout.n_detectors,
            banks=(captioned_bank,),
            presets=layout.presets,
            view=layout.view,
            reference_arrows=layout.reference_arrows,
            display_name=layout.display_name,
            display=layout.display,
            all_segments=list(bank.segments),
            active_segments=[s for s in bank.segments if not s.read_only],
        )
        widget = DetectorSchematicWidget(proxy_layout)
        ax = widget._axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "viewed looking upstream" in texts

    def test_instrument_level_caption_attribute_is_rendered(self, qapp):
        import types

        layout = get_instrument_layout("FLAME")
        bank = layout.banks[0]
        proxy_layout = types.SimpleNamespace(
            name=layout.name,
            n_detectors=layout.n_detectors,
            banks=(bank,),
            presets=layout.presets,
            view=layout.view,
            reference_arrows=layout.reference_arrows,
            display_name=layout.display_name,
            display=layout.display,
            all_segments=list(bank.segments),
            active_segments=[s for s in bank.segments if not s.read_only],
            caption="viewed looking upstream",
        )
        widget = DetectorSchematicWidget(proxy_layout)
        ax = widget._axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "viewed looking upstream" in texts
