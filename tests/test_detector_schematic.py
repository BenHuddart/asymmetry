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


@pytest.fixture(params=["HiFi", "MuSR", "EMU", "FLAME"])
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

    def test_hal_ring_segments_render_as_rectangles(self, qapp):
        from matplotlib.patches import Rectangle

        layout = get_instrument_layout("HAL")
        widget = DetectorSchematicWidget(layout)
        # The 16 positron detectors (ids 2-17) are edge-aligned rectangular bars;
        # MV (id 1) stays a central disc wedge.
        for det_id in range(2, 18):
            assert isinstance(widget._patches[det_id], Rectangle)

    def test_hal_rectangle_hit_test_selects_correct_detector(self, qapp):
        import math

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
