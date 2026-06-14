"""Tests for the instrument layout definitions in core/instrument.py."""

from __future__ import annotations

import pytest

from asymmetry.core.instrument import (
    INSTRUMENT_NAMES,
    PROJECTION_TINTS,
    AsymmetryProjection,
    DetectorSegment,
    InstrumentLayout,
    PresetGrouping,
    derive_projection_pairs,
    detect_instrument,
    get_instrument_layout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_detector_ids(layout: InstrumentLayout) -> list[int]:
    """Return all detector IDs across all banks of *layout*."""
    ids = []
    for bank in layout.banks:
        for seg in bank.segments:
            ids.append(seg.detector_id)
    return ids


def _preset_all_ids(preset: PresetGrouping) -> list[int]:
    """Return all detector IDs across all groups in *preset* (may have duplicates)."""
    ids = []
    for gdef in preset.groups.values():
        ids.extend(gdef.detector_ids)
    return ids


# ---------------------------------------------------------------------------
# detect_instrument
# ---------------------------------------------------------------------------


class TestDetectInstrument:
    def test_64_returns_hifi(self):
        assert detect_instrument(64) == "HiFi"

    def test_metadata_instrument_overrides_64_histogram_heuristic(self):
        assert detect_instrument(64, metadata={"instrument": "MUSR"}) == "MuSR"

    def test_source_file_prefix_can_hint_instrument(self):
        assert detect_instrument(64, source_file="/tmp/emu000123.nxs") == "EMU"

    def test_flame_can_be_detected_from_psi_metadata_or_filename(self):
        assert detect_instrument(8, metadata={"facility": "PSI", "instrument": "FLAME"}) == "FLAME"
        assert (
            detect_instrument(
                8,
                metadata={"facility": "PSI", "instrument": ""},
                source_file="/tmp/run_flame_001.bin",
            )
            == "FLAME"
        )

    def test_flame_can_be_detected_from_psi_detector_labels(self):
        labels = ["Forw", "Back", "Righ", "Left", "R_F", "R_B", "L_F", "L_B"]
        assert (
            detect_instrument(
                8,
                metadata={"facility": "PSI", "instrument": "PSI", "histogram_labels": labels},
            )
            == "FLAME"
        )

    def test_psi_metadata_does_not_select_isis_layout(self):
        # PSI "HIFI" is the HAL-9500 high-field instrument, not the ISIS HiFi.
        assert detect_instrument(64, metadata={"facility": "PSI", "instrument": "HIFI"}) == "HAL"
        # A PSI file claiming an ISIS-only instrument name still must not map to
        # that ISIS layout.
        assert (
            detect_instrument(64, metadata={"psi_format": "psi-bin", "instrument": "MuSR"}) is None
        )

    def test_psi_hal_detected_from_metadata_and_filename(self):
        md = {"facility": "PSI", "psi_format": "psi-mdu", "instrument": "HIFI"}
        assert detect_instrument(9, metadata=md) == "HAL"
        assert detect_instrument(17, metadata=md) == "HAL"
        assert (
            detect_instrument(
                17,
                metadata={"facility": "PSI", "psi_format": "psi-mdu", "instrument": ""},
                source_file="/data/tdc_hifi_2025_01425.mdu",
            )
            == "HAL"
        )

    def test_isis_hifi_still_resolves_without_psi_flag(self):
        # Without a PSI marker, the bare "HiFi" name remains the ISIS layout.
        assert detect_instrument(64, metadata={"instrument": "HiFi"}) == "HiFi"

    def test_96_returns_emu(self):
        assert detect_instrument(96) == "EMU"

    def test_unknown_returns_none(self):
        assert detect_instrument(0) is None
        assert detect_instrument(32) is None
        assert detect_instrument(128) is None

    def test_95_returns_none(self):
        assert detect_instrument(95) is None

    def test_65_returns_none(self):
        assert detect_instrument(65) is None


# ---------------------------------------------------------------------------
# get_instrument_layout
# ---------------------------------------------------------------------------


class TestGetInstrumentLayout:
    def test_returns_layout_for_each_known_name(self):
        for name in INSTRUMENT_NAMES:
            layout = get_instrument_layout(name)
            assert isinstance(layout, InstrumentLayout)
            assert layout.name == name

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_instrument_layout("UnknownInstrument")

    def test_same_object_returned_on_repeated_calls(self):
        a = get_instrument_layout("HiFi")
        b = get_instrument_layout("HiFi")
        assert a is b  # cached singleton

    def test_freeform_flame_alias_returns_flame_layout(self):
        assert get_instrument_layout("LMU_BULKMUSR_FLAME").name == "FLAME"


# ---------------------------------------------------------------------------
# FLAME layout
# ---------------------------------------------------------------------------


class TestFlameLayout:
    @pytest.fixture(scope="class")
    def layout(self):
        return get_instrument_layout("FLAME")

    def test_n_detectors(self, layout):
        assert layout.n_detectors == 8

    def test_plan_view(self, layout):
        assert layout.view == "plan"
        assert len(layout.banks) == 1
        assert layout.reference_arrows

    def test_detector_ids_and_labels(self, layout):
        labels = {seg.detector_id: seg.label for seg in layout.all_segments}
        assert labels == {
            1: "Forward",
            2: "Backward",
            3: "Right",
            4: "Left",
            5: "R_F",
            6: "R_B",
            7: "L_F",
            8: "L_B",
        }
        assert {seg.shape for seg in layout.all_segments} == {"rectangle"}

    def test_side_bank_rectangles_share_height_with_wide_middle(self, layout):
        segments = {seg.detector_id: seg for seg in layout.all_segments}
        assert segments[3].height == pytest.approx(segments[5].height)
        assert segments[3].height == pytest.approx(segments[6].height)
        assert segments[4].height == pytest.approx(segments[7].height)
        assert segments[4].height == pytest.approx(segments[8].height)
        assert segments[3].width > segments[5].width * 2
        assert segments[4].width > segments[7].width * 2

    def test_spin_arrow_points_toward_backward_detector(self, layout):
        spin_arrow = next(arrow for arrow in layout.reference_arrows if "spin" in arrow.label)
        backward = next(seg for seg in layout.all_segments if seg.detector_id == 2)
        assert spin_arrow.end[0] < spin_arrow.start[0]
        assert abs(spin_arrow.end[1] - backward.y_center) < backward.height / 2

    def test_longitudinal_preset(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.forward_group == 1
        assert preset.backward_group == 2
        assert preset.groups[1].detector_ids == (1,)
        assert preset.groups[2].detector_ids == (2,)

    def test_transverse_preset(self, layout):
        preset = layout.presets["Transverse"]
        assert preset.forward_group == 1
        assert preset.backward_group == 2
        assert set(preset.groups[1].detector_ids) == {3, 5, 6}
        assert set(preset.groups[2].detector_ids) == {4, 7, 8}


# ---------------------------------------------------------------------------
# HiFi layout
# ---------------------------------------------------------------------------


class TestHiFiLayout:
    @pytest.fixture(scope="class")
    def layout(self):
        return get_instrument_layout("HiFi")

    def test_n_detectors(self, layout):
        assert layout.n_detectors == 64

    def test_two_banks(self, layout):
        assert len(layout.banks) == 2

    def test_bank_names(self, layout):
        assert layout.banks[0].name == "Forward"
        assert layout.banks[1].name == "Backward"

    def test_forward_bank_detector_count(self, layout):
        assert len(layout.banks[0].segments) == 32

    def test_backward_bank_detector_count(self, layout):
        assert len(layout.banks[1].segments) == 32

    def test_forward_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[0].segments}
        assert ids == set(range(1, 33))

    def test_backward_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[1].segments}
        assert ids == set(range(33, 65))

    def test_no_duplicate_ids(self, layout):
        all_ids = _all_detector_ids(layout)
        assert len(all_ids) == len(set(all_ids))

    def test_all_ids_covered(self, layout):
        assert set(_all_detector_ids(layout)) == set(range(1, 65))

    def test_forward_detector_1_angle(self, layout):
        # Detector 1 centred at 270° (bottom of disc, counterclockwise numbering)
        seg = next(s for s in layout.banks[0].segments if s.detector_id == 1)
        assert abs(seg.angle_center_deg - 270.0) < 1e-6

    def test_forward_detector_2_angle(self, layout):
        # Each step is +11.25°
        seg = next(s for s in layout.banks[0].segments if s.detector_id == 2)
        assert abs(seg.angle_center_deg - (270.0 + 11.25)) < 1e-6

    def test_backward_detector_33_angle(self, layout):
        # Detector 33 centred at 264.375°
        seg = next(s for s in layout.banks[1].segments if s.detector_id == 33)
        assert abs(seg.angle_center_deg - 264.375) < 1e-6

    def test_segment_half_width(self, layout):
        for seg in layout.banks[0].segments:
            assert abs(seg.angle_half_width_deg - 11.25 / 2) < 1e-6

    def test_default_preset_is_longitudinal(self, layout):
        assert layout.default_preset_name == "Longitudinal"

    def test_presets_present(self, layout):
        assert "Longitudinal" in layout.presets
        assert "Transverse (Vector)" in layout.presets

    def test_longitudinal_forward_group(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.forward_group == 1
        assert preset.backward_group == 2
        assert set(preset.groups[1].detector_ids) == set(range(1, 33))
        assert set(preset.groups[2].detector_ids) == set(range(33, 65))

    def test_longitudinal_group_names(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.groups[1].name == "Forward"
        assert preset.groups[2].name == "Backward"

    def test_transverse_lr_left_detectors(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        left_ids = set(preset.groups[1].detector_ids)
        assert set(range(5, 14)).issubset(left_ids)
        assert set(range(52, 61)).issubset(left_ids)

    def test_transverse_lr_right_detectors(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        right_ids = set(preset.groups[2].detector_ids)
        assert set(range(21, 30)).issubset(right_ids)
        assert set(range(36, 45)).issubset(right_ids)

    def test_transverse_tb_top_detectors(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        top_ids = set(preset.groups[3].detector_ids)
        assert set(range(13, 22)).issubset(top_ids)
        assert set(range(44, 53)).issubset(top_ids)

    def test_transverse_tb_bottom_detectors(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        bottom_ids = set(preset.groups[4].detector_ids)
        assert set(range(1, 6)).issubset(bottom_ids)
        assert set(range(29, 37)).issubset(bottom_ids)
        assert set(range(60, 65)).issubset(bottom_ids)

    def test_transverse_vector_declares_two_projections(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        labels = [p.label for p in preset.projections]
        assert labels == ["Left-Right", "Top-Bottom"]
        pairs = derive_projection_pairs(
            {gid: list(g.detector_ids) for gid, g in preset.groups.items()},
            projections=[p.to_payload() for p in preset.projections],
        )
        assert pairs == {"Left-Right": (1, 2), "Top-Bottom": (3, 4)}

    def test_preset_detector_ids_in_valid_range(self, layout):
        for preset in layout.presets.values():
            for gdef in preset.groups.values():
                for det_id in gdef.detector_ids:
                    assert 1 <= det_id <= 64, f"{det_id} out of range for HiFi"


# ---------------------------------------------------------------------------
# HAL-9500 layout
# ---------------------------------------------------------------------------


class TestHALLayout:
    @pytest.fixture(scope="class")
    def layout(self):
        return get_instrument_layout("HAL")

    def test_registered_in_instrument_names(self):
        assert "HAL" in INSTRUMENT_NAMES

    def test_n_detectors(self, layout):
        # MV + 8 forward + 8 backward.
        assert layout.n_detectors == 17

    def test_radial_view(self, layout):
        assert layout.view == "radial"

    def test_two_banks(self, layout):
        assert len(layout.banks) == 2
        assert layout.banks[0].name == "Forward"
        assert layout.banks[1].name == "Backward"

    def test_forward_bank_contains_mv_and_eight_wedges(self, layout):
        ids = {s.detector_id for s in layout.banks[0].segments}
        assert ids == {1} | set(range(2, 10))  # MV + F1..F8

    def test_backward_bank_eight_wedges(self, layout):
        ids = {s.detector_id for s in layout.banks[1].segments}
        assert ids == set(range(10, 18))  # B1..B8

    def test_no_duplicate_ids(self, layout):
        all_ids = _all_detector_ids(layout)
        assert len(all_ids) == len(set(all_ids))

    def test_all_ids_covered(self, layout):
        assert set(_all_detector_ids(layout)) == set(range(1, 18))

    def test_detector_labels_map_to_histogram_order(self, layout):
        # Histogram order is MV, F1..F8, B1..B8 -> detector N is histogram N-1.
        by_id = {s.detector_id: s for s in layout.all_segments}
        assert by_id[1].label == "MV"
        assert [by_id[i].label for i in range(2, 10)] == [f"F{k}" for k in range(1, 9)]
        assert [by_id[i].label for i in range(10, 18)] == [f"B{k}" for k in range(1, 9)]

    def test_mv_is_central_disc(self, layout):
        mv = next(s for s in layout.all_segments if s.detector_id == 1)
        assert mv.r_inner < 0.3
        # Effectively a full disc so the radial hit-test selects it anywhere.
        assert mv.angle_half_width_deg > 179.0

    def test_forward_wedges_are_octagonal(self, layout):
        fwd = [s for s in layout.banks[0].segments if s.detector_id != 1]
        assert len(fwd) == 8
        for seg in fwd:
            # Each detector is a rectangular bar along one octagon edge.
            assert seg.shape == "rectangle"
            assert seg.width > 0 and seg.height > 0
        # F1 at the top, numbering clockwise by 45 degrees.
        f1 = next(s for s in fwd if s.label == "F1")
        f5 = next(s for s in fwd if s.label == "F5")
        assert abs(f1.angle_center_deg - 90.0) < 1e-6
        # F1's bar sits at the top (positive y, ~zero x); F5 at the bottom.
        assert f1.y_center > 0.5 and abs(f1.x_center) < 1e-6
        assert f5.y_center < -0.5 and abs(f5.x_center) < 1e-6
        # F5 is diametrically opposite F1 (musrfit opposed pair).
        assert abs(((f5.angle_center_deg - f1.angle_center_deg) % 360.0) - 180.0) < 1e-6

    def test_default_preset_is_longitudinal(self, layout):
        assert layout.default_preset_name == "Longitudinal"

    def test_presets_present(self, layout):
        assert set(layout.presets) == {
            "Longitudinal",
            "Transverse (opposed pairs)",
            "Per-octant",
        }

    def test_longitudinal_forward_ring_vs_backward_ring(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.forward_group == 1
        assert preset.backward_group == 2
        assert set(preset.groups[1].detector_ids) == set(range(2, 10))
        assert set(preset.groups[2].detector_ids) == set(range(10, 18))

    def test_transverse_opposed_pair_default(self, layout):
        preset = layout.presets["Transverse (opposed pairs)"]
        # Each forward detector is its own group; default pair is F1 (group 1)
        # vs F5 (group 5), which are 180 degrees apart.
        assert preset.groups[1].detector_ids == (2,)
        assert preset.groups[5].detector_ids == (6,)
        assert preset.forward_group == 1
        assert preset.backward_group == 5

    def test_per_octant_combines_forward_and_backward_wedge(self, layout):
        preset = layout.presets["Per-octant"]
        # Octant k pairs F_k with B_k (same azimuth).
        assert preset.groups[1].detector_ids == (2, 10)
        assert preset.groups[8].detector_ids == (9, 17)

    def test_preset_detector_ids_in_valid_range(self, layout):
        for preset in layout.presets.values():
            for gdef in preset.groups.values():
                for det_id in gdef.detector_ids:
                    assert 1 <= det_id <= 17, f"{det_id} out of range for HAL"


# ---------------------------------------------------------------------------
# MuSR layout
# ---------------------------------------------------------------------------


class TestMuSRLayout:
    @pytest.fixture(scope="class")
    def layout(self):
        return get_instrument_layout("MuSR")

    def test_n_detectors(self, layout):
        assert layout.n_detectors == 64

    def test_two_banks(self, layout):
        assert len(layout.banks) == 2

    def test_bank_names(self, layout):
        assert layout.banks[0].name == "Backward"
        assert layout.banks[1].name == "Forward"

    def test_backward_ring_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[0].segments}
        assert ids == set(range(1, 33))

    def test_forward_ring_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[1].segments}
        assert ids == set(range(33, 65))

    def test_no_duplicate_ids(self, layout):
        all_ids = _all_detector_ids(layout)
        assert len(all_ids) == len(set(all_ids))

    def test_all_ids_covered(self, layout):
        assert set(_all_detector_ids(layout)) == set(range(1, 65))

    def test_backward_detector_1_angle(self, layout):
        # Detector 1 starts at 230.625°
        seg = next(s for s in layout.banks[0].segments if s.detector_id == 1)
        assert abs(seg.angle_center_deg - 230.625) < 1e-6

    def test_forward_detector_33_angle(self, layout):
        # Detector 33 starts at 309.375°
        seg = next(s for s in layout.banks[1].segments if s.detector_id == 33)
        assert abs(seg.angle_center_deg - 309.375) < 1e-6

    def test_longitudinal_preset(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.forward_group == 2
        assert preset.backward_group == 1
        assert set(preset.groups[1].detector_ids) == set(range(1, 33))
        assert set(preset.groups[2].detector_ids) == set(range(33, 65))

    def test_transverse_top_bottom_preset(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        top = set(preset.groups[1].detector_ids)
        bottom = set(preset.groups[2].detector_ids)
        # Verify expected detectors
        assert set(range(17, 25)).issubset(top)
        assert set(range(49, 57)).issubset(top)
        assert set(range(1, 9)).issubset(bottom)
        assert set(range(33, 41)).issubset(bottom)

    def test_transverse_fwd_bwd_preset(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        fwd = set(preset.groups[3].detector_ids)
        bwd = set(preset.groups[4].detector_ids)
        assert set(range(9, 17)).issubset(fwd)
        assert set(range(57, 65)).issubset(fwd)
        assert set(range(25, 33)).issubset(bwd)
        assert set(range(41, 49)).issubset(bwd)

    def test_transverse_vector_declares_two_projections(self, layout):
        preset = layout.presets["Transverse (Vector)"]
        labels = [p.label for p in preset.projections]
        assert labels == ["Top-Bottom", "Fwd-Back"]
        pairs = derive_projection_pairs(
            {gid: list(g.detector_ids) for gid, g in preset.groups.items()},
            projections=[p.to_payload() for p in preset.projections],
        )
        assert pairs == {"Top-Bottom": (1, 2), "Fwd-Back": (3, 4)}

    def test_preset_detector_ids_in_valid_range(self, layout):
        for preset in layout.presets.values():
            for gdef in preset.groups.values():
                for det_id in gdef.detector_ids:
                    assert 1 <= det_id <= 64, f"{det_id} out of range for MuSR"


# ---------------------------------------------------------------------------
# EMU layout
# ---------------------------------------------------------------------------


class TestEMULayout:
    @pytest.fixture(scope="class")
    def layout(self):
        return get_instrument_layout("EMU")

    def test_n_detectors(self, layout):
        assert layout.n_detectors == 96

    def test_two_banks(self, layout):
        assert len(layout.banks) == 2

    def test_bank_names(self, layout):
        assert layout.banks[0].name == "Forward"
        assert layout.banks[1].name == "Backward"

    def test_forward_bank_detector_count(self, layout):
        assert len(layout.banks[0].segments) == 48

    def test_backward_bank_detector_count(self, layout):
        assert len(layout.banks[1].segments) == 48

    def test_forward_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[0].segments}
        assert ids == set(range(1, 49))

    def test_backward_detector_ids(self, layout):
        ids = {s.detector_id for s in layout.banks[1].segments}
        assert ids == set(range(49, 97))

    def test_no_duplicate_ids(self, layout):
        all_ids = _all_detector_ids(layout)
        assert len(all_ids) == len(set(all_ids))

    def test_all_ids_covered(self, layout):
        assert set(_all_detector_ids(layout)) == set(range(1, 97))

    def test_three_rings_per_sector(self, layout):
        """Every azimuthal sector in each bank has exactly 3 ring segments."""
        for bank in layout.banks:
            sectors: dict[int, list] = {}
            for seg in bank.segments:
                sectors.setdefault(seg.sector_index, []).append(seg)
            for s_idx, segs in sectors.items():
                assert len(segs) == 3, f"Bank {bank.name} sector {s_idx} has {len(segs)} segments"

    def test_sixteen_sectors_per_bank(self, layout):
        for bank in layout.banks:
            sectors = {s.sector_index for s in bank.segments}
            assert sectors == set(range(16))

    def test_sector_0_angle(self, layout):
        """Sector 0 should be at 90° (12 o'clock)."""
        for bank in layout.banks:
            s0_segs = [s for s in bank.segments if s.sector_index == 0]
            for seg in s0_segs:
                assert abs(seg.angle_center_deg - 90.0) < 1e-6

    def test_forward_bank_numbering_formula(self, layout):
        """Forward bank: inner = 1+3s, middle = 2+3s, outer = 3+3s."""
        forward_segs = {s.detector_id: s for s in layout.banks[0].segments}
        for s in range(16):
            inner_id = 1 + 3 * s
            middle_id = 2 + 3 * s
            outer_id = 3 + 3 * s
            assert inner_id in forward_segs
            assert middle_id in forward_segs
            assert outer_id in forward_segs
            # ring_index 0 = inner, 1 = middle, 2 = outer
            assert forward_segs[inner_id].ring_index == 0
            assert forward_segs[middle_id].ring_index == 1
            assert forward_segs[outer_id].ring_index == 2

    def test_backward_bank_numbering_formula(self, layout):
        """Backward bank: inner = 49+3s, middle = 50+3s, outer = 51+3s."""
        backward_segs = {s.detector_id: s for s in layout.banks[1].segments}
        for s in range(16):
            inner_id = 49 + 3 * s
            middle_id = 50 + 3 * s
            outer_id = 51 + 3 * s
            assert inner_id in backward_segs
            assert middle_id in backward_segs
            assert outer_id in backward_segs

    def test_longitudinal_preset(self, layout):
        preset = layout.presets["Longitudinal"]
        assert preset.forward_group == 1
        assert preset.backward_group == 2
        assert set(preset.groups[1].detector_ids) == set(range(1, 49))
        assert set(preset.groups[2].detector_ids) == set(range(49, 97))

    def test_vector_polarization_six_groups(self, layout):
        preset = layout.presets["Vector Polarization"]
        assert len(preset.groups) == 6

    def test_vector_polarization_group_names(self, layout):
        preset = layout.presets["Vector Polarization"]
        names = {g.name for g in preset.groups.values()}
        assert "Pz Forward" in names
        assert "Pz Backward" in names
        assert "Px Left" in names
        assert "Px Right" in names
        assert "Py Top" in names
        assert "Py Bottom" in names

    def test_vector_polarization_pz_covers_all(self, layout):
        """Pz Forward + Backward should cover all 96 detectors once."""
        preset = layout.presets["Vector Polarization"]
        pz_fwd = next(g for g in preset.groups.values() if g.name == "Pz Forward")
        pz_bwd = next(g for g in preset.groups.values() if g.name == "Pz Backward")
        combined = set(pz_fwd.detector_ids) | set(pz_bwd.detector_ids)
        assert combined == set(range(1, 97))
        assert len(pz_fwd.detector_ids) + len(pz_bwd.detector_ids) == 96  # no overlap

    def test_vector_polarization_no_duplicates_within_group(self, layout):
        preset = layout.presets["Vector Polarization"]
        for gdef in preset.groups.values():
            assert len(gdef.detector_ids) == len(set(gdef.detector_ids)), (
                f"Duplicate detector IDs in group '{gdef.name}'"
            )

    def test_vector_polarization_transverse_groups_each_have_48_detectors(self, layout):
        """Each transverse group spans half of both banks: 8 sectors × 3 rings × 2 = 48."""
        preset = layout.presets["Vector Polarization"]
        for gid, gdef in preset.groups.items():
            if gdef.name in {"Py Top", "Py Bottom", "Px Right", "Px Left"}:
                assert len(gdef.detector_ids) == 48, (
                    f"Group '{gdef.name}' has {len(gdef.detector_ids)} detectors, expected 48"
                )

    def test_vector_polarization_transverse_group_overlap_pattern(self, layout):
        """Py and Px halves overlap by one octant (24 detectors) per adjacent pair."""
        preset = layout.presets["Vector Polarization"]
        py_top = set(next(g.detector_ids for g in preset.groups.values() if g.name == "Py Top"))
        py_bottom = set(
            next(g.detector_ids for g in preset.groups.values() if g.name == "Py Bottom")
        )
        px_right = set(next(g.detector_ids for g in preset.groups.values() if g.name == "Px Right"))
        px_left = set(next(g.detector_ids for g in preset.groups.values() if g.name == "Px Left"))

        assert py_top.isdisjoint(py_bottom)
        assert px_right.isdisjoint(px_left)

        # Top-right and top-left overlaps are each one octant across both banks.
        assert len(py_top & px_right) == 24
        assert len(py_top & px_left) == 24
        # Bottom-right and bottom-left overlaps are each one octant across both banks.
        assert len(py_bottom & px_right) == 24
        assert len(py_bottom & px_left) == 24

    def test_vector_polarization_pz_groups_overlap_transverse_groups(self, layout):
        """Pz groups intentionally overlap Px/Py groups for vector decomposition."""
        preset = layout.presets["Vector Polarization"]
        pz_forward = set(
            next(g.detector_ids for g in preset.groups.values() if g.name == "Pz Forward")
        )
        py_up = set(next(g.detector_ids for g in preset.groups.values() if g.name == "Py Top"))
        px_right = set(next(g.detector_ids for g in preset.groups.values() if g.name == "Px Right"))

        assert pz_forward & py_up
        assert pz_forward & px_right

    def test_vector_polarization_all_ids_in_range(self, layout):
        preset = layout.presets["Vector Polarization"]
        for gdef in preset.groups.values():
            for det_id in gdef.detector_ids:
                assert 1 <= det_id <= 96, f"{det_id} out of range for EMU"

    def test_preset_detector_ids_in_valid_range(self, layout):
        for preset in layout.presets.values():
            for gdef in preset.groups.values():
                for det_id in gdef.detector_ids:
                    assert 1 <= det_id <= 96


# ---------------------------------------------------------------------------
# DetectorSegment helpers
# ---------------------------------------------------------------------------


class TestDetectorSegment:
    def test_angle_start_end(self):
        seg = DetectorSegment(
            detector_id=1,
            sector_index=0,
            ring_index=0,
            angle_center_deg=45.0,
            angle_half_width_deg=5.625,
            r_inner=0.28,
            r_outer=1.0,
        )
        assert abs(seg.angle_start_deg - 39.375) < 1e-9
        assert abs(seg.angle_end_deg - 50.625) < 1e-9


# ---------------------------------------------------------------------------
# InstrumentLayout helpers
# ---------------------------------------------------------------------------


class TestInstrumentLayoutHelpers:
    def test_all_segments_returns_all(self):
        layout = get_instrument_layout("HiFi")
        segs = layout.all_segments
        assert len(segs) == 64

    def test_default_preset_name_is_longitudinal(self):
        for name in INSTRUMENT_NAMES:
            layout = get_instrument_layout(name)
            assert layout.default_preset_name == "Longitudinal"


# ---------------------------------------------------------------------------
# Asymmetry projections
# ---------------------------------------------------------------------------


class TestAsymmetryProjections:
    def test_emu_vector_preset_declares_three_projections(self):
        preset = get_instrument_layout("EMU").presets["Vector Polarization"]
        labels = [p.label for p in preset.projections]
        assert labels == ["P_x", "P_y", "P_z"]

    def test_emu_projections_point_at_correct_group_pairs(self):
        preset = get_instrument_layout("EMU").presets["Vector Polarization"]
        by_label = {p.label: p for p in preset.projections}
        # Pz = forward/backward banks (groups 1/2); Py = top/bottom (3/4);
        # Px = left/right (5/6).
        assert (by_label["P_z"].forward_group, by_label["P_z"].backward_group) == (1, 2)
        assert (by_label["P_y"].forward_group, by_label["P_y"].backward_group) == (3, 4)
        assert (by_label["P_x"].forward_group, by_label["P_x"].backward_group) == (5, 6)

    def test_emu_projections_carry_semantic_tints(self):
        preset = get_instrument_layout("EMU").presets["Vector Polarization"]
        for proj in preset.projections:
            assert proj.tint == PROJECTION_TINTS[proj.label]

    def test_longitudinal_presets_declare_no_projections(self):
        for name in INSTRUMENT_NAMES:
            preset = get_instrument_layout(name).presets["Longitudinal"]
            assert preset.projections == ()

    def test_projection_to_payload_roundtrips(self):
        proj = AsymmetryProjection("P_x", 5, 6, alpha=1.2, tint="#abcdef")
        assert proj.to_payload() == {
            "label": "P_x",
            "forward_group": 5,
            "backward_group": 6,
            "alpha": 1.2,
            "tint": "#abcdef",
        }

    def test_to_payload_omits_absent_tint(self):
        assert "tint" not in AsymmetryProjection("X", 1, 2).to_payload()


class TestDeriveProjectionPairs:
    def _emu_groups(self):
        preset = get_instrument_layout("EMU").presets["Vector Polarization"]
        groups = {gid: list(gdef.detector_ids) for gid, gdef in preset.groups.items()}
        names = {gid: gdef.name for gid, gdef in preset.groups.items()}
        return preset, groups, names

    def test_declared_projections_resolve_in_declaration_order(self):
        preset, groups, _names = self._emu_groups()
        pairs = derive_projection_pairs(groups, None, preset.projections)
        assert list(pairs) == ["P_x", "P_y", "P_z"]
        assert pairs == {"P_x": (5, 6), "P_y": (3, 4), "P_z": (1, 2)}

    def test_legacy_name_fallback_matches_declared(self):
        """With no declaration, the legacy group-name path reproduces the pairs."""
        preset, groups, names = self._emu_groups()
        declared = derive_projection_pairs(groups, names, preset.projections)
        legacy = derive_projection_pairs(groups, names, None)
        assert declared == legacy

    def test_legacy_fallback_is_all_or_nothing(self):
        _preset, groups, names = self._emu_groups()
        # Drop one required group: the legacy path returns nothing.
        groups.pop(5)
        names.pop(5)
        assert derive_projection_pairs(groups, names, None) == {}

    def test_declared_projections_return_resolvable_subset(self):
        """A two-projection (TF-style) declaration resolves just those pairs."""
        groups = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8]}
        projections = [
            AsymmetryProjection("Top-Bottom", 1, 2),
            AsymmetryProjection("Fwd-Back", 3, 4),
        ]
        pairs = derive_projection_pairs(groups, None, projections)
        assert pairs == {"Top-Bottom": (1, 2), "Fwd-Back": (3, 4)}

    def test_declared_projection_with_missing_group_is_skipped(self):
        groups = {1: [1, 2], 2: [3, 4]}
        projections = [
            AsymmetryProjection("A", 1, 2),
            AsymmetryProjection("B", 1, 9),  # group 9 absent
        ]
        assert derive_projection_pairs(groups, None, projections) == {"A": (1, 2)}

    def test_declared_projection_with_empty_group_is_skipped(self):
        groups = {1: [1, 2], 2: []}
        projections = [AsymmetryProjection("A", 1, 2)]
        assert derive_projection_pairs(groups, None, projections) == {}

    def test_payload_dict_projections_accepted(self):
        """Projections persisted as plain dicts resolve like dataclasses."""
        groups = {1: [1], 2: [2]}
        projections = [{"label": "A", "forward_group": 1, "backward_group": 2}]
        assert derive_projection_pairs(groups, None, projections) == {"A": (1, 2)}

    def test_empty_or_non_vector_grouping_returns_empty(self):
        assert derive_projection_pairs({}, None, None) == {}
        assert derive_projection_pairs(None, None, None) == {}
        assert derive_projection_pairs({1: [1], 2: [2]}, {1: "Forward", 2: "Backward"}) == {}
