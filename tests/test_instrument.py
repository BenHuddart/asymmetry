"""Tests for the instrument layout definitions in core/instrument.py."""

from __future__ import annotations

import pytest

from asymmetry.core.instrument import (
    INSTRUMENT_NAMES,
    DetectorSegment,
    InstrumentLayout,
    PresetGrouping,
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

    def test_psi_metadata_does_not_select_isis_layout(self):
        assert detect_instrument(64, metadata={"facility": "PSI", "instrument": "HIFI"}) is None
        assert (
            detect_instrument(64, metadata={"psi_format": "psi-bin", "instrument": "MuSR"}) is None
        )

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
        assert "Transverse (Left\u2013Right)" in layout.presets
        assert "Transverse (Top\u2013Bottom)" in layout.presets

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
        preset = layout.presets["Transverse (Left\u2013Right)"]
        left_ids = set(preset.groups[1].detector_ids)
        assert set(range(5, 14)).issubset(left_ids)
        assert set(range(52, 61)).issubset(left_ids)

    def test_transverse_lr_right_detectors(self, layout):
        preset = layout.presets["Transverse (Left\u2013Right)"]
        right_ids = set(preset.groups[2].detector_ids)
        assert set(range(21, 30)).issubset(right_ids)
        assert set(range(36, 45)).issubset(right_ids)

    def test_transverse_tb_top_detectors(self, layout):
        preset = layout.presets["Transverse (Top\u2013Bottom)"]
        top_ids = set(preset.groups[1].detector_ids)
        assert set(range(13, 22)).issubset(top_ids)
        assert set(range(44, 53)).issubset(top_ids)

    def test_transverse_tb_bottom_detectors(self, layout):
        preset = layout.presets["Transverse (Top\u2013Bottom)"]
        bottom_ids = set(preset.groups[2].detector_ids)
        assert set(range(1, 6)).issubset(bottom_ids)
        assert set(range(29, 37)).issubset(bottom_ids)
        assert set(range(60, 65)).issubset(bottom_ids)

    def test_preset_detector_ids_in_valid_range(self, layout):
        for preset in layout.presets.values():
            for gdef in preset.groups.values():
                for det_id in gdef.detector_ids:
                    assert 1 <= det_id <= 64, f"{det_id} out of range for HiFi"


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
        preset = layout.presets["Transverse (Top\u2013Bottom)"]
        top = set(preset.groups[1].detector_ids)
        bottom = set(preset.groups[2].detector_ids)
        # Verify expected detectors
        assert set(range(17, 25)).issubset(top)
        assert set(range(49, 57)).issubset(top)
        assert set(range(1, 9)).issubset(bottom)
        assert set(range(33, 41)).issubset(bottom)

    def test_transverse_fwd_bwd_preset(self, layout):
        preset = layout.presets["Transverse (Forward\u2013Backward)"]
        fwd = set(preset.groups[1].detector_ids)
        bwd = set(preset.groups[2].detector_ids)
        assert set(range(9, 17)).issubset(fwd)
        assert set(range(57, 65)).issubset(fwd)
        assert set(range(25, 33)).issubset(bwd)
        assert set(range(41, 49)).issubset(bwd)

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
