"""Readout identity, ordinal/sign fidelity, and independent schematic anchors."""

from dataclasses import replace

import pytest

from asymmetry.core.instruments import FLAME, GPS, HAL9500
from asymmetry.core.instruments.geometry import (
    BankRow,
    Composite,
    Directional,
    Occurrence,
    expand_arrangement,
)
from asymmetry.core.instruments.numbering import numbering_template
from asymmetry.core.instruments.resolve import layout_from_profile


@pytest.mark.parametrize(
    "profile, acquisition, keys",
    (
        (GPS, "bin_combined", "F B U D R L"),
        (GPS, "root_split", "F B UB UF DB DF RB RF LB LF M"),
        (GPS, "root_combined_split", "F B U D R L UB UF DB DF RB RF LB LF M"),
        (FLAME, "eight_channel", "F B R L RF RB LF LB"),
        (HAL9500, "full17", "MV F1 F2 F3 F4 F5 F6 F7 F8 B1 B2 B3 B4 B5 B6 B7 B8"),
    ),
)
def test_exact_readout_maps_and_slot_identity_survive_view_reordering(profile, acquisition, keys):
    slots = numbering_template(profile, acquisition)
    assert [slot.channel for slot in slots] == keys.split()
    assert [slot.ordinal for slot in slots] == list(range(1, len(slots) + 1))
    assert [slot.id for slot in slots] == [
        f"{profile.identity.key}/{acquisition}/{key}" for key in keys.split()
    ]
    config = next(config for config in profile.detectors.acquisitions if config.key == acquisition)
    changed = replace(
        profile,
        detectors=replace(
            profile.detectors,
            acquisitions=tuple(
                replace(c, views=tuple(reversed(c.views))) if c.key == acquisition else c
                for c in profile.detectors.acquisitions
            ),
        ),
    )
    assert numbering_template(changed, acquisition) == slots
    layout = layout_from_profile(profile, acquisition)
    assert sorted(s.detector_id for s in layout.active_segments) == list(range(1, len(slots) + 1))
    assert {s.detector_id: s.label for s in layout.active_segments} == {
        c.ordinal: c.label for c in config.channels
    }
    expected_aux = {"M", "MV"} & set(keys.split())
    assert {s.detector_id for s in layout.auxiliary} == {
        slot.ordinal for slot in slots if slot.channel in expected_aux
    }
    assert all(
        s.detector_id not in {a.detector_id for a in layout.auxiliary}
        for b in layout.banks
        for s in b.segments
    )


def test_same_gps_channel_key_is_qualified_by_acquisition():
    split = {s.channel: s for s in numbering_template(GPS, "root_split")}
    both = {s.channel: s for s in numbering_template(GPS, "root_combined_split")}
    assert (split["UB"].ordinal, both["UB"].ordinal) == (3, 7)
    assert split["UB"].id != both["UB"].id
    assert both["U"].physical == ("UB", "UF")
    assert both["F"].physical == both["M"].physical == ()
    assert (
        len({s.id for c in GPS.detectors.acquisitions for s in numbering_template(GPS, c.key)})
        == 32
    )


@pytest.mark.parametrize(
    "acquisition,up,down,right,left",
    (
        ("bin_combined", (3,), (4,), (5,), (6,)),
        ("root_split", (3, 4), (5, 6), (7, 8), (9, 10)),
        ("root_combined_split", (7, 8), (9, 10), (11, 12), (13, 14)),
    ),
)
def test_gps_ordered_analysis_legs_and_separate_acquired_roads(acquisition, up, down, right, left):
    layout = layout_from_profile(GPS, acquisition)
    longitudinal = layout.presets["Longitudinal"]
    assert (longitudinal.forward_group, longitudinal.backward_group) == (2, 1)
    assert [g.detector_ids for g in longitudinal.groups.values()] == [(1,), (2,)]
    vector = layout.presets["Transverse (Vector)"]
    assert [g.detector_ids for g in vector.groups.values()] == [up, down, right, left]
    assert (vector.forward_group, vector.backward_group) == (1, 2)
    spin = layout.presets["Spin-rotated (B+U/F+D)"]
    assert [g.detector_ids for g in spin.groups.values()] == [(2, *up), (1, *down)]
    assert (spin.forward_group, spin.backward_group) == (1, 2)
    wep = layout.presets["WEP (spin-rotated)"]
    assert [g.detector_ids for g in wep.groups.values()] == [(1,), (2,), up, down]
    assert [(p.label, p.forward_group, p.backward_group, p.alpha) for p in wep.projections] == [
        ("FB", 2, 1, 0.75),
        ("UD", 3, 4, 1.0),
    ]
    assert [bank.beam.screen for bank in layout.banks] == [(1.0, 0.0)] * 2
    assert all(bank.beam.toward is None for bank in layout.banks)
    by_id = {s.detector_id: s for s in layout.active_segments}
    assert all(by_id[i].y_center > 0 for i in up + left)
    assert all(by_id[i].y_center < 0 for i in down + right)
    if len(up) == 2:
        for back, front in (up, down, right, left):
            assert by_id[back].x_center < 0 < by_id[front].x_center
    assert not any(s.shape.startswith("endon") for s in layout.all_segments)


def test_flame_flankers_and_hal_tangent_clockwise_rectangles():
    flame = layout_from_profile(FLAME, "eight_channel")
    by_id = {s.detector_id: s for s in flame.active_segments}
    assert by_id[8].x_center < by_id[4].x_center < by_id[7].x_center
    assert by_id[6].x_center < by_id[3].x_center < by_id[5].x_center
    assert by_id[4].y_center > 0 > by_id[3].y_center
    assert flame.reference_arrows == ()
    hal = layout_from_profile(HAL9500, "full17")
    for bank, first in zip(hal.banks, (2, 10), strict=True):
        assert [s.detector_id for s in bank.segments] == list(range(first, first + 8))
        assert [s.angle_center_deg for s in bank.segments] == [90, 45, 0, 315, 270, 225, 180, 135]
        assert bank.segments[0].rotation_deg == 0
        assert bank.segments[0].y_center > 0
        assert bank.segments[2].x_center > 0
        assert bank.beam.screen == (0.0, 0.0) and bank.beam.toward is None
    assert all(
        "Axial viewing sense is unknown" in view.convention
        for view in HAL9500.detectors.acquisitions[0].views
    )


def test_directional_rows_keep_order_across_opposed_banks_and_reject_bad_data():
    row = BankRow((Occurrence(channel="B"), Occurrence(channel="F")), (1, 2))
    for direction, sign in (("up", 1), ("down", -1)):
        a, b = expand_arrangement(Directional(direction, (row,), 3))
        assert a.x < b.x and a.y == b.y == sign * 3
        assert a.shape.width == 1 and b.shape.width == 2
    for constructor in (
        lambda: BankRow((), ()),
        lambda: BankRow(row.targets, (1,)),
        lambda: BankRow(row.targets, (1, float("nan"))),
        lambda: Directional("oblique", (row,)),
        lambda: Directional("up", (row,), gap=-1),
        lambda: Composite(()),
        lambda: Composite((Composite((Directional("up", (row,)),)),)),
    ):
        with pytest.raises(ValueError):
            constructor()
    with pytest.raises(ValueError, match="Unknown acquisition"):
        numbering_template(GPS, "not-a-readout")


def test_auxiliary_validation_prevents_duplicate_or_unavailable_editable_readouts():
    config = GPS.detectors.acquisitions[-1]
    for auxiliary in (
        (Occurrence(channel="missing"),),
        (Occurrence(channel="F"),),
        (Occurrence(channel="M", editable=False),),
    ):
        with pytest.raises(ValueError, match="Auxiliary|exactly one editable"):
            replace(
                GPS,
                detectors=replace(
                    GPS.detectors,
                    acquisitions=(
                        *GPS.detectors.acquisitions[:-1],
                        replace(config, auxiliary=auxiliary),
                    ),
                ),
            )


@pytest.mark.parametrize(
    "direction,axis", (("up", "y"), ("down", "y"), ("left", "x"), ("right", "x"))
)
def test_unequal_thickness_rows_keep_the_declared_edge_gap(direction, axis):
    rows = (
        BankRow((Occurrence(channel="a"),), (1,), thickness=0.5),
        BankRow((Occurrence(channel="b"),), (1,), thickness=2),
    )
    a, b = expand_arrangement(Directional(direction, rows, distance=3, gap=0.15))
    a_center, b_center = (a.x, b.x) if axis == "x" else (a.y, b.y)
    assert abs(a_center) == 3
    assert abs(b_center) == pytest.approx(4.4)
    assert (abs(b_center) - 1) - (abs(a_center) + 0.25) == pytest.approx(0.15)
