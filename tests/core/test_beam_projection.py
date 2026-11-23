"""Independent coordinate controls for geometry-relative beam projection."""

import math
from dataclasses import asdict, replace

import pytest

from asymmetry.core.instruments.catalog import PROFILES
from asymmetry.core.instruments.geometry import (
    BeamDirection,
    BeamProjection,
    ViewOrientation,
    project_beam,
)
from asymmetry.core.instruments.resolve import layout_from_profile
from tests._instrument_profiles import synthetic_profiles


@pytest.mark.parametrize(
    "right,up,toward,expected",
    [
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0), ((1, 0), 0)),  # Side: beam right.
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), ((0, 0), 1)),  # Looking upstream.
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1), ((0, 0), -1)),  # Looking downstream.
        ((0, 1, 0), (0, 0, 1), (1, 0, 0), ((0, 1), 0)),  # Rotated side: beam up.
    ],
)
def test_independent_side_upstream_downstream_and_rotated_signs(right, up, toward, expected):
    # In this fixed arrangement frame +z is downstream, +y is up, and +x is transverse.
    result = project_beam(BeamDirection((0, 0, 1)), ViewOrientation(right, up, toward))
    assert result == BeamProjection(*expected)


def test_oblique_rotation_preserves_signed_components_and_direction_only_magnitude():
    orientation = ViewOrientation((1, 1, 0), (-1, 1, 0), (0, 0, 1))
    result = project_beam(BeamDirection((3, 0, 0)), orientation)
    assert result.screen == pytest.approx((math.sqrt(0.5), -math.sqrt(0.5)))
    assert result.toward == 0
    assert BeamDirection((1e308, 1e308, 1e308)).vector == pytest.approx((1 / math.sqrt(3),) * 3)
    assert BeamDirection((1e-320, 0, 0)).vector == (1, 0, 0)


def test_explicit_mirrored_basis_is_supported_without_claiming_a_physical_rotation():
    beam = BeamDirection((0, 0, 1))
    frame = ViewOrientation((1, 0, 0), (0, 1, 0), (0, 0, -1))
    assert project_beam(beam, frame) == BeamProjection((0, 0), -1)


@pytest.mark.parametrize("beam", [(1, 0, 0), (0, 0, 1)])
def test_missing_toward_axis_retains_unknown_sense_even_for_zero_screen_projection(beam):
    result = project_beam(BeamDirection(beam), ViewOrientation((1, 0, 0), (0, 1, 0)))
    assert result.screen == beam[:2]
    assert result.toward is None


@pytest.mark.parametrize("missing", ["beam", "orientation", "both"])
def test_missing_beam_or_view_orientation_has_no_projection(missing):
    beam = None if missing in {"beam", "both"} else BeamDirection((1, 0, 0))
    frame = None if missing in {"orientation", "both"} else ViewOrientation((1, 0, 0), (0, 1, 0))
    assert project_beam(beam, frame) is None


@pytest.mark.parametrize(
    "vector", [(), (1, 0), (1, 0, 0, 0), (0, 0, 0), (math.nan, 0, 1), (0, math.inf, 0)]
)
def test_invalid_beam_directions_are_rejected_at_construction(vector):
    with pytest.raises(ValueError, match="direction"):
        BeamDirection(vector)


@pytest.mark.parametrize(
    "right,up,toward",
    [
        ((0, 0, 0), (0, 1, 0), None),
        ((1, 0, 0), (math.nan, 0, 0), None),
        ((1, 0, 0), (0, 1, 0), (0, 0, math.inf)),
        ((1, 0, 0), (1, 0, 0), None),
        ((1, 0, 0), (1, 1, 0), None),
        ((1, 0, 0), (0, 1, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 1, 1)),
    ],
)
def test_nonfinite_or_degenerate_view_bases_are_rejected(right, up, toward):
    with pytest.raises(ValueError, match="direction|perpendicular"):
        ViewOrientation(right, up, toward)


def test_actual_profile_resolution_projects_each_view_without_changing_grouping_or_segments():
    profile = synthetic_profiles()["multiview"]
    original = profile.detectors.acquisitions[0]
    orientations = (
        ViewOrientation((0, 0, 1), (0, 1, 0)),
        ViewOrientation((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    )
    acquisition = replace(
        original,
        beam=BeamDirection((0, 0, 1)),
        views=tuple(
            replace(view, orientation=frame)
            for view, frame in zip(original.views, orientations, strict=True)
        ),
    )
    declared = replace(profile, detectors=replace(profile.detectors, acquisitions=(acquisition,)))
    before, after = (layout_from_profile(p, "test") for p in (profile, declared))
    assert [bank.beam for bank in after.banks] == [
        BeamProjection((1, 0), None),
        BeamProjection((0, 0), -1),
    ]
    assert after.presets == before.presets
    assert after.all_segments == before.all_segments
    assert after.reference_arrows == before.reference_arrows
    assert after.apply_default_preset_on_load == before.apply_default_preset_on_load
    unknown = replace(
        acquisition, views=tuple(replace(v, orientation=None) for v in original.views)
    )
    unknown_profile = replace(
        profile, detectors=replace(profile.detectors, acquisitions=(unknown,))
    )
    assert all(bank.beam is None for bank in layout_from_profile(unknown_profile, "test").banks)


def test_only_explicitly_declared_profiles_gain_beam_authority():
    for profile in PROFILES:
        for config in profile.detectors.acquisitions:
            declared = profile.identity.key in {"psi.gps", "psi.flame", "psi.hal9500"}
            assert (config.beam is not None) is declared
            assert all((view.orientation is not None) is declared for view in config.views)
            layout = layout_from_profile(profile, config.key)
            assert all((bank.beam is not None) is declared for bank in layout.banks)
            assert layout.reference_arrows == config.reference_arrows


def test_direction_records_are_frozen_and_do_not_keep_mutable_input():
    from dataclasses import FrozenInstanceError

    direction = [0, 0, 2]
    beam = BeamDirection(direction)
    direction[2] = -2
    assert asdict(beam) == {"vector": (0.0, 0.0, 1.0)}
    with pytest.raises(FrozenInstanceError):
        beam.vector = (0, 0, -1)


def test_profile_boundaries_require_typed_beam_and_view_orientation():
    original = synthetic_profiles()["multiview"].detectors.acquisitions[0]
    with pytest.raises(ValueError, match="typed BeamDirection"):
        replace(original, beam=(0, 0, 1))
    with pytest.raises(ValueError, match="typed ViewOrientation"):
        replace(original.views[0], orientation=((1, 0, 0), (0, 1, 0)))
