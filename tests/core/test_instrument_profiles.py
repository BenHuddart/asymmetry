"""Profile migration fidelity and independent facility-manual anchors."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from asymmetry.core.instruments import ALC, DOLLY, FLAME, GPD, GPS, HAL9500, LEM, LTF
from asymmetry.core.instruments.isis_emu import PROFILE
from asymmetry.core.instruments.resolve import layout_from_profile


def test_emu_resolved_layout_preserves_every_existing_channel_and_preset():
    expected = json.loads(
        (Path(__file__).parents[1] / "data/instruments/emu-layout.json").read_text()
    )
    actual = json.loads(json.dumps(asdict(layout_from_profile(PROFILE, "nexus96"))))
    assert actual == expected


def test_emu_manual_triplets_and_bank_grouping_are_independent_anchors():
    # EMU User Guide, §8.1, printed p.33: clockwise from the top viewed upstream.
    layout = layout_from_profile(PROFILE, "nexus96")
    by_id = {segment.detector_id: segment for segment in layout.active_segments}
    for ids, angle in (
        ((1, 2, 3), 90.0),
        ((13, 14, 15), 0.0),
        ((25, 26, 27), 270.0),
        ((49, 50, 51), 90.0),
    ):
        assert [by_id[i].angle_center_deg for i in ids] == [angle] * 3
        assert [by_id[i].ring_index for i in ids] == [0, 1, 2]
    preset = layout.presets["Longitudinal"]
    assert preset.groups[preset.forward_group].detector_ids == tuple(range(1, 49))
    assert preset.groups[preset.backward_group].detector_ids == tuple(range(49, 97))


def test_metadata_only_identity_needs_no_invented_acquisition():
    import pytest

    from tests._instrument_profiles import synthetic_profile

    profile = synthetic_profile()
    assert profile.detectors.acquisitions == ()
    assert profile.presets == ()
    layout = layout_from_profile(profile, n_histograms=7)
    assert layout.name == "Generic"
    assert [element.detector_id for element in layout.active_segments] == list(range(1, 8))
    assert layout.presets == {}
    with pytest.raises(ValueError, match="Unknown acquisition"):
        layout_from_profile(profile, "misspelled", n_histograms=7)


def test_synthetic_arrangements_have_independent_coordinate_anchors():
    import pytest

    from tests._instrument_profiles import synthetic_profiles

    profiles = synthetic_profiles()
    concentric = layout_from_profile(profiles["concentric"], "test")
    assert [
        (seg.detector_id, seg.angle_center_deg, seg.ring_index)
        for seg in concentric.active_segments
    ] == [(1, 90.0, 0), (2, 90.0, 1), (3, 0.0, 0), (4, 0.0, 1)]
    grid = layout_from_profile(profiles["grid"], "test")
    assert [(seg.detector_id, seg.x_center, seg.y_center) for seg in grid.active_segments] == [
        (2, -1.0, 2.0),
        (1, 1.0, 2.0),
        (4, -1.0, -1.0),
        (3, 1.0, -1.0),
    ]
    ring = layout_from_profile(profiles["ring_rectangles"], "test")
    assert [seg.rotation_deg for seg in ring.active_segments] == [90.0, 180.0, 270.0, 360.0]
    assert ring.active_segments[1].x_center == pytest.approx(0.0, abs=1e-15)
    assert ring.active_segments[1].y_center == pytest.approx(0.8)
    explicit = layout_from_profile(profiles["explicit"], "test")
    assert [seg.rotation_deg for seg in explicit.active_segments] == [30.0, -45.0]
    assert explicit.all_segments[-1].detector_id is None
    assert explicit.all_segments[-1].read_only


def test_combined_split_configurations_share_hardware_without_count_equations():
    from tests._instrument_profiles import synthetic_profiles

    profile = synthetic_profiles()["combined_split"]
    assert [len(config.channels) for config in profile.detectors.acquisitions] == [6, 11, 15]
    assert len(profile.detectors.elements) == 12
    combined = profile.detectors.acquisitions[-1]
    assert next(channel for channel in combined.channels if channel.key == "U").physical == (
        "UB",
        "UF",
    )
    assert next(channel for channel in combined.channels if channel.key == "F").physical == ()
    assert next(channel for channel in combined.channels if channel.key == "M").physical == ()


def test_profile_construction_rejects_invalid_cross_references_and_evidence():
    from dataclasses import replace

    import pytest

    from asymmetry.core.instruments.geometry import Explicit, Occurrence, Placement, Rectangle

    config = PROFILE.detectors.acquisitions[0]
    with pytest.raises(ValueError, match="Duplicate channel ordinals"):
        channels = (
            config.channels[0],
            replace(config.channels[1], ordinal=1),
            *config.channels[2:],
        )
        replace(
            PROFILE,
            detectors=replace(
                PROFILE.detectors, acquisitions=(replace(config, channels=channels),)
            ),
        )
    with pytest.raises(ValueError, match="unknown acquired channel"):
        bad_view = replace(
            config.views[0],
            arrangement=Explicit((Placement(Occurrence(channel="missing"), Rectangle(1, 1)),)),
        )
        replace(
            PROFILE,
            detectors=replace(
                PROFILE.detectors,
                acquisitions=(replace(config, views=(bad_view, config.views[1])),),
            ),
        )
    with pytest.raises(ValueError, match="exactly one editable"):
        replace(
            PROFILE,
            detectors=replace(
                PROFILE.detectors, acquisitions=(replace(config, views=(config.views[0],)),)
            ),
        )
    with pytest.raises(ValueError, match="Evidence must cover"):
        replace(PROFILE, sources=replace(PROFILE.sources, evidence=PROFILE.sources.evidence[:-1]))
    with pytest.raises(ValueError, match="this profile revision"):
        replace(PROFILE, review=replace(PROFILE.review, revision="wrong"))
    with pytest.raises(ValueError, match="Hardware context cannot be editable"):
        Occurrence(hardware="h1")


def test_view_construction_rejects_primitives_the_renderer_cannot_draw():
    import pytest

    from asymmetry.core.instruments.geometry import (
        EndOn,
        Explicit,
        Occurrence,
        Placement,
        View,
        Wedge,
    )

    with pytest.raises(ValueError, match="Plan views cannot contain wedges"):
        View(
            "bad",
            "Bad",
            "plan",
            "Synthetic",
            Explicit((Placement(Occurrence(channel="a"), Wedge(0.1, 1.0, 30.0)),)),
        )
    with pytest.raises(ValueError, match="Radial views cannot contain end-on markers"):
        View(
            "bad",
            "Bad",
            "radial",
            "Synthetic",
            Explicit((Placement(Occurrence(hardware="h", editable=False), EndOn()),)),
        )


def test_emu_recognition_through_preset_reduces_independent_count_sentinels():
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
    from asymmetry.core.instrument import detect_instrument
    from asymmetry.core.project.profiles import ProfileFingerprint, profile_fingerprint_for_run
    from asymmetry.core.transform.t0 import source_is_pulsed
    from asymmetry.core.workflow import apply_grouping_preset

    run = Run(
        run_number=42,
        histograms=[
            Histogram(np.full(8, number, dtype=float), 0.1, 0, 0, 7) for number in range(1, 97)
        ],
        metadata={"instrument": "EMU"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "good_frames": 100000,
            "background_correction": False,
        },
    )
    data = MuonDataset(np.array([]), np.array([]), np.array([]), metadata=run.metadata, run=run)
    assert detect_instrument(len(run.histograms), metadata=run.metadata) == "EMU"
    assert profile_fingerprint_for_run(run) == ProfileFingerprint("isis.emu", 96, "nexus96")
    result = apply_grouping_preset(data, "Longitudinal")
    assert result.run.grouping["groups"] == {1: list(range(1, 49)), 2: list(range(49, 97))}
    # Manual bank memberships, independent arithmetic sums: 1+...+48=1176;49+...+96=3480.
    np.testing.assert_allclose(
        result.asymmetry, 100 * (1176 - 3480) / (1176 + 3480), rtol=0, atol=1e-12
    )
    assert PROFILE.source.pulsed is True
    assert source_is_pulsed(run.metadata) is None
    assert result.run.histograms is run.histograms
    assert run.grouping["groups"] == {1: [1], 2: [2]}


def test_profile_import_and_resolution_do_not_import_gui_or_plotting():
    import subprocess
    import sys
    import textwrap

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.startswith(("PySide6","matplotlib","asymmetry.gui")):
                    raise AssertionError(f"Pure profile imported {fullname}")
        sys.meta_path.insert(0,BlockFrontend())
        from asymmetry.core.instruments.isis_emu import PROFILE
        from asymmetry.core.instruments.resolve import layout_from_profile
        from asymmetry.core.instrument import detect_instrument,preset_grouping_for_run
        assert detect_instrument(96,metadata={"instrument":"EMU"}) == "EMU"
        layout=layout_from_profile(PROFILE,"nexus96")
        assert len(layout.active_segments)==96
        groups=preset_grouping_for_run(layout,"Longitudinal",n_histograms=96)
        assert groups=={1:list(range(1,49)),2:list(range(49,97))}
        assert not any(name.startswith(("PySide6","matplotlib","asymmetry.gui")) for name in sys.modules)
    """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_geometry_dimensions_and_profile_frozen_state_are_explicit():
    from dataclasses import FrozenInstanceError

    import pytest

    from asymmetry.core.instruments.geometry import EndOn, Rectangle, Wedge
    from asymmetry.core.instruments.schema import Dimension

    for constructor in (
        lambda: Wedge(0.5, 0.1, 10),
        lambda: Rectangle(0, 1),
        lambda: EndOn(radius=-1),
        lambda: Dimension("distance", 1, ""),
    ):
        with pytest.raises(ValueError):
            constructor()
    with pytest.raises(FrozenInstanceError):
        PROFILE.identity.name = "Changed"
    assert all(
        isinstance(config.channels, tuple) and isinstance(config.views, tuple)
        for config in PROFILE.detectors.acquisitions
    )


def test_acquired_histogram_numbers_are_integers_at_construction():
    from dataclasses import replace

    import pytest

    config = PROFILE.detectors.acquisitions[0]
    channels = (replace(config.channels[0], ordinal=1.5), *config.channels[1:])
    with pytest.raises(ValueError, match="positive integer"):
        replace(
            PROFILE,
            detectors=replace(
                PROFILE.detectors, acquisitions=(replace(config, channels=channels),)
            ),
        )


def test_acquisition_preserves_legacy_variant_identity_and_declared_arrows():
    from dataclasses import replace

    import pytest

    from asymmetry.core.instruments.layout import ReferenceArrow
    from tests._instrument_profiles import synthetic_profiles

    profile = synthetic_profiles()["multiview"]
    original = profile.detectors.acquisitions[0]
    arrows = (ReferenceArrow("Synthetic direction", (-2.0, 0.0), (2.0, 0.0)),)
    config = replace(original, layout_key="Fixture-split", reference_arrows=arrows)
    variant = replace(profile, detectors=replace(profile.detectors, acquisitions=(config,)))
    layout = layout_from_profile(variant, "test")
    assert layout.name == "Fixture-split"
    assert layout.display_name == "Fixture"
    assert layout.reference_arrows == arrows
    inherited = layout_from_profile(profile, "test")
    assert inherited.name == "Fixture"
    assert inherited.display_name is None
    assert inherited.reference_arrows == ()
    with pytest.raises(ValueError, match="layout key"):
        replace(original, layout_key=" ")
    with pytest.raises(ValueError, match="finite"):
        replace(original, reference_arrows=(ReferenceArrow("Bad", (float("nan"), 0), (1, 0)),))


@pytest.mark.parametrize(
    "profile, acquisition, legacy",
    (
        (GPS, "bin_combined", "GPS"),
        (GPS, "root_split", "GPS-RD"),
        (GPS, "root_combined_split", "GPS-RD15"),
        (FLAME, "eight_channel", "FLAME"),
        (HAL9500, "full17", "HAL"),
    ),
)
def test_psi_layouts_preserve_channel_geometry_and_scientific_presets(profile, acquisition, legacy):
    # End-on width0 means radius0.16 in the legacy renderer; snapshots encode that radius explicitly.
    expected = json.loads(
        (Path(__file__).parents[1] / f"data/instruments/{legacy.lower()}-layout.json").read_text()
    )
    actual = json.loads(json.dumps(asdict(layout_from_profile(profile, acquisition))))
    assert actual == expected


@pytest.mark.parametrize("acquisition", ("bin_combined", "root_split", "root_combined_split"))
def test_gps_paper_sign_and_configuration_limits_are_independent_anchors(acquisition):
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
    from asymmetry.core.workflow import apply_grouping_preset

    config = next(config for config in GPS.detectors.acquisitions if config.key == acquisition)
    assert next(channel for channel in config.channels if channel.key == "F").physical == ()
    if acquisition != "bin_combined":
        assert next(channel for channel in config.channels if channel.key == "M").physical == ()
    # Amato et al. (2017), Eq2: A=(B-alpha F)/(B+alpha F); choose B30,F10.
    run = Run(
        42,
        [
            Histogram(np.full(8, 30.0 if channel.key == "B" else 10.0), 0.1, 0, 0, 7)
            for channel in config.channels
        ],
        metadata={"facility": "PSI", "instrument": "GPS"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "good_frames": 1000,
            "background_correction": False,
        },
    )
    dataset = MuonDataset(np.array([]), np.array([]), np.array([]), metadata=run.metadata, run=run)
    result = apply_grouping_preset(dataset, "Longitudinal")
    assert result.run.grouping["forward_group"] == 2
    np.testing.assert_allclose(result.asymmetry, 50.0, atol=1e-12)
    assert result.run.grouping["alpha"] == 1.0
    # Figure2/TableI distinguish hardware and acquired roads, not independent counts.
    assert len(GPS.detectors.elements) == 12
    assert all("independent" in channel.relationship for channel in config.channels)


@pytest.mark.parametrize("profile", (LTF, GPD, DOLLY, ALC, LEM))
def test_psi_metadata_only_identities_preserve_generic_unknown_geometry(profile):
    assert profile.identity.facility == "PSI"
    assert profile.detectors.acquisitions == profile.presets == ()
    layout = layout_from_profile(profile, n_histograms=5)
    assert layout.name == "Generic"
    assert len(layout.active_segments) == 5
    assert layout.presets == {}


@pytest.mark.parametrize(
    "raw, facility, expected",
    (
        ("HAL", "PSI", "HAL"),
        ("hal", "PSI", "HAL"),
        ("halite", "PSI", None),
        ("HIFI", "PSI", "HAL"),
        ("HiFi", "ISIS", "HiFi"),
        ("HAL9500", "PSI", "HAL"),
        ("HAL", "ISIS", None),
    ),
)
def test_psi_hal_exact_names_preserve_facility_scope_and_reject_substrings(raw, facility, expected):
    from asymmetry.core.instrument import detect_instrument

    assert detect_instrument(17, metadata={"facility": facility, "instrument": raw}) == expected
