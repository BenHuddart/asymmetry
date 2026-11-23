"""Readout sets qualify acquired channels without duplicating physical detectors."""

import pytest

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.core.instruments import GPS
from asymmetry.core.instruments.geometry import Explicit, Occurrence, Placement, Rectangle, View
from asymmetry.core.instruments.numbering import numbering_template
from asymmetry.core.instruments.readouts import layout_for_readout_set, readout_presets
from asymmetry.core.instruments.schema import (
    Acquisition,
    Channel,
    CombinedReadout,
    ConditionalContribution,
    ReadoutSet,
)
from tests._instrument_profiles import synthetic_profile


@pytest.mark.parametrize(
    "layout,expected",
    [
        ("GPS", [("hardware_combined", (1, 2, 3, 4, 5, 6))]),
        ("GPS-RD", [("individual", tuple(range(1, 12)))]),
        (
            "GPS-RD15",
            [
                ("individual", (1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15)),
                ("hardware_combined", (1, 2, 3, 4, 5, 6, 15)),
            ],
        ),
    ],
)
def test_explicit_maps_and_filtered_views_keep_real_ordinals(layout, expected):
    source = get_instrument_layout(layout)
    assert [(s.key, s.detector_ids) for s in source.readout_sets] == expected
    for key, ordinals in expected:
        view = layout_for_readout_set(source, key)
        assert sorted(s.detector_id for s in view.active_segments) == list(ordinals)
        assert view.n_detectors == source.n_detectors
        assert len(view.banks) == 2
        assert len(view.auxiliary) == (1 if source.n_detectors > 6 else 0)
    config = next(
        c for c in GPS.detectors.acquisitions if (c.layout_key or GPS.identity.layout_key) == layout
    )
    assert len(numbering_template(GPS, config.key)) == source.n_detectors
    for readouts in config.readout_sets:
        assert any(
            e.fact == f"readout_set.{config.key}.{readouts.key}" for e in GPS.sources.evidence
        )


def test_set_specific_presets_have_literal_memberships_and_unchanged_signs():
    source = get_instrument_layout("GPS-RD15")
    individual = readout_presets(source, "individual")
    combined = readout_presets(source, "hardware_combined")
    assert set(individual) == {
        "Longitudinal",
        "Transverse (Vector)",
        "Spin-rotated (B+U/F+D)",
        "WEP (spin-rotated)",
    }
    assert set(combined) == {
        "Longitudinal",
        "Transverse (Vector, combined)",
        "Spin-rotated (combined)",
        "WEP (combined)",
    }
    for name, expected in (
        ("WEP (spin-rotated)", ((7, 8), (9, 10))),
        ("WEP (combined)", ((3,), (4,))),
    ):
        preset = source.presets[name]
        assert (preset.groups[3].detector_ids, preset.groups[4].detector_ids) == expected
        assert (preset.forward_group, preset.backward_group) == (2, 1)
        assert [
            (p.label, p.forward_group, p.backward_group, p.alpha) for p in preset.projections
        ] == [("FB", 2, 1, 0.75), ("UD", 3, 4, 1.0)]
    assert source.readout_sets[1].notice == "May include Mobile counts"
    with pytest.raises(ValueError, match="unavailable"):
        layout_for_readout_set(get_instrument_layout("GPS"), "individual")


def declaration(config_overrides):
    channels = tuple(Channel(key, index + 1) for index, key in enumerate(("a", "b", "c")))
    view = View(
        "view",
        "View",
        "plan",
        "Synthetic",
        Explicit(
            tuple(
                Placement(Occurrence(channel=c.key), Rectangle(1, 1), i * 2, 1)
                for i, c in enumerate(channels)
            )
        ),
    )
    return synthetic_profile((Acquisition("test", "Test", channels, (view,), **config_overrides),))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"readout_sets": (ReadoutSet("all", "All", ("a", "b"), "Pair"),)}, "cover exactly"),
        (
            {"readout_sets": (ReadoutSet("all", "All", ("a", "b", "c"), "Missing"),)},
            "default preset",
        ),
        ({"combined_readouts": (CombinedReadout("a", ("a", "b")),)}, "non-combined"),
        ({"combined_readouts": (CombinedReadout("a", ("b", "missing")),)}, "non-combined"),
        (
            {"conditional_contributions": (ConditionalContribution("c", ("a",), "Auxiliary"),)},
            "combined destinations",
        ),
    ],
)
def test_invalid_relationships_fail_at_profile_boundary(overrides, match):
    with pytest.raises(ValueError, match=match):
        declaration(overrides)


def test_conditional_destinations_do_not_assert_common_mobile_events():
    config = next(c for c in GPS.detectors.acquisitions if c.key == "root_combined_split")
    assert [(r.channel, r.constituents) for r in config.combined_readouts] == [
        ("U", ("UB", "UF")),
        ("D", ("DB", "DF")),
        ("R", ("RB", "RF")),
        ("L", ("LB", "LF")),
    ]
    assert config.conditional_contributions == (ConditionalContribution("M", ("R", "L"), "Mobile"),)
    safe = declaration({"combined_readouts": (CombinedReadout("a", ("b", "c")),)})
    assert safe.detectors.acquisitions[0].combined_readouts[0].channel == "a"
