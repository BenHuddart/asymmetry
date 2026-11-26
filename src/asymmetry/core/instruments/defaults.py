"""Resolve profile grouping defaults for supported file conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from asymmetry.core.instruments.catalog import PROFILES
from asymmetry.core.instruments.layout import PresetGrouping
from asymmetry.core.instruments.resolve import layout_from_profile

if TYPE_CHECKING:
    from asymmetry.core.instruments.resolution import InstrumentSelection


def absent_file_grouping_default(
    selection: InstrumentSelection | None, metadata: dict
) -> PresetGrouping | None:
    """Read a compatible map's conditional preset, using normalized detector orientation.

    The acquisition boundary supplies an unambiguous selection and calls this
    only for a fresh unconfigured run whose file grouping table is absent.
    Applied-field state does not establish detector-bank orientation.
    """
    return file_grouping_default(selection, metadata, "absent_file_grouping")


def file_grouping_default(
    selection: InstrumentSelection | None,
    metadata: dict,
    condition: Literal["absent_file_grouping", "legacy_v1_zero_grouping"],
) -> PresetGrouping | None:
    """Use the preset belonging to a supported acquisition and detector orientation."""
    if selection is None:
        return None
    profile = next((p for p in PROFILES if p.identity.key == selection.profile_key), None)
    if profile is None:
        return None
    rule = (
        profile.defaults.absent_file_grouping
        if condition == "absent_file_grouping"
        else profile.defaults.legacy_v1_zero_grouping
    )
    if (
        rule is None
        or selection.acquisition != rule.acquisition
        or metadata.get("detector_orientation") != rule.detector_orientation
    ):
        return None
    return layout_from_profile(profile, rule.acquisition).presets[rule.preset]


def retained_grouping_provenance(requested: dict, acquired: dict) -> dict:
    """Retain a captured default's attribution only while its scientific grouping is unchanged."""
    keys = ("groups", "forward_group", "backward_group")
    unchanged = all(requested.get(key) == acquired.get(key) for key in keys) and all(
        requested.get(key, []) == acquired.get(key, [])
        for key in ("projections", "excluded_detectors")
    )
    return (
        {"grouping_provenance": acquired["grouping_provenance"]}
        if "grouping_provenance" in acquired and unchanged
        else {}
    )
