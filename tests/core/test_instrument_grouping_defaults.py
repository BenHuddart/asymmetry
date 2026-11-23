"""Absent-table defaults require a declared map and explicit detector orientation."""

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.instruments.catalog import PROFILES
from asymmetry.core.instruments.isis_musr import PROFILE as MUSR
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    UnmappedAcquisition,
    assess_instrument,
)


def test_absent_longitudinal_musr_default_has_independent_manual_membership_and_sign():
    from asymmetry.core.instruments.defaults import absent_file_grouping_default

    selected = assess_instrument(64, metadata={"instrument": "MuSR"}).selection
    preset = absent_file_grouping_default(selected, {"detector_orientation": "Longitudinal"})
    # MuSR Manual §2.6.1 p.12: F33–64/B1–32, independent of applied-field state.
    assert preset.name == "Longitudinal"
    assert preset.forward_group == 2
    assert preset.backward_group == 1
    forward = preset.groups[preset.forward_group].detector_ids
    backward = preset.groups[preset.backward_group].detector_ids
    assert forward == tuple(range(33, 65))
    assert backward == tuple(range(1, 33))
    counts = np.repeat([1000.0, 3000.0], 32)
    f = counts[np.asarray(forward) - 1].sum()
    b = counts[np.asarray(backward) - 1].sum()
    assert (f, b) == (96000, 32000)
    assert (f - b) / (f + b) == 0.5
    assert not MUSR.defaults.apply_default_preset_on_load
    assert MUSR.identity.revision == MUSR.review.revision == "2"


@pytest.mark.parametrize("orientation", [None, "", "Transverse", "Unknown", "L", "LF"])
def test_missing_or_noncanonical_detector_orientation_never_selects_default(orientation):
    from asymmetry.core.instruments.defaults import absent_file_grouping_default

    selected = assess_instrument(64, metadata={"instrument": "MuSR"}).selection
    metadata = {"field_state": "LF", "field_direction": "Longitudinal"}
    if orientation is not None:
        metadata["detector_orientation"] = orientation
    assert absent_file_grouping_default(selected, metadata) is None


@pytest.mark.parametrize("field_state", [None, "LF", "TF", "ZF"])
def test_detector_orientation_default_does_not_invent_an_applied_field_requirement(field_state):
    from asymmetry.core.instruments.defaults import absent_file_grouping_default

    selected = assess_instrument(64, metadata={"instrument": "MuSR"}).selection
    metadata = {"detector_orientation": "Longitudinal", "field_state": field_state}
    assert absent_file_grouping_default(selected, metadata).name == "Longitudinal"


@pytest.mark.parametrize(
    "selected",
    [
        None,
        InstrumentSelection("isis.musr", UnmappedAcquisition("incompatible"), "2", "metadata"),
        InstrumentSelection("isis.musr", UnmappedAcquisition("undeclared"), "2", "metadata"),
        InstrumentSelection("isis.musr", "removed", "old", "legacy"),
        InstrumentSelection("missing.instrument", "removed", "old", "legacy"),
        assess_instrument(64, metadata={"instrument": "HiFi"}).selection,
        assess_instrument(96, metadata={"instrument": "EMU"}).selection,
        assess_instrument(63, metadata={"instrument": "MuSR"}).selection,
        assess_instrument(64, metadata={"instrument": "MuSR", "facility": "PSI"}).selection,
    ],
)
def test_undeclared_or_incompatible_identity_and_map_never_select_default(selected):
    from asymmetry.core.instruments.defaults import absent_file_grouping_default

    assert absent_file_grouping_default(selected, {"detector_orientation": "Longitudinal"}) is None


@pytest.mark.parametrize(
    "acquisition,orientation,preset",
    [
        ("", "Longitudinal", "Longitudinal"),
        ("nexus64", "", "Longitudinal"),
        ("nexus64", "LF", "Longitudinal"),
        ("nexus64", "Longitudinal", ""),
    ],
)
def test_absent_grouping_rule_rejects_incomplete_or_noncanonical_facts(
    acquisition, orientation, preset
):
    from asymmetry.core.instruments.schema import AbsentGroupingDefault

    with pytest.raises(ValueError, match="Absent-file grouping"):
        AbsentGroupingDefault(acquisition, orientation, preset)


@pytest.mark.parametrize(
    "acquisition,preset", [("unavailable", "Longitudinal"), ("nexus64", "unknown")]
)
def test_profile_rejects_absent_default_referencing_unavailable_acquisition_or_preset(
    acquisition, preset
):
    from asymmetry.core.instruments.schema import AbsentGroupingDefault, Defaults

    with pytest.raises(ValueError, match="Absent-file grouping"):
        replace(
            MUSR,
            defaults=Defaults(
                absent_file_grouping=AbsentGroupingDefault(acquisition, "Longitudinal", preset)
            ),
        )


def test_only_declared_instrument_receives_absent_grouping_default():
    assert {p.identity.key for p in PROFILES if p.defaults.absent_file_grouping is not None} == {
        "isis.musr"
    }
