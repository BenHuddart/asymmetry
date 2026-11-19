"""Canonical decisions preserve evidence and distinguish identity from mapping."""

from dataclasses import replace

import pytest

from asymmetry.core.instruments.resolution import InstrumentSelection


@pytest.mark.parametrize(
    "metadata,count,key,acquisition",
    [
        ({"instrument": "EMU"}, 96, "isis.emu", "nexus96"),
        ({"instrument": "HiFi"}, 64, "isis.hifi", "nexus64"),
        ({"instrument": "MuSR"}, 64, "isis.musr", "nexus64"),
        ({"facility": "PSI", "instrument": "HIFI"}, 17, "psi.hal9500", "full17"),
        ({"facility": "PSI", "instrument": "HAL"}, 17, "psi.hal9500", "full17"),
        ({"instrument": "GPS"}, 11, "psi.gps", "root_split"),
    ],
)
def test_known_evidence_resolves_stable_instrument_and_configuration(
    metadata, count, key, acquisition
):
    from asymmetry.core.instruments.resolution import assess_instrument

    result = assess_instrument(count, metadata=metadata)
    assert result.selection.profile_key == key
    assert result.selection.acquisition == acquisition


def test_metadata_only_and_incompatible_mapping_are_distinct():
    from asymmetry.core.instruments.resolution import UnmappedAcquisition, assess_instrument

    identity = assess_instrument(4, metadata={"facility": "PSI", "instrument": "LTF"})
    mismatch = assess_instrument(4, metadata={"instrument": "EMU"})
    assert identity.selection.profile_key == "psi.ltf"
    assert identity.selection.acquisition == UnmappedAcquisition("undeclared")
    assert mismatch.selection.profile_key == "isis.emu"
    assert mismatch.selection.acquisition == UnmappedAcquisition("incompatible")


def test_contradictory_evidence_is_not_resolved_by_registry_order():
    from asymmetry.core.instruments.resolution import assess_instrument

    result = assess_instrument(64, metadata={"instrument": "HiFi", "instrument_name": "MuSR"})
    assert result.selection is None
    assert result.unresolved_reason == "conflict"
    assert {candidate.profile_key for candidate in result.candidates} == {"isis.hifi", "isis.musr"}
    assert ("instrument", "HiFi") in result.evidence
    assert ("instrument_name", "MuSR") in result.evidence


def test_candidate_identity_excludes_provenance_and_revision():
    from asymmetry.core.instruments.resolution import InstrumentResolution, assess_instrument

    result = assess_instrument(96, metadata={"instrument": "EMU"})
    candidate = result.selection
    with pytest.raises(ValueError, match="distinct"):
        InstrumentResolution(
            (candidate, replace(candidate, profile_revision="future", origin="filename")),
            result.evidence,
            "conflict",
        )


def test_anonymous_count_never_becomes_identity():
    from asymmetry.core.instruments.resolution import assess_instrument

    for count in (64, 96):
        result = assess_instrument(count, metadata={})
        assert result.selection is None
        assert result.unresolved_reason == "absent"


def test_explicit_recorded_unknown_is_a_choice_not_missing_evidence():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.instruments.resolution import (
        RecordedInstrumentChoice,
        UnresolvedInstrumentIdentity,
        assess_instrument,
        resolve_instrument,
    )

    run = Run(metadata={"instrument": "EMU"})
    evidence = assess_instrument(0, metadata=run.metadata).evidence
    choice = RecordedInstrumentChoice(
        UnresolvedInstrumentIdentity("Unlisted saved spectrometer"), evidence
    )
    run.instrument_choice = choice
    context = resolve_instrument(run)
    assert not context.requires_choice
    assert context.selection is None
    assert context.recorded.selection.label == "Unlisted saved spectrometer"
    assert RecordedInstrumentChoice.from_dict(choice.to_dict()) == choice


def test_accepted_choice_is_not_reprompted_but_new_conflicting_evidence_is_visible():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.instruments.resolution import (
        RecordedInstrumentChoice,
        assess_instrument,
        resolve_instrument,
    )

    run = Run(metadata={"instrument": "HiFi", "instrument_name": "MuSR"})
    observed = assess_instrument(0, metadata=run.metadata)
    selected = next(
        candidate for candidate in observed.candidates if candidate.profile_key == "isis.hifi"
    )
    run.instrument_choice = RecordedInstrumentChoice(
        replace(selected, origin="user"), observed.evidence
    )
    assert not resolve_instrument(run).requires_choice
    assert resolve_instrument(run).selection.profile_key == "isis.hifi"
    run.metadata = {"instrument": "EMU"}
    assert resolve_instrument(run).requires_choice
    assert run.instrument_choice.selection.profile_key == "isis.hifi"


def test_profile_revision_change_does_not_replace_a_captured_choice():
    from asymmetry.core.data.dataset import Run
    from asymmetry.core.instruments.resolution import (
        RecordedInstrumentChoice,
        assess_instrument,
        resolve_instrument,
    )

    run = Run(metadata={"instrument": "EMU"})
    observed = assess_instrument(0, metadata=run.metadata)
    historical = replace(observed.selection, profile_revision="old", origin="legacy")
    run.instrument_choice = RecordedInstrumentChoice(historical, ())
    context = resolve_instrument(run)
    assert not context.requires_choice
    assert context.selection is historical


def test_new_contrary_group_labels_are_part_of_the_accepted_evidence():
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.instruments.resolution import RecordedInstrumentChoice, resolve_instrument

    run = Run(
        histograms=[Histogram(np.ones(4), 0.1, 0, 0, 3) for _ in range(8)],
        metadata={"instrument": "EMU"},
    )
    observed = resolve_instrument(run).observed
    run.instrument_choice = RecordedInstrumentChoice(observed.selection, observed.evidence)
    run.metadata["group_names"] = {
        1: "Forward",
        2: "Backward",
        3: "Left",
        4: "Right",
        5: "RF",
        6: "RB",
        7: "LF",
        8: "LB",
    }
    context = resolve_instrument(run)
    assert context.observed.unresolved_reason == "conflict"
    assert context.requires_choice
    assert dict(context.observed.evidence)["group_names"] == str(run.metadata["group_names"])


@pytest.mark.parametrize(
    "key,configuration", [("missing.instrument", "removed"), ("isis.emu", "removed")]
)
def test_unavailable_historical_selection_cannot_authorize_a_layout(key, configuration):
    from asymmetry.core.instruments.resolution import layout_key_for_selection

    selected = InstrumentSelection(key, configuration, "old", "legacy")
    assert InstrumentSelection.from_dict(selected.to_dict()) == selected
    assert layout_key_for_selection(selected, 96) is None


def test_recorded_selection_rejects_an_unqualified_key():
    with pytest.raises(ValueError, match="facility-qualified"):
        InstrumentSelection("EMU", "made_up", "1", "manual")
