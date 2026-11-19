"""Identity resolution cannot discard authored analysis requests or acquired facts."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.instruments.resolution import (
    InstrumentSelection,
    RecordedInstrumentChoice,
)
from asymmetry.core.project.profiles import (
    ProfileFingerprint,
    choose_instrument_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.transform.grouping import InvalidGroupingError


def test_positive_identity_conflict_preserves_the_complete_saved_payload_and_run():
    run = Run(
        histograms=[Histogram(np.full(8, 100.0), 0.1, 0, 0, 7) for _ in range(8)],
        metadata={"facility": "PSI", "instrument": "FLAME"},
        grouping={"instrument": "FLAME", "groups": {1: [1], 2: [2]}},
    )
    payload = {
        "groups": {1: [2, 3], 7: [5, 6]},
        "group_names": {1: "Saved positive", 7: "Saved negative"},
        "forward_group": 1,
        "backward_group": 7,
        "excluded_detectors": [8],
        "projections": [{"label": "Owned", "forward_group": 1, "backward_group": 7, "alpha": 0.9}],
        "alpha_x": 0.93,
        "alpha_x_error": 0.02,
        "alpha_x_reference_run": 8,
        "alpha_policy": {"mode": "manual", "value": 0.9},
        "background_run": {
            "source_file": "reference.nxs",
            "timing": {"t0_policy": {"mode": "manual", "offset_bins": 2}},
        },
        "t0_policy": {"mode": "manual", "offset_bins": -1},
        "detector_first_good_bins": [0] * 8,
        "detector_last_good_bins": [7] * 8,
        "deadtime_policy": {"mode": "manual", "manual_us": 0.01},
        "owned_extra": {"history": ["retained"], "calibration_capture": {"value": 0.9}},
    }
    before = deepcopy(payload)
    grouping_before, metadata_before = deepcopy(run.grouping), deepcopy(run.metadata)
    run.instrument_choice = RecordedInstrumentChoice(
        InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
    )
    profile = profile_from_payload(
        payload, "Saved", ProfileFingerprint("psi.gps", 8, "bin_combined")
    )
    with pytest.raises(InvalidGroupingError, match="Instrument identity conflicts") as error:
        resolve_effective_grouping(profile, run)
    assert error.value.grouping["groups"] == before["groups"]
    assert error.value.grouping["background_run"] == before["background_run"]
    assert payload == before
    assert run.grouping == grouping_before
    assert run.metadata == metadata_before


@pytest.mark.parametrize(
    "metadata,pulsed,reason",
    [
        ({}, None, "absent"),
        ({"instrument": "EMU"}, None, "absent"),
        ({"source": "pulsed"}, None, "unsupported"),
        ({"facility": "ISIS", "source": "PSI"}, None, "conflict"),
        ({"source_pulsed": False, "facility": "ISIS"}, False, "known"),
        ({"facility": "TRIUMF"}, False, "known"),
    ],
)
def test_source_assessment_retains_unknown_reason_and_explicit_boolean_precedence(
    metadata, pulsed, reason
):
    from asymmetry.core.instruments.resolution import SourceAssessment, assess_source

    before = deepcopy(metadata)
    result = assess_source(metadata)
    assert result.pulsed is pulsed
    assert result.reason == reason
    assert SourceAssessment.from_dict(result.to_dict()) == result
    assert metadata == before


def test_a_captured_unknown_source_survives_clone_without_metadata_reinference():
    from dataclasses import replace

    from asymmetry.core.instruments.source import SourceAssessment, source_for_run

    captured = SourceAssessment(None, "absent", "metadata")
    run = Run(metadata={"facility": "ISIS"}, source_assessment=captured)
    assert source_for_run(run) is captured
    assert source_for_run(replace(run)) is captured
    assert source_for_run(Run(metadata={"facility": "ISIS"})).pulsed is True


def test_identity_release_retains_authored_policies_without_resolving_reference():
    run = Run(
        histograms=[Histogram(np.full(8, 100.0), 0.1, 0, 0, 7) for _ in range(8)],
        metadata={"facility": "PSI", "instrument": "FLAME"},
        grouping={"groups": {1: [1], 2: [2]}, "alpha": 0.91, "owned_extra": ["kept"]},
        instrument_choice=RecordedInstrumentChoice(
            InstrumentSelection("psi.gps", "bin_combined", "1", "legacy"), ()
        ),
    )
    payload = {
        "groups": {1: [2], 2: [3]},
        "background_run": {"source_file": "missing.nxs"},
        "background_mode": "reference_run",
        "background_correction": True,
        "alpha_policy": {"mode": "per_run_estimate"},
        "t0_policy": {"mode": "manual", "offset_bins": 2},
    }
    profile = profile_from_payload(
        payload, "Saved", ProfileFingerprint("psi.gps", 8, "bin_combined")
    )
    chosen = RecordedInstrumentChoice(
        InstrumentSelection("psi.flame", "eight_channel", "1", "manual"), ()
    )
    updated, released = choose_instrument_for_run(run, chosen, profile)
    assert released
    assert updated.grouping["groups"] == payload["groups"]
    assert updated.grouping["background_run"] == payload["background_run"]
    assert updated.grouping["alpha_policy"] == {"mode": "per_run_estimate"}
    assert updated.grouping["alpha"] == 0.91
    assert updated.grouping["owned_extra"] == ["kept"]
    assert updated.grouping["t0_policy"] == payload["t0_policy"]
    assert updated.metadata == run.metadata
    assert profile.name == "Saved"
