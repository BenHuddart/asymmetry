"""Only fresh absent evidence can use a documented instrument source family."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.instruments.catalog import profile_by_key
from asymmetry.core.instruments.source import SourceAssessment, source_for_run


@pytest.mark.parametrize(
    "instrument,count,key,pulsed",
    [
        ("EMU", 96, "isis.emu", True),
        ("MuSR", 64, "isis.musr", True),
        ("HiFi", 64, "isis.hifi", True),
        ("GPS", 6, "psi.gps", False),
        ("FLAME", 8, "psi.flame", False),
        ("GPS", 5, "psi.gps", False),
    ],
)
def test_fresh_absent_source_uses_documented_family_without_requiring_a_channel_map(
    instrument, count, key, pulsed
):
    run = Run(
        histograms=[Histogram(np.ones(8), 0.1) for _ in range(count)],
        metadata={"instrument": instrument},
    )
    before = deepcopy(run.metadata)
    actual = source_for_run(run)
    assert actual == SourceAssessment(
        pulsed, "known", "profile", key, profile_by_key(key).identity.revision
    )
    assert run.metadata == before
    assert run.source_assessment is None


@pytest.mark.parametrize(
    "metadata,pulsed,reason,origin",
    [
        ({"instrument": "EMU", "source_pulsed": False}, False, "known", "explicit"),
        ({"instrument": "GPS", "facility": "ISIS"}, True, "known", "metadata"),
        ({"instrument": "GPS", "source": "pulsed"}, None, "unsupported", "metadata"),
        ({"instrument": "EMU", "facility": "ISIS", "source": "PSI"}, None, "conflict", "metadata"),
        ({"instrument": "LTF"}, None, "absent", "metadata"),
        ({"instrument": "Unknown"}, None, "absent", "metadata"),
    ],
)
def test_explicit_unsupported_conflicting_and_unverified_source_evidence_is_preserved(
    metadata, pulsed, reason, origin
):
    before = deepcopy(metadata)
    run = Run(metadata=metadata)
    assert source_for_run(run) == SourceAssessment(pulsed, reason, origin)
    assert metadata == before


@pytest.mark.parametrize(
    "captured",
    [
        SourceAssessment(None, "absent", "project"),
        SourceAssessment(None, "absent", "legacy:metadata"),
        SourceAssessment(False, "known", "explicit"),
        SourceAssessment(True, "known", "profile", "isis.emu", "saved-revision"),
    ],
)
def test_captured_source_wins_over_current_profile_and_contrary_metadata(captured):
    run = Run(metadata={"instrument": "GPS", "facility": "ISIS"}, source_assessment=captured)
    assert source_for_run(run) is captured


def test_conflicting_instrument_evidence_does_not_authorize_source_fallback():
    run = Run(metadata={"instrument": "EMU"}, source_file="gps000001.bin")
    actual = source_for_run(run)
    assert actual.pulsed is None
    assert actual.origin == "metadata"


def test_captured_unknown_is_authoritative_in_timing_context_and_explicit_strategy_can_override():
    from asymmetry.core.transform.corrections import reference_observation_metadata
    from asymmetry.core.transform.t0 import find_t0_for_run

    counts = np.array([0, 0, 0, 0, 0, 10, 30, 60, 90, 100, 95, 90], dtype=float)
    captured = SourceAssessment(None, "absent", "project")
    run = Run(
        histograms=[Histogram(counts, 0.02)],
        metadata={"instrument": "EMU", "facility": "ISIS"},
        source_assessment=captured,
    )
    raw = deepcopy(run.metadata)
    context = reference_observation_metadata(run)
    assert context["source_assessment"] is captured
    unknown = find_t0_for_run(run.histograms, context)
    assert not unknown.ok
    assert unknown.strategy == "unknown"
    explicit = find_t0_for_run(run.histograms, context, pulsed=True)
    assert explicit.ok
    assert [estimate.t0_bin for estimate in explicit.estimates] == [7]
    assert run.metadata == raw


@pytest.mark.parametrize("invalid", [None, True, {"pulsed": True}])
def test_present_nonrecord_timing_context_is_rejected_instead_of_reinferred(invalid):
    from asymmetry.core.transform.t0 import find_t0_for_run

    with pytest.raises(ValueError, match="typed SourceAssessment"):
        find_t0_for_run(
            [Histogram(np.array([0.0, 1.0, 3.0, 2.0]), 0.02)],
            {"facility": "ISIS", "source_assessment": invalid},
        )


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"instrument": "GPS"}, 0.0),
        ({"instrument": "GPS", "facility": "PSI"}, 0.01975),
        ({"instrument": "GPS", "psi_format": "bin"}, 0.01975),
        ({"facility": "TRIUMF"}, 0.04337),
    ],
)
def test_profile_source_does_not_activate_an_additional_rf_period(metadata, expected):
    from asymmetry.core.instruments.source import beam_period_us, resolve_facility
    from asymmetry.core.transform.background import apply_grouped_background_correction

    run = Run(metadata=metadata)
    if metadata.get("instrument") == "GPS":
        assert source_for_run(run).pulsed is False
    assert beam_period_us(resolve_facility(metadata=metadata)) == expected
    counts = np.full(120, 100.0)
    common = dict(grouping={"background_mode": "range"}, t0_bin=100, bin_width_us=0.005)
    opted_out = apply_grouped_background_correction(
        counts, counts, metadata=metadata, facility="", **common
    )
    neutral = apply_grouped_background_correction(counts, counts, facility="", **common)
    assert opted_out.outcome == neutral.outcome


def test_raw_combined_source_is_captured_once_for_repeat_and_observation_replay(monkeypatch):
    from asymmetry.core.data.combine import combine_runs
    from asymmetry.core.instruments import catalog
    from asymmetry.core.transform.preparation import prepare_histogram_asymmetry
    from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

    pulse = np.array([0, 0, 0, 0, 0, 10, 30, 60, 90, 100, 95, 90, 85, 80, 75, 70], dtype=float)
    runs = [
        Run(
            number,
            histograms=[
                Histogram(scale * pulse, 0.02, good_bin_start=10, good_bin_end=14)
                for scale in (3, 1)
            ],
            metadata={"instrument": "EMU", "run_number": number},
            source_file=f"raw{number}.nxs",
            grouping={
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "t0_source": "missing",
                "t0_policy": {"mode": "from_file"},
            },
        )
        for number in (1, 2)
    ]
    metadata = [deepcopy(run.metadata) for run in runs]
    combined = combine_runs(runs)
    reduced = reduce_run(combined, ReductionSettings())
    captured = reduced.histogram_observation.source
    assert all(run.source_assessment.pulsed is True for run in combined.grouping["period_inputs"])
    assert all(run.source_assessment.pulsed is True for run in captured.grouping["period_inputs"])
    from dataclasses import replace

    monkeypatch.setattr(
        catalog,
        "PROFILES",
        tuple(
            replace(profile, source=replace(profile.source, kind="continuous"))
            if profile.identity.key == "isis.emu"
            else profile
            for profile in catalog.PROFILES
        ),
    )
    repeated = reduce_run(combined, ReductionSettings())
    replayed = prepare_histogram_asymmetry(captured, merge=1, facility=None, reference=None)
    np.testing.assert_array_equal(repeated.time, reduced.time)
    np.testing.assert_array_equal(replayed.time, reduced.time)
    assert [run.metadata for run in runs] == metadata
