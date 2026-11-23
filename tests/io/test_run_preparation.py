"""Replay selectors retain source authority only on their acquired run/period owner."""

import pytest

from asymmetry.core.data.dataset import Run
from asymmetry.core.instruments.source import SourceAssessment
from asymmetry.core.io.run_preparation import ReplayState, prepare_loaded_run

pytestmark = pytest.mark.io


@pytest.mark.parametrize("number", [42, 42001])
def test_base_or_encoded_run_and_period_capture_the_selected_owner(number):
    captured = SourceAssessment(None, "absent", "project")
    run = Run(
        run_number=42001,
        metadata={"source_run_number": 42, "period_number": 1, "instrument": "EMU"},
    )
    result = prepare_loaded_run(
        run, ReplayState(source=captured, run_number=number, period_number=1)
    )
    assert result.source_assessment is captured
    assert run.source_assessment is None
    assert result.metadata == run.metadata


def test_contradictory_encoded_run_and_period_is_an_explicit_error():
    run = Run(run_number=42001, metadata={"source_run_number": 42, "period_number": 1})
    with pytest.raises(ValueError, match="selectors contradict"):
        prepare_loaded_run(run, ReplayState(run_number=42001, period_number=2))


@pytest.mark.parametrize("number,period", [(42, 2), (99, 1)])
def test_unselected_period_remains_legacy_replay_without_new_profile_defaults(number, period):
    captured = SourceAssessment(False, "known", "project")
    run = Run(
        run_number=42001,
        metadata={"source_run_number": 42, "period_number": 1, "instrument": "EMU"},
    )
    result = prepare_loaded_run(
        run, ReplayState(source=captured, run_number=number, period_number=period)
    )
    assert result.source_assessment == SourceAssessment(None, "absent", "legacy:metadata")
    assert result.source_assessment is not captured


@pytest.mark.parametrize("state", [ReplayState(), ReplayState(run_number=42)])
def test_present_replay_request_with_omitted_source_is_not_fresh(state):
    run = Run(run_number=42, metadata={"instrument": "EMU"})
    result = prepare_loaded_run(run, state)
    assert result.source_assessment == SourceAssessment(None, "absent", "legacy:metadata")
