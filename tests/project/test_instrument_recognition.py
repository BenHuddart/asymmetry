"""Recorded identity and authored grouping survive inconclusive or contrary file evidence."""

import copy

import pytest

from asymmetry.core.instruments.resolution import (
    accept_instrument_choice,
    layout_key_for_selection,
    resolve_instrument,
)
from asymmetry.core.project import prepare_project
from asymmetry.core.project.profiles import (
    profile_fingerprint_for_run,
    profile_from_payload,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from asymmetry.core.transform.grouping import InvalidGroupingError
from tests.core.test_grouping_profiles import _run


def _restore_released_request(path, run, payload):
    save_project(
        {
            "schema_version": 40,
            "datasets": [
                {
                    "run_number": run.run_number,
                    "source_file": run.source_file,
                    "grouping_overrides": payload,
                }
            ],
        },
        path,
    )
    prepared = prepare_project(load_project(path))
    run.instrument_choice = prepared.instrument_choices[run.run_number]
    run.grouping = copy.deepcopy(prepared.groupings[run.run_number].payload)
    return prepared.state["datasets"][0]


@pytest.mark.parametrize("n_hist", [64, 96])
def test_unknown_acquisition_evidence_preserves_migrated_musr_choice_and_authored_grouping(
    n_hist, tmp_path
):
    run = _run(instrument="Unknown", n_hist=n_hist, metadata={"instrument": "Unknown"})
    run.source_file = "run000123.nxs"
    payload = {
        "instrument": "MuSR",
        "groups": {"1": [1, 4, 9], "2": [2, 7]},
        "group_names": {"1": "Sample forward", "2": "Sample backward"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha": 1.23,
        "grouping_preset": "Custom",
    }
    original = copy.deepcopy(payload)
    entry = _restore_released_request(tmp_path / "unknown.asymp", run, payload)
    expected = {key: value for key, value in original.items() if key != "instrument"}
    assert entry["grouping_overrides"] == expected
    assert run.grouping == expected
    assert payload == original
    context = resolve_instrument(run)
    assert context.observed.selection is None
    assert context.observed.unresolved_reason == "unsupported"
    assert not context.requires_choice
    assert context.selection.profile_key == "isis.musr"
    # The saved layout has no acquired count; its chosen map cannot authorise 96 channels.
    assert context.selection.acquisition == "nexus64"
    assert layout_key_for_selection(context.selection, n_hist) == ("MuSR" if n_hist == 64 else None)
    assert run.grouping == expected
    assert run.metadata == {"instrument": "Unknown"}
    saved_path = tmp_path / "resaved.asymp"
    save_project({"schema_version": CURRENT_SCHEMA_VERSION, "datasets": [entry]}, saved_path)
    assert load_project(saved_path)["datasets"][0] == entry


def test_known_bare_hal_requires_explicit_resolution_without_discarding_saved_gps_groups(tmp_path):
    run = _run(instrument="HAL", n_hist=17, metadata={"facility": "PSI", "instrument": "HAL"})
    run.source_file = "neutral.root"
    payload = {
        "instrument": "GPS",
        "groups": {"1": [2, 4], "2": [6, 8]},
        "group_names": {"1": "Authored positive", "2": "Authored negative"},
        "forward_group": 2,
        "backward_group": 1,
        "alpha": 1.23,
        "grouping_preset": "Custom",
    }
    entry = _restore_released_request(tmp_path / "hal.asymp", run, payload)
    before = copy.deepcopy((run.grouping, run.metadata, run.instrument_choice))
    context = resolve_instrument(run)
    assert context.observed.selection.identity == ("psi.hal9500", "full17")
    assert context.recorded.selection.identity == ("psi.gps", "bin_combined")
    assert context.requires_choice
    assert context.selection is None
    assert run.grouping == entry["grouping_overrides"]
    profile = profile_from_payload(run.grouping, "Retained", profile_fingerprint_for_run(run))
    with pytest.raises(InvalidGroupingError, match="Instrument identity conflicts") as error:
        resolve_effective_grouping(profile, run)
    assert error.value.grouping["groups"] == {1: [2, 4], 2: [6, 8]}
    assert error.value.grouping["group_names"] == {1: "Authored positive", 2: "Authored negative"}
    assert error.value.grouping["forward_group"] == 2
    assert error.value.grouping["backward_group"] == 1
    assert error.value.grouping["alpha"] == 1.23
    assert error.value.grouping["grouping_preset"] == "Custom"
    assert (run.grouping, run.metadata, run.instrument_choice) == before
    run.instrument_choice = accept_instrument_choice(run, context.recorded.selection)
    assert not resolve_instrument(run).requires_choice
    assert run.instrument_choice.selection.identity == ("psi.gps", "bin_combined")
    assert run.grouping == before[0]
    assert run.metadata == before[1]
