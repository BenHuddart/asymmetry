"""Readout choice survives existing request/profile transport without upgrading legacy settings."""

from copy import deepcopy

import pytest

from asymmetry.core.project.grouping_request import (
    merge_grouping_request,
    persistent_grouping_for_run,
)
from asymmetry.core.project.profiles import (
    GroupingProfile,
    profile_fingerprint_for_run,
    profile_from_payload,
)
from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, load_project, save_project
from tests.core.test_readout_overlap import gps_run


@pytest.mark.parametrize("choice", [None, "individual", "hardware_combined"])
def test_profile_and_project_request_roundtrip_retains_exact_choice_and_conflicting_groups(
    tmp_path, choice
):
    run = gps_run({1: [6, 13, 14], 2: [1]})
    if choice is not None:
        run.grouping["readout_set"] = choice
    before = deepcopy(run.grouping)
    profile = profile_from_payload(run.grouping, "Saved", profile_fingerprint_for_run(run))
    restored = GroupingProfile.from_dict(profile.to_dict())
    assert restored.extra.get("readout_set") == choice
    assert ("readout_set" in restored.extra) == (choice is not None)
    request = persistent_grouping_for_run(run)
    state = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "datasets": [],
        "combined_datasets": [],
        "grouping_profiles": [restored.to_dict()],
        "retained_request": request,
    }
    path = tmp_path / "readouts.asymp"
    save_project(state, path)
    reopened = load_project(path)
    request = reopened["retained_request"]
    merged = merge_grouping_request({}, request)
    assert {int(gid): ids for gid, ids in merged["groups"].items()} == before["groups"]
    assert merged.get("readout_set") == choice
    assert ("readout_set" in merged) == (choice is not None)
    assert run.grouping == before
