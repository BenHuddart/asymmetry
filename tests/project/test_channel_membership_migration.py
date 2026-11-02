"""Released grouping payloads keep authored channel selection, including invalid requests."""

import json
from copy import deepcopy

import pytest

from asymmetry.core.project.profiles import GroupingProfile
from asymmetry.core.project.schema import load_project
from asymmetry.core.transform.grouping import group_forward_backward
from tests.core.test_group_membership import histograms


@pytest.mark.parametrize("members", [[[2, 4], [1, 9]], [0], [True], [1.5], ["bad"], [1, 1], [1, 4]])
def test_saved_group_membership_is_migrated_without_repair(tmp_path, members):
    grouping = {"groups": {"1": members, "2": [3]}, "forward_group": 2, "backward_group": 1}
    state = {
        "schema_version": 29,
        "datasets": [{"grouping_overrides": grouping}],
        "grouping_profiles": [{"name": "sample", "groups": grouping["groups"]}],
    }
    before = deepcopy(state)
    path = tmp_path / "membership.asymp"
    path.write_text(json.dumps(state))
    loaded = load_project(path)
    migrated = loaded["datasets"][0]["grouping_overrides"]
    expected = [2, 1] if members == [[2, 4], [1, 9]] else members
    assert migrated["groups"]["1"] == expected
    profile = GroupingProfile.from_dict(loaded["grouping_profiles"][0])
    assert profile.to_dict()["groups"][1] == expected
    if expected == [2, 1]:
        fb = group_forward_backward(histograms(), migrated)
        assert fb.forward[0] == 50
        assert fb.backward[0] == 40
    else:
        with pytest.raises(ValueError):
            group_forward_backward(histograms(), migrated)
    assert state == before


@pytest.mark.parametrize("members", [[[1, 0], [2, 0]], [0], [1, 1]])
def test_old_project_with_detector_timing_loads_before_profile_inference(members):
    from asymmetry.core.project.schema import migrate_to_current

    grouping = {
        "instrument": "EMU",
        "groups": {"1": members, "2": [3]},
        "forward_group": 1,
        "backward_group": 2,
        "t0_bin": 2,
        "detector_t0_bins": [0, 0, 0],
    }
    loaded = migrate_to_current(
        {"schema_version": 11, "datasets": [{"grouping_overrides": grouping}]}
    )
    if members == [[1, 0], [2, 0]]:
        profile = GroupingProfile.from_dict(loaded["grouping_profiles"][0])
        assert profile.groups[1] == [1, 2]
        assert profile.t0_policy.offset_bins == 2
    else:
        assert loaded["datasets"][0]["grouping_overrides"]["groups"]["1"] == members


def test_old_project_keeps_override_when_acquired_count_is_unknown():
    from asymmetry.core.project.schema import migrate_to_current

    grouping = {"instrument": "EMU", "groups": {"1": [1, 4], "2": [2]}, "alpha": 1.7}
    loaded = migrate_to_current(
        {"schema_version": 11, "datasets": [{"grouping_overrides": grouping}]}
    )
    assert loaded["datasets"][0]["grouping_overrides"] == grouping
    assert loaded["grouping_profiles"] == []
