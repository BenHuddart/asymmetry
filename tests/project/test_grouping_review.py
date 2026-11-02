"""Grouping boundary recovery preserves project and correction-policy intent."""

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    BetaPolicy,
    DeadtimePolicy,
    GoodWindowPolicy,
    GroupingProfile,
    ProfileFingerprint,
    resolve_effective_grouping,
)
from asymmetry.core.project.schema import migrate_to_current
from asymmetry.core.transform.grouping import InvalidGroupingError


@pytest.mark.parametrize("fragment", [{}, {"groups": None}, {"groups": []}])
def test_v11_missing_or_malformed_group_map_stays_an_editable_override(fragment):
    grouping = {"instrument": "GPS", "detector_t0_bins": [1, 2, 3, 4], "alpha": 1.2} | fragment
    migrated = migrate_to_current(
        {"schema_version": 11, "datasets": [{"grouping_overrides": grouping}]}
    )
    assert migrated["datasets"][0]["grouping_overrides"] == grouping
    assert migrated["grouping_profiles"] == []


@pytest.mark.parametrize(
    "fields",
    [{"groups": None}, {"groups": []}, {"groups": {"bad": [1]}}, {"excluded_detectors": None}],
)
def test_profile_container_errors_use_the_membership_boundary(fields):
    with pytest.raises(InvalidGroupingError):
        GroupingProfile.from_dict(fields)


def test_background_details_keep_their_position_after_timing_resolution():
    run = Run(
        histograms=[Histogram(np.full(20, 20.0), 0.1) for _ in range(2)],
        grouping={"first_good_bin": 0, "last_good_bin": 19},
    )
    profile = GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("test", 2),
        groups={1: [1], 2: [2]},
        good_window_policy=GoodWindowPolicy(
            mode="manual", first_offset_bins=2, last_offset_bins=15
        ),
        background_policy=BackgroundPolicy(
            mode="fixed", details={"first_good_bin": 4, "last_good_bin": 10}
        ),
    )
    grouping = resolve_effective_grouping(profile, run)
    assert (grouping["first_good_bin"], grouping["last_good_bin"]) == (4, 10)


@pytest.mark.parametrize("members", [[0], [1, 2]])
def test_unavailable_profile_still_exposes_fixed_correction_configuration(members):
    run = Run(histograms=[Histogram(np.full(20, 20.0), 0.1) for _ in range(2)])
    profile = GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("test", 2),
        groups={1: members, 2: [2]},
        alpha_policy=AlphaPolicy(value=1.7),
        beta_policy=BetaPolicy(value=1.3),
        deadtime_policy=DeadtimePolicy(mode="manual", values=[0.01, 0.02]),
        background_policy=BackgroundPolicy(
            mode="fixed", details={"background_fixed_values": [3.0, 4.0]}
        ),
    )
    with pytest.raises(InvalidGroupingError) as error:
        resolve_effective_grouping(profile, run)
    grouping = error.value.grouping
    assert grouping["alpha"] == 1.7
    assert grouping["beta"] == 1.3
    assert grouping["dead_time_us"] == [0.01, 0.02]
    assert grouping["background_fixed_values"] == [3.0, 4.0]


@pytest.mark.parametrize("from_profile", [False, True])
def test_unavailable_pair_preserves_selected_period(from_profile):
    profile = GroupingProfile(
        name="sample",
        fingerprint=ProfileFingerprint("test", 2),
        groups={1: [1, 2], 2: [2]},
        period_mode="green" if from_profile else None,
    )
    run = Run(
        histograms=[Histogram(np.full(20, level), 0.1) for level in (10, 30)],
        grouping={"period_mode": "red" if from_profile else "green"},
    )
    with pytest.raises(InvalidGroupingError) as error:
        resolve_effective_grouping(profile, run)
    assert error.value.grouping["period_mode"] == "green"
