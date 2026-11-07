"""Released scale snapshots convert once without inventing exposure evidence."""

from copy import deepcopy

import pytest

from asymmetry.core.project.schema import (
    CURRENT_SCHEMA_VERSION,
    load_project,
    migrate_to_current,
    save_project,
)


@pytest.mark.parametrize(
    "legacy,expected_request",
    [
        ({}, {"mode": "automatic"}),
        ({"scale": 7}, {"mode": "manual", "coefficient": 7}),
        ({"good_frames_sample": 200, "good_frames_reference": 100}, {"mode": "automatic"}),
        ({"good_frames_sample": 1, "good_frames_reference": 1, "scale": 7}, {"mode": "automatic"}),
        ({"good_frames_sample": None, "scale": 7}, {"mode": "automatic"}),
        ({"scale": "unusable"}, {"mode": "automatic"}),
        ({"scale": 10**310}, {"mode": "automatic"}),
    ],
)
@pytest.mark.parametrize("location", ["profile", "run"])
def test_released_shapes_convert_and_reopen_without_promoting_unknown_origin(
    legacy, expected_request, location, tmp_path
):
    declaration = {"run_number": 42, "period_number": 1, **legacy}
    grouping = {"background_run": declaration}
    state = {"schema_version": 33, "datasets": [], "grouping_profiles": []}
    if location == "profile":
        state["grouping_profiles"] = [
            {"background_policy": {"mode": "reference_run", "details": grouping}}
        ]
    else:
        state["datasets"] = [
            {"run_number": 1, "source_file": "sample.nxs", "grouping_overrides": grouping}
        ]
    original = deepcopy(state)
    migrated = migrate_to_current(state)
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert state == original

    def payload(value):
        return (
            value["grouping_profiles"][0]["background_policy"]["details"]
            if location == "profile"
            else value["datasets"][0]["grouping_overrides"]
        )["background_run"]

    actual = payload(migrated)
    assert actual["scale_request"] == expected_request
    assert not any(
        key in actual
        for key in ("scale", "good_frames_sample", "good_frames_reference", "exposure_pair")
    )
    if legacy:
        assert actual["historical_scale"] == legacy | {"exposure_origin": "unverified"}
    assert migrate_to_current(migrated) == migrated
    path = tmp_path / "scale.asymp"
    save_project(migrated, path)
    assert payload(load_project(path)) == actual


def test_cached_applied_coefficients_do_not_become_authored_manual_settings():
    state = {
        "schema_version": 33,
        "datasets": [
            {
                "background_details": {"reference": {"scale": 2, "scale_source": "live_frames"}},
                "grouping_overrides": {"background_run": {"run_number": 42}},
            }
        ],
    }
    migrated = migrate_to_current(state)
    item = migrated["datasets"][0]
    assert item["background_details"] == state["datasets"][0]["background_details"]
    assert item["grouping_overrides"]["background_run"]["scale_request"] == {"mode": "automatic"}
