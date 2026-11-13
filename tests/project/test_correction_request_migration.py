"""Released correction names retain precedence and coefficient provenance."""

from copy import deepcopy

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.project.schema import migrate_to_current
from asymmetry.core.transform.deadtime import file_deadtime_table, resolve_deadtime
from asymmetry.core.transform.reduce import corrected_grouped_counts, correction_flags_from_grouping


def migrated_grouping(grouping):
    project = {"schema_version": 38, "datasets": [{"grouping_overrides": grouping}]}
    original = deepcopy(project)
    current = migrate_to_current(project)
    assert project == original
    assert migrate_to_current(current) == current
    return current["datasets"][0]["grouping_overrides"]


@pytest.mark.parametrize("key", ["background_fixed_values", "background_fix", "bkg_fix"])
def test_background_aliases_reach_canonical_numerical_boundary(key):
    grouping = migrated_grouping(
        {
            key: [2.0, 3.0],
            "background_correction": True,
            "background_mode": "fixed",
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "t0_bin": 0,
            "first_good_bin": 0,
            "last_good_bin": 2,
        }
    )
    assert not {"background_fix", "bkg_fix"}.intersection(grouping)
    flags = correction_flags_from_grouping(grouping)
    histograms = [Histogram(np.array([100.0, 80.0, 60.0]), 0.1) for _ in range(2)]
    counts = corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=[0],
        backward_idx=[1],
        use_background=flags.use_background,
        use_deadtime=False,
        deadtime_mode="off",
    )
    np.testing.assert_array_equal(counts.forward.counts, [98.0, 78.0, 58.0])
    np.testing.assert_array_equal(counts.backward.counts, [97.0, 77.0, 57.0])


def test_background_precedence_does_not_choose_a_later_alias():
    grouping = migrated_grouping(
        {"background_fixed_values": [1.0, 2.0], "background_fix": [3.0, 4.0], "bkg_fix": [5.0, 6.0]}
    )
    assert grouping["background_fixed_values"] == [1.0, 2.0]
    with pytest.raises(ValueError, match="fixed background"):
        migrated_grouping({"background_fixed_values": "bad", "bkg_fix": [1.0, 2.0]})


@pytest.mark.parametrize("method", ["load", "value", "calibrate", "count_fit", "maxent_fit"])
def test_request_aliases_keep_the_actual_coefficient_origin(method):
    grouping = migrated_grouping(
        {
            "deadtime_correction": True,
            "deadtime_method": method,
            "deadtime_loaded_us": [0.01, 0.02],
            "deadtime_reference_run": 17,
            "deadtime_errors_us": [0.001, 0.002],
            "good_frames": 1000.0,
        }
    )
    assert grouping["deadtime_mode"] == "manual"
    assert grouping["deadtime_method"] == method
    assert grouping["dead_time_us"] == [0.01, 0.02]
    assert grouping["deadtime_reference_run"] == 17
    assert grouping["deadtime_errors_us"] == [0.001, 0.002]
    assert resolve_deadtime(grouping, 2, True).effective["source"] == method
    assert file_deadtime_table(grouping) == []


def test_file_request_does_not_borrow_historical_calibration_table():
    grouping = migrated_grouping(
        {
            "deadtime_correction": True,
            "deadtime_mode": "file",
            "deadtime_method": "calibrate",
            "dead_time_us": [0.01],
            "good_frames": 1000.0,
        }
    )
    assert grouping["deadtime_mode"] == "from_file"
    assert file_deadtime_table(grouping) == []
    assert resolve_deadtime(grouping, 1, True).status == "skipped"
    assert grouping["dead_time_us"] == [0.01]


def test_frozen_calibration_and_recipe_history_is_not_rewritten():
    frozen = {"deadtime_mode": "load", "bkg_fix": [4.0, 5.0], "computation_revision": "older"}
    project = {
        "schema_version": 38,
        "datasets": [
            {
                "grouping_overrides": {"deadtime_mode": "load"},
                "metadata_overrides": {"effective_recipe": frozen},
            }
        ],
        "calibration_history": [frozen],
    }
    converted = migrate_to_current(project)
    assert converted["datasets"][0]["metadata_overrides"]["effective_recipe"] == frozen
    assert converted["calibration_history"] == [frozen]


def test_authored_request_does_not_infer_mode_from_estimator_provenance():
    with pytest.raises(ValueError, match="Set deadtime_mode explicitly"):
        correction_flags_from_grouping({"deadtime_correction": True, "deadtime_method": "estimate"})
    request = {"deadtime_mode": "from_file", "deadtime_method": "calibrate", "dead_time_us": [0.01]}
    assert file_deadtime_table(request) == []
    request["deadtime_file_us"] = [0.02]
    assert file_deadtime_table(request) == [0.02]


def test_empty_released_project_mode_keeps_the_recorded_off_request():
    grouping = migrated_grouping({"deadtime_correction": True, "deadtime_mode": ""})
    assert correction_flags_from_grouping(grouping).deadtime_mode == "off"


def test_acquisition_lookup_does_not_require_an_authored_request():
    # The numerical caller may supply Manual as a separate typed argument.
    assert file_deadtime_table({"deadtime_method": "estimate", "dead_time_us": [0.01]}) == []
