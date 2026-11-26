"""A saved toolbar factor must not multiply its recorded grouping a second time."""

from copy import deepcopy

import pytest

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current


@pytest.mark.parametrize("panel_factor", [1, 3])
def test_recorded_binning_survives_the_display_merge_transition(panel_factor):
    project = {
        "schema_version": 28,
        "datasets": [{"run_number": 1, "grouping_overrides": {"bunching_factor": 4}}],
        "grouping_profiles": [{"name": "sample", "bunching_factor": 4}],
        "plot_state": {"bunch_factor": panel_factor},
        "view_modes_state": {
            "active_index": 1,
            "modes": [
                {"bunch_factor": 2, "x_min": 0.1},
                {"bunch_factor": 4, "x_min": 0.2},
                {"bunch_factor": 8, "x_min": 0.3},
            ],
        },
    }
    before = deepcopy(project)
    result = migrate_to_current(project)
    assert result["schema_version"] == CURRENT_SCHEMA_VERSION
    assert result["datasets"] == project["datasets"]
    assert result["grouping_profiles"] == project["grouping_profiles"]
    assert result["plot_state"] == project["plot_state"]
    modes = result["view_modes_state"]["modes"]
    assert [mode["bunch_factor"] for mode in modes] == [1, panel_factor, 1]
    assert [mode["axis_limits"]["axes"]["x"]["held"][0] for mode in modes] == [0.1, 0.2, 0.3]
    assert migrate_to_current(result) == result
    assert project == before


def test_projects_before_numbered_slots_keep_the_plot_policy_and_bunching():
    axes = {
        "x": {"auto": True, "held": [0.1, 4.0], "quantity": "time"},
        "y:P_x": {"auto": False, "held": [-10.0, 20.0], "quantity": "fb_asymmetry|False|None"},
        "y:P_y": {"auto": False, "held": [-8.0, 18.0], "quantity": "fb_asymmetry|False|None"},
    }
    project = {
        "schema_version": 42,
        "datasets": [],
        "plot_state": {"bunch_factor": 5, "axis_limits": {"axes": deepcopy(axes)}},
    }
    result = migrate_to_current(project)
    saved = result["view_modes_state"]
    assert saved["active_index"] == 0
    assert saved["modes"] == [{"bunch_factor": 5, "axis_limits": {"axes": axes}}, None, None]
    assert "view_modes_state" not in project
    assert migrate_to_current(result) == result


def test_legacy_active_count_limits_keep_only_recoverable_asymmetry_bounds():
    asymmetry = {"auto": True, "held": [-10.0, 20.0], "quantity": "fb_asymmetry|False|None"}
    time = {"auto": False, "held": [0.5, 2.5], "quantity": "time"}
    counts = {"auto": False, "held": [400.0, 1800.0], "quantity": "raw_counts|False|None"}
    project = {
        "schema_version": 42,
        "datasets": [],
        "plot_state": {
            "time_view_mode": "raw_counts",
            "axis_limits": {"axes": {"x": time, "y:P_z": asymmetry, "y:group:1": counts}},
        },
        "view_modes_state": {
            "active_index": 0,
            "modes": [{"bunch_factor": 4, "y_min": 400.0, "y_max": 1800.0}],
        },
    }
    result = migrate_to_current(project)
    assert result["view_modes_state"]["modes"][0] == {
        "bunch_factor": 4,
        "axis_limits": {"axes": {"x": time, "y:P_z": asymmetry}},
    }
    assert result["plot_state"] == project["plot_state"]
