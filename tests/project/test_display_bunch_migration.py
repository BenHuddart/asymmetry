"""A saved toolbar factor must not multiply its recorded grouping a second time."""

from copy import deepcopy

import pytest

from asymmetry.core.project.schema import CURRENT_SCHEMA_VERSION, migrate_to_current
from asymmetry.core.project.view_migration import diagnostic_axis_id


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
    assert result["plot_state"]["axis_limits"]["axes"] == {
        "x": time,
        "y:P_z": asymmetry,
        diagnostic_axis_id("y:group:1"): counts,
    }


@pytest.mark.parametrize("view", ["fb_asymmetry", "raw_counts"])
def test_saved_slots_exclude_count_and_reconstruction_axis_identities(view):
    asymmetry = {"auto": False, "held": [-10.0, 20.0], "quantity": "fb_asymmetry|False|None"}
    untyped = {"auto": False, "held": [-5.0, 10.0], "quantity": None}
    project = {
        "schema_version": 42,
        "datasets": [
            {"run_number": 1, "grouping_overrides": {"projections": [{"label": "Counter pair"}]}}
        ],
        "plot_state": {
            "time_view_mode": view,
            "axis_limits": {
                "axes": {
                    "y": untyped,
                    "y:Counter pair": untyped,
                    "y:-1001": asymmetry,
                    "y:-1001:raw": asymmetry,
                    "y:recon:combined:1": asymmetry,
                }
            },
        },
    }
    result = migrate_to_current(project)
    axes = result["view_modes_state"]["modes"][0]["axis_limits"]["axes"]
    expected = {"y:Counter pair": untyped}
    if view == "fb_asymmetry":
        expected["y"] = untyped
    assert axes == expected
    plot_axes = result["plot_state"]["axis_limits"]["axes"]
    assert plot_axes[diagnostic_axis_id("y:-1001")] == asymmetry
    assert plot_axes[diagnostic_axis_id("y:-1001:raw")] == asymmetry
    assert plot_axes[diagnostic_axis_id("y:recon:combined:1")] == asymmetry


def test_v17_count_project_recovers_its_declared_projection():
    project = {
        "schema_version": 17,
        "datasets": [],
        "plot_state": {
            "time_view_mode": "raw_counts",
            "projection_selection": ["Counter pair"],
            "y_min": 850.0,
            "y_max": 1500.0,
            "y_limits_by_polarization": {"Counter pair": [-5.0, 10.0], "-1001": [850.0, 1500.0]},
        },
    }
    axes = migrate_to_current(project)["view_modes_state"]["modes"][0]["axis_limits"]["axes"]
    assert axes["y:Counter pair"] == {"auto": False, "held": [-5.0, 10.0], "quantity": None}
    assert set(axes) == {"x", "y:Counter pair"}


def test_v17_count_project_recovers_inferred_vector_axes():
    names = ["Px left", "Px right", "Py up", "Py down", "Pz forward", "Pz backward"]
    project = {
        "schema_version": 17,
        "datasets": [
            {
                "run_number": 101,
                "grouping_overrides": {
                    "groups": {i: [i] for i in range(1, 7)},
                    "group_names": dict(enumerate(names, start=1)),
                },
            }
        ],
        "plot_state": {
            "time_view_mode": "raw_counts",
            "y_min": 850.0,
            "y_max": 1500.0,
            "y_limits_by_polarization": {"P_x": [-5.0, 10.0], "P_z": [-8.0, 13.0]},
        },
    }
    axes = migrate_to_current(project)["view_modes_state"]["modes"][0]["axis_limits"]["axes"]
    assert set(axes) == {"x", "y:P_x", "y:P_z"}
    assert axes["y:P_x"]["held"] == [-5.0, 10.0]
    assert axes["y:P_z"]["held"] == [-8.0, 13.0]
