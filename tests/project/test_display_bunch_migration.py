"""A saved toolbar factor must not multiply its recorded grouping a second time."""

from copy import deepcopy

import pytest

from asymmetry.core.project.schema import migrate_to_current


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
    assert result["schema_version"] == 29
    assert result["datasets"] == project["datasets"]
    assert result["grouping_profiles"] == project["grouping_profiles"]
    assert result["plot_state"] == project["plot_state"]
    modes = result["view_modes_state"]["modes"]
    assert [mode["bunch_factor"] for mode in modes] == [1, panel_factor, 1]
    assert [mode["x_min"] for mode in modes] == [0.1, 0.2, 0.3]
    assert migrate_to_current(result) == result
    assert project == before
