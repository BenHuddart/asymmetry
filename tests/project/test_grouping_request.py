"""Project request transport retains extensions and excludes acquired raw buffers."""

from copy import deepcopy

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.grouping_request import persistent_grouping_for_run


def test_request_transport_retains_detached_extensions_and_captured_reference_source():
    grouping = {
        "groups": {1: [1], 2: [2]},
        "t0_bin": 3,
        "first_good_bin": 5,
        "last_good_bin": 9,
        "alpha_x": 0.93,
        "alpha_x_error": 0.02,
        "background_run": {
            "source_file": "unavailable.nxs",
            "timing": {
                "t0_policy": {"mode": "from_file"},
                "source": {"pulsed": None, "reason": "absent", "origin": "metadata"},
            },
        },
        "owned_extra": {"history": ["retained"]},
        "period_histograms": [[Histogram(np.ones(10), 0.01)]],
        "period_inputs": (Run(),),
        "period_reduced": [object()],
    }
    run = Run(histograms=[Histogram(np.ones(10), 0.01)], grouping=grouping)
    before = deepcopy(grouping["background_run"])
    payload = persistent_grouping_for_run(run)
    assert payload["t_good_offset"] == 2
    assert payload["background_run"] == before
    assert payload["owned_extra"] == {"history": ["retained"]}
    assert payload["alpha_x"] == 0.93
    assert payload["alpha_x_error"] == 0.02
    assert not {"period_histograms", "period_inputs", "period_reduced"} & payload.keys()
    payload["owned_extra"]["history"].append("edited")
    assert grouping["owned_extra"]["history"] == ["retained"]


def test_file_deadtime_is_not_promoted_to_an_authored_manual_override():
    run = Run(grouping={"groups": {}, "deadtime_mode": "from_file", "dead_time_us": [0.01]})
    payload = persistent_grouping_for_run(run)
    assert payload["groups"] == {}
    assert payload["deadtime_mode"] == "from_file"
    assert "dead_time_us" not in payload
    run.grouping["deadtime_mode"] = "manual"
    assert persistent_grouping_for_run(run)["dead_time_us"] == [0.01]
