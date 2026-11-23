"""Keep an authored group-7 acquisition available for explicit grouping repair."""

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.io.run_preparation import prepare_loaded_run

from ._base import Scenario, register
from .grouping_window_t0_row import _pump_until


class MuSRSingleGroupScenario(Scenario):
    name = "musr_single_group"
    description = (
        "Authored MuSR group 7 with unavailable independent F/B preview and preset repair."
    )
    size = (1180, 800)

    def build(self):
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        histograms = [
            Histogram(np.full(24, counts), 0.02, 2, 4, 19)
            for counts in [1000.0] * 32 + [3000.0] * 32
        ]
        run = prepare_loaded_run(
            Run(
                run_number=42,
                histograms=histograms,
                metadata={"instrument": "MuSR", "title": "Synthetic single-group acquisition"},
                grouping={
                    "groups": {7: list(range(1, 65))},
                    "group_names": {7: "Group 7"},
                    "detector_t0_sources": ["file"] * 64,
                    "t0_bin": 2,
                    "first_good_bin": 4,
                    "last_good_bin": 19,
                    "good_frames": 100000,
                },
            )
        )
        dataset = MuonDataset(np.array([]), np.array([]), np.array([]), run=run)
        return GroupingDialog([dataset], selected_run_number=42)

    def settle(self, widget):
        _pump_until(lambda: widget._t0_cache_key() in widget._t0_search_cache)
        super().settle(widget)

    def teardown(self, widget):
        widget._clear_dirty()
        super().teardown(widget)


register(MuSRSingleGroupScenario())
