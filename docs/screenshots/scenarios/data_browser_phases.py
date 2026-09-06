"""Data Browser with a partitioned series: two phases plus one excluded run.

Populates a standalone Data Browser panel with the synthetic two-phase ZF
scan (:func:`make_two_phase_zf_tscan`) grouped into a series, then applies a
hand-built partition through the same core API the Global Fit Wizard's
**Apply phases** button calls (``ProjectModel.create_phase_groups``) — see
``tests/gui/test_data_browser_phases.py::_panel_with_partition`` for the same
construction pattern. The hottest run is deliberately left out of both
``PhaseSpec``s, so it renders as an excluded member of the series rather than
as a legitimate mismatch: the point is to show the excluded-row treatment
(hatched stripe, italic, badge), not a real gate failure.

No fit runs at capture time; the phase colours, ranges, and provenance are
plain data, not a live analysis.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_two_phase_zf_tscan
from ._base import Scenario, register


class DataBrowserPhasesScenario(Scenario):
    name = "data_browser_phases"
    description = (
        "Data browser with a temperature-scan group split into two coloured "
        "phases, plus one excluded run."
    )
    size = (760, 460)

    def build(self) -> QWidget:
        from asymmetry.core.representation.group import PhaseSpec
        from asymmetry.core.representation.project_model import ProjectModel
        from asymmetry.gui.panels.data_browser import DataBrowserPanel
        from asymmetry.gui.utils.phase_colors import phase_color

        datasets = make_two_phase_zf_tscan()
        runs = [int(dataset.run_number) for dataset in datasets]
        cold_runs = tuple(runs[0:5])
        warm_runs = tuple(runs[5:8])
        excluded_run = runs[8]

        model = ProjectModel()
        panel = DataBrowserPanel()
        panel.set_project_model(model)
        with panel.batch_updates():
            for dataset in datasets:
                panel.add_dataset(dataset)

        parent_id = panel.create_data_group(runs, name="Synthetic ZF scan", order_key="temperature")
        model.create_phase_groups(
            parent_id,
            [
                PhaseSpec(
                    ordinal=1,
                    name="Phase I",
                    member_run_numbers=cold_runs,
                    phase_range=(4.0, 18.0),
                    phase_boundaries={"lower": None, "upper": (21.0, 3.0)},
                    phase_color=phase_color(1),
                    phase_provenance={
                        "model_title": "Oscillatory × Exponential + Constant",
                        "confidence": "high",
                        "axis_key": "temperature",
                        "found_at": "2026-09-06T09:00:00",
                        "selected_breaks": 1,
                        "gains": [92.4],
                        "shared_parameters": ["A_1", "A_bg"],
                        "fit_state": "converged",
                        "reduced_chi_squared": "1.03",
                    },
                ),
                PhaseSpec(
                    ordinal=2,
                    name="Phase II",
                    member_run_numbers=warm_runs,
                    phase_range=(24.0, 38.0),
                    phase_boundaries={"lower": (21.0, 3.0), "upper": None},
                    phase_color=phase_color(2),
                    phase_provenance={
                        "model_title": "Exponential + Constant",
                        "confidence": "high",
                        "axis_key": "temperature",
                        "found_at": "2026-09-06T09:00:00",
                        "selected_breaks": 1,
                        "gains": [92.4],
                        "shared_parameters": ["A_bg"],
                        "fit_state": "converged",
                        "reduced_chi_squared": "1.03",
                    },
                ),
            ],
        )
        panel.sync_groups_from_project_model()
        panel.select_runs([excluded_run])
        panel._table.horizontalHeader().resizeSection(0, 210)
        return panel


register(DataBrowserPhasesScenario())
