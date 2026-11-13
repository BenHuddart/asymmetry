"""Restored synthetic exponential–Gaussian envelope with archived affine constraints."""

from __future__ import annotations

from ._base import Scenario, register


class HistoricalFitConstraintsScenario(Scenario):
    name = "historical_fit_constraints"
    description = "Canonical snapshot values and the explicit constraint-editing requirement."
    size = (760, 640)

    def build(self):
        from asymmetry.core.project.analysis_migration import fit_editor
        from asymmetry.gui.panels.fit.single_tab import SingleFitTab

        state = {
            "composite_model": {
                "component_names": ["Exponential", "Gaussian"],
                "operators": ["*"],
                "open_parentheses": [1, 0],
                "close_parentheses": [0, 1],
                "fraction_groups": [],
            },
            "parameters": [
                {"name": "A_1", "value": 2.0, "tie": {"main": "x", "scale": 2.0}},
                {"name": "A_2", "value": 3.0},
                {"name": "Lambda", "value": 0.5},
                {"name": "sigma", "value": 0.3},
                {"name": "x", "value": 1.0, "fixed": True},
            ],
        }
        fit_editor(state)
        tab = SingleFitTab()
        tab.restore_state(state)
        return tab


register(HistoricalFitConstraintsScenario())
