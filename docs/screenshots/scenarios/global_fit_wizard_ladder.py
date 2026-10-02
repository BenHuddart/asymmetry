"""Global Fit Wizard — the Compare step under "Best for trending": a sharing ladder.

A real recommendation of the trend objective, built synchronously by
``build_global_fit_wizard_recommendation`` on the synthetic hopping scan
(:func:`make_hopping_zf_tscan`: one static width, a hop rate rising with
temperature, one run with 10 % more asymmetry) and handed to the window through
``set_cached_recommendation``. Two models are climbed: Dynamic GKT + Constant,
and Exponential + Constant as the poor rival that falls outside the 3 % band.

The recommended rung shares the background, the amplitude and Δ and leaves the
hop rate ν local, with the anomalous run exempt. The scenario unfolds that
model's ladder, so the other rungs show beneath it: the earlier ones, where Δ or
the amplitude is still fitted run by run and trends badly, and the last one,
which shares ν as well and costs far too much. The strip under the list traces
ν against temperature for the recommended rung.

Seed 58 (the generator's default): every run passes its residual checks, so the
summary carries the exemption and nothing else. On about half of the seeds one
of the ten runs trips the runs test by chance and is named there too.

Marked ``requires_fit = True`` because the ladders are real coupled fits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_hopping_zf_tscan
from ._base import Scenario, _process_events_for, register


class GlobalFitWizardLadderScenario(Scenario):
    name = "global_fit_wizard_ladder"
    description = (
        "Global Fit Wizard Compare step under Best for trending — the sharing "
        "ladder of a dynamic Kubo–Toyabe model on a synthetic hopping scan, with "
        "the trace strip for the recommended rung."
    )
    size = (1180, 1000)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.global_fit_wizard import (
            build_global_fit_wizard_recommendation,
        )
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_hopping_zf_tscan()
        # Two ticked families, as on the Screen step: only these climb a ladder.
        # The scan is in zero field, so the model's applied field is held at zero.
        recommendation = build_global_fit_wizard_recommendation(
            datasets,
            selected_template_keys=("dynamic_gkt_constant", "exp_constant"),
            current_parameter_types={"B_L": "Fixed"},
            current_values={"B_L": 0.0},
        )

        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=60)
        window.set_cached_recommendation(recommendation)
        # Unfold the recommended model's ladder, as a click on its toggle would.
        window._compare_panel._folds[0].toggle.click()
        _process_events_for(milliseconds=200)
        return window


register(GlobalFitWizardLadderScenario())
