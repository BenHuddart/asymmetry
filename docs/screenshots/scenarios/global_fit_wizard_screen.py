"""Global Fit Wizard — the Screen step's family leaderboard after a real screening.

Screens the four-field Ag LF decoupling series with
``build_global_fit_wizard_screening_recommendation`` and hands the result to the
window through ``set_cached_recommendation``, which lands on Screen because
nothing is optimised yet. The scope is small so the build takes seconds: the
runs are answered Longitudinal, and the generic relaxation families are screened
alongside three Kubo–Toyabe models. The families no partition of the series
could select are dropped before scoring, so three rows remain:
``Longitudinal-field KT + Constant`` leads with every run's χ²ᵣ graded good.
The two dynamic and broadened KT families trail by several hundred AICc units,
with a fair cell at 15 G. Only the leader is within Δ ≤ 10, so it alone is
pre-ticked and the footer reads "Optimise 1 family →". The preview on the right
draws the leader's per-run fits over residual strips.

Marked ``requires_fit = True``: the screening runs real fits.
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from ..data import make_ag_lf_decoupling
from ._base import Scenario, _process_events_for, register


class GlobalFitWizardScreenScenario(Scenario):
    name = "global_fit_wizard_screen"
    description = (
        "Global Fit Wizard Screen step on the Ag LF-KT decoupling series — the "
        "family leaderboard with per-run χ²ᵣ cells beside the leader's fit preview."
    )
    size = (1180, 760)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.component_tags import FieldGeometry, PhysicsClass
        from asymmetry.core.fitting.global_fit_wizard import (
            build_global_fit_wizard_screening_recommendation,
        )
        from asymmetry.core.fitting.wizard_scope import WizardScope, set_user_field_direction
        from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow

        datasets = make_ag_lf_decoupling(fields_g=(0.0, 15.0, 50.0, 100.0))
        set_user_field_direction(datasets, FieldGeometry.LF)
        scope = WizardScope(
            physics=frozenset({PhysicsClass.GENERIC_RELAXATION}),
            include_components=frozenset(
                {"LongitudinalFieldKT", "DynamicGaussianKT", "GaussianBroadenedKT"}
            ),
            exclude_components=frozenset({"Abragam"}),
        )
        recommendation = build_global_fit_wizard_screening_recommendation(datasets, scope=scope)

        window = GlobalFitWizardWindow()
        window.set_analysis_context(datasets)
        _process_events_for(milliseconds=60)
        window.set_cached_recommendation(recommendation, signature={"scope": scope.to_payload()})
        _process_events_for(milliseconds=200)
        return window


register(GlobalFitWizardScreenScenario())
