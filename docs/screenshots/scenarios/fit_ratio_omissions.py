"""Inspect Gaussian omission reporting and selected residual support on sparse Ag."""

from PySide6.QtWidgets import QWidget

from ._base import Scenario, _process_events_for, register


class FitRatioOmissionsScenario(Scenario):
    name = "fit_ratio_omissions"
    description = (
        "Sparse synthetic Ag histogram fits report omissions and unavailable residual spectra."
    )
    size = (1180, 160)
    requires_fit = True

    def build(self) -> QWidget:
        from asymmetry.core.fitting.fit_wizard import (
            build_fit_wizard_recommendation_for_templates,
            build_null_baseline_templates,
        )
        from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
        from docs.screenshots.data.archetypes import make_sparse_ag_ratio_control

        dataset = make_sparse_ag_ratio_control()
        recommendation = build_fit_wizard_recommendation_for_templates(
            dataset, build_null_baseline_templates()[:1], variant_budget=1
        )
        result = recommendation.assessments[0].fit_result
        assert result.success and result.observation_report.omitted > 0
        assert result.residuals.size == result.observation_report.used
        assert dataset.n_points > result.residuals.size
        window = FitWizardWindow()
        window.set_analysis_context(dataset)
        window.set_cached_recommendation(recommendation)
        window._answer_card._residuals_toggle.setChecked(True)
        window._follow_a(recommendation.assessments[0].template.key)
        details = window._compare_warning_text
        details.setParent(None)
        window.close()
        window.deleteLater()
        _process_events_for(milliseconds=150)
        return details


register(FitRatioOmissionsScenario())
