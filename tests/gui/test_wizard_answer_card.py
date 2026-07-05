"""Standalone unit tests for the window-agnostic WizardAnswerCard widget."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    ConfidenceTier,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    SpectrumFingerprint,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.wizard_answer_card import WizardAnswerCard


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fingerprint() -> SpectrumFingerprint:
    return SpectrumFingerprint(
        tail_estimate=0.01,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.0,
        dominant_fft_snr=0.0,
        dominant_fft_cycles_in_window=0.0,
        monotonic_decay_fraction=1.0,
        early_time_curvature=-0.1,
        semilog_slope_ratio=1.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=False,
        kt_like_hint=False,
        multi_rate_hint=False,
    )


def _assessment(
    key: str,
    title: str,
    *,
    params: int,
    curve: np.ndarray,
    aic: float = 8.0,
    aicc: float = 8.2,
    bic: float = 10.0,
) -> CandidateAssessment:
    names = ["A_1", "Lambda", "A_bg", "sigma"][:params]
    pset = ParameterSet([Parameter(n, value=0.1, min=0.0, max=5.0) for n in names])
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    result = FitResult(
        success=True,
        chi_squared=5.0,
        reduced_chi_squared=0.05,
        parameters=pset,
        uncertainties={n: 0.01 for n in names},
        residuals=np.zeros_like(curve),
        message="ok",
    )
    template = CandidateTemplate(
        key=key, title=title, category="General", rationale="r", model=model
    )
    return CandidateAssessment(
        template=template,
        fit_result=result,
        aic=aic,
        aicc=aicc,
        bic=bic,
        selected_score=aicc,
        residual_rms=0.9,
        runs_z_score=0.2,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.2,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.linspace(0, 8, curve.size),
        fitted_curve=curve,
        component_curves=(),
    )


def _recommendation(
    *,
    verdict=RecommendationVerdict.STRUCTURED,
    confidence=ConfidenceTier.HIGH,
    comparable=(),
    caveat="",
    gauss_aicc: float = 8.2,
) -> FitWizardRecommendation:
    t = np.linspace(0, 8, 60)
    exp_curve = 0.2 * np.exp(-0.4 * t) + 0.01
    gauss_curve = 0.18 * np.exp(-0.5 * t * t) + 0.02
    exp = _assessment("exp_constant", "Exponential + Constant", params=3, curve=exp_curve)
    gauss = _assessment(
        "gaussian_constant",
        "Gaussian + Constant",
        params=2,
        curve=gauss_curve,
        aicc=gauss_aicc,
    )
    return FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(exp.template, gauss.template),
        assessments=(exp, gauss),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=comparable,
        summary="Recommended: Exponential + Constant by AICc.",
        confidence=confidence,
        verdict=verdict,
        caveat=caveat,
    )


def _plot_arrays():
    t = np.linspace(0, 8, 60)
    y = 0.2 * np.exp(-0.4 * t) + 0.01
    e = np.full_like(t, 0.01)
    return t, y, e


def test_high_confidence_reads_as_confident(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation())
    assert "Exponential + Constant" in card._verdict_label.text()
    assert "High confidence" in card._confidence_label.text()
    # No "failed" language.
    assert "failed" not in card._verdict_label.text().lower()


def test_medium_confidence_caveat_on_card(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(
        _recommendation(
            confidence=ConfidenceTier.MEDIUM,
            caveat="Structured residuals remain: review before publishing.",
        )
    )
    text = card._confidence_label.text()
    assert "Medium confidence" in text
    assert "Structured residuals remain" in text


def test_no_structure_is_framed_as_result(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(
        _recommendation(
            verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
            confidence=ConfidenceTier.NONE,
        )
    )
    verdict = card._verdict_label.text().lower()
    confidence = card._confidence_label.text().lower()
    # Result-framing present; failure language absent.
    assert "simple decay" in verdict
    assert "no oscillation" in verdict
    assert "failed" not in verdict
    assert "failed" not in confidence
    assert "simple decay" in confidence


def test_apply_emits_selected_assessment(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation())
    emitted: list[object] = []
    card.apply_requested.connect(emitted.append)
    card._on_apply_clicked()
    assert emitted and emitted[0].template.key == "exp_constant"


def test_alternatives_swap_changes_applied_key(qapp: QApplication) -> None:
    # gauss has fewer params (2 < 3) so it is offered as an alternative.
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation())
    assert "gaussian_constant" in card._alt_buttons
    button = card._alt_buttons["gaussian_constant"]
    # The simpler-model descriptor moved to the tooltip; the button label is
    # the plain display name (plus an optional delta badge).
    assert "simpler model" not in button.text()
    assert "simpler model" in button.toolTip()

    card.set_selected_key("gaussian_constant")
    assert card.selected_key() == "gaussian_constant"
    emitted: list[object] = []
    card.apply_requested.connect(emitted.append)
    card._on_apply_clicked()
    assert emitted[0].template.key == "gaussian_constant"
    # Selected alternative is visually explicit.
    assert card._alt_buttons["gaussian_constant"].isChecked() is True


def test_comparable_keys_lead_alternatives(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation(comparable=("gaussian_constant",)))
    assert list(card._alt_buttons.keys())[0] == "gaussian_constant"


def test_residuals_toggle_redraws_without_error(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_plot_data(*_plot_arrays())
    card.set_recommendation(_recommendation())
    card._residuals_toggle.setChecked(True)
    card._residuals_toggle.setChecked(False)
    # No exception is success; the figure has at least one axis after a draw.
    figure = card._plot_widget._figure
    assert figure is not None


def test_high_confidence_winner_gets_success_frame(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation(confidence=ConfidenceTier.HIGH))
    assert tokens.SUCCESS_BG in card._card_frame.styleSheet()


def test_non_high_confidence_gets_neutral_frame(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation(confidence=ConfidenceTier.MEDIUM))
    style = card._card_frame.styleSheet()
    assert tokens.SUCCESS_BG not in style
    assert tokens.SURFACE in style


def test_no_recommendation_gets_neutral_frame(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(None)
    style = card._card_frame.styleSheet()
    assert tokens.SUCCESS_BG not in style
    assert tokens.SURFACE in style


def test_confidence_chip_text_per_tier(qapp: QApplication) -> None:
    card = WizardAnswerCard()

    card.set_recommendation(_recommendation(confidence=ConfidenceTier.HIGH))
    assert card._confidence_chip is not None
    assert card._confidence_chip.text() == "High confidence"

    card.set_recommendation(_recommendation(confidence=ConfidenceTier.MEDIUM))
    assert card._confidence_chip is not None
    assert card._confidence_chip.text() == "Medium confidence"

    card.set_recommendation(
        _recommendation(
            verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
            confidence=ConfidenceTier.NONE,
        )
    )
    assert card._confidence_chip is not None
    assert card._confidence_chip.text() == "No structure to fit"


def test_confidence_chip_hidden_for_none_tier_with_winner(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation(confidence=ConfidenceTier.NONE))
    assert card._confidence_chip is None


def test_alternative_delta_badge_formatting(qapp: QApplication) -> None:
    # exp (recommended) has aicc=8.2; gauss set to 10.3 -> delta = +2.1.
    card = WizardAnswerCard()
    card.set_recommendation(_recommendation(gauss_aicc=10.3))
    button = card._alt_buttons["gaussian_constant"]
    assert "+2.1" in button.text()
    assert "+2.10" in button.toolTip()
    assert "AICc" in button.toolTip()


def test_apply_button_uses_primary_qss(qapp: QApplication) -> None:
    card = WizardAnswerCard()
    assert tokens.ACCENT_SOFT in card._apply_btn.styleSheet()
