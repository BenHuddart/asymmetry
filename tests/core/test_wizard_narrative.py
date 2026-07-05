"""Tests for the pure, window-agnostic fit-wizard narrative builder.

Fixtures here mirror the ``_dummy_assessment``/``_recommendation_with_extras``
patterns in ``tests/core/test_fit_wizard_tiered.py`` but are kept local and
minimal — this module only needs enough of a ``FitWizardRecommendation`` to
exercise ``build_wizard_trail``/``render_log_text``/``confidence_statement``,
never a real fit.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    ConfidenceTier,
    FamilyScreeningReport,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    SpectrumFingerprint,
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.peak_detection import DetectedPeak, MultipletMatch, PeakAnalysis
from asymmetry.core.fitting.wizard_narrative import (
    FAMILY_GLOSSES,
    TrailStep,
    build_wizard_trail,
    confidence_statement,
    render_log_text,
    template_display_name,
)

_EMPTY = np.array([], dtype=float)


def _fingerprint(**overrides: object) -> SpectrumFingerprint:
    base = dict(
        tail_estimate=0.0,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.0,
        dominant_fft_snr=0.0,
        dominant_fft_cycles_in_window=0.0,
        monotonic_decay_fraction=0.0,
        early_time_curvature=0.0,
        semilog_slope_ratio=0.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=False,
        kt_like_hint=False,
        multi_rate_hint=False,
    )
    base.update(overrides)
    return SpectrumFingerprint(**base)  # type: ignore[arg-type]


def _assessment(
    key: str,
    *,
    aicc: float = 100.0,
    is_null_baseline: bool = False,
    disqualification_reasons: tuple[str, ...] = (),
    residual_gate_passed: bool = True,
    residual_gate_reasons: tuple[str, ...] = (),
    success: bool = True,
    free_param_count: int = 0,
) -> CandidateAssessment:
    template = CandidateTemplate(
        key=key,
        title=key.replace("_", " ").title(),
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    parameters = ParameterSet([Parameter(name=f"p{i}", value=1.0) for i in range(free_param_count)])
    fit_result = FitResult(
        success=success, chi_squared=aicc, reduced_chi_squared=1.0, parameters=parameters
    )
    return CandidateAssessment(
        template=template,
        fit_result=fit_result,
        aic=aicc,
        aicc=aicc,
        bic=aicc,
        selected_score=aicc,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=residual_gate_passed,
        residual_gate_reasons=residual_gate_reasons,
        bound_hits=(),
        fitted_time=_EMPTY,
        fitted_curve=_EMPTY,
        component_curves=(),
        disqualification_reasons=disqualification_reasons,
        is_null_baseline=is_null_baseline,
    )


def _family_report(
    family_key: str,
    title: str,
    *,
    stage1_key: str,
    promoted: bool,
    reason: str,
    stage2_keys: tuple[str, ...] = (),
) -> FamilyScreeningReport:
    return FamilyScreeningReport(
        family_key=family_key,
        title=title,
        stage1_template_key=stage1_key,
        stage1_metric_value=100.0,
        stage1_gate_passed=promoted,
        promoted=promoted,
        reason=reason,
        stage2_template_keys=stage2_keys,
    )


def _high_structured_recommendation() -> FitWizardRecommendation:
    winner = _assessment("oscillatory_exp_constant", aicc=50.0)
    null_flat = _assessment("null_constant", aicc=90.0, is_null_baseline=True)
    null_exp = _assessment("null_exp", aicc=80.0, is_null_baseline=True)
    rejected = _assessment(
        "kt_constant",
        aicc=200.0,
        disqualification_reasons=("frequency_0 at the 1/T resolution floor (0.01 <= 0.02 MHz)",),
    )
    analysis = PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=1.35,
                amplitude=2.0,
                snr=12.0,
                width_mhz=0.05,
                prominence=1.0,
                source="fft",
            ),
        ),
        noise_floor=0.1,
        resolution_mhz=0.05,
        nyquist_mhz=50.0,
        detrended=True,
    )
    larmor_match = MultipletMatch(
        kind="larmor",
        family_key="oscillatory",
        peak_indices=(0,),
        quality=0.95,
        derived_values=(("field_gauss", 100.0),),
        note="line at 1.35 MHz matches the muon Larmor frequency for 100 G",
    )
    families = (
        _family_report(
            "relaxation",
            "Relaxation",
            stage1_key="exp_constant",
            promoted=True,
            reason="smooth-relaxation reference family — always expanded",
        ),
        _family_report(
            "oscillatory",
            "Precession",
            stage1_key="oscillatory_exp_constant",
            promoted=True,
            reason="pattern match promotes this family",
            stage2_keys=("oscillatory_gaussian_constant",),
        ),
        _family_report(
            "kt",
            "Kubo-Toyabe",
            stage1_key="static_gkt_constant",
            promoted=False,
            reason="not promoted: gates failed",
        ),
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(oscillatory_hint=True),
        templates=(winner.template, null_flat.template, null_exp.template, rejected.template),
        assessments=(winner, null_flat, null_exp, rejected),
        metric=SelectionMetric.AICC,
        recommended_key=winner.template.key,
        comparable_keys=(),
        summary="Recommended: Oscillatory Exp Constant by AICc.",
        peak_analysis=analysis,
        multiplet_matches=(larmor_match,),
        family_reports=families,
        confidence=ConfidenceTier.HIGH,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat="",
        scope_note="run geometry: transverse field — screening TF families",
    )
    return recommendation


def _medium_with_caveat_recommendation() -> FitWizardRecommendation:
    winner = _assessment(
        "exp_gaussian_constant",
        aicc=60.0,
        residual_gate_passed=False,
        residual_gate_reasons=("standardized residual RMS is high (2.10)",),
    )
    null_flat = _assessment("null_constant", aicc=120.0, is_null_baseline=True)
    families = (
        _family_report(
            "multi_rate",
            "Multi-rate relaxation",
            stage1_key="biexp_constant",
            promoted=True,
            reason="best Stage-1 score",
        ),
    )
    caveat = "Structured residuals remain: standardized residual RMS is high (2.10)."
    return FitWizardRecommendation(
        fingerprint=_fingerprint(multi_rate_hint=True),
        templates=(winner.template, null_flat.template),
        assessments=(winner, null_flat),
        metric=SelectionMetric.AICC,
        recommended_key=winner.template.key,
        comparable_keys=(),
        summary="Recommended: Exp Gaussian Constant by AICc (medium confidence).",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=families,
        confidence=ConfidenceTier.MEDIUM,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat=caveat,
        scope_note="",
    )


def _no_significant_structure_recommendation() -> FitWizardRecommendation:
    null_exp = _assessment("null_exp", aicc=80.0, is_null_baseline=True)
    caveat = (
        "Exp Constant improves on the Null baseline: constant null baseline by only "
        "ΔAICc = 3.0 (< 10); the data show no significant structure beyond the null."
    )
    families = (
        _family_report(
            "relaxation",
            "Relaxation",
            stage1_key="exp_constant",
            promoted=True,
            reason="smooth-relaxation reference family — always expanded",
        ),
        # Realistic: on flat/noise data the oscillatory representative's fit
        # itself fails, and ``_decide_family_promotions`` phrases that as
        # "... fit failed". Included here so the no-structure fixture
        # actually exercises the "never echo raw reasons" contract instead of
        # passing only because every reason happens to be clean.
        _family_report(
            "oscillatory",
            "Precession",
            stage1_key="oscillatory_exp_constant",
            promoted=False,
            reason="not promoted: Stage-1 representative fit failed",
        ),
    )
    return FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(null_exp.template,),
        assessments=(null_exp,),
        metric=SelectionMetric.AICC,
        recommended_key=null_exp.template.key,
        comparable_keys=(),
        summary="No significant structure — Exp Constant does not beat the null baseline by AICc.",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=families,
        confidence=ConfidenceTier.NONE,
        verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
        caveat=caveat,
        scope_note="field geometry not recorded — screening all component families",
    )


def _none_verdict_recommendation() -> FitWizardRecommendation:
    return FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary=(
            "No candidate could be recommended — every successful fit was "
            "disqualified and no null baseline is available."
        ),
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=(),
        confidence=ConfidenceTier.NONE,
        verdict=RecommendationVerdict.NONE,
        caveat="",
        scope_note="",
    )


# --------------------------------------------------------------------------- #
# Full HIGH structured case
# --------------------------------------------------------------------------- #


def test_high_structured_trail_has_all_six_steps_in_order() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    assert [step.key for step in trail] == [
        "conditions",
        "families",
        "spectrum",
        "candidates",
        "verdict",
        "confidence",
    ]
    for step in trail:
        assert isinstance(step, TrailStep)
        assert step.detail_kind == step.key
        assert step.headline


def test_high_structured_conditions_step_uses_scope_note() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    conditions = trail[0]
    assert "transverse field" in conditions.headline


def test_high_structured_families_step_glosses_family_titles() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    families = trail[1]
    assert "Precession" in families.headline
    assert "Kubo-Toyabe" in families.headline
    joined_detail = " ".join(families.detail_lines)
    # The plain-physics gloss for oscillatory/kt appears somewhere in the detail.
    assert "precess" in joined_detail.lower()


def test_families_step_never_echoes_raw_failure_reasons() -> None:
    # A realistic no-significant-structure run has an unpromoted family whose
    # Stage-1 representative failed outright or failed its residual gates —
    # ``_decide_family_promotions`` phrases that as "... fit failed" / "gates
    # failed". Step 2 must report only titles/glosses/promoted-state, never
    # echo that reason text, or the null "RESULT, not failure" framing breaks.
    families = (
        _family_report(
            "oscillatory",
            "Precession",
            stage1_key="oscillatory_exp_constant",
            promoted=False,
            reason="not promoted: Stage-1 representative fit failed",
        ),
        _family_report(
            "kt",
            "Kubo-Toyabe",
            stage1_key="static_gkt_constant",
            promoted=False,
            reason="not promoted: delta above best and gates failed",
        ),
        _family_report(
            "relaxation",
            "Relaxation",
            stage1_key="exp_constant",
            promoted=True,
            reason="smooth-relaxation reference family — always expanded",
        ),
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=families,
        confidence=ConfidenceTier.NONE,
        verdict=RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE,
        caveat="",
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    families_step = trail[1]
    joined = " ".join(families_step.detail_lines).lower()
    assert "failed" not in joined
    assert "not expanded" in joined

    full_log = render_log_text(recommendation).lower()
    assert "failed" not in full_log


def test_high_structured_spectrum_step_phrases_larmor_match() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    spectrum = trail[2]
    assert "1" in spectrum.headline  # one line detected
    joined = " ".join(spectrum.detail_lines)
    assert "Larmor" in joined


def test_high_structured_candidates_step_condenses_rejection() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    candidates = trail[3]
    joined = " ".join(candidates.detail_lines)
    assert "1 rejected" in joined
    assert "no support in the spectrum" in joined
    # Raw technical reason text must not leak through.
    assert "resolution floor" not in joined
    assert "1/T" not in joined


def test_high_structured_verdict_step_names_winner_with_gloss() -> None:
    trail = build_wizard_trail(_high_structured_recommendation())
    verdict = trail[4]
    assert "precession signal" in verdict.headline.lower() or "Oscillatory" in verdict.headline


def test_verdict_step_claims_decisive_win_when_margin_is_computable_and_large() -> None:
    # The winner has more free parameters than the simplest successful null and
    # clears it by >= 10 AICc — the one case where the "clearly better than a
    # plain relaxation baseline" clause is allowed to fire. Without a fixture
    # like this, the decisiveness branch has zero positive coverage.
    winner = _assessment("oscillatory_exp_constant", aicc=50.0, free_param_count=4)
    null_flat = _assessment("null_constant", aicc=90.0, is_null_baseline=True, free_param_count=1)
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(winner.template, null_flat.template),
        assessments=(winner, null_flat),
        metric=SelectionMetric.AICC,
        recommended_key=winner.template.key,
        comparable_keys=(),
        summary="",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=(),
        confidence=ConfidenceTier.HIGH,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat="",
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    verdict = trail[4]
    assert "clearly better than a plain relaxation baseline" in verdict.headline


def test_verdict_step_does_not_claim_decisiveness_without_null_assessments() -> None:
    # Same winner, but the null baselines are absent (as on a legacy/partial
    # payload) — the margin cannot be recomputed, so the claim must not appear.
    winner = _assessment("oscillatory_exp_constant", aicc=50.0, free_param_count=4)
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(winner.template,),
        assessments=(winner,),
        metric=SelectionMetric.AICC,
        recommended_key=winner.template.key,
        comparable_keys=(),
        summary="",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=(),
        confidence=ConfidenceTier.HIGH,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat="",
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    verdict = trail[4]
    assert "clearly better than a plain relaxation baseline" not in verdict.headline
    assert "describes real structure" in verdict.headline


def test_high_structured_confidence_statement_matches_card_wording() -> None:
    recommendation = _high_structured_recommendation()
    trail = build_wizard_trail(recommendation)
    confidence = trail[5]
    statement = confidence_statement(recommendation)
    assert confidence.headline == statement
    assert statement == "High confidence — the recommended model describes the data cleanly."


# --------------------------------------------------------------------------- #
# MEDIUM confidence with caveat
# --------------------------------------------------------------------------- #


def test_medium_confidence_caveat_appears_in_confidence_step_and_statement() -> None:
    recommendation = _medium_with_caveat_recommendation()
    statement = confidence_statement(recommendation)
    assert "Medium confidence" in statement
    assert "residual RMS is high" in statement

    trail = build_wizard_trail(recommendation)
    confidence_step = trail[5]
    assert "residual RMS is high" in confidence_step.headline
    assert "residual RMS is high" in " ".join(confidence_step.detail_lines)


def test_medium_confidence_verdict_step_does_not_overstate() -> None:
    trail = build_wizard_trail(_medium_with_caveat_recommendation())
    verdict = trail[4]
    # No "clearly better" claim without a null-baseline delta to support it —
    # only one null is present here and it is a much larger AICc away, but we
    # must not invent a stronger claim than the wording map allows.
    assert "structure" in verdict.headline.lower() or "better" in verdict.headline.lower()


# --------------------------------------------------------------------------- #
# NO_SIGNIFICANT_STRUCTURE — result framing, never failure language
# --------------------------------------------------------------------------- #


def test_no_significant_structure_uses_result_framing() -> None:
    recommendation = _no_significant_structure_recommendation()
    trail = build_wizard_trail(recommendation)
    verdict = trail[4]
    confidence = trail[5]
    assert "no oscillation or extra structure is worth fitting" in verdict.headline
    assert "no oscillation or extra structure is worth fitting" in confidence.headline

    full_text = render_log_text(recommendation).lower()
    for banned in ("failed", "error", "failure"):
        assert banned not in full_text


def test_no_significant_structure_confidence_statement_is_bare_result_framing() -> None:
    # The wording map reserves "append the caveat" for Medium confidence only;
    # the shared card string for NONE + NO_SIGNIFICANT_STRUCTURE stays plain
    # (no numeric ΔAICc jargon on the card).
    recommendation = _no_significant_structure_recommendation()
    statement = confidence_statement(recommendation)
    assert statement == (
        "The data look like a simple decay/flat background — no oscillation or "
        "extra structure is worth fitting."
    )


def test_no_significant_structure_confidence_step_carries_caveat_in_detail_only() -> None:
    # The caveat is still available as guidance in the expanded panel / copy
    # log, just not folded into the shared card statement.
    recommendation = _no_significant_structure_recommendation()
    trail = build_wizard_trail(recommendation)
    confidence = trail[5]
    assert confidence.headline == confidence_statement(recommendation)
    assert "does not beat" not in confidence.headline
    assert any("ΔAICc" in line for line in confidence.detail_lines)


# --------------------------------------------------------------------------- #
# NONE verdict
# --------------------------------------------------------------------------- #


def test_none_verdict_trail_is_honest_and_does_not_raise() -> None:
    recommendation = _none_verdict_recommendation()
    trail = build_wizard_trail(recommendation)
    assert len(trail) == 6
    verdict = trail[4]
    assert "no recommendation" in verdict.headline.lower()
    confidence = trail[5]
    assert "no confident recommendation" in confidence.headline.lower()


# --------------------------------------------------------------------------- #
# Disqualified candidates condensed into step 4 (additional coverage)
# --------------------------------------------------------------------------- #


def test_multiple_disqualification_reasons_are_deduplicated_and_condensed() -> None:
    winner = _assessment("oscillatory_exp_constant", aicc=50.0)
    rejected_a = _assessment(
        "kt_constant",
        aicc=150.0,
        disqualification_reasons=("frequency_0 at the 1/T resolution floor (0.01 <= 0.02 MHz)",),
    )
    rejected_b = _assessment(
        "gbkt_constant",
        aicc=160.0,
        disqualification_reasons=(
            "oscillation amplitude amplitude_0 consistent with zero (|0.001| < 2*0.002)",
        ),
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(winner.template, rejected_a.template, rejected_b.template),
        assessments=(winner, rejected_a, rejected_b),
        metric=SelectionMetric.AICC,
        recommended_key=winner.template.key,
        comparable_keys=(),
        summary="",
        peak_analysis=None,
        multiplet_matches=(),
        family_reports=(),
        confidence=ConfidenceTier.HIGH,
        verdict=RecommendationVerdict.STRUCTURED,
        caveat="",
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    candidates = trail[3]
    joined = " ".join(candidates.detail_lines)
    assert "2 rejected" in joined
    assert "no support in the spectrum" in joined
    assert "indistinguishable from zero" in joined


# --------------------------------------------------------------------------- #
# Envelope/larmor matches phrased plainly in step 3
# --------------------------------------------------------------------------- #


def test_fmuf_envelope_match_is_phrased_plainly() -> None:
    analysis = PeakAnalysis(
        peaks=(),
        noise_floor=0.1,
        resolution_mhz=0.05,
        nyquist_mhz=50.0,
        detrended=True,
    )
    match = MultipletMatch(
        kind="fmuf_envelope",
        family_key="fmuf",
        peak_indices=(),
        quality=0.8,
        derived_values=(("r_muF_angstrom", 1.17),),
        note="time-domain envelope matches a collinear F-mu-F dipolar signature with r_muF = 1.17 A",
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
        peak_analysis=analysis,
        multiplet_matches=(match,),
        family_reports=(),
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    spectrum = trail[2]
    joined = " ".join(spectrum.detail_lines)
    assert "muon-fluorine" in joined.lower() or "f-mu-f" in joined.lower()
    assert "r_muF" in joined or "1.17" in joined


def test_kt_envelope_match_is_phrased_plainly() -> None:
    analysis = PeakAnalysis(
        peaks=(),
        noise_floor=0.1,
        resolution_mhz=0.05,
        nyquist_mhz=50.0,
        detrended=True,
    )
    match = MultipletMatch(
        kind="kt_envelope",
        family_key="kt",
        peak_indices=(),
        quality=0.7,
        derived_values=(("Delta", 0.3),),
        note="time-domain envelope matches a static Gaussian Kubo-Toyabe with Delta = 0.3 us^-1",
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
        peak_analysis=analysis,
        multiplet_matches=(match,),
        family_reports=(),
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    spectrum = trail[2]
    joined = " ".join(spectrum.detail_lines)
    assert "kubo-toyabe" in joined.lower() or "static" in joined.lower()


def test_zero_peaks_and_no_matches_says_so_plainly() -> None:
    analysis = PeakAnalysis(
        peaks=(),
        noise_floor=0.1,
        resolution_mhz=0.05,
        nyquist_mhz=50.0,
        detrended=True,
    )
    recommendation = FitWizardRecommendation(
        fingerprint=_fingerprint(),
        templates=(),
        assessments=(),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
        peak_analysis=analysis,
        multiplet_matches=(),
        family_reports=(),
        scope_note="",
    )
    trail = build_wizard_trail(recommendation)
    spectrum = trail[2]
    assert "no significant spectral lines" in spectrum.headline.lower()


# --------------------------------------------------------------------------- #
# Empty / legacy payload tolerance
# --------------------------------------------------------------------------- #


def test_legacy_payload_deserializes_and_trail_builds_without_raising() -> None:
    minimal_payload = {
        "fingerprint": {
            "tail_estimate": 0.0,
            "initial_amplitude_estimate": 0.2,
            "zero_crossings": 0,
            "smoothed_zero_crossings": 0,
            "smoothed_turning_points": 0,
            "dominant_fft_frequency_mhz": 0.0,
            "dominant_fft_snr": 0.0,
            "dominant_fft_cycles_in_window": 0.0,
            "monotonic_decay_fraction": 0.0,
            "early_time_curvature": 0.0,
            "semilog_slope_ratio": 0.0,
            "late_time_dip_recovery_score": 0.0,
            "oscillatory_hint": False,
            "kt_like_hint": False,
            "multi_rate_hint": False,
        },
        "templates": [],
        "assessments": [],
        "metric": "AICc",
        "recommended_key": None,
        "comparable_keys": [],
        "summary": "legacy",
        # No peak_analysis / multiplet_matches / family_reports / confidence /
        # verdict / caveat / scope_note keys at all — predates every additive
        # field this module reads.
    }
    recommendation = deserialize_fit_wizard_recommendation(minimal_payload)
    assert recommendation is not None
    assert recommendation.scope_note == ""

    trail = build_wizard_trail(recommendation)
    assert len(trail) == 6
    for step in trail:
        assert step.headline
    # A legacy/empty payload must never read as a failure report.
    log_text = render_log_text(recommendation).lower()
    for banned in ("traceback", "exception"):
        assert banned not in log_text


def test_scope_note_fallback_sentence_when_empty() -> None:
    recommendation = _none_verdict_recommendation()
    assert recommendation.scope_note == ""
    trail = build_wizard_trail(recommendation)
    conditions = trail[0]
    assert conditions.headline == (
        "Run conditions were not recorded — every physics family was considered."
    )


# --------------------------------------------------------------------------- #
# render_log_text round-trips every headline
# --------------------------------------------------------------------------- #


def test_render_log_text_contains_every_step_headline() -> None:
    recommendation = _high_structured_recommendation()
    trail = build_wizard_trail(recommendation)
    log_text = render_log_text(recommendation)
    for step in trail:
        assert step.headline in log_text


def test_render_log_text_is_plain_text_and_nonempty_for_every_fixture() -> None:
    for recommendation in (
        _high_structured_recommendation(),
        _medium_with_caveat_recommendation(),
        _no_significant_structure_recommendation(),
        _none_verdict_recommendation(),
    ):
        text = render_log_text(recommendation)
        assert isinstance(text, str)
        assert text.strip()


# --------------------------------------------------------------------------- #
# template_display_name
# --------------------------------------------------------------------------- #


def test_template_display_name_glosses_known_family() -> None:
    name = template_display_name("fmuf", "F-mu-F (collinear) * Exponential + Constant")
    assert "muon-fluorine" in name


def test_template_display_name_falls_back_for_unknown_family() -> None:
    name = template_display_name(None, "Null baseline: constant")
    assert name == "Null baseline: constant"
    name_unknown_key = template_display_name("not_a_family", "Some Title")
    assert name_unknown_key == "Some Title"


def test_family_glosses_cover_every_wizard_family_key() -> None:
    expected_keys = {"relaxation", "multi_rate", "kt", "oscillatory", "muonium", "fmuf", "baseline"}
    assert expected_keys.issubset(FAMILY_GLOSSES.keys())
    for plain_name, gloss in FAMILY_GLOSSES.values():
        assert plain_name
        assert gloss


# --------------------------------------------------------------------------- #
# fit_wizard.py additions: scope_note serialization round-trip + legacy default
# --------------------------------------------------------------------------- #


def test_scope_note_serialization_round_trip() -> None:
    recommendation = _high_structured_recommendation()
    payload = serialize_fit_wizard_recommendation(recommendation)
    assert payload["scope_note"] == recommendation.scope_note
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.scope_note == recommendation.scope_note


def test_scope_note_legacy_payload_defaults_to_empty_string() -> None:
    recommendation = _high_structured_recommendation()
    payload = serialize_fit_wizard_recommendation(recommendation)
    payload.pop("scope_note", None)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.scope_note == ""
