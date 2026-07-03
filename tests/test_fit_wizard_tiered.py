"""Scaffolding tests for the tiered fit-wizard screening substrate.

These cover the mechanical pieces added ahead of the tiered-screening flow:
``build_wizard_families``, the ``FamilyScreeningReport`` serializer, the
additive recommendation/assessment serializer keys, and the
``_run_template_assessments`` threading helper. The orchestrating flow tests
land later in the same file.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import ComputationalCost
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    _FIT_WIZARD_TITLES,
    CandidateAssessment,
    CandidateTemplate,
    FamilyScreeningReport,
    FitWizardRecommendation,
    SelectionMetric,
    SpectrumFingerprint,
    WizardFamily,
    _decide_family_promotions,
    _run_template_assessments,
    build_fit_wizard_recommendation,
    build_wizard_families,
    deserialize_family_screening_report,
    deserialize_fit_wizard_recommendation,
    serialize_family_screening_report,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.muon_fluorine.polarization import linear_fmuf_polarization
from asymmetry.core.fitting.peak_detection import (
    DetectedPeak,
    MultipletMatch,
    PeakAnalysis,
)
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    WizardScopePreset,
    resolve_scope,
)


def _plain_fingerprint(**overrides: object) -> SpectrumFingerprint:
    base = dict(
        tail_estimate=0.0,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.0,
        dominant_fft_snr=0.0,
        dominant_fft_cycles_in_window=0.0,
        monotonic_decay_fraction=1.0,
        early_time_curvature=0.0,
        semilog_slope_ratio=1.0,
        late_time_dip_recovery_score=0.0,
        oscillatory_hint=False,
        kt_like_hint=False,
        multi_rate_hint=False,
    )
    base.update(overrides)
    return SpectrumFingerprint(**base)  # type: ignore[arg-type]


_EXPECTED_REPS = {
    "relaxation": "exp_constant",
    "multi_rate": "biexp_constant",
    "kt": "static_gkt_constant",
    "oscillatory": "oscillatory_exp_constant",
    "muonium": "muonium_low_tf_constant",
    "fmuf": "fmuf_linear_exp_constant",
}

_CANONICAL_ORDER = ("relaxation", "multi_rate", "kt", "oscillatory", "muonium", "fmuf")


# --------------------------------------------------------------------------- #
# build_wizard_families
# --------------------------------------------------------------------------- #


def test_plain_fingerprint_yields_all_six_families_in_canonical_order() -> None:
    families = build_wizard_families(_plain_fingerprint())
    assert tuple(f.key for f in families) == _CANONICAL_ORDER
    for family in families:
        assert family.stage1_rep.key == _EXPECTED_REPS[family.key]
        assert family.priority == 0.0
        # The representative is never repeated among stage-2 members.
        member_keys = [m.key for m in family.stage2_members]
        assert family.stage1_rep.key not in member_keys


def test_no_duplicate_template_keys_across_the_table() -> None:
    families = build_wizard_families(_plain_fingerprint())
    all_keys: list[str] = []
    for family in families:
        all_keys.append(family.stage1_rep.key)
        all_keys.extend(m.key for m in family.stage2_members)
    assert len(all_keys) == len(set(all_keys))


def test_every_template_key_is_registered_in_titles() -> None:
    families = build_wizard_families(
        _plain_fingerprint(),
        current_model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    for family in families:
        for template in (family.stage1_rep, *family.stage2_members):
            assert template.key in _FIT_WIZARD_TITLES
            assert template.title == _FIT_WIZARD_TITLES[template.key]


def test_multi_rate_hint_raises_priority_ordering() -> None:
    families = build_wizard_families(_plain_fingerprint(multi_rate_hint=True))
    assert families[0].key == "multi_rate"
    assert families[0].priority == 1.0
    # Ties (the remaining zero-priority families) keep canonical order.
    rest = tuple(f.key for f in families[1:])
    assert rest == ("relaxation", "kt", "oscillatory", "muonium", "fmuf")


def test_kt_and_oscillatory_hints_raise_their_priority() -> None:
    families = build_wizard_families(_plain_fingerprint(kt_like_hint=True, oscillatory_hint=True))
    prioritised = [f.key for f in families if f.priority == 1.0]
    assert set(prioritised) == {"kt", "oscillatory"}
    # Between the two priority-1 families, canonical order (kt before oscillatory).
    assert prioritised == ["kt", "oscillatory"]


# --------------------------------------------------------------------------- #
# Scope filtering
# --------------------------------------------------------------------------- #


def test_fluoride_fmuf_scope_reduces_families() -> None:
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    keys = {f.key for f in families}
    # No transverse-precession or Kubo-Toyabe families survive the ZF/LF molecular scope.
    assert "oscillatory" not in keys
    assert "kt" not in keys
    # The fmuf family survives with its collinear representative.
    fmuf = next(f for f in families if f.key == "fmuf")
    assert fmuf.stage1_rep.key == "fmuf_linear_exp_constant"


def test_scope_rep_fallback_promotes_cheapest_surviving_member() -> None:
    # Excluding MuoniumLowTF (the rep) while keeping the other muonium forms
    # forces promotion of the cheapest surviving member; all muonium members are
    # CHEAP, so the tie breaks alphabetically -> muonium_high_tf_constant.
    resolution = resolve_scope(
        WizardScope(
            preset=WizardScopePreset.MUONIUM_RADICAL,
            exclude_components=frozenset({"MuoniumLowTF"}),
        )
    )
    assert "MuoniumLowTF" not in resolution.included_set
    assert "MuoniumTF" in resolution.included_set
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    muonium = next(f for f in families if f.key == "muonium")
    assert muonium.stage1_rep.key == "muonium_high_tf_constant"
    assert "muonium_high_tf_constant" not in {m.key for m in muonium.stage2_members}


def test_scope_omits_family_with_nothing_surviving() -> None:
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(_plain_fingerprint(), scope_resolution=resolution)
    # Muonium has no ZF/LF molecular component in scope, so the family is omitted.
    assert "muonium" not in {f.key for f in families}


def test_baseline_family_is_last_and_never_scope_filtered() -> None:
    # A narrow scope that would drop the current model's components must still
    # keep the baseline family (a Bessel oscillatory model has no ZF-molecular
    # component, yet the baseline family is exempt from scope filtering).
    current_model = CompositeModel(["Bessel", "Exponential", "Constant"], operators=["*", "+"])
    resolution = resolve_scope(WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF))
    families = build_wizard_families(
        _plain_fingerprint(), current_model=current_model, scope_resolution=resolution
    )
    assert families[-1].key == "baseline"
    baseline = families[-1]
    assert baseline.stage1_rep.is_current_model_baseline is True
    assert baseline.stage2_members == ()
    assert baseline.must_run_stage1 is True


# --------------------------------------------------------------------------- #
# FamilyScreeningReport serialization
# --------------------------------------------------------------------------- #


def test_family_screening_report_round_trip_with_inf_metric() -> None:
    report = FamilyScreeningReport(
        family_key="kt",
        title="Kubo-Toyabe",
        stage1_template_key="static_gkt_constant",
        stage1_metric_value=math.inf,
        stage1_gate_passed=False,
        promoted=False,
        reason="representative fit failed",
        stage2_template_keys=("dynamic_gkt_constant",),
    )
    payload = serialize_family_screening_report(report)
    # inf cannot live in JSON: it is stored as None.
    assert payload["stage1_metric_value"] is None
    restored = deserialize_family_screening_report(payload)
    assert restored is not None
    assert math.isinf(restored.stage1_metric_value)
    assert restored == report


def test_family_screening_report_round_trip_finite_metric() -> None:
    report = FamilyScreeningReport(
        family_key="relaxation",
        title="Relaxation",
        stage1_template_key="exp_constant",
        stage1_metric_value=12.5,
        stage1_gate_passed=True,
        promoted=True,
        reason="best Stage-1 family",
        stage2_template_keys=("gaussian_constant", "stretched_constant"),
    )
    restored = deserialize_family_screening_report(serialize_family_screening_report(report))
    assert restored == report


def test_family_screening_report_legacy_payload_tolerance() -> None:
    # A sparse/legacy dict deserializes with defaults; missing metric -> inf.
    restored = deserialize_family_screening_report({"family_key": "muonium"})
    assert restored is not None
    assert restored.family_key == "muonium"
    assert restored.title == ""
    assert restored.stage1_template_key == ""
    assert math.isinf(restored.stage1_metric_value)
    assert restored.stage1_gate_passed is False
    assert restored.promoted is False
    assert restored.stage2_template_keys == ()
    assert deserialize_family_screening_report("not-a-dict") is None


# --------------------------------------------------------------------------- #
# Recommendation / assessment serializer additions
# --------------------------------------------------------------------------- #


def _dummy_assessment(key: str, *, stage: int = 2) -> CandidateAssessment:
    template = CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_result = FitResult(success=True, chi_squared=1.0, reduced_chi_squared=1.0)
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=template,
        fit_result=fit_result,
        aic=1.0,
        aicc=1.0,
        bic=1.0,
        selected_score=1.0,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        stage=stage,
    )


def _recommendation_with_extras() -> FitWizardRecommendation:
    analysis = PeakAnalysis(
        peaks=(
            DetectedPeak(
                frequency_mhz=1.5,
                amplitude=2.0,
                snr=8.0,
                width_mhz=0.1,
                prominence=1.0,
                source="fft",
                burg_confirmed=True,
            ),
        ),
        noise_floor=0.25,
        resolution_mhz=0.1,
        nyquist_mhz=50.0,
        detrended=True,
        detrend_template_key="exp_constant",
        burg_order=8,
    )
    match = MultipletMatch(
        kind="larmor",
        family_key="oscillatory",
        peak_indices=(0,),
        quality=0.9,
        derived_values=(("field_gauss", 110.0),),
        note="line matches Larmor frequency",
    )
    report = FamilyScreeningReport(
        family_key="oscillatory",
        title="Precession",
        stage1_template_key="oscillatory_exp_constant",
        stage1_metric_value=math.inf,
        stage1_gate_passed=False,
        promoted=False,
        reason="rep failed",
    )
    return FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(),
        assessments=(_dummy_assessment("exp_constant"),),
        metric=SelectionMetric.AICC,
        recommended_key="exp_constant",
        comparable_keys=(),
        summary="ok",
        peak_analysis=analysis,
        multiplet_matches=(match,),
        family_reports=(report,),
    )


def test_recommendation_round_trip_with_new_fields() -> None:
    recommendation = _recommendation_with_extras()
    payload = serialize_fit_wizard_recommendation(recommendation)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None

    assert restored.peak_analysis is not None
    assert len(restored.peak_analysis.peaks) == 1
    assert restored.peak_analysis.peaks[0].frequency_mhz == 1.5
    assert restored.peak_analysis.detrend_template_key == "exp_constant"

    assert len(restored.multiplet_matches) == 1
    assert restored.multiplet_matches[0].kind == "larmor"
    assert restored.multiplet_matches[0].derived("field_gauss") == 110.0

    assert len(restored.family_reports) == 1
    assert restored.family_reports[0].family_key == "oscillatory"
    assert math.isinf(restored.family_reports[0].stage1_metric_value)


def test_recommendation_legacy_payload_defaults_new_fields() -> None:
    recommendation = _recommendation_with_extras()
    payload = serialize_fit_wizard_recommendation(recommendation)
    # Simulate an older persisted payload predating the tiered fields.
    for key in ("peak_analysis", "multiplet_matches", "family_reports"):
        payload.pop(key, None)
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.peak_analysis is None
    assert restored.multiplet_matches == ()
    assert restored.family_reports == ()


def test_candidate_assessment_stage_default_on_legacy_payload() -> None:
    assessment = _dummy_assessment("exp_constant", stage=1)
    recommendation = FitWizardRecommendation(
        fingerprint=_plain_fingerprint(),
        templates=(),
        assessments=(assessment,),
        metric=SelectionMetric.AICC,
        recommended_key=None,
        comparable_keys=(),
        summary="",
    )
    payload = serialize_fit_wizard_recommendation(recommendation)
    assert payload["assessments"][0]["stage"] == 1

    # Drop the stage key to emulate a legacy assessment payload -> defaults to 2.
    payload["assessments"][0].pop("stage")
    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assert restored.assessments[0].stage == 2


# --------------------------------------------------------------------------- #
# _run_template_assessments threading helper
# --------------------------------------------------------------------------- #


def _order_preserving_tasks() -> list:
    keys = ["a", "b", "c", "d", "e"]
    return [(lambda k=k: _dummy_assessment(k)) for k in keys], keys


def test_run_template_assessments_preserves_order_parallel() -> None:
    tasks, keys = _order_preserving_tasks()
    results = _run_template_assessments(tasks, max_workers=4)
    assert [a.template.key for a in results] == keys


def test_run_template_assessments_preserves_order_serial() -> None:
    tasks, keys = _order_preserving_tasks()
    results = _run_template_assessments(tasks, max_workers=1)
    assert [a.template.key for a in results] == keys


def test_run_template_assessments_empty_returns_empty() -> None:
    assert _run_template_assessments([]) == []


# --------------------------------------------------------------------------- #
# Tiered orchestrator (end-to-end)
# --------------------------------------------------------------------------- #


def _tiered_dataset(
    t: np.ndarray, y: np.ndarray, *, error: float = 0.01, metadata: dict | None = None
) -> MuonDataset:
    payload = {"run_number": 1}
    payload.update(metadata or {})
    return MuonDataset(
        time=np.asarray(t, dtype=float),
        asymmetry=np.asarray(y, dtype=float),
        error=np.full_like(np.asarray(t, dtype=float), error),
        metadata=payload,
    )


@pytest.mark.integration
def test_tiered_flow_screens_all_families_and_reports() -> None:
    rng = np.random.default_rng(21)
    t = np.linspace(0.02, 10.0, 220)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    report_keys = {report.family_key for report in recommendation.family_reports}
    assert report_keys == {"relaxation", "multi_rate", "kt", "oscillatory", "muonium", "fmuf"}
    assert all(report.reason for report in recommendation.family_reports)
    assert recommendation.peak_analysis is not None
    stage1 = [a for a in recommendation.assessments if a.stage == 1]
    # Every family fitted at least its representative in Stage 1.
    assert len(stage1) >= len(report_keys)
    assert recommendation.recommended_key == "exp_constant"


def _strip_expensive_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop EXPENSIVE Stage-2 members (numerical powder averages, strong-collision
    solvers) so end-to-end orchestrator tests stay inside the CI per-test timeout
    (same precedent as the global wizard's template-restriction helpers)."""
    from asymmetry.core.fitting import fit_wizard as fw

    original = fw.build_wizard_families

    def _cheap(*args: object, **kwargs: object) -> tuple:
        families = original(*args, **kwargs)
        threshold = fw._COST_RANK[ComputationalCost.EXPENSIVE]
        return tuple(
            replace(
                family,
                stage2_members=tuple(
                    member
                    for member in family.stage2_members
                    if fw._template_cost_rank(member) < threshold
                ),
            )
            for family in families
        )

    monkeypatch.setattr(fw, "build_wizard_families", _cheap)


@pytest.mark.integration
def test_pattern_promotion_expands_fmuf_family(monkeypatch: pytest.MonkeyPatch) -> None:
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(22)
    t = np.linspace(0.02, 24.0, 480)
    y = 0.25 * linear_fmuf_polarization(t, 1.17) + 0.02
    y = y + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004, metadata={"field_direction": "Zero field"})

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    fmuf_report = next(
        report for report in recommendation.family_reports if report.family_key == "fmuf"
    )
    assert fmuf_report.promoted
    assert fmuf_report.reason.startswith(("pattern match", "best", "residual gates"))
    assert fmuf_report.reason.startswith("pattern match")
    matches = [m for m in recommendation.multiplet_matches if m.kind == "fmuf_linear"]
    assert matches and matches[0].quality > 0.8
    # With EXPENSIVE members stripped for CI, a 3-cosine multiplet seeded at the
    # F-mu-F line frequencies can legitimately out-score the cheap fmuf members,
    # so accept either description of the triplet.
    recommended = recommendation.recommended_key or ""
    assert recommended.startswith(("fmuf", "muf", "dynamic_fmuf", "dipolar", "oscillatory"))


@pytest.mark.integration
def test_multiplet_templates_generated_for_two_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _strip_expensive_members(monkeypatch)
    rng = np.random.default_rng(23)
    t = np.linspace(0.0, 16.0, 2048)
    envelope = np.exp(-0.15 * t)
    y = (
        0.15 * np.cos(2.0 * np.pi * 1.3 * t) + 0.10 * np.cos(2.0 * np.pi * 3.7 * t)
    ) * envelope + 0.02
    y = y + rng.normal(0.0, 0.005, t.size)
    dataset = _tiered_dataset(t, y, error=0.005)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.peak_analysis is not None
    assert len(recommendation.peak_analysis.peaks) >= 2
    template_keys = {template.key for template in recommendation.templates}
    assert "oscillatory2_exp_constant" in template_keys


@pytest.mark.integration
def test_scope_restricts_screened_families() -> None:
    rng = np.random.default_rng(24)
    t = np.linspace(0.02, 10.0, 200)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    recommendation = build_fit_wizard_recommendation(dataset, scope=scope, max_workers=1)

    report_keys = {report.family_key for report in recommendation.family_reports}
    assert "fmuf" in report_keys
    assert "kt" not in report_keys
    assert "oscillatory" not in report_keys
    assert "muonium" not in report_keys


@pytest.mark.integration
def test_user_frequencies_merge_into_peaks() -> None:
    rng = np.random.default_rng(25)
    t = np.linspace(0.02, 10.0, 200)
    y = 0.22 * np.exp(-0.8 * t) + 0.03 + rng.normal(0.0, 0.004, t.size)
    dataset = _tiered_dataset(t, y, error=0.004)

    recommendation = build_fit_wizard_recommendation(
        dataset, user_frequencies_mhz=[2.5], max_workers=1
    )

    assert recommendation.peak_analysis is not None
    user_peaks = [peak for peak in recommendation.peak_analysis.peaks if peak.source == "user"]
    assert user_peaks
    assert user_peaks[0].frequency_mhz == 2.5


@pytest.mark.integration
def test_empty_scope_reports_no_candidates() -> None:
    t = np.linspace(0.02, 10.0, 50)
    dataset = _tiered_dataset(t, np.exp(-t))
    scope = WizardScope(preset=WizardScopePreset.ALL, exclude_components=frozenset(COMPONENTS))
    recommendation = build_fit_wizard_recommendation(dataset, scope=scope, max_workers=1)
    assert recommendation.recommended_key is None
    assert recommendation.templates == ()
    assert "scope" in recommendation.summary


# --------------------------------------------------------------------------- #
# Promotion decisions (unit)
# --------------------------------------------------------------------------- #


def _scored_assessment(
    key: str, value: float, *, gate: bool = False, success: bool = True
) -> CandidateAssessment:
    template = CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="",
        model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
    )
    fit_result = FitResult(success=success, chi_squared=value, reduced_chi_squared=1.0)
    empty = np.array([], dtype=float)
    return CandidateAssessment(
        template=template,
        fit_result=fit_result,
        aic=value,
        aicc=value,
        bic=value,
        selected_score=value,
        residual_rms=1.0,
        runs_z_score=0.0,
        max_abs_autocorrelation=0.0,
        residual_fft_peak_snr=0.0,
        residual_gate_passed=gate,
        residual_gate_reasons=() if gate else ("standardized residual RMS is high",),
        bound_hits=(),
        fitted_time=empty,
        fitted_curve=empty,
        component_curves=(),
        stage=1,
    )


def _unit_family(key: str) -> WizardFamily:
    return WizardFamily(
        key=key,
        title=key,
        stage1_rep=CandidateTemplate(
            key=f"{key}_rep",
            title=key,
            category="General",
            rationale="",
            model=CompositeModel(["Exponential", "Constant"], operators=["+"]),
        ),
        stage2_members=(),
    )


def test_promotion_margin_boundary() -> None:
    families = [_unit_family(k) for k in ("a", "b", "c")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 109.0),
        _scored_assessment("c_rep", 111.0),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: (promoted, reason) for family, _a, promoted, reason in decisions}
    assert by_key["a"][0] is True  # best
    assert by_key["b"][0] is True  # within delta 10
    assert by_key["c"][0] is False
    assert "not promoted" in by_key["c"][1]


def test_promotion_gate_beats_delta() -> None:
    families = [_unit_family(k) for k in ("a", "b")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 500.0, gate=True),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: reason for family, _a, promoted, reason in decisions if promoted}
    assert "b" in by_key
    assert "gates" in by_key["b"]


def test_promotion_cap_and_pattern_exemption() -> None:
    keys = ["f1", "f2", "f3", "f4", "f5", "f6", "fmuf"]
    families = [_unit_family(k) for k in keys]
    reps = [_scored_assessment(f"{k}_rep", 100.0 + i, gate=True) for i, k in enumerate(keys)]
    # fmuf scores worst but is pattern-matched.
    reps[-1] = _scored_assessment("fmuf_rep", 1000.0)
    decisions = _decide_family_promotions(families, reps, frozenset({"fmuf"}), SelectionMetric.AICC)
    promoted = {family.key for family, _a, ok, _r in decisions if ok}
    assert "fmuf" in promoted
    assert len(promoted - {"fmuf"}) == 4  # score promotions capped
    demoted_reasons = [r for family, _a, ok, r in decisions if not ok]
    assert any("cap" in reason for reason in demoted_reasons)


def test_promotion_hint_rescues_shape_mismatched_family() -> None:
    families = [_unit_family(k) for k in ("a", "kt")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("kt_rep", 500.0),
    ]
    decisions = _decide_family_promotions(
        families,
        reps,
        frozenset(),
        SelectionMetric.AICC,
        hint_family_keys=frozenset({"kt"}),
    )
    by_key = {family.key: (ok, reason) for family, _a, ok, reason in decisions}
    assert by_key["kt"][0] is True
    assert "hint" in by_key["kt"][1]


def test_promotion_failed_rep_not_promoted() -> None:
    families = [_unit_family(k) for k in ("a", "b")]
    reps = [
        _scored_assessment("a_rep", 100.0),
        _scored_assessment("b_rep", 90.0, success=False),
    ]
    decisions = _decide_family_promotions(families, reps, frozenset(), SelectionMetric.AICC)
    by_key = {family.key: (ok, reason) for family, _a, ok, reason in decisions}
    assert by_key["b"][0] is False
    assert "failed" in by_key["b"][1]
