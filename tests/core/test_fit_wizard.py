"""Tests for the single-spectrum fit wizard core service."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]

import numpy as np
import pytest

import asymmetry.core.fitting.fit_wizard as wizard_module
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_wizard import (
    build_candidate_templates,
    build_fit_wizard_recommendation,
    build_fit_wizard_recommendation_for_templates,
    compute_information_criteria,
    deserialize_fit_wizard_recommendation,
    fingerprint_spectrum,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _configure_scipy_fit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wizard_module.FitEngine,
        "fit",
        lambda self, dataset, model_fn, parameters, *args, **kwargs: (
            wizard_module._scipy_fit_fallback(  # type: ignore[attr-defined]
                dataset,
                model_fn,
                parameters,
            )
        ),
    )


def _dataset_for(model: CompositeModel, **params: float) -> MuonDataset:
    t = np.linspace(0.0, 8.0, 140)
    y = model.function(t, **params)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=y, error=e, metadata={"run_number": 1})


def test_information_criteria_formulae() -> None:
    aic, aicc, bic = compute_information_criteria(
        chi_squared=10.0, parameter_count=3, sample_count=100
    )
    assert aic == pytest.approx(16.0)
    assert aicc == pytest.approx(16.25)
    assert bic == pytest.approx(10.0 + 3.0 * np.log(100.0))


def test_information_criteria_aicc_falls_back_to_aic_when_sample_count_is_small() -> None:
    aic, aicc, bic = compute_information_criteria(
        chi_squared=10.0, parameter_count=3, sample_count=4
    )
    assert aic == pytest.approx(16.0)
    assert aicc is None
    assert bic == pytest.approx(10.0 + 3.0 * np.log(4.0))


def test_fit_wizard_recommends_exponential_for_exponential_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "exp_constant"
    assert recommendation.recommended_assessment is not None
    assert recommendation.recommended_assessment.residual_gate_passed is True


def test_fit_wizard_recommends_two_exponentials_for_biexponential_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    dataset = _dataset_for(model, A_1=0.14, Lambda_1=2.4, A_2=0.08, Lambda_2=0.18, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "biexp_constant"


def test_fit_wizard_recommends_exponential_plus_gaussian_for_mixed_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Gaussian", "Constant"], operators=["+", "+"])
    dataset = _dataset_for(model, A_1=0.13, Lambda=1.7, A_2=0.09, sigma=0.28, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "exp_gaussian_constant"


def test_fit_wizard_recommends_gaussian_for_gaussian_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, sigma=0.6, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "gaussian_constant"


def test_fit_wizard_recommends_stretched_exponential_for_stretched_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["StretchedExponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.35, beta=1.6, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "stretched_constant"


def test_fit_wizard_recommends_static_kt_for_static_kt_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["StaticGKT_ZF", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.25, Delta=0.5, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "static_gkt_constant"


def test_fit_wizard_recommends_static_kt_times_exponential_for_damped_kt_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["StaticGKT_ZF", "Exponential", "Constant"], operators=["*", "+"])
    dataset = _dataset_for(model, A_1=0.25, Delta=0.45, Lambda=0.2, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "static_gkt_exp_constant"


def test_fit_wizard_recommends_damped_oscillation_for_oscillatory_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Oscillatory", "Exponential", "Constant"], operators=["*", "+"])
    dataset = _dataset_for(model, A_1=0.2, frequency=0.6, phase=0.0, Lambda=0.3, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "oscillatory_exp_constant"


# ── selectively adopted wimda-fit-function-parity candidates (Item 5) ─────────


def test_fit_wizard_recommends_risch_kehr_for_1d_diffusion_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["RischKehr", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.25, Gamma=2.0, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "risch_kehr_constant"


def test_fit_wizard_recommends_bessel_for_bessel_oscillation_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Bessel", "Exponential", "Constant"], operators=["*", "+"])
    dataset = _dataset_for(model, A_1=0.25, frequency=0.6, phase=0.0, Lambda=0.2, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "bessel_exp_constant"


def test_fit_wizard_recommends_gbkt_for_broadened_kt_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["GaussianBroadenedKT", "Constant"], operators=["+"])
    # Weak LF (B_L = 5 G) so B_L sits off its lower bound — the distributed-Δ
    # regime GBKT is for. Pure ZF (B_L = 0) correctly defers to StaticGKT_ZF.
    dataset = _dataset_for(model, A_1=0.25, Delta=0.6, B_L=5.0, w_rel=0.3, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "gbkt_constant"


#: The ONLY components the wizard may use as automatic candidates. Anything
#: outside this allowlist — every muonium/F–µ–F/dipolar specialist form, which
#: needs field or geometry context the deterministic fingerprint cannot supply —
#: must never leak into the portfolio. An allowlist (not a denylist) makes the
#: guard leak-proof: a future _add() of any unlisted component fails the test.
#: See docs/porting/wimda-fit-function-parity/.
_ALLOWED_WIZARD_COMPONENTS = {
    "Exponential",
    "Constant",
    "Gaussian",
    "StretchedExponential",
    "StaticGKT_ZF",
    "DynamicGaussianKT",
    "Oscillatory",
    # Selectively adopted parity components (Item 5):
    "RischKehr",
    "GaussianBroadenedKT",
    "Bessel",
}


def test_only_allowlisted_components_are_wizard_candidates() -> None:
    """Across every fingerprint-hint combination, the candidate portfolio uses
    only allowlisted components — no specialist parity form (muonium, F–µ–F,
    dipolar) can leak in. Allowlist, so a new unlisted _add() fails here."""
    from itertools import product

    from asymmetry.core.fitting.fit_wizard import SpectrumFingerprint

    base = dict(
        tail_estimate=0.0,
        initial_amplitude_estimate=0.2,
        zero_crossings=0,
        smoothed_zero_crossings=0,
        smoothed_turning_points=0,
        dominant_fft_frequency_mhz=0.5,
        dominant_fft_snr=4.0,
        dominant_fft_cycles_in_window=2.0,
        monotonic_decay_fraction=0.5,
        early_time_curvature=-0.1,
        semilog_slope_ratio=2.0,
        late_time_dip_recovery_score=0.1,
    )
    for kt, osc, multi in product((False, True), repeat=3):
        fp = SpectrumFingerprint(
            **base, kt_like_hint=kt, oscillatory_hint=osc, multi_rate_hint=multi
        )
        templates = build_candidate_templates(fp)
        used = {name for tpl in templates for name in tpl.model.component_names}
        leaked = used - _ALLOWED_WIZARD_COMPONENTS
        assert not leaked, f"non-allowlisted components leaked into candidates: {leaked}"


def test_bound_hit_detection_ignores_infinite_bounds() -> None:
    """A finite value far from a one-sided [0, inf) bound is not a bound hit
    (regression: an infinite |max| made the tolerance infinite)."""
    from asymmetry.core.fitting.fit_wizard import _bound_hit_names

    params = ParameterSet([Parameter("Gamma", 2.0, min=0.0, max=float("inf"))])
    assert _bound_hit_names(params) == []
    # A value genuinely at the finite lower bound is still flagged.
    pinned = ParameterSet([Parameter("Gamma", 0.0, min=0.0, max=float("inf"))])
    assert _bound_hit_names(pinned) == ["Gamma at lower bound"]


def test_monotonic_low_frequency_spectrum_prefers_multi_rate_over_oscillation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    t = np.linspace(0.0, 10.0, 420)
    model = CompositeModel(["Exponential", "Exponential", "Constant"], operators=["+", "+"])
    y = model.function(t, A_1=0.16, Lambda_1=3.5, A_2=0.07, Lambda_2=0.18, A_bg=0.0)
    dataset = MuonDataset(
        time=t,
        asymmetry=y,
        error=np.full_like(t, 0.01),
        metadata={"run_number": 2},
    )

    fingerprint = fingerprint_spectrum(dataset)
    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert fingerprint.oscillatory_hint is False
    assert fingerprint.multi_rate_hint is True
    assert fingerprint.dominant_fft_cycles_in_window < 1.5
    assert recommendation.recommended_key == "biexp_constant"


def test_fit_wizard_recommends_three_exponentials_for_three_rate_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(
        ["Exponential", "Exponential", "Exponential", "Constant"], operators=["+", "+", "+"]
    )
    # Three well-separated rates (8.0 / 1.2 / 0.2 µs⁻¹) over a long window so
    # that AICc genuinely favours the three-rate model — under the
    # confidence-tier policy the recommendation is the metric winner outright,
    # not a gate-passing runner-up, so the third (slow) rate must be resolvable.
    # ``_dataset_for``'s 8 µs / 140-point window leaves the slow rate degenerate
    # with the constant, so build a longer window inline.
    t = np.linspace(0.0, 12.0, 240)
    params = dict(A_1=0.10, Lambda_1=8.0, A_2=0.08, Lambda_2=1.2, A_3=0.06, Lambda_3=0.2, A_bg=0.01)
    dataset = MuonDataset(
        time=t,
        asymmetry=model.function(t, **params),
        error=np.full_like(t, 0.004),
        metadata={"run_number": 1},
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key == "triple_exp_constant"


def test_fit_wizard_analysis_is_deterministic_for_the_same_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Gaussian", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, sigma=0.6, A_bg=0.01)

    first = build_fit_wizard_recommendation(dataset, max_workers=1)
    second = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert first.recommended_key == second.recommended_key
    first_scores = [
        assessment.metric_value(first.metric) for assessment in first.sorted_assessments()
    ]
    second_scores = [
        assessment.metric_value(second.metric) for assessment in second.sorted_assessments()
    ]
    assert first_scores == pytest.approx(second_scores)


def test_residual_gate_demotes_structured_residuals() -> None:
    t = np.linspace(0.0, 8.0, 140)
    residuals = 0.05 * np.sin(2.0 * np.pi * 0.8 * t)
    dataset = MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 1},
    )
    fit_result = FitResult(
        success=True,
        chi_squared=float(np.sum((residuals / dataset.error) ** 2)),
        reduced_chi_squared=1.0,
        parameters=ParameterSet([Parameter("A_1", value=0.1), Parameter("Lambda", value=0.5)]),
        residuals=residuals,
        message="ok",
    )

    rms, runs_z, autocorr, fft_snr = wizard_module._residual_diagnostics(dataset, fit_result)  # type: ignore[attr-defined]
    reasons = wizard_module._residual_gate_reasons(  # type: ignore[attr-defined]
        fit_result=fit_result,
        residual_rms=rms,
        runs_z_score=runs_z,
        max_abs_autocorrelation=autocorr,
        residual_fft_peak_snr=fft_snr,
        bound_hits=[],
    )

    assert reasons
    assert any(
        "autocorrelation" in reason or "FFT" in reason or "RMS" in reason for reason in reasons
    )


def test_medium_confidence_recommendation_when_every_candidate_is_forced_to_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under the confidence-tier policy the gates classify rather than veto: an
    # all-gates-failing spectrum still gets a best-by-metric recommendation, now
    # at Medium confidence with the gate reasons carried as a caveat (F1 fix).
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)

    monkeypatch.setattr(
        wizard_module,
        "_residual_gate_reasons",
        lambda **kwargs: ["forced warning"],
    )

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)

    assert recommendation.recommended_key is not None
    assert recommendation.confidence is wizard_module.ConfidenceTier.MEDIUM
    assert recommendation.verdict is wizard_module.RecommendationVerdict.STRUCTURED
    assert recommendation.caveat


def test_failed_candidate_fit_is_retained_in_the_comparison_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = wizard_module._scipy_fit_fallback  # type: ignore[attr-defined]

    def _patched_fit(self, dataset, model_fn, parameters, *args, **kwargs):
        param_names = {parameter.name for parameter in parameters}
        if "sigma" in param_names and "beta" not in param_names and "frequency" not in param_names:
            return FitResult(success=False, message="forced failure")
        return original_fit(dataset, model_fn, parameters)

    monkeypatch.setattr(wizard_module.FitEngine, "fit", _patched_fit)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)
    {assessment.template.key: assessment for assessment in recommendation.assessments}


def test_fit_wizard_recommendation_serialization_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)

    recommendation = build_fit_wizard_recommendation(dataset, max_workers=1)
    restored = deserialize_fit_wizard_recommendation(
        serialize_fit_wizard_recommendation(recommendation)
    )

    assert restored is not None
    assert restored.recommended_key == recommendation.recommended_key
    assert [template.key for template in restored.templates] == [
        template.key for template in recommendation.templates
    ]
    assert [assessment.template.key for assessment in restored.assessments] == [
        assessment.template.key for assessment in recommendation.assessments
    ]
    assert restored.assessment_for_key(recommendation.recommended_key) is not None
    # New additive policy fields survive the round trip.
    assert restored.confidence is recommendation.confidence
    assert restored.verdict is recommendation.verdict
    assert restored.caveat == recommendation.caveat


def test_deserialize_recommendation_migrates_legacy_fraction_params() -> None:
    """A recommendation cached before the fraction rework migrates on load.

    Its assessment's fitted ``fraction_<k>`` params (and matching uncertainty
    keys) are renamed to the n-1 free scheme against the template's model, so the
    cached fit still applies onto the new ``f_<Component>`` rows.
    """
    model = CompositeModel.from_expression("( Exponential + Gaussian + Constant ){frac}")
    payload = {
        "fingerprint": {},
        "templates": [
            {"key": "cand", "title": "", "category": "", "rationale": "", "model": model.to_dict()}
        ],
        "assessments": [
            {
                "template": {
                    "key": "cand",
                    "title": "",
                    "category": "",
                    "rationale": "",
                    "model": model.to_dict(),
                },
                "fit_result": {
                    "success": True,
                    "chi_squared": 1.0,
                    "reduced_chi_squared": 1.0,
                    "parameters": [
                        {"name": "A_1", "value": 20.0, "min": 0.0, "max": 1e9, "fixed": False},
                        {"name": "Lambda", "value": 0.5, "min": 0.0, "max": 1e9, "fixed": False},
                        {
                            "name": "fraction_1",
                            "value": 2.0,
                            "min": 0.0,
                            "max": 1.0,
                            "fixed": False,
                        },
                        {"name": "sigma", "value": 0.3, "min": 0.0, "max": 1e9, "fixed": False},
                        {
                            "name": "fraction_2",
                            "value": 1.0,
                            "min": 0.0,
                            "max": 1.0,
                            "fixed": False,
                        },
                        {
                            "name": "fraction_3",
                            "value": 1.0,
                            "min": 0.0,
                            "max": 1.0,
                            "fixed": False,
                        },
                    ],
                    "uncertainties": {"fraction_1": 0.01, "fraction_2": 0.02, "fraction_3": 0.03},
                },
            }
        ],
        "metric": "aicc",
        "recommended_key": "cand",
        "comparable_keys": ["cand"],
        "summary": "",
    }
    # A valid fingerprint is required (deserialize returns None otherwise).
    payload["fingerprint"] = _minimal_fingerprint_payload()

    restored = deserialize_fit_wizard_recommendation(payload)
    assert restored is not None
    assessment = restored.assessments[0]
    names = {parameter.name for parameter in assessment.fit_result.parameters}
    assert "f_Exponential" in names
    assert "f_Gaussian" in names
    assert not any(name.startswith("fraction_") for name in names)
    # Uncertainties re-keyed to the surviving free params; the dropped last key gone.
    assert "f_Exponential" in assessment.fit_result.uncertainties
    assert "f_Gaussian" in assessment.fit_result.uncertainties
    assert not any(key.startswith("fraction_") for key in assessment.fit_result.uncertainties)


def _minimal_fingerprint_payload() -> dict:
    return {
        "tail_estimate": 0.0,
        "initial_amplitude_estimate": 0.0,
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
    }


def test_fit_wizard_can_evaluate_explicit_template_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_scipy_fit_backend(monkeypatch)
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)
    fingerprint = fingerprint_spectrum(dataset)
    templates = tuple(build_candidate_templates(fingerprint))

    recommendation = build_fit_wizard_recommendation_for_templates(
        dataset,
        templates,
        fingerprint=fingerprint,
    )

    assert [template.key for template in recommendation.templates] == [
        template.key for template in templates
    ]
    assert len(recommendation.assessments) == len(templates)
    assert recommendation.recommended_key == "exp_constant"

    assessments = {assessment.template.key: assessment for assessment in recommendation.assessments}
    assert "gaussian_constant" in assessments
    assert assessments["exp_constant"].fit_result.success is True
    assert assessments["exp_constant"].metric_value(recommendation.metric) < assessments[
        "gaussian_constant"
    ].metric_value(recommendation.metric)


def test_background_parameter_can_take_negative_bounds() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    dataset = _dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01)
    shifted = MuonDataset(
        time=np.asarray(dataset.time, dtype=float).copy(),
        asymmetry=np.asarray(dataset.asymmetry, dtype=float) - 0.08,
        error=np.asarray(dataset.error, dtype=float).copy(),
        metadata=dict(dataset.metadata),
    )

    fingerprint = fingerprint_spectrum(shifted)
    template = next(
        template
        for template in build_candidate_templates(fingerprint)
        if template.key == "exp_constant"
    )
    parameters = wizard_module._initial_parameters_for_template(shifted, fingerprint, template)  # type: ignore[attr-defined]
    background = next(parameter for parameter in parameters if parameter.name == "A_bg")

    assert background.min < 0.0
    assert background.value < 0.0


def test_bound_hit_names_report_parameters_stuck_on_bounds() -> None:
    parameters = ParameterSet(
        [
            Parameter("Lambda", value=0.0, min=0.0, max=1.0),
            Parameter("A_bg", value=-1.0, min=-1.0, max=1.0),
            Parameter("beta", value=3.0, min=0.1, max=3.0),
        ]
    )

    hits = wizard_module._bound_hit_names(parameters)  # type: ignore[attr-defined]

    assert "Lambda at lower bound" in hits
    assert "beta at upper bound" in hits
    assert "A_bg at lower bound" in hits


def test_default_candidate_pool_excludes_muon_fluorine_components() -> None:
    model = CompositeModel(["Exponential", "Constant"], operators=["+"])
    fingerprint = fingerprint_spectrum(_dataset_for(model, A_1=0.2, Lambda=0.4, A_bg=0.01))
    templates = build_candidate_templates(fingerprint)

    template_keys = {template.key for template in templates}
    component_names = {
        component for template in templates for component in template.model.component_names
    }
    assert "biexp_constant" in template_keys
    assert "exp_gaussian_constant" in template_keys
    assert "double_gaussian_constant" in template_keys
    assert "triple_exp_constant" in template_keys
    assert "double_exp_gaussian_constant" in template_keys
    assert "exp_double_gaussian_constant" in template_keys
    assert "triple_gaussian_constant" in template_keys
    assert "MuF" not in component_names
    assert "FmuF_Linear" not in component_names
    assert "FmuF_General" not in component_names
