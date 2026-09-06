"""Scope and multiplet-pattern integration for the global fit wizard portfolio."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    CandidateTemplate,
    FitWizardRecommendation,
    SelectionMetric,
    build_candidate_templates,
    fingerprint_spectrum,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import (
    _aggregate_fingerprints,
    build_global_fit_wizard_candidate_portfolio,
    series_template_alphabet,
)
from asymmetry.core.fitting.muon_fluorine.polarization import linear_fmuf_polarization
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    WizardScopePreset,
    resolve_scope_for_datasets,
)


def _dataset(
    run_number: int,
    t: np.ndarray,
    y: np.ndarray,
    *,
    metadata: dict | None = None,
) -> MuonDataset:
    payload = {"run_number": run_number, "temperature": float(run_number)}
    payload.update(metadata or {})
    return MuonDataset(
        time=np.asarray(t, dtype=float),
        asymmetry=np.asarray(y, dtype=float),
        error=np.full_like(np.asarray(t, dtype=float), 0.004),
        metadata=payload,
    )


def _exp_series(n: int = 2) -> list[MuonDataset]:
    rng = np.random.default_rng(31)
    t = np.linspace(0.02, 10.0, 200)
    return [
        _dataset(run, t, 0.2 * np.exp(-0.7 * t) + 0.02 + rng.normal(0.0, 0.004, t.size))
        for run in range(1, n + 1)
    ]


def _fmuf_series(n: int = 2) -> list[MuonDataset]:
    rng = np.random.default_rng(32)
    t = np.linspace(0.02, 24.0, 480)
    return [
        _dataset(
            run,
            t,
            0.25 * linear_fmuf_polarization(t, 1.17) + 0.02 + rng.normal(0.0, 0.004, t.size),
            metadata={"field_direction": "Zero field"},
        )
        for run in range(1, n + 1)
    ]


def test_portfolio_scope_filters_templates() -> None:
    datasets = _exp_series()
    scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets, scope=scope)

    resolution = resolve_scope_for_datasets(datasets, scope)
    for template in portfolio.templates:
        assert all(name in resolution.included_set for name in template.model.component_names), (
            template.key
        )
    keys = {template.key for template in portfolio.templates}
    assert "fmuf_linear_exp_constant" in keys
    assert "oscillatory_exp_constant" not in keys
    assert "static_gkt_constant" not in keys


def test_portfolio_legacy_default_unchanged_without_patterns() -> None:
    datasets = _exp_series()
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets)

    fingerprints = [fingerprint_spectrum(dataset) for dataset in datasets]
    legacy = build_candidate_templates(_aggregate_fingerprints(fingerprints))
    assert [template.key for template in portfolio.templates] == [
        template.key for template in legacy
    ]
    assert portfolio.pattern_template_keys == ()


def _assessment(template: CandidateTemplate, *, score: float) -> CandidateAssessment:
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    return CandidateAssessment(
        template=template,
        fit_result=FitResult(
            success=True,
            chi_squared=score,
            reduced_chi_squared=score / 10.0,
            parameters=ParameterSet([Parameter("A_1", value=0.2)]),
            message="ok",
        ),
        aic=score,
        aicc=score,
        bic=score,
        selected_score=score,
        residual_rms=0.001,
        runs_z_score=0.1,
        max_abs_autocorrelation=0.1,
        residual_fft_peak_snr=1.0,
        residual_gate_passed=True,
        residual_gate_reasons=(),
        bound_hits=(),
        fitted_time=np.array([0.0, 1.0]),
        fitted_curve=np.array([0.2, 0.1]),
        component_curves=(),
    )


def _template(
    key: str, components: tuple[str, ...] = ("Exponential", "Constant")
) -> CandidateTemplate:
    return CandidateTemplate(
        key=key,
        title=key,
        category="General",
        rationale="test",
        model=CompositeModel(list(components), operators=["+"] * (len(components) - 1)),
    )


def _recommendation(scores: dict[str, float]) -> FitWizardRecommendation:
    templates = tuple(_template(key) for key in scores)
    return rerank_fit_wizard_recommendation(
        FitWizardRecommendation(
            fingerprint=fingerprint_spectrum(_exp_series(1)[0]),
            templates=templates,
            assessments=tuple(
                _assessment(template, score=scores[template.key]) for template in templates
            ),
            metric=SelectionMetric.AICC,
            recommended_key=None,
            comparable_keys=(),
            summary="",
        ),
        SelectionMetric.AICC,
    )


def test_alphabet_keeps_a_template_only_one_run_supports() -> None:
    """A minority phase's model is in the series alphabet, not out-voted by it.

    This is the whole point of the union: the old half-of-the-runs quorum threw
    away exactly the template that describes the other half of a series.
    """
    alphabet = series_template_alphabet(
        {
            1: _recommendation({"exp_constant": 10.0, "biexp_constant": 30.0}),
            2: _recommendation({"exp_constant": 11.0, "biexp_constant": 31.0}),
            3: _recommendation({"exp_constant": 12.0, "biexp_constant": 32.0}),
            4: _recommendation({"oscillatory2_exp_relax_constant": 5.0, "exp_constant": 40.0}),
        }
    )

    keys = [template.key for template in alphabet]
    assert "oscillatory2_exp_relax_constant" in keys
    # Templates a run chose come first: run 4's winner and runs 1-3's winner.
    assert set(keys[:2]) == {"exp_constant", "oscillatory2_exp_relax_constant"}
    assert "biexp_constant" in keys


def test_alphabet_excludes_null_baselines_and_failures() -> None:
    from asymmetry.core.fitting.engine import FitResult

    recommendation = _recommendation({"exp_constant": 10.0, "biexp_constant": 20.0})
    assessments = (
        replace(recommendation.assessments[0], is_null_baseline=True),
        replace(
            recommendation.assessments[1],
            fit_result=FitResult(success=False, message="no"),
        ),
    )
    alphabet = series_template_alphabet({1: replace(recommendation, assessments=assessments)})

    assert alphabet == ()


def test_alphabet_is_capped() -> None:
    scores = {f"template_{index}": float(index) for index in range(30)}
    alphabet = series_template_alphabet({1: _recommendation(scores)}, cap=5)

    assert [template.key for template in alphabet] == [f"template_{index}" for index in range(5)]


def test_alphabet_portfolio_drops_templates_out_of_scope_for_every_run() -> None:
    datasets = _exp_series(2)
    scope = WizardScope(preset=WizardScopePreset.FLUORIDE_FMUF)
    oscillatory = _template("oscillatory_exp_constant", ("Oscillatory", "Exponential", "Constant"))
    recommendations = {
        int(dataset.run_number): replace(
            _recommendation({"exp_constant": 10.0, "oscillatory_exp_constant": 5.0}),
            templates=(_template("exp_constant"), oscillatory),
        )
        for dataset in datasets
    }
    for run_number, recommendation in list(recommendations.items()):
        recommendations[run_number] = replace(
            recommendation,
            assessments=(
                recommendation.assessments[0],
                replace(recommendation.assessments[1], template=oscillatory),
            ),
        )
    portfolio = build_global_fit_wizard_candidate_portfolio(
        datasets,
        scope=scope,
        single_fit_recommendations_by_run=recommendations,
    )

    resolution = resolve_scope_for_datasets(datasets, scope)
    assert "Oscillatory" not in resolution.included_set
    assert [template.key for template in portfolio.templates] == ["exp_constant"]


def test_multiplet_templates_are_protected_from_effort_tier_trimming() -> None:
    datasets = _exp_series(2)
    recommendations = {
        int(dataset.run_number): _recommendation(
            {"exp_constant": 10.0, "oscillatory3_gaussian_constant": 12.0}
        )
        for dataset in datasets
    }
    portfolio = build_global_fit_wizard_candidate_portfolio(
        datasets,
        single_fit_recommendations_by_run=recommendations,
    )

    assert portfolio.pattern_template_keys == ("oscillatory3_gaussian_constant",)


def test_user_frequencies_no_longer_change_the_preview_portfolio() -> None:
    """The preview quotes the scope's families; frequencies land in phase 1 instead.

    The cross-run peak vote that used to fold a user's declared lines into the
    portfolio is gone: those frequencies now reach the per-run single-fit wizard,
    whose assessed templates become the series alphabet.
    """
    datasets = _fmuf_series(2)
    portfolio = build_global_fit_wizard_candidate_portfolio(datasets)

    assert portfolio.pattern_template_keys == ()
