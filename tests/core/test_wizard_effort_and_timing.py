"""Effort tiers, seeding parity, and the wizards' timing instrumentation.

These cover the three defects a downstream headless caller hit: screening that
could not be bought cheaper at any effort tier, a seed ladder that did not scale
with a family's rate dimensionality, and no way to tell a slow run from a hung
one without reading the process tree.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateTemplate,
    SelectionMetric,
    _additive_relaxation_mixture_variants,
    _initial_parameters_for_template,
    _rate_dimension,
    _stage2_variant_budget,
    build_fit_wizard_recommendation,
    build_fit_wizard_recommendation_for_templates,
    fingerprint_spectrum,
    single_fit_build_signature,
)
from asymmetry.core.fitting.global_fit_wizard import (
    _screening_no_recommendation_summary,
    build_global_fit_wizard_candidate_portfolio,
    build_global_fit_wizard_screening_recommendation,
    build_or_complete_single_fit_wizard_recommendations_for_global_portfolio,
    screening_templates_for_effort_tier,
)
from asymmetry.core.fitting.wizard_scope import (
    EffortTier,
    WizardScope,
    WizardScopePreset,
)
from asymmetry.core.fitting.wizard_timing import (
    TIMING_KEY,
    WizardStageProgress,
    stage_timer,
    timing_block,
)

ZF_SCOPE = WizardScope(preset=WizardScopePreset.ZF_STATIC_MAGNETISM)


def _series(count: int = 4, points: int = 220, seed: int = 0) -> list[MuonDataset]:
    """A synthetic two-component relaxation series, one dataset per temperature."""
    rng = np.random.default_rng(seed)
    datasets: list[MuonDataset] = []
    for index in range(count):
        time = np.linspace(0.0, 6.0, points)
        fast = 3.0 * np.exp(-index / 9.0) + 0.4
        slow = 0.55 * np.exp(-index / 12.0) + 0.05
        asymmetry = (
            0.10 * (0.75 * np.exp(-fast * time) + 0.25 * np.exp(-slow * time))
            + 0.01
            + rng.normal(0.0, 0.0025, time.size)
        )
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=asymmetry,
                error=np.full_like(time, 0.0025),
                metadata={
                    "run_number": 1000 + index,
                    "temperature": 20.0 + 6.0 * index,
                    "field": 0.0,
                    "run_label": str(1000 + index),
                },
            )
        )
    return datasets


def _template(expression: str, key: str) -> CandidateTemplate:
    return CandidateTemplate(
        key=key,
        title=key,
        category="Relaxation",
        rationale="test",
        model=CompositeModel.from_expression(expression),
    )


def _stub_phase_one(
    monkeypatch: pytest.MonkeyPatch,
    templates: tuple[CandidateTemplate, ...],
) -> None:
    """Replace phase 1's per-run single-run wizard with a fixed template table.

    Phase 1 runs the whole single-run Fit Wizard on every dataset. That is the
    behaviour under test in ``tests/core/test_global_fit_wizard.py``; here it is
    only the fixture that produces a series alphabet, so a fixed, signed table
    stands in for it.
    """
    import asymmetry.core.fitting.global_fit_wizard as module

    def _fake_build(dataset, current_model=None, **kwargs):
        recommendation = build_fit_wizard_recommendation_for_templates(
            dataset,
            templates,
            metric=kwargs.get("metric", SelectionMetric.AICC),
        )
        return replace(
            recommendation,
            build_signature=single_fit_build_signature(
                kwargs.get("scope"), kwargs.get("user_frequencies_mhz")
            ),
        )

    monkeypatch.setattr(module, "build_fit_wizard_recommendation", _fake_build)


def _covering_tables(
    datasets: list[MuonDataset],
    templates: tuple[CandidateTemplate, ...],
) -> dict[int, object]:
    """A per-run score table covering ``templates`` — enough to skip phase 1."""
    return {
        int(dataset.run_number): build_fit_wizard_recommendation_for_templates(dataset, templates)
        for dataset in datasets
    }


# --------------------------------------------------------------------------- #
# Effort tiers actually buy a cheaper screen
# --------------------------------------------------------------------------- #


def test_low_and_balanced_tiers_prune_the_screened_portfolio() -> None:
    portfolio = build_global_fit_wizard_candidate_portfolio(_series(), scope=ZF_SCOPE)
    full = portfolio.templates
    assert len(full) > 10  # the portfolio this scope offers is what makes screening expensive

    low, low_skipped = screening_templates_for_effort_tier(full, EffortTier.LOW)
    balanced, balanced_skipped = screening_templates_for_effort_tier(full, EffortTier.BALANCED)
    thorough, thorough_skipped = screening_templates_for_effort_tier(full, EffortTier.THOROUGH)

    assert len(low) < len(balanced) < len(full)
    assert thorough_skipped == ()
    assert thorough == full
    # Nothing is lost: every template lands in exactly one of the two halves.
    for kept, skipped in ((low, low_skipped), (balanced, balanced_skipped)):
        assert len(kept) + len(skipped) == len(full)
        assert {t.key for t in kept} | {t.key for t in skipped} == {t.key for t in full}
    # Model order is preserved so two tiers' tables read the same way.
    assert [t.key for t in low] == [t.key for t in full if t in low]


def test_cheap_tiers_drop_the_numerically_expensive_candidates() -> None:
    """The dynamic Kubo-Toyabe family dominates screening cost; Low must shed it."""
    portfolio = build_global_fit_wizard_candidate_portfolio(_series(), scope=ZF_SCOPE)
    low, _skipped = screening_templates_for_effort_tier(portfolio.templates, EffortTier.LOW)

    kept = {template.key for template in low}
    assert "dynamic_lkt_constant" not in kept
    assert "dynamic_gkt_constant" not in kept
    # ...while the cheap workhorses survive, so the answer is coarser, not absent.
    assert "exp_constant" in kept
    assert "biexp_constant" in kept


def test_pattern_matched_templates_are_never_pruned() -> None:
    """A positively identified candidate would change the answer, not coarsen it."""
    portfolio = build_global_fit_wizard_candidate_portfolio(_series(), scope=ZF_SCOPE)
    expensive = next(
        template for template in portfolio.templates if template.key == "dynamic_lkt_constant"
    )

    low, skipped = screening_templates_for_effort_tier(
        portfolio.templates,
        EffortTier.LOW,
        pattern_template_keys=(expensive.key,),
    )

    assert expensive.key in {template.key for template in low}
    assert expensive.key not in {template.key for template in skipped}


def test_effort_tier_trims_the_series_alphabet(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tier now narrows the *alphabet* phase 1 produced, not a guessed portfolio.

    Low is compared against Balanced rather than Thorough: an unpruned screen of
    a full alphabet is exactly the multi-minute job the tiers exist to avoid, so
    running one here would make the test the thing it is testing.
    """
    datasets = _series(count=2, points=120)
    alphabet_source = tuple(
        _template(expression, key)
        for expression, key in (
            ("Exponential + Constant", "exp_constant"),
            ("Exponential + Exponential + Constant", "biexp_constant"),
            ("Gaussian + Constant", "gaussian_constant"),
            ("StretchedExponential + Constant", "stretched_constant"),
            ("Exponential + Gaussian + Constant", "exp_gaussian_constant"),
            ("Gaussian + Gaussian + Constant", "double_gaussian_constant"),
            ("Exponential + Exponential + Exponential + Constant", "triple_exp_constant"),
            ("Gaussian + Gaussian + Gaussian + Constant", "triple_gaussian_constant"),
        )
    )
    _stub_phase_one(monkeypatch, alphabet_source)
    instrumentation: dict[str, object] = {}

    low = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        effort_tier=EffortTier.LOW,
        instrumentation=instrumentation,
    )
    balanced = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        effort_tier=EffortTier.BALANCED,
    )

    assert len(low.portfolio.templates) < len(balanced.portfolio.templates)
    assert instrumentation["alphabet_size"] == len(low.portfolio.templates)
    # The skipped candidates are named, never silently dropped.
    skipped = instrumentation.get("screening_skipped_template_keys")
    assert isinstance(skipped, list) and skipped
    assert instrumentation.get("screening_effort_tier") == EffortTier.LOW.value


def test_multiplet_alphabet_entries_survive_the_cheapest_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peak-seeded multiplet is evidence, so no tier may trim it away."""
    datasets = _series(count=2, points=120)
    multiplet = _template(
        "Oscillatory * Exponential + Oscillatory * Exponential + Constant",
        "oscillatory2_exp_constant",
    )
    alphabet_source = (
        multiplet,
        *(
            _template(expression, key)
            for expression, key in (
                ("Exponential + Constant", "exp_constant"),
                ("Exponential + Exponential + Constant", "biexp_constant"),
                ("Gaussian + Constant", "gaussian_constant"),
                ("StretchedExponential + Constant", "stretched_constant"),
                ("Exponential + Gaussian + Constant", "exp_gaussian_constant"),
                ("Gaussian + Gaussian + Constant", "double_gaussian_constant"),
                ("Exponential + Exponential + Exponential + Constant", "triple_exp_constant"),
            )
        ),
    )
    _stub_phase_one(monkeypatch, alphabet_source)

    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(
        datasets,
        effort_tier=EffortTier.LOW,
    )

    assert "oscillatory2_exp_constant" in table.portfolio.pattern_template_keys
    assert "oscillatory2_exp_constant" in {t.key for t in table.portfolio.templates}


# --------------------------------------------------------------------------- #
# The "no recommendation" outcome names itself
# --------------------------------------------------------------------------- #


def _screened(datasets: list[MuonDataset], monkeypatch: pytest.MonkeyPatch):
    """A screening recommendation over a small fixed alphabet, without phase 1."""
    templates = tuple(
        _template(expression, key)
        for expression, key in (
            ("Exponential + Constant", "exp_constant"),
            ("Gaussian + Constant", "gaussian_constant"),
            ("Exponential + Exponential + Constant", "biexp_constant"),
        )
    )
    _stub_phase_one(monkeypatch, templates)
    table = build_or_complete_single_fit_wizard_recommendations_for_global_portfolio(datasets)
    return build_global_fit_wizard_screening_recommendation(
        datasets,
        metric=SelectionMetric.AICC,
        portfolio=table.portfolio,
        single_fit_recommendations_by_run=table.recommendations_by_run,
    )


def test_screening_summary_distinguishes_a_ranked_table_from_a_failed_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _series(count=3, points=120)
    recommendation = _screened(datasets, monkeypatch)

    # Screening never recommends: prescreen assessments are not evidence about a
    # coupled global fit. That is by design — and the summary now says so, and
    # names the candidate a caller should select.
    assert recommendation.recommended_key is None
    assert "candidates scored" in recommendation.summary
    assert "select one or more" in recommendation.summary.lower()

    scored = [
        assessment
        for assessment in recommendation.assessments
        if np.isfinite(assessment.metric_value(recommendation.metric))
    ]
    assert scored
    assert min(scored, key=lambda a: a.selected_score).template.key in recommendation.summary


def test_screening_summary_calls_an_unscoreable_table_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genuinely empty case must not read like an ordinary screen."""
    datasets = _series(count=3, points=120)
    recommendation = _screened(datasets, monkeypatch)

    broken = replace(
        recommendation,
        assessments=tuple(
            replace(
                assessment,
                aic=float("inf"),
                aicc=None,
                bic=float("inf"),
                selected_score=float("inf"),
                series_warnings=("Missing or failed single-fit assessment for run 1000.",),
            )
            for assessment in recommendation.assessments
        ),
    )

    summary = _screening_no_recommendation_summary(broken)

    assert "scored none of its" in summary
    assert "failed screen" in summary
    assert "run 1000" in summary


def test_screening_summary_reports_an_empty_portfolio_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _series(count=3, points=120)
    recommendation = _screened(datasets, monkeypatch)

    summary = _screening_no_recommendation_summary(replace(recommendation, assessments=()))

    assert "no candidate assessments at all" in summary
    assert "failure" in summary


# --------------------------------------------------------------------------- #
# Seeding parity
# --------------------------------------------------------------------------- #


def test_rate_dimension_counts_a_templates_independent_rates() -> None:
    assert _rate_dimension(_template("Exponential + Constant", "one")) == 1
    assert _rate_dimension(_template("Exponential + Exponential + Constant", "two")) == 2
    assert (
        _rate_dimension(_template("Exponential + Exponential + Gaussian + Constant", "three")) == 3
    )


def test_stage2_budget_scales_with_rate_dimensionality() -> None:
    """Comparable depth means comparable *per rate dimension*, not per seed count."""
    kwargs = dict(is_expensive=False, is_peak_seeded=False, family_key="relaxation", supported=True)

    one = _stage2_variant_budget(**kwargs, rate_dimension=1)
    two = _stage2_variant_budget(**kwargs, rate_dimension=2)
    three = _stage2_variant_budget(**kwargs, rate_dimension=3)

    assert one < two < three
    assert two - one == three - two  # a flat allowance per extra rate

    # Expensive members keep their smaller base but still scale: their search
    # space is not smaller just because each fit costs more.
    expensive_one = _stage2_variant_budget(**{**kwargs, "is_expensive": True}, rate_dimension=1)
    expensive_two = _stage2_variant_budget(**{**kwargs, "is_expensive": True}, rate_dimension=2)
    assert expensive_one < one
    assert expensive_two - expensive_one == two - one


def test_mixture_ladder_explores_rate_separation_not_just_overall_scale() -> None:
    """The extra parity seeds must buy new rate *ratios*, not repeats of one."""
    template = _template("Exponential + Exponential + Constant", "biexp_constant")
    dataset = _series(count=2, points=120)[0]
    base = _initial_parameters_for_template(
        dataset,
        fingerprint_spectrum(dataset),
        template,
    )

    variants = _additive_relaxation_mixture_variants(base, template)

    assert len(variants) >= _stage2_variant_budget(
        is_expensive=False,
        is_peak_seeded=False,
        family_key="relaxation",
        supported=True,
        rate_dimension=2,
    )
    ratios = sorted(
        abs(variant["Lambda_1"].value / variant["Lambda_2"].value) for variant in variants
    )
    # Distinct separations, and the widest is far beyond the original 3x ladder.
    assert len(set(round(ratio, 6) for ratio in ratios)) >= 5
    assert max(ratios) / min(ratios) > 100.0


def test_refinement_records_convergence_quality_on_the_top_candidates() -> None:
    """Every refined candidate carries a measured verdict on its own convergence."""
    time = np.linspace(0.0, 8.0, 400)
    rng = np.random.default_rng(5)
    asymmetry = (0.075 * np.exp(-3.2 * time) + 0.025 * np.exp(-0.42 * time) + 0.012) + rng.normal(
        0.0, 0.0018, time.size
    )
    dataset = MuonDataset(
        time=time,
        asymmetry=asymmetry,
        error=np.full_like(time, 0.0018),
        metadata={"run_number": 1, "run_label": "1", "temperature": 20.0, "field": 0.0},
    )

    shallow = build_fit_wizard_recommendation(
        dataset, scope=ZF_SCOPE, max_workers=1, refine_top_candidates=0
    )
    refined = build_fit_wizard_recommendation(dataset, scope=ZF_SCOPE, max_workers=1)

    assert all(
        assessment.refinement_delta_chi_squared == 0.0 and not assessment.under_converged
        for assessment in shallow.assessments
    )
    # The deeper ladder never loses: the reported score is the better of the two.
    for assessment in refined.assessments:
        shallow_assessment = shallow.assessment_for_key(assessment.template.key)
        if shallow_assessment is None or not assessment.is_successful:
            continue
        assert assessment.fit_result.chi_squared <= (
            shallow_assessment.fit_result.chi_squared + 1e-6
        )
        assert assessment.refinement_delta_chi_squared >= 0.0


# --------------------------------------------------------------------------- #
# Timing instrumentation
# --------------------------------------------------------------------------- #


def test_stage_timer_records_a_timing_block_and_emits_events() -> None:
    instrumentation: dict[str, object] = {}
    events: list[WizardStageProgress] = []

    with stage_timer(
        instrumentation,
        "demo.stage",
        items_total=2,
        stage_callback=events.append,
        message="demo",
    ) as advance:
        advance(1, "one done")
        advance(2, "two done")

    block = timing_block(instrumentation)
    assert block is not None
    assert block["elapsed_seconds"] >= 0.0
    assert block["cpu_seconds"] >= 0.0
    stages = block["stages"]
    assert [stage["stage"] for stage in stages] == ["demo.stage"]
    assert stages[0]["items_total"] == 2

    assert [event.event for event in events] == ["start", "item", "item", "end"]
    assert events[1].fraction_done == pytest.approx(0.5)
    assert events[-1].cpu_cores >= 0.0


def test_stage_timer_records_the_stage_even_when_it_raises() -> None:
    instrumentation: dict[str, object] = {}
    events: list[WizardStageProgress] = []

    with pytest.raises(RuntimeError):
        with stage_timer(instrumentation, "demo.boom", stage_callback=events.append):
            raise RuntimeError("boom")

    block = timing_block(instrumentation)
    assert block is not None
    assert [stage["stage"] for stage in block["stages"]] == ["demo.boom"]
    assert events[-1].event == "end"


def test_screening_populates_the_standard_timing_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _series(count=3, points=120)
    _stub_phase_one(
        monkeypatch,
        (
            _template("Exponential + Constant", "exp_constant"),
            _template("Gaussian + Constant", "gaussian_constant"),
        ),
    )
    instrumentation: dict[str, object] = {}
    events: list[WizardStageProgress] = []

    build_global_fit_wizard_screening_recommendation(
        datasets,
        metric=SelectionMetric.AICC,
        effort_tier=EffortTier.LOW,
        instrumentation=instrumentation,
        stage_callback=events.append,
    )

    block = instrumentation[TIMING_KEY]
    assert isinstance(block, dict)
    stage_names = {stage["stage"] for stage in block["stages"]}
    assert "screening.single_fit_tables" in stage_names
    assert "screening.completion_fits" in stage_names
    assert "screening.aggregate_assessments" in stage_names
    assert block["elapsed_seconds"] > 0.0
    # CPU is what distinguishes "slow" from "hung", so it must be present and
    # attributable — including the pool workers' share once they are reaped.
    assert block["cpu_seconds"] > 0.0
    assert block["cpu_cores"] > 0.0

    assert {event.stage for event in events} == stage_names
    assert events[0].event == "start"
    assert events[-1].event == "end"


def test_single_fit_wizard_populates_the_standard_timing_block() -> None:
    dataset = _series(count=2, points=200)[0]
    instrumentation: dict[str, object] = {}
    events: list[WizardStageProgress] = []

    build_fit_wizard_recommendation(
        dataset,
        scope=ZF_SCOPE,
        max_workers=1,
        instrumentation=instrumentation,
        stage_callback=events.append,
    )

    block = instrumentation[TIMING_KEY]
    stage_names = [stage["stage"] for stage in block["stages"]]
    assert "single_fit.stage1" in stage_names
    assert "single_fit.refinement" in stage_names
    # The segments between the fitting fan-outs are timed too. They used to be
    # invisible, and on a 10⁵-point record they were half the wall clock: the
    # envelope banks alone ran 19.9 s inside what the timing block reported as a
    # gap between Stage 1 and Stage 2.
    assert "single_fit.detection" in stage_names
    assert "single_fit.pattern_match" in stage_names
    assert block["cpu_seconds"] > 0.0
    assert all(event.elapsed_seconds >= 0.0 for event in events)
