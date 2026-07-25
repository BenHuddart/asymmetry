"""Tests for the per-component resolution / identifiability diagnostic.

Every dataset here is synthesized in the test from a stated truth, so the
verdicts are checked against something known rather than against a stored fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.resolution import (
    ResolutionVerdict,
    assess_component_resolution,
    relaxation_rate_parameter_names,
)

BIEXP = CompositeModel.from_expression("Exponential + Exponential + Constant")
SINGLE_EXP = CompositeModel.from_expression("Exponential + Constant")


def _dataset(time: np.ndarray, asymmetry: np.ndarray, noise: float, seed: int) -> MuonDataset:
    rng = np.random.default_rng(seed)
    return MuonDataset(
        time=time,
        asymmetry=asymmetry + rng.normal(0.0, noise, time.size),
        error=np.full_like(time, noise),
        metadata={"run_number": 1, "run_label": "1", "temperature": 20.0, "field": 0.0},
    )


def _biexp_seed(a_1: float, lam_1: float, a_2: float, lam_2: float, bg: float) -> ParameterSet:
    return ParameterSet(
        [
            Parameter("A_1", a_1, min=-1.0, max=1.0),
            Parameter("Lambda_1", lam_1, min=1e-4, max=500.0),
            Parameter("A_2", a_2, min=-1.0, max=1.0),
            Parameter("Lambda_2", lam_2, min=1e-4, max=500.0),
            Parameter("A_bg", bg, min=-1.0, max=1.0),
        ]
    )


def _verdicts(assessment) -> dict[str, ResolutionVerdict]:
    return {component.parameter_name: component.verdict for component in assessment.components}


@pytest.fixture
def time_axis() -> np.ndarray:
    return np.linspace(0.0, 8.0, 800)


def test_rate_parameters_come_from_the_component_registry() -> None:
    """Rate parameters are the ``µs⁻¹`` ones; amplitudes and backgrounds are not."""
    assert relaxation_rate_parameter_names(BIEXP) == ("Lambda_1", "Lambda_2")
    assert relaxation_rate_parameter_names(SINGLE_EXP) == ("Lambda",)
    assert relaxation_rate_parameter_names(CompositeModel.from_expression("Constant")) == ()


def test_genuine_two_component_fit_is_resolved(time_axis: np.ndarray) -> None:
    """Both branches sit well inside the resolvable window and stay there."""
    truth = 0.075 * np.exp(-3.2 * time_axis) + 0.025 * np.exp(-0.42 * time_axis) + 0.012
    dataset = _dataset(time_axis, truth, 0.0016, seed=3)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.08, 3.0, 0.02, 0.5, 0.01))
    assert result.success

    assessment = assess_component_resolution(result, BIEXP, dataset)

    assert assessment.neighbourhood_source == "profile"
    assert assessment.is_resolved
    assert set(_verdicts(assessment).values()) == {ResolutionVerdict.RESOLVED}
    assert assessment.disqualification_reasons() == ()


def test_early_spike_rails_a_branch_past_the_binning(time_axis: np.ndarray) -> None:
    """A few-bin excess is absorbed by a branch whose 1/e time is sub-resolution.

    The truth is a single slow exponential plus an excess confined to the first
    three bins — no second *rate* exists. The two-exponential fit still converges
    happily, and the fast branch is the pathology the rule exists to name.
    """
    truth = 0.06 * np.exp(-0.5 * time_axis) + 0.01
    truth = truth.copy()
    truth[:3] += 0.03
    dataset = _dataset(time_axis, truth, 0.0016, seed=5)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.03, 60.0, 0.06, 0.5, 0.01))
    assert result.success

    assessment = assess_component_resolution(result, BIEXP, dataset)
    verdicts = _verdicts(assessment)

    assert verdicts["Lambda_1"] is ResolutionVerdict.UNRESOLVED_FAST
    assert verdicts["Lambda_2"] is ResolutionVerdict.RESOLVED
    assert not assessment.is_resolved
    railed = next(c for c in assessment.components if c.parameter_name == "Lambda_1")
    assert railed.bins_per_e_folding < assessment.min_bins_per_e_folding
    assert "not resolved at this binning" in railed.detail


def test_flat_ridge_is_undetermined_not_resolved(time_axis: np.ndarray) -> None:
    """A two-exponential fit to one-exponential data leaves the rates undetermined.

    Both branches land at finite, individually plausible rates — a point-estimate
    rule would call them resolved — but the Δχ² ≤ 1 neighbourhood runs the whole
    width of the resolvable window, so neither is measured.
    """
    truth = 0.07 * np.exp(-1.1 * time_axis) + 0.01
    dataset = _dataset(time_axis, truth, 0.0025, seed=13)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.07, 1.1, 0.004, 8.0, 0.01))
    assert result.success

    assessment = assess_component_resolution(result, BIEXP, dataset)
    verdicts = _verdicts(assessment)

    assert set(verdicts.values()) == {ResolutionVerdict.UNDETERMINED}
    assert not assessment.is_resolved
    for component in assessment.components:
        low, high = component.admissible_rate_range
        slow_edge, fast_edge = component.resolved_rate_window
        # The point estimate is inside the window; the neighbourhood is not.
        assert slow_edge <= component.rate <= fast_edge
        assert low < slow_edge or high > fast_edge
        assert "undetermined" in component.detail


def test_point_estimate_mode_is_the_documented_fallback(time_axis: np.ndarray) -> None:
    """Without a probe or solutions the verdict is the (weaker) point-estimate one."""
    truth = 0.07 * np.exp(-1.1 * time_axis) + 0.01
    dataset = _dataset(time_axis, truth, 0.0025, seed=13)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.07, 1.1, 0.004, 8.0, 0.01))

    assessment = assess_component_resolution(result, BIEXP, dataset, probe_neighbourhood=False)

    assert assessment.neighbourhood_source == "point"
    # Same fit, same rates — but with no neighbourhood explored the ridge is
    # invisible and every branch reads as resolved. This is exactly why the
    # probe is the default.
    assert set(_verdicts(assessment).values()) == {ResolutionVerdict.RESOLVED}
    for component in assessment.components:
        assert component.admissible_rate_range is None


def test_supplied_multistart_solutions_drive_the_verdict(time_axis: np.ndarray) -> None:
    """A caller's own multistart solution list is an accepted neighbourhood source."""
    truth = 0.075 * np.exp(-3.2 * time_axis) + 0.025 * np.exp(-0.42 * time_axis) + 0.012
    dataset = _dataset(time_axis, truth, 0.0016, seed=3)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.08, 3.0, 0.02, 0.5, 0.01))
    best = result.chi_squared

    # One statistically indistinguishable solution puts Lambda_1 at 200 µs⁻¹ —
    # the optimizer-dependence the neighbourhood rule exists to absorb.
    solutions = [
        (best, {"Lambda_1": 3.16, "Lambda_2": 0.39}),
        (best + 0.4, {"Lambda_1": 200.0, "Lambda_2": 0.39}),
        (best + 9.0, {"Lambda_1": 0.001, "Lambda_2": 0.39}),
    ]
    assessment = assess_component_resolution(result, BIEXP, dataset, solutions=solutions)

    assert assessment.neighbourhood_source == "solutions"
    verdicts = _verdicts(assessment)
    assert verdicts["Lambda_1"] is ResolutionVerdict.UNDETERMINED
    # The Δχ² = 9 solution is outside the neighbourhood and must not contribute.
    lambda_1 = next(c for c in assessment.components if c.parameter_name == "Lambda_1")
    assert lambda_1.admissible_rate_range[0] > 0.001


def test_slow_edge_is_two_sided(time_axis: np.ndarray) -> None:
    """A branch slower than the fit window is unresolved too, not just fast ones."""
    truth = 0.075 * np.exp(-3.2 * time_axis) + 0.03
    dataset = _dataset(time_axis, truth, 0.0016, seed=21)
    # Seed the second branch at a rate whose 1/e time far exceeds the window; it
    # stays there because it is degenerate with the free baseline.
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.075, 3.2, 0.03, 0.002, 0.0))
    assert result.success

    assessment = assess_component_resolution(result, BIEXP, dataset, probe_neighbourhood=False)
    verdicts = _verdicts(assessment)

    assert verdicts["Lambda_2"] is ResolutionVerdict.UNRESOLVED_SLOW
    slow = next(c for c in assessment.components if c.parameter_name == "Lambda_2")
    assert "degenerate with a constant baseline" in slow.detail


def test_min_bins_per_e_folding_is_configurable(time_axis: np.ndarray) -> None:
    """Tightening the fast edge moves the verdict, and the window reports it."""
    truth = 0.06 * np.exp(-0.5 * time_axis) + 0.01
    truth = truth.copy()
    truth[:3] += 0.03
    dataset = _dataset(time_axis, truth, 0.0016, seed=5)
    result = FitEngine().fit(dataset, BIEXP.function, _biexp_seed(0.03, 60.0, 0.06, 0.5, 0.01))

    permissive = assess_component_resolution(
        result, BIEXP, dataset, min_bins_per_e_folding=0.5, probe_neighbourhood=False
    )
    strict = assess_component_resolution(
        result, BIEXP, dataset, min_bins_per_e_folding=200.0, probe_neighbourhood=False
    )

    assert _verdicts(permissive)["Lambda_1"] is ResolutionVerdict.RESOLVED
    assert _verdicts(strict)["Lambda_1"] is ResolutionVerdict.UNRESOLVED_FAST
    assert (
        permissive.components[0].resolved_rate_window[1]
        > (strict.components[0].resolved_rate_window[1])
    )


def test_model_without_relaxation_channels_is_vacuously_resolved(
    time_axis: np.ndarray,
) -> None:
    model = CompositeModel.from_expression("Constant")
    dataset = _dataset(time_axis, np.full_like(time_axis, 0.02), 0.001, seed=1)
    result = FitEngine().fit(
        dataset, model.function, ParameterSet([Parameter("A_bg", 0.02, min=-1.0, max=1.0)])
    )

    assessment = assess_component_resolution(result, model, dataset)

    assert assessment.components == ()
    assert assessment.is_resolved


def test_requires_a_time_axis() -> None:
    result = FitEngine().fit(
        _dataset(np.linspace(0.0, 1.0, 10), np.zeros(10), 0.01, seed=0),
        SINGLE_EXP.function,
        ParameterSet(
            [
                Parameter("A_1", 0.1, min=-1.0, max=1.0),
                Parameter("Lambda", 1.0, min=1e-4, max=100.0),
                Parameter("A_bg", 0.0, min=-1.0, max=1.0),
            ]
        ),
    )
    with pytest.raises(ValueError, match="dataset or an explicit time axis"):
        assess_component_resolution(result, SINGLE_EXP)


# --------------------------------------------------------------------------- #
# Wizard integration
# --------------------------------------------------------------------------- #


def test_wizard_disqualifies_an_over_parameterised_candidate(time_axis: np.ndarray) -> None:
    """A candidate whose extra branch is undetermined must not win the table.

    The truth has exactly two exponential components. Richer candidates fit it
    just as well — AICc alone separates them by only a few units — but their
    extra branches are not measured, and the ranked table now says so.
    """
    from asymmetry.core.fitting.fit_wizard import (
        SelectionMetric,
        build_fit_wizard_recommendation,
    )
    from asymmetry.core.fitting.wizard_scope import WizardScope, WizardScopePreset

    truth = 0.075 * np.exp(-3.2 * time_axis) + 0.025 * np.exp(-0.42 * time_axis) + 0.012
    dataset = _dataset(time_axis, truth, 0.0016, seed=3)

    recommendation = build_fit_wizard_recommendation(
        dataset,
        metric=SelectionMetric.AICC,
        scope=WizardScope(preset=WizardScopePreset.ZF_STATIC_MAGNETISM),
        max_workers=1,
    )

    two_component = recommendation.assessment_for_key("biexp_constant")
    three_component = recommendation.assessment_for_key("triple_exp_constant")
    assert two_component is not None and three_component is not None

    # The true model is clean...
    assert not two_component.disqualification_reasons
    # ...and the redundant third branch is named — either railed past the
    # binning or simply undetermined, both of which mean "not a measured rate" —
    # rather than silently rewarded for the χ² it buys.
    assert three_component.is_disqualified
    assert any(
        "is not resolved" in reason or "is undetermined" in reason
        for reason in three_component.disqualification_reasons
    )
    assert recommendation.recommended_key == "biexp_constant"


def test_wizard_skips_the_check_for_single_channel_candidates(
    time_axis: np.ndarray,
) -> None:
    """The probe is a composite-identifiability check, and costs fits — so it is gated."""
    from asymmetry.core.fitting.fit_wizard import (
        CandidateTemplate,
        component_resolution_for_assessment,
    )

    truth = 0.07 * np.exp(-1.1 * time_axis) + 0.01
    dataset = _dataset(time_axis, truth, 0.0025, seed=13)
    template = CandidateTemplate(
        key="exp_constant",
        title="exp_constant",
        category="Relaxation",
        rationale="test",
        model=SINGLE_EXP,
    )
    result = FitEngine().fit(
        dataset,
        SINGLE_EXP.function,
        ParameterSet(
            [
                Parameter("A_1", 0.07, min=-1.0, max=1.0),
                Parameter("Lambda", 1.1, min=1e-4, max=500.0),
                Parameter("A_bg", 0.01, min=-1.0, max=1.0),
            ]
        ),
    )

    assert component_resolution_for_assessment(dataset, template, result) is None
