"""Unit tests for the shared fit_result_summary helper (Phase 2)."""

from __future__ import annotations

import pytest
from scipy import stats

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import (
    fit_result_summary,
    parameters_at_bound,
)


def test_summary_extracts_values_and_uncertainties():
    params = ParameterSet([Parameter("lambda", 0.5), Parameter("A", 0.2)])
    result = FitResult(
        success=True,
        chi_squared=12.0,
        reduced_chi_squared=1.1,
        parameters=params,
        uncertainties={"lambda": 0.01, "A": 0.02},
        objective=Objective.GAUSSIAN,
    )

    summary = fit_result_summary(result)

    assert summary["success"] is True
    assert summary["chi_squared"] == 12.0
    assert summary["reduced_chi_squared"] == 1.1
    assert summary["parameters"] == {"lambda": 0.5, "A": 0.2}
    assert summary["uncertainties"] == {"lambda": 0.01, "A": 0.02}


def test_quality_band_tracks_configurable_confidence():
    """The χ² quality band edges are chi2 quantiles at the requested confidence,
    and a non-default confidence can flip the verdict (Item 4)."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 41
    # chi2_r = 50/41 ~= 1.22: inside the 95% band ("good"), outside a tight 60%.
    result = FitResult(
        success=True,
        chi_squared=50.0,
        reduced_chi_squared=50.0 / dof,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    q95 = fit_result_summary(result, confidence=0.95)["quality"]
    q60 = fit_result_summary(result, confidence=0.60)["quality"]

    assert q95["confidence"] == pytest.approx(0.95)
    assert q60["confidence"] == pytest.approx(0.60)
    # Band edges are exactly the two-sided chi2 quantiles over dof.
    for q, R in ((q95, 0.95), (q60, 0.60)):
        assert q["band_low"] == pytest.approx(stats.chi2.ppf((1 - R) / 2, dof) / dof, rel=1e-9)
        assert q["band_high"] == pytest.approx(stats.chi2.ppf((1 + R) / 2, dof) / dof, rel=1e-9)
    # Higher confidence => wider band => more forgiving verdict.
    assert q60["band_low"] > q95["band_low"]
    assert q60["band_high"] < q95["band_high"]
    assert q95["verdict"] == "good"
    assert q60["verdict"] == "poor"
    # Default confidence is the muon-tuned 0.999 (corpus finding #6), not 0.95.
    assert fit_result_summary(result)["quality"]["confidence"] == pytest.approx(0.999)


# --- muon-tuned default: routine high-ν fits read "good", not red "poor" ------


def test_routine_high_ndof_muon_fit_is_good_at_default():
    """Live regression (corpus finding #6): a routine time-domain fit with
    χ²ᵣ≈1.21 at ν≈488 was labelled "poor" in red at WiMDA's 0.95 band. At the
    muon-tuned 0.999 default it reads "good" (band ≈ [0.80, 1.22]) — excellent
    high-statistics muon fits are no longer alarmed."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 488
    chi2 = 1.21 * dof
    result = FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "good"
    assert quality["marginal"] is False


def test_cuprate_high_ndof_near_unity_chi2_is_good_at_default():
    """The cuprate case: χ²ᵣ≈1.10 at ν≈1927 is also "good" at the 0.999 default
    (band ≈ [0.90, 1.11]) — at the old 0.95 band it read "poor" and #139 could
    only soften it to "poor (marginal)" amber; now it is genuinely good."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 1927
    chi2 = 1.10 * dof
    result = FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "good"
    assert quality["marginal"] is False


# --- (a) near-unity "marginal" softening: safety net at extreme ν ------------


def test_extreme_ndof_near_unity_chi2_is_poor_but_marginal():
    """The #139 ``marginal`` softening still guards the residual edge: at very
    large ν even the 0.999 band is tighter than ±0.2, so a near-unity χ²ᵣ≈1.15 at
    ν≈2000 reads "poor" yet is flagged ``marginal`` for a softer amber chip. The
    verdict itself is unchanged — WiMDA band math is preserved."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 2000
    chi2 = 1.15 * dof
    result = FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "poor"  # band math unchanged (parity preserved)
    assert quality["marginal"] is True


def test_genuinely_high_chi2_after_rebin_is_poor_not_marginal():
    """A bunched fit with χ²ᵣ≈8 (LiFeAs) is an honest "poor": rebin propagates
    errors correctly, so we never soften it to marginal or rescale the errors."""
    params = ParameterSet([Parameter("sigma", 0.3)])
    dof = 200
    chi2 = 8.0 * dof
    result = FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=params,
        uncertainties={"sigma": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "poor"
    assert quality["marginal"] is False


def test_overdone_verdict_is_never_marginal():
    """A near-unity χ²ᵣ in the lower tail reads "overdone" at high ν; it is left
    as-is (non-alarming accent), never softened to "marginal"."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 1927
    chi2 = 0.88 * dof  # below the tight 0.999 band (≈0.90) -> overdone, |χ²ᵣ-1|=0.12
    result = FitResult(
        success=True,
        chi_squared=chi2,
        reduced_chi_squared=chi2 / dof,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "overdone"
    assert quality["marginal"] is False


def test_good_fit_is_not_marginal():
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 100
    result = FitResult(
        success=True,
        chi_squared=1.0 * dof,
        reduced_chi_squared=1.0,
        parameters=params,
        uncertainties={"A": 0.01},
        objective=Objective.GAUSSIAN,
        dof=dof,
    )
    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "good"
    assert quality["marginal"] is False


# --- (b) parameters-at-bound detection ---------------------------------------


def test_params_at_bound_flags_free_param_on_its_max():
    params = ParameterSet(
        [
            Parameter("A", 0.2, min=0.0, max=1.0),  # interior
            Parameter("r", 2.5, min=1.0, max=2.5),  # railed to max (FµF case)
        ]
    )
    assert parameters_at_bound(params) == ["r"]
    # And it rides along on the full summary.
    result = FitResult(
        success=True,
        chi_squared=50.0,
        reduced_chi_squared=1.2,
        parameters=params,
        uncertainties={},
        objective=Objective.GAUSSIAN,
        dof=41,
    )
    assert fit_result_summary(result)["params_at_bound"] == ["r"]


def test_params_at_bound_flags_param_on_its_min():
    # Maleic A_Mu → 0 lower bound; near-rail within tolerance also fires.
    params = ParameterSet(
        [
            Parameter("A_Mu", 0.0, min=0.0, max=0.3),
            Parameter("Delta", 0.9999, min=0.0, max=1.0),  # ZF-KT Δ→1.0 near-rail
        ]
    )
    flagged = parameters_at_bound(params)
    assert set(flagged) == {"A_Mu", "Delta"}


def test_clean_interior_fit_has_no_bound_flag():
    params = ParameterSet(
        [Parameter("A", 0.2, min=0.0, max=1.0), Parameter("lambda", 0.5, min=0.0, max=5.0)]
    )
    assert parameters_at_bound(params) == []
    result = FitResult(
        success=True,
        chi_squared=50.0,
        reduced_chi_squared=1.2,
        parameters=params,
        uncertainties={},
        objective=Objective.GAUSSIAN,
        dof=41,
    )
    assert fit_result_summary(result)["params_at_bound"] == []


def test_user_fixed_param_at_value_is_not_flagged():
    # A parameter the user FIXED is meant to hold its value — never flag it,
    # even though its value sits on what would be a bound.
    params = ParameterSet(
        [
            Parameter("A", 0.2, min=0.0, max=1.0),
            Parameter("field_1", 2.5, min=1.0, max=2.5, fixed=True),
        ]
    )
    assert parameters_at_bound(params) == []


def test_param_with_infinite_bounds_is_not_flagged():
    # Default ±inf bounds (unbounded free param) can never be "at bound".
    params = ParameterSet([Parameter("c", 0.0)])  # min=-inf, max=+inf
    assert parameters_at_bound(params) == []


def test_tied_and_linked_followers_are_not_flagged():
    # Equality-link followers drop out of the free set, so a follower sitting on
    # a bound is not an unconstrained-rail signal.
    params = ParameterSet(
        [
            Parameter("A1", 1.0, min=0.0, max=1.0, link_group=1),
            Parameter("A2", 1.0, min=0.0, max=1.0, link_group=1),
        ]
    )
    # Only the link main (A1) is free; A2 follows. Detection ranges over the
    # free set, so at most the main is considered.
    flagged = parameters_at_bound(params)
    assert "A2" not in flagged


@pytest.mark.parametrize(
    "minimum,has_quality", [(4.999, False), (5.0, True), (20.0, True), (None, False)]
)
def test_poisson_quality_requires_recorded_dense_expected_counts(minimum, has_quality):
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.likelihood import Objective

    result = FitResult(
        success=True,
        objective=Objective.POISSON,
        chi_squared=10.0,
        reduced_chi_squared=1.0,
        dof=10,
        minimum_expected_count=minimum,
    )
    summary = fit_result_summary(result)
    assert summary["objective"] == "poisson"
    assert (summary["quality"] is not None) is has_quality
    if not has_quality:
        assert summary["quality_reason"]


def test_gaussian_quality_marks_its_approximation():
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.likelihood import Objective

    summary = fit_result_summary(
        FitResult(
            success=True,
            objective=Objective.GAUSSIAN,
            chi_squared=10.0,
            reduced_chi_squared=1.0,
            dof=10,
        )
    )
    assert summary["objective"] == "gaussian"
    assert "approximate for nonlinear or bounded fits" in summary["quality"]["assumption"]


@pytest.mark.parametrize(
    "objective,label", [("gaussian", "χ²/ν"), ("poisson", "Poisson deviance/ν")]
)
def test_reduced_statistic_summaries_keep_likelihoods_separate(objective, label):
    from asymmetry.core.fitting.result_summary import reduced_statistic_summary

    rows = [{"objective": objective, "reduced_chi_squared": value} for value in [1.0, 2.0]]
    assert reduced_statistic_summary(rows) == f"{label} 1–2"
    assert reduced_statistic_summary(rows, mean=True) == f"average {label} = 1.5000"
    rows[0]["objective"] = None
    assert reduced_statistic_summary(rows) == "Mixed fit objectives."
    rows[1]["objective"] = None
    assert reduced_statistic_summary(rows) == "Fit objective was not recorded."


@pytest.mark.parametrize("strategy", ["joint", "profiled"])
def test_coupled_poisson_quality_requires_dense_expectations_for_the_whole_objective(strategy):
    import numpy as np

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.engine import FitEngine
    from asymmetry.core.fitting.likelihood import CountSource, PoissonCounts

    datasets = [
        MuonDataset(
            np.full(4, index, dtype=float),
            np.full(4, counts, dtype=float),
            np.ones(4),
            {"run_number": index},
        )
        for index, counts in [(1, 20), (2, 0)]
    ]
    observations = {
        ds.run_number: PoissonCounts(ds.time, ds.asymmetry, CountSource.RAW) for ds in datasets
    }
    parameters = {ds.run_number: ParameterSet([Parameter("c", 20.0, min=0.0)]) for ds in datasets}

    def model(time, c):
        return c * np.where(time == 1, 1.0, 0.01)

    results, _ = FitEngine().global_fit(
        datasets, model, ["c"], [], parameters, observations=observations, strategy=strategy
    )
    assert all(result.success for result in results.values())
    for result in results.values():
        assert result.minimum_expected_count == pytest.approx(0.19801980198019803, abs=1e-4)
        assert fit_result_summary(result)["quality"] is None


def test_forward_backward_card_uses_the_sparse_bank_in_its_quality_policy():
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
    from asymmetry.core.fitting.count_domain import fit_fb_alpha
    from asymmetry.core.utils.constants import MUON_LIFETIME_US
    from tests.core.test_count_domain_fits import _fb_seed_params, _tf

    time = np.arange(20) * 0.1
    run = Run(
        run_number=1,
        histograms=[
            Histogram(np.round(100 * np.exp(-time / MUON_LIFETIME_US)), 0.1, 0),
            Histogram(np.zeros(20), 0.1, 0),
        ],
        grouping={"groups": {1: [0], 2: [1]}, "first_good_bin": 0, "last_good_bin": 19},
    )
    dataset = MuonDataset(time, np.zeros(20), np.ones(20), run=run)
    parameters = _fb_seed_params(n0=10.0)
    for parameter in parameters:
        parameter.fixed = parameter.name != "N0"
    parameters["alpha"].value = 100.0
    parameters["A"].value = 0.0
    result = fit_fb_alpha(dataset, 1, 2, _tf, parameters, cost="poisson")
    assert result.success
    forward, backward = result.group_results[1], result.group_results[2]
    assert forward.minimum_expected_count == backward.minimum_expected_count
    assert 0 < forward.minimum_expected_count < 1
    assert fit_result_summary(forward)["quality"] is None


def test_summary_is_json_serializable_with_numpy_scalar_diagnostics():
    import json

    import numpy as np

    result = FitResult(
        success=np.bool_(True),
        objective=Objective.POISSON,
        chi_squared=np.float32(10.0),
        reduced_chi_squared=np.float32(1.0),
        minimum_expected_count=np.float32(3.0),
        dof=10,
    )
    summary = fit_result_summary(result)
    assert json.loads(json.dumps(summary)) == summary
