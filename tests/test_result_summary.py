"""Unit tests for the shared fit_result_summary helper (Phase 2)."""

from __future__ import annotations

import pytest
from scipy import stats

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import (
    fit_result_summary,
    parameters_at_bound,
)


class _FakeResult:
    def __init__(self, success, chi, red, parameters, uncertainties, dof=None):
        self.success = success
        self.chi_squared = chi
        self.reduced_chi_squared = red
        self.parameters = parameters
        self.uncertainties = uncertainties
        if dof is not None:
            self.dof = dof


def test_summary_extracts_values_and_uncertainties():
    params = ParameterSet([Parameter("lambda", 0.5), Parameter("A", 0.2)])
    result = _FakeResult(True, 12.0, 1.1, params, {"lambda": 0.01, "A": 0.02})

    summary = fit_result_summary(result)

    assert summary["success"] is True
    assert summary["chi_squared"] == 12.0
    assert summary["reduced_chi_squared"] == 1.1
    assert summary["parameters"] == {"lambda": 0.5, "A": 0.2}
    assert summary["uncertainties"] == {"lambda": 0.01, "A": 0.02}


def test_summary_tolerates_missing_attributes():
    class _Bare:
        pass

    summary = fit_result_summary(_Bare())

    assert summary["success"] is False
    assert summary["chi_squared"] == 0.0
    assert summary["reduced_chi_squared"] == 0.0
    assert summary["parameters"] == {}
    assert summary["uncertainties"] == {}


def test_quality_band_tracks_configurable_confidence():
    """The χ² quality band edges are chi2 quantiles at the requested confidence,
    and a non-default confidence can flip the verdict (Item 4)."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 41
    # chi2_r = 50/41 ~= 1.22: inside the 95% band ("good"), outside a tight 60%.
    result = _FakeResult(True, 50.0, 50.0 / dof, params, {"A": 0.01}, dof=dof)

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
    # Default confidence is unchanged at WiMDA's Rgoodfit = 0.95.
    assert fit_result_summary(result)["quality"]["confidence"] == pytest.approx(0.95)


# --- (a) near-unity "marginal" softening at high ν ---------------------------


def test_high_ndof_near_unity_chi2_is_poor_but_marginal():
    """The cuprate case: χ²ᵣ≈1.10 at ν≈1927 is statistically "poor" (the band is
    tight) yet numerically near-ideal, so it is flagged ``marginal`` for a softer
    chip. The verdict itself is unchanged — WiMDA band math is preserved."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 1927
    chi2 = 1.10 * dof
    result = _FakeResult(True, chi2, chi2 / dof, params, {"A": 0.01}, dof=dof)

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "poor"  # band math unchanged (parity preserved)
    assert quality["marginal"] is True


def test_genuinely_high_chi2_after_rebin_is_poor_not_marginal():
    """A bunched fit with χ²ᵣ≈8 (LiFeAs) is an honest "poor": rebin propagates
    errors correctly, so we never soften it to marginal or rescale the errors."""
    params = ParameterSet([Parameter("sigma", 0.3)])
    dof = 200
    chi2 = 8.0 * dof
    result = _FakeResult(True, chi2, chi2 / dof, params, {"sigma": 0.01}, dof=dof)

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "poor"
    assert quality["marginal"] is False


def test_overdone_verdict_is_never_marginal():
    """A near-unity χ²ᵣ in the lower tail reads "overdone" at high ν; it is left
    as-is (non-alarming accent), never softened to "marginal"."""
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 1927
    chi2 = 0.90 * dof  # below the tight band -> overdone, |χ²ᵣ-1| = 0.10
    result = _FakeResult(True, chi2, chi2 / dof, params, {"A": 0.01}, dof=dof)

    quality = fit_result_summary(result)["quality"]
    assert quality["verdict"] == "overdone"
    assert quality["marginal"] is False


def test_good_fit_is_not_marginal():
    params = ParameterSet([Parameter("A", 0.2)])
    dof = 100
    result = _FakeResult(True, 1.0 * dof, 1.0, params, {"A": 0.01}, dof=dof)
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
    result = _FakeResult(True, 50.0, 1.2, params, {}, dof=41)
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
    result = _FakeResult(True, 50.0, 1.2, params, {}, dof=41)
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
