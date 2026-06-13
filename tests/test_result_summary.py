"""Unit tests for the shared fit_result_summary helper (Phase 2)."""

from __future__ import annotations

import pytest
from scipy import stats

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary


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
