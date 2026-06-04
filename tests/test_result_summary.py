"""Unit tests for the shared fit_result_summary helper (Phase 2)."""

from __future__ import annotations

from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.result_summary import fit_result_summary


class _FakeResult:
    def __init__(self, success, chi, red, parameters, uncertainties):
        self.success = success
        self.chi_squared = chi
        self.reduced_chi_squared = red
        self.parameters = parameters
        self.uncertainties = uncertainties


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
