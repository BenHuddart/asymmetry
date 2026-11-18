"""Spectral caveats qualify existing numbers without changing their estimator."""

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.result_payload import _deserialize_fit_result, _serialize_fit_result
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fourier.moments import moments_trend_row, spectrum_moments


def test_moment_recipe_and_trend_retain_approximation_scope():
    x = np.linspace(-2, 2, 41)
    y = np.exp(-x * x)
    result = spectrum_moments(
        x,
        y,
        x_range=None,
        cutoff_fraction=0.1,
        errors=np.full(len(x), 0.01),
        n_bootstrap=16,
        seed=8,
        error_oversampling=4,
    )
    assert "independent spectral-bin noise" in result.recipe["uncertainty_scope"]
    assert result.recipe["error_oversampling"] == 4
    row = moments_trend_row(result, run_number=1)
    assert row["warnings"] == [result.recipe["uncertainty_scope"]]
    assert row["moments_recipe"] == result.recipe


def test_frequency_fit_warns_at_padding_one_and_survives_payload():
    x = np.linspace(-2, 2, 41)
    datasets = [
        MuonDataset(time=x, asymmetry=2 * x + 3, error=np.full(len(x), 0.1), metadata=meta)
        for meta in ({}, {"plot_domain": "frequency"})
    ]

    def model(x, **p):
        return p["m"] * x + p["b"]

    results = [
        FitEngine().fit(
            ds, model, ParameterSet([Parameter(name="m", value=1), Parameter(name="b", value=2)])
        )
        for ds in datasets
    ]
    assert not results[0].warnings
    assert results[0].input_attribution["identity"] == results[1].input_attribution["identity"]
    assert results[0].uncertainties == results[1].uncertainties
    assert any("Approximate spectral fit" in w for w in results[1].warnings)
    np.testing.assert_array_equal(results[0].covariance, results[1].covariance)
    assert results[0].chi_squared == results[1].chi_squared
    restored = _deserialize_fit_result(_serialize_fit_result(results[1]))
    assert fit_result_summary(restored)["warnings"] == results[1].warnings


def test_unknown_historical_scope_is_not_invented():
    from asymmetry.core.fitting.likelihood import Objective
    from asymmetry.core.fitting.results import FitResult

    result = FitResult(
        success=True, objective=Objective.GAUSSIAN, input_attribution={"error_oversampling": 4}
    )
    assert not result.warnings
    values = spectrum_moments(
        [1, 2, 3], [1, 2, 1], x_range=None, cutoff_fraction=0, uncertainty="none"
    )
    assert "uncertainty_scope" not in values.recipe
    assert moments_trend_row(values, run_number=1)["warnings"] == []
