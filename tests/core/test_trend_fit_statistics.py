"""Applicability of trend residual statistics under the effective weights."""

import numpy as np
import pytest

from asymmetry.core.fitting.cross_group import run_cross_group_fit_from_config
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
    ParameterModelFitResult,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.trend_result import ColumnErrorSource


@pytest.mark.parametrize("scale", [1.0, 100.0])
@pytest.mark.parametrize(
    "mode, measured, effective",
    [
        (ErrorMode.COLUMN, False, ErrorMode.NONE),
        (ErrorMode.NONE, False, ErrorMode.NONE),
        (ErrorMode.SCATTER, False, ErrorMode.SCATTER),
        (ErrorMode.COLUMN, True, ErrorMode.COLUMN),
    ],
)
def test_fit_trend_quality_depends_on_effective_error_mode(scale, mode, measured, effective):
    x = np.arange(12.0)
    y = np.tile([scale, -scale], 6)
    result = fit_parameter_model(
        x,
        y,
        np.full_like(y, scale) if measured else None,
        ParameterCompositeModel(["Constant"]),
        ParameterSet([Parameter("c", 0.0)]),
        error_mode=mode,
    )

    assert result.success
    assert result.error_mode == mode.value
    assert result.effective_error_mode == effective.value
    assert result.chi_squared == pytest.approx(12.0 if measured else 12.0 * scale**2)
    assert result.parameters["c"].value == pytest.approx(0.0, abs=1e-6)
    assert result.dof == 11
    quality = result.assess_quality()
    if measured:
        assert quality.verdict == "good"
    else:
        assert quality is None
        assert result.quality_reason


def test_trend_preserves_x_errors_when_column_y_errors_are_missing():
    x = np.arange(4.0)
    y = np.array([0.0, 1.0, 4.0, 3.0])
    xerr = np.array([0.0, 0.0, 10.0, 0.0])
    model = ParameterCompositeModel(["Linear"])
    params = ParameterSet([Parameter("m", 1.0), Parameter("b", 0.0)])

    mixed = fit_parameter_model(x, y, None, model, params, xerr=xerr)
    measured = fit_parameter_model(x, y, np.ones_like(y), model, params, xerr=xerr)
    unit = fit_parameter_model(x, y, None, model, params, xerr=xerr, error_mode=ErrorMode.NONE)

    assert mixed.success and measured.success and unit.success
    slope = mixed.parameters["m"].value
    intercept = mixed.parameters["b"].value
    assert slope == pytest.approx(1.01078018, abs=5e-4)
    assert intercept == pytest.approx(-0.00795556, abs=5e-4)
    expected = np.sum((y - slope * x - intercept) ** 2 / (1 + slope**2 * xerr**2))
    assert mixed.chi_squared == pytest.approx(expected)
    assert measured.chi_squared == pytest.approx(expected)
    assert unit.chi_squared == pytest.approx(2.8)
    assert mixed.error_mode == ErrorMode.COLUMN.value
    assert mixed.effective_error_mode == ErrorMode.NONE.value
    assert mixed.uses_effective_variance is True
    assert mixed.assess_quality() is None
    assert "assumed unit y errors and supplied x errors" in mixed.quality_reason
    assert measured.assess_quality() is not None

    recovered = ParameterModelFitResult.from_dict(mixed.to_dict())
    assert recovered.column_error_source is ColumnErrorSource.ASSUMED_UNIT
    assert recovered.uses_effective_variance is True
    assert recovered.quality_reason == mixed.quality_reason


def test_saturated_trend_fit_roundtrips_unavailable_reduced_statistic():
    result = fit_parameter_model(
        [0.0],
        [0.5],
        [0.1],
        ParameterCompositeModel(["Constant"]),
        ParameterSet([Parameter("c", 0.5)]),
    )

    assert result.success
    assert result.chi_squared == 0.0
    assert result.dof == 0
    assert np.isnan(result.reduced_chi_squared)
    assert result.to_dict()["reduced_chi_squared"] is None
    recovered = ParameterModelFitResult.from_dict(result.to_dict())
    assert recovered.dof == 0
    assert np.isnan(recovered.reduced_chi_squared)
    assert recovered.assess_quality() is None


def test_cross_group_keeps_weight_sources_of_retained_observations():
    x = np.arange(3.0)
    y = np.array([1.0, 3.0, 4.0])
    groups = [
        ParameterGroupData("unit", "Unit", x, y, None, 0.0),
        ParameterGroupData(
            "mixed",
            "Mixed",
            x,
            y,
            np.ones_like(y),
            0.0,
            yerr_estimated=np.array([False, False, True]),
        ),
    ]
    config = {"model": ParameterCompositeModel(["Constant"]).to_dict(), "windows": [[0, 1]]}
    cropped = run_cross_group_fit_from_config(groups, config).result
    assert cropped.success
    assert cropped.global_parameters["c"].value == pytest.approx(2.0)
    assert cropped.chi_squared == pytest.approx(4.0)
    assert cropped.dof == 3
    assert cropped.column_error_sources == {
        "unit": ColumnErrorSource.ASSUMED_UNIT,
        "mixed": ColumnErrorSource.SUPPLIED,
    }
    assert "assumed unit" in cropped.quality_reason

    full = run_cross_group_fit_from_config(groups, config | {"windows": []}).result
    assert full.success
    assert full.column_error_sources["mixed"] is ColumnErrorSource.INCLUDES_ESTIMATES
    assert "include estimates" in full.quality_reason
    recovered = CrossGroupFitResult.from_dict(full.to_dict())
    assert recovered.column_error_sources == full.column_error_sources
    assert recovered.quality_reason == full.quality_reason
