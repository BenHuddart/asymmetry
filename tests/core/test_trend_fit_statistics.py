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


@pytest.mark.parametrize(
    ("mode", "expected", "standard_error", "effective_sigma"),
    [
        (ErrorMode.COLUMN, 106 / 102, 1 / np.sqrt(102), [0.1, 1.0, 1.0]),
        (ErrorMode.COLUMN_STABILIZED, 5 / 3, 1 / np.sqrt(6), [0.5, 1.0, 1.0]),
    ],
)
@pytest.mark.parametrize("with_invalid_points", [False, True])
def test_column_fit_policy_on_selected_points(
    mode, expected, standard_error, effective_sigma, with_invalid_points
) -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 3.0, 3.0])
    sigma = np.array([0.1, 1.0, 1.0])
    windows = None
    if with_invalid_points:
        x = np.r_[x, 0.25, 1.25, 1.75, 2.1, 3.0, 4.0, 5.0]
        y = np.r_[y, np.full(7, 100.0)]
        sigma = np.r_[sigma, 0.0, -0.2, np.inf, np.nan, 10.0, 10.0, 10.0]
        windows = [(0.0, 0.5), (0.75, 2.25)]
    result = fit_parameter_model(
        x,
        y,
        sigma,
        ParameterCompositeModel(["Constant"]),
        ParameterSet([Parameter("c", 0.0)]),
        windows=windows,
        error_mode=mode,
    )

    assert result.success
    assert result.n_points == 3
    assert result.parameters["c"].value == pytest.approx(expected, abs=1e-4)
    assert result.uncertainties["c"] == pytest.approx(standard_error, abs=1e-4)
    assert result.chi_squared == pytest.approx(
        np.sum(((y[:3] - expected) / effective_sigma) ** 2), abs=1e-4
    )
    restored = ParameterModelFitResult.from_dict(result.to_dict())
    assert restored.error_mode == mode.value
    assert restored.column_error_floor_applied is (mode is ErrorMode.COLUMN_STABILIZED)
    assert (restored.quality_reason is None) is (mode is ErrorMode.COLUMN)


def test_inactive_column_error_floor_keeps_quality_verdict() -> None:
    x = np.arange(3.0)
    result = fit_parameter_model(
        x,
        x,
        np.ones_like(x),
        ParameterCompositeModel(["Constant"]),
        ParameterSet([Parameter("c", 0.0)]),
        error_mode=ErrorMode.COLUMN_STABILIZED,
    )

    assert result.success
    assert result.column_error_floor_applied is False
    assert result.assess_quality() is not None


def test_column_fit_recommends_alternative_when_all_errors_are_invalid() -> None:
    x = np.arange(3.0)
    result = fit_parameter_model(
        x,
        x,
        np.array([0.0, -1.0, np.nan]),
        ParameterCompositeModel(["Constant"]),
        ParameterSet([Parameter("c", 0.0)]),
    )

    assert not result.success
    assert "Choose None (unit weights) or Estimate from scatter" in result.message


@pytest.mark.slow
def test_fit_parameter_model_error_floor_blocks_high_field_collapse() -> None:
    x = np.array(
        [25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 35000.0]
    )
    y = np.array(
        [
            0.604866,
            0.504736,
            0.337137,
            0.203568,
            0.174864,
            0.159764,
            0.140585,
            0.139061,
            0.0664971,
            0.00056813,
            0.0,
        ]
    )
    yerr = np.array(
        [
            0.0331,
            0.0386,
            0.0239,
            0.0131,
            0.0111,
            0.0102,
            0.00929,
            0.0093,
            0.00486,
            0.000115,
            1.39e-11,
        ]
    )
    model = ParameterCompositeModel(["DiffusionLF_2D", "BallisticLF_2D"], operators=["+"])

    params = ParameterSet(
        [
            Parameter("A_1", value=1.0, min=0.0, max=500.0),
            Parameter("D_2D", value=1.0, min=0.0, max=1.0e6),
            Parameter("D_perp", value=0.0, min=0.0, max=0.0, fixed=True),
            Parameter("A_2", value=1.0, min=0.0, max=500.0),
            Parameter("D_hop", value=1.0, min=0.0, max=1.0e6),
        ]
    )

    result = fit_parameter_model(x, y, yerr, model, params, error_mode=ErrorMode.COLUMN_STABILIZED)

    assert result.success
    assert result.column_error_floor_applied
    fitted = {parameter.name: parameter.value for parameter in result.parameters}
    assert result.reduced_chi_squared < 5.0
    assert fitted["A_1"] > 20.0
    assert 100.0 <= fitted["D_2D"] <= 1000.0
    assert fitted["A_2"] > 100.0
    assert fitted["D_hop"] > 1.0e4
