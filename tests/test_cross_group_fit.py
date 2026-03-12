"""Tests for cross-group parameter model fitting."""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    ParameterGroupData,
    global_fit_parameter_model,
)


def test_global_fit_parameter_model_recovers_shared_slope() -> None:
    model = ParameterCompositeModel(["Linear"])

    # Shared slope m=2.0 across groups, local intercepts b_i.
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    groups = []
    true_intercepts = [1.0, -0.5, 3.0]
    for idx, b in enumerate(true_intercepts):
        y = 2.0 * x + b
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x,
                y=y,
                yerr=np.full_like(x, 0.05),
                group_variable_value=float(idx),
            )
        )

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )

    assert result.success
    assert "m" in result.global_parameters
    np.testing.assert_allclose(
        result.global_parameters["m"].value,
        2.0,
        rtol=1e-3,
        atol=1e-3,
    )


def test_global_fit_parameter_model_requires_two_groups() -> None:
    model = ParameterCompositeModel(["Linear"])
    single_group = [
        ParameterGroupData(
            group_id="g0",
            group_name="G0",
            x=np.array([0.0, 1.0], dtype=float),
            y=np.array([1.0, 2.0], dtype=float),
            yerr=np.array([0.1, 0.1], dtype=float),
            group_variable_value=0.0,
        )
    ]

    result = global_fit_parameter_model(
        groups=single_group,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
    )

    assert not result.success
    assert "Need at least two groups" in result.message


def test_global_fit_parameter_model_enforces_parameter_bounds() -> None:
    model = ParameterCompositeModel(["Linear"])

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    groups = []
    for idx, b in enumerate([0.0, 1.0, 2.0]):
        y = 2.0 * x + b
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x,
                y=y,
                yerr=np.full_like(x, 0.05),
                group_variable_value=float(idx),
            )
        )

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 0.5, "b": 0.0},
        parameter_bounds={"m": (0.0, 1.0), "b": (-10.0, 10.0)},
    )

    assert result.success
    assert "m" in result.global_parameters
    assert result.global_parameters["m"].value <= 1.0 + 1e-6
    assert result.global_parameters["m"].min == 0.0
    assert result.global_parameters["m"].max == 1.0


def test_global_fit_parameter_model_returns_fixed_parameters() -> None:
    model = ParameterCompositeModel(["Linear"])

    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    groups = []
    for idx in range(3):
        y = 2.0 * x + 1.0
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x,
                y=y,
                yerr=np.full_like(x, 0.05),
                group_variable_value=float(idx),
            )
        )

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=[],
        fixed_params={"b": 1.0},
        initial_params={"m": 1.0},
    )

    assert result.success
    assert "b" in result.fixed_parameters
    assert result.fixed_parameters["b"].fixed
    assert np.isclose(result.fixed_parameters["b"].value, 1.0)
