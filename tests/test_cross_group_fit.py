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


# ---------------------------------------------------------------------------
# Phase C — cross-group x-uncertainty (effective variance)
# ---------------------------------------------------------------------------


def _two_line_groups(*, with_xerr: bool, seed: int = 7) -> list[ParameterGroupData]:
    """Two groups sharing slope m and intercept b, with finite σ_y (and σ_x)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, 15)
    groups: list[ParameterGroupData] = []
    for idx in range(2):
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, size=x.size)
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x.copy(),
                y=y,
                yerr=np.full_like(x, 0.05),
                group_variable_value=float(idx),
                xerr=(np.full_like(x, 0.4) if with_xerr else None),
            )
        )
    return groups


def test_cross_group_xerr_zero_is_identical_to_ols() -> None:
    """σ_x = 0 (or xerr=None) ⇒ byte-identical to the ordinary-least-squares
    cross-group fit (the regression-safety oracle)."""
    model = ParameterCompositeModel(["Linear"])
    groups = _two_line_groups(with_xerr=False)

    base = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )
    # xerr present but all zero -> the per-group branch contributes nothing.
    zero_groups = [
        ParameterGroupData(
            group_id=g.group_id,
            group_name=g.group_name,
            x=g.x,
            y=g.y,
            yerr=g.yerr,
            group_variable_value=g.group_variable_value,
            xerr=np.zeros_like(g.x),
        )
        for g in groups
    ]
    zeroed = global_fit_parameter_model(
        groups=zero_groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        xerr={g.group_id: g.xerr for g in zero_groups},
    )

    assert base.success and zeroed.success
    np.testing.assert_allclose(
        base.global_parameters["m"].value, zeroed.global_parameters["m"].value, atol=1e-12
    )
    np.testing.assert_allclose(
        base.global_parameters["b"].value, zeroed.global_parameters["b"].value, atol=1e-12
    )
    np.testing.assert_allclose(
        base.global_uncertainties["m"], zeroed.global_uncertainties["m"], atol=1e-12
    )
    np.testing.assert_allclose(
        base.global_uncertainties["b"], zeroed.global_uncertainties["b"], atol=1e-12
    )


def test_cross_group_xerr_matches_single_series_effective_variance() -> None:
    """Two identical groups with σ_x: the cross-group effective-variance fit's
    shared parameters equal the single-series effective-variance fit on one copy
    (the two paths share one estimator and must not drift)."""
    from asymmetry.core.fitting.parameter_models import fit_parameter_model

    model = ParameterCompositeModel(["Linear"])
    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 15)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, size=x.size)
    yerr = np.full_like(x, 0.05)
    xerr = np.full_like(x, 0.4)

    params = ParameterCompositeModel(["Linear"]).param_names
    assert params == ["m", "b"]

    single = fit_parameter_model(
        x=x,
        y=y,
        yerr=yerr,
        model=model,
        parameters=_linear_params(),
        xerr=xerr,
    )

    groups = [
        ParameterGroupData(
            group_id=f"g{idx}",
            group_name=f"G{idx}",
            x=x.copy(),
            y=y.copy(),
            yerr=yerr.copy(),
            group_variable_value=float(idx),
            xerr=xerr.copy(),
        )
        for idx in range(2)
    ]
    cross = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        xerr={g.group_id: g.xerr for g in groups},
    )

    assert single.success and cross.success
    # The two paths share one effective-variance estimator, so they recover the
    # same minimum; the residual difference is migrad's tolerance on the
    # 2×-scaled cross-group cost (two identical groups), not a numerics drift.
    np.testing.assert_allclose(
        cross.global_parameters["m"].value, single.parameters["m"].value, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        cross.global_parameters["b"].value, single.parameters["b"].value, rtol=1e-3, atol=1e-3
    )


def test_cross_group_xerr_inflates_uncertainties() -> None:
    """Finite σ_x inflates the global parameter errors vs ignoring it."""
    model = ParameterCompositeModel(["Linear"])
    groups = _two_line_groups(with_xerr=True)
    kwargs = dict(
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )
    without = global_fit_parameter_model(groups=groups, **kwargs)
    withx = global_fit_parameter_model(
        groups=groups, xerr={g.group_id: g.xerr for g in groups}, **kwargs
    )
    assert without.success and withx.success
    assert withx.global_uncertainties["m"] > without.global_uncertainties["m"]


def test_cross_group_xerr_ignored_under_scatter_and_none() -> None:
    """x-errors carry no scale under None/Scatter weights, so they are ignored
    (the result equals the same mode with no xerr)."""
    from asymmetry.core.fitting.parameter_models import ErrorMode

    model = ParameterCompositeModel(["Linear"])
    groups = _two_line_groups(with_xerr=True)
    for mode in (ErrorMode.NONE, ErrorMode.SCATTER):
        kwargs = dict(
            model=model,
            global_params=["m", "b"],
            local_params=[],
            fixed_params={},
            initial_params={"m": 1.0, "b": 0.0},
            error_mode=mode,
        )
        without = global_fit_parameter_model(groups=groups, **kwargs)
        withx = global_fit_parameter_model(
            groups=groups, xerr={g.group_id: g.xerr for g in groups}, **kwargs
        )
        assert without.success and withx.success
        np.testing.assert_allclose(
            without.global_parameters["m"].value,
            withx.global_parameters["m"].value,
            atol=1e-12,
        )


def _linear_params():
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    ps = ParameterSet()
    ps.add(Parameter(name="m", value=1.0))
    ps.add(Parameter(name="b", value=0.0))
    return ps
