"""Tests for cross-group parameter model fitting."""

from __future__ import annotations

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Phase 1A — per-group chi-squared, correlations, serialization
# ---------------------------------------------------------------------------


def _three_line_groups(*, seed: int = 3) -> list[ParameterGroupData]:
    """Three groups sharing slope m, local intercepts b_i, noisy y."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, 12)
    groups: list[ParameterGroupData] = []
    for idx, b in enumerate([1.0, -0.5, 3.0]):
        y = 2.0 * x + b + rng.normal(0.0, 0.1, size=x.size)
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"G{idx}",
                x=x.copy(),
                y=y,
                yerr=np.full_like(x, 0.1),
                group_variable_value=float(idx),
            )
        )
    return groups


def test_per_group_chi_squared_sums_to_total_column_mode() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )

    assert result.success
    assert set(result.per_group_chi_squared) == {"g0", "g1", "g2"}
    assert set(result.per_group_n_points) == {"g0", "g1", "g2"}
    np.testing.assert_allclose(
        sum(result.per_group_chi_squared.values()), result.chi_squared, rtol=1e-9
    )
    assert sum(result.per_group_n_points.values()) == result.n_points


def test_per_group_chi_squared_sums_to_total_with_windows() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        windows=[(2.0, 8.0)],
    )

    assert result.success
    np.testing.assert_allclose(
        sum(result.per_group_chi_squared.values()), result.chi_squared, rtol=1e-9
    )
    assert sum(result.per_group_n_points.values()) == result.n_points
    # The window excludes some of the 12 points per group.
    assert result.n_points < 12 * len(groups)


def test_per_group_chi_squared_sums_to_total_with_xerr() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _two_line_groups(with_xerr=True)

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        xerr={g.group_id: g.xerr for g in groups},
    )

    assert result.success
    np.testing.assert_allclose(
        sum(result.per_group_chi_squared.values()), result.chi_squared, rtol=1e-9
    )
    assert sum(result.per_group_n_points.values()) == result.n_points


def test_per_group_chi_squared_reported_pre_rescale_under_scatter() -> None:
    """Under SCATTER, per-group chi-squared still sums to the (pre-rescale)
    total chi_squared -- the rescale only touches uncertainties."""
    from asymmetry.core.fitting.parameter_models import ErrorMode

    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        error_mode=ErrorMode.SCATTER,
    )

    assert result.success
    np.testing.assert_allclose(
        sum(result.per_group_chi_squared.values()), result.chi_squared, rtol=1e-9
    )


def test_global_correlations_symmetric_unit_diagonal_shape() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )

    assert result.success
    assert result.global_correlations is not None
    names, matrix = result.global_correlations
    assert set(names) == {"m", "b"}
    arr = np.array(matrix)
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(np.diag(arr), 1.0, atol=1e-8)
    np.testing.assert_allclose(arr, arr.T, atol=1e-10)


def test_global_correlations_none_when_fewer_than_two_free_globals() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    # Only one global parameter ("m"); "b" is local per group.
    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )

    assert result.success
    assert result.global_correlations is None


def test_global_correlations_none_when_all_params_fixed_or_local() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    # No global parameters at all -> zero free globals.
    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=[],
        local_params=["m", "b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )

    assert result.success
    assert result.global_correlations is None


# ---------------------------------------------------------------------------
# Phase 1A — canonical serialization
# ---------------------------------------------------------------------------


def test_parameter_group_data_round_trip_with_xerr() -> None:
    group = ParameterGroupData(
        group_id="g0",
        group_name="G0",
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([1.0, 2.0, 3.0]),
        yerr=np.array([0.1, 0.1, 0.1]),
        group_variable_value=4.5,
        xerr=np.array([0.2, 0.2, 0.2]),
    )

    restored = ParameterGroupData.from_dict(group.to_dict())

    assert restored.group_id == group.group_id
    assert restored.group_name == group.group_name
    assert restored.group_variable_value == group.group_variable_value
    np.testing.assert_allclose(restored.x, group.x)
    np.testing.assert_allclose(restored.y, group.y)
    np.testing.assert_allclose(restored.yerr, group.yerr)
    assert restored.xerr is not None
    np.testing.assert_allclose(restored.xerr, group.xerr)


def test_parameter_group_data_round_trip_without_xerr() -> None:
    group = ParameterGroupData(
        group_id="g1",
        group_name="G1",
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 2.0]),
        yerr=np.array([0.1, 0.1]),
        group_variable_value=1.0,
    )

    restored = ParameterGroupData.from_dict(group.to_dict())

    assert restored.xerr is None
    np.testing.assert_allclose(restored.x, group.x)


def test_cross_group_fit_result_round_trip() -> None:
    model = ParameterCompositeModel(["Linear"])
    groups = _three_line_groups()

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m"],
        local_params=["b"],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
    )
    assert result.success

    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult

    restored = CrossGroupFitResult.from_dict(result.to_dict())

    assert restored.success == result.success
    np.testing.assert_allclose(restored.chi_squared, result.chi_squared)
    np.testing.assert_allclose(restored.reduced_chi_squared, result.reduced_chi_squared)
    assert restored.n_points == result.n_points
    assert restored.error_mode == result.error_mode
    assert restored.message == result.message

    assert set(restored.global_parameters.names) == set(result.global_parameters.names)
    for name in result.global_parameters.names:
        assert restored.global_parameters[name].value == pytest.approx(
            result.global_parameters[name].value
        )
        assert restored.global_parameters[name].min == pytest.approx(
            result.global_parameters[name].min
        )
        assert restored.global_parameters[name].max == pytest.approx(
            result.global_parameters[name].max
        )
        assert restored.global_parameters[name].fixed == result.global_parameters[name].fixed

    assert set(restored.local_parameters) == set(result.local_parameters)
    for gid, pset in result.local_parameters.items():
        restored_pset = restored.local_parameters[gid]
        for name in pset.names:
            assert restored_pset[name].value == pytest.approx(pset[name].value)

    assert restored.global_uncertainties == pytest.approx(result.global_uncertainties)
    assert set(restored.local_uncertainties) == set(result.local_uncertainties)

    assert restored.per_group_chi_squared == pytest.approx(result.per_group_chi_squared)
    assert restored.per_group_n_points == result.per_group_n_points

    if result.global_correlations is None:
        assert restored.global_correlations is None
    else:
        names, matrix = result.global_correlations
        restored_names, restored_matrix = restored.global_correlations
        assert restored_names == names
        np.testing.assert_allclose(np.array(restored_matrix), np.array(matrix))


def test_cross_group_fit_result_round_trip_with_xerr_groups() -> None:
    """Round trip a result produced from xerr-bearing groups (covers the
    effective-variance fit path feeding the same serializer)."""
    model = ParameterCompositeModel(["Linear"])
    groups = _two_line_groups(with_xerr=True)

    result = global_fit_parameter_model(
        groups=groups,
        model=model,
        global_params=["m", "b"],
        local_params=[],
        fixed_params={},
        initial_params={"m": 1.0, "b": 0.0},
        xerr={g.group_id: g.xerr for g in groups},
    )
    assert result.success

    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult

    restored = CrossGroupFitResult.from_dict(result.to_dict())
    np.testing.assert_allclose(restored.chi_squared, result.chi_squared)
    assert restored.per_group_chi_squared == pytest.approx(result.per_group_chi_squared)


def test_cross_group_fit_result_from_dict_legacy_payload_without_new_keys() -> None:
    """A legacy dict recorded before Phase 1A (no per-group/correlation keys)
    must still load, with empty/None defaults for the new fields."""
    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult

    legacy = {
        "success": True,
        "chi_squared": 12.3,
        "reduced_chi_squared": 1.2,
        "global_parameters": [
            {"name": "m", "value": 2.0, "min": -1e300, "max": 1e300, "fixed": False}
        ],
        "local_parameters": {
            "g0": [{"name": "b", "value": 1.0, "min": -1e300, "max": 1e300, "fixed": False}]
        },
        "fixed_parameters": [],
        "global_uncertainties": {"m": 0.1},
        "local_uncertainties": {"g0": {"b": 0.05}},
        "message": "Fit successful",
        "error_mode": "column",
        "n_points": 24,
    }

    restored = CrossGroupFitResult.from_dict(legacy)

    assert restored.success is True
    assert restored.chi_squared == pytest.approx(12.3)
    assert restored.n_points == 24
    assert restored.per_group_chi_squared == {}
    assert restored.per_group_n_points == {}
    assert restored.global_correlations is None
    assert "m" in restored.global_parameters


def test_cross_group_fit_result_from_dict_tolerates_malformed_entries() -> None:
    """Malformed parameter entries (missing name, junk correlation shape) are
    skipped rather than raising."""
    from asymmetry.core.fitting.parameter_models import CrossGroupFitResult

    junky = {
        "success": True,
        "chi_squared": 1.0,
        "reduced_chi_squared": 1.0,
        "global_parameters": [
            {"name": "m", "value": 2.0, "min": -1.0, "max": 1.0, "fixed": False},
            {"value": 3.0},  # missing name -- skipped
            "not-a-dict",  # skipped
        ],
        "local_parameters": {"g0": "not-a-list"},
        "per_group_chi_squared": {"g0": "nan-ish-but-not-a-number"},
        "global_correlations": ["not", "two-elements", "either"],
    }

    restored = CrossGroupFitResult.from_dict(junky)

    assert "m" in restored.global_parameters
    assert len(restored.global_parameters.names) == 1
    assert restored.local_parameters["g0"].names == []
    assert restored.per_group_chi_squared == {}
    assert restored.global_correlations is None
