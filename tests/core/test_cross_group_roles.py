"""Tests for the cross-group role-suggestion engine.

The "truth" tests (paper-shaped and all-shared) use small, well-conditioned
synthetic data so the real cross-group fits converge quickly. The
strategy/plumbing tests (cancellation, budget, determinism) monkeypatch
``global_fit_parameter_model`` with a fast deterministic stub so they never pay
the cost of a real fit.
"""

from __future__ import annotations

import numpy as np
import pytest

import asymmetry.core.fitting.cross_group_roles as cross_group_roles_module
from asymmetry.core import fitting as fitting_api
from asymmetry.core.fitting.cross_group_roles import (
    CrossGroupRoleRecommendation,
    suggest_cross_group_roles,
)
from asymmetry.core.fitting.parameter_models import (
    CrossGroupFitResult,
    ErrorMode,
    ParameterCompositeModel,
    ParameterGroupData,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

pytestmark = [pytest.mark.integration]


# --------------------------------------------------------------------------
# Synthetic data builders
# --------------------------------------------------------------------------


def _exp_groups(
    tau_values: list[float],
    *,
    a_true: float = 0.9,
    c_true: float = 0.1,
    seed: int = 1,
    noise: float = 0.01,
) -> tuple[ParameterCompositeModel, list[ParameterGroupData]]:
    """ExponentialDecay across N groups: y = a·exp(-x/tau) + c.

    ``ExponentialDecay`` (params ``a``, ``tau``, ``c``) is the well-conditioned
    cross-group analogue of the paper's DiffusionLF_2D + Constant shape: a
    transport-rate-like parameter (``tau``) that genuinely varies per group,
    plus a shared amplitude (``a``) and background (``c``). It converges in
    milliseconds so the whole test file stays well under budget.

    (The paper's own DiffusionLF_2D + Constant is exercised through the GUI /
    integration layer; its cross-group fits take tens of seconds each and its
    high-field tail is near-degenerate, so it is unsuitable for a fast unit
    test — see the module report.)
    """
    model = ParameterCompositeModel(["ExponentialDecay"])  # a, tau, c
    x = np.linspace(0.0, 5.0, 10)
    rng = np.random.default_rng(seed)
    groups: list[ParameterGroupData] = []
    for idx, tau in enumerate(tau_values):
        y = model.function(x, a=a_true, tau=tau, c=c_true)
        y = y + rng.normal(0.0, noise, size=x.shape)
        groups.append(
            ParameterGroupData(
                group_id=f"g{idx}",
                group_name=f"T{idx}",
                x=x.copy(),
                y=y,
                yerr=np.full_like(x, noise),
                group_variable_value=float(10 * (idx + 1)),
            )
        )
    return model, groups


def _exp_fit_kwargs() -> dict:
    return {
        "initial_params": {"a": 0.9, "tau": 2.0, "c": 0.1},
        "parameter_bounds": {"a": (0.01, 5.0), "tau": (0.05, 50.0), "c": (-1.0, 1.0)},
        "error_mode": ErrorMode.COLUMN,
    }


# --------------------------------------------------------------------------
# 1. Paper-shaped truth: tau genuinely per-group, a and c shared
# --------------------------------------------------------------------------


def test_paper_shaped_recommends_rate_local_and_others_global() -> None:
    model, groups = _exp_groups([0.8, 1.5, 3.0, 6.0])

    recommendation = suggest_cross_group_roles(
        groups,
        model,
        criterion="aicc",
        max_fits=20,
        **_exp_fit_kwargs(),
    )

    assert isinstance(recommendation, CrossGroupRoleRecommendation)
    assert recommendation.recommended is not None
    recommended = recommendation.recommended

    # tau should be recommended Local; a and c shared (Global).
    assert "tau" in recommended.local_params
    assert "a" in recommended.global_params
    assert "c" in recommended.global_params

    roles = {row.name: row.recommended_role for row in recommendation.parameters}
    assert roles["tau"] == "local"
    assert roles["a"] == "global"
    assert roles["c"] == "global"

    # The recommended candidate must beat the all-global and all-local extremes.
    by_partition = {(c.global_params, c.local_params): c for c in recommendation.candidates}
    all_global = by_partition.get((("a", "c", "tau"), ()))
    assert all_global is not None and all_global.success
    assert recommended.aicc < all_global.aicc

    all_local = by_partition.get(((), ("a", "c", "tau")))
    assert all_local is not None and all_local.success
    assert recommended.aicc <= all_local.aicc

    # tau row should carry a real per-group spread signal.
    tau_row = next(row for row in recommendation.parameters if row.name == "tau")
    assert tau_row.total_variation > 0.0
    assert tau_row.score_delta > 0.0  # global scores worse than local => favour local


# --------------------------------------------------------------------------
# 2. All-shared truth: nothing should be localized
# --------------------------------------------------------------------------


def test_all_shared_truth_recommends_nothing_local() -> None:
    # Identical tau across all groups -> localizing any param cannot pay for
    # the extra free parameters.
    model, groups = _exp_groups([2.5, 2.5, 2.5, 2.5], seed=7)

    recommendation = suggest_cross_group_roles(
        groups,
        model,
        criterion="aicc",
        max_fits=20,
        **_exp_fit_kwargs(),
    )

    assert recommendation.recommended is not None
    assert recommendation.recommended.local_params == ()
    assert all(row.recommended_role == "global" for row in recommendation.parameters)


# --------------------------------------------------------------------------
# Fast deterministic stub for strategy/plumbing tests
# --------------------------------------------------------------------------


def _stub_result(
    groups: list[ParameterGroupData],
    global_params: list[str],
    local_params: list[str],
    *,
    chi_squared: float,
) -> CrossGroupFitResult:
    global_set = ParameterSet(
        [Parameter(name, value=1.0, min=-10.0, max=10.0) for name in global_params]
    )
    local_sets: dict[str, ParameterSet] = {}
    for gidx, group in enumerate(groups):
        local_sets[group.group_id] = ParameterSet(
            [Parameter(name, value=1.0 + 0.5 * gidx, min=-10.0, max=10.0) for name in local_params]
        )
    n_points = sum(int(np.asarray(group.x).size) for group in groups)
    return CrossGroupFitResult(
        success=True,
        chi_squared=chi_squared,
        reduced_chi_squared=chi_squared / max(n_points, 1),
        global_parameters=global_set,
        local_parameters=local_sets,
        n_points=n_points,
    )


def _linear_groups() -> tuple[ParameterCompositeModel, list[ParameterGroupData]]:
    model = ParameterCompositeModel(["Linear"])  # params: m, b
    x = np.linspace(0.0, 3.0, 6)
    groups = [
        ParameterGroupData(
            group_id=f"g{idx}",
            group_name=f"G{idx}",
            x=x.copy(),
            y=2.0 * x + idx,
            yerr=np.full_like(x, 0.1),
            group_variable_value=float(idx),
        )
        for idx in range(3)
    ]
    return model, groups


# --------------------------------------------------------------------------
# 3. Cancellation
# --------------------------------------------------------------------------


def test_cancellation_after_first_fit_returns_partial(monkeypatch) -> None:
    model, groups = _linear_groups()
    call_count = {"n": 0}

    def _fake_fit(groups_arg, model_arg, *, global_params, local_params, **kwargs):
        call_count["n"] += 1
        return _stub_result(groups_arg, global_params, local_params, chi_squared=10.0)

    monkeypatch.setattr(cross_group_roles_module, "global_fit_parameter_model", _fake_fit)

    fits_seen = {"n": 0}

    def _cancel() -> bool:
        # Allow the very first fit, then cancel.
        result = fits_seen["n"] >= 1
        fits_seen["n"] += 1
        return result

    recommendation = suggest_cross_group_roles(
        groups,
        model,
        cancel_callback=_cancel,
        max_fits=40,
    )

    assert "cancelled" in recommendation.message
    # Exactly one real fit ran before cancellation kicked in.
    assert call_count["n"] == 1


# --------------------------------------------------------------------------
# 4. Budget cap
# --------------------------------------------------------------------------


def test_max_fits_budget_is_respected(monkeypatch) -> None:
    model, groups = _linear_groups()
    call_count = {"n": 0}

    def _fake_fit(groups_arg, model_arg, *, global_params, local_params, **kwargs):
        call_count["n"] += 1
        # Give localizing "m" a big improvement so greedy search wants more fits.
        chi = 5.0 if "m" in local_params else 100.0
        return _stub_result(groups_arg, global_params, local_params, chi_squared=chi)

    monkeypatch.setattr(cross_group_roles_module, "global_fit_parameter_model", _fake_fit)

    recommendation = suggest_cross_group_roles(
        groups,
        model,
        max_fits=3,
    )

    assert call_count["n"] <= 3
    assert isinstance(recommendation, CrossGroupRoleRecommendation)


# --------------------------------------------------------------------------
# 5. Determinism
# --------------------------------------------------------------------------


def test_two_runs_produce_identical_candidate_ordering(monkeypatch) -> None:
    model, groups = _linear_groups()

    def _fake_fit(groups_arg, model_arg, *, global_params, local_params, **kwargs):
        # Deterministic chi2 depending only on the partition.
        chi = 50.0 - 10.0 * len(local_params)
        return _stub_result(groups_arg, global_params, local_params, chi_squared=chi)

    monkeypatch.setattr(cross_group_roles_module, "global_fit_parameter_model", _fake_fit)

    rec_a = suggest_cross_group_roles(groups, model, max_fits=40)
    rec_b = suggest_cross_group_roles(groups, model, max_fits=40)

    order_a = [(c.global_params, c.local_params, c.aicc) for c in rec_a.candidates]
    order_b = [(c.global_params, c.local_params, c.aicc) for c in rec_b.candidates]
    assert order_a == order_b
    assert rec_a.recommended is not None
    assert rec_b.recommended is not None
    assert (rec_a.recommended.global_params, rec_a.recommended.local_params) == (
        rec_b.recommended.global_params,
        rec_b.recommended.local_params,
    )


# --------------------------------------------------------------------------
# Guards + exports
# --------------------------------------------------------------------------


def test_fewer_than_two_groups_returns_empty_message() -> None:
    model, groups = _linear_groups()
    recommendation = suggest_cross_group_roles(groups[:1], model)
    assert recommendation.recommended is None
    assert "two groups" in recommendation.message


def test_all_fixed_params_returns_message() -> None:
    model, groups = _linear_groups()
    recommendation = suggest_cross_group_roles(
        groups,
        model,
        fixed_params={"m": 2.0, "b": 0.0},
    )
    assert recommendation.recommended is None
    assert "fixed" in recommendation.message.lower()


def test_none_error_mode_flags_relative_only(monkeypatch) -> None:
    model, groups = _linear_groups()

    def _fake_fit(groups_arg, model_arg, *, global_params, local_params, **kwargs):
        chi = 50.0 - 10.0 * len(local_params)
        return _stub_result(groups_arg, global_params, local_params, chi_squared=chi)

    monkeypatch.setattr(cross_group_roles_module, "global_fit_parameter_model", _fake_fit)

    recommendation = suggest_cross_group_roles(
        groups,
        model,
        error_mode=ErrorMode.NONE,
        max_fits=40,
    )
    assert "relative" in recommendation.message.lower()


def test_all_failed_fits_leaves_no_recommendation(monkeypatch) -> None:
    model, groups = _linear_groups()

    def _fake_fit(groups_arg, model_arg, *, global_params, local_params, **kwargs):
        return CrossGroupFitResult(
            success=False,
            chi_squared=float("inf"),
            reduced_chi_squared=float("inf"),
            message="forced failure",
            n_points=18,
        )

    monkeypatch.setattr(cross_group_roles_module, "global_fit_parameter_model", _fake_fit)

    recommendation = suggest_cross_group_roles(groups, model, max_fits=40)
    assert recommendation.recommended is None
    assert any(not c.success for c in recommendation.candidates)
    assert "No candidate fit converged" in recommendation.message


def test_symbols_are_exported() -> None:
    assert fitting_api.suggest_cross_group_roles is not None
    assert fitting_api.CrossGroupCandidate is not None
    assert fitting_api.CrossGroupParameterRecommendation is not None
    assert fitting_api.CrossGroupRoleRecommendation is not None
