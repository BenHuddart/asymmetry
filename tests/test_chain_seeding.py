"""Sequential chain-seeding for grouped series (fit-workflow-diagnostics).

WiMDA ``itPrevious`` analogue: member N+1 is seeded from member N's fitted values
(re-normalised to the grouped contract), iterating in the series order key. Auto
picks chaining only for ordered scans.
"""

from __future__ import annotations

import numpy as np

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import (
    GROUP_NUISANCE_PARAMS,
    GroupedTimeDomainFitResult,
    GroupedTimeDomainGroup,
    SeedingRecommendation,
    fit_grouped_series,
    recommend_grouped_series_seeding,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _two_groups(run: int) -> list[GroupedTimeDomainGroup]:
    return [
        GroupedTimeDomainGroup(
            group_id="forward",
            group_name="Forward",
            time=np.array([0.0, 0.1]),
            counts=np.array([100.0, 99.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=run,
        ),
        GroupedTimeDomainGroup(
            group_id="backward",
            group_name="Backward",
            time=np.array([0.0, 0.1]),
            counts=np.array([95.0, 94.0]),
            error=np.array([1.0, 1.0]),
            source_run_number=run,
        ),
    ]


def _provided_group_seed() -> ParameterSet:
    ps = ParameterSet()
    ps.add(Parameter(name="frequency", value=1.0, min=0.0))
    ps.add(Parameter(name="N0", value=100.0))
    ps.add(Parameter(name="background", value=9.0))
    ps.add(Parameter(name="amplitude", value=0.5))
    ps.add(Parameter(name="relative_phase", value=0.0))
    return ps


def _series_initial(runs) -> dict:
    return {run: {"forward": _provided_group_seed(), "backward": _provided_group_seed()} for run in runs}


def _fake_member_factory(received: dict):
    """A fake member fit that records its received seed and returns distinct fits."""

    def _fake(groups, _model_fn, **kwargs):
        run = int(groups[0].source_run_number)
        received[run] = {
            gid: {p.name: p.value for p in ps} for gid, ps in kwargs["initial_params"].items()
        }
        group_results = {}
        for g in groups:
            fitted = ParameterSet()
            fitted.add(Parameter(name="frequency", value=float(run)))  # distinct per run
            fitted.add(Parameter(name="N0", value=500.0 + run))
            fitted.add(Parameter(name="background", value=0.05))  # fitted away from 0
            fitted.add(Parameter(name="amplitude", value=0.7))  # fitted away from 1
            fitted.add(Parameter(name="relative_phase", value=0.2))
            group_results[g.group_id] = FitResult(
                success=True, chi_squared=1.0, reduced_chi_squared=0.1, parameters=fitted
            )
        return GroupedTimeDomainFitResult(
            success=True,
            group_results=group_results,
            shared_parameters=ParameterSet(),
            message="ok",
        )

    return _fake


# --- recommendation policy --------------------------------------------------


def test_recommend_chains_ordered_scan():
    rec = recommend_grouped_series_seeding([10, 11, 12], {10: 5.0, 11: 10.0, 12: 15.0})
    assert rec.mode == "chain"
    assert "chain" in rec.reason.lower()


def test_recommend_skips_when_too_few_members():
    rec = recommend_grouped_series_seeding([10, 11], {10: 5.0, 11: 10.0})
    assert rec.mode == "as_provided"


def test_recommend_skips_without_order_key():
    rec = recommend_grouped_series_seeding([10, 11, 12], None)
    assert rec.mode == "as_provided"


def test_recommend_skips_constant_order_key():
    rec = recommend_grouped_series_seeding([10, 11, 12], {10: 5.0, 11: 5.0, 12: 5.0})
    assert rec.mode == "as_provided"


# --- chaining mechanism -----------------------------------------------------


def test_chain_carries_fitted_values_renormalised_in_order(monkeypatch):
    received: dict = {}
    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain",
        _fake_member_factory(received),
    )
    # Deliberately insert members out of physical order to prove ordering by key.
    members = {12: _two_groups(12), 10: _two_groups(10), 11: _two_groups(11)}
    order_key = {10: 5.0, 11: 10.0, 12: 15.0}

    result = fit_grouped_series(
        "individual",
        members,
        lambda t: t,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=_series_initial(members),
        seeding="chain",
        order_key=order_key,
    )

    assert result.seeding_used == "chain"
    # Member 10 (lowest temperature) used its provided seed.
    assert received[10]["forward"]["frequency"] == 1.0
    # Member 11 was seeded from member 10's fit: frequency carried, amplitude/
    # background re-pinned to the contract, N0/phase carried.
    assert received[11]["forward"]["frequency"] == 10.0
    assert received[11]["forward"]["amplitude"] == 1.0  # re-normalised
    assert received[11]["forward"]["background"] == 0.0  # re-normalised
    assert received[11]["forward"]["N0"] == 510.0  # carried (500 + 10)
    assert received[11]["forward"]["relative_phase"] == 0.2  # carried
    # Member 12 was seeded from member 11's fit.
    assert received[12]["forward"]["frequency"] == 11.0


def test_auto_resolves_and_records_reason(monkeypatch):
    received: dict = {}
    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain",
        _fake_member_factory(received),
    )
    members = {10: _two_groups(10), 11: _two_groups(11), 12: _two_groups(12)}
    result = fit_grouped_series(
        "individual",
        members,
        lambda t: t,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=_series_initial(members),
        seeding="auto",
        order_key={10: 5.0, 11: 10.0, 12: 15.0},
    )
    assert result.seeding_used == "chain"
    assert result.seeding_reason  # non-empty explanation for the log
    # Auto with no order key falls back to independent seeds.
    received.clear()
    result2 = fit_grouped_series(
        "individual",
        members,
        lambda t: t,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=_series_initial(members),
        seeding="auto",
        order_key=None,
    )
    assert result2.seeding_used == "as_provided"
    assert received[11]["forward"]["frequency"] == 1.0  # provided, not chained


def test_failed_member_resets_chain_to_provided(monkeypatch):
    received: dict = {}

    def _fake(groups, _model_fn, **kwargs):
        run = int(groups[0].source_run_number)
        received[run] = {
            gid: {p.name: p.value for p in ps} for gid, ps in kwargs["initial_params"].items()
        }
        success = run != 11  # member 11 fails
        group_results = {}
        for g in groups:
            fitted = ParameterSet()
            fitted.add(Parameter(name="frequency", value=float(run)))
            for name in GROUP_NUISANCE_PARAMS:
                fitted.add(Parameter(name=name, value=1.0))
            group_results[g.group_id] = FitResult(
                success=success, chi_squared=1.0, reduced_chi_squared=0.1, parameters=fitted
            )
        return GroupedTimeDomainFitResult(
            success=success,
            group_results=group_results,
            shared_parameters=ParameterSet(),
            message="ok" if success else "failed",
        )

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain", _fake
    )
    members = {10: _two_groups(10), 11: _two_groups(11), 12: _two_groups(12)}
    fit_grouped_series(
        "individual",
        members,
        lambda t: t,
        global_params=["frequency"],
        local_params=list(GROUP_NUISANCE_PARAMS),
        initial_params=_series_initial(members),
        seeding="chain",
        order_key={10: 5.0, 11: 10.0, 12: 15.0},
    )
    # Member 11 chained from 10 (frequency 10); member 11 FAILED, so member 12 falls
    # back to its provided seed (frequency 1.0) rather than chaining a diverged fit.
    assert received[11]["forward"]["frequency"] == 10.0
    assert received[12]["forward"]["frequency"] == 1.0


def test_recommend_recommendation_type():
    rec = recommend_grouped_series_seeding([1, 2, 3], {1: 1.0, 2: 2.0, 3: 3.0})
    assert isinstance(rec, SeedingRecommendation)
