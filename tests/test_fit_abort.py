"""Mid-fit abort contract (fit-workflow-diagnostics).

A cancelled fit raises :class:`FitCancelledError` and records nothing — verified
both in-fit (cost-function raise) and between member fits in a series.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitCancelledError, FitEngine
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.grouped_time_domain import (
    GROUP_NUISANCE_PARAMS,
    GroupedTimeDomainFitResult,
    GroupedTimeDomainGroup,
    fit_grouped_series,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _model(t, A0, lam):  # noqa: N803
    return A0 * np.exp(-lam * np.asarray(t, dtype=float))


def _dataset() -> MuonDataset:
    t = np.linspace(0.0, 8.0, 400)
    err = np.full_like(t, 0.004)
    y = _model(t, 0.22, 0.6) + np.random.default_rng(1).normal(0.0, err)
    return MuonDataset(time=t, asymmetry=y, error=err)


def _seed() -> ParameterSet:
    ps = ParameterSet()
    ps.add(Parameter(name="A0", value=0.2, min=0.0, max=1.0))
    ps.add(Parameter(name="lam", value=0.5, min=0.0))
    return ps


# --- in-fit abort -----------------------------------------------------------


def test_in_fit_abort_raises_and_returns_no_result():
    with pytest.raises(FitCancelledError):
        FitEngine().fit(_dataset(), _model, _seed(), cancel_callback=lambda: True)


def test_no_cancel_completes_normally():
    result = FitEngine().fit(_dataset(), _model, _seed(), cancel_callback=lambda: False)
    assert result.success


def test_default_path_unaffected_by_cancel_machinery():
    # The None callback (default) is a no-op guard — existing behaviour preserved.
    result = FitEngine().fit(_dataset(), _model, _seed())
    assert result.success


# --- between-member abort ---------------------------------------------------


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


def _group_seed() -> ParameterSet:
    ps = ParameterSet()
    ps.add(Parameter(name="frequency", value=1.0, min=0.0))
    for name in GROUP_NUISANCE_PARAMS:
        ps.add(Parameter(name=name, value=0.0))
    return ps


def _series_initial() -> dict:
    return {"forward": _group_seed(), "backward": _group_seed()}


def test_between_member_abort_records_no_partial_result(monkeypatch):
    members = {10: _two_groups(10), 11: _two_groups(11), 12: _two_groups(12)}
    initial = {run: _series_initial() for run in members}

    fitted_runs: list[int] = []
    cancel_state = {"stop": False}

    def _fake_member_fit(groups, _model_fn, **kwargs):
        run = int(groups[0].source_run_number)
        fitted_runs.append(run)
        # Trip cancellation after the first member fit completes, so the *between
        # member* check (not an in-fit raise) is what stops the series.
        cancel_state["stop"] = True
        return GroupedTimeDomainFitResult(
            success=True,
            group_results={
                g.group_id: FitResult(
                    success=True,
                    chi_squared=1.0,
                    reduced_chi_squared=0.1,
                    parameters=kwargs["initial_params"][g.group_id],
                    message=str(g.group_id),
                )
                for g in groups
            },
            shared_parameters=ParameterSet(),
            message="ok",
        )

    monkeypatch.setattr(
        "asymmetry.core.fitting.grouped_time_domain.fit_grouped_time_domain",
        _fake_member_fit,
    )

    with pytest.raises(FitCancelledError):
        fit_grouped_series(
            "individual",
            members,
            _model,
            global_params=["frequency"],
            local_params=list(GROUP_NUISANCE_PARAMS),
            initial_params=initial,
            cancel_callback=lambda: cancel_state["stop"],
        )

    # Only the first member ran; the series raised before fitting the rest and
    # returned no GroupedSeriesFitResult to record.
    assert fitted_runs == [10]
