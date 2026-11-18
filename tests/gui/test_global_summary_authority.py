"""Recorded errors and free counts use result authority, never mutable recipe roles."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting import FitEngine, Parameter, ParameterSet
from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.parameters import AffineTie
from asymmetry.core.representation.base import FitSlot, RepresentationType
from asymmetry.core.representation.series import FitSeries

pytestmark = pytest.mark.gui


def test_recorded_fixed_parameter_does_not_inflate_free_count_or_data_key(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    time = np.linspace(0, 3, 30)
    result = FitEngine().fit(
        MuonDataset(time=time, asymmetry=2 * time + 1, error=np.ones(30)),
        lambda t, **p: p["m"] * t + p["b"],
        ParameterSet([Parameter("m", 2, fixed=True), Parameter("b", 0)]),
    )
    summary = MainWindow._fit_result_summary(SimpleNamespace(_settings=None), result)
    assert summary["npar"] == 1
    assert summary["ndof"] == 29
    assert FitSlot(result=summary).data_key()[1] == 30


@pytest.mark.parametrize("constraint", ["linked", "affine", "expression", "fixed"])
def test_recorded_free_count_excludes_constrained_parameters(qapp, constraint):
    from asymmetry.gui.mainwindow import MainWindow

    follower = Parameter("b", 2)
    main = Parameter("a", 2)
    if constraint == "linked":
        main.link_group = follower.link_group = 1
    elif constraint == "affine":
        follower.tie = AffineTie("a")
    elif constraint == "expression":
        follower.expr = "a"
    else:
        main.fixed = follower.fixed = True
    result = FitResult(
        success=True,
        objective=Objective.GAUSSIAN,
        parameters=ParameterSet([main, follower]),
        uncertainties={"a": 0.1, "b": 0.1},
    )
    summary = MainWindow._fit_result_summary(SimpleNamespace(_settings=None), result)
    assert summary["npar"] == (0 if constraint == "fixed" else 1)


def test_mutable_series_roles_cannot_relabel_missing_historical_errors(qapp):
    from asymmetry.gui.mainwindow import MainWindow

    series = FitSeries(
        "historical",
        RepresentationType.TIME_FB_ASYMMETRY,
        member_run_numbers=[1],
        param_roles={"a": "local", "b": "fixed"},
        results_by_run={
            1: {
                "success": True,
                "parameters": {"a": 2.0, "b": 3.0},
                "uncertainties": {"b": float("nan")},
                "field": 0.0,
                "temperature": 1.0,
                "computation_revision": "2",
            }
        },
    )
    window = SimpleNamespace(_data_browser=SimpleNamespace(get_dataset=lambda run: None))
    for role in ("local", "fixed"):
        series.param_roles["a"] = role
        row = MainWindow._build_series_rows(window, series)[0]
        assert all(math.isnan(row["errors"][name]) for name in ("a", "b"))
        assert row["computation_revision"] == "2"
        assert row["covariance"] is None
