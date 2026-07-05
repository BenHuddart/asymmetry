"""Parity tests for the profiled/nested-locals global-fit strategy (technique L).

The profiled strategy runs an outer Minuit over the free globals only and solves
each dataset's locals independently (globals held fixed). It shares the joint
objective's minimum, so at convergence the fitted values, per-dataset χ², and the
resulting information criterion must match the joint solver's within fit
tolerance. These tests are the acceptance gate for L: values within fit
tolerance, IC within < 0.5 on representative cases, and the global HESSE error
(profile curvature = marginal curvature) reproduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


def _make_series(
    *,
    n_datasets: int,
    a0_true: float,
    lambdas: list[float],
    baseline_true: float = 0.0,
    seed: int = 0,
    sigma: float = 0.4,
) -> tuple[list[MuonDataset], dict[int, ParameterSet]]:
    """A synthetic exponential-relaxation series with a shared A0 and per-run Λ."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 8.0, 250)
    model = MODELS["ExponentialRelaxation"].function
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for i in range(n_datasets):
        clean = model(t, A0=a0_true, Lambda=lambdas[i], baseline=baseline_true)
        y = clean + rng.normal(0.0, sigma, t.size)
        err = np.full_like(t, sigma)
        datasets.append(MuonDataset(time=t, asymmetry=y, error=err, metadata={"run_number": i}))
        ps = ParameterSet()
        ps.add(Parameter("A0", 20.0, min=0.0))
        ps.add(Parameter("Lambda", 0.5, min=0.0))
        ps.add(Parameter("baseline", baseline_true, fixed=True))
        inits[i] = ps
    return datasets, inits


def _total_ic(results: dict, k: int) -> float:
    """Total-model AIC = Σ_d χ²_d + 2k (the wizard's additive AIC across runs)."""
    total_chi2 = sum(float(r.chi_squared) for r in results.values())
    return total_chi2 + 2.0 * k


def test_profiled_matches_joint_values_and_ic():
    """Profiled vs joint: shared global, per-run local; values, χ², and IC agree."""
    model = MODELS["ExponentialRelaxation"].function
    datasets, inits = _make_series(n_datasets=4, a0_true=22.0, lambdas=[0.3, 0.5, 0.8, 1.1], seed=1)
    engine = FitEngine()

    joint, g_joint = engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="joint")
    prof, g_prof = engine.global_fit(
        datasets, model, ["A0"], ["Lambda"], inits, strategy="profiled"
    )

    # Shared global agrees within fit tolerance.
    assert g_prof["A0"].value == pytest.approx(g_joint["A0"].value, abs=5e-3)

    # Every per-dataset local agrees within fit tolerance.
    for i in range(len(datasets)):
        assert prof[i].parameters["Lambda"].value == pytest.approx(
            joint[i].parameters["Lambda"].value, abs=5e-3
        )
        assert prof[i].chi_squared == pytest.approx(joint[i].chi_squared, abs=5e-2)

    # IC gate: k is identical for both (same partition: 1 global + 1 local * G).
    k = 1 + 1 * len(datasets)
    ic_delta = abs(_total_ic(prof, k) - _total_ic(joint, k))
    assert ic_delta < 0.5

    # The shared-global HESSE error is the profile curvature, which equals the
    # marginal (joint) curvature, so it must reproduce the joint error.
    assert prof[0].uncertainties["A0"] == pytest.approx(joint[0].uncertainties["A0"], rel=0.05)


def test_profiled_matches_joint_multiple_globals():
    """Two shared globals (A0 and baseline free) still profile to the joint minimum."""
    model = MODELS["ExponentialRelaxation"].function
    rng = np.random.default_rng(7)
    t = np.linspace(0.05, 8.0, 250)
    lambdas = [0.4, 0.7, 1.0]
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for i in range(3):
        clean = model(t, A0=18.0, Lambda=lambdas[i], baseline=1.5)
        y = clean + rng.normal(0.0, 0.35, t.size)
        datasets.append(
            MuonDataset(
                time=t, asymmetry=y, error=np.full_like(t, 0.35), metadata={"run_number": i}
            )
        )
        ps = ParameterSet()
        ps.add(Parameter("A0", 15.0, min=0.0))
        ps.add(Parameter("Lambda", 0.6, min=0.0))
        ps.add(Parameter("baseline", 0.0))
        inits[i] = ps
    engine = FitEngine()

    joint, g_joint = engine.global_fit(
        datasets, model, ["A0", "baseline"], ["Lambda"], inits, strategy="joint"
    )
    prof, g_prof = engine.global_fit(
        datasets, model, ["A0", "baseline"], ["Lambda"], inits, strategy="profiled"
    )

    assert g_prof["A0"].value == pytest.approx(g_joint["A0"].value, abs=1e-2)
    assert g_prof["baseline"].value == pytest.approx(g_joint["baseline"].value, abs=1e-2)
    for i in range(3):
        assert prof[i].parameters["Lambda"].value == pytest.approx(
            joint[i].parameters["Lambda"].value, abs=1e-2
        )

    k = 2 + 1 * 3
    assert abs(_total_ic(prof, k) - _total_ic(joint, k)) < 0.5


def test_profiled_single_dataset_reduces_to_single_fit():
    """A one-dataset profiled global fit matches the joint one-dataset fit."""
    model = MODELS["ExponentialRelaxation"].function
    datasets, inits = _make_series(n_datasets=1, a0_true=20.0, lambdas=[0.6], seed=3)
    engine = FitEngine()
    joint, gj = engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="joint")
    prof, gp = engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="profiled")
    assert gp["A0"].value == pytest.approx(gj["A0"].value, abs=5e-3)
    assert prof[0].parameters["Lambda"].value == pytest.approx(
        joint[0].parameters["Lambda"].value, abs=5e-3
    )


def test_profiled_rejects_unknown_strategy():
    model = MODELS["ExponentialRelaxation"].function
    datasets, inits = _make_series(n_datasets=2, a0_true=20.0, lambdas=[0.4, 0.9], seed=5)
    engine = FitEngine()
    with pytest.raises(ValueError, match="strategy"):
        engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="bogus")


def test_profiled_no_free_global_falls_back_to_separable():
    """With all globals fixed the objective is separable; profiled must still work."""
    model = MODELS["ExponentialRelaxation"].function
    datasets, inits = _make_series(n_datasets=3, a0_true=20.0, lambdas=[0.4, 0.7, 1.0], seed=9)
    # Fix the "global" so there is no free global parameter.
    for ps in inits.values():
        ps["A0"].fixed = True
    engine = FitEngine()
    prof, _ = engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="profiled")
    joint, _ = engine.global_fit(datasets, model, ["A0"], ["Lambda"], inits, strategy="joint")
    for i in range(3):
        assert prof[i].parameters["Lambda"].value == pytest.approx(
            joint[i].parameters["Lambda"].value, abs=5e-3
        )
