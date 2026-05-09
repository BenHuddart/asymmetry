"""Tests for FitEngine single and global fitting flows."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.parameters import Parameter, ParameterSet


class _FakeMinuit:
    def __init__(self, _cost, *initial_values, name):
        self.values = list(initial_values)
        self.errors = [0.1 for _ in initial_values]
        self.valid = True
        self.fval = 1.23
        self.covariance = np.eye(len(initial_values)) if initial_values else np.empty((0, 0))
        self.limits = [(-float("inf"), float("inf")) for _ in initial_values]
        self.method_called: str | None = None
        self.last_ncall: int | None = None

    def migrad(self, ncall=None):
        self.method_called = "migrad"
        self.last_ncall = ncall

    def simplex(self, ncall=None):
        self.method_called = "simplex"
        self.last_ncall = ncall


@pytest.fixture
def dataset() -> MuonDataset:
    t = np.linspace(0.0, 5.0, 80)
    a = 0.2 * np.exp(-0.4 * t)
    e = np.full_like(t, 0.01)
    return MuonDataset(time=t, asymmetry=a, error=e, metadata={"run_number": 1})


def _install_fake_iminuit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_iminuit = ModuleType("iminuit")
    fake_iminuit.Minuit = _FakeMinuit

    fake_cost = ModuleType("iminuit.cost")
    fake_cost.LeastSquares = lambda x, y, err, fn: SimpleNamespace(x=x, y=y, err=err, fn=fn)

    monkeypatch.setitem(sys.modules, "iminuit", fake_iminuit)
    monkeypatch.setitem(sys.modules, "iminuit.cost", fake_cost)


def _install_broken_iminuit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_iminuit = ModuleType("iminuit")
    # Deliberately omit Minuit attribute to trigger ImportError in "from iminuit import Minuit".
    monkeypatch.setitem(sys.modules, "iminuit", fake_iminuit)
    monkeypatch.setitem(sys.modules, "iminuit.cost", ModuleType("iminuit.cost"))


def _exp_model(t: np.ndarray, A0: float, Lambda: float, baseline: float = 0.0) -> np.ndarray:
    return A0 * np.exp(-Lambda * t) + baseline


def test_fit_success_with_fixed_and_free_parameters(
    dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_iminuit(monkeypatch)

    params = ParameterSet(
        [
            Parameter("A0", value=0.2, min=0.0, max=1.0),
            Parameter("Lambda", value=0.4, fixed=True),
            Parameter("baseline", value=0.0, fixed=True),
        ]
    )

    result = FitEngine().fit(dataset, _exp_model, params)

    assert result.success is True
    assert result.parameters["A0"].value == pytest.approx(0.2)
    assert result.parameters["Lambda"].value == pytest.approx(0.4)
    assert "A0" in result.uncertainties
    assert "Lambda" not in result.uncertainties
    assert result.reduced_chi_squared == pytest.approx(1.23 / (len(dataset.time) - 1))


def test_fit_import_error_returns_failure(
    dataset: MuonDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_broken_iminuit(monkeypatch)

    params = ParameterSet([Parameter("A0", value=0.2), Parameter("Lambda", value=0.4)])
    result = FitEngine().fit(dataset, _exp_model, params)

    assert result.success is False
    assert "iminuit import error" in result.message


def test_global_fit_requires_non_empty_dataset_list() -> None:
    with pytest.raises(ValueError, match="No datasets provided"):
        FitEngine().global_fit([], _exp_model, ["A0"], ["Lambda"], {})


def test_global_fit_import_error_returns_failed_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_broken_iminuit(monkeypatch)

    ds1 = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.array([0.2, 0.1]),
        error=np.array([0.01, 0.01]),
        metadata={"run_number": 1},
    )
    ds2 = MuonDataset(
        time=np.array([0.0, 1.0]),
        asymmetry=np.array([0.3, 0.2]),
        error=np.array([0.01, 0.01]),
        metadata={"run_number": 2},
    )

    init = {
        1: ParameterSet([Parameter("A0", 0.2), Parameter("Lambda", 0.4)]),
        2: ParameterSet([Parameter("A0", 0.2), Parameter("Lambda", 0.5)]),
    }

    results, fitted_global = FitEngine().global_fit(
        [ds1, ds2], _exp_model, ["A0"], ["Lambda"], init
    )

    assert set(results) == {1, 2}
    assert all(not r.success for r in results.values())
    assert len(fitted_global) == 0


def test_global_fit_success_with_fake_iminuit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_iminuit(monkeypatch)

    t = np.linspace(0.0, 3.0, 30)
    ds1 = MuonDataset(t, _exp_model(t, 0.2, 0.3), np.full_like(t, 0.01), {"run_number": 1})
    ds2 = MuonDataset(t, _exp_model(t, 0.2, 0.5), np.full_like(t, 0.01), {"run_number": 2})

    init = {
        1: ParameterSet(
            [
                Parameter("A0", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.3, min=0.0, max=2.0),
                Parameter("baseline", 0.0, fixed=True),
            ]
        ),
        2: ParameterSet(
            [
                Parameter("A0", 0.2, min=0.0, max=1.0),
                Parameter("Lambda", 0.5, min=0.0, max=2.0),
                Parameter("baseline", 0.0, fixed=True),
            ]
        ),
    }

    results, fitted_global = FitEngine().global_fit(
        [ds1, ds2],
        _exp_model,
        global_params=["A0"],
        local_params=["Lambda", "baseline"],
        initial_params=init,
        method="simplex",
        max_calls=123,
    )

    assert set(results) == {1, 2}
    assert fitted_global["A0"].value == pytest.approx(0.2)
    assert results[1].parameters["Lambda"].value == pytest.approx(0.3)
    assert results[2].parameters["Lambda"].value == pytest.approx(0.5)
    assert results[1].parameters["baseline"].fixed is True
    assert results[1].success is True


def test_global_fit_rejects_non_finite_initial_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_iminuit(monkeypatch)

    t = np.linspace(0.0, 1.0, 10)
    ds = MuonDataset(t, _exp_model(t, 0.2, 0.3), np.full_like(t, 0.01), {"run_number": 1})

    init = {
        1: ParameterSet(
            [
                Parameter("A0", 0.2),
                Parameter("Lambda", np.nan),
            ]
        )
    }

    with pytest.raises(ValueError, match="non-finite initial value"):
        FitEngine().global_fit([ds], _exp_model, ["A0"], ["Lambda"], init)


def test_global_fit_all_local_reuses_single_fit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    t = np.linspace(0.0, 2.0, 20)
    ds1 = MuonDataset(t, _exp_model(t, 0.2, 0.3), np.full_like(t, 0.01), {"run_number": 1})
    ds2 = MuonDataset(t, _exp_model(t, 0.3, 0.5), np.full_like(t, 0.01), {"run_number": 2})

    init = {
        1: ParameterSet([Parameter("A0", 0.2), Parameter("Lambda", 0.3)]),
        2: ParameterSet([Parameter("A0", 0.3), Parameter("Lambda", 0.5)]),
    }

    captured_calls: list[tuple[int, str]] = []

    def _fake_fit(self, dataset, _model_fn, parameters, t_min=None, t_max=None, method="migrad"):
        captured_calls.append((int(dataset.run_number), method))
        return FitResult(
            success=True,
            chi_squared=float(dataset.run_number),
            reduced_chi_squared=0.1,
            parameters=parameters,
            message=f"single-fit-{int(dataset.run_number)}",
        )

    monkeypatch.setattr(FitEngine, "fit", _fake_fit)

    results, fitted_global = FitEngine().global_fit(
        [ds1, ds2],
        _exp_model,
        global_params=[],
        local_params=["A0", "Lambda"],
        initial_params=init,
        method="simplex",
    )

    assert captured_calls == [(1, "simplex"), (2, "simplex")]
    assert len(fitted_global) == 0
    assert results[1].message == "single-fit-1"
    assert results[2].message == "single-fit-2"
