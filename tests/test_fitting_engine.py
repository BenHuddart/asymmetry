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


def test_global_fit_requires_initial_params_for_each_dataset() -> None:
    t = np.linspace(0.0, 1.0, 10)
    ds1 = MuonDataset(t, _exp_model(t, 0.2, 0.3), np.full_like(t, 0.01), {"run_number": 1})
    ds2 = MuonDataset(t, _exp_model(t, 0.2, 0.4), np.full_like(t, 0.01), {"run_number": 2})

    init = {
        1: ParameterSet([Parameter("A0", 0.2), Parameter("Lambda", 0.3)]),
    }

    with pytest.raises(
        KeyError, match=r"initial parameter sets missing for dataset run numbers \[2\]"
    ):
        FitEngine().global_fit([ds1, ds2], _exp_model, ["A0"], ["Lambda"], init)


def test_global_fit_rejects_duplicate_dataset_run_numbers() -> None:
    t = np.linspace(0.0, 1.0, 10)
    ds1 = MuonDataset(t, _exp_model(t, 0.2, 0.3), np.full_like(t, 0.01), {"run_number": 1})
    ds2 = MuonDataset(t, _exp_model(t, 0.2, 0.4), np.full_like(t, 0.01), {"run_number": 1})

    init = {
        1: ParameterSet([Parameter("A0", 0.2), Parameter("Lambda", 0.3)]),
    }

    with pytest.raises(ValueError, match="Global fitting requires unique dataset run numbers"):
        FitEngine().global_fit([ds1, ds2], _exp_model, ["A0"], ["Lambda"], init)


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

    def _fake_fit(
        self,
        dataset,
        _model_fn,
        parameters,
        t_min=None,
        t_max=None,
        method="migrad",
        minos=False,
        cancel_callback=None,
        frequency_offsets=None,
        cost_factory=None,
    ):
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


# --- selectable fit cost (Gaussian √N vs Poisson Cash) ----------------------


def _count_dataset(counts: np.ndarray, times: np.ndarray) -> MuonDataset:
    """A raw-count dataset: ``asymmetry`` holds the counts, error = √N (Cash ignores it)."""
    return MuonDataset(
        time=times,
        asymmetry=counts.astype(float),
        error=np.sqrt(np.clip(counts, 1.0, None)),
        metadata={},
    )


def _decay_count_model(t, N0, tau):  # noqa: N803 — physics names
    """Expected counts N0·e^(−t/τ) — a positive count expectation for Cash."""
    return N0 * np.exp(-t / tau)


def test_cost_factory_gaussian_matches_default_least_squares():
    """GAUSSIAN_COST must reproduce the no-factory least-squares fit exactly."""
    from asymmetry.core.fitting.engine import GAUSSIAN_COST

    t = np.linspace(0.05, 6.0, 120)
    rng = np.random.default_rng(7)
    counts = rng.poisson(800.0 * np.exp(-t / 2.2)).astype(float)
    ds = _count_dataset(counts, t)
    params = ParameterSet([Parameter("N0", 700.0, min=1.0), Parameter("tau", 2.0, min=0.1)])
    engine = FitEngine()

    base = engine.fit(ds, _decay_count_model, params)
    via_factory = engine.fit(ds, _decay_count_model, params, cost_factory=GAUSSIAN_COST)
    assert base.success and via_factory.success
    assert via_factory.chi_squared == pytest.approx(base.chi_squared, rel=1e-12)
    assert via_factory.parameters["N0"].value == pytest.approx(base.parameters["N0"].value, rel=1e-9)
    assert via_factory.parameters["tau"].value == pytest.approx(
        base.parameters["tau"].value, rel=1e-9
    )


def test_cost_factory_poisson_and_gaussian_agree_at_high_counts():
    """At high counts the Poisson and Gaussian fits converge to the same answer."""
    from asymmetry.core.fitting.engine import GAUSSIAN_COST, POISSON_COST

    t = np.linspace(0.05, 6.0, 200)
    rng = np.random.default_rng(11)
    counts = rng.poisson(5000.0 * np.exp(-t / 2.2)).astype(float)
    ds = _count_dataset(counts, t)
    params = ParameterSet([Parameter("N0", 4000.0, min=1.0), Parameter("tau", 2.0, min=0.1)])
    engine = FitEngine()

    gauss = engine.fit(ds, _decay_count_model, params, cost_factory=GAUSSIAN_COST)
    pois = engine.fit(ds, _decay_count_model, params, cost_factory=POISSON_COST)
    assert gauss.success and pois.success
    # √N weighting and Cash agree to <0.5% when every bin is well-populated.
    assert pois.parameters["N0"].value == pytest.approx(gauss.parameters["N0"].value, rel=5e-3)
    assert pois.parameters["tau"].value == pytest.approx(gauss.parameters["tau"].value, rel=5e-3)


def test_cost_factory_poisson_less_biased_than_gaussian_at_low_counts():
    """√N weighting biases the fitted normalisation low at low counts; Cash does not.

    A small Monte-Carlo over independent low-count realisations: the mean Poisson
    estimate of N0 sits closer to the truth than the mean Gaussian estimate, and
    the Gaussian mean is biased *low* (the known √N-weighting pathology). This is
    the bias the fgAll→Poisson migration removes by default.
    """
    from asymmetry.core.fitting.engine import GAUSSIAN_COST, POISSON_COST

    t = np.linspace(0.05, 6.0, 60)
    n0_true, tau_true = 12.0, 2.2
    mean_counts = n0_true * np.exp(-t / tau_true)
    rng = np.random.default_rng(2024)
    engine = FitEngine()

    gauss_n0: list[float] = []
    pois_n0: list[float] = []
    for _ in range(120):
        counts = rng.poisson(mean_counts).astype(float)
        ds = _count_dataset(counts, t)
        seed = ParameterSet([Parameter("N0", 12.0, min=0.1), Parameter("tau", 2.2, min=0.1)])
        g = engine.fit(ds, _decay_count_model, seed, cost_factory=GAUSSIAN_COST)
        p = engine.fit(ds, _decay_count_model, seed, cost_factory=POISSON_COST)
        if g.success:
            gauss_n0.append(g.parameters["N0"].value)
        if p.success:
            pois_n0.append(p.parameters["N0"].value)

    gauss_mean = float(np.mean(gauss_n0))
    pois_mean = float(np.mean(pois_n0))
    # Gaussian √N is biased low; Poisson Cash recovers the truth far better.
    assert gauss_mean < n0_true
    assert abs(pois_mean - n0_true) < abs(gauss_mean - n0_true)


def test_poisson_cash_primitive_matches_definition():
    """engine.poisson_cash == 2·Σ(μ − n + n·ln(n/μ)), with n=0 → 2μ."""
    from asymmetry.core.fitting.engine import poisson_cash

    n = np.array([0.0, 5.0, 10.0])
    mu = np.array([3.0, 5.0, 8.0])
    expected = 2.0 * np.sum(mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0))
    assert poisson_cash(n, mu) == pytest.approx(expected)
    # Exact minimum at μ = n (for n > 0): Cash → 0.
    assert poisson_cash(np.array([7.0]), np.array([7.0])) == pytest.approx(0.0, abs=1e-12)
