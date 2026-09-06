"""Robust chained F-B asymmetry series fit (the EuO near-T_C bistability).

A block-separable batch (every free parameter Local) is fit one run at a time. With
``seeding="chain"`` each run warm-starts from the previous good run, and a run that
converges onto the spurious branch (amplitude collapsed / frequency off the trend) is
reseeded from the good-run trend and refit. These tests use a deterministic *bistable*
fake engine — the real branch is found only when the seed frequency is near the run's
true frequency, otherwise the fit "converges" to the spurious high-frequency, near-zero
amplitude solution — so chain vs. reseed behaviour is exercised without iminuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.engine import (
    COST_FACTORIES,
    GAUSSIAN_COST,
    POISSON_COST,
    CostFactory,
    FitCancelledError,
    FitResult,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet
from asymmetry.core.fitting.series import fit_asymmetry_series

# Runs ordered by temperature; the true precession frequency descends toward T_C.
_REAL_FREQ = {2960: 30.0, 2955: 26.0, 2950: 18.0, 2945: 12.0, 2940: 8.0}
_ORDER = {2960: 10.0, 2955: 30.0, 2950: 50.0, 2945: 60.0, 2940: 67.0}
_CAPTURE_WINDOW = 5.0  # the real branch is found only within this of the truth
_SPURIOUS_FREQ = 30.0


def _datasets():
    time = np.linspace(0.1, 8.0, 16)
    out = []
    for run in _REAL_FREQ:
        out.append(
            MuonDataset(
                time=time,
                asymmetry=np.zeros_like(time),
                error=np.ones_like(time),
                metadata={"run_number": run, "temperature": _ORDER[run]},
            )
        )
    return out


def _seed(freq: float = _SPURIOUS_FREQ) -> ParameterSet:
    ps = ParameterSet()
    ps.add(Parameter(name="A_1", value=20.0, min=0.0, max=100.0))
    ps.add(Parameter(name="frequency", value=freq, min=0.0, max=100.0))
    ps.add(Parameter(name="lambda", value=0.1, min=0.0))
    return ps


def _initial(seed_freq: float = _SPURIOUS_FREQ) -> dict[int, ParameterSet]:
    return {run: _seed(seed_freq) for run in _REAL_FREQ}


class _BistableEngine:
    """Fake engine: finds the real branch only when seeded near the truth."""

    def __init__(self) -> None:
        self.seed_freqs: dict[int, list[float]] = {}

    def fit(self, dataset, _model_fn, parameters, **_kwargs) -> FitResult:
        run = int(dataset.run_number)
        seed_freq = float(parameters["frequency"].value)
        self.seed_freqs.setdefault(run, []).append(seed_freq)
        real = _REAL_FREQ[run]
        fitted = ParameterSet()
        if abs(seed_freq - real) <= _CAPTURE_WINDOW:
            fitted.add(Parameter(name="A_1", value=20.0))
            fitted.add(Parameter(name="frequency", value=real))
            fitted.add(Parameter(name="lambda", value=0.1))
            return FitResult(success=True, reduced_chi_squared=1.0, parameters=fitted)
        # Spurious branch: high frequency, amplitude collapsed to ~0.
        fitted.add(Parameter(name="A_1", value=0.05))
        fitted.add(Parameter(name="frequency", value=_SPURIOUS_FREQ))
        fitted.add(Parameter(name="lambda", value=0.1))
        return FitResult(success=True, reduced_chi_squared=3.0, parameters=fitted)


def _run(seeding: str, engine=None):
    engine = engine or _BistableEngine()
    result = fit_asymmetry_series(
        _datasets(),
        lambda t: t,
        global_params=[],
        local_params=["A_1", "frequency", "lambda"],
        initial_params=_initial(),
        fit_engine=engine,
        seeding=seeding,
        order_key=_ORDER,
        amplitude_param="A_1",
        frequency_param="frequency",
    )
    return result, engine


def _fitted_freq(result, run: int) -> float:
    return float(result.results[run].parameters["frequency"].value)


def test_independent_seeds_strand_runs_on_the_spurious_branch():
    # Every run seeded at 30 MHz: only the base-T run (real 30) lands on the real
    # branch; the descending runs stick to the spurious high frequency.
    result, _ = _run("as_provided")
    assert _fitted_freq(result, 2960) == 30.0
    assert _fitted_freq(result, 2950) == _SPURIOUS_FREQ  # stranded


def test_chain_with_reseed_recovers_the_descending_trend():
    result, engine = _run("chain")
    # Every run ends on its real descending frequency, not the spurious branch.
    for run, real in _REAL_FREQ.items():
        assert _fitted_freq(result, run) == real, run
    # The run whose naive chain seed overshot the capture window was reseeded.
    assert result.reseeded_runs, "expected at least one detect-and-reseed"
    assert result.seeding_used == "chain"


def test_chain_visits_runs_in_scan_order():
    result, _ = _run("chain")
    assert list(result.order) == sorted(_REAL_FREQ, key=lambda r: _ORDER[r])


def test_auto_resolves_to_chain_for_ordered_scan():
    result, _ = _run("auto")
    assert result.seeding_used == "chain"
    assert result.seeding_reason


def test_reseed_only_fires_for_converged_spurious_not_hard_failure():
    # A run that fails outright must not be reseeded (the chain resets to its own
    # provided seed for the next run instead).
    class _FailingEngine(_BistableEngine):
        def fit(self, dataset, model_fn, parameters, **kwargs):
            if int(dataset.run_number) == 2950:
                ps = ParameterSet()
                ps.add(Parameter(name="frequency", value=0.0))
                return FitResult(success=False, parameters=ps, message="diverged")
            return super().fit(dataset, model_fn, parameters, **kwargs)

    result, _ = _run("chain", engine=_FailingEngine())
    assert result.results[2950].success is False
    assert 2950 not in result.reseeded_runs
    # The run after the failure falls back to its provided seed (30 MHz) → real 30
    # is outside its capture window, so it lands spurious rather than chaining a
    # diverged seed forward. (Confirms the chain is not poisoned by the failure.)
    assert 2945 in result.results


def test_global_echoed_for_block_separable():
    result, _ = _run("chain")
    assert len(result.fitted_global) == 0  # no global params in this batch


# --- Parallel ``as_provided`` execution --------------------------------------------
#
# ``as_provided`` runs share no state, so a process pool cannot change results — only
# their wall-clock. These exercise the equivalence, the graceful fallbacks (pool
# unavailable / unpicklable payload), the chain-stays-sequential guarantee, and prompt
# cancellation without orphaned workers, mirroring the grouped-series parallel tests.


def _exp_model(t, A0, Lambda):  # noqa: N803 — module-level so it is picklable under spawn
    return A0 * np.exp(-Lambda * np.asarray(t, dtype=float))


def _real_batch():
    """A tiny noise-free exponential-relaxation batch that fits deterministically."""
    time = np.linspace(0.1, 8.0, 32)
    truth = {101: (0.24, 0.30), 102: (0.20, 0.55), 103: (0.18, 0.80)}
    datasets: list[MuonDataset] = []
    initial: dict[int, ParameterSet] = {}
    for run, (a0, lam) in truth.items():
        datasets.append(
            MuonDataset(
                time=time,
                asymmetry=a0 * np.exp(-lam * time),
                error=np.full_like(time, 0.01),
                metadata={"run_number": run},
            )
        )
        ps = ParameterSet()
        ps.add(Parameter(name="A0", value=0.2, min=0.0, max=1.0))
        ps.add(Parameter(name="Lambda", value=0.5, min=0.0, max=5.0))
        initial[run] = ps
    return datasets, initial


def _fit_real_batch(max_workers, *, model=_exp_model, **overrides):
    datasets, initial = _real_batch()
    kwargs = dict(
        global_params=[],
        local_params=["A0", "Lambda"],
        initial_params=initial,
        seeding="as_provided",
        amplitude_param="A0",
        max_workers=max_workers,
    )
    kwargs.update(overrides)
    return fit_asymmetry_series(datasets, model, **kwargs)


def test_resolve_series_workers_is_opt_in_and_clamped():
    from asymmetry.core.fitting.series import _resolve_series_workers

    # Opt-in: None (default) and 1 keep the sequential path.
    assert _resolve_series_workers(None, 10) == 1
    assert _resolve_series_workers(1, 10) == 1
    # A positive count is honoured but never exceeds the run count.
    assert _resolve_series_workers(4, 10) == 4
    assert _resolve_series_workers(16, 3) == 3
    # A degenerate batch never spins up a pool.
    assert _resolve_series_workers(8, 1) == 1


def test_parallel_as_provided_matches_sequential():
    serial = _fit_real_batch(1)
    parallel = _fit_real_batch(4)

    # Independent runs → the worker count cannot change any result.
    assert serial.order == parallel.order
    assert set(serial.results) == set(parallel.results)
    for run, serial_result in serial.results.items():
        parallel_result = parallel.results[run]
        for name in serial_result.parameters.names:
            assert float(serial_result.parameters[name].value) == pytest.approx(
                float(parallel_result.parameters[name].value), rel=0, abs=0
            )
    # The batch-wide diagnosis and per-run quality are byte-identical too.
    assert serial.member_quality == parallel.member_quality
    assert serial.reseeded_runs == parallel.reseeded_runs == ()


def test_parallel_falls_back_when_pool_unavailable(monkeypatch):
    # A constrained environment where no spawn pool can start must still complete via
    # the sequential path with identical results.
    import asymmetry.core.fitting.series as series_module

    calls: list[int] = []

    def _no_pool(workers):
        calls.append(workers)
        return None

    monkeypatch.setattr(series_module, "open_spawn_pool", _no_pool)
    result = _fit_real_batch(4)
    assert calls  # parallel was attempted
    # Equal to the sequential path.
    serial = _fit_real_batch(1)
    for run, serial_result in serial.results.items():
        for name in serial_result.parameters.names:
            assert float(result.results[run].parameters[name].value) == pytest.approx(
                float(serial_result.parameters[name].value), rel=0, abs=0
            )


def test_parallel_falls_back_for_unpicklable_model():
    # A model captured as a local closure cannot cross a process boundary; the batch
    # must still complete by transparently falling back to the sequential path.
    def _closure_model(t, A0, Lambda):  # noqa: N803 — local closure, not picklable
        return _exp_model(t, A0, Lambda)

    result = _fit_real_batch(4, model=_closure_model)
    assert set(result.results) == {101, 102, 103}
    assert all(r.success for r in result.results.values())


def test_chain_stays_sequential_even_with_workers(monkeypatch):
    # Chain seeding is inherently sequential: the parallel path must never be reached,
    # and the chain/reseed behaviour is unchanged from the sequential contract.
    import asymmetry.core.fitting.series as series_module

    monkeypatch.setattr(
        series_module,
        "_fit_runs_parallel",
        lambda *a, **k: pytest.fail("chain must not dispatch to the process pool"),
    )
    engine = _BistableEngine()
    result = fit_asymmetry_series(
        _datasets(),
        lambda t: t,
        global_params=[],
        local_params=["A_1", "frequency", "lambda"],
        initial_params=_initial(),
        fit_engine=engine,
        seeding="chain",
        order_key=_ORDER,
        amplitude_param="A_1",
        frequency_param="frequency",
        max_workers=8,
    )
    for run, real in _REAL_FREQ.items():
        assert _fitted_freq(result, run) == real, run
    assert result.reseeded_runs
    assert result.seeding_used == "chain"


class _FakeProc:
    """Stand-in worker process recording that teardown killed and reaped it."""

    def __init__(self) -> None:
        self.killed = False
        self.joined = False

    def kill(self) -> None:
        self.killed = True

    def join(self, timeout=None) -> None:
        self.joined = True


class _EagerFakePool:
    """In-process fake pool: runs each submission eagerly, records teardown.

    Avoids the cost of real spawn workers while still driving the cancellation path
    (``open_spawn_pool`` is monkeypatched to return this). ``_processes`` lets the real
    ``terminate_spawn_pool`` exercise its kill-and-reap so a cancel leaves no orphan.
    """

    def __init__(self) -> None:
        from concurrent.futures import Future

        self._future_cls = Future
        self._processes = {1: _FakeProc(), 2: _FakeProc()}
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, fn, payload):
        future = self._future_cls()
        future.set_result(fn(payload))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def test_parallel_cancellation_tears_down_pool_and_raises(monkeypatch):
    import asymmetry.core.fitting.series as series_module

    pool = _EagerFakePool()
    monkeypatch.setattr(series_module, "open_spawn_pool", lambda workers: pool)

    # False on the pre-open check, then True once collection is under way → cancel fires
    # between completions, tearing the pool down before all runs are collected.
    state = {"calls": 0}

    def _cancel() -> bool:
        state["calls"] += 1
        return state["calls"] >= 3

    with pytest.raises(FitCancelledError):
        _fit_real_batch(4, cancel_callback=_cancel)

    # terminate_spawn_pool killed and reaped every worker (no orphans, no zombies) and
    # tore the pool down without waiting on in-flight fits.
    assert all(proc.killed and proc.joined for proc in pool._processes.values())
    assert any(cancel_futures for _wait, cancel_futures in pool.shutdown_calls)


# --- What a batch payload carries across the process boundary ----------------------
#
# A payload pickles whatever its dataset still references. The source run's raw
# detector histograms are two orders of magnitude larger than the fitted arrays they
# hang off, and a time-domain fit never reads them, so a time-domain payload ships a
# fit record. A count-domain (Poisson) cost does read them, so its payload keeps the
# full record. The cost factory declares which through ``needs_histograms``.


class _RecordingFakePool(_EagerFakePool):
    """An eager in-process pool that also keeps every payload it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[tuple] = []

    def submit(self, fn, payload):
        self.payloads.append(payload)
        return super().submit(fn, payload)


def _histogram_backed_batch():
    """The exponential batch, each run backed by 15 detector histograms."""
    datasets, initial = _real_batch()
    for dataset in datasets:
        dataset.run = Run(
            run_number=int(dataset.metadata["run_number"]),
            histograms=[
                Histogram(counts=np.full(4096, 120.0), bin_width=0.016, t0_bin=10)
                for _ in range(15)
            ],
            metadata={"title": "Synthetic", "temperature": 5.0, "field": 100.0},
            grouping={"groups": {0: [0, 1], 1: [2, 3]}},
        )
    return datasets, initial


def _fit_histogram_backed_batch(max_workers, **overrides):
    datasets, initial = _histogram_backed_batch()
    kwargs = dict(
        global_params=[],
        local_params=["A0", "Lambda"],
        initial_params=initial,
        seeding="as_provided",
        amplitude_param="A0",
        max_workers=max_workers,
    )
    kwargs.update(overrides)
    return datasets, fit_asymmetry_series(datasets, _exp_model, **kwargs)


def _record_batch_payloads(monkeypatch, **overrides):
    """Fit the histogram-backed batch through a recording pool; return (pool, datasets)."""
    import asymmetry.core.fitting.series as series_module

    pool = _RecordingFakePool()
    monkeypatch.setattr(series_module, "open_spawn_pool", lambda workers: pool)
    datasets, _ = _fit_histogram_backed_batch(4, **overrides)
    return pool, datasets


def test_time_domain_batch_payloads_carry_no_histograms(monkeypatch):
    # No cost factory → the default Gaussian objective, which reads only the fitted
    # arrays. Every payload ships a fit record: same provenance, empty histograms.
    pool, datasets = _record_batch_payloads(monkeypatch)

    assert len(pool.payloads) == len(datasets)
    for payload in pool.payloads:
        shipped = payload[1]
        assert shipped.run is not None
        assert shipped.run.histograms == []
        # Provenance is what results are keyed and labelled by, so it survives.
        assert shipped.run_number in {101, 102, 103}
        assert shipped.run_label == str(shipped.run_number)
        assert shipped.run.grouping == {"groups": {0: [0, 1], 1: [2, 3]}}
    # The caller's own datasets are untouched — they still hold their counts.
    assert all(len(dataset.run.histograms) == 15 for dataset in datasets)


def _payloads_for_cost(monkeypatch, cost_factory):
    """Payloads the parallel dispatcher builds under one cost factory.

    Drives ``_fit_runs_parallel`` directly rather than through
    ``fit_asymmetry_series``: the built-in factories carry lambdas, so the caller's
    ``_series_payload_picklable`` check would divert any explicitly-passed factory to
    the sequential path before a payload is ever built. The payload site itself is
    what this pins.
    """
    import asymmetry.core.fitting.series as series_module

    pool = _RecordingFakePool()
    monkeypatch.setattr(series_module, "open_spawn_pool", lambda workers: pool)
    datasets, initial = _histogram_backed_batch()
    dataset_by_run = {int(ds.run_number): ds for ds in datasets}
    series_module._fit_runs_parallel(
        list(dataset_by_run),
        dataset_by_run,
        _exp_model,
        initial,
        t_min=None,
        t_max=None,
        method="migrad",
        minos=False,
        cost_factory=cost_factory,
        error_oversampling=1.0,
        cancel_callback=None,
        workers=4,
    )
    return pool


def test_gaussian_cost_batch_payloads_also_carry_no_histograms(monkeypatch):
    # An explicitly-passed Gaussian factory declares needs_histograms=False and so
    # reaches the same fit-record payload as the implicit default.
    pool = _payloads_for_cost(monkeypatch, GAUSSIAN_COST)

    assert len(pool.payloads) == 3
    assert all(payload[1].run.histograms == [] for payload in pool.payloads)


def test_count_domain_batch_payloads_keep_the_counts(monkeypatch):
    # The Poisson cost declares needs_histograms=True: the count-domain setup behind it
    # reads run.histograms, so the full record has to cross the boundary.
    pool = _payloads_for_cost(monkeypatch, POISSON_COST)

    assert len(pool.payloads) == 3
    assert all(len(payload[1].run.histograms) == 15 for payload in pool.payloads)


def test_every_cost_factory_declares_whether_it_reads_counts():
    # The distinction is a declared property of the cost, not a name the payload site
    # pattern-matches on, and a new factory cannot be built without stating it.
    assert GAUSSIAN_COST.needs_histograms is False
    assert POISSON_COST.needs_histograms is True
    assert all(isinstance(f.needs_histograms, bool) for f in COST_FACTORIES.values())
    with pytest.raises(TypeError):
        CostFactory("undeclared", lambda *a: None, lambda *a: 0.0)


def test_dropping_the_counts_does_not_change_the_fit(monkeypatch):
    # Parallel ships fit records, sequential fits the full records in-process. The
    # fitted arrays are identical either way, so the results must be too — this is a
    # payload-size change, not a numerical one.
    import asymmetry.core.fitting.series as series_module

    monkeypatch.setattr(series_module, "open_spawn_pool", lambda workers: _RecordingFakePool())
    _, parallel = _fit_histogram_backed_batch(4)
    monkeypatch.undo()
    _, sequential = _fit_histogram_backed_batch(1)

    assert parallel.order == sequential.order
    assert set(parallel.results) == set(sequential.results)
    for run, from_record in parallel.results.items():
        from_full = sequential.results[run]
        assert from_record.success == from_full.success
        for name in from_record.parameters.names:
            assert float(from_record.parameters[name].value) == pytest.approx(
                float(from_full.parameters[name].value), rel=0, abs=0
            )
    assert parallel.member_quality == sequential.member_quality
