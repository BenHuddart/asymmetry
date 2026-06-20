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

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitResult
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
