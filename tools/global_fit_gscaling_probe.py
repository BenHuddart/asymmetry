"""G-scaling timing probe for the global-fit engine (PR 3 deliverable).

Fits a synthetic shared-parameter series at several dataset counts G with the
``joint`` and ``profiled`` strategies and reports per-fit wall time. The joint
solver builds one Minuit problem over ``n_global + n_local·G`` parameters, whose
per-fit cost grows super-linearly in G (a fitted power-law exponent well above
1). The profiled solver runs an outer Minuit over the globals only with G small
per-dataset local solves, so its cost grows ~linearly (exponent near 1).

Run directly::

    .venv/bin/python tools/global_fit_gscaling_probe.py

It is ``__main__``-guarded and self-contained (no corpus, no process pool), so it
can be embedded in a PR body verbatim. Determinism: fixed RNG seed per group.
"""

from __future__ import annotations

import math
import time

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitEngine
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_MODEL = MODELS["ExponentialRelaxation"].function
_N_POINTS = 300
_T_MAX = 8.0
_A0_TRUE = 22.0
_SIGMA = 0.4


def _make_series(n_datasets: int, seed: int = 0):
    """A shared-A0, per-run-Lambda exponential series with fixed noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, _T_MAX, _N_POINTS)
    # Spread the local rates smoothly across the series.
    lambdas = np.linspace(0.3, 1.2, n_datasets)
    datasets: list[MuonDataset] = []
    inits: dict[int, ParameterSet] = {}
    for i in range(n_datasets):
        clean = _MODEL(t, A0=_A0_TRUE, Lambda=float(lambdas[i]), baseline=0.0)
        y = clean + rng.normal(0.0, _SIGMA, t.size)
        datasets.append(
            MuonDataset(
                time=t,
                asymmetry=y,
                error=np.full_like(t, _SIGMA),
                metadata={"run_number": i},
            )
        )
        ps = ParameterSet()
        ps.add(Parameter("A0", 20.0, min=0.0))
        ps.add(Parameter("Lambda", 0.5, min=0.0))
        ps.add(Parameter("baseline", 0.0, fixed=True))
        inits[i] = ps
    return datasets, inits


def _time_fit(strategy: str, datasets, inits, repeats: int = 3) -> float:
    """Best-of-``repeats`` wall time (seconds) for one global fit."""
    engine = FitEngine()
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.global_fit(
            datasets,
            _MODEL,
            ["A0"],
            ["Lambda"],
            inits,
            strategy=strategy,
        )
        best = min(best, time.perf_counter() - t0)
    return best


def _fit_power_law(gs: list[int], times: list[float]) -> float:
    """Least-squares slope of log(time) vs log(G) — the scaling exponent."""
    logg = np.log(np.asarray(gs, dtype=float))
    logt = np.log(np.asarray(times, dtype=float))
    slope, _ = np.polyfit(logg, logt, 1)
    return float(slope)


def main() -> None:
    g_values = [2, 4, 8, 16, 32, 48]
    joint_times: list[float] = []
    prof_times: list[float] = []

    print(f"{'G':>3} | {'joint (s)':>11} | {'profiled (s)':>13} | {'speedup':>8}")
    print("-" * 48)
    for g in g_values:
        datasets, inits = _make_series(g, seed=1)
        tj = _time_fit("joint", datasets, inits)
        tp = _time_fit("profiled", datasets, inits)
        joint_times.append(tj)
        prof_times.append(tp)
        print(f"{g:>3} | {tj:>11.4f} | {tp:>13.4f} | {tj / tp:>7.2f}x")

    joint_exp = _fit_power_law(g_values, joint_times)
    prof_exp = _fit_power_law(g_values, prof_times)
    print("-" * 48)
    print(f"joint    scaling exponent (time ~ G^p): p = {joint_exp:.2f}")
    print(f"profiled scaling exponent (time ~ G^p): p = {prof_exp:.2f}")


if __name__ == "__main__":
    main()
