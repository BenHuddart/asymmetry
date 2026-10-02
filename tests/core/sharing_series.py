"""Simulated series for the sharing ladder and the trend objective's tests.

Each is simulated from the repo's own components with muon-decay noise, on a
zero-field temperature scan, and modelled on a case in
``docs/plans/global-wizard-trend-objective.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from itertools import count
from types import MappingProxyType

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.global_search.sharing_ladder import (
    SharingLadder,
    climb_sharing_ladder,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

_MUON_LIFETIME_US = 2.197
_LIMITS = {"A_1": 100.0, "A_2": 100.0, "Lambda": 50.0, "sigma": 50.0, "Delta": 10.0, "nu": 100.0}


class SimulatedSeries:
    """A simulated series and its independent per-run fits, runs numbered 1, 2, 3…"""

    def __init__(
        self,
        expression: str,
        truths: list[dict[str, float]],
        *,
        sigma: float,
        n_points: int = 400,
        t_max: float = 10.0,
        fixed: tuple[str, ...] = (),
        axis: list[float] | None = None,
        starts: list[dict[str, float]] | None = None,
    ) -> None:
        """``starts`` are where each run's independent fit begins: the truth by default."""
        self.model = CompositeModel.from_expression(expression)
        self.axis = axis or [float(run) for run in range(1, len(truths) + 1)]
        rng = np.random.default_rng(11)
        time = np.linspace(0.02, t_max, n_points)
        # Counting errors grow as the muons decay.
        error = sigma * np.exp(time / (2.0 * _MUON_LIFETIME_US))
        self.datasets: list[MuonDataset] = []
        self.base_by_run: dict[int, ParameterSet] = {}
        for run, (truth, start) in enumerate(zip(truths, starts or truths, strict=True), start=1):
            dataset = MuonDataset(
                time=time,
                asymmetry=self.model.function(time, **truth) + rng.normal(0.0, error),
                error=error,
                metadata={
                    "run_number": run,
                    "run_label": str(run),
                    "temperature": self.axis[run - 1],
                    "field": 0.0,
                },
            )
            base = ParameterSet(
                [
                    Parameter(
                        name,
                        start[name],
                        min=0.1 if name == "beta" else 0.0 if name in _LIMITS else -np.inf,
                        max=3.0 if name == "beta" else _LIMITS.get(name, np.inf),
                        fixed=name in fixed,
                    )
                    for name in self.model.param_names
                ]
            )
            self.datasets.append(dataset)
            self.base_by_run[run] = base

    @cached_property
    def all_local(self) -> dict[int, FitResult]:
        """Each run fitted on its own, from its start."""
        return {
            int(dataset.run_number): FitEngine().fit(
                dataset, self.model.function, self.base_by_run[int(dataset.run_number)]
            )
            for dataset in self.datasets
        }

    def climb(self, *, further: int | None = None, **options) -> SharingLadder:
        """Climb the ladder, trying at most ``further`` further parameters (``None``: all)."""
        tried = count()
        return climb_sharing_ladder(
            self.datasets,
            self.model,
            all_local_results=self.all_local,
            base_by_run=self.base_by_run,
            axis_values=self.axis,
            climb_further=lambda: further is None or next(tried) < further,
            **{"cancel_callback": lambda: False, **options},
        )


def glassy_series(*, runs_with_lost_asymmetry: int) -> SimulatedSeries:
    """Stretched exponential + constant, the rate falling by three decades (YMnAl).

    The five slowest runs cannot tell amplitude from background: their
    independent fits start ten units of asymmetry along that valley and stay
    there. The exponent truly varies. The first ``runs_with_lost_asymmetry``
    runs carry less amplitude.
    """
    rates = np.logspace(0.5, -2.5, 12)
    exponents = np.linspace(0.5, 1.0, 12)
    truths = [
        {
            "A_1": 13.0 if index < runs_with_lost_asymmetry else 20.0,
            "Lambda": rates[index],
            "beta": exponents[index],
            "A_bg": 5.0,
        }
        for index in range(12)
    ]
    valley = [
        {**truth, "A_1": truth["A_1"] + 10.0, "A_bg": truth["A_bg"] - 10.0} for truth in truths
    ]
    return SimulatedSeries(
        "StretchedExponential + Constant", truths, sigma=0.15, starts=truths[:7] + valley[7:]
    )


def hopping_series(
    *,
    runs: int = 10,
    sigma: float = 0.15,
    amplitude_scale: Mapping[int, float] = MappingProxyType({}),
    background: bool = True,
) -> SimulatedSeries:
    """Dynamic Gaussian KT + constant: one static width, a hop rate rising (copper).

    ``amplitude_scale`` multiplies the asymmetry of the runs it names (Re₆Zr:
    three runs with 2 % more).
    """
    hop_rates = np.logspace(-1.5, 0.4, runs)
    return SimulatedSeries(
        "DynamicGaussianKT" + " + Constant" * background,
        [
            {
                "A_1": 18.0 * amplitude_scale.get(run, 1.0),
                "Delta": 0.39,
                "nu": hop_rates[run - 1],
                "B_L": 0.0,
                **({"A_bg": 4.0} if background else {}),
            }
            for run in range(1, runs + 1)
        ],
        sigma=sigma,
        n_points=300,
        t_max=12.0,
        fixed=("B_L",),
    )


def two_line_series(
    fractions: Sequence[float],
    *,
    second_line_sign: float = 1.0,
    fixed: tuple[str, ...] = ("phase_1", "phase_3"),
) -> SimulatedSeries:
    """Two Gaussian-damped precession lines sharing 20 units of asymmetry, + constant.

    ``fractions`` is the first line's share run by run; both widths rise along
    the series. The amplitudes are not limited, so either may be fitted negative.
    """
    return SimulatedSeries(
        "Oscillatory * Gaussian + Oscillatory * Gaussian + Constant",
        [
            {
                "A_1": 20.0 * fraction,
                "frequency_1": 2.0,
                "phase_1": 0.0,
                "sigma_2": 0.5 + 0.03 * index,
                "A_3": second_line_sign * 20.0 * (1.0 - fraction),
                "frequency_3": 2.4,
                "phase_3": 0.0,
                "sigma_4": 0.12 + 0.01 * index,
                "A_bg": 2.0,
            }
            for index, fraction in enumerate(fractions)
        ],
        sigma=0.12,
        n_points=300,
        t_max=8.0,
        fixed=fixed,
    )


#: The first line's share rising through a transition at the middle of twelve runs.
TRANSITION = 0.15 + 0.7 / (1.0 + np.exp(-(np.arange(12) - 5.5) / 1.2))
