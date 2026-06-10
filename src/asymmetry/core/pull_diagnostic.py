"""Pull-distribution diagnostic: is the analysis chain's error bar honest?

Given a completed fit — a model, its generating parameter values and the run it
was fit on — re-simulate many synthetic runs at matched statistics, refit each,
and collect the *pulls*

    pull = (θ̂ − θ_true) / σ_θ̂

for every free parameter. For a correctly calibrated analysis the pulls are
standard normal: a mean consistent with zero says the fit is unbiased, and a
**width consistent with one** says the reported errors are neither over- nor
under-estimated. A width persistently below one flags over-estimated errors,
above one under-estimated errors — the single most informative check that the
whole reduction → fit → covariance chain is trustworthy.

This module is Qt-free and engine-agnostic: the caller injects a ``refit``
callable so the diagnostic never imports the fitting engine (or iminuit)
itself. It builds on :func:`asymmetry.core.simulate.simulate_run`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.simulate import reduce_run_to_dataset, simulate_run

#: A refit maps a reduced dataset to ``(values, errors)`` dicts keyed by
#: parameter name, or ``None`` when the fit did not converge.
RefitResult = tuple[Mapping[str, float], Mapping[str, float]] | None
Refit = Callable[[MuonDataset], RefitResult]


@dataclass(frozen=True)
class ParameterPull:
    """Collected pulls for one fitted parameter."""

    name: str
    truth: float
    pulls: NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.pulls.size)

    @property
    def mean(self) -> float:
        """Sample mean of the pulls (consistent with 0 for an unbiased fit)."""
        return float(self.pulls.mean()) if self.n else float("nan")

    @property
    def mean_uncertainty(self) -> float:
        """Standard error of the mean under the N(0, 1) null (1/√N)."""
        return 1.0 / np.sqrt(self.n) if self.n else float("nan")

    @property
    def width(self) -> float:
        """Sample standard deviation (consistent with 1 for honest errors)."""
        return float(self.pulls.std(ddof=1)) if self.n > 1 else float("nan")

    @property
    def width_uncertainty(self) -> float:
        """Standard error of the width under the N(0, 1) null (1/√(2(N−1)))."""
        return 1.0 / np.sqrt(2.0 * (self.n - 1)) if self.n > 1 else float("nan")

    def verdict(self) -> str:
        """One-line calibration verdict for this parameter's error bar."""
        if self.n <= 1:
            return f"{self.name}: too few converged fits to judge"
        deviation = (self.width - 1.0) / self.width_uncertainty
        width_text = f"width {self.width:.2f}({self.width_uncertainty * 100:.0f})"
        if abs(deviation) <= 2.0:
            return f"{self.name}: errors well-calibrated ({width_text})"
        if deviation > 0:
            return f"{self.name}: errors UNDER-estimated ({width_text} > 1)"
        return f"{self.name}: errors OVER-estimated ({width_text} < 1)"


@dataclass(frozen=True)
class PullDistribution:
    """Result of a pull-distribution run over many seeds."""

    parameters: dict[str, ParameterPull]
    truth: dict[str, float]
    n_seeds: int
    n_converged: int
    total_events: float

    def verdict(self) -> str:
        """Overall headline plus a per-parameter calibration line."""
        if self.n_converged <= 1:
            return (
                f"Only {self.n_converged}/{self.n_seeds} fits converged — "
                "too few to judge error calibration."
            )
        lines = [
            f"{self.n_converged}/{self.n_seeds} fits converged at "
            f"{self.total_events / 1e6:g} MEv matched statistics."
        ]
        lines.extend(pull.verdict() for pull in self.parameters.values())
        return "\n".join(lines)


def run_pull_distribution(
    template: Run,
    model: Any,
    parameters: Mapping[str, float],
    refit: Refit,
    *,
    total_events: float,
    n_seeds: int = 200,
    seed_start: int = 0,
    track: Sequence[str] | None = None,
    alpha: float | None = None,
    background_per_bin: float = 0.0,
    time_range: tuple[float | None, float | None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PullDistribution:
    """Re-simulate, refit and histogram parameter pulls over ``n_seeds`` seeds.

    ``parameters`` are the generating (true) values bound into ``model`` for
    every synthetic run; ``track`` selects which of them to collect pulls for
    (default: all of them — pass the *free* parameters to skip fixed ones).
    Each seed simulates a run from ``template`` at ``total_events`` matched
    statistics, reduces it, optionally restricts it to ``time_range`` (the
    fit window), and refits it via the injected ``refit``. Seeds whose refit
    fails to converge — or returns a non-finite/non-positive error for a
    tracked parameter — are dropped and counted in ``n_converged``.

    ``progress(done, total)`` is called after each seed when supplied.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1.")
    tracked = list(track) if track is not None else list(parameters.keys())
    if not tracked:
        raise ValueError("No parameters to track.")

    collected: dict[str, list[float]] = {name: [] for name in tracked}
    n_converged = 0
    for index in range(n_seeds):
        run = simulate_run(
            template,
            model,
            dict(parameters),
            total_events=total_events,
            seed=seed_start + index,
            alpha=alpha,
            background_per_bin=background_per_bin,
        )
        dataset = reduce_run_to_dataset(run)
        if time_range is not None:
            dataset = dataset.time_range(*time_range)

        result = refit(dataset)
        if result is not None:
            values, errors = result
            contributions: dict[str, float] = {}
            ok = True
            for name in tracked:
                error = errors.get(name)
                if name not in values or error is None or not np.isfinite(error) or error <= 0:
                    ok = False
                    break
                contributions[name] = (float(values[name]) - float(parameters[name])) / float(error)
            if ok:
                n_converged += 1
                for name, pull in contributions.items():
                    collected[name].append(pull)

        if progress is not None:
            progress(index + 1, n_seeds)

    return PullDistribution(
        parameters={
            name: ParameterPull(
                name=name,
                truth=float(parameters[name]),
                pulls=np.asarray(values, dtype=float),
            )
            for name, values in collected.items()
        },
        truth={name: float(parameters[name]) for name in tracked},
        n_seeds=n_seeds,
        n_converged=n_converged,
        total_events=float(total_events),
    )
