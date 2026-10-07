"""Synthetic vector-polarisation runs for the projection tests (invented numbers, no data).

Six detectors look along ±x, ±y, ±z, so with alpha 1 each projection's
asymmetry is exactly ``amplitude · P_k(t)`` plus Poisson noise.  The time of
bin i is ``(i − t0_bin) · bin_width``, the reducer's convention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.utils.constants import MUON_LIFETIME_US

T0_BIN = 10
BIN_WIDTH_US = 0.016
#: Detector order and the unit vector each one looks along.
_DIRECTIONS = np.array(
    [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], dtype=float
)
PROJECTIONS = [
    {"label": "P_x", "forward_group": 5, "backward_group": 6},
    {"label": "P_y", "forward_group": 3, "backward_group": 4},
    {"label": "P_z", "forward_group": 1, "backward_group": 2},
]


def bin_times(n_bins: int) -> NDArray[np.float64]:
    return (np.arange(n_bins) - T0_BIN) * BIN_WIDTH_US


def _histograms(
    polarisation: NDArray[np.float64],
    amplitudes: tuple[float, float, float],
    rate: float,
    rng: np.random.Generator,
) -> list[Histogram]:
    n_bins = polarisation.shape[1]
    t = np.clip(bin_times(n_bins), 0.0, None)
    scaled = np.asarray(amplitudes)[:, None] * polarisation
    histograms = []
    for direction in _DIRECTIONS:
        expected = rate * np.exp(-t / MUON_LIFETIME_US) * (1.0 + direction @ scaled)
        counts = rng.poisson(expected).astype(float) if rng is not None else expected
        histograms.append(Histogram(counts=counts, bin_width=BIN_WIDTH_US, t0_bin=T0_BIN))
    return histograms


def synthetic_vector_run(
    polarisations: Sequence[Callable[[NDArray[np.float64]], NDArray[np.float64]]],
    *,
    n_bins: int = 700,
    amplitudes: tuple[float, float, float] = (0.12, 0.12, 0.2),
    rate: float = 5.0e4,
    run_number: int = 900,
    field: float = 110.0,
    alphas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int | None = 7,
) -> Run:
    """A run with one period per polarisation callable ``P(t) -> (3, n)``.

    ``seed=None`` gives noiseless expected counts.
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    t = bin_times(n_bins)
    periods = [_histograms(p(t), amplitudes, rate, rng) for p in polarisations]
    grouping = {
        "groups": {gid: [gid] for gid in range(1, 7)},
        "group_names": {1: "F", 2: "B", 3: "T", 4: "Bo", 5: "L", 6: "R"},
        "projections": [dict(p) for p in PROJECTIONS],
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "alpha_x": alphas[0],
        "alpha_y": alphas[1],
        "alpha_z": alphas[2],
        "t0_bin": T0_BIN,
        "first_good_bin": T0_BIN + 2,
        "last_good_bin": n_bins - 1,
        "deadtime_correction": False,
    }
    metadata = {"run_number": run_number, "field": field, "period_count": len(periods)}
    if len(periods) > 1:
        reduced = [(t, np.zeros_like(t), np.ones_like(t)) for _ in periods]
        grouping |= {"period_histograms": periods, "period_reduced": reduced}
    return Run(
        run_number=run_number,
        histograms=periods[0],
        metadata=metadata,
        grouping=grouping,
        source_file="synthetic",
    )
