"""Compatibility wrapper for grouped MaxEnt spectral reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.maxent import MaxEntConfig, MaxEntResult
from asymmetry.core.maxent import maxent as grouped_maxent


def maxent(
    dataset: MuonDataset | Run,
    n_freq: int = 512,
    f_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a MaxEnt spectrum as ``(frequencies, values)``.

    This preserves the historical Fourier-module import path while delegating
    to :mod:`asymmetry.core.maxent`.  MaxEnt is a grouped raw-count algorithm;
    callers passing a :class:`MuonDataset` must provide one with ``dataset.run``.
    """
    run = dataset if isinstance(dataset, Run) else dataset.run
    if run is None:
        raise ValueError("MaxEnt requires a Run with raw detector histograms.")
    config = MaxEntConfig(n_spectrum_points=int(n_freq), f_max_mhz=f_max)
    result: MaxEntResult = grouped_maxent(run, config)
    return result.frequencies_mhz, result.spectrum
