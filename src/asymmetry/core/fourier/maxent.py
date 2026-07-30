"""Compatibility wrapper for grouped MaxEnt spectral reconstruction.

**There is no asymmetry-domain MaxEnt, and there cannot be one.** MaxEnt fits a
forward model of the raw detector *counts*, weighting every bin by its own
Poisson error and carrying per-group phases, amplitudes and backgrounds; an
asymmetry curve has already combined detectors, divided out the normalisations
and destroyed exactly those statistics. The frequency-domain estimator for an
asymmetry curve is :func:`asymmetry.core.fourier.fft.fft_arrays` (or a
frequency-domain fit). Callers holding counts but no
:class:`~asymmetry.core.data.dataset.Run` want
:func:`~asymmetry.core.maxent.maxent_from_counts`, re-exported here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.maxent import MaxEntConfig, MaxEntResult, maxent_from_counts
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

    There is no asymmetry-domain MaxEnt (see the module docstring): an
    asymmetry curve has destroyed the per-group Poisson statistics the
    algorithm fits. Use :func:`asymmetry.core.fourier.fft.fft_arrays` for a
    curve, or :func:`~asymmetry.core.maxent.maxent_from_counts` for counts
    without a run.
    """
    run = dataset if isinstance(dataset, Run) else dataset.run
    if run is None:
        raise ValueError("MaxEnt requires a Run with raw detector histograms.")
    config = MaxEntConfig(n_spectrum_points=int(n_freq), f_max_mhz=f_max)
    result: MaxEntResult = grouped_maxent(run, config)
    return result.frequencies_mhz, result.spectrum


__all__ = ["maxent", "maxent_from_counts"]
