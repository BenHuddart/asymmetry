"""Compatibility wrapper for grouped MaxEnt spectral reconstruction.

Asymmetry's MaxEnt implementation fits a forward model of grouped detector
*counts*, with count-domain errors, phases, amplitudes and backgrounds. An
asymmetry curve does not retain those separate group signals and cannot be used
as input to this algorithm. Use :func:`asymmetry.core.fourier.fft.fft_arrays`
or a frequency-domain fit for a reduced curve. Callers holding counts but no
:class:`~asymmetry.core.data.dataset.Run` want
:func:`~asymmetry.core.maxent.engine.maxent_from_counts`, re-exported here.
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

    Delegates to :mod:`asymmetry.core.maxent`, which reconstructs a shared
    spectrum from grouped raw counts. Callers passing a
    :class:`~asymmetry.core.data.dataset.MuonDataset` must provide ``dataset.run``.

    This implementation requires the separate detector-group signals and their
    count-domain errors. Use :func:`asymmetry.core.fourier.fft.fft_arrays` for a
    curve, or :func:`~asymmetry.core.maxent.engine.maxent_from_counts` for counts
    without a run.
    """
    run = dataset if isinstance(dataset, Run) else dataset.run
    if run is None:
        raise ValueError("MaxEnt requires a Run with raw detector histograms.")
    config = MaxEntConfig(n_spectrum_points=int(n_freq), f_max_mhz=f_max)
    result: MaxEntResult = grouped_maxent(run, config)
    return result.frequencies_mhz, result.spectrum


__all__ = ["maxent", "maxent_from_counts"]
