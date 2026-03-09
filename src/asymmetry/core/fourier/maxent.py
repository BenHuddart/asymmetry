"""Maximum-entropy spectral reconstruction.

This is a placeholder — the full MaxEnt algorithm will be implemented in a
future milestone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset


def maxent(
    dataset: MuonDataset,
    n_freq: int = 512,
    f_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Maximum-entropy frequency spectrum (stub).

    Parameters
    ----------
    dataset
        Time-domain asymmetry data.
    n_freq
        Number of frequency points.
    f_max
        Maximum frequency (MHz).

    Returns
    -------
    frequencies, spectrum
    """
    raise NotImplementedError(
        "Maximum-entropy reconstruction is planned for a future milestone."
    )
