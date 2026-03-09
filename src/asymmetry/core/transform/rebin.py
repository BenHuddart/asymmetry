"""Rebinning utilities for time-domain data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rebin(
    time: NDArray[np.float64],
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    factor: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Rebin data by combining *factor* consecutive bins.

    Parameters
    ----------
    time, values, errors
        Equal-length arrays of the original data.
    factor
        Number of bins to merge into one.

    Returns
    -------
    (time_rebinned, values_rebinned, errors_rebinned)
    """
    if factor < 1:
        raise ValueError("Rebinning factor must be >= 1")
    if factor == 1:
        return time.copy(), values.copy(), errors.copy()

    n = len(time)
    n_new = n // factor
    trimmed = n_new * factor

    t = time[:trimmed].reshape(n_new, factor).mean(axis=1)
    v = values[:trimmed].reshape(n_new, factor).mean(axis=1)
    e = np.sqrt((errors[:trimmed].reshape(n_new, factor) ** 2).sum(axis=1)) / factor

    return t, v, e
