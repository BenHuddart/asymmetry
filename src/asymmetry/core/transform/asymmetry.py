"""Compute the μSR asymmetry from grouped histograms.

The standard asymmetry is defined as

    A(t) = [N_F(t) − α N_B(t)] / [N_F(t) + α N_B(t)]

where N_F and N_B are the forward and backward group counts and α is
the balance parameter.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_asymmetry(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    alpha: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry and its statistical error.

    Parameters
    ----------
    forward, backward
        Counts in the forward and backward detector groups (same length).
    alpha
        Balance parameter (α).

    Returns
    -------
    asymmetry, error
        Arrays of the same length as the inputs.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)

    denominator = f + alpha * b
    # Avoid division by zero
    safe = denominator > 0
    asym = np.zeros_like(f)
    err = np.zeros_like(f)

    asym[safe] = (f[safe] - alpha * b[safe]) / denominator[safe]

    # Error propagation (Poisson statistics on counts)
    err[safe] = (
        2.0
        * alpha
        * np.sqrt(f[safe] * b[safe] ** 2 + b[safe] * f[safe] ** 2)
        / denominator[safe] ** 2
    )

    return asym, err
