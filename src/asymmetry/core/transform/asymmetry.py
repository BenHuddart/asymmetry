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

    numerator = f - alpha * b
    denominator = f + alpha * b
    # Mantid-compatible handling: only divide on non-zero denominator.
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    err = np.zeros_like(f)

    asym[safe] = numerator[safe] / denominator[safe]

    # Match Mantid AsymmetryCalc error model:
    # error = sqrt((F + alpha^2 B) * (1 + (num/den)^2)) / den
    # with a default error of 1.0 where denominator is zero.
    err[~safe] = 1.0
    if np.any(safe):
        den_safe = denominator[safe]
        num_safe = numerator[safe]
        q1 = f[safe] + (alpha * alpha) * b[safe]
        q2 = 1.0 + (num_safe * num_safe) / (den_safe * den_safe)
        err[safe] = np.sqrt(np.maximum(q1 * q2, 0.0)) / np.abs(den_safe)

    return asym, err


def compute_asymmetry_with_count_errors(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    forward_error: NDArray[np.float64],
    backward_error: NDArray[np.float64],
    alpha: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate asymmetry from counts with supplied count uncertainties.

    This is used for musrfit-style background-corrected histograms, where the
    count uncertainties are formed before or during background subtraction.
    The asymmetry convention remains Asymmetry's convention, with ``alpha``
    multiplying the backward group.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    ef = np.asarray(forward_error, dtype=np.float64)
    eb = np.asarray(backward_error, dtype=np.float64)

    n = min(f.size, b.size, ef.size, eb.size)
    f = f[:n]
    b = b[:n]
    ef = ef[:n]
    eb = eb[:n]

    numerator = f - alpha * b
    denominator = f + alpha * b
    safe = denominator != 0.0
    asym = np.zeros_like(f)
    err = np.ones_like(f)

    asym[safe] = numerator[safe] / denominator[safe]
    if np.any(safe):
        den_safe = denominator[safe]
        variance_term = (b[safe] * ef[safe]) ** 2 + (f[safe] * eb[safe]) ** 2
        err[safe] = (
            2.0
            * abs(float(alpha))
            * np.sqrt(np.maximum(variance_term, 0.0))
            / (den_safe * den_safe)
        )

    return asym, err


def estimate_alpha(
    forward: NDArray[np.float64],
    backward: NDArray[np.float64],
    *,
    first_good_bin: int | None = None,
    last_good_bin: int | None = None,
) -> float:
    r"""Estimate the detector-balance parameter ``alpha`` from grouped counts.

    This follows the same approach used by Mantid's ``AlphaCalc`` algorithm:

    .. math::

        \alpha = \frac{\sum_i F_i}{\sum_i B_i}

    where :math:`F_i` and :math:`B_i` are forward and backward grouped counts
    integrated over the selected good-bin window.

    Parameters
    ----------
    forward, backward
        Forward and backward grouped count arrays.
    first_good_bin, last_good_bin
        Optional inclusive bin range for integration. If omitted, the full
        overlap of the two arrays is used.

    Returns
    -------
    float
        Estimated alpha value. Returns ``1.0`` when the backward integral is
        not positive or when no valid bins are available.
    """
    f = np.asarray(forward, dtype=np.float64)
    b = np.asarray(backward, dtype=np.float64)
    n = min(len(f), len(b))
    if n <= 0:
        return 1.0

    lo = 0 if first_good_bin is None else max(0, int(first_good_bin))
    hi = n - 1 if last_good_bin is None else min(n - 1, int(last_good_bin))
    if lo > hi:
        return 1.0

    fs = float(np.sum(f[lo : hi + 1]))
    bs = float(np.sum(b[lo : hi + 1]))
    if bs <= 0.0:
        return 1.0
    return fs / bs
