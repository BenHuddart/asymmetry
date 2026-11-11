"""Sign-run and short-lag diagnostics for fit residuals."""

import math

import numpy as np
from numpy.typing import NDArray


def runs_test_z_score(values: NDArray[np.float64]) -> float:
    """Return the normalised sign-run statistic after numerical-zero filtering."""
    if values.size < 3:
        return 0.0
    threshold = max(1e-6, 1e-3 * float(np.max(np.abs(values))))
    filtered = np.asarray(values, dtype=float).copy()
    filtered[np.abs(filtered) <= threshold] = 0.0
    if np.max(np.abs(filtered)) <= threshold:
        return 0.0
    signs = np.sign(filtered)
    signs = signs[signs != 0]
    if signs.size < 3:
        return 0.0
    positives = int(np.count_nonzero(signs > 0))
    negatives = int(np.count_nonzero(signs < 0))
    if positives == 0 or negatives == 0:
        return 0.0
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    total = positives + negatives
    mean = 1.0 + 2.0 * positives * negatives / total
    variance = (
        2.0
        * positives
        * negatives
        * (2.0 * positives * negatives - positives - negatives)
        / max(total * total * (total - 1), 1)
    )
    if variance <= 0.0:
        return 0.0
    return float((runs - mean) / math.sqrt(variance))


def max_abs_autocorrelation(values: NDArray[np.float64]) -> float:
    """Return the largest absolute residual correlation over the first four lags."""
    if values.size < 4:
        return 0.0
    threshold = max(1e-6, 1e-3 * float(np.max(np.abs(values))))
    filtered = np.asarray(values, dtype=float).copy()
    filtered[np.abs(filtered) <= threshold] = 0.0
    if np.max(np.abs(filtered)) <= threshold:
        return 0.0
    centered = filtered - float(np.mean(filtered))
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return 0.0
    max_lag = min(4, centered.size // 2)
    if max_lag <= 0:
        return 0.0
    correlations = []
    for lag in range(1, max_lag + 1):
        corr = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        correlations.append(abs(corr))
    return float(max(correlations, default=0.0))
