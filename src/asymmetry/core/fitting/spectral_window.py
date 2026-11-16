"""Shared informative windows and centering for internal spectral diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import MuonDataset


def effective_analysis_window(
    time: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    factor: float = 5.0,
    min_points: int = 32,
) -> int:
    """Return the exclusive end index of the noise-truncated analysis window.

    μSR error bars grow roughly exponentially with time (dying-muon statistics)
    and are capped at 100 %; a full-window FFT is then dominated by the late-time
    pure-noise tail, whitening the spectrum so even clean lines vanish.  This
    truncates the record at the first point whose per-point error exceeds
    ``factor`` times the early-time error — i.e. where the per-point information
    ``1/σ²`` has fallen below ``1/factor²`` of its early value — keeping the
    statistically informative early window and shedding the noise tail.

    The criterion is strictly **per-point**, so flat-error records (constant σ,
    the synthetic/test convention) are never truncated: the returned index is the
    full length and the reported resolution is unchanged.  Returns the full
    length for short records (``≤ min_points``) or degenerate error arrays.
    """
    err = np.asarray(error, dtype=float)
    n = err.size
    if n <= int(min_points) or int(time.size) != n:
        return n
    finite = np.isfinite(err) & (err > 0.0)
    if not np.any(finite):
        return n
    early = max(5, n // 20)
    sigma_early = float(np.median(err[:early][finite[:early]])) if np.any(finite[:early]) else 0.0
    if not np.isfinite(sigma_early) or sigma_early <= 0.0:
        return n
    # First point that is finite and exceeds the SNR-truncation threshold.
    exceeds = finite & (err > float(factor) * sigma_early)
    if not np.any(exceeds):
        return n
    end = int(np.argmax(exceeds))
    return max(end, int(min_points))


def _centered_signal(
    dataset: MuonDataset, detrend_curve: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], bool]:
    """Return ``(signal, detrended)`` — the residual to transform.

    With ``detrend_curve`` the residual is ``asymmetry − detrend_curve``;
    otherwise it is ``asymmetry −`` tail estimate (mean of the last ~20 %),
    mirroring the fingerprint centering in ``fit_wizard``.
    """
    y = np.asarray(dataset.asymmetry, dtype=float)
    if detrend_curve is not None:
        curve = np.asarray(detrend_curve, dtype=float)
        if curve.shape != y.shape:
            raise ValueError(
                f"detrend_curve shape {curve.shape} does not match asymmetry {y.shape}"
            )
        return y - curve, True
    n = y.size
    late = min(n, max(5, n // 5))  # last ~20 %
    tail = float(np.mean(y[-late:])) if n else 0.0
    return y - tail, False
