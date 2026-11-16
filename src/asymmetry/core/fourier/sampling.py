"""One sampling geometry and tolerance authority for uniform Fourier views."""

import numpy as np
from numpy.typing import NDArray

_SAMPLING_GRID_TOLERANCE = 1e-6


def is_uniform_sampling(time: NDArray[np.float64]) -> bool:
    """Whether a grid satisfies the FFT's existing sampling tolerance."""
    if time.size < 2 or not np.all(np.isfinite(time)):
        return False
    dt = float(np.mean(np.diff(time)))
    return bool(
        np.isfinite(dt)
        and dt > 0
        and np.max(_sampling_grid_residual(time, dt)) <= _SAMPLING_GRID_TOLERANCE
    )


def _sampling_grid_residual(time: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    return np.abs((time - time[0]) - np.arange(time.size) * dt) / dt


def _uniform_sample_interval(
    time: NDArray[np.float64], *, caller: str, t_min: float | None, t_max: float | None
) -> float:
    """Require a finite, increasing uniform grid in the retained transform window."""
    if time.size < 2:
        raise ValueError(
            f"{caller}: the transform window t_min={t_min!r}, t_max={t_max!r} holds "
            f"{time.size} time sample(s); the FFT needs at least 2 to fix its sampling interval."
        )
    if not np.all(np.isfinite(time)):
        raise ValueError(
            f"{caller}: time must contain only finite samples in the transform window."
        )
    dt = float(np.mean(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(
            f"{caller}: time must increase; the window runs from {time[0]:g} µs to {time[-1]:g} µs."
        )
    # At Nyquist a displacement δ changes phase by πδ/dt, bounded here by π·1e-6.
    # Positional residuals prevent interval errors accumulating across a long record.
    residual = _sampling_grid_residual(time, dt)
    index = int(np.argmax(residual))
    if residual[index] > _SAMPLING_GRID_TOLERANCE:
        raise ValueError(
            f"{caller}: time is not uniformly sampled: sample {index} at {time[index]:g} µs "
            f"lies {residual[index]:.3g} intervals from the uniform grid {time[0]:g} + k·{dt:g} µs "
            f"(tolerance {_SAMPLING_GRID_TOLERANCE:g}). The FFT assigns frequencies only to a "
            "uniform grid; use Fixed binning, or resample explicitly before transforming."
        )
    return dt
