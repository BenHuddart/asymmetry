"""Selection of bin centres on a microsecond time axis."""

import numpy as np
from numpy.typing import NDArray

# A 1e-12 µs margin covers arithmetic roundoff in bin times and is more than
# seven orders of magnitude smaller than the 2.44e-5 µs HAL .mdu bin width.
TIME_WINDOW_ATOL_US = 1e-12


def time_window_mask(
    time: NDArray[np.float64],
    t_min: float | None,
    t_max: float | None,
    *,
    support_start: float | None = None,
) -> NDArray[np.bool_]:
    """Select centres within inclusive bounds in µs, allowing for roundoff.

    A model's exact support can restrict the added margin while leaving the
    literal window intact for model validation.
    """
    if t_min is not None and t_max is not None and t_min > t_max:
        return np.zeros(time.shape, dtype=bool)
    mask = np.ones(time.shape, dtype=bool)
    eligible = True if support_start is None else time >= support_start
    with np.errstate(invalid="ignore", over="ignore"):
        if t_min is not None:
            mask &= (time >= t_min) | (eligible & (t_min - time <= TIME_WINDOW_ATOL_US))
        if t_max is not None:
            mask &= (time <= t_max) | (eligible & (time - t_max <= TIME_WINDOW_ATOL_US))
    return mask
