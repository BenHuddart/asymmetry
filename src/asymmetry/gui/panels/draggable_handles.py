"""Shared hit-testing for draggable x-handles on a matplotlib axis.

Both the time-spectrum fit-range span (:mod:`plot_panel`) and the ALC scan view
(:mod:`alc_panel`) let the user grab a vertical handle on the plot and drag it.
The grab test — device-pixel distance from a handle's data-x to the cursor,
within a tolerance — is the same for both and lives here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def nearest_handle(
    axis,
    handles: Iterable[tuple[float, T]],
    event_x_px: float,
    tolerance_px: float,
) -> T | None:
    """Return the key of the handle nearest *event_x_px*, within tolerance.

    *handles* is an iterable of ``(data_x, key)``; each ``data_x`` is projected
    to device pixels via ``axis.transData`` and compared to the cursor pixel
    ``event_x_px``. Returns the key of the closest handle within
    *tolerance_px* (inclusive), preferring the first on a tie, or ``None``.
    """
    nearest: T | None = None
    best: float | None = None
    for data_x, key in handles:
        px = axis.transData.transform((data_x, 0.0))[0]
        distance = abs(px - event_x_px)
        if distance <= tolerance_px and (best is None or distance < best):
            best = distance
            nearest = key
    return nearest
