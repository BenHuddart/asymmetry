"""Shared point-count bounding for lightweight, display-only preview plots.

Several small "preview" plots draw a matplotlib ``errorbar`` over an entire
reduced/raw asymmetry curve synchronously on the GUI thread: the grouping
editor's live preview (``gui/windows/grouping/preview_pane.py``) and the fit
wizard's fingerprint plot (``gui/windows/fit_wizard_window.py``). ``errorbar``
builds a ``LineCollection`` for the error bars and computing its data limits
(``get_path_collection_extents``) is O(points) with real per-point overhead —
on a run with hundreds of thousands to millions of bins this stalls the GUI
for seconds. These preview surfaces are visual fingerprints, not precision
analysis surfaces, so a uniform stride down to a fixed point budget is
visually adequate and keeps the draw bounded regardless of input size.

:func:`decimate_for_preview` is the one shared implementation — do not re-roll
a second stride/bucketing helper for a new preview plot without a concrete
reason the uniform-stride contract doesn't fit (see
``plot_panel.py::_decimated_plot_indices`` for the interactive-view sibling,
which additionally does min-max bucketing for frequency-domain spectra where
a stride could drop a narrow peak — not needed here since these preview plots
are small and re-drawn from scratch each time, not zoom/pan interactive).
"""

from __future__ import annotations

import numpy as np


def decimate_for_preview(
    time: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly stride three aligned arrays down to at most ``max_points``.

    All three arrays share one stride so points stay aligned. Curves already
    at or below the cap pass through unchanged. Advisory-only sampling: no
    min/max envelope, just a plain stride — cheap and visually adequate for
    a small preview plot.
    """
    n = int(time.size)
    if n <= max_points or max_points <= 0:
        return time, y, yerr
    step = (n + max_points - 1) // max_points  # ceil(n / max_points)
    return time[::step], y[::step], yerr[::step]
