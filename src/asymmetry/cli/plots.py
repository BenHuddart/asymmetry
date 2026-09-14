"""Headless PNG plots for ``--plot``.

The only module in this package (or in ``asymmetry`` at all, outside
``asymmetry.gui``) that imports matplotlib — through the object API
(:class:`matplotlib.figure.Figure` + ``FigureCanvasAgg``), never ``pyplot``
and never ``FigureCanvasQTAgg`` (the structural check reserves that
construction for ``asymmetry.gui.widgets.mpl_canvas``). Every function here
takes plain arrays and scalars and an output path, draws one fixed-size PNG
at 120 dpi with a tight layout, and returns the path it wrote — argument
parsing, model evaluation, and file naming stay in the command modules.

matplotlib is imported lazily *inside* each function (and inside
:func:`require_matplotlib`, via ``importlib.util.find_spec`` rather than a
real import) so that building the argument parser — and so ``asymmetry
--help`` — never pays its import cost, and a command that never plots never
needs it installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.cli._output import UserError

#: Shown to the user when ``--plot`` is requested without the ``agent`` extra.
MATPLOTLIB_HINT = "--plot needs matplotlib: pip install 'asymmetry[agent]'"

#: Figure geometry shared by every plot this module draws — fixed, not
#: content-dependent, so a PNG's size never varies with how much data it holds.
_FIGSIZE = (6.4, 4.8)
_FIGSIZE_TALL = (6.4, 6.4)
_DPI = 120

#: x-axis label for each scan coordinate a series may be ordered along.
_ORDER_AXIS_LABELS = {
    "temperature": "temperature / K",
    "field": "field / G",
    "run": "run",
}


def require_matplotlib() -> None:
    """Raise :class:`UserError` when matplotlib is not installed.

    Called by a command before it does any plotting work, so a missing
    optional dependency is one clear stderr line (exit 1) rather than an
    import traceback surfacing as an internal error (exit 2).
    """
    if importlib.util.find_spec("matplotlib") is None:
        raise UserError(MATPLOTLIB_HINT)


def _new_figure(figsize: tuple[float, float]):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize, dpi=_DPI)
    FigureCanvasAgg(figure)
    return figure


def _save(figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out_path, format="png")
    return out_path


def _order_axis_label(order_key: str) -> str:
    return _ORDER_AXIS_LABELS.get(order_key, order_key)


def plot_reduced(
    time: np.ndarray,
    asymmetry: np.ndarray,
    error: np.ndarray,
    *,
    run_number: int,
    temperature: float | None,
    field: float | None,
    title: str,
    out_path: str | Path,
) -> Path:
    """Asymmetry vs time for one reduced run: points with error bars, no line."""
    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
    axes.errorbar(time, asymmetry, yerr=error, fmt="o", ms=3, elinewidth=0.8, capsize=0, color="C0")
    axes.set_xlabel("time / µs")
    axes.set_ylabel("asymmetry / %")
    t_text = "-" if temperature is None else f"{temperature:g}"
    b_text = "-" if field is None else f"{field:g}"
    axes.set_title(f"Run {run_number} — {t_text} K, {b_text} G, {title}")
    return _save(figure, out_path)


def plot_fit(
    time: np.ndarray,
    asymmetry: np.ndarray,
    error: np.ndarray,
    *,
    model_function: Callable[..., np.ndarray],
    parameters: Mapping[str, float],
    t_min: float | None,
    t_max: float | None,
    run_number: int,
    expression: str,
    out_path: str | Path,
    n_curve_points: int = 400,
) -> Path:
    """Data, the fitted curve, and a residuals panel beneath.

    The curve is *model_function* (``recipe.model().function``) evaluated at
    *parameters* (the fitted values from a ``fit_result_summary``) on a dense
    time axis spanning the fitted window — *t_min*/*t_max*, or the data's own
    range where either is ``None``. Residuals are ``(data - model) / error``
    over that same window; the raw data is drawn over its full range.
    """
    time = np.asarray(time, dtype=np.float64)
    asymmetry = np.asarray(asymmetry, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)

    window_min = float(time.min()) if t_min is None else float(t_min)
    window_max = float(time.max()) if t_max is None else float(t_max)
    mask = (time >= window_min) & (time <= window_max)

    dense_time = np.linspace(window_min, window_max, n_curve_points)
    curve = np.asarray(model_function(dense_time, **parameters), dtype=np.float64)
    model_at_data = np.asarray(model_function(time[mask], **parameters), dtype=np.float64)
    residuals = (asymmetry[mask] - model_at_data) / error[mask]

    figure = _new_figure(_FIGSIZE_TALL)
    data_axes = figure.add_subplot(211)
    data_axes.errorbar(
        time,
        asymmetry,
        yerr=error,
        fmt="o",
        ms=3,
        elinewidth=0.8,
        capsize=0,
        color="0.4",
        label="data",
    )
    data_axes.plot(dense_time, curve, "-", color="C1", lw=1.5, label="model")
    data_axes.set_ylabel("asymmetry / %")
    data_axes.set_title(f"Run {run_number} — {expression}")
    data_axes.legend(loc="best", fontsize="small")

    residual_axes = figure.add_subplot(212, sharex=data_axes)
    residual_axes.axhline(0.0, color="0.6", lw=0.8)
    residual_axes.plot(time[mask], residuals, "o", ms=3, color="0.2")
    residual_axes.set_xlabel("time / µs")
    residual_axes.set_ylabel("(data − model) / error")

    return _save(figure, out_path)


def plot_trend(
    rows: Sequence[Mapping[str, Any]],
    *,
    param_name: str,
    order_key: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Value vs scan coordinate for one free parameter, ordered, with error bars.

    *rows* is a :class:`~asymmetry.core.workflow.series.TrendTable`'s own
    ``rows`` — already ordered along the scan — so this reads ``"x"``,
    *param_name*, ``f"{param_name}_err"`` and ``"flags"`` straight off each
    row rather than the caller re-deriving parallel arrays. A row whose
    ``"flags"`` is non-empty is drawn hollow and grouped into a "flagged"
    legend entry instead of being dropped.
    """
    from asymmetry.gui.utils.formatting import format_param_label

    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row.get(param_name) for row in rows], dtype=np.float64)
    y_err = np.asarray([row.get(f"{param_name}_err") for row in rows], dtype=np.float64)
    flagged = np.asarray([bool(row.get("flags")) for row in rows], dtype=bool)

    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
    clean = ~flagged
    if np.any(clean):
        axes.errorbar(
            x[clean],
            y[clean],
            yerr=y_err[clean],
            fmt="o",
            ms=4,
            elinewidth=0.8,
            capsize=0,
            color="C0",
            label=param_name,
        )
    if np.any(flagged):
        axes.errorbar(
            x[flagged],
            y[flagged],
            yerr=y_err[flagged],
            fmt="o",
            ms=4,
            elinewidth=0.8,
            capsize=0,
            markerfacecolor="none",
            markeredgecolor="0.5",
            ecolor="0.5",
            label="flagged",
        )
    axes.set_xlabel(_order_axis_label(order_key))
    axes.set_ylabel(format_param_label(param_name))
    axes.set_title(param_name if title is None else title)
    axes.legend(loc="best", fontsize="small")

    return _save(figure, out_path)


__all__ = [
    "MATPLOTLIB_HINT",
    "plot_fit",
    "plot_reduced",
    "plot_trend",
    "require_matplotlib",
]
