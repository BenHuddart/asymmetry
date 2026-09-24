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

Framing
-------

μSR errors grow with time (dying-muon statistics) and are capped at 100 %; an
unframed plot's y-axis is then set by the noise tail, squashing the
informative early region into a thin band and, for a fit, flattening the
model curve to a line. :func:`plot_reduced` and :func:`plot_fit` frame the x
range on the same SNR-truncated informative window the fit wizard's own
fingerprint and peak detector use
(:func:`~asymmetry.core.fitting.peak_detection.effective_analysis_window`,
via :func:`frame_for_record`), draw at most a few hundred points per panel
(bunching a longer record for *display only* — the fit itself is untouched),
and set the y range from what actually ends up on screen. :func:`plot_trend`
frames its y range on the unflagged points, so one wildly-flagged run cannot
squash every other point onto the axis edge; a flagged point outside that
range is drawn clamped, with a distinct marker, rather than silently moving
the frame or being dropped.

Every frame is taken over the *finite* points only. A run that failed to fit
carries no value and no error, so a range over those would be NaN and
``set_ylim`` would raise — exactly when every run in a series is flagged. With
nothing finite to frame on, the panel is drawn with no y range and a note
saying so, and the PNG is still written.
"""

from __future__ import annotations

import importlib.util
import math
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
    "sample_temperature_logged": "logged sample temperature / K",
    "field": "field / G",
    "run": "run",
}

#: A panel drawn beyond this many points is bunched for display (see
#: :func:`_bunch_factor`) — the fit is unaffected, only what is drawn.
_MAX_DISPLAY_POINTS = 400

#: Margin (as a fraction of the framed span) added around a data/fit y range.
_DATA_Y_MARGIN = 0.05

#: Margin added around a trend's y range, framed on its unflagged points.
_TREND_Y_MARGIN = 0.10

#: Drawn on a panel that has no finite point to frame on — every run in the
#: series failed — in place of a y range.
_NO_FINITE_VALUES_NOTE = "no finite values to frame"


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


def frame_for_record(
    time: np.ndarray,
    error: np.ndarray,
    *,
    t_min: float | None,
    t_max: float | None,
) -> tuple[float, float]:
    """The ``(window_min, window_max)`` x range a reduced/fit plot draws over.

    ``window_min`` is *t_min*, or the record's own start; ``window_max`` is
    the smaller of *t_max* (or the record's own end) and the end of the
    SNR-truncated informative window
    (:func:`~asymmetry.core.fitting.peak_detection.effective_analysis_window`
    — the same one the fit wizard's own fingerprint and peak detector use):
    late-time μSR points routinely carry ±100 % errors, and an unframed plot's
    y-axis follows them rather than the signal.

    ``effective_analysis_window`` returns an *exclusive end index* into
    *time*/*error* (the full length when nothing needs truncating); this
    reports the corresponding time value — the last point still inside the
    informative window.
    """
    from asymmetry.core.fitting.peak_detection import effective_analysis_window

    time = np.asarray(time, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)

    window_min = float(time.min()) if t_min is None else float(t_min)
    requested_max = float(time.max()) if t_max is None else float(t_max)

    end_index = int(effective_analysis_window(time, error))
    end_index = min(max(end_index, 1), time.size)
    effective_end = float(time[end_index - 1])

    return window_min, min(requested_max, effective_end)


def _bunch_factor(n_points: int, *, max_points: int = _MAX_DISPLAY_POINTS) -> int:
    """The smallest integer rebin factor that brings *n_points* to at most *max_points*."""
    if n_points <= max_points:
        return 1
    return math.ceil(n_points / max_points)


def _drawn_record(
    time: np.ndarray,
    asymmetry: np.ndarray,
    error: np.ndarray,
    *,
    window_min: float,
    window_max: float,
):
    """The framed record a panel actually draws: windowed, then bunched for display.

    Returns ``(drawn_dataset, bunch_factor)``. The fit itself never sees
    *drawn_dataset* — only :func:`plot_fit`'s own residual/curve drawing does.
    """
    from asymmetry.core.data.dataset import MuonDataset

    full = MuonDataset(
        time=np.asarray(time, dtype=np.float64),
        asymmetry=np.asarray(asymmetry, dtype=np.float64),
        error=np.asarray(error, dtype=np.float64),
    )
    windowed = full.time_range(window_min, window_max)
    factor = _bunch_factor(windowed.n_points)
    drawn = windowed if factor <= 1 else windowed.rebin(factor)
    return drawn, factor


def _margin_range(
    values: np.ndarray, errors: np.ndarray, *, fraction: float
) -> tuple[float, float] | None:
    """``(min(value - error), max(value + error))`` over the finite points, plus a margin.

    Only finite *values* are framed on, and only a finite error widens one: a
    run that failed to fit carries no value and no error, and a range taken
    over those would hand ``set_ylim`` NaN limits — which raises, exactly when
    every run in a series is flagged. A point whose value is finite but whose
    error is missing frames as the bare value. Returns ``None`` when no finite
    value is left to frame at all, which is the caller's cue to draw the axes
    with no y range rather than an impossible one.

    A degenerate (zero-span) range — e.g. a single point, or a parameter held
    fixed across a whole trend — gets a margin scaled off the value itself
    (or, at zero, a fixed absolute margin) rather than collapsing to a
    zero-height axis.
    """
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    framed = values[finite]
    widths = np.where(np.isfinite(errors[finite]), errors[finite], 0.0)
    lo = float(np.min(framed - widths))
    hi = float(np.max(framed + widths))
    span = hi - lo
    if span > 0:
        margin = span * fraction
    elif lo != 0.0:
        margin = abs(lo) * fraction
    else:
        margin = 1.0
    return lo - margin, hi + margin


def _format_time(value: float) -> str:
    return f"{value:.4g}"


def _frame_note(
    window_min: float, window_max: float, requested_max: float, factor: int
) -> str | None:
    """A small axes note describing what was cropped/bunched for display, if anything."""
    parts: list[str] = []
    if window_max < requested_max - 1e-9:
        parts.append(
            f"showing {_format_time(window_min)}–{_format_time(window_max)} µs of "
            f"{_format_time(window_min)}–{_format_time(requested_max)}"
        )
    if factor > 1:
        parts.append(f"bunched ×{factor}")
    return "; ".join(parts) if parts else None


def _set_y_range(axes, y_range: tuple[float, float] | None) -> None:
    """Apply a :func:`_margin_range` result, noting on the axes when there is none."""
    if y_range is None:
        _annotate_frame(axes, _NO_FINITE_VALUES_NOTE)
        return
    axes.set_ylim(*y_range)


def _annotate_frame(axes, note: str | None) -> None:
    if note is not None:
        axes.text(
            0.99,
            0.02,
            note,
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            fontsize="x-small",
            color="0.4",
        )


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
    """Asymmetry vs time for one reduced run: points with error bars, no line.

    Framed on the SNR-truncated informative window (see :func:`frame_for_record`)
    and bunched for display when that window still holds many points — see the
    module docstring.
    """
    time = np.asarray(time, dtype=np.float64)
    asymmetry = np.asarray(asymmetry, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)

    window_min, window_max = frame_for_record(time, error, t_min=None, t_max=None)
    drawn, factor = _drawn_record(
        time, asymmetry, error, window_min=window_min, window_max=window_max
    )

    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
    axes.errorbar(
        drawn.time,
        drawn.asymmetry,
        yerr=drawn.error,
        fmt="o",
        ms=3,
        elinewidth=0.8,
        capsize=0,
        color="C0",
    )
    axes.set_xlim(window_min, window_max)
    _set_y_range(axes, _margin_range(drawn.asymmetry, drawn.error, fraction=_DATA_Y_MARGIN))
    axes.set_xlabel("time / µs")
    axes.set_ylabel("asymmetry / %")
    t_text = "-" if temperature is None else f"{temperature:g}"
    b_text = "-" if field is None else f"{field:g}"
    axes.set_title(f"Run {run_number} — {t_text} K, {b_text} G, {title}")
    _annotate_frame(axes, _frame_note(window_min, window_max, float(time.max()), factor))
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
    *parameters* (the fitted values from a ``fit_result_summary``). Both
    panels are framed on the SNR-truncated informative window inside
    *t_min*/*t_max* (see :func:`frame_for_record`) — the curve is evaluated
    densely across that same range, and the residuals ``(data - model) /
    error`` are computed on the (possibly display-bunched, see the module
    docstring) drawn points rather than the full record.
    """
    time = np.asarray(time, dtype=np.float64)
    asymmetry = np.asarray(asymmetry, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)

    window_min, window_max = frame_for_record(time, error, t_min=t_min, t_max=t_max)
    requested_max = float(time.max()) if t_max is None else float(t_max)
    drawn, factor = _drawn_record(
        time, asymmetry, error, window_min=window_min, window_max=window_max
    )

    dense_time = np.linspace(window_min, window_max, n_curve_points)
    curve = np.asarray(model_function(dense_time, **parameters), dtype=np.float64)
    model_at_drawn = np.asarray(model_function(drawn.time, **parameters), dtype=np.float64)
    residuals = (drawn.asymmetry - model_at_drawn) / drawn.error

    figure = _new_figure(_FIGSIZE_TALL)
    data_axes = figure.add_subplot(211)
    data_axes.errorbar(
        drawn.time,
        drawn.asymmetry,
        yerr=drawn.error,
        fmt="o",
        ms=3,
        elinewidth=0.8,
        capsize=0,
        color="0.4",
        label="data",
    )
    data_axes.plot(dense_time, curve, "-", color="C1", lw=1.5, label="model")
    data_axes.set_xlim(window_min, window_max)
    _set_y_range(data_axes, _margin_range(drawn.asymmetry, drawn.error, fraction=_DATA_Y_MARGIN))
    data_axes.set_ylabel("asymmetry / %")
    data_axes.set_title(f"Run {run_number} — {expression}")
    data_axes.legend(loc="best", fontsize="small")
    _annotate_frame(data_axes, _frame_note(window_min, window_max, requested_max, factor))

    residual_axes = figure.add_subplot(212, sharex=data_axes)
    residual_axes.axhline(0.0, color="0.6", lw=0.8)
    residual_axes.plot(drawn.time, residuals, "o", ms=3, color="0.2")
    residual_axes.set_xlabel("time / µs")
    residual_axes.set_ylabel("(data − model) / error")

    return _save(figure, out_path)


def _trend_frame(
    y: np.ndarray, y_err: np.ndarray, flagged: np.ndarray
) -> tuple[float | None, float | None, np.ndarray]:
    """``(y_lo, y_hi, outside)`` for :func:`plot_trend` — see its docstring.

    Framed on the unflagged points when any of them carries a finite value,
    else on all of them (so a wholly-flagged trend still gets a usable frame);
    *outside* marks the flagged points whose value falls outside that frame.
    ``(None, None, nothing outside)`` when no point carries a finite value at
    all — a series in which every run failed — so the caller can draw the axes
    without a y range instead of an exception.
    """
    clean = ~flagged
    framed = None
    if np.any(clean):
        framed = _margin_range(y[clean], y_err[clean], fraction=_TREND_Y_MARGIN)
    if framed is None:
        framed = _margin_range(y, y_err, fraction=_TREND_Y_MARGIN)
    if framed is None:
        return None, None, np.zeros_like(flagged)
    y_lo, y_hi = framed
    outside = flagged & np.isfinite(y) & ((y < y_lo) | (y > y_hi))
    return y_lo, y_hi, outside


def plot_trend(
    rows: Sequence[Mapping[str, Any]],
    *,
    param_name: str,
    order_key: str,
    out_path: str | Path,
    title: str | None = None,
    model: tuple[Callable[..., np.ndarray], Mapping[str, float], tuple[float, float]] | None = None,
) -> Path:
    """Value vs scan coordinate for one free parameter, ordered, with error bars.

    *model* — ``(function, parameters, (x_lo, x_hi))`` — overlays a trend fit,
    drawn across the x range of the points it was fitted to.

    *rows* is a :class:`~asymmetry.core.workflow.series.TrendTable`'s own
    ``rows`` — already ordered along the scan — so this reads ``"x"``,
    *param_name*, ``f"{param_name}_err"`` and ``"flags"`` straight off each
    row rather than the caller re-deriving parallel arrays.

    The y range is framed on the *unflagged* points (plus a margin), so a
    single wildly-flagged run cannot squash every other point onto the axis
    edge — see the module docstring. A flagged point that still falls inside
    that range is drawn hollow, as before; one that falls outside it is
    clamped to the nearest edge and drawn with a distinct (triangular)
    marker, in its own legend entry naming how many were clamped, rather than
    being dropped or moving the frame. When every point is flagged, the frame
    covers all of them, so nothing is clamped; when no point carries a finite
    value at all — every run in the series failed — the axes are drawn with no
    y range and a note saying so, and the PNG is still written.
    """
    from asymmetry.gui.utils.formatting import format_param_label

    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row.get(param_name) for row in rows], dtype=np.float64)
    y_err = np.asarray([row.get(f"{param_name}_err") for row in rows], dtype=np.float64)
    flagged = np.asarray([bool(row.get("flags")) for row in rows], dtype=bool)
    clean = ~flagged

    y_lo, y_hi, outside = _trend_frame(y, y_err, flagged)
    inside_flagged = flagged & ~outside

    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
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
    if np.any(inside_flagged):
        axes.errorbar(
            x[inside_flagged],
            y[inside_flagged],
            yerr=y_err[inside_flagged],
            fmt="o",
            ms=4,
            elinewidth=0.8,
            capsize=0,
            markerfacecolor="none",
            markeredgecolor="0.5",
            ecolor="0.5",
            label="flagged",
        )
    n_outside = int(np.count_nonzero(outside))
    if n_outside:
        y_clamped = np.clip(y[outside], y_lo, y_hi)
        axes.scatter(
            x[outside],
            y_clamped,
            marker="^",
            color="0.5",
            label=f"flagged, {n_outside} outside frame",
        )

    if model is not None:
        function, parameters, (x_lo, x_hi) = model
        dense_x = np.linspace(x_lo, x_hi, 500)
        axes.plot(dense_x, function(dense_x, **parameters), "-", color="C1", lw=1.5, label="fit")
    _set_y_range(axes, None if y_lo is None else (y_lo, y_hi))
    axes.set_xlabel(_order_axis_label(order_key))
    axes.set_ylabel(format_param_label(param_name))
    axes.set_title(param_name if title is None else title)
    axes.legend(loc="best", fontsize="small")

    return _save(figure, out_path)


def plot_scan(
    x: np.ndarray,
    value: np.ndarray,
    error: np.ndarray,
    *,
    expression: str | None,
    model_function: Callable[..., np.ndarray] | None,
    parameters: Mapping[str, float] | None,
    x_label: str,
    y_label: str,
    title: str,
    out_path: str | Path,
) -> Path:
    """Integral asymmetry against field/temperature, with an optional fit."""
    x = np.asarray(x, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
    axes.errorbar(x, value, yerr=error, fmt="o", ms=4, elinewidth=0.8, color="C0", label="data")
    if model_function is not None and parameters is not None and x.size:
        dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 500)
        curve = np.asarray(model_function(dense_x, **parameters), dtype=np.float64)
        axes.plot(dense_x, curve, "-", color="C1", lw=1.5, label=expression or "model")
        axes.legend(loc="best", fontsize="small")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    _set_y_range(axes, _margin_range(value, error, fraction=_DATA_Y_MARGIN))
    return _save(figure, out_path)


def plot_spectrum(
    frequency: np.ndarray,
    real: np.ndarray,
    magnitude: np.ndarray,
    *,
    run_number: int,
    out_path: str | Path,
) -> Path:
    """Real and magnitude Fourier channels for one reduced run."""
    figure = _new_figure(_FIGSIZE)
    axes = figure.add_subplot(111)
    axes.plot(frequency, magnitude, color="C0", lw=1.2, label="magnitude")
    axes.plot(frequency, real, color="C1", lw=1.0, alpha=0.8, label="real")
    axes.set_xlabel("frequency / MHz")
    axes.set_ylabel("Fourier amplitude")
    axes.set_title(f"Run {run_number} — Fourier spectrum")
    axes.legend(loc="best", fontsize="small")
    return _save(figure, out_path)


__all__ = [
    "MATPLOTLIB_HINT",
    "frame_for_record",
    "plot_fit",
    "plot_reduced",
    "plot_scan",
    "plot_spectrum",
    "plot_trend",
    "require_matplotlib",
]
