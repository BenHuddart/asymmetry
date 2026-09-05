"""Fast "dots with error bars" rendering for interactive time-domain plots.

``Axes.errorbar`` is the natural idiom for a muSR asymmetry trace, but it is
built for small point counts: the vertical bars become a ``LineCollection``
with one :class:`~matplotlib.path.Path` per point (``vlines``), constructed
through per-segment masked-array conversions. At the interactive view's
4 000-point display budget that construction alone costs 40–90 ms per trace
on the GUI thread — on every run switch, pan tick and redraw.

:func:`add_errorbar_dots` draws the same picture with two artists: a
marker-only :class:`~matplotlib.lines.Line2D` for the dots and a
``LineCollection`` holding a *single* NaN-separated path for all the bars
(Matplotlib breaks a path at every NaN vertex, so one path carries N disjoint
vertical segments and Agg strokes them in one pass). Construction is a
handful of vectorised numpy allocations (~1 ms). Keeping the bars in a
collection rather than a second ``Line2D`` also matters for the legend:
``loc="best"`` scores candidate positions against every ``Line2D`` vertex,
and a 12 000-vertex bar line would cost ~10 ms per draw there, whereas a
collection contributes only its offsets (none). The legend keeps the
errorbar glyph (dot with a vertical bar) through a handler registered for
the marker line's class.

This is the one shared implementation for interactive errorbar-dot traces
(``plot_panel.py::_plot_errorbar_masked`` routes every time-domain trace
through it); preview surfaces with a bounded, one-shot point budget
(``gui/utils/plot_decimation.py``) may keep plain ``errorbar``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["add_errorbar_dots", "errorbar_dots_line_class"]


def _import_mpl():
    from matplotlib.legend import Legend
    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.lines import Line2D

    return Legend, HandlerLine2D, Line2D


def _first(value: object) -> object:
    """Return the first entry of a per-item collection property (or *value*)."""
    try:
        return value[0]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return value


def _make_classes():
    """Build (once) the marker-line class and its legend handler."""
    legend_cls, handler_line2d_cls, line2d_cls = _import_mpl()

    class ErrorbarDotsLine(line2d_cls):
        """Marker-only ``Line2D`` that remembers its companion error-bar line.

        The legend handler reads :attr:`bar_line` (the bars'
        ``LineCollection``) to draw the bar glyph with the bars' own
        colour/width, as ``HandlerErrorbar`` does for a real
        ``ErrorbarContainer``.
        """

        bar_line: object | None = None

    class _HandlerErrorbarDots(handler_line2d_cls):
        """``HandlerLine2D`` plus a vertical bar through the legend marker."""

        def __init__(self, yerr_size: float = 0.5, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._yerr_size = yerr_size

        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
            artists = super().create_artists(
                legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
            )
            bar = getattr(orig_handle, "bar_line", None)
            if bar is None or not artists:
                return artists
            marker_line = artists[0]
            xdata = np.asarray(marker_line.get_xdata(), dtype=float)
            ydata = np.asarray(marker_line.get_ydata(), dtype=float)
            if xdata.size == 0:
                return artists
            # HandlerLine2D with numpoints == 1 draws three x samples and
            # marks only the middle one; mirror the marker position.
            x = float(xdata[xdata.size // 2])
            y = float(ydata[ydata.size // 2])
            half = self._yerr_size * fontsize
            bar_glyph = line2d_cls(
                [x, x],
                [y - half, y + half],
                color=_first(bar.get_color()),
                linewidth=_first(bar.get_linewidth()),
                alpha=bar.get_alpha(),
            )
            bar_glyph.set_transform(trans)
            # The bar sits behind the marker, as on the axes.
            return [bar_glyph, *artists]

    legend_cls.update_default_handler_map({ErrorbarDotsLine: _HandlerErrorbarDots()})
    return ErrorbarDotsLine


_line_cls: type | None = None


def errorbar_dots_line_class() -> type:
    """Return the marker-line class (built lazily so importing never needs matplotlib).

    Every ``ax.legend()`` call picks up the errorbar glyph for instances of
    this class through the default handler map registered on first use.
    """
    global _line_cls
    if _line_cls is None:
        _line_cls = _make_classes()
    return _line_cls


def _parse_fmt(fmt: str) -> tuple[str | None, str | None, str | None]:
    """Return ``(linestyle, marker, color)`` from a ``plot``-style format."""
    if not fmt:
        return None, None, None
    try:
        from matplotlib.axes._base import _process_plot_format

        linestyle, marker, color = _process_plot_format(fmt)
        return linestyle, marker, color
    except Exception:  # pragma: no cover - private-API fallback
        return None, fmt, None


def add_errorbar_dots(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    fmt: str = ".",
    color: object | None = None,
    ecolor: object | None = None,
    markersize: float | None = None,
    label: str | None = None,
    zorder: float = 2.0,
    alpha: float | None = None,
    elinewidth: float | None = None,
    **marker_kwargs: Any,
):
    """Draw *y* ± *yerr* at *x* as dots with vertical error bars.

    Mirrors the ``Axes.errorbar(x, y, yerr=..., fmt=".", ...)`` picture: the
    bars take ``ecolor`` (default: the marker colour), the marker sits just
    above them (``zorder + 0.1``, as errorbar's ``barsabove=False``), a
    ``None`` colour advances the axes' property cycle, and a non-finite
    ``yerr`` draws no bar for that point. Returns ``(marker_line, bars)`` —
    the marker ``Line2D`` (which carries *label*) and the bars'
    ``LineCollection``.
    """
    line_cls = errorbar_dots_line_class()
    from matplotlib import rcParams
    from matplotlib.collections import LineCollection

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(yerr, dtype=float)
    if err.shape != y.shape:
        err = np.broadcast_to(err, y.shape)

    _linestyle, marker, fmt_color = _parse_fmt(fmt)
    if marker is None:
        marker = "."
    if color is None:
        color = fmt_color
    if color is None:
        # Advance the cycle exactly as ``plot``/``errorbar`` would.
        color = ax._get_lines.get_next_color()
    if ecolor is None:
        ecolor = color

    n = x.size
    xs = np.empty(3 * n, dtype=float)
    ys = np.empty(3 * n, dtype=float)
    xs[0::3] = x
    xs[1::3] = x
    xs[2::3] = np.nan
    ys[0::3] = y - err
    ys[1::3] = y + err
    ys[2::3] = np.nan

    # One path for every bar: ``LineCollection`` builds a Path per *segment*
    # handed to it, so the NaN-separated vertex array is passed as a single
    # segment (N points → one Path), not as N two-point segments.
    bar_line = LineCollection(
        [np.column_stack([xs, ys])],
        linewidths=float(elinewidth)
        if elinewidth is not None
        else float(rcParams["lines.linewidth"]),
        colors=ecolor,
        linestyles="solid",
        label="_nolegend_",
        zorder=zorder,
        alpha=alpha,
    )
    marker_line = line_cls(
        x,
        y,
        linestyle="none",
        marker=marker,
        markersize=markersize if markersize is not None else rcParams["lines.markersize"],
        color=color,
        label=label,
        zorder=zorder + 0.1,
        alpha=alpha,
        **marker_kwargs,
    )
    marker_line.bar_line = bar_line
    # The NaN-separated path would poison the collection's own data limits,
    # so add it without autolim and feed the finite bar ends in explicitly —
    # the same limits ``errorbar`` contributes (markers plus bar extents).
    ax.add_collection(bar_line, autolim=False)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    if np.any(finite):
        xf = x[finite]
        ax.update_datalim(np.column_stack([xf, (y - err)[finite]]))
        ax.update_datalim(np.column_stack([xf, (y + err)[finite]]))
    ax.add_line(marker_line)
    # ``add_line``/``add_collection(autolim=False)`` only extend the data
    # limits; ``plot``/``errorbar`` additionally ask for an autoscale pass at
    # the next draw, and an axis with no explicit limits relies on that.
    request = getattr(ax, "_request_autoscale_view", None)
    if callable(request):
        request()
    return marker_line, bar_line
