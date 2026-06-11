"""Centralised matplotlib styling for the BENCH design language.

All functions are safe to call even when matplotlib is not installed — they
import matplotlib lazily and no-op silently on ImportError.
"""

from __future__ import annotations

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import register_matplotlib_fonts

_FONTS_REGISTERED = False


def _ensure_fonts() -> None:
    """Lazily register the bundled mono face with matplotlib, once."""
    global _FONTS_REGISTERED
    if not _FONTS_REGISTERED:
        register_matplotlib_fonts()
        _FONTS_REGISTERED = True


def style_axes(ax: object) -> None:
    """Apply BENCH spine, tick, grid, and background styling to one Axes.

    Matches the design-handoff plot grammar (prototype ``PlotSVG``): an OPEN
    frame — left and bottom spines only — with small outward tick marks in a
    lighter grey than their monospaced labels. The closed four-spine box and
    the default tick treatment are most of what reads as "very matplotlib".
    """
    _ensure_fonts()
    try:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)  # type: ignore[union-attr]
        for side in ("left", "bottom"):
            spine = ax.spines[side]  # type: ignore[union-attr]
            spine.set_color(tokens.PLOT_AXIS)
            spine.set_linewidth(1.0)
        ax.tick_params(  # type: ignore[union-attr]
            which="both",
            direction="out",
            length=3.5,
            width=0.8,
            color=tokens.PLOT_TICK_MARK,
            labelcolor=tokens.PLOT_TICK_LABEL,
            labelsize=9,
            labelfontfamily="IBM Plex Mono",
        )
        r, g, b, a = tokens.PLOT_GRID
        ax.grid(True, color=(r, g, b), alpha=a, linewidth=0.6)  # type: ignore[union-attr]
        ax.set_axisbelow(True)  # type: ignore[union-attr]
        ax.set_facecolor(tokens.SURFACE)  # type: ignore[union-attr]
        # Axis labels: sans (per the handoff only symbols are italic; words
        # and units stay roman) in the muted label grey, a step above the
        # tick-label size. Colour/size live on the persistent label Text
        # objects, so later set_xlabel calls keep them.
        for axis_label in (ax.xaxis.label, ax.yaxis.label):  # type: ignore[union-attr]
            axis_label.set_color(tokens.PLOT_TICK_LABEL)
            axis_label.set_fontsize(10)
    except Exception:
        pass


def draw_zero_line(ax: object) -> None:
    """Draw the handoff's y = 0 reference line (excluded from autoscaling).

    Invisible-by-construction on positive-only data (it coincides with the
    bottom spine or sits outside the view), so callers can apply it from a
    shared rendering chokepoint without per-domain branching.
    """
    try:
        ax.axhline(  # type: ignore[union-attr]
            0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.8, zorder=1.5
        )
    except Exception:
        pass


def style_figure(fig: object) -> None:
    """Apply BENCH background colour to a Figure patch."""
    try:
        fig.patch.set_facecolor(tokens.SURFACE)  # type: ignore[union-attr]
    except Exception:
        pass


def style_legend(legend: object) -> None:
    """Apply BENCH frame and text styling to a Legend, if present.

    Matches the handoff legend: white 95%-opaque card with a 1px border and
    monospaced entries (the prototype lists runs/temperatures in mono).
    """
    if legend is None:
        return
    _ensure_fonts()
    try:
        frame = legend.get_frame()  # type: ignore[union-attr]
        r, g, b, a = tokens.PLOT_LEGEND_BG
        frame.set_facecolor((r, g, b))
        frame.set_alpha(a)
        frame.set_edgecolor(tokens.BORDER)
        frame.set_linewidth(1.0)
        for text in legend.get_texts():  # type: ignore[union-attr]
            text.set_fontfamily("IBM Plex Mono")
            text.set_fontsize(9)
            text.set_color(tokens.TEXT)
        title = legend.get_title()  # type: ignore[union-attr]
        if title is not None and title.get_text():
            title.set_fontsize(9)
            title.set_color(tokens.TEXT_MUTED)
    except Exception:
        pass


def draw_fit_range_span(ax: object, x_min: float, x_max: float) -> tuple[object, object, object]:
    """Draw a BENCH-styled fit-range span on *ax* and return the three artists.

    Replaces the ad-hoc "gold" / "darkorange" colours used previously.
    Returns (span, left_line, right_line).
    """
    span = ax.axvspan(  # type: ignore[union-attr]
        x_min,
        x_max,
        color=tokens.PLOT_FIT_RANGE_FACE,
        alpha=0.06,
        zorder=1,
    )
    left_line = ax.axvline(  # type: ignore[union-attr]
        x_min,
        color=tokens.PLOT_FIT_RANGE_EDGE,
        alpha=0.40,
        linestyle="--",
        linewidth=1.5,
        zorder=4,
    )
    right_line = ax.axvline(  # type: ignore[union-attr]
        x_max,
        color=tokens.PLOT_FIT_RANGE_EDGE,
        alpha=0.40,
        linestyle="--",
        linewidth=1.5,
        zorder=4,
    )
    return span, left_line, right_line
