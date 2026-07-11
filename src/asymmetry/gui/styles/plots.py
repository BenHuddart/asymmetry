"""Centralised matplotlib styling for the BENCH design language.

All functions are safe to call even when matplotlib is not installed — they
import matplotlib lazily and no-op silently on ImportError.
"""

from __future__ import annotations

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import register_matplotlib_fonts

# ── Data-trace overlay palettes ───────────────────────────────────────────────
# Okabe-Ito colour-blind-safe qualitative cycle (the named colours live in
# ``styles/tokens.py``). Used to give each overlaid run a distinct, high-contrast
# trace colour, and the fallback ordering when a plot's base colour is not a
# known period-mode hue. Order preserves the pre-centralisation default palette
# exactly (blue, orange, sky, yellow, green, magenta, black, vermillion) so the
# overlay colour assignment is unchanged. The per-period-mode orderings below
# drop the mode's own base hue so an overlay never collides with the selected
# period trace.
_OKABE_ITO = (
    tokens.TRACE_BLUE,
    tokens.TRACE_ORANGE,
    tokens.TRACE_SKY,
    tokens.TRACE_YELLOW,
    tokens.TRACE_GREEN,
    tokens.TRACE_MAGENTA,
    tokens.TRACE_BLACK,
    tokens.TRACE_VERMILLION,
)

#: Overlay orderings keyed by the two-period mode base colour. Each list omits
#: the hue closest to its base so overlays stay visually separable from the
#: selected period trace.
_PERIOD_OVERLAY_PALETTES: dict[str, tuple[str, ...]] = {
    tokens.PERIOD_RED: (
        tokens.TRACE_BLUE,
        tokens.TRACE_SKY,
        tokens.TRACE_GREEN,
        tokens.TRACE_YELLOW,
        tokens.TRACE_MAGENTA,
        tokens.TRACE_BLACK,
        tokens.TRACE_ORANGE,
    ),
    tokens.PERIOD_GREEN: (
        tokens.TRACE_BLUE,
        tokens.TRACE_SKY,
        tokens.TRACE_YELLOW,
        tokens.TRACE_MAGENTA,
        tokens.TRACE_BLACK,
        tokens.TRACE_ORANGE,
        tokens.TRACE_VERMILLION,
    ),
    tokens.PERIOD_DIFF: (
        tokens.TRACE_ORANGE,
        tokens.TRACE_YELLOW,
        tokens.TRACE_GREEN,
        tokens.TRACE_MAGENTA,
        tokens.TRACE_BLACK,
        tokens.TRACE_VERMILLION,
    ),
    tokens.PERIOD_SUM: (
        tokens.TRACE_BLUE,
        tokens.TRACE_SKY,
        tokens.TRACE_GREEN,
        tokens.TRACE_YELLOW,
        tokens.TRACE_ORANGE,
        tokens.TRACE_BLACK,
    ),
}


def period_overlay_palette(base_color: str) -> tuple[str, ...]:
    """Return the high-contrast overlay ordering for a period-mode base colour.

    The first trace of a two-period (RG) plot uses *base_color* itself; any
    additional overlaid runs cycle through the returned palette so they stay
    distinct from the base hue. Falls back to the full Okabe-Ito cycle when the
    base colour is not one of the known period-mode colours.
    """
    return _PERIOD_OVERLAY_PALETTES.get(base_color.lower(), _OKABE_ITO)


def style_axes(ax: object) -> None:
    """Apply BENCH spine, tick, grid, and background styling to one Axes.

    Matches the design-handoff plot grammar (prototype ``PlotSVG``): an OPEN
    frame — left and bottom spines only — with small outward tick marks in a
    lighter grey than their monospaced labels. The closed four-spine box and
    the default tick treatment are most of what reads as "very matplotlib".
    """
    register_matplotlib_fonts()
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
        )
        try:
            # labelfontfamily needs matplotlib >= 3.8; the project supports
            # 3.7, where ticks simply keep the default face. Kept separate so
            # a rejected kwarg cannot abort the styling that follows.
            ax.tick_params(which="both", labelfontfamily="IBM Plex Mono")  # type: ignore[union-attr]
        except (TypeError, ValueError):
            pass
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


def draw_empty_state_message(ax: object, message: str) -> None:
    """Draw a centred grey placeholder over an axis-off plot for an empty state.

    The shared cue for a compute-on-demand view that has nothing to show yet —
    the frequency panel before an FFT/MaxEnt is computed, or the ALC view before
    a scan is built. *message* is centred in axes-fraction coordinates and the
    axis is hidden, so the result reads as an intentional placeholder rather than
    a blank graph. Callers ``ax.clear()`` first; this only adds the text.
    """
    try:
        ax.text(  # type: ignore[union-attr]
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="gray",
            wrap=True,
        )
        ax.set_axis_off()  # type: ignore[union-attr]
    except Exception:
        pass


def draw_zero_line(
    ax: object,
    y: float = 0.0,
    *,
    linewidth: float = 0.8,
    alpha: float | None = None,
    zorder: float = 1.5,
) -> None:
    """Draw the handoff's y = 0 reference line, excluded from autoscaling.

    ``axhline`` registers its y-value in the data limits, which would anchor
    positive-only plots (grouped counts ≈ N0) to zero and squash the signal.
    Building the Line2D by hand and attaching it via ``add_artist`` keeps it
    out of the autoscale computation entirely, so callers can apply it from a
    shared rendering chokepoint without per-domain branching.

    *y*, *linewidth*, *alpha*, and *zorder* let the waterfall overlay reuse
    the same idiom for its fainter per-trace baselines at each stacked
    trace's shifted zero; the defaults preserve the classic y = 0 line.
    """
    try:
        from matplotlib import lines as mlines

        # x in axes-fraction (always spans the full width), y in data coords.
        transform = ax.get_yaxis_transform(which="grid")  # type: ignore[union-attr]
        line = mlines.Line2D(
            [0.0, 1.0],
            [float(y), float(y)],
            transform=transform,
            color=tokens.PLOT_ZERO_LINE,
            linewidth=linewidth,
            zorder=zorder,
        )
        if alpha is not None:
            line.set_alpha(alpha)
        ax.add_artist(line)  # type: ignore[union-attr]
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
    register_matplotlib_fonts()
    try:
        # Keep the legend out of the layout solver. With many overlaid traces
        # (e.g. an angle scan of 20+ FFT spectra) the legend is tall; if
        # tight/constrained layout tries to reserve room for it on a short plot
        # pane it collapses the axes to a thin band ("squashed" plot). Ignoring
        # it lets the legend overlap the axes (as it already does when there is
        # room) while the plot keeps its full height.
        legend.set_in_layout(False)  # type: ignore[union-attr]
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
