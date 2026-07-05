"""Read-only matplotlib preview canvas for the trend model-fit dialogs.

Shows the "parameter vs X" data points and a candidate model curve so the user
can see their fit range, window gaps, and seeds *before* running a fit. In this
phase the canvas is READ-ONLY: :meth:`TrendPreviewCanvas.enable_drag` merely
stores the flag; the drag signals declared here (``range_edge_dragged``,
``window_edge_dragged``, ``exclude_region_requested``) are wired to matplotlib
events in a later phase.

The canvas is built exclusively via
:func:`asymmetry.gui.widgets.mpl_canvas.create_canvas` (a structural check
forbids constructing ``FigureCanvasQTAgg`` directly). ``create_canvas`` imports
matplotlib lazily, so importing *this* module never requires matplotlib to be
installed — canvas creation is guarded by ``try/except ImportError`` and the
widget degrades to a placeholder label when matplotlib is absent.

All plot styling comes from :mod:`asymmetry.gui.styles.plots` and the
:mod:`asymmetry.gui.styles.tokens` palette so the preview matches the BENCH
plot grammar used elsewhere in the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from asymmetry.gui.styles import plots as plot_styles
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.mpl_canvas import create_canvas


@dataclass
class PreviewSeries:
    """One data trace (base dialog draws 1; cross-group draws N)."""

    label: str
    x: NDArray
    y: NDArray
    yerr: NDArray | None
    xerr: NDArray | None


@dataclass
class PreviewRange:
    """What to draw for one fit range: its extent, windows, mask, and curve."""

    idx: int
    x_min: float | None
    x_max: float | None
    windows: list[tuple[float, float]] | None
    #: Per-point of the PRIMARY series; True = point enters the fit.
    in_mask: NDArray[np.bool_]
    curve_x: NDArray
    curve_y: NDArray
    #: Solid (fitted) vs dashed (seed) curve.
    fitted: bool


_State = Literal["ready", "empty", "loading", "error"]


class TrendPreviewCanvas(QWidget):
    """Preview canvas for a trend fit: data points + candidate model curve(s).

    Public surface matches the frozen C7/C8 contract. Setters mutate internal
    state and schedule a redraw; ``_redraw`` is the single chokepoint that
    repaints from stored state.
    """

    # Phase-2 drag signals (declared now so the seams exist; not yet emitted).
    range_edge_dragged = Signal(int, float, float)  # (range_idx, x_min, x_max)
    window_edge_dragged = Signal(int, int, float, float)  # (range_idx, window_idx, lo, hi)
    exclude_region_requested = Signal(int, float, float)  # (range_idx, lo, hi)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ── Internal state repainted by _redraw ──────────────────────────────
        self._series: list[PreviewSeries] = []
        self._ranges: list[PreviewRange] = []
        self._active_idx: int | None = None
        self._state: _State = "empty"
        self._message: str = ""
        self._drag_enabled: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._has_mpl = False
        self._figure = None
        self._canvas = None
        try:
            self._figure, self._canvas = create_canvas(layout="constrained")
            layout.addWidget(self._canvas, 1)
            self._has_mpl = True
            plot_styles.style_figure(self._figure)
        except ImportError:
            layout.addWidget(QLabel("matplotlib not installed - preview disabled"), 1)

        self._redraw()

    # ── Public API (frozen C7/C8 contract) ───────────────────────────────────
    def set_series(self, series: list[PreviewSeries]) -> None:
        """Replace the data traces, then redraw from stored state."""
        self._series = list(series)
        self._redraw()

    def set_active_range(self, idx: int | None) -> None:
        """Set which range is active (full span/edge treatment), then redraw."""
        self._active_idx = idx
        self._redraw()

    def set_ranges(self, ranges: list[PreviewRange]) -> None:
        """Replace the fit ranges (and their candidate curves), then redraw."""
        self._ranges = list(ranges)
        self._redraw()

    def set_state(self, state: _State, message: str = "") -> None:
        """Set the render state (``ready``/``empty``/``loading``/``error``)."""
        self._state = state
        self._message = message
        self._redraw()

    def enable_drag(self, enabled: bool) -> None:
        """Store the drag flag. Phase 1: no behaviour; Phase 2 adds mpl events."""
        self._drag_enabled = enabled

    # ── Rendering ─────────────────────────────────────────────────────────────
    def _redraw(self) -> None:
        """Repaint the whole canvas from stored state (single chokepoint)."""
        if not self._has_mpl or self._figure is None or self._canvas is None:
            return

        # "loading" overlays the last rendered content rather than clearing, so
        # a debounced stream of updates does not flicker.
        if self._state == "loading":
            self._draw_loading_overlay()
            self._canvas.draw_idle()
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if self._state == "empty" or not self._series:
            self._draw_empty(ax)
            self._canvas.draw_idle()
            return

        plot_styles.style_axes(ax)

        primary = self._series[0]
        self._draw_series(ax, primary, is_primary=True)
        for extra in self._series[1:]:
            self._draw_series(ax, extra, is_primary=False)

        # Curves are suppressed in the "error" state (points still drawn above).
        if self._state != "error":
            self._draw_ranges(ax)

        if self._state == "error" and self._message:
            self._draw_message_banner(ax, self._message, tokens.ERROR)

        self._canvas.draw_idle()

    def _draw_empty(self, ax: object) -> None:
        """Clear to a titled, muted centred placeholder (no blank white box)."""
        message = self._message or "No data to preview"
        # Mirrors the titled-empty-axis approach in fit_parameters_panel; here we
        # use the muted token explicitly rather than the generic "gray".
        try:
            ax.clear()  # type: ignore[union-attr]
            ax.text(  # type: ignore[union-attr]
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                transform=ax.transAxes,  # type: ignore[union-attr]
                color=tokens.TEXT_MUTED,
                wrap=True,
            )
            ax.set_axis_off()  # type: ignore[union-attr]
        except Exception:
            pass

    def _draw_series(self, ax: object, series: PreviewSeries, *, is_primary: bool) -> None:
        """Draw one data trace: in-mask points coloured, excluded points greyed.

        Only the primary series carries a per-point ``in_mask`` (from the active
        range); secondary series are drawn wholly in the data colour.
        """
        x = np.asarray(series.x, dtype=float)
        y = np.asarray(series.y, dtype=float)
        yerr = None if series.yerr is None else np.asarray(series.yerr, dtype=float)
        xerr = None if series.xerr is None else np.asarray(series.xerr, dtype=float)

        mask = self._primary_mask(x.size) if is_primary else np.ones(x.size, dtype=bool)

        # Split into in-fit (coloured) and excluded (greyed) subsets.
        self._draw_points(ax, x, y, yerr, xerr, mask, color=tokens.PLOT_DATA, label=series.label)
        excluded = ~mask
        if np.any(excluded):
            self._draw_points(
                ax, x, y, yerr, xerr, excluded, color=tokens.PLOT_LOW_COUNT, label=None
            )

    def _primary_mask(self, n: int) -> NDArray[np.bool_]:
        """In-fit mask for the primary series, taken from the active range.

        Falls back to all-in when no active range, no mask, or a length
        mismatch (e.g. the mask lags a just-replaced series).
        """
        active = self._active_range()
        if active is None or active.in_mask is None:
            return np.ones(n, dtype=bool)
        mask = np.asarray(active.in_mask, dtype=bool)
        if mask.size != n:
            return np.ones(n, dtype=bool)
        return mask

    def _draw_points(
        self,
        ax: object,
        x: NDArray,
        y: NDArray,
        yerr: NDArray | None,
        xerr: NDArray | None,
        sel: NDArray[np.bool_],
        *,
        color: str,
        label: str | None,
    ) -> None:
        """Draw the selected subset of points as an errorbar marker series."""
        if not np.any(sel):
            return
        xs, ys = x[sel], y[sel]
        ye = yerr[sel] if yerr is not None else None
        xe = xerr[sel] if xerr is not None else None
        try:
            ax.errorbar(  # type: ignore[union-attr]
                xs,
                ys,
                yerr=ye,
                xerr=xe,
                fmt="o",
                markersize=4,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2,
                linestyle="none",
                label=label,
                zorder=3,
            )
        except Exception:
            pass

    def _draw_ranges(self, ax: object) -> None:
        """Draw each range's curve + span; active range gets full treatment."""
        for rng in self._ranges:
            is_active = rng.idx == self._active_idx
            self._draw_curve(ax, rng, is_active=is_active)
            self._draw_span(ax, rng, is_active=is_active)
        # Excluded window gaps are shaded only for the active range so the
        # "excluded region" reads at a glance.
        active = self._active_range()
        if active is not None:
            self._draw_window_gaps(ax, active)

    def _draw_curve(self, ax: object, rng: PreviewRange, *, is_active: bool) -> None:
        """Solid line if fitted, dashed if seed; non-active curves are dimmed."""
        cx = np.asarray(rng.curve_x, dtype=float)
        cy = np.asarray(rng.curve_y, dtype=float)
        if cx.size == 0 or cy.size == 0:
            return
        if rng.fitted:
            color = tokens.PLOT_FIT
            linestyle = "-"
        else:
            color = tokens.PLOT_FIT_PREVIEW
            linestyle = "--"
        alpha = 1.0 if is_active else 0.35
        try:
            ax.plot(  # type: ignore[union-attr]
                cx,
                cy,
                linestyle=linestyle,
                color=color,
                linewidth=1.6,
                alpha=alpha,
                zorder=5 if is_active else 2,
            )
        except Exception:
            pass

    def _draw_span(self, ax: object, rng: PreviewRange, *, is_active: bool) -> None:
        """Fit-range span: full BENCH treatment for active, dimmer for others."""
        if rng.x_min is None or rng.x_max is None:
            return
        if is_active:
            plot_styles.draw_fit_range_span(ax, rng.x_min, rng.x_max)
        else:
            try:
                ax.axvspan(  # type: ignore[union-attr]
                    rng.x_min,
                    rng.x_max,
                    color=tokens.PLOT_FIT_RANGE_FACE,
                    alpha=0.03,
                    zorder=1,
                )
            except Exception:
                pass

    def _draw_window_gaps(self, ax: object, rng: PreviewRange) -> None:
        """Shade+hatch the EXCLUDED sub-intervals between the active windows.

        With windows set, only the union of windows is in-fit; the gaps between
        them (within the range extent) are excluded and shaded so they read as
        excluded regions.
        """
        windows = rng.windows
        if not windows:
            return
        lo_bound = rng.x_min
        hi_bound = rng.x_max
        wins = sorted((lo, hi) for lo, hi in windows)
        # Gap segments: [range_start .. first_win], between consecutive windows,
        # and [last_win .. range_end]. Bounds are omitted when the range extent
        # is open on that side.
        gaps: list[tuple[float, float]] = []
        if lo_bound is not None and wins and lo_bound < wins[0][0]:
            gaps.append((lo_bound, wins[0][0]))
        for (lo_a, hi_a), (lo_b, _hi_b) in zip(wins, wins[1:]):
            if hi_a < lo_b:
                gaps.append((hi_a, lo_b))
        if hi_bound is not None and wins and wins[-1][1] < hi_bound:
            gaps.append((wins[-1][1], hi_bound))

        for lo, hi in gaps:
            try:
                ax.axvspan(  # type: ignore[union-attr]
                    lo,
                    hi,
                    facecolor=tokens.PLOT_LOW_COUNT,
                    edgecolor=tokens.PLOT_LOW_COUNT,
                    alpha=0.18,
                    hatch="///",
                    linewidth=0.0,
                    zorder=1.5,
                )
            except Exception:
                pass

    def _draw_loading_overlay(self) -> None:
        """Overlay a subtle accent "updating…" note, keeping the last content."""
        if self._figure is None:
            return
        try:
            axes = self._figure.get_axes()  # type: ignore[union-attr]
            ax = axes[0] if axes else self._figure.add_subplot(111)  # type: ignore[union-attr]
            self._draw_message_banner(ax, self._message or "updating…", tokens.ACCENT)
        except Exception:
            pass

    def _draw_message_banner(self, ax: object, message: str, color: str) -> None:
        """Draw a small top-left annotation in axes-fraction coordinates."""
        try:
            ax.text(  # type: ignore[union-attr]
                0.02,
                0.97,
                message,
                ha="left",
                va="top",
                transform=ax.transAxes,  # type: ignore[union-attr]
                color=color,
                fontsize=9,
                zorder=10,
            )
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _active_range(self) -> PreviewRange | None:
        """The PreviewRange whose idx matches the active index, if any."""
        if self._active_idx is None:
            return None
        for rng in self._ranges:
            if rng.idx == self._active_idx:
                return rng
        return None
