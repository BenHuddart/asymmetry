"""Read-only matplotlib preview canvas for the trend model-fit dialogs.

Shows the "parameter vs X" data points and a candidate model curve so the user
can see their fit range, window gaps, and seeds *before* running a fit. When
:meth:`TrendPreviewCanvas.enable_drag` is on, the active range's edges (and its
window edges) are draggable with the left button, and a right-button drag over
the active range requests an exclusion; these emit ``range_edge_dragged``,
``window_edge_dragged``, and ``exclude_region_requested`` respectively. With
drag disabled (the default) the canvas is read-only.

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

from asymmetry.gui.panels.draggable_handles import nearest_handle
from asymmetry.gui.styles import plots as plot_styles
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.mpl_canvas import create_canvas

#: Cursor-to-edge grab tolerance in device pixels (matches plot_panel's span).
_DRAG_TOLERANCE_PX = 7.0
#: Minimum right-drag extent (device pixels) that counts as an exclude gesture
#: rather than a bare click.
_EXCLUDE_MIN_PX = 4.0


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

    # Drag signals (emitted only while enable_drag(True); see _on_* handlers).
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
        #: Optional residual strip below the main plot (item 4.1). Default OFF;
        #: the dialog opts in via set_show_residuals.
        self._show_residuals: bool = False

        # ── Drag state (all None/idle until a grab; see _on_button_press) ─────
        #: The grabbed edge key, one of ("range","min"/"max") or
        #: ("window", window_idx, "lo"/"hi"); None when not dragging.
        self._active_handle: tuple | None = None
        #: Data-x where a right-button (exclude) gesture began; None otherwise.
        self._exclude_start_x: float | None = None
        #: Pixel-x where the right-button gesture began (click-vs-drag test).
        self._exclude_start_px: float | None = None

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
            # Connect the drag events ONCE; the handler bodies early-return
            # unless self._drag_enabled, so there is no connect/disconnect churn.
            self._canvas.mpl_connect("button_press_event", self._on_button_press)
            self._canvas.mpl_connect("motion_notify_event", self._on_motion_notify)
            self._canvas.mpl_connect("button_release_event", self._on_button_release)
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

    def set_show_residuals(self, enabled: bool) -> None:
        """Toggle the residual strip below the main plot, then redraw.

        When ON, the figure lays out as a 2-row gridspec (main plot ~4×, a
        residual strip ~1× below) sharing the x-axis; when OFF, a single axis.
        The layout is rebuilt from scratch inside ``_redraw`` on each repaint,
        so this only needs to flip the flag and trigger one redraw.
        """
        enabled = bool(enabled)
        if enabled == self._show_residuals:
            return
        self._show_residuals = enabled
        self._redraw()

    def enable_drag(self, enabled: bool) -> None:
        """Toggle drag interaction.

        The mpl events are connected once in ``__init__``; the handlers gate on
        this flag. Disabling mid-session cleanly abandons any in-progress drag.
        """
        self._drag_enabled = enabled
        if not enabled:
            self._active_handle = None
            self._exclude_start_x = None
            self._exclude_start_px = None

    # ── Drag interaction (active range only) ──────────────────────────────────
    def _active_axes(self):
        """The current subplot, or None when nothing is drawn."""
        if not self._has_mpl or self._figure is None:
            return None
        axes = self._figure.get_axes()
        return axes[0] if axes else None

    def _active_handles(self) -> list[tuple[float, tuple]]:
        """``(data_x, key)`` handles for the ACTIVE range only.

        Range edges give keys ``("range", "min"/"max")``; each window gives
        ``("window", window_idx, "lo"/"hi")``. Non-active ranges are omitted, so
        only the active range is draggable.
        """
        active = self._active_range()
        if active is None:
            return []
        handles: list[tuple[float, tuple]] = []
        if active.x_min is not None:
            handles.append((float(active.x_min), ("range", "min")))
        if active.x_max is not None:
            handles.append((float(active.x_max), ("range", "max")))
        for w_idx, (lo, hi) in enumerate(active.windows or []):
            handles.append((float(lo), ("window", w_idx, "lo")))
            handles.append((float(hi), ("window", w_idx, "hi")))
        return handles

    def _on_button_press(self, event) -> None:
        """Grab the nearest active-range edge (left) or start an exclude (right)."""
        if not self._drag_enabled:
            return
        ax = self._active_axes()
        if ax is None or event.inaxes is not ax:
            return
        active = self._active_range()
        if active is None:
            return

        if event.button == 3:
            # Right-button: begin an exclude gesture from this data-x.
            if event.xdata is not None:
                self._exclude_start_x = float(event.xdata)
                self._exclude_start_px = float(event.x)
            return

        if event.button != 1:
            return

        handle = nearest_handle(ax, self._active_handles(), event.x, _DRAG_TOLERANCE_PX)
        if handle is not None:
            self._active_handle = handle

    def _on_motion_notify(self, event) -> None:
        """Move the grabbed edge to the cursor and emit the matching signal."""
        if not self._drag_enabled or self._active_handle is None:
            return
        # Ignore motion that leaves the axes (no data-x to snap to).
        if event.xdata is None:
            return
        active = self._active_range()
        if active is None:
            return
        new_x = float(event.xdata)
        kind = self._active_handle[0]
        if kind == "range":
            self._drag_range_edge(active, self._active_handle[1], new_x)
        elif kind == "window":
            self._drag_window_edge(active, self._active_handle[1], self._active_handle[2], new_x)

    def _drag_range_edge(self, active: PreviewRange, which: str, new_x: float) -> None:
        """Update+clamp a range min/max edge and emit ``range_edge_dragged``."""
        x_min = active.x_min if active.x_min is not None else new_x
        x_max = active.x_max if active.x_max is not None else new_x
        if which == "min":
            # Clamp so min never crosses above max.
            x_min = min(new_x, x_max)
        else:
            # Clamp so max never crosses below min.
            x_max = max(new_x, x_min)
        active.x_min = x_min
        active.x_max = x_max
        self._redraw()
        self.range_edge_dragged.emit(active.idx, x_min, x_max)

    def _drag_window_edge(self, active: PreviewRange, w_idx: int, which: str, new_x: float) -> None:
        """Update+clamp one window's lo/hi edge and emit ``window_edge_dragged``."""
        windows = active.windows or []
        if not (0 <= w_idx < len(windows)):
            return
        lo, hi = windows[w_idx]
        if which == "lo":
            lo = min(new_x, hi)  # clamp: lo cannot pass hi
        else:
            hi = max(new_x, lo)  # clamp: hi cannot pass lo
        windows[w_idx] = (lo, hi)
        active.windows = windows
        self._redraw()
        self.window_edge_dragged.emit(active.idx, w_idx, lo, hi)

    def _on_button_release(self, event) -> None:
        """Settle a left-drag, or fire the exclude gesture on right-release."""
        if not self._drag_enabled:
            return

        if event.button == 1 and self._active_handle is not None:
            active = self._active_range()
            if active is not None and event.xdata is not None:
                kind = self._active_handle[0]
                new_x = float(event.xdata)
                if kind == "range":
                    self._drag_range_edge(active, self._active_handle[1], new_x)
                elif kind == "window":
                    self._drag_window_edge(
                        active, self._active_handle[1], self._active_handle[2], new_x
                    )
            self._active_handle = None
            return

        if event.button == 3 and self._exclude_start_x is not None:
            start_x = self._exclude_start_x
            start_px = self._exclude_start_px
            self._exclude_start_x = None
            self._exclude_start_px = None
            active = self._active_range()
            if active is None or event.xdata is None:
                return
            # Click-vs-drag: a negligible pixel move is a click, not an exclude.
            if start_px is not None and abs(float(event.x) - start_px) < _EXCLUDE_MIN_PX:
                return
            lo, hi = sorted((start_x, float(event.xdata)))
            self.exclude_region_requested.emit(active.idx, lo, hi)

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
        ax, resid_ax = self._add_axes()

        if self._state == "empty" or not self._series:
            self._draw_empty(ax)
            if resid_ax is not None:
                self._style_residual_axis(resid_ax)
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

        # Residual strip (item 4.1): standardized residuals of the active range's
        # primary series against its candidate curve. Empty axis when there is no
        # curve to residual against.
        if resid_ax is not None:
            self._draw_residuals(resid_ax, ax)

        self._canvas.draw_idle()

    def _add_axes(self) -> tuple[object, object | None]:
        """Add the main axis (and, when enabled, a shared-x residual strip below).

        Returns ``(main_ax, residual_ax_or_None)``. The main axis is always the
        FIRST axis on the figure so ``_active_axes`` (which drives dragging)
        keeps returning it. With residuals off this is just ``add_subplot(111)``;
        with them on we build a 2-row height-ratio gridspec (4:1) sharing x.
        """
        if not self._show_residuals:
            return self._figure.add_subplot(111), None
        try:
            gs = self._figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
            ax = self._figure.add_subplot(gs[0])
            resid_ax = self._figure.add_subplot(gs[1], sharex=ax)
            # Hide the main axis's x tick labels; the residual strip carries them.
            try:
                ax.tick_params(labelbottom=False)  # type: ignore[union-attr]
            except Exception:
                pass
            return ax, resid_ax
        except Exception:
            # Any layout failure degrades to a single axis rather than crashing.
            return self._figure.add_subplot(111), None

    def _style_residual_axis(self, resid_ax: object) -> None:
        """Apply BENCH axis styling + the small "resid/σ" label to the strip."""
        plot_styles.style_axes(resid_ax)
        try:
            resid_ax.set_ylabel("resid/σ", fontsize=8)  # type: ignore[union-attr]
        except Exception:
            pass

    def _draw_residuals(self, resid_ax: object, main_ax: object) -> None:
        """Draw standardized residuals for the active range's primary series.

        For each in-mask primary-series point, plot ``(y - f(x)) / σ`` where
        ``f(x)`` is the active range's candidate curve interpolated to the data x
        (``np.interp(series_x, curve_x, curve_y)``) and ``σ`` is the series yerr
        (falling back to 1 where yerr is missing/non-positive). A ±1 band and a
        zero line give scale. With no active-range curve the axis is left empty.
        """
        self._style_residual_axis(resid_ax)
        # Zero line + ±1 guide band, always drawn so the strip reads as a residual
        # plot even before a curve exists.
        try:
            resid_ax.axhspan(-1.0, 1.0, color=tokens.PLOT_ZERO_LINE, alpha=0.12, zorder=0)  # type: ignore[union-attr]
            resid_ax.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.8, zorder=1)  # type: ignore[union-attr]
        except Exception:
            pass

        active = self._active_range()
        if active is None or not self._series:
            return
        curve_x = np.asarray(active.curve_x, dtype=float)
        curve_y = np.asarray(active.curve_y, dtype=float)
        if curve_x.size == 0 or curve_y.size == 0:
            return

        primary = self._series[0]
        x = np.asarray(primary.x, dtype=float)
        y = np.asarray(primary.y, dtype=float)
        if x.size == 0 or y.size == 0:
            return

        mask = self._primary_mask(x.size)
        xs = x[mask]
        ys = y[mask]
        if xs.size == 0:
            return

        yerr = None if primary.yerr is None else np.asarray(primary.yerr, dtype=float)
        if yerr is not None and yerr.size == x.size:
            sigma = yerr[mask]
        else:
            sigma = np.ones(xs.size, dtype=float)
        # Guard: a missing / non-positive σ would blow up the standardized
        # residual — fall back to unit weight for those points.
        sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, 1.0)

        fitted_y = np.interp(xs, curve_x, curve_y)
        residuals = (ys - fitted_y) / sigma
        try:
            resid_ax.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=0.8, zorder=1)  # type: ignore[union-attr]
            resid_ax.plot(  # type: ignore[union-attr]
                xs,
                residuals,
                marker="o",
                markersize=3,
                linestyle="none",
                color=tokens.PLOT_DATA,
                zorder=3,
            )
        except Exception:
            pass

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
