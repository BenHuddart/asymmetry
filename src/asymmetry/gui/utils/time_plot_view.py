"""Owned time-display snapshots and artists retained across viewport changes.

Every plot publication creates a fresh owner. Viewport changes consume that
snapshot; publishing data, masks, transforms, fits or labels replaces it.
No source array is frozen, hashed, or used as a cross-publication cache key.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from asymmetry.gui.styles.plots import style_legend
from asymmetry.gui.utils.connector_geometry import ConnectorGeometry
from asymmetry.gui.utils.errorbar_dots import add_errorbar_dots, update_errorbar_dots


@cache
def connector_line_class():
    """Build the viewport artist only when matplotlib is available."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.lines import Line2D

    class DataConnector(Line2D):
        def __init__(self, arrays, mask, color, decimate):
            super().__init__(
                [],
                [],
                color=color,
                linewidth=0.65,
                linestyle=(0, (2.0, 1.3)),
                alpha=0.65,
                zorder=1.6,
                label="_nolegend_",
                visible=False,
            )
            self.source = (*arrays, mask)
            self.geometry = None
            self.viewport = None
            self.decimate = decimate

        def draw(self, renderer):
            if not self.get_visible():
                return
            if self.geometry is None:
                self.geometry = ConnectorGeometry(*self.source)
            window = tuple(self.axes.get_xlim())
            columns = (
                max(1, int(np.ceil(2 * self.axes.bbox.width)))
                if self.decimate and isinstance(renderer, RendererAgg)
                else None
            )
            key = (window, columns)
            if key != self.viewport:
                vertices = self.geometry.vertices(window, columns)
                self.set_data(vertices[:, 0], vertices[:, 1])
                self.viewport = key
            super().draw(renderer)

    return DataConnector


@dataclass
class TimeTrace:
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
    mask: np.ndarray
    marker: object
    bars: object
    connector: object

    def set_connected(self, enabled):
        self.connector.set_visible(enabled)
        plane = 3 if enabled else 2
        self.bars.set_zorder(plane)
        self.marker.set_zorder(plane + 0.1)

    def update(self, panel, window):
        time, values, error = self.arrays
        visible = panel._visible_plot_indices(time, self.mask, window)
        indices = panel._decimated_plot_indices(
            time, self.mask, values=values, visible_indices=visible
        )
        panel._decimation_points_shown += int(indices.size)
        panel._decimation_points_total += int(visible.size)
        panel._decimation_applied_for_current_view |= indices.size < visible.size
        update_errorbar_dots(self.marker, self.bars, time[indices], values[indices], error[indices])
        self.connector.decimate = panel.decimation_enabled()
        return int(indices.size)


class TimePlotView:
    """One publication's immutable arrays, traces, bounds and viewport artists."""

    def __init__(self):
        self.traces: list[TimeTrace] = []
        self.arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.y_bounds: dict = {}
        self.x_bounds = None
        self.indicator_axis = None
        self.fits: list[tuple] = []

    def own(self, source):
        """Own a copy even when a writable alias exists to a read-only source."""
        key = id(source)
        if key not in self.arrays:
            owned = np.array(source, copy=True)
            owned.flags.writeable = False
            self.arrays[key] = (source, owned)
            self.arrays[id(owned)] = (owned, owned)
        return self.arrays[key][1]

    def add_trace(self, panel, ax, time, values, error, mask, window, **kwargs):
        arrays = tuple(self.own(source) for source in (time, values, error))
        owned_mask = self.own(mask)
        marker, bars = add_errorbar_dots(ax, [], [], [], **kwargs)
        connector = connector_line_class()(
            arrays,
            owned_mask,
            marker.get_color(),
            panel.decimation_enabled(),
        )
        ax.add_artist(connector)
        marker.connector_line = connector
        trace = TimeTrace(arrays, owned_mask, marker, bars, connector)
        trace.set_connected(panel.connect_points_enabled())
        self.traces.append(trace)
        return trace.update(panel, window)

    def capture_readout(self, panel):
        for name in (
            "_last_plot_time",
            "_last_plot_asymmetry",
            "_last_plot_error",
            "_last_low_count_mask",
            "_last_plot_sentinel_mask",
        ):
            source = getattr(panel, name)
            if source is not None:
                setattr(panel, name, self.own(source))
        self.arrays = {id(owned): (owned, owned) for _source, owned in self.arrays.values()}

    def set_connected(self, enabled):
        for trace in self.traces:
            trace.set_connected(enabled)
        for line, off_label, fit_label in self.fits:
            line.set_label(f"Fit: {fit_label}" if enabled else off_label)
        for ax in dict.fromkeys(trace.marker.axes for trace in self.traces):
            legend = ax.get_legend()
            if legend is not None:
                style_legend(ax.legend(loc=legend._loc, title=legend.get_title().get_text()))

    def refresh(self, panel):
        """Resolve current limits against this publication without analysis work."""
        if not self.traces:
            return False
        panel._sync_auto_from_buttons(list(self.y_bounds))
        resolved, window = panel._resolve_x(self.x_bounds)
        panel._reset_decimation_view_state()
        for trace in self.traces:
            trace.update(panel, window)
        for axis_id, bounds in list(self.y_bounds.items()):
            resolved.update(panel._resolve_y(axis_id, bounds))
        panel._apply_x_axis_decimation_indicator(self.indicator_axis or panel._ax)
        panel._apply_limits(resolved)
        if panel._vector_subplot_datasets:
            panel._refresh_fit_target_decoration()
        return True


def draw_masked_errorbar(panel, ax, time, values, error, mask, window, **kwargs):
    """Draw the shared marker/error idiom, retaining supported time-view artists."""
    if panel._is_frequency_domain_dataset(panel._current_dataset):
        panel._time_view = None
    if panel._time_view is not None:
        return panel._time_view.add_trace(panel, ax, time, values, error, mask, window, **kwargs)
    visible = panel._visible_plot_indices(time, mask, window)
    indices = panel._decimated_plot_indices(time, mask, values=values, visible_indices=visible)
    panel._decimation_applied_for_current_view |= indices.size < visible.size
    panel._decimation_points_shown += int(indices.size)
    panel._decimation_points_total += int(visible.size)
    if indices.size:
        add_errorbar_dots(ax, time[indices], values[indices], error[indices], **kwargs)
    return int(indices.size)


def plot_fit_curve(panel, ax, time, values, color, fit_identity, off_label):
    """Keep solid fit identity available when the connector setting changes."""
    dataset, fit_label = fit_identity
    if len(panel._current_datasets) > 1:
        fit_label = f"{panel._dataset_label_for(dataset)} — {fit_label}"
    label = (
        f"Fit: {fit_label}"
        if panel._time_view is not None and panel.connect_points_enabled()
        else off_label
    )
    lines = ax.plot(
        time, values, "-", color=color, linewidth=2, marker="None", zorder=2, label=label
    )
    if panel._time_view is not None:
        panel._time_view.fits.append((lines[0], off_label, fit_label))


def reuse_fit_guides(panel):
    """Update attached time-range artists, including their draggable endpoints."""
    from matplotlib.patches import Rectangle

    axes = panel._fit_range_axes()
    lo, hi = panel._fit_guide_window()
    if lo is None or hi is None or len(panel._fit_span_artists) != len(axes):
        return False
    artists = list(
        zip(
            axes,
            panel._fit_span_artists,
            panel._fit_min_handles,
            panel._fit_max_handles,
            strict=True,
        )
    )
    if not all(
        span in ax.patches and left in ax.lines and right in ax.lines
        for ax, span, left, right in artists
    ):
        return False
    for _ax, span, left, right in artists:
        if isinstance(span, Rectangle):
            span.set_bounds(lo, 0, hi - lo, 1)
        else:
            span.set_xy([[lo, 0], [lo, 1], [hi, 1], [hi, 0], [lo, 0]])
        left.set_xdata([lo, lo])
        right.set_xdata([hi, hi])
    return True
