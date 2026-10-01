"""The trace strip: one small plot per local parameter of a sharing-ladder rung.

Each plot is a parameter's fitted value ± error against the series axis, titled
with its trend quality; hovering one names the three factors behind the number.
Runs that keep their own amplitude are drawn as hollow diamonds, and phase
boundaries as dashed lines. Design:
``docs/plans/global-wizard-trend-objective.md`` (D6, Phase 5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from asymmetry.core.fitting.global_search.trend_objective import RungSummary
from asymmetry.core.fitting.trend_quality import TraceQuality
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import char_width, row_height
from asymmetry.gui.styles.plots import draw_empty_state_message, style_axes, style_figure
from asymmetry.gui.utils.errorbar_dots import add_errorbar_dots
from asymmetry.gui.utils.formatting import format_param_label
from asymmetry.gui.widgets.mpl_canvas import create_canvas

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

NO_LOCAL_PARAMETER = "Every parameter is shared: nothing varies from run to run"
EXEMPT_LEGEND = "◇ run that keeps its own amplitude"
#: Plots per row of the strip, and each row's height in table rows.
_COLUMNS = 2
_ROW_HEIGHT_ROWS = 7
#: Rows of plots the strip asks for before it scrolls.
_PREFERRED_PLOT_ROWS = 2
_FONT_SIZE = 7.5


@dataclass(frozen=True)
class TraceSeries:
    """One rung's values of one local parameter, in the rung's run order."""

    #: Who the values belong to: ``"A"``, ``"B"`` or ``"Phase 2"``.
    label: str
    x: NDArray[np.float64]
    value: NDArray[np.float64]
    #: NaN where the fit gave no uncertainty.
    error: NDArray[np.float64]
    #: Runs that keep their own amplitude (plan D12).
    exempt: NDArray[np.bool_]
    colour: str
    filled: bool
    quality: TraceQuality


@dataclass(frozen=True)
class ParameterTrace:
    """One parameter's plot: a series per rung that leaves it local."""

    name: str
    series: tuple[TraceSeries, ...]

    @property
    def title(self) -> str:
        qualities = " · ".join(f"{series.quality.quality:.2f}" for series in self.series)
        return f"{format_param_label(self.name)} — trend {qualities}"

    @property
    def tooltip(self) -> str:
        """The three factors of each series' trend quality, in words."""
        return "\n".join(
            [
                format_param_label(self.name),
                *(
                    f"{series.label}: trend quality {series.quality.quality:.2f} = "
                    f"determined share {series.quality.determined:.0%} × "
                    f"signal {series.quality.signal:.2f} × "
                    f"(1 − zigzag share {series.quality.zigzag:.0%})"
                    for series in self.series
                ),
                "Determined share: runs whose value the data pin down.",
                "Signal: the trace's span against its median error bar.",
                "Zigzag share: interior points that jump out of line with both neighbours.",
            ]
        )


@dataclass(frozen=True)
class TracedRung:
    """A rung to trace, with how its points are drawn."""

    summary: RungSummary
    label: str
    colour: str
    filled: bool = True


def rung_traces(rungs: Sequence[TracedRung]) -> tuple[ParameterTrace, ...]:
    """One trace per parameter that any of ``rungs`` leaves local, in first-seen order."""
    series_by_name: dict[str, list[TraceSeries]] = {}
    for traced in rungs:
        summary = traced.summary
        rows = {row.name: row for row in summary.parameters}
        x = np.array([run.axis_value for run in summary.runs], dtype=float)
        exempt = np.array(
            [run.run_number in summary.rung.exempt_runs for run in summary.runs], dtype=bool
        )
        for name, quality in summary.rung.trend.parameters.items():
            estimates = rows[name].values
            series_by_name.setdefault(name, []).append(
                TraceSeries(
                    label=traced.label,
                    x=x,
                    value=np.array([estimate.value for estimate in estimates], dtype=float),
                    error=np.array([estimate.error for estimate in estimates], dtype=float),
                    exempt=exempt,
                    colour=traced.colour,
                    filled=traced.filled,
                    quality=quality,
                )
            )
    return tuple(ParameterTrace(name, tuple(series)) for name, series in series_by_name.items())


class ParameterTraceStrip(QWidget):
    """Small multiples of local-parameter traces; scrolls when there are many."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._traces: tuple[ParameterTrace, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._legend = QLabel(EXEMPT_LEGEND, self)
        self._legend.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(self._legend)
        self._figure, self._canvas = create_canvas(layout="constrained")
        style_figure(self._figure)
        self._canvas.mpl_connect("motion_notify_event", self._show_factors)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._canvas)
        layout.addWidget(scroll, 1)
        self.set_traces((), "")

    @property
    def figure(self) -> Figure:
        return self._figure

    def _plot_rows(self) -> int:
        return max(-(-len(self._traces) // _COLUMNS), 1)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        # The plots it holds, up to two rows: a canvas's own hint is a full-size figure.
        rows = min(self._plot_rows(), _PREFERRED_PLOT_ROWS)
        return QSize(char_width(60), (rows * _ROW_HEIGHT_ROWS + 1) * row_height())

    def set_traces(
        self,
        traces: Sequence[ParameterTrace],
        axis_label: str,
        boundaries: Sequence[float] = (),
    ) -> None:
        """Draw ``traces`` against the series axis, with ``boundaries`` between phases."""
        self._traces = tuple(traces)
        self._legend.setVisible(
            any(series.exempt.any() for trace in self._traces for series in trace.series)
        )
        figure = self._figure
        figure.clear()
        rows = self._plot_rows()
        # Tall enough for every row of plots, so many parameters scroll rather than shrink.
        self._canvas.setMinimumHeight(rows * _ROW_HEIGHT_ROWS * row_height())
        self.updateGeometry()
        if not self._traces:
            draw_empty_state_message(figure.add_subplot(), NO_LOCAL_PARAMETER)
            self._canvas.draw_idle()
            return
        columns = min(len(self._traces), _COLUMNS)
        for index, trace in enumerate(self._traces):
            axes = figure.add_subplot(rows, columns, index + 1)
            style_axes(axes)
            axes.set_title(trace.title, fontsize=_FONT_SIZE, loc="left", color=tokens.TEXT)
            axes.tick_params(labelsize=_FONT_SIZE)
            axes.locator_params(axis="y", nbins=3)
            for boundary in boundaries:
                axes.axvline(boundary, color=tokens.PLOT_ZERO_LINE, linestyle="--", linewidth=0.8)
            for series in trace.series:
                order = np.argsort(series.x)
                axes.plot(
                    series.x[order],
                    series.value[order],
                    color=series.colour,
                    linewidth=0.8,
                    linestyle="-" if series.filled else (0, (4, 3)),
                )
                for mask, marker in ((~series.exempt, "o"), (series.exempt, "D")):
                    add_errorbar_dots(
                        axes,
                        series.x[mask],
                        series.value[mask],
                        series.error[mask],
                        fmt=marker,
                        color=series.colour,
                        markersize=4,
                        markerfacecolor=(
                            series.colour if series.filled and marker == "o" else tokens.SURFACE
                        ),
                    )
            if index >= len(self._traces) - columns:
                axes.set_xlabel(axis_label, fontsize=_FONT_SIZE)
        self._canvas.draw_idle()

    def trace_at(self, axes: object) -> ParameterTrace:
        """The trace drawn in ``axes``."""
        return self._traces[self._figure.axes.index(axes)]

    def _show_factors(self, event: MouseEvent) -> None:
        self._canvas.setToolTip(
            self.trace_at(event.inaxes).tooltip if event.inaxes is not None and self._traces else ""
        )
