"""One figure overlaying a series' data with up to two candidates' fits.

The top axes draw every run's asymmetry as small translucent markers with
candidate A's curve solid and B's dashed, one colour per run along the series
axis. Below, one strip per run shows the normalised residuals (y − f)/σ, A in
the run colour over B in grey, clipped to ±4σ. Design:
``docs/plans/global-wizard-stepper.md`` (D4).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from asymmetry.core.fitting.model_comparison import CandidateSummary, RunFit
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.plots import draw_zero_line, style_axes, style_figure
from asymmetry.gui.utils.formatting import format_reduced_chi_squared
from asymmetry.gui.utils.plot_decimation import preview_stride
from asymmetry.gui.utils.series_colours import series_colours
from asymmetry.gui.widgets.mpl_canvas import create_canvas

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from asymmetry.core.data.dataset import MuonDataset

#: Markers drawn per run in the overlay and per residual strip.
DISPLAY_POINTS_PER_RUN = 300
#: Residual strips shown at most; a longer series shows runs spread evenly along it.
MAX_RESIDUAL_STRIPS = 12
#: Residuals beyond ±this many σ are drawn on the strip's edge.
RESIDUAL_CLIP_SIGMA = 4.0

# Figure layout in logical pixels, converted to figure fractions on every draw.
_LEFT_PX = 58
_RIGHT_PX = 84
_TOP_PX = 8
_OVERLAY_BOTTOM_PX = 40  # time tick labels and the axis label
_OVERLAY_MIN_PX = 150
_RESIDUAL_HEADER_PX = 20
_STRIP_GAP_PX = 3
_STRIP_MAX_PX = 44
_STRIP_MIN_PX = 20
_RESIDUAL_BUDGET_PX = 4 * _STRIP_MAX_PX  # strips thin past four runs, down to the floor
_RESIDUAL_SHARE = 0.4  # nor may the strips take more of the plotting height than this
_NOTE_PX = 16
_BOTTOM_PX = 6
_LABEL_PX = 15  # vertical room one right-edge run label needs, with a little air
_LABEL_FONT_SIZE = 8
_B_DASH = (0, (5, 3))


def _strip_height(strip_count: int, plot_px: float) -> float:
    """Strips share the smaller of the budget and their share of ``plot_px``, within the floor and max."""
    budget = min(_RESIDUAL_BUDGET_PX, _RESIDUAL_SHARE * plot_px)
    return min(_STRIP_MAX_PX, max(_STRIP_MIN_PX, budget / strip_count))


def spread_labels(targets: Sequence[float], gap: float, low: float, high: float) -> list[float]:
    """Label positions in ``[low, high]`` at least ``gap`` apart, each as near its target as fits.

    Positions keep the targets' order. When ``gap`` cannot fit them all, they are
    spaced evenly across the range.
    """
    order = sorted(range(len(targets)), key=lambda index: targets[index])
    if len(order) > 1 and gap * (len(order) - 1) > high - low:
        gap = (high - low) / (len(order) - 1)
    placed = [0.0] * len(targets)
    floor = low
    for index in order:
        placed[index] = max(floor, min(max(targets[index], low), high))
        floor = placed[index] + gap
    ceiling = high
    for index in reversed(order):
        placed[index] = min(placed[index], ceiling)
        ceiling = placed[index] - gap
    return placed


class SeriesFitCanvas(QWidget):
    """Data overlay with candidate A solid, B dashed, and optional residual strips."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure, self._canvas = create_canvas(layout="none")
        style_figure(self._figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._datasets: tuple[MuonDataset, ...] = ()
        self._labels: tuple[str, ...] = ()
        self._colours: tuple[str, ...] = ()
        self._a: CandidateSummary | None = None
        self._b: CandidateSummary | None = None
        self._residuals_visible = True
        self._canvas.mpl_connect("resize_event", self._redraw)
        self._redraw()

    @property
    def figure(self) -> Figure:
        return self._figure

    def set_series(
        self,
        datasets: Sequence[MuonDataset],
        run_labels: Sequence[str],
        axis_values: Sequence[float],
        colours: Sequence[str] | None = None,
    ) -> None:
        """Show ``datasets`` in series order; ``colours`` (e.g. phases) replace the axis gradient."""
        if not len(datasets) == len(run_labels) == len(axis_values):
            raise ValueError("SeriesFitCanvas needs one label and one axis value per dataset")
        if colours is not None and len(colours) != len(datasets):
            raise ValueError("SeriesFitCanvas needs one colour per dataset")
        self._datasets = tuple(datasets)
        self._labels = tuple(run_labels)
        self._colours = (
            tuple(colours) if colours is not None else tuple(series_colours(axis_values))
        )
        self._redraw()

    def set_curves(self, a: CandidateSummary | None, b: CandidateSummary | None) -> None:
        """Overlay candidate ``a`` solid and ``b`` dashed, matched to the runs by run number."""
        self._a = a
        self._b = b
        self._redraw()

    def set_residuals_visible(self, visible: bool) -> None:
        self._residuals_visible = visible
        self._redraw()

    def _strip_runs(self) -> list[int]:
        """Indices of the runs that get a residual strip (none without candidate A)."""
        if not self._residuals_visible or self._a is None:
            return []
        run_count = len(self._datasets)
        if run_count <= MAX_RESIDUAL_STRIPS:
            return list(range(run_count))
        return sorted({round(i) for i in np.linspace(0, run_count - 1, MAX_RESIDUAL_STRIPS)})

    def _redraw(self, _event: object = None) -> None:
        strips = self._strip_runs()
        residual_chrome_px = 0.0
        if strips:
            residual_chrome_px = (
                _RESIDUAL_HEADER_PX
                + len(strips) * _STRIP_GAP_PX
                + (_NOTE_PX if len(strips) < len(self._datasets) else 0)
            )
        chrome_px = _TOP_PX + _OVERLAY_BOTTOM_PX + _BOTTOM_PX + residual_chrome_px
        self._canvas.setMinimumHeight(
            round(chrome_px + _OVERLAY_MIN_PX + len(strips) * _STRIP_MIN_PX)
        )
        width = max(self._canvas.width(), _LEFT_PX + _RIGHT_PX + 1)
        height = max(self._canvas.height(), 1)
        strip_height = _strip_height(len(strips), height - chrome_px) if strips else 0.0
        residual_px = residual_chrome_px + len(strips) * strip_height
        self._figure.clear()

        def rect(bottom_px: float, height_px: float) -> list[float]:
            return [
                _LEFT_PX / width,
                bottom_px / height,
                (width - _LEFT_PX - _RIGHT_PX) / width,
                height_px / height,
            ]

        overlay_bottom = _BOTTOM_PX + residual_px + _OVERLAY_BOTTOM_PX
        overlay_height = max(height - _TOP_PX - overlay_bottom, 1.0)
        overlay = self._figure.add_axes(rect(overlay_bottom, overlay_height))
        style_axes(overlay)
        overlay.set_xlabel("Time (µs)")
        overlay.set_ylabel("Asymmetry")
        a_runs = _runs_by_number(self._a)
        b_runs = _runs_by_number(self._b)
        label_targets: list[float] = []
        for dataset, colour in zip(self._datasets, self._colours, strict=True):
            time = np.asarray(dataset.time, dtype=float)
            asymmetry = np.asarray(dataset.asymmetry, dtype=float)
            step = preview_stride(time.size, DISPLAY_POINTS_PER_RUN)
            overlay.plot(
                time[::step],
                asymmetry[::step],
                linestyle="none",
                marker="o",
                markersize=2.4,
                markeredgewidth=0,
                alpha=0.35,
                color=colour,
            )
            run_number = int(dataset.run_number)
            if run_number in b_runs:
                overlay.plot(
                    *b_runs[run_number].curves.fit, color=colour, linewidth=1.3, linestyle=_B_DASH
                )
            if run_number in a_runs:
                a_time, a_value = a_runs[run_number].curves.fit
                overlay.plot(a_time, a_value, color=colour, linewidth=1.7)
                label_targets.append(float(a_value[-1]))
            else:
                tail = asymmetry[-max(1, asymmetry.size // 10) :]
                label_targets.append(float(np.mean(tail)))
        if self._datasets:
            overlay.set_xlim(
                min(float(np.min(d.time)) for d in self._datasets),
                max(float(np.max(d.time)) for d in self._datasets),
            )
            y_low, y_high = overlay.get_ylim()
            fractions = [(target - y_low) / (y_high - y_low) for target in label_targets]
            for label, colour, fraction in zip(
                self._labels,
                self._colours,
                spread_labels(fractions, _LABEL_PX / overlay_height, 0.0, 1.0),
                strict=True,
            ):
                overlay.text(
                    1.01,
                    fraction,
                    label,
                    transform=overlay.transAxes,
                    color=colour,
                    fontsize=_LABEL_FONT_SIZE,
                    va="center",
                    ha="left",
                )

        if strips:
            top = _BOTTOM_PX + residual_px
            self._figure.text(
                _LEFT_PX / width,
                (top - _RESIDUAL_HEADER_PX / 2) / height,
                f"Normalised residuals (±{RESIDUAL_CLIP_SIGMA:g}σ)",
                fontsize=_LABEL_FONT_SIZE,
                fontweight="bold",
                color=tokens.TEXT,
                va="center",
            )
            self._figure.text(
                1 - _RIGHT_PX / width + 0.005,
                (top - _RESIDUAL_HEADER_PX / 2) / height,
                "χ²ᵣ A · B" if self._b is not None else "χ²ᵣ",
                fontsize=_LABEL_FONT_SIZE,
                color=tokens.TEXT_MUTED,
                va="center",
            )
            strip_top = top - _RESIDUAL_HEADER_PX
            for position, index in enumerate(strips):
                bottom = strip_top - (position + 1) * (strip_height + _STRIP_GAP_PX)
                axes = self._figure.add_axes(rect(bottom, strip_height), sharex=overlay)
                self._draw_strip(axes, index, a_runs, b_runs)
            if len(strips) < len(self._datasets):
                self._figure.text(
                    _LEFT_PX / width,
                    (_BOTTOM_PX + _NOTE_PX / 2) / height,
                    f"+{len(self._datasets) - len(strips)} more runs",
                    fontsize=_LABEL_FONT_SIZE,
                    color=tokens.TEXT_MUTED,
                    va="center",
                )
        self._canvas.draw_idle()

    def _draw_strip(
        self,
        axes: Axes,
        index: int,
        a_runs: dict[int, RunFit],
        b_runs: dict[int, RunFit],
    ) -> None:
        run_number = int(self._datasets[index].run_number)
        axes.set_axis_off()
        axes.set_ylim(-RESIDUAL_CLIP_SIGMA * 1.1, RESIDUAL_CLIP_SIGMA * 1.1)
        draw_zero_line(axes)
        for runs, colour, zorder in (
            (a_runs, self._colours[index], 3),
            (b_runs, tokens.TEXT_DIM, 2),
        ):
            if run_number not in runs:
                continue
            time, residual = runs[run_number].curves.residuals
            step = preview_stride(time.size, DISPLAY_POINTS_PER_RUN)
            axes.plot(
                time[::step],
                np.clip(residual[::step], -RESIDUAL_CLIP_SIGMA, RESIDUAL_CLIP_SIGMA),
                linestyle="none",
                marker="o",
                markersize=1.8,
                markeredgewidth=0,
                color=colour,
                zorder=zorder,
            )
        # A run a pre-screen never fitted has no χ²ᵣ.
        chi_texts = [
            format_reduced_chi_squared(runs[run_number].reduced_chi_squared)
            if run_number in runs
            else "—"
            for runs in ((a_runs, b_runs) if self._b is not None else (a_runs,))
        ]
        axes.text(
            -0.01,
            0.5,
            self._labels[index],
            transform=axes.transAxes,
            fontsize=_LABEL_FONT_SIZE,
            color=tokens.PLOT_TICK_LABEL,
            ha="right",
            va="center",
        )
        axes.text(
            1.01,
            0.5,
            " · ".join(chi_texts),
            transform=axes.transAxes,
            fontsize=_LABEL_FONT_SIZE,
            color=tokens.PLOT_TICK_LABEL,
            ha="left",
            va="center",
        )


def _runs_by_number(summary: CandidateSummary | None) -> dict[int, RunFit]:
    return {} if summary is None else {run.run_number: run for run in summary.runs}
