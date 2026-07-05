"""Window-agnostic answer card for the *global* (series) fit wizard.

Where :mod:`asymmetry.gui.widgets.wizard_answer_card` is the answer-first surface
for a single spectrum, this card is its counterpart for a whole ordered *series*:
one shared model fitted across many datasets (runs at different fields or
temperatures), with some parameters shared "Global" and others left per-run
"Local". The card leads with a plain verdict headline and confidence line, then a
hero figure that overlays every run's data and fit, and — when the host supplies
one — a "local parameter trend" panel that shows what the global fit bought you.
A primary Apply button and an alternatives strip round it out.

It is deliberately window- and core-agnostic. The widget imports nothing from
``asymmetry.core``: the host adapts its recommendation objects (e.g.
``GlobalCandidateAssessment`` / ``GlobalFitWizardRecommendation``) into the plain
:class:`SeriesRunTrace` / :class:`SeriesTrend` records below, and owns what
"apply" means — the card only emits :attr:`apply_requested`. Keeping the contract
plain-data keeps the card testable in isolation and reusable across host windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    build_primary_button_qss,
    build_segmented_button_qss,
    make_confidence_chip,
)

#: Okabe-Ito trace colours cycled when the series axis cannot grade the runs.
_FALLBACK_TRACE_COLOURS = (
    tokens.TRACE_BLUE,
    tokens.TRACE_GREEN,
    tokens.TRACE_ORANGE,
    tokens.TRACE_MAGENTA,
    tokens.TRACE_SKY,
    tokens.TRACE_VERMILLION,
)

#: Viridis sample range — the top end is too light on a white surface, so grade
#: only across the darker/mid band.
_VIRIDIS_LO = 0.10
_VIRIDIS_HI = 0.85

#: Legend is only drawn for a series small enough to read at a glance.
_MAX_LEGEND_RUNS = 8


@dataclass(frozen=True)
class SeriesRunTrace:
    """One dataset's data + fit overlay in the series hero figure.

    ``axis_value`` is the run's position along the series axis (field /
    temperature); ``None`` means the run cannot be graded, so the card falls back
    to a cycled palette. ``fitted_time`` / ``fitted_curve`` are optional — a run
    with no converged fit still draws its data.
    """

    run_label: str
    axis_value: float | None
    time: np.ndarray
    asymmetry: np.ndarray
    error: np.ndarray | None
    fitted_time: np.ndarray | None
    fitted_curve: np.ndarray | None


@dataclass(frozen=True)
class SeriesTrend:
    """A local parameter's value across the series — the trend panel's data."""

    parameter_label: str
    axis_label: str
    axis_values: tuple[float, ...]
    values: tuple[float, ...]
    errors: tuple[float, ...] | None


class WizardSeriesCard(QWidget):
    """Answer-first card for a global fit: verdict + series overlay + trend + apply."""

    #: Emitted when Apply is pressed; the host decides what applying means.
    apply_requested = Signal()
    #: Emitted with the selected alternative-candidate key when it changes.
    selection_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._runs: list[SeriesRunTrace] = []
        self._axis_label: str = ""
        self._trend: SeriesTrend | None = None
        self._selected_key: str | None = None
        self._alt_buttons: dict[str, QPushButton] = {}
        self._chip: QLabel | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Card chrome — the result-box frame owns the tint; children never inherit it.
        self._frame = QFrame(self)
        self._frame.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._frame.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        outer.addWidget(self._frame)

        layout = QVBoxLayout(self._frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Header row: headline + chip + stretch.
        self._header_row = QHBoxLayout()
        self._verdict_label = QLabel("", self._frame)
        self._verdict_label.setWordWrap(True)
        verdict_font = self._verdict_label.font()
        verdict_font.setPointSize(max(verdict_font.pointSize() + 3, 14))
        verdict_font.setBold(True)
        self._verdict_label.setFont(verdict_font)
        self._header_row.addWidget(self._verdict_label)
        self._header_row.addStretch()
        layout.addLayout(self._header_row)

        # Confidence prose line (hidden when empty).
        self._confidence_label = QLabel("", self._frame)
        self._confidence_label.setWordWrap(True)
        self._confidence_label.setVisible(False)
        layout.addWidget(self._confidence_label)

        # Hero figure.
        self._plot_widget = self._build_plot_widget()
        layout.addWidget(self._plot_widget, 1)

        # Alternatives strip.
        self._alternatives_row = QHBoxLayout()
        self._alternatives_label = QLabel("Alternatives:", self._frame)
        self._alternatives_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._alternatives_row.addWidget(self._alternatives_label)
        self._alternatives_row.addStretch()
        self._alternatives_container = QWidget(self._frame)
        self._alternatives_container.setLayout(self._alternatives_row)
        self._alternatives_container.setVisible(False)
        layout.addWidget(self._alternatives_container)

        # Apply row: primary-styled button + stretch.
        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply recommended fit", self._frame)
        self._apply_btn.setStyleSheet(build_primary_button_qss())
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        apply_row.addWidget(self._apply_btn)
        apply_row.addStretch()
        layout.addLayout(apply_row)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_verdict(self, headline: str, confidence_text: str, tier: str | None) -> None:
        """Set the headline, confidence line, chip, and card frame tint.

        ``tier`` is one of ``"high"``, ``"medium"``, ``"none"`` or ``None``.
        ``None`` shows no chip at all; ``"none"`` shows a muted chip. The frame
        goes green (:data:`RESULT_BOX_SUCCESS_STYLE`) only for ``"high"`` — every
        other tier stays neutral.
        """
        self._verdict_label.setText(headline)
        self._confidence_label.setText(confidence_text)
        self._confidence_label.setVisible(bool(confidence_text))
        self._rebuild_chip(confidence_text, tier)
        style = RESULT_BOX_SUCCESS_STYLE if tier == "high" else RESULT_BOX_NEUTRAL_STYLE
        self._frame.setStyleSheet(style)

    def set_series(self, runs: Sequence[SeriesRunTrace], axis_label: str) -> None:
        """Provide the ordered run traces and the series axis label."""
        self._runs = list(runs)
        self._axis_label = axis_label
        self._redraw()

    def set_trend(self, trend: SeriesTrend | None) -> None:
        """Attach (or clear) the local-parameter trend panel."""
        self._trend = trend
        self._redraw()

    def set_alternatives(self, items: Sequence[tuple[str, str, str]]) -> None:
        """Populate the alternatives strip from ``(key, label, tooltip)`` triples."""
        self._clear_alternatives()
        for key, label, tooltip in items:
            button = QPushButton(label, self._alternatives_container)
            button.setCheckable(True)
            if tooltip:
                button.setToolTip(tooltip)
            button.setStyleSheet(build_segmented_button_qss(padding_h=8))
            button.clicked.connect(lambda _checked=False, k=key: self.set_selected_key(k))
            self._alternatives_row.insertWidget(self._alternatives_row.count() - 1, button)
            self._alt_buttons[key] = button
        self._alternatives_container.setVisible(bool(self._alt_buttons))
        self._sync_alternative_styles()

    def set_selected_key(self, key: str | None) -> None:
        """Select ``key``, syncing chip checked states; emits only when changed."""
        if key == self._selected_key:
            return
        self._selected_key = key
        self._sync_alternative_styles()
        if isinstance(key, str):
            self.selection_changed.emit(key)

    def selected_key(self) -> str | None:
        return self._selected_key

    def set_apply_text(self, text: str) -> None:
        self._apply_btn.setText(text)

    def set_apply_enabled(self, enabled: bool) -> None:
        self._apply_btn.setEnabled(enabled)

    def clear(self) -> None:
        """Reset to an empty state: no verdict, no chip, cleared figure, no alternatives."""
        self._runs = []
        self._axis_label = ""
        self._trend = None
        self._selected_key = None
        self._verdict_label.setText("")
        self._confidence_label.setText("")
        self._confidence_label.setVisible(False)
        self._rebuild_chip("", None)
        self._frame.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._clear_alternatives()
        self._redraw()

    # ── Confidence chip ────────────────────────────────────────────────────

    def _rebuild_chip(self, text: str, tier: str | None) -> None:
        """Rebuild the header chip in place (chip colours bake at construction)."""
        if self._chip is not None:
            self._header_row.removeWidget(self._chip)
            self._chip.setParent(None)
            self._chip.deleteLater()
            self._chip = None
        if tier is None:
            return
        chip = make_confidence_chip(text, tier)  # type: ignore[arg-type]
        # Insert after the headline, before the trailing stretch.
        self._header_row.insertWidget(1, chip)
        self._chip = chip

    # ── Alternatives strip ─────────────────────────────────────────────────

    def _clear_alternatives(self) -> None:
        for button in self._alt_buttons.values():
            self._alternatives_row.removeWidget(button)
            button.setParent(None)
            button.deleteLater()
        self._alt_buttons.clear()
        self._alternatives_container.setVisible(False)

    def _sync_alternative_styles(self) -> None:
        for key, button in self._alt_buttons.items():
            button.setChecked(key == self._selected_key)

    # ── Apply ──────────────────────────────────────────────────────────────

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit()

    # ── Hero figure ────────────────────────────────────────────────────────

    def _build_plot_widget(self) -> QWidget:
        container = QWidget(self._frame)
        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            figure, canvas = create_canvas(layout="tight")
            # The card lives inside a scroll page, where the canvas would
            # otherwise collapse to its tiny sizeHint; the hero figure needs
            # real height to read.
            canvas.setMinimumHeight(340)
            self._figure = figure
            self._canvas = canvas
            inner.addWidget(canvas)
        except ImportError:
            self._figure = None
            self._canvas = None
            fallback = QLabel("matplotlib not available — plot preview disabled", container)
            fallback.setWordWrap(True)
            inner.addWidget(fallback)
        return container

    def _trace_colours(self) -> list[str]:
        """One colour per run: viridis-graded along the axis, else cycled palette.

        Grading needs every run to carry an ``axis_value`` and a non-degenerate
        range; otherwise the six Okabe-Ito trace colours cycle by index.
        """
        values = [run.axis_value for run in self._runs]
        if values and all(v is not None for v in values):
            floats = [float(v) for v in values]  # type: ignore[arg-type]
            lo, hi = min(floats), max(floats)
            if hi > lo:
                from matplotlib import colormaps

                cmap = colormaps["viridis"]
                span = _VIRIDIS_HI - _VIRIDIS_LO
                colours = []
                for value in floats:
                    frac = _VIRIDIS_LO + span * (value - lo) / (hi - lo)
                    rgba = cmap(frac)
                    colours.append((rgba[0], rgba[1], rgba[2]))
                return colours  # type: ignore[return-value]
        return [
            _FALLBACK_TRACE_COLOURS[i % len(_FALLBACK_TRACE_COLOURS)]
            for i in range(len(self._runs))
        ]

    def _redraw(self) -> None:
        figure = getattr(self, "_figure", None)
        canvas = getattr(self, "_canvas", None)
        if figure is None or canvas is None:
            return
        figure.clear()

        if self._trend is not None:
            grid = figure.add_gridspec(1, 2, width_ratios=[2.2, 1])
            ax_overlay = figure.add_subplot(grid[0, 0])
            ax_trend = figure.add_subplot(grid[0, 1])
        else:
            ax_overlay = figure.add_subplot(1, 1, 1)
            ax_trend = None

        self._draw_overlay(ax_overlay)
        if ax_trend is not None:
            self._draw_trend(ax_trend)

        canvas.draw_idle()

    def _draw_overlay(self, ax) -> None:
        colours = self._trace_colours()
        for run, colour in zip(self._runs, colours):
            yerr = run.error if run.error is not None else None
            ax.errorbar(
                run.time,
                run.asymmetry,
                yerr=yerr,
                fmt=".",
                markersize=2.5,
                alpha=0.75,
                color=colour,
                label=run.run_label,
            )
            if run.fitted_curve is not None and run.fitted_time is not None:
                ax.plot(run.fitted_time, run.fitted_curve, linewidth=1.6, color=colour)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Asymmetry")
        if self._runs and len(self._runs) <= _MAX_LEGEND_RUNS:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def _draw_trend(self, ax) -> None:
        trend = self._trend
        if trend is None:
            return
        yerr = trend.errors if trend.errors is not None else None
        ax.errorbar(
            trend.axis_values,
            trend.values,
            yerr=yerr,
            fmt="o-",
            markersize=4,
            color=tokens.FIT,
        )
        ax.set_xlabel(trend.axis_label)
        ax.set_ylabel(trend.parameter_label)
        ax.set_title("Local parameter trend")
