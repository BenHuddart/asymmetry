"""The wizards' comparison workspace: pick candidate A, pin B, read them side by side.

Left, the candidates grouped by title in rank order, each with its Global/Local
chips, a Δ(metric) bar, its evidence weight, a gate badge and flag lines. Right,
the series overlay (A solid, B dashed) over residual strips, the A-vs-B
parameter table, and one local parameter's trend along the series. Every
judgement shown comes from ``core/fitting/model_comparison.py``. The global
wizard's series has N runs; the single-run wizard's has one, with no role chips
and no trend. Design: ``docs/plans/global-wizard-stepper.md`` (D4, D8, D9) and
``docs/plans/fit-wizard-compare.md`` (D1–D3).

Under the Global Fit Wizard's trend objective the candidates are rungs of
sharing ladders (:meth:`ModelComparePanel.set_ladders`): each row shows its cost
and trend quality in place of Δ and weight, each model folds to its
pre-selected rung, and the trend plot becomes a strip of every local
parameter's trace. Design: ``docs/plans/global-wizard-trend-objective.md``
(D9, Phase 5).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.global_search.sharing_ladder import ADEQUACY_SIGMA
from asymmetry.core.fitting.global_search.trend_objective import RungSummary, cost_text
from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    Estimate,
    ParameterFlag,
    ParameterPair,
    ParameterRole,
    ParameterRow,
    RunFit,
    compare_parameters,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import char_width, row_height
from asymmetry.gui.styles.plots import (
    draw_empty_state_message,
    style_axes,
    style_figure,
    style_legend,
)
from asymmetry.gui.styles.typography import header_font
from asymmetry.gui.styles.widgets import build_primary_button_qss, clear_layout
from asymmetry.gui.utils.errorbar_dots import add_errorbar_dots
from asymmetry.gui.utils.formatting import format_param_label, format_value_uncertainty
from asymmetry.gui.widgets.elided_label import ElidedLabel
from asymmetry.gui.widgets.flow_layout import FlowLayout
from asymmetry.gui.widgets.mpl_canvas import create_canvas
from asymmetry.gui.widgets.parameter_trace_strip import (
    ParameterTraceStrip,
    TracedRung,
    rung_traces,
)
from asymmetry.gui.widgets.screening_leaderboard import format_delta, paint_delta_bar
from asymmetry.gui.widgets.series_fit_canvas import SeriesFitCanvas

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from asymmetry.core.data.dataset import MuonDataset

#: A sub-row shows at most this many flag lines; the rest sit behind a "+k more" line.
MAX_FLAG_LINES = 3
NO_B_LEGEND = "Pin another row as B to overlay it dashed"
NO_LOCAL_TREND = "A shares every parameter across the series"
EMPTY_BOARD = "No optimised candidates yet"
CONTINUE_WITH_A = "Continue with A →"
LADDER_CAPTION = "Sharing ladders · cost against every parameter local · trend of the worst one"
RECOMMENDED = "Recommended"
PRESELECTED = "Pre-selected"
COSTS_TOO_MUCH = "Costs too much"
FIT_FAILED = "Fit failed"
#: The trace strip's height floor, in table rows.
_STRIP_MIN_ROWS = 9
#: The cost meter is full at this many tolerances, so the tolerance mark sits mid-track.
_COST_METER_TOLERANCES = 2.0

_ROLE_SUFFIX = {
    ParameterRole.GLOBAL: "shared",
    ParameterRole.LOCAL: "per run",
    ParameterRole.FITTED: "fitted",
    ParameterRole.FIXED: "fixed",
}
_CHIP_COLOURS = {
    ParameterRole.GLOBAL: (tokens.ACCENT_SOFT, tokens.ACCENT),
    ParameterRole.LOCAL: (tokens.WARN_BANNER_BG, tokens.WARN_BANNER_TEXT),
}
_GATE_BADGES = {
    True: ("Pass", tokens.SUCCESS_SOFT, tokens.OK),
    False: ("Warn", tokens.WARN_SOFT, tokens.WARN_BANNER_TEXT),
}
#: Row frame (background, border, border style) per slot; B is dashed like its curve.
_ROW_LOOKS = {
    "A": (tokens.ACCENT_SOFT, tokens.ACCENT, "solid"),
    "B": (tokens.SURFACE_ALT, tokens.BORDER_STRONG, "dashed"),
    "": (tokens.SURFACE, tokens.BORDER, "solid"),
}
_PARAMETER_COLUMN, _A_COLUMN, _B_COLUMN, _NOTE_COLUMN = range(4)
_TREND_COLOUR = tokens.PLOT_DATA


def _symbol(name: str) -> str:
    return format_param_label(name, include_unit=False)


def format_weight(weight: float) -> str:
    """An evidence weight as a whole percentage; a sliver of support reads ``"<1%"``."""
    return "<1%" if 0.0 < weight < 0.005 else f"{weight:.0%}"


def format_cost(cost: float) -> str:
    """A rung's cost in standard deviations of χ²ᵣ; whole numbers once it is far over."""
    return f"{cost:+.1f}σ" if abs(cost) < 100.0 else f"{cost:+.0f}σ"


def split_text(summary: CandidateSummary) -> str:
    """``"shares Δ, A_bg"`` with globals, ``"all per run"`` with locals only, else empty."""
    shared = summary.names(ParameterRole.GLOBAL)
    if shared:
        return "shares " + ", ".join(_symbol(name) for name in shared)
    return "all per run" if summary.names(ParameterRole.LOCAL) else ""


def headline(summary: CandidateSummary) -> str:
    """The title, then the role split when the candidate has one."""
    return " · ".join(part for part in (summary.title, split_text(summary)) if part)


def flag_lines(summary: CandidateSummary, run_labels: Mapping[int, str]) -> list[str]:
    """One line per parameter flag, naming the runs that earned it.

    A shared global's flag, or any flag of a single-run candidate, needs no run;
    a flag every run of several earned reads "at every run".
    """
    lines = []
    for row in summary.parameters:
        for flag in row.flags:
            runs = [
                run for run, flags in zip(summary.runs, row.run_flags, strict=True) if flag in flags
            ]
            if isinstance(row.values, Estimate) or len(summary.runs) == 1:
                where = ""
            elif len(runs) == len(summary.runs) > 1:
                where = " at every run"
            else:
                where = " at " + ", ".join(run_labels[run.run_number] for run in runs)
            lines.append(f"{_symbol(row.name)} {flag.value}{where}")
    return lines


def _chip(text: str, background: str, foreground: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"QLabel {{ background: {background}; color: {foreground};"
        " border-radius: 4px; padding: 1px 6px; }"
    )
    return label


class _TagDisc(QWidget):
    """A disc naming the row's slot: filled with "A" or "B", an outline otherwise."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.letter = ""
        side = row_height()
        self.setFixedSize(side, side)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        disc = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        if self.letter:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tokens.TEXT))
            painter.drawEllipse(disc)
            font = self.font()
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(tokens.WHITE))
            painter.drawText(disc, Qt.AlignmentFlag.AlignCenter, self.letter)
        else:
            painter.setPen(QPen(QColor(tokens.BORDER_STRONG), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(disc)


class _DeltaBar(QWidget):
    """The leaderboard's Δ bar, as a widget."""

    def __init__(self, delta: float, metric_label: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._delta = delta
        self.setToolTip(f"{metric_label}: {format_delta(delta)}")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(char_width(18), row_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(char_width(10), row_height())

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        paint_delta_bar(painter, QRectF(self.rect()), self._delta, self.fontMetrics())


class Meter(QWidget):
    """A track filled to ``fraction``, its value in words beside it, an optional mark across it."""

    def __init__(
        self,
        fraction: float,
        text: str,
        colour: str,
        parent: QWidget,
        *,
        mark: float | None = None,
    ) -> None:
        super().__init__(parent)
        self.fraction = min(max(fraction, 0.0), 1.0)
        self.text = text
        self.colour = colour
        self._mark = mark

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(char_width(14), row_height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(char_width(10), row_height())

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = QRectF(self.rect())
        text_width = self.fontMetrics().horizontalAdvance("+000.0σ")
        track = QRectF(area.left(), area.center().y() - 3, area.width() - text_width - 6, 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens.SURFACE_HI))
        painter.drawRoundedRect(track, 3, 3)
        painter.setBrush(QColor(self.colour))
        painter.drawRoundedRect(track.adjusted(0, 0, -(1 - self.fraction) * track.width(), 0), 3, 3)
        if self._mark is not None:
            x = track.left() + self._mark * track.width()
            painter.setPen(QPen(QColor(tokens.TEXT_MUTED), 1.2))
            painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
        painter.setPen(QColor(tokens.TEXT_MUTED))
        painter.drawText(
            area, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self.text
        )


class CompareRow(QFrame):
    """One candidate: its slot disc, role chips and pin. A click, Enter or Space picks it as A.

    A subclass adds its scores with :meth:`_add_scores` and its warnings with
    :meth:`_add_flag_lines`. ``slots=False`` is a row outside an A/B comparison:
    it has no disc and no pin, and :meth:`set_slot` only frames it as picked.
    """

    picked = Signal(str)

    def __init__(
        self,
        summary: CandidateSummary,
        parent: QWidget,
        *,
        leading_chips: Sequence[QLabel] = (),
        slots: bool = True,
    ) -> None:
        super().__init__(parent)
        self.key = summary.key
        self._slots = slots
        self.setObjectName("compareRow")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(headline(summary))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        self.disc = _TagDisc(self)
        self.disc.setVisible(slots)
        layout.addWidget(self.disc, 0, Qt.AlignmentFlag.AlignTop)
        self._body = QVBoxLayout()
        self._body.setSpacing(4)
        layout.addLayout(self._body, 1)

        top = QHBoxLayout()
        chips = QWidget(self)
        flow = FlowLayout(chips)
        flow.setContentsMargins(0, 0, 0, 0)
        self.chips: list[QLabel] = [
            _chip(f"{role.value} {_symbol(name)}", *_CHIP_COLOURS[role])
            for role, names in (
                (ParameterRole.GLOBAL, summary.names(ParameterRole.GLOBAL)),
                (ParameterRole.LOCAL, summary.names(ParameterRole.LOCAL)),
            )
            for name in names
        ]
        for chip in (*leading_chips, *self.chips):
            flow.addWidget(chip)
        top.addWidget(chips, 1)
        self.pin_button = QPushButton(self)
        self.pin_button.setFlat(True)
        self.pin_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.pin_button.setStyleSheet(
            f"QPushButton {{ color: {tokens.ACCENT}; border: none; padding: 0 2px; }}"
        )
        top.addWidget(self.pin_button, 0, Qt.AlignmentFlag.AlignTop)
        self._body.addLayout(top)

        fixed_names = summary.names(ParameterRole.FIXED)
        if fixed_names:
            fixed = QLabel("Fixed: " + ", ".join(_symbol(name) for name in fixed_names), self)
            fixed.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            self._body.addWidget(fixed)
        self.flag_labels: list[ElidedLabel] = []
        self.set_slot("")

    def _add_scores(self, widgets: Sequence[tuple[QWidget, int]]) -> None:
        """A line of ``(widget, stretch)`` scores under the chips."""
        scores = QHBoxLayout()
        scores.setSpacing(8)
        for widget, stretch in widgets:
            scores.addWidget(widget, stretch)
        self._body.addLayout(scores)

    def _add_flag_lines(self, lines: Sequence[tuple[str, str]]) -> None:
        """``(line, hover text)`` warnings; past ``MAX_FLAG_LINES`` they fold into "+k more"."""
        for line, hover in lines[:MAX_FLAG_LINES]:
            label = ElidedLabel(line, self)
            label.set_hover_text(hover)
            label.set_pen_color(tokens.WARN)
            self.flag_labels.append(label)
        if len(lines) > MAX_FLAG_LINES:
            rest = lines[MAX_FLAG_LINES:]
            more = ElidedLabel(f"+{len(rest)} more", self)
            more.set_hover_text("\n".join(hover for _line, hover in rest))
            more.set_pen_color(tokens.TEXT_MUTED)
            self.flag_labels.append(more)
        for label in self.flag_labels:
            self._body.addWidget(label)

    def set_slot(self, letter: str) -> None:
        """Show the row as candidate ``"A"``, ``"B"`` or neither (``""``)."""
        self.disc.letter = letter
        self.disc.update()
        self.pin_button.setVisible(self._slots and letter != "A")
        self.pin_button.setText("Unpin B" if letter == "B" else "Pin as B")
        background, border, style = _ROW_LOOKS[letter]
        self.setStyleSheet(
            f"#compareRow {{ background: {background}; border: 1px {style} {border};"
            f" border-radius: 6px; }}"
            f" #compareRow:focus {{ border: 1px {style} {tokens.ACCENT}; }}"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.picked.emit(self.key)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.picked.emit(self.key)
            return
        super().keyPressEvent(event)


class CandidateRow(CompareRow):
    """A candidate ranked by an information criterion: Δ, evidence weight and gate."""

    def __init__(
        self,
        summary: CandidateSummary,
        metric_label: str,
        run_labels: Mapping[int, str],
        parent: QWidget,
    ) -> None:
        super().__init__(summary, parent)
        self.delta_bar = _DeltaBar(summary.delta, metric_label, self)
        # An unranked candidate (Δ "—") has no share of the evidence to show.
        self.weight = QLabel(
            format_weight(summary.weight) if math.isfinite(summary.delta) else "—", self
        )
        self.weight.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self.weight.setToolTip("Evidence weight: w ∝ exp(−Δ/2) across the candidates listed")
        if summary.gate_passed is None:
            # No residual gate was run (a saved fit): its χ²ᵣ is the verdict to hand.
            (run,) = summary.runs
            chi2 = run.reduced_chi_squared
            self.gate = QLabel(f"χ²ᵣ {chi2:.3g}" if math.isfinite(chi2) else "χ²ᵣ —", self)
            self.gate.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        else:
            text, background, foreground = _GATE_BADGES[summary.gate_passed]
            self.gate = _chip(text, background, foreground)
            self.gate.setToolTip(summary.gate_summary or "Every run passes the residual gate")
        self._add_scores([(self.delta_bar, 1), (self.weight, 0), (self.gate, 0)])
        self._add_flag_lines(_warning_lines(summary, run_labels))


class RungRow(CompareRow):
    """A rung of a sharing ladder: what it costs, how what it leaves local trends, what was found.

    A rung that costs too much says so and stays pickable (plan D9).
    """

    def __init__(
        self,
        summary: RungSummary,
        run_labels: Mapping[int, str],
        parent: QWidget,
        *,
        recommended: bool,
        slots: bool = True,
    ) -> None:
        rung = summary.rung
        cost_sentence = cost_text(rung, run_labels)
        if recommended:
            mark = _chip(RECOMMENDED, tokens.SUCCESS_SOFT, tokens.OK)
            mark.setToolTip(
                "The rung the wizard recommends: the best trend among the models that fit."
            )
        elif not summary.converged:
            mark = _chip(FIT_FAILED, tokens.ERROR_SOFT, tokens.ERROR)
            mark.setToolTip("The coupled fit did not converge on every run.")
        elif not rung.within_tolerance:
            mark = _chip(COSTS_TOO_MUCH, tokens.WARN_SOFT, tokens.WARN_BANNER_TEXT)
            mark.setToolTip(cost_sentence)
        elif rung.preselected:
            mark = _chip(PRESELECTED, tokens.ACCENT_SOFT, tokens.ACCENT)
            mark.setToolTip("This model's rung that trends best among those that fit adequately.")
        else:
            mark = None
        super().__init__(
            summary, parent, leading_chips=() if mark is None else (mark,), slots=slots
        )
        #: The chip that says where the rung stands, or ``None`` for an adequate also-ran.
        self.mark = mark

        self.cost = Meter(
            rung.series_cost / (_COST_METER_TOLERANCES * ADEQUACY_SIGMA),
            format_cost(rung.series_cost),
            tokens.BORDER_STRONG if rung.within_tolerance else tokens.WARN,
            self,
            mark=1.0 / _COST_METER_TOLERANCES,
        )
        self.cost.setToolTip(cost_sentence)
        is_trend, worst, mean = rung.trend.ordering_key
        self.trend = Meter(worst, f"{worst:.2f}" if is_trend else "—", tokens.ACCENT, self)
        self.trend.setToolTip(
            "\n".join(
                [
                    f"Trend quality of the worst local parameter (mean {mean:.2f}); 1 is best.",
                    *(
                        f"{_symbol(name)}: {quality.quality:.2f}"
                        for name, quality in rung.trend.parameters.items()
                    ),
                ]
            )
            if is_trend
            else "Every parameter is shared: nothing is left to trend."
        )
        self._add_scores(
            [
                (_muted("Cost", self), 0),
                (self.cost, 1),
                (_muted("Trend", self), 0),
                (self.trend, 1),
            ]
        )
        self._add_flag_lines(
            [
                *((finding.label, finding.detail) for finding in summary.findings),
                *_warning_lines(summary, run_labels),
            ]
        )


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
    return label


def _warning_lines(
    summary: CandidateSummary, run_labels: Mapping[int, str]
) -> list[tuple[str, str]]:
    """The gate reasons, then the parameter flags; each line is its own hover text."""
    lines = [
        *([summary.gate_summary] if summary.gate_summary else []),
        *flag_lines(summary, run_labels),
    ]
    return [(line, line) for line in lines]


class _Fold(QWidget):
    """A ladder's rungs other than the one its model leads with, behind a toggle."""

    opened = Signal(bool)

    def __init__(self, rows: Sequence[CompareRow], parent: QWidget) -> None:
        super().__init__(parent)
        self._keys = {row.key for row in rows}
        self._noun = f"{len(rows)} other rung{'' if len(rows) == 1 else 's'}"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.toggle = QPushButton(self)
        self.toggle.setFlat(True)
        self.toggle.setCheckable(True)
        self.toggle.setStyleSheet(
            f"QPushButton {{ color: {tokens.ACCENT}; border: none; padding: 0 2px;"
            " text-align: left; }"
        )
        layout.addWidget(self.toggle)
        self._rows = QWidget(self)
        rows_layout = QVBoxLayout(self._rows)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(6)
        for row in rows:
            rows_layout.addWidget(row)
        layout.addWidget(self._rows)
        self.toggle.toggled.connect(self._show_rows)
        self._show_rows(False)

    def _show_rows(self, opened: bool) -> None:
        self._rows.setVisible(opened)
        self.toggle.setText(f"{'▾' if opened else '▸'} {self._noun}")
        self.opened.emit(opened)

    def open_for(self, keys: Sequence[str | None]) -> None:
        """Open when one of ``keys`` is a row in here, so a picked rung is never hidden."""
        if self._keys.intersection(keys):
            self.toggle.setChecked(True)


def _value_cell(
    row: ParameterRow | None, runs: Sequence[RunFit], labels: Mapping[int, str]
) -> QTableWidgetItem:
    """One side's value(err), per-run values joined by " · "; flagged values in WARN or ERROR."""
    if row is None:
        item = QTableWidgetItem("—")
        item.setToolTip("Not a parameter of this model")
        item.setForeground(QColor(tokens.TEXT_DIM))
        return item
    if isinstance(row.values, Estimate):
        text = format_value_uncertainty(row.values.value, row.values.error)
        tooltip = text + "".join(f" — {flag.value}" for flag in row.flags)
    else:
        values = [format_value_uncertainty(e.value, e.error) for e in row.values]
        text = " · ".join(values)
        tooltip = "\n".join(
            f"{labels[run.run_number]}: {value}" + "".join(f" — {flag.value}" for flag in flags)
            for run, value, flags in zip(runs, values, row.run_flags, strict=True)
        )
    item = QTableWidgetItem(text)
    item.setToolTip(tooltip)
    if ParameterFlag.NOT_FINITE in row.flags:
        item.setForeground(QColor(tokens.ERROR))
    elif row.flags:
        item.setForeground(QColor(tokens.WARN))
    return item


class ModelComparePanel(QWidget):
    """Candidates on the left; A against B on the right; the host's A action below.

    Call :meth:`set_series` before :meth:`set_candidates` or :meth:`set_ladders`:
    every candidate run must be one of the series' runs, which name the runs in
    flag lines and tooltips. ``set_candidates`` lists candidates ranked by an
    information criterion; ``set_ladders`` lists rungs of sharing ladders.
    ``continue_text`` labels the footer button that emits :attr:`continue_requested`.
    :attr:`curves_required` names an A or B whose dense curves are not built yet;
    the host builds them off the GUI thread and answers with :meth:`refresh_curves`.
    """

    a_changed = Signal(str)
    b_changed = Signal(object)
    continue_requested = Signal(str)
    curves_required = Signal(str)

    def __init__(
        self, parent: QWidget | None = None, *, continue_text: str = CONTINUE_WITH_A
    ) -> None:
        super().__init__(parent)
        self._summaries: dict[str, CandidateSummary] = {}
        self._metric_label = ""
        self._a: str | None = None  # None only while there are no candidates
        self._b: str | None = None
        self._run_labels: dict[int, str] = {}
        self._axis_label = ""
        self._trend: str | None = None
        self._pairs: tuple[ParameterPair, ...] = ()
        self._rows: dict[str, CompareRow] = {}
        self._continue_allowed = True
        # Whether the candidates are ladder rungs (set_ladders) or ranked rows (set_candidates).
        self._ladders = False
        self._recommended: str | None = None
        # Titles of the ladders the user unfolded, kept across rebuilds.
        self._unfolded: set[str] = set()
        self._folds: list[_Fold] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        body = QHBoxLayout()
        body.setSpacing(12)
        layout.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(4)
        self._caption = QLabel(self)
        self._caption.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        left.addWidget(self._caption)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        board = QWidget(self._scroll)
        self._board = QVBoxLayout(board)
        self._board.setContentsMargins(0, 0, 4, 0)
        self._board.setSpacing(6)
        self._scroll.setWidget(board)
        left.addWidget(self._scroll, 1)
        # Under the ladders, where the board has height to spare: beside the
        # overlay, the traces would add to the tallest column's floor.
        self._strip = ParameterTraceStrip(self)
        self._strip.setMinimumHeight(_STRIP_MIN_ROWS * row_height())
        left.addWidget(self._strip)
        body.addLayout(left, 2)

        right = QVBoxLayout()
        right.setSpacing(4)
        self._legend_a = ElidedLabel("", self)
        self._legend_a.set_pen_color(tokens.TEXT)
        self._legend_b = ElidedLabel("", self)
        right.addWidget(self._legend_a)
        right.addWidget(self._legend_b)
        self._canvas = SeriesFitCanvas(self)
        self._canvas.curves_required.connect(self.curves_required)
        right.addWidget(self._canvas, 3)
        lower = QHBoxLayout()
        lower.setSpacing(8)
        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Parameter", "A", "B", ""])
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(row_height())
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self._table.horizontalHeader()
        header.setFont(header_font())
        header.setHighlightSections(False)
        header.setSectionResizeMode(_PARAMETER_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_A_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_B_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_NOTE_COLUMN, QHeaderView.ResizeMode.Interactive)
        self._table.setColumnWidth(_NOTE_COLUMN, char_width(16))
        self._table.cellClicked.connect(self._on_table_clicked)
        lower.addWidget(self._table, 3)
        self._trend_figure, self._trend_canvas = create_canvas(layout="tight")
        style_figure(self._trend_figure)
        lower.addWidget(self._trend_canvas, 2)
        right.addLayout(lower, 2)
        body.addLayout(right, 3)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._continue = QPushButton(continue_text, self)
        self._continue.setStyleSheet(build_primary_button_qss())
        self._continue.clicked.connect(lambda: self.continue_requested.emit(self._a))
        footer.addWidget(self._continue)
        layout.addLayout(footer)

        self._rebuild_board()
        self._show_pair()

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def trend_figure(self) -> Figure:
        return self._trend_figure

    @property
    def trace_strip(self) -> ParameterTraceStrip:
        return self._strip

    def set_series(
        self,
        datasets: Sequence[MuonDataset],
        run_labels: Sequence[str],
        axis_values: Sequence[float],
        axis_label: str,
        colours: Sequence[str] | None = None,
    ) -> None:
        """Show ``datasets`` in series order; ``axis_label`` titles the trend's x axis."""
        self._canvas.set_series(datasets, run_labels, axis_values, colours)
        self._run_labels = {
            int(dataset.run_number): label
            for dataset, label in zip(datasets, run_labels, strict=True)
        }
        self._axis_label = axis_label
        self._rebuild_board()
        self._show_pair()

    def set_candidates(
        self,
        summaries: Sequence[CandidateSummary],
        metric_label: str,
        *,
        a_key: str | None = None,
    ) -> None:
        """Show ``summaries`` in rank order, without emitting a signal.

        A is ``a_key``, or the first summary; B is kept while it is still listed
        and is not the new A.
        """
        self._metric_label = metric_label
        self._ladders = False
        self._list(summaries, a_key)

    def set_ladders(
        self,
        summaries: Sequence[RungSummary],
        *,
        recommended_key: str | None,
        a_key: str | None = None,
    ) -> None:
        """Show sharing-ladder rungs model by model, without emitting a signal.

        Each model leads with its first rung (its pre-selected one, in the
        wizard's order) and folds the others behind a toggle, so the default
        view is one answer per model and a ladder is read only when it is
        opened. ``recommended_key`` marks the rung the wizard recommends, or
        ``None`` when it recommends none. A and B are as for
        :meth:`set_candidates`.
        """
        self._ladders = True
        self._recommended = recommended_key
        self._list(summaries, a_key)

    def _list(self, summaries: Sequence[CandidateSummary], a_key: str | None) -> None:
        self._summaries = {summary.key: summary for summary in summaries}
        if a_key is not None and a_key not in self._summaries:
            raise KeyError(f"candidate A {a_key!r} is not among the summaries")
        self._a = a_key if a_key is not None else next(iter(self._summaries), None)
        if self._b not in self._summaries or self._b == self._a:
            self._b = None
        self._rebuild_board()
        self._show_pair()

    def refresh_curves(self, summaries: Sequence[CandidateSummary]) -> None:
        """Swap in the ranking already shown with more curves built; redraw A and B, not the board."""
        if [summary.key for summary in summaries] != list(self._summaries):
            raise ValueError("refresh_curves needs the ranking the panel already shows")
        self._summaries = {summary.key: summary for summary in summaries}
        self._show_pair()

    def a_key(self) -> str | None:
        """Candidate A's key; ``None`` only while there are no candidates."""
        return self._a

    def b_key(self) -> str | None:
        return self._b

    def set_a(self, key: str) -> None:
        """Make ``key`` candidate A; picking the current B as A clears B."""
        if key not in self._summaries:
            raise KeyError(f"candidate {key!r} is not in the Compare panel")
        if key == self._a:
            return
        cleared_b = key == self._b
        self._a = key
        if cleared_b:
            self._b = None
        self._show_pair()
        self.a_changed.emit(key)
        if cleared_b:
            self.b_changed.emit(None)

    def set_b(self, key: str | None) -> None:
        """Pin ``key`` as candidate B, or clear B with ``None``."""
        if key == self._b:
            return
        if key is not None and key not in self._summaries:
            raise KeyError(f"candidate {key!r} is not in the Compare panel")
        if key is not None and key == self._a:
            raise ValueError("candidate B must differ from candidate A")
        self._b = key
        self._show_pair()
        self.b_changed.emit(key)

    def set_continue_enabled(self, enabled: bool) -> None:
        """Let the host hold Continue back, e.g. while A's fit has failed."""
        self._continue_allowed = enabled
        self._sync_continue()

    # ── Rendering ───────────────────────────────────────────────────────────

    def _a_and_b(self) -> tuple[CandidateSummary | None, CandidateSummary | None]:
        return (
            self._summaries[self._a] if self._a is not None else None,
            self._summaries[self._b] if self._b is not None else None,
        )

    def _sync_continue(self) -> None:
        self._continue.setEnabled(self._continue_allowed and self._a is not None)

    def _toggle_pin(self, key: str) -> None:
        self.set_b(None if key == self._b else key)

    def _remember_fold(self, title: str, opened: bool) -> None:
        (self._unfolded.add if opened else self._unfolded.discard)(title)

    def _rebuild_board(self) -> None:
        """Rebuild the leaderboard rows: one bold title line per template group."""
        noun = "Candidates" if len(self._run_labels) == 1 else "Role splits"
        self._caption.setText(LADDER_CAPTION if self._ladders else f"{noun} · {self._metric_label}")
        self._caption.setVisible(bool(self._summaries))
        # A lone run has nothing to trend along; ladders trace every local parameter.
        self._trend_canvas.setVisible(len(self._run_labels) > 1 and not self._ladders)
        self._strip.setVisible(self._ladders)
        clear_layout(self._board)
        self._rows = {}
        self._folds = []
        if not self._summaries:
            empty = QLabel(EMPTY_BOARD)
            empty.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            self._board.addWidget(empty)
        for title, group in groupby(self._summaries.values(), key=lambda s: s.title):
            heading = QLabel(title)
            heading.setStyleSheet(f"color: {tokens.TEXT}; font-weight: 600; padding-top: 4px;")
            self._board.addWidget(heading)
            parent = self._board.parentWidget()
            rows: list[CompareRow] = [
                RungRow(
                    summary, self._run_labels, parent, recommended=summary.key == self._recommended
                )
                if self._ladders
                else CandidateRow(summary, self._metric_label, self._run_labels, parent)
                for summary in group
            ]
            for row in rows:
                row.picked.connect(self.set_a)
                row.pin_button.clicked.connect(partial(self._toggle_pin, row.key))
                self._rows[row.key] = row
            # A ladder leads with one rung and folds the rest (see set_ladders).
            shown = rows[:1] if self._ladders else rows
            for row in shown:
                self._board.addWidget(row)
            if len(shown) < len(rows):
                fold = _Fold(rows[len(shown) :], self._board.parentWidget())
                fold.toggle.setChecked(title in self._unfolded)
                fold.opened.connect(partial(self._remember_fold, title))
                self._folds.append(fold)
                self._board.addWidget(fold)
        self._board.addStretch(1)

    def _show_pair(self) -> None:
        """Render everything that follows A and B: slots, legend, overlay, table, trend."""
        a, b = self._a_and_b()
        for key, row in self._rows.items():
            row.set_slot("A" if key == self._a else "B" if key == self._b else "")
        for fold in self._folds:
            fold.open_for((self._a, self._b))
        self._legend_a.setVisible(a is not None)
        self._legend_b.setVisible(a is not None)
        if a is not None:
            self._legend_a.setText(f"— A: {headline(a)}")
        if b is not None:
            self._legend_b.setText(f"- - B: {headline(b)}")
            self._legend_b.set_pen_color(tokens.TEXT)
        else:
            self._legend_b.setText(NO_B_LEGEND)
            self._legend_b.set_pen_color(tokens.TEXT_MUTED)
        self._canvas.set_curves(a, b)

        if a is None:
            self._pairs = ()
        elif b is None:
            self._pairs = tuple(ParameterPair(row.name, row, None) for row in a.parameters)
        else:
            self._pairs = compare_parameters(a, b)
        if self._trend not in self._trend_names():
            # Ladders trace every local parameter, so none is the picked one.
            local_names = () if a is None or self._ladders else a.names(ParameterRole.LOCAL)
            self._trend = local_names[0] if local_names else None
        self._table.setColumnHidden(_B_COLUMN, b is None)
        self._fill_table()
        self._draw_trend()
        self._sync_continue()

    def _trend_names(self) -> set[str]:
        """Parameters local to A or B, the ones the single trend plot can show."""
        if self._ladders:
            return set()
        return {
            pair.name
            for pair in self._pairs
            for row in (pair.a, pair.b)
            if row is not None and row.role is ParameterRole.LOCAL
        }

    def _fill_table(self) -> None:
        a, b = self._a_and_b()
        trend_names = self._trend_names()
        table = self._table
        table.setRowCount(len(self._pairs))
        for index, pair in enumerate(self._pairs):
            roles = {side.role for side in (pair.a, pair.b) if side is not None}
            if roles == {ParameterRole.FITTED}:
                label = _symbol(pair.name)  # a single-run fit: every free parameter is fitted
            elif len(roles) == 1:
                label = f"{_symbol(pair.name)} · {_ROLE_SUFFIX[roles.pop()]}"
            else:
                label = (
                    f"{_symbol(pair.name)} · {_ROLE_SUFFIX[pair.a.role]} in A,"
                    f" {_ROLE_SUFFIX[pair.b.role]} in B"
                )
            name = QTableWidgetItem(label)
            name.setToolTip(
                f"Click to plot {format_param_label(pair.name)} against the series"
                if pair.name in trend_names
                else format_param_label(pair.name)
            )
            # The pair's kσ difference, then each side's flags.
            note = QTableWidgetItem(
                " · ".join(
                    [
                        *([pair.difference_text] if pair.difference_text else []),
                        *(
                            f"{side}: " + ", ".join(flag.value for flag in side_row.flags)
                            for side, side_row in (("A", pair.a), ("B", pair.b))
                            if side_row is not None and side_row.flags
                        ),
                    ]
                )
            )
            note.setToolTip(note.text())
            note.setForeground(QColor(tokens.WARN))
            items = (
                name,
                _value_cell(pair.a, a.runs if a is not None else (), self._run_labels),
                _value_cell(pair.b, b.runs if b is not None else (), self._run_labels),
                note,
            )
            for column, item in enumerate(items):
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if pair.name == self._trend:
                    item.setBackground(QColor(tokens.ACCENT_SOFT))
                table.setItem(index, column, item)

    def _on_table_clicked(self, row: int, _column: int) -> None:
        name = self._pairs[row].name
        if name in self._trend_names():
            self._trend = name
            self._fill_table()
            self._draw_trend()

    def _draw_trend(self) -> None:
        """Plot the trend parameter along the series: A filled with a line, B hollow.

        Ladders draw the strip instead: every parameter A or B leaves local.
        """
        if self._ladders:
            self._strip.set_traces(
                rung_traces(
                    [
                        TracedRung(summary, letter, _TREND_COLOUR, filled=letter == "A")
                        for letter, summary in zip("AB", self._a_and_b(), strict=True)
                        if summary is not None
                    ]
                ),
                self._axis_label,
            )
            return
        if self._trend_canvas.isHidden():
            return
        figure = self._trend_figure
        figure.clear()
        axes = figure.add_subplot()
        style_axes(axes)
        a, b = self._a_and_b()
        if a is None:
            axes.set_axis_off()
        elif self._trend is None:
            draw_empty_state_message(axes, NO_LOCAL_TREND)
        else:
            for letter, summary in (("A", a), ("B", b)):
                rows = {} if summary is None else {row.name: row for row in summary.parameters}
                if self._trend not in rows:
                    continue
                values = rows[self._trend].values
                linestyle = "-" if letter == "A" else (0, (4, 3))
                if isinstance(values, Estimate):
                    # A shared value is one line across the series, with its ±σ band.
                    axes.axhline(
                        values.value,
                        color=_TREND_COLOUR,
                        linestyle=linestyle,
                        linewidth=1.2,
                        label=letter,
                    )
                    if np.isfinite(values.error):
                        axes.axhspan(
                            values.value - values.error,
                            values.value + values.error,
                            color=_TREND_COLOUR,
                            alpha=0.1,
                            linewidth=0,
                        )
                    continue
                x = np.array([run.axis_value for run in summary.runs], dtype=float)
                y = np.array([estimate.value for estimate in values], dtype=float)
                order = np.argsort(x)
                axes.plot(
                    x[order], y[order], color=_TREND_COLOUR, linestyle=linestyle, linewidth=1.0
                )
                add_errorbar_dots(
                    axes,
                    x,
                    y,
                    np.array([estimate.error for estimate in values], dtype=float),
                    fmt="o",
                    color=_TREND_COLOUR,
                    markersize=5,
                    markerfacecolor=_TREND_COLOUR if letter == "A" else tokens.SURFACE,
                    label=letter,
                )
            axes.set_xlabel(self._axis_label)
            axes.set_ylabel(format_param_label(self._trend))
            if b is not None:
                style_legend(axes.legend(loc="best", fontsize=8))
        self._trend_canvas.draw_idle()
