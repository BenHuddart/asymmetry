"""The Screen step's family leaderboard (plan D3, D7).

One row per pre-screened family, in rank order: a shortlist checkbox with the
title, a bar for Δ(metric) from the best row, one graded χ²ᵣ cell per run, and
the family's optimisation status. Clicking a row selects it for the preview.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    FitGrade,
    grade_reduced_chi_squared,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import char_width, row_height
from asymmetry.gui.styles.typography import header_font
from asymmetry.gui.utils.formatting import format_reduced_chi_squared


class OptimisationStatus(Enum):
    """Whether a family's coupled role search has run; the value is its badge text."""

    NOT_OPTIMISED = "Not optimised"
    RUNNING = "Running"
    OPTIMISED = "Optimised"
    FAILED = "Failed"


#: Above this many runs the χ²ᵣ cells become a heat strip: graded squares whose
#: value is in the tooltip, since nine or more numbers no longer fit legibly.
NUMERIC_CHI2_MAX_RUNS = 8
#: The Δ at which the bar is full; a larger Δ prints its value over a full bar.
DELTA_BAR_FULL = 20.0
#: Collapsed, the board shows the ticked rows, the selected row and this many
#: unticked rows by rank.
COLLAPSED_UNTICKED_ROWS = 3

_STATUS_COLOURS = {
    OptimisationStatus.NOT_OPTIMISED: (tokens.SURFACE_ALT, tokens.TEXT_MUTED),
    OptimisationStatus.RUNNING: (tokens.ACCENT_SOFT, tokens.ACCENT),
    OptimisationStatus.OPTIMISED: (tokens.SUCCESS_SOFT, tokens.OK),
    OptimisationStatus.FAILED: (tokens.ERROR_SOFT, tokens.ERROR),
}
_GRADE_COLOURS = {
    FitGrade.GOOD: (tokens.SUCCESS_SOFT, tokens.OK),
    FitGrade.FAIR: (tokens.WARN_SOFT, tokens.WARN_BANNER_TEXT),
    FitGrade.POOR: (tokens.ERROR_SOFT, tokens.ERROR),
}
_UNFITTED_COLOURS = (tokens.SURFACE_ALT, tokens.TEXT_DIM)

_KEY_ROLE = Qt.ItemDataRole.UserRole
_DELTA_ROLE = Qt.ItemDataRole.UserRole + 1
_ROW_PADDING = 4
_CHIP_PADDING = 8
_TITLE_COLUMN = 0
_DELTA_COLUMN = 1
_FIRST_RUN_COLUMN = 2


def format_delta(delta: float) -> str:
    """``"best"`` for Δ = 0, one decimal below the full bar, grouped integers beyond."""
    if not math.isfinite(delta):
        return "—"
    if delta == 0:
        return "best"
    if delta < DELTA_BAR_FULL:
        return f"+{delta:.1f}"
    return "+" + f"{delta:,.0f}".replace(",", "\u202f")  # narrow no-break space


def _draw_item_frame(option: QStyleOptionViewItem, painter: QPainter) -> None:
    """The item's selection and hover background, without its own background or text."""
    frame = QStyleOptionViewItem(option)
    frame.text = ""
    frame.backgroundBrush = QBrush()
    frame.widget.style().drawControl(
        QStyle.ControlElement.CE_ItemViewItem, frame, painter, frame.widget
    )


class _RowDelegate(QStyledItemDelegate):
    """Selection is by row, so no single cell draws the keyboard-focus treatment."""

    def initStyleOption(  # noqa: N802 - Qt override
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> None:
        super().initStyleOption(option, index)
        option.state &= ~QStyle.StateFlag.State_HasFocus


class _ChipDelegate(_RowDelegate):
    """Paints an item's background as a rounded chip under its centred text, selected or not."""

    def __init__(self, parent: QWidget, *, fill_cell: bool) -> None:
        super().__init__(parent)
        self._fill_cell = fill_cell

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        _draw_item_frame(opt, painter)
        text = opt.text
        cell = QRectF(opt.rect)
        if self._fill_cell:
            chip = cell.adjusted(2, 3, -2, -3)
        else:
            metrics = opt.fontMetrics
            chip = QRectF(
                0, 0, metrics.horizontalAdvance(text) + 2 * _CHIP_PADDING, metrics.height() + 4
            )
            chip.moveCenter(cell.center())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        foreground = index.data(Qt.ItemDataRole.ForegroundRole).color()
        if self._fill_cell:
            # A faint edge keeps a pale cell apart from the selected row's tint.
            edge = QColor(foreground)
            edge.setAlphaF(0.3)
            painter.setPen(edge)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(opt.backgroundBrush)
        painter.drawRoundedRect(chip, 4, 4)
        painter.setPen(foreground)
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


def paint_delta_bar(painter: QPainter, area: QRectF, delta: float, metrics: QFontMetrics) -> None:
    """A track filling with Δ up to ``DELTA_BAR_FULL``, :func:`format_delta` right of it."""
    text_width = metrics.horizontalAdvance(format_delta(1e5))
    track = QRectF(area.left(), area.center().y() - 3, area.width() - text_width - 6, 6)
    fraction = min(delta, DELTA_BAR_FULL) / DELTA_BAR_FULL if math.isfinite(delta) else 0.0
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(tokens.SURFACE_HI))
    painter.drawRoundedRect(track, 3, 3)
    painter.setBrush(QColor(tokens.BORDER_STRONG))
    painter.drawRoundedRect(
        QRectF(track.topLeft(), track.size()).adjusted(0, 0, -(1 - fraction) * track.width(), 0),
        3,
        3,
    )
    painter.setPen(QColor(tokens.OK if delta == 0 else tokens.TEXT_MUTED))
    painter.drawText(
        area, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, format_delta(delta)
    )
    painter.restore()


class _DeltaBarDelegate(_RowDelegate):
    """A bar growing with Δ up to ``DELTA_BAR_FULL``, with the formatted Δ beside it."""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        _draw_item_frame(opt, painter)
        paint_delta_bar(
            painter,
            QRectF(opt.rect).adjusted(_CHIP_PADDING, 0, -_CHIP_PADDING, 0),
            float(index.data(_DELTA_ROLE)),
            opt.fontMetrics,
        )


def _chip_item(text: str, colours: tuple[str, str], tooltip: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    background, foreground = colours
    item.setBackground(QColor(background))
    item.setForeground(QColor(foreground))
    item.setToolTip(tooltip)
    return item


class ScreeningLeaderboard(QWidget):
    """Pre-screened families ranked by Δ(metric), with a shortlist tick per row."""

    ticked_changed = Signal(tuple)
    selected_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget(self)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(row_height() + 2 * _ROW_PADDING)
        self._table.horizontalHeader().setFont(header_font())
        self._table.horizontalHeader().setHighlightSections(False)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._table.setItemDelegate(_RowDelegate(self._table))
        self._table.setItemDelegateForColumn(_DELTA_COLUMN, _DeltaBarDelegate(self._table))
        self._cell_delegate = _ChipDelegate(self._table, fill_cell=True)
        self._badge_delegate = _ChipDelegate(self._table, fill_cell=False)
        self._table.itemChanged.connect(lambda _item: self.ticked_changed.emit(self.ticked()))
        self._table.currentCellChanged.connect(self._on_current_cell_changed)
        self._heat_caption = QLabel(self)
        self._heat_caption.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._heat_caption.hide()
        layout.addWidget(self._heat_caption)
        layout.addWidget(self._table)

        self._show_all = QPushButton(self)
        self._show_all.setCheckable(True)
        self._show_all.setFlat(True)
        self._show_all.setStyleSheet(
            f"QPushButton {{ color: {tokens.ACCENT}; border: none; padding: 2px 0; }}"
        )
        self._show_all.toggled.connect(self._apply_row_visibility)
        layout.addWidget(self._show_all, 0, Qt.AlignmentFlag.AlignLeft)
        # The table is fixed-height, so spare height goes below it, not around it.
        layout.addStretch()

    def set_candidates(
        self,
        summaries: Sequence[CandidateSummary],
        run_labels: Mapping[int, str],
        metric_label: str,
        *,
        ticked: Iterable[str],
        statuses: Mapping[str, OptimisationStatus],
    ) -> None:
        """Show ``summaries`` in rank order, one χ²ᵣ column per run of ``run_labels``.

        ``run_labels`` maps run number to its column heading, in series order. The
        previously selected family stays selected when it is still listed, else the
        first row is; neither emits a signal.
        """
        previous_key = self.selected_key()
        ticked_keys = set(ticked)
        runs = list(run_labels.items())
        numeric = len(runs) <= NUMERIC_CHI2_MAX_RUNS
        status_column = _FIRST_RUN_COLUMN + len(runs)
        self._heat_caption.setVisible(not numeric)
        if not numeric:
            self._heat_caption.setText(
                f"χ²ᵣ per run, {runs[0][1]} → {runs[-1][1]}: hover a square for its value"
            )
        table = self._table
        with QSignalBlocker(table):
            table.clear()
            table.setRowCount(len(summaries))
            table.setColumnCount(status_column + 1)
            table.setHorizontalHeaderLabels(
                ["Family", metric_label, *(label if numeric else "" for _, label in runs), "Status"]
            )
            for column, (_, label) in enumerate(runs, start=_FIRST_RUN_COLUMN):
                table.horizontalHeaderItem(column).setToolTip(label)
                table.setItemDelegateForColumn(column, self._cell_delegate)
            table.setItemDelegateForColumn(status_column, self._badge_delegate)
            for row, summary in enumerate(summaries):
                title = QTableWidgetItem(summary.title)
                title.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                title.setCheckState(
                    Qt.CheckState.Checked if summary.key in ticked_keys else Qt.CheckState.Unchecked
                )
                title.setData(_KEY_ROLE, summary.key)
                title.setToolTip(
                    f"{summary.title}\n{summary.gate_summary}"
                    if summary.gate_summary
                    else summary.title
                )
                table.setItem(row, _TITLE_COLUMN, title)
                delta = QTableWidgetItem(format_delta(summary.delta))
                delta.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                delta.setData(_DELTA_ROLE, summary.delta)
                table.setItem(row, _DELTA_COLUMN, delta)
                fits = {run.run_number: run for run in summary.runs}
                for column, (run_number, label) in enumerate(runs, start=_FIRST_RUN_COLUMN):
                    if run_number in fits:
                        chi2 = fits[run_number].reduced_chi_squared
                        text = format_reduced_chi_squared(chi2)
                        item = _chip_item(
                            text if numeric else "",
                            _GRADE_COLOURS[grade_reduced_chi_squared(chi2)],
                            f"{label}: χ²ᵣ {text}",
                        )
                    else:
                        item = _chip_item(
                            "—" if numeric else "", _UNFITTED_COLOURS, f"{label}: not fitted"
                        )
                    table.setItem(row, column, item)
                status = statuses[summary.key]
                table.setItem(
                    row,
                    status_column,
                    _chip_item(status.value, _STATUS_COLOURS[status], status.value),
                )
            keys = [summary.key for summary in summaries]
            table.setCurrentCell(keys.index(previous_key) if previous_key in keys else 0, 0)

        header = table.horizontalHeader()
        header_metrics = QFontMetrics(header_font())
        header.setSectionResizeMode(_TITLE_COLUMN, QHeaderView.ResizeMode.Stretch)
        for column in range(_DELTA_COLUMN, status_column + 1):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(
            _DELTA_COLUMN,
            max(char_width(16), header_metrics.horizontalAdvance(metric_label) + 3 * _CHIP_PADDING),
        )
        for column, (_, label) in enumerate(runs, start=_FIRST_RUN_COLUMN):
            table.setColumnWidth(
                column,
                max(char_width(7), header_metrics.horizontalAdvance(label) + 2 * _CHIP_PADDING)
                if numeric
                else row_height() + _ROW_PADDING,
            )
        table.setColumnWidth(
            status_column,
            max(
                QFontMetrics(table.font()).horizontalAdvance(status.value)
                for status in OptimisationStatus
            )
            + 4 * _CHIP_PADDING,
        )
        self._apply_row_visibility(self._show_all.isChecked())

    def ticked(self) -> tuple[str, ...]:
        """Keys of the ticked families, in rank order."""
        return tuple(
            self._table.item(row, _TITLE_COLUMN).data(_KEY_ROLE)
            for row in range(self._table.rowCount())
            if self._table.item(row, _TITLE_COLUMN).checkState() == Qt.CheckState.Checked
        )

    def selected_key(self) -> str | None:
        """Key of the family selected for the preview; ``None`` only when the board is empty."""
        row = self._table.currentRow()
        return None if row < 0 else self._table.item(row, _TITLE_COLUMN).data(_KEY_ROLE)

    def _on_current_cell_changed(
        self, row: int, _column: int, previous_row: int, _previous_column: int
    ) -> None:
        if row != previous_row and row >= 0:
            self.selected_changed.emit(self._table.item(row, _TITLE_COLUMN).data(_KEY_ROLE))

    def _apply_row_visibility(self, expanded: bool) -> None:
        """Hide the rows a collapsed board leaves out, and fit the table to what is shown.

        Runs on new candidates and on the toggle only, never on a tick, so a row
        never vanishes from under the pointer that unticked it.
        """
        table = self._table
        current = table.currentRow()
        unticked_seen = 0
        left_out = 0
        for row in range(table.rowCount()):
            ticked = table.item(row, _TITLE_COLUMN).checkState() == Qt.CheckState.Checked
            kept = ticked or row == current or unticked_seen < COLLAPSED_UNTICKED_ROWS
            unticked_seen += not ticked
            left_out += not kept
            table.setRowHidden(row, not (expanded or kept))
        self._show_all.setVisible(left_out > 0)
        self._show_all.setText(
            "Show fewer families" if expanded else f"Show all {table.rowCount()} families"
        )
        table.setFixedHeight(
            table.horizontalHeader().sizeHint().height()
            + sum(
                table.rowHeight(row)
                for row in range(table.rowCount())
                if not table.isRowHidden(row)
            )
            + 2 * table.frameWidth()
        )
