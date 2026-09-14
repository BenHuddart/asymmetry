"""Read-out of a solved fit — a parameter's trend model, or one run's asymmetry fit.

The parameters panel's cards carry a χ²ᵣ chip; clicking it opens one of these
windows for that parameter. The fit tabs' χ²ᵣ chip opens the same window for the
run that was just fitted. It is a pure view: the caller hands it the plain
:class:`FitResults` snapshot below and this module imports nothing from the
panels, so the window can be built and tested on its own. ``editable`` says
whether the fit behind the snapshot has an editor to hand off to — the trend
fit does (:attr:`FitResultsWindow.edit_requested`), a run fit does not. The
parameters panel keeps one window per parameter, refreshes it whenever the fit
changes, and drops it when the parameter's card goes away.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    VERDICT_CHIP_OBJECT_NAME,
    apply_param_table_style,
    clear_layout,
    make_context_chip,
    verdict_chip_qss,
)
from asymmetry.gui.utils.formatting import (
    format_value_error,
    format_value_uncertainty,
)

__all__ = [
    "FitParameterRow",
    "FitRangeResults",
    "FitResults",
    "FitResultsWindow",
]

#: The table's columns, in order.
_COLUMNS = ("Parameter", "Value", "Unit")
#: Row label introducing the fixed parameters underneath the fitted ones.
_FIXED_SEPARATOR = "held fixed"


@dataclass(frozen=True)
class FitParameterRow:
    """One parameter of a solved range, as the table shows it."""

    #: Raw parameter name — what the clipboard copy quotes.
    name: str
    #: Display symbol without its unit ("T꜀").
    symbol: str
    #: Unit, or "" for a dimensionless parameter.
    unit: str
    value: float
    #: ``None`` for a parameter the fit did not vary, or one Minuit gave no
    #: uncertainty for.
    error: float | None
    fixed: bool


@dataclass(frozen=True)
class FitRangeResults:
    """One solved range of a parameter's model fit."""

    #: The range's composite model expression ("SC_TwoGap_SS").
    model: str
    #: The χ²ᵣ chip's text ("χ²ᵣ 0.89").
    chi_squared: str
    #: The verdict phrase the chip's tooltip leads with.
    verdict: str
    #: The chip's (background, border, text) colours.
    colours: tuple[str, str, str]
    #: The fitted x interval, already formatted ("100 – 200").
    bounds: str
    #: The error mode the fit ran under ("column", "scatter", …).
    error_mode: str
    #: Free parameters first, then the fixed ones.
    parameters: tuple[FitParameterRow, ...]


@dataclass(frozen=True)
class FitResults:
    """Everything :class:`FitResultsWindow` renders for one fit."""

    #: The window's title — a trend fit names its parameter, a run fit its run.
    title: str
    parameter_name: str
    #: The trend's x axis, as the panel labels it ("Temperature (K)").
    x_label: str
    #: Trend provenance ("4 / 4 runs · 1 range").
    runs: str
    #: The solved ranges, in fit order; never empty.
    ranges: tuple[FitRangeResults, ...]


class FitResultsWindow(QDialog):
    """Non-modal, reusable read-out of one parameter's trend model fit."""

    #: The user asked to edit this fit; carries the parameter name.
    edit_requested = Signal(str)

    def __init__(
        self,
        results: FitResults,
        parent: QWidget | None = None,
        *,
        editable: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)

        layout = QVBoxLayout(self)
        self._body = QVBoxLayout()
        layout.addLayout(self._body)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        copy_button = QPushButton("Copy")
        copy_button.setToolTip("Copy the fitted parameters to the clipboard.")
        copy_button.clicked.connect(self._copy)
        buttons.addWidget(copy_button)
        if editable:
            edit_button = QPushButton("Edit model fit…")
            edit_button.clicked.connect(self._edit)
            buttons.addWidget(edit_button)
        buttons.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self.set_results(results)
        self.adjustSize()

    def set_results(self, results: FitResults) -> None:
        """Re-render the window from *results* (the fit changed, or is new)."""
        self._results = results
        self.setWindowTitle(results.title)
        clear_layout(self._body)

        first = results.ranges[0]
        header = QWidget(self)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(make_context_chip(first.model))
        header_row.addWidget(_verdict_chip(first))
        header_row.addStretch(1)
        header_row.addWidget(_muted_label(results.runs))
        self._body.addWidget(header)

        for index, fit_range in enumerate(results.ranges, start=1):
            if len(results.ranges) > 1:
                self._body.addWidget(
                    _muted_label(f"Range {index} · {fit_range.bounds} · {fit_range.chi_squared}")
                )
            self._body.addWidget(_parameter_table(fit_range))

        # A run's asymmetry fit has neither a trended x axis nor an error mode of
        # its own, so those segments are simply absent rather than shown empty.
        footer = [f"fit range {first.bounds}"]
        if results.x_label:
            footer.insert(0, f"x: {results.x_label}")
        if first.error_mode:
            footer.append(f"errors: {first.error_mode}")
        self._body.addWidget(_muted_label(" · ".join(footer)))

    def _copy(self) -> None:
        lines: list[str] = []
        for fit_range in self._results.ranges:
            lines.append(f"Model: {fit_range.model}")
            lines.append(fit_range.chi_squared)
            for row in fit_range.parameters:
                value = format_value_error(row.value, row.error or 0.0)
                lines.append(f"{row.name}\t{value}\t{row.unit}")
        QApplication.clipboard().setText("\n".join(lines))

    def _edit(self) -> None:
        self.edit_requested.emit(self._results.parameter_name)


def _verdict_chip(fit_range: FitRangeResults) -> QLabel:
    chip = QLabel(fit_range.chi_squared)
    chip.setObjectName(VERDICT_CHIP_OBJECT_NAME)
    chip.setToolTip(f"{fit_range.chi_squared} · {fit_range.verdict}")
    chip.setStyleSheet(verdict_chip_qss(fit_range.colours, widget="QLabel"))
    return chip


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _fill_row(table: QTableWidget, index: int, row: FitParameterRow) -> None:
    cells = (row.symbol, format_value_uncertainty(row.value, row.error), row.unit)
    for column, text in enumerate(cells):
        item = QTableWidgetItem(text)
        if row.fixed:
            item.setForeground(QColor(tokens.TEXT_MUTED))
        table.setItem(index, column, item)


def _parameter_table(fit_range: FitRangeResults) -> QTableWidget:
    """Return the read-only Parameter/Value/Unit table for one solved range."""
    free = [row for row in fit_range.parameters if not row.fixed]
    fixed = [row for row in fit_range.parameters if row.fixed]
    # The separator row only exists when there is something to separate.
    table = QTableWidget(len(free) + (1 + len(fixed) if fixed else 0), len(_COLUMNS))
    table.setHorizontalHeaderLabels(list(_COLUMNS))
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    apply_param_table_style(table)

    for index, row in enumerate(free):
        _fill_row(table, index, row)
    if fixed:
        separator = QTableWidgetItem(_FIXED_SEPARATOR)
        separator.setForeground(QColor(tokens.TEXT_DIM))
        table.setItem(len(free), 0, separator)
        table.setSpan(len(free), 0, 1, len(_COLUMNS))
        for offset, row in enumerate(fixed):
            _fill_row(table, len(free) + 1 + offset, row)

    table.resizeColumnsToContents()
    frame = 2 * table.frameWidth()
    table.setMinimumWidth(table.horizontalHeader().length() + frame)
    # The window is as wide as its button row; the unit column takes the slack
    # rather than leaving an empty gutter beside the last column.
    table.horizontalHeader().setStretchLastSection(True)
    table.setFixedHeight(
        table.horizontalHeader().height()
        + table.rowCount() * table.verticalHeader().defaultSectionSize()
        + frame
    )
    return table
