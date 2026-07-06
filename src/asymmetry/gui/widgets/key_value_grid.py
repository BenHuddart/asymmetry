"""Read-only aligned name/value results grid for the BENCH UI.

A compact two-column grid for "labelled result" readouts (fit summaries,
run-info panels, reconstruction stats) where a full ``QTableWidget`` is heavier
than the content warrants. Labels are muted body text; values are monospaced so
numbers line up, and the value text is selectable so a user can copy a figure.
Values may be rich text (a chip from ``styles/widgets.py``) — set via
:meth:`set_rows`, which repopulates without leaking the previous row widgets.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import SIZE_BODY, SIZE_NUMERIC
from asymmetry.gui.styles.widgets import clear_layout


class KeyValueGrid(QWidget):
    """A compact, chrome-free grid of read-only ``name → value`` result rows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(3)
        # Values take the slack so the label column stays tight to its text.
        self._grid.setColumnStretch(1, 1)

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        """Replace the grid contents with ``(label, value)`` pairs.

        *value* may be plain or rich text (e.g. a chip HTML snippet from
        ``styles/widgets.py``). Any previously added rows are cleared and their
        widgets scheduled for deletion, so repeated calls do not leak.
        """
        clear_layout(self._grid)
        for row, (label_text, value_text) in enumerate(rows):
            label = QLabel(str(label_text))
            label.setFont(mono_font(SIZE_BODY))
            label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

            value = QLabel(str(value_text))
            value.setFont(mono_font(SIZE_NUMERIC))
            value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            value.setTextFormat(Qt.TextFormat.RichText)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            self._grid.addWidget(label, row, 0)
            self._grid.addWidget(value, row, 1)
