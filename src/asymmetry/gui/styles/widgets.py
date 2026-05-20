"""Shared widget-styling helpers for the BENCH design language.

Import these in any panel or window that needs BENCH-consistent table headers,
formula code-boxes, or result-group success states.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QLabel, QSizePolicy, QTableWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font

# ── Result group ──────────────────────────────────────────────────────────────

RESULTS_GROUP_SUCCESS_STYLE = (
    "QGroupBox {"
    " background-color: #f4f8f4;"
    " border: 1px solid #cbe1cf;"
    " border-radius: 4px;"
    " margin-top: 10px;"
    "}"
)


def apply_results_group_success(group: QGroupBox) -> None:
    """Apply green-tint success style to a results QGroupBox."""
    group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)


def clear_results_group_style(group: QGroupBox) -> None:
    """Remove any inline style from a results QGroupBox (revert to bench.qss default)."""
    group.setStyleSheet("")


# ── Table style ───────────────────────────────────────────────────────────────

def apply_param_table_style(table: QTableWidget) -> None:
    """Apply BENCH styling to any parameter or data table.

    Sets a DemiBold section header, 11pt mono cell font, hides the row-number
    column, and tints alternating rows with surfaceAlt.
    """
    hdr_font = QFont()
    hdr_font.setPointSizeF(9.5)
    hdr_font.setWeight(QFont.Weight.DemiBold)
    hdr_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.3)
    table.horizontalHeader().setFont(hdr_font)
    table.verticalHeader().setVisible(False)
    table.setFont(mono_font(11.0))
    table.setAlternatingRowColors(True)
    table.setStyleSheet(
        f"QTableWidget {{ alternate-background-color: {tokens.SURFACE_ALT}; }}"
    )


# ── Formula label ─────────────────────────────────────────────────────────────

def configure_formula_label(label: QLabel) -> None:
    """Style a QLabel as a read-only mono code-box for formula display."""
    label.setFont(mono_font(11.0))
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
    label.setStyleSheet(
        f"QLabel {{ background-color: {tokens.SURFACE_ALT}; border: 1px solid {tokens.BORDER};"
        " border-radius: 3px; padding: 6px 8px; }"
    )
    line_height = label.fontMetrics().lineSpacing()
    label.setMinimumHeight(line_height * 3 + 16)
