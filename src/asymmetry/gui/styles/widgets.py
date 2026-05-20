"""Shared widget-styling helpers for the BENCH design language.

Import these in any panel or window that needs BENCH-consistent table headers,
formula code-boxes, result-group success states, button QSS, or HTML snippets.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QSizePolicy, QTableWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import header_font

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


# ── Table header font ─────────────────────────────────────────────────────────

def apply_param_table_header_font(table: QTableWidget) -> None:
    """Apply the BENCH column-header font to a table's horizontal header.

    Separated from apply_param_table_style so callers that handle row colours
    themselves can still pick up the canonical header font.
    """
    table.horizontalHeader().setFont(header_font())


# ── Segmented toolbar / toggle-button QSS ────────────────────────────────────

def build_segmented_button_qss(*, min_width: int | None = None) -> str:
    """Return QSS for a checkable QPushButton used as a segmented-control cell.

    Checked state uses ACCENT_SOFT background with ACCENT border and text.
    Unchecked state uses SURFACE background with BORDER.
    """
    width_rule = f" min-width: {min_width}px;" if min_width is not None else ""
    return (
        f"QPushButton {{{width_rule} background-color: {tokens.SURFACE};"
        f" color: {tokens.TEXT}; border: 1px solid {tokens.BORDER}; border-radius: 4px; }}"
        f" QPushButton:checked {{ background-color: {tokens.ACCENT_SOFT};"
        f" color: {tokens.ACCENT}; border-color: {tokens.ACCENT}; font-weight: 600; }}"
    )


def build_nav_button_qss() -> str:
    """Return QSS for plot-panel pan/zoom/auto navigation buttons."""
    return (
        f"QPushButton {{ min-width: 60px; border: 1px solid {tokens.BORDER_STRONG};"
        f" border-radius: 4px; }}"
        f" QPushButton:checked {{ background-color: {tokens.ACCENT_SOFT};"
        f" color: {tokens.ACCENT}; border: 2px solid {tokens.ACCENT}; font-weight: 600; }}"
    )


# ── HTML snippets ─────────────────────────────────────────────────────────────

def success_html(label: str, *, detail: str | None = None) -> str:
    """Return rich-text HTML for a success status line (green header + muted detail).

    Args:
        label:  Primary line, e.g. "Fit converged".
        detail: Optional secondary line in muted colour, e.g. "χ²/ν = 1.04 · npar = 5".
    """
    html = f'<span style="color:{tokens.OK};font-weight:600;">{label}</span>'
    if detail:
        html += f'<br><span style="color:{tokens.TEXT_MUTED};">{detail}</span>'
    return html


def info_html(label: str) -> str:
    """Return rich-text HTML for an informational status line (accent colour)."""
    return f'<span style="color:{tokens.ACCENT};">{label}</span>'


def warning_html(label: str) -> str:
    """Return rich-text HTML for a warning status line (warn colour)."""
    return f'<span style="color:{tokens.WARN};">{label}</span>'


def error_html(label: str) -> str:
    """Return rich-text HTML for an error status line (error colour)."""
    return f'<span style="color:{tokens.ERROR};">{label}</span>'


# ── Group-state buttons ───────────────────────────────────────────────────────

def style_group_state_button(
    button: QPushButton,
    state: Literal["active", "selected", "unselected"],
) -> None:
    """Apply BENCH styling to a group-state toggle button in fit_parameters_panel.

    States:
        active:     Currently focused/active group — ACCENT_SOFT fill, ACCENT border.
        selected:   Included in multi-group fit — ACCENT_SOFT2 fill, ACCENT border, dim text.
        unselected: Not participating — SURFACE fill, BORDER border, muted text.
    """
    if state == "active":
        button.setStyleSheet(
            f"QPushButton {{ border: 2px solid {tokens.ACCENT};"
            f" background: {tokens.ACCENT_SOFT}; color: {tokens.ACCENT};"
            f" font-weight: 700; }}"
        )
    elif state == "selected":
        button.setStyleSheet(
            f"QPushButton {{ border: 1px solid {tokens.ACCENT};"
            f" background: {tokens.ACCENT_SOFT2}; color: {tokens.TEXT};"
            f" font-weight: 500; }}"
        )
    else:
        button.setStyleSheet(
            f"QPushButton {{ border: 1px solid {tokens.BORDER};"
            f" background: {tokens.SURFACE}; color: {tokens.TEXT_MUTED};"
            f" font-weight: 400; }}"
        )


# ── Footer hint label ─────────────────────────────────────────────────────────

def apply_footer_hint(label: QLabel) -> None:
    """Apply the BENCH footer-hint style to a QLabel (surfaceAlt bg, muted text)."""
    label.setStyleSheet(
        f"QLabel {{ background-color: {tokens.SURFACE_ALT};"
        f" color: {tokens.TEXT_MUTED};"
        f" border-top: 1px solid {tokens.BORDER}; }}"
    )
