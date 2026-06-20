"""BENCH type-scale constants and QFont builders.

Import from here instead of instantiating bare QFont() objects in panels, so the
type scale stays in one place. The point-size constants below are the single
source of truth for the BENCH type scale: the QFont builders here consume them,
and ``ui_manager.build_stylesheet`` references the same constants for the chrome
QSS selectors (dock titles, table headers, group-box titles) so the Python fonts
and the stylesheet can never drift (the historical pt-vs-px split).

All builders multiply their point size by the active UI font scale (see
``fonts.set_ui_font_scale``) so chrome built after a scale change is born at the
active scale.
"""

from __future__ import annotations

from PySide6.QtGui import QFont

from asymmetry.gui.styles.fonts import mono_font, scaled_point_size

# ── Letter-spacing (absolute, pixels) ────────────────────────────────────────

LETTER_SPACING_HEADER = 0.3  # column headers, data-browser table
LETTER_SPACING_LABEL = 0.4  # 9.5pt uppercase section headers (QGroupBox titles)

# ── Point sizes ───────────────────────────────────────────────────────────────

SIZE_BODY = 11.5  # form labels, body text
SIZE_NUMERIC = 11.0  # monospaced numeric display
SIZE_HEADER = 9.5  # section/column headers (uppercase DemiBold)
SIZE_STATUS = 10.5  # status bar and log entries
SIZE_FOOTER = 10.0  # footer hints, tab badges

# ── QFont builders ────────────────────────────────────────────────────────────


def header_font() -> QFont:
    """9.5pt DemiBold sans with 0.3px letter-spacing — column and table headers."""
    f = QFont()
    f.setPointSizeF(scaled_point_size(SIZE_HEADER))
    f.setWeight(QFont.Weight.DemiBold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, LETTER_SPACING_HEADER)
    return f


def section_label_font() -> QFont:
    """9.5pt Bold sans with 0.4px letter-spacing — inspector section titles.

    Caller is responsible for uppercasing the label text; Qt QSS text-transform
    is not guaranteed across all platforms.
    """
    f = QFont()
    f.setPointSizeF(scaled_point_size(SIZE_HEADER))
    f.setWeight(QFont.Weight.Bold)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, LETTER_SPACING_LABEL)
    return f


def status_font() -> QFont:
    """10.5pt monospace — status bar coordinates label and log entries."""
    return mono_font(SIZE_STATUS)


def footer_font() -> QFont:
    """10.0pt monospace — footer hints and badge counts."""
    return mono_font(SIZE_FOOTER)
