"""Font-metric-relative sizing helpers for the BENCH UI.

Panels historically hard-coded pixel widths (``setFixedWidth(70)``, dialog
minimums, table row heights) that were tuned for one font size and then drift
once the font-driven UI zoom scales everything else. These helpers derive the
same figures from the *current* application font at call time, so a widget sized
through them tracks the active zoom instead of freezing at design size.

Deliberately QtGui-only (``QGuiApplication`` lives in QtGui) and free of any
``ui_manager`` import, so the helpers stay usable from plain widget code and
from unit contexts without pulling in the whole GUI stack. Every helper reads
the live font via :func:`QGuiApplication.font` unless a font is passed
explicitly, and raises a clear error when no ``QGuiApplication`` exists yet.
"""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontMetrics, QGuiApplication

#: Frame + horizontal padding allowance (px) added to a raw character width so a
#: :func:`field_width_for` result comfortably fits a QLineEdit / spinbox with its
#: native frame and text margins. Additive (never multiplied) so widths stay
#: monotonic in the character count.
_FIELD_PADDING = 16

#: Vertical padding (px) added to the font line height for a table row so cell
#: text is not cramped against the grid lines.
_ROW_PADDING = 6


def _resolve_font(font: QFont | None) -> QFont:
    """Return *font*, or the live application font, raising if no app exists."""
    if font is not None:
        return font
    app = QGuiApplication.instance()
    if app is None:
        raise RuntimeError(
            "asymmetry.gui.styles.metrics requires a running QGuiApplication; "
            "call these helpers after the application is created (or pass an "
            "explicit font)."
        )
    return app.font()


def char_width(n: int, font: QFont | None = None) -> int:
    """Return the width in px of *n* average characters in *font*.

    Uses ``QFontMetrics.averageCharWidth`` on the given font, or the live
    application font when *font* is ``None`` — so the result tracks the active
    UI font scale. Rounded to a whole pixel.
    """
    metrics = QFontMetrics(_resolve_font(font))
    return round(max(0, int(n)) * metrics.averageCharWidth())


def field_width_for(chars: int, widget=None) -> int:
    """Return a pixel width sized to hold *chars* characters in a text field.

    Suitable for ``QLineEdit`` / spinbox minimums: the raw character width plus
    a fixed frame/padding allowance so the field's own chrome does not clip the
    text. When *widget* is given its font is used (so a field styled at a
    non-default size measures against its own metrics); otherwise the live
    application font is used.
    """
    font = widget.font() if widget is not None else None
    return char_width(chars, font) + _FIELD_PADDING


def dialog_width(chars: int) -> int:
    """Return a convenience dialog minimum width sized for *chars* characters.

    A thin wrapper over :func:`field_width_for` for the common "make this dialog
    at least wide enough for N characters of content" case.
    """
    return field_width_for(chars)


def row_height(font: QFont | None = None) -> int:
    """Return a table row height in px: font line height plus vertical padding.

    Tracks the live application font (or *font* when given) so rows grow with the
    UI scale rather than clipping taller glyphs at a frozen height.
    """
    metrics = QFontMetrics(_resolve_font(font))
    return metrics.height() + _ROW_PADDING
