"""Render LaTeX-style math expressions for Qt widgets."""

from __future__ import annotations

import base64
from functools import lru_cache
import io

from PySide6.QtGui import QPixmap


@lru_cache(maxsize=256)
def render_latex_png_bytes(latex: str, font_size: int = 16, dpi: int = 160) -> bytes | None:
    """Render math text to PNG bytes using matplotlib mathtext.

    Returns ``None`` if matplotlib is unavailable or rendering fails.
    """
    expression = latex.strip()
    if not expression:
        return None
    if not expression.startswith("$"):
        expression = f"${expression}$"

    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
    except Exception:
        return None

    try:
        buffer = io.BytesIO()
        math_to_image(
            expression,
            buffer,
            prop=FontProperties(size=font_size),
            dpi=dpi,
            format="png",
        )
    except Exception:
        return None

    return buffer.getvalue() or None


def render_latex_to_pixmap(latex: str, font_size: int = 16, dpi: int = 160) -> QPixmap | None:
    """Render a math expression into a ``QPixmap`` for Qt views."""
    png = render_latex_png_bytes(latex, font_size=font_size, dpi=dpi)
    if not png:
        return None

    pixmap = QPixmap()
    if not pixmap.loadFromData(png, "PNG"):
        return None
    return pixmap


def render_latex_to_html_image(latex: str, font_size: int = 16, dpi: int = 160) -> str | None:
    """Return a data-URI HTML ``<img>`` for a LaTeX expression."""
    png = render_latex_png_bytes(latex, font_size=font_size, dpi=dpi)
    if not png:
        return None

    encoded = base64.b64encode(png).decode("ascii")
    return (
        "<img "
        "style='max-width:100%; height:auto; display:block; margin: 0.4em 0;' "
        f"src='data:image/png;base64,{encoded}' "
        "alt='LaTeX equation'/>"
    )
