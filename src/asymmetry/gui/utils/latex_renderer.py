"""Render LaTeX-style math expressions for Qt widgets."""

from __future__ import annotations

import base64
import io
from functools import lru_cache

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


@lru_cache(maxsize=128)
def render_colored_equation_pixmap(
    fragments: tuple[tuple[str, str], ...],
    *,
    font_size: int = 13,
    dpi: int = 160,
) -> QPixmap | None:
    """Render *fragments* as one equation, each fragment tinted its own color.

    ``fragments`` is a sequence of ``(mathtext_fragment, color_hex)`` pairs,
    where each fragment is mathtext *without* surrounding ``$``. Fragments are
    drawn left-to-right on a shared baseline in a single matplotlib figure
    (rather than composited from separate per-fragment images), so ascenders/
    descenders and baselines line up exactly as they would in one expression.

    Measurement uses the cheapest approach available: draw each fragment via
    ``fig.text`` and read back its rendered pixel extent from the Agg canvas
    (``Text.get_window_extent``) after a canvas draw, then place the next
    fragment starting at the previous one's right edge. This avoids a second
    matplotlib backend (e.g. ``mathtext.math_to_image``, which only measures
    one string at a time and can't share a baseline across colors).

    Returns ``None`` (never raises) if matplotlib is unavailable, ``fragments``
    is empty, or any fragment fails to parse as mathtext. The background is
    transparent; the pixmap is rendered at 2x and marked with
    ``setDevicePixelRatio(2.0)`` so it stays crisp on HiDPI displays without
    appearing oversized in Qt layouts.
    """
    if not fragments:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib.figure import Figure
    except Exception:
        return None

    device_pixel_ratio = 2.0
    render_dpi = dpi * device_pixel_ratio

    try:
        # A generously-sized scratch figure; the true bbox is measured and
        # cropped below, so this canvas size only needs to be large enough
        # that no fragment clips against its edge.
        fig = Figure(figsize=(14, 2), dpi=render_dpi)
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        renderer = canvas.get_renderer()

        x_pos = 0.0
        text_artists = []
        for fragment, color in fragments:
            expression = fragment if fragment.startswith("$") else f"${fragment}$"
            text_artist = fig.text(
                0.0,
                0.5,
                expression,
                fontsize=font_size,
                color=color,
                transform=fig.dpi_scale_trans,
                horizontalalignment="left",
                verticalalignment="center",
            )
            # Position in inches via the figure's dpi-scale transform so each
            # fragment starts exactly where the previous one's measured extent
            # ended (converted from rendered pixels back to inches).
            text_artist.set_position((x_pos, 0.5))
            canvas.draw()
            bbox = text_artist.get_window_extent(renderer)
            width_in = bbox.width / render_dpi
            x_pos += width_in
            text_artists.append((text_artist, bbox))

        if not text_artists:
            return None

        # Overall tight bbox across every fragment, in display (pixel) coords.
        x0 = min(bbox.x0 for _artist, bbox in text_artists)
        x1 = max(bbox.x1 for _artist, bbox in text_artists)
        y0 = min(bbox.y0 for _artist, bbox in text_artists)
        y1 = max(bbox.y1 for _artist, bbox in text_artists)

        pad_px = max(4.0, font_size * 0.25 * device_pixel_ratio)
        x0 -= pad_px
        y0 -= pad_px
        x1 += pad_px
        y1 += pad_px

        width_in = max((x1 - x0) / render_dpi, 0.05)
        height_in = max((y1 - y0) / render_dpi, 0.05)

        # Re-render at the exact tight size so there is no wasted transparent
        # margin around the composed equation (a fresh figure sized to the
        # measured bbox, with each fragment repositioned relative to it).
        fig2 = Figure(figsize=(width_in, height_in), dpi=render_dpi)
        fig2.patch.set_alpha(0.0)
        canvas2 = FigureCanvasAgg(fig2)

        # Recompute each fragment's left edge relative to the new tight
        # origin, preserving the shared baseline (vertical center).
        x_pos = -x0 / render_dpi
        baseline_y = 0.5 - (y0 + y1) / 2.0 / render_dpi + height_in / 2.0
        for fragment, color in fragments:
            expression = fragment if fragment.startswith("$") else f"${fragment}$"
            text_artist = fig2.text(
                0.0,
                0.0,
                expression,
                fontsize=font_size,
                color=color,
                transform=fig2.dpi_scale_trans,
                horizontalalignment="left",
                verticalalignment="center",
            )
            text_artist.set_position((x_pos, baseline_y))
            canvas2.draw()
            bbox2 = text_artist.get_window_extent(canvas2.get_renderer())
            x_pos += bbox2.width / render_dpi

        buffer = io.BytesIO()
        fig2.savefig(buffer, format="png", dpi=render_dpi, transparent=True)
    except Exception:
        return None

    png = buffer.getvalue()
    if not png:
        return None

    pixmap = QPixmap()
    if not pixmap.loadFromData(png, "PNG"):
        return None
    pixmap.setDevicePixelRatio(device_pixel_ratio)
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
