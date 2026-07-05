"""Tests for GUI LaTeX rendering helpers."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from asymmetry.gui.utils.latex_renderer import (
    render_colored_equation_pixmap,
    render_latex_png_bytes,
    render_latex_to_html_image,
    render_latex_to_pixmap,
)


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_render_latex_to_pixmap(qapp: QApplication) -> None:
    pixmap = render_latex_to_pixmap(r"\lambda(B)=\frac{A}{1+(B/B_0)^2}")

    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0


def test_render_latex_png_bytes_cache_hits() -> None:
    render_latex_png_bytes.cache_clear()

    first = render_latex_png_bytes(r"A(t)=A e^{-\Lambda t}")
    info_after_first = render_latex_png_bytes.cache_info()
    second = render_latex_png_bytes(r"A(t)=A e^{-\Lambda t}")
    info_after_second = render_latex_png_bytes.cache_info()

    assert first is not None
    assert second == first
    assert info_after_second.hits == info_after_first.hits + 1


def test_render_latex_to_html_image_contains_data_uri() -> None:
    html_img = render_latex_to_html_image(r"y(T)=a e^{-E_a/(k_B T)}")

    assert html_img is not None
    assert "data:image/png;base64," in html_img


# --------------------------------------------------- composed colored equation
def _distinct_colors(pixmap) -> set[tuple[int, int, int]]:
    image = pixmap.toImage()
    colors: set[tuple[int, int, int]] = set()
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() > 50:
                colors.add((color.red(), color.green(), color.blue()))
    return colors


def test_render_colored_equation_pixmap_two_fragments(qapp: QApplication) -> None:
    fragments = (
        (r"A_1 e^{-\lambda t}", "#005a9c"),
        (r"+ A_2", "#a44a00"),
    )
    pixmap = render_colored_equation_pixmap(fragments)

    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0

    colors = _distinct_colors(pixmap)
    # Each fragment's glyphs render in its own color, so at least two
    # visually-distinct non-transparent colors should appear (antialiasing
    # adds many near-duplicates, hence a coarse bucket rather than an exact
    # count).
    buckets = {(r // 32, g // 32, b // 32) for r, g, b in colors}
    assert len(buckets) >= 2


def test_render_colored_equation_pixmap_garbage_fragment_returns_none(
    qapp: QApplication,
) -> None:
    render_colored_equation_pixmap.cache_clear()
    result = render_colored_equation_pixmap((("\\frac{", "#000000"),))

    assert result is None


def test_render_colored_equation_pixmap_empty_returns_none(qapp: QApplication) -> None:
    assert render_colored_equation_pixmap(()) is None


def test_render_colored_equation_pixmap_cache_hit_same_object(qapp: QApplication) -> None:
    render_colored_equation_pixmap.cache_clear()
    fragments = ((r"A_1", "#005a9c"),)

    first = render_colored_equation_pixmap(fragments)
    second = render_colored_equation_pixmap(fragments)

    assert first is not None
    assert second is first
