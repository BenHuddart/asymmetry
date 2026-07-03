"""Tests for GUI LaTeX rendering helpers."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from asymmetry.gui.utils.latex_renderer import (
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
