"""Tests for the SectionOverflowIndicator viewport pill."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea, QVBoxLayout, QWidget

from asymmetry.gui.widgets.section_overflow_indicator import SectionOverflowIndicator


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _tall_scroll_area(section_height: int = 400, viewport_height: int = 300):
    """A short scroll area over a tall 3-section content widget.

    Returns ``(scroll, [top, middle, bottom])`` where each section is a fixed-height
    labelled widget so the middle/bottom fall below a short viewport's fold.
    """
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    sections = []
    for name in ("Top", "Middle", "Bottom"):
        widget = QLabel(name)
        widget.setFixedHeight(section_height)
        layout.addWidget(widget)
        sections.append(widget)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.resize(200, viewport_height)
    return scroll, sections


def _settle(scroll: QScrollArea) -> None:
    scroll.show()
    QApplication.processEvents()


def test_pill_names_the_hidden_sections(qapp):
    scroll, sections = _tall_scroll_area()
    labels = ["Top", "Middle", "Bottom"]
    pill = SectionOverflowIndicator(scroll, lambda: list(zip(labels, sections)))
    _settle(scroll)
    pill.refresh()

    # At the top of a 300px viewport with 400px sections, Middle and Bottom are
    # below the fold; Top is not.
    assert pill.isVisible()
    assert pill.text() == "↓ Middle · Bottom"


def test_click_scrolls_first_hidden_into_view_and_updates_label(qapp):
    scroll, sections = _tall_scroll_area()
    labels = ["Top", "Middle", "Bottom"]
    pill = SectionOverflowIndicator(scroll, lambda: list(zip(labels, sections)))
    _settle(scroll)
    pill.refresh()
    assert pill.text() == "↓ Middle · Bottom"

    pill.click()
    QApplication.processEvents()

    # The first hidden section (Middle) is now visible in the viewport, so the
    # label no longer names it. Bottom may still be below the fold.
    viewport = scroll.viewport()
    middle_top = sections[1].mapTo(viewport, sections[1].rect().topLeft()).y()
    assert middle_top < viewport.height()
    assert "Middle" not in pill.text()


def test_refresh_drops_a_section_hidden_via_setvisible(qapp):
    scroll, sections = _tall_scroll_area()
    labels = ["Top", "Middle", "Bottom"]
    pill = SectionOverflowIndicator(scroll, lambda: list(zip(labels, sections)))
    _settle(scroll)
    pill.refresh()
    assert pill.text() == "↓ Middle · Bottom"

    sections[1].setVisible(False)
    pill.refresh()

    # Middle is no longer a visible section, so the pill only names Bottom.
    assert pill.isVisible()
    assert pill.text() == "↓ Bottom"


def test_pill_hidden_when_content_fits(qapp):
    # Tall viewport, short sections -> everything fits, no pill.
    scroll, sections = _tall_scroll_area(section_height=80, viewport_height=600)
    labels = ["Top", "Middle", "Bottom"]
    pill = SectionOverflowIndicator(scroll, lambda: list(zip(labels, sections)))
    _settle(scroll)
    pill.refresh()

    assert not pill.isVisible()


def test_empty_sections_hide_the_pill(qapp):
    scroll, _ = _tall_scroll_area()
    pill = SectionOverflowIndicator(scroll, lambda: [])
    _settle(scroll)
    pill.refresh()

    assert not pill.isVisible()
