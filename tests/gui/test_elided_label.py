"""Tests for the shared ElidedLabel (zero minimum width, elide + tooltip)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from asymmetry.gui.widgets.elided_label import ElidedLabel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_minimum_width_is_zero_regardless_of_text(qapp) -> None:
    label = ElidedLabel("a very long status line · that would otherwise set a hard minimum width")
    assert label.minimumSizeHint().width() == 0
    # The height hint stays a real text height (the row must not collapse).
    assert label.minimumSizeHint().height() > 0


def test_tooltip_only_while_elided(qapp) -> None:
    text = "10.000 ns × 4 detectors · max correction at t=0: 99999900.0%"
    label = ElidedLabel(text)
    # Resize events reach hidden widgets only on show, so show first; from then
    # on resizeEvent refreshes the tooltip eagerly (no paint needed).
    label.show()
    label.resize(60, 18)
    assert label.toolTip() == text

    label.resize(label.fontMetrics().horizontalAdvance(text) + 20, 18)
    assert label.toolTip() == ""

    label.close()


def test_hover_text_overrides_the_elision_rule(qapp) -> None:
    label = ElidedLabel("2 series")
    label.show()
    label.resize(400, 18)
    # Nothing is hidden, so the default rule would leave no tooltip at all.
    assert label.toolTip() == ""

    label.set_hover_text("Series A\nSeries B")
    assert label.toolTip() == "Series A\nSeries B"
    # It survives both a relayout and a new text.
    label.resize(20, 18)
    assert label.toolTip() == "Series A\nSeries B"
    label.setText("3 series")
    assert label.toolTip() == "Series A\nSeries B"

    label.set_hover_text("")
    assert label.toolTip() == "3 series"
    label.close()


def test_pen_color_override_and_default(qapp) -> None:
    label = ElidedLabel("x")
    default = label.pen_color()
    label.set_pen_color("#a8332a")
    assert label.pen_color().name() == "#a8332a"
    assert default.isValid()
