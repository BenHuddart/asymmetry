"""Standalone unit tests for the wrapping FlowLayout."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QRect  # type: ignore
from PySide6.QtWidgets import QApplication, QPushButton, QWidget  # type: ignore

from asymmetry.gui.widgets.flow_layout import FlowLayout


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _host(count: int = 4) -> tuple[QWidget, FlowLayout, list[QPushButton]]:
    host = QWidget()
    layout = FlowLayout(host)
    buttons = [QPushButton(f"Parameter {index}", host) for index in range(count)]
    for button in buttons:
        layout.addWidget(button)
    return host, layout, buttons


def test_items_share_one_row_when_wide(qapp: QApplication) -> None:
    host, layout, buttons = _host()
    widest = sum(button.sizeHint().width() for button in buttons) + 200

    layout.setGeometry(QRect(0, 0, widest, 400))

    tops = {button.geometry().top() for button in buttons}
    assert len(tops) == 1
    lefts = [button.geometry().left() for button in buttons]
    assert lefts == sorted(lefts)
    host.deleteLater()


def test_items_wrap_when_narrow(qapp: QApplication) -> None:
    host, layout, buttons = _host()
    narrow = buttons[0].sizeHint().width() + 4

    layout.setGeometry(QRect(0, 0, narrow, 400))

    tops = [button.geometry().top() for button in buttons]
    assert len(set(tops)) == len(buttons)
    assert tops == sorted(tops)
    host.deleteLater()


def test_height_for_width_grows_as_width_shrinks(qapp: QApplication) -> None:
    host, layout, buttons = _host()
    widest = sum(button.sizeHint().width() for button in buttons) + 200
    narrow = buttons[0].sizeHint().width() + 4

    assert layout.hasHeightForWidth()
    assert layout.heightForWidth(narrow) > layout.heightForWidth(widest)
    host.deleteLater()


def test_item_protocol_counts_and_takes(qapp: QApplication) -> None:
    host, layout, buttons = _host(3)

    assert layout.count() == 3
    assert layout.itemAt(0).widget() is buttons[0]
    assert layout.itemAt(3) is None

    taken = layout.takeAt(0)
    assert taken.widget() is buttons[0]
    assert layout.count() == 2
    assert layout.itemAt(0).widget() is buttons[1]
    assert layout.takeAt(5) is None
    host.deleteLater()
