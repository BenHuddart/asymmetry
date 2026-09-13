"""The shared ⓘ popover frame.

``InfoPopover`` is the chrome behind every ⓘ indicator: the popup window, the
two-column row grid with its width cap, and the placement under the anchor that
opened it. The Data Browser's phase popover subclasses it
(``tests/gui/test_data_browser_phases.py`` covers that surface); the Batch tab's
parameter-role help uses it as it stands.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from asymmetry.gui.widgets.info_popover import InfoPopover  # noqa: E402

_ROWS = [
    ("Global", "one value shared by every run"),
    ("Local", "fitted separately for each run"),
]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _texts(popover: InfoPopover) -> list[str]:
    return [label.text() for label in popover.findChildren(QLabel) if label.text()]


def test_rows_render_as_name_and_text(qapp) -> None:
    popover = InfoPopover()
    try:
        popover.set_rows(_ROWS)
        assert _texts(popover) == [
            "Global",
            "one value shared by every run",
            "Local",
            "fitted separately for each run",
        ]
    finally:
        popover.close()
        popover.deleteLater()


def test_the_name_column_is_bold(qapp) -> None:
    """A definition list: the term stands out from its description."""
    popover = InfoPopover()
    try:
        popover.set_rows(_ROWS)
        names = [label for label in popover.findChildren(QLabel) if label.text() == "Global"]
        assert [label.font().bold() for label in names] == [True]
    finally:
        popover.close()
        popover.deleteLater()


def test_a_pathological_row_is_elided_at_the_cap(qapp) -> None:
    """The frame sizes itself to its rows, but never past the width cap."""
    popover = InfoPopover()
    try:
        popover.set_rows([("Model", "x" * 400)])
        assert popover.width() <= popover.maximumWidth()
        assert "…" in " ".join(_texts(popover))
    finally:
        popover.close()
        popover.deleteLater()


def test_show_below_puts_the_frame_under_its_anchor(qapp) -> None:
    anchor = QPushButton("ⓘ")
    anchor.resize(30, 20)
    anchor.show()
    popover = InfoPopover()
    try:
        popover.set_rows(_ROWS)
        popover.show_below(anchor)

        assert popover.isVisible()
        assert popover.pos() == anchor.mapToGlobal(anchor.rect().bottomLeft())
        # A Qt.Popup closes itself on the first click outside the frame.
        assert popover.windowFlags() & Qt.WindowType.Popup
    finally:
        popover.close()
        popover.deleteLater()
        anchor.close()
        anchor.deleteLater()
