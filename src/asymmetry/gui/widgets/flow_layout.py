"""Wrapping flow layout for strips of small items (chips, toggles, tags).

Places items left to right and starts a new line whenever the next item would
overflow the available width, reporting its height for that width so the strip
grows downwards instead of clipping. A ``QHBoxLayout`` would push the tail of a
chip row off-screen at narrow widths and a ``QGridLayout`` would freeze the
column count at whatever happens to fit today; this is the canonical Qt flow
layout, kept in one place so no panel grows its own copy.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QApplication, QLayout, QLayoutItem, QSizePolicy, QStyle, QWidget

__all__ = ["FlowLayout"]


def _style_spacing(orientation: Qt.Orientation) -> int:
    """Return the active style's layout spacing along *orientation*.

    Styles answer this through either of two entry points and return ``-1`` from
    the other: Fusion knows ``PM_Layout*Spacing`` but not the control-type
    ``layoutSpacing``, the macOS style the reverse. Taking the larger of the two
    picks whichever the style actually implements.
    """
    style = QApplication.style()
    metric = (
        QStyle.PixelMetric.PM_LayoutHorizontalSpacing
        if orientation == Qt.Orientation.Horizontal
        else QStyle.PixelMetric.PM_LayoutVerticalSpacing
    )
    return max(
        style.pixelMetric(metric),
        style.layoutSpacing(
            QSizePolicy.ControlType.DefaultType,
            QSizePolicy.ControlType.DefaultType,
            orientation,
        ),
    )


class FlowLayout(QLayout):
    """A layout that arranges its items in rows, wrapping at the available width."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h_space = _style_spacing(Qt.Orientation.Horizontal)
        self._v_space = _style_spacing(Qt.Orientation.Vertical)

    # ── QLayout item protocol ───────────────────────────────────────────────

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 — Qt override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 — Qt override
        # Qt walks a layout by calling this with rising indices until it gets
        # None, so out-of-range is a normal query rather than a caller error.
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 — Qt override
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    # ── Geometry ────────────────────────────────────────────────────────────

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802 — Qt override
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — Qt override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — Qt override
        return self._lay_out(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 — Qt override
        super().setGeometry(rect)
        self._lay_out(rect, apply_geometry=True)

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 — Qt override
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _lay_out(self, rect: QRect, *, apply_geometry: bool) -> int:
        """Place the items inside *rect*; return the height the content needs."""
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            # A hidden widget reports a zero size hint but would still cost a
            # spacing gap; rows that swap one button for another (Fit ↔ Stop) or
            # reveal a chip only after a fit must not leave that gap behind.
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_space
            if next_x - self._h_space > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + self._v_space
                next_x = x + hint.width() + self._h_space
                line_height = 0
            if apply_geometry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + margins.bottom()
