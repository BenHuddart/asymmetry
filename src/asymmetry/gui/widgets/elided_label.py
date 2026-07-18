"""A QLabel that elides right instead of imposing its text width as a minimum.

A plain non-wrapping ``QLabel``'s minimum size hint is its full text width, so
one long status line ("10.000 ns × 4 detectors · max correction at t=0: …")
can force a whole scroll column into a horizontal scrollbar. ``ElidedLabel``
reports a zero minimum width and paints elided ("…") when squeezed, showing
the full text as a tooltip only while elided. The pen colour is held directly
(:meth:`set_pen_color`) because the custom paint bypasses QSS colour rules;
it defaults to the palette's window text.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QLabel, QWidget

__all__ = ["ElidedLabel"]


class ElidedLabel(QLabel):
    """A right-eliding label that never widens its layout (see module doc)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._pen_color: QColor | None = None

    def set_pen_color(self, color: str) -> None:
        """Paint the text in *color* (a ``#rrggbb`` token) instead of the palette."""
        self._pen_color = QColor(color)
        self.update()

    def pen_color(self) -> QColor:
        """Current text colour (test seam)."""
        if self._pen_color is not None:
            return QColor(self._pen_color)
        return self.palette().color(self.foregroundRole())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt override
        # Width 0: the layout may shrink the label freely; paint elides.
        return QSize(0, super().minimumSizeHint().height())

    def _elided_text(self) -> str:
        return self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, max(0, self.width())
        )

    def _update_tooltip(self) -> None:
        # Full text on hover only when something is actually hidden. Kept out of
        # paintEvent so the tooltip is right even before the first paint.
        self.setToolTip(self.text() if self._elided_text() != self.text() else "")

    def setText(self, text: str) -> None:  # noqa: N802 — Qt override
        super().setText(text)
        self._update_tooltip()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._update_tooltip()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        painter = QPainter(self)
        painter.setPen(self.pen_color())
        painter.drawText(
            self.rect(),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._elided_text(),
        )
