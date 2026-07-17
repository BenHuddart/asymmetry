"""A viewport-overlay pill that names the scroll sections hidden below the fold.

A tall stack of labelled sections inside a :class:`QScrollArea` can push the
sections users most need to discover (background subtraction, α calibration)
entirely below the visible fold on a short window, with nothing on screen to
say they exist. This widget overlays a small pill at the bottom-right of the
scroll viewport naming those hidden sections — ``↓ Background · α (detector
balance)`` — and clicking it scrolls the first one into view. When nothing is
hidden the pill is invisible, so at sizes where everything fits it never shows.

It owns no threads and does no work of its own. Recompute is driven purely by
cheap geometry reads on the scrollbar's ``valueChanged``/``rangeChanged`` (the
latter covers content-height changes, e.g. a section growing/shrinking as a
mode toggles), on viewport resize (via an event filter), and on an explicit
:meth:`refresh` the owner calls when a section's visibility flips (so the label
tracks e.g. a section hidden in vector mode). ``sections`` is a *callable*
returning the current ``list[(label, widget)]`` for exactly that reason — the
set of sections is not fixed.
"""

from __future__ import annotations

from collections.abc import Callable

import shiboken6
from PySide6.QtCore import QEvent, QObject, QPoint, Qt
from PySide6.QtWidgets import QPushButton, QScrollArea, QWidget

from asymmetry.gui.styles import tokens

__all__ = ["SectionOverflowIndicator"]

#: Gap (px) between the pill and the viewport's bottom/right edges.
_MARGIN = 8


class SectionOverflowIndicator(QPushButton):
    """Overlay pill naming the *sections* currently hidden below *scroll_area*'s fold.

    *sections* is a callable returning ``list[(label, widget)]`` in top-to-bottom
    order; it is re-invoked on every recompute so callers can vary the set (e.g.
    dropping the α section in vector mode). Clicking scrolls the first hidden
    section into view.
    """

    def __init__(
        self,
        scroll_area: QScrollArea,
        sections: Callable[[], list[tuple[str, QWidget]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(scroll_area if parent is None else parent)
        self._scroll = scroll_area
        self._sections = sections
        self._first_hidden: QWidget | None = None

        # A small fixed-footprint pill: it accepts clicks on itself only and, as a
        # plain button, ignores wheel events so they propagate to the viewport —
        # no grab, no event stealing.
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAutoDefault(False)
        self.setDefault(False)
        self.setToolTip("Scroll the first hidden section into view")
        self.setStyleSheet(
            "SectionOverflowIndicator {"
            f"  background-color: {tokens.SURFACE_ALT};"
            f"  color: {tokens.TEXT_MUTED};"
            f"  border: 1px solid {tokens.BORDER};"
            "  border-radius: 10px;"
            "  padding: 2px 10px;"
            "  font-size: 11px;"
            "}"
            "SectionOverflowIndicator:hover {"
            f"  background-color: {tokens.SURFACE_HI};"
            f"  border-color: {tokens.BORDER_STRONG};"
            "}"
        )
        self.clicked.connect(self._on_clicked)

        self._viewport = scroll_area.viewport()
        vbar = scroll_area.verticalScrollBar()
        vbar.valueChanged.connect(self._recompute)
        vbar.rangeChanged.connect(self._recompute)
        self._viewport.installEventFilter(self)

        self.hide()
        self._recompute()

    def refresh(self) -> None:
        """Recompute the label and visibility — call after a section's visibility flips."""
        self._recompute()

    def _hidden_sections(self) -> list[tuple[str, QWidget]]:
        """The *sections* whose top sits at or below the viewport's bottom edge.

        Uses ``isVisibleTo(content)`` (not ``isVisible()``) so a section on the
        non-current tab still counts, while one explicitly ``setVisible(False)``
        is dropped. Each section's top is mapped into viewport coordinates — that
        mapping already folds in the scroll offset — and a top ``>= viewport
        height`` means it is fully below the fold.
        """
        content = self._scroll.widget()
        if content is None:
            return []
        viewport = self._viewport
        vp_height = viewport.height()
        hidden: list[tuple[str, QWidget]] = []
        for label, widget in self._sections() or []:
            if widget is None or not widget.isVisibleTo(content):
                continue
            top = widget.mapTo(viewport, QPoint(0, 0)).y()
            if top >= vp_height:
                hidden.append((label, widget))
        return hidden

    def _recompute(self, *_: object) -> None:
        # Scrollbar signals can fire while the scroll area is being torn down
        # (the pill is a child, destroyed alongside it); bail if it is gone.
        if not shiboken6.isValid(self._scroll):
            return
        hidden = self._hidden_sections()
        if not hidden:
            self._first_hidden = None
            self.setVisible(False)
            return
        self._first_hidden = hidden[0][1]
        self.setText("↓ " + " · ".join(label for label, _ in hidden))
        self.adjustSize()
        self.setVisible(True)
        self._reposition()
        self.raise_()

    def _reposition(self) -> None:
        """Anchor the pill to the bottom-right of the viewport (inside any scrollbar)."""
        geo = self._viewport.geometry()
        x = max(geo.left(), geo.right() - self.width() - _MARGIN)
        y = max(geo.top(), geo.bottom() - self.height() - _MARGIN)
        self.move(x, y)

    def _on_clicked(self) -> None:
        if self._first_hidden is not None:
            # ensureWidgetVisible scrolls; the resulting valueChanged recomputes
            # the label. The explicit recompute covers the no-scroll edge case.
            self._scroll.ensureWidgetVisible(self._first_hidden)
        self._recompute()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        # Identity check against the stored wrapper — never a method call on the
        # (possibly torn-down) scroll area.
        if obj is self._viewport and event.type() == QEvent.Type.Resize:
            self._recompute()
        return super().eventFilter(obj, event)
