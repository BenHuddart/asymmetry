"""Size a stacked/tabbed container by its *current* page only."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QSize

#: Qt's QWIDGETSIZE_MAX — the "no maximum" sentinel for setMaximumHeight.
_QWIDGETSIZE_MAX = (1 << 24) - 1


class CurrentPageSizingMixin:
    """Make a QStackedWidget/QTabWidget size to its current page, not the maximum.

    A plain stacked or tab widget reports the largest size across *every* page,
    so a large hidden page (e.g. a wide Batch tab, or a tall multi-group fit
    surface) imposes its size on the whole container even while a compact page is
    showing. This sizes to the visible page instead, in two ways:

    * the size *hints* report the current page (so a parent layout reserves the
      right width/height), and
    * the container's maximum height is capped to the current page's content (or
      the available height, whichever is larger). The cap is the part that
      actually binds the height a ``widgetResizable`` QScrollArea hands the
      widget — without it the scroll area sizes to the tallest page's size hint
      and scrolls into empty space below a short visible page.

    Subclasses add surrounding chrome (e.g. a QTabWidget's tab bar) via
    :meth:`_page_extra`. Mix in BEFORE the Qt base so the overrides win the MRO::

        class Deck(CurrentPageSizingMixin, QStackedWidget): ...
        class Tabs(CurrentPageSizingMixin, QTabWidget):
            def _page_extra(self): return self.tabBar().sizeHint()
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Page swaps change the (current-page-derived) hints and height cap.
        self.currentChanged.connect(lambda _index: self._sync_to_current_page())

    def _page_extra(self) -> QSize:
        """Extra size added around the current page (e.g. a tab bar). Default none."""
        return QSize(0, 0)

    def _current_page_hint(self, *, minimum: bool) -> QSize:
        current = self.currentWidget()
        if current is None:
            return super().minimumSizeHint() if minimum else super().sizeHint()
        page = current.minimumSizeHint() if minimum else current.sizeHint()
        extra = self._page_extra()
        return QSize(max(page.width(), extra.width()), page.height() + extra.height())

    def sizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return self._current_page_hint(minimum=False)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 — Qt override
        return self._current_page_hint(minimum=True)

    def _sync_to_current_page(self) -> None:
        if self.currentWidget() is None:
            target = _QWIDGETSIZE_MAX
        else:
            content = self._current_page_hint(minimum=False).height()
            parent = self.parentWidget()
            available = parent.height() if parent is not None else 0
            # max(content, available): a short page still fills (and stops at) the
            # viewport — no scroll; a tall page scrolls exactly its own content.
            target = max(content, available)
        if self.maximumHeight() != target:
            self.setMaximumHeight(target)

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self._sync_to_current_page()  # the available height changed

    def event(self, event: QEvent) -> bool:
        handled = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self._sync_to_current_page()  # the current page's content changed
        return handled
