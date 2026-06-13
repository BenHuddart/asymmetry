"""Size a stacked/tabbed container by its *current* page only."""

from __future__ import annotations

from PySide6.QtCore import QSize


class CurrentPageSizingMixin:
    """Make a QStackedWidget/QTabWidget size to its current page, not the maximum.

    A plain stacked or tab widget reports the largest size hint across *every*
    page, so a large hidden page (e.g. a wide Batch tab, or a tall multi-group
    fit surface) imposes its size on the whole container even while a compact
    page is showing — forcing the dock wider/taller than the visible content
    needs. Mixing this in sizes to the visible page instead. Subclasses add
    surrounding chrome (e.g. a QTabWidget's tab bar) via :meth:`_page_extra`.

    Mix in BEFORE the Qt base so the overrides win the MRO::

        class Deck(CurrentPageSizingMixin, QStackedWidget): ...
        class Tabs(CurrentPageSizingMixin, QTabWidget):
            def _page_extra(self): return self.tabBar().sizeHint()
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Page swaps change the (current-page-derived) hints; tell the layout.
        self.currentChanged.connect(lambda _index: self.updateGeometry())

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
