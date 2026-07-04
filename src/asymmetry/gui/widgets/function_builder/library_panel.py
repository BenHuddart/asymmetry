"""Searchable component library panel (left side of the function builder).

Contract consumed by :mod:`asymmetry.gui.widgets.function_builder.dialog`:

- ``ComponentLibraryPanel(component_definitions, parent=None)``
- ``component_activated = Signal(str)`` — emitted with the component name when
  the user double-clicks an entry, presses Enter in the search flow, or presses
  Enter/Return with a row selected in the tree.
- ``set_components(component_definitions)`` — replace the searchable pool
  (e.g. on domain switch), preserving the current search text.
- ``current_component_name() -> str | None`` — the highlighted entry.
- ``set_search_text(text)`` / ``search_text()`` — programmatic search access.

Behavior
--------
An empty query groups every component under category headers (canonical
order from :func:`search_components`); a non-empty query flattens the list
into ranked search hits, highlighting the matched span in the name and
annotating non-name matches (alias/parameter/description/category/fuzzy)
with a short muted reason. See module-level widgets below for the rest of
the affordances (info button, user/missing badges, keyboard flow).
"""

from __future__ import annotations

import html
from collections.abc import Mapping

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.component_search import SearchResult, search_components
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

_NAME_ROLE = int(Qt.ItemDataRole.UserRole)
_CATEGORY_ITEM_TYPE = QTreeWidgetItem.ItemType.UserType + 1
_COMPONENT_ITEM_TYPE = QTreeWidgetItem.ItemType.UserType + 2

#: Short, human-readable annotation for each non-name ``matched_field`` value,
#: shown muted next to the component name so a hit that isn't an obvious
#: substring of the name is still explainable at a glance.
_FIELD_ANNOTATIONS: dict[str, str] = {
    "alias": "alias",
    "category": "category",
    "param": "parameter",
    "description": "description",
    "fuzzy": "similar name",
}


def _category_of(definition: object) -> str:
    return getattr(definition, "category", "General") or "General"


class _RowLabel(QLabel):
    """Rich-text row label used as the tree's per-item widget.

    Using ``setItemWidget`` (rather than a delegate) keeps the highlight
    styling in plain Qt rich text while the tree itself still drives keyboard
    navigation, selection, and the standard selected-row background — the
    label paints on top of (and does not intercept) that background because
    it never sets an opaque one of its own.
    """

    def __init__(self, html_text: str, *, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(html_text, parent)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setToolTip(tooltip)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setContentsMargins(2, 0, 2, 0)


def _escape(text: str) -> str:
    return html.escape(text)


def _name_html(name: str, name_span: tuple[int, int] | None, *, muted: bool) -> str:
    color = tokens.TEXT_MUTED if muted else tokens.TEXT
    if name_span is None:
        return f"<span style='color:{color};'>{_escape(name)}</span>"

    start, end = name_span
    start = max(0, min(start, len(name)))
    end = max(start, min(end, len(name)))
    before, matched, after = name[:start], name[start:end], name[end:]
    highlight_color = tokens.ACCENT
    return (
        f"<span style='color:{color};'>{_escape(before)}"
        f"<b style='color:{highlight_color};'>{_escape(matched)}</b>"
        f"{_escape(after)}</span>"
    )


def _row_html(
    name: str,
    *,
    name_span: tuple[int, int] | None = None,
    annotation: str | None = None,
    is_user: bool = False,
    is_missing: bool = False,
) -> str:
    parts = [_name_html(name, name_span, muted=is_missing)]
    if is_user:
        parts.append(f"<span style='color:{tokens.TEXT_MUTED};'>&#8194;&middot; user</span>")
    if annotation:
        parts.append(
            f"<span style='color:{tokens.TEXT_DIM};'>&#8194;&middot; {_escape(annotation)}</span>"
        )
    if is_missing:
        parts.append(f"<span style='color:{tokens.WARN};'>&#8194;&middot; missing</span>")
    return "".join(parts)


class ComponentLibraryPanel(QWidget):
    """Search box over a ranked, category-grouped component tree."""

    component_activated = Signal(str)  # noqa: N815

    def __init__(
        self,
        component_definitions: Mapping[str, object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._definitions: dict[str, object] = dict(component_definitions)
        self.setMinimumWidth(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._search_edit = QLineEdit(self)
        self._search_edit.setPlaceholderText("Search functions")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._refresh)
        self._search_edit.returnPressed.connect(self._activate_current)
        self._search_edit.installEventFilter(self)
        layout.addWidget(self._search_edit)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(2)
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(False)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(False)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.itemActivated.connect(self._on_item_activated)
        self._tree.itemSelectionChanged.connect(self._update_info_button_state)
        self._tree.installEventFilter(self)
        layout.addWidget(self._tree, 1)

        self._empty_label = QLabel(self)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; padding: 8px;")
        self._empty_label.setVisible(False)
        layout.addWidget(self._empty_label)

        info_row = QHBoxLayout()
        info_row.addStretch(1)
        self._info_button = QPushButton("ⓘ Info", self)
        self._info_button.setToolTip("Show details for the selected function")
        self._info_button.clicked.connect(self._show_current_info)
        info_row.addWidget(self._info_button)
        layout.addLayout(info_row)

        self._refresh()

    # -- Public contract ---------------------------------------------------

    def set_components(self, component_definitions: Mapping[str, object]) -> None:
        """Replace the searchable pool, keeping the current query text."""
        self._definitions = dict(component_definitions)
        self._refresh()

    def current_component_name(self) -> str | None:
        item = self._tree.currentItem()
        if item is None or item.type() != _COMPONENT_ITEM_TYPE:
            return None
        name = item.data(0, _NAME_ROLE)
        return name if isinstance(name, str) else None

    def search_text(self) -> str:
        return self._search_edit.text()

    def set_search_text(self, text: str) -> None:
        self._search_edit.setText(text)

    # -- Internal: (re)build the tree ---------------------------------------

    def _refresh(self) -> None:
        self._tree.clear()
        query = self._search_edit.text()
        stripped = query.strip()

        if not stripped:
            self._populate_grouped()
        else:
            self._populate_flat(stripped)

        self._select_first_component()
        self._update_info_button_state()

    def _populate_grouped(self) -> None:
        self._empty_label.setVisible(False)
        self._tree.setRootIsDecorated(False)
        results = search_components("", components=self._definitions)

        categories_in_order: list[str] = []
        seen_categories: set[str] = set()
        for result in results:
            definition = self._definitions.get(result.name)
            category = _category_of(definition)
            if category not in seen_categories:
                seen_categories.add(category)
                categories_in_order.append(category)

        category_items: dict[str, QTreeWidgetItem] = {}
        for category in categories_in_order:
            category_item = QTreeWidgetItem(self._tree, _CATEGORY_ITEM_TYPE)
            category_item.setText(0, category)
            category_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            font = category_item.font(0)
            font.setBold(True)
            category_item.setFont(0, font)
            category_item.setForeground(0, _brush(tokens.TEXT_MUTED))
            category_items[category] = category_item

        for result in results:
            definition = self._definitions.get(result.name)
            category = _category_of(definition)
            parent_item = category_items[category]
            self._add_component_item(parent_item, result, definition)

        self._tree.expandAll()

    def _populate_flat(self, query: str) -> None:
        self._tree.setRootIsDecorated(False)
        results = search_components(query, components=self._definitions)

        if not results:
            self._empty_label.setText("No matches — try 'KT', 'muonium', 'background'…")
            self._empty_label.setVisible(True)
            return

        self._empty_label.setVisible(False)
        for result in results:
            definition = self._definitions.get(result.name)
            self._add_component_item(self._tree, result, definition)

    def _add_component_item(
        self,
        parent: QTreeWidget | QTreeWidgetItem,
        result: SearchResult,
        definition: object | None,
    ) -> None:
        item = QTreeWidgetItem(parent, _COMPONENT_ITEM_TYPE)
        item.setData(0, _NAME_ROLE, result.name)
        description = getattr(definition, "description", "") or ""
        item.setToolTip(0, description)

        is_user = bool(getattr(definition, "user", False))
        is_missing = bool(getattr(definition, "missing", False))
        annotation = None
        if result.matched_field != "name" and result.matched_field in _FIELD_ANNOTATIONS:
            annotation = _FIELD_ANNOTATIONS[result.matched_field]

        row_html = _row_html(
            result.name,
            name_span=result.name_span,
            annotation=annotation,
            is_user=is_user,
            is_missing=is_missing,
        )
        label = _RowLabel(row_html, tooltip=description or result.name, parent=self._tree)
        self._tree.setItemWidget(item, 0, label)
        # Keep a plain-text fallback in column 0 for accessibility/testing
        # tools that read QTreeWidgetItem.text() rather than the item widget.
        item.setText(0, result.name)

    def _select_first_component(self) -> None:
        iterator_item = self._first_component_item()
        if iterator_item is not None:
            self._tree.setCurrentItem(iterator_item)

    def _first_component_item(self) -> QTreeWidgetItem | None:
        root = self._tree.invisibleRootItem()
        return self._first_component_item_under(root)

    def _first_component_item_under(self, item: QTreeWidgetItem) -> QTreeWidgetItem | None:
        for index in range(item.childCount()):
            child = item.child(index)
            if child.type() == _COMPONENT_ITEM_TYPE:
                return child
            found = self._first_component_item_under(child)
            if found is not None:
                return found
        return None

    # -- Activation / selection ----------------------------------------------

    def _activate_current(self) -> None:
        name = self.current_component_name()
        if name:
            self.component_activated.emit(name)

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.type() != _COMPONENT_ITEM_TYPE:
            return
        name = item.data(0, _NAME_ROLE)
        if isinstance(name, str):
            self.component_activated.emit(name)

    def _update_info_button_state(self) -> None:
        self._info_button.setEnabled(self.current_component_name() is not None)

    def _show_current_info(self) -> None:
        name = self.current_component_name()
        if not name:
            return
        definition = self._definitions.get(name)
        if definition is None:
            return
        show_component_info_dialog(self, definition)

    # -- Keyboard flow --------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._search_edit and event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Down:
                self._move_selection(1)
                return True
            if key == Qt.Key.Key_Up:
                self._move_selection(-1)
                return True
            if key == Qt.Key.Key_Escape:
                self._search_edit.clear()
                return True
        if watched is self._tree and event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._activate_current()
                return True
        return super().eventFilter(watched, event)

    def _move_selection(self, delta: int) -> None:
        items = self._component_items_in_display_order()
        if not items:
            return
        current = self.current_component_name()
        if current is None:
            index = 0
        else:
            positions = [i for i, item in enumerate(items) if item.data(0, _NAME_ROLE) == current]
            index = (positions[0] + delta) if positions else 0
            index = max(0, min(index, len(items) - 1))
        self._tree.setCurrentItem(items[index])
        self._tree.scrollToItem(items[index])

    def _component_items_in_display_order(self) -> list[QTreeWidgetItem]:
        items: list[QTreeWidgetItem] = []

        def _walk(parent: QTreeWidgetItem) -> None:
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child.type() == _COMPONENT_ITEM_TYPE:
                    items.append(child)
                _walk(child)

        _walk(self._tree.invisibleRootItem())
        return items


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color))
