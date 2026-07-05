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
- ``create_requested = Signal()`` — emitted when the user asks to author a new
  user function (footer button, or the empty-search-results invitation).
  Opt-in: disabled by default via ``set_creation_enabled(False)``; a caller
  that wants the affordance calls ``set_creation_enabled(True)``.

Behavior
--------
An empty query groups every component under category headers (canonical
order from :func:`search_components`); a non-empty query flattens the list
into ranked search hits, highlighting the matched span in the name and
annotating non-name matches (alias/parameter/description/category/fuzzy)
with a short muted reason. Each row carries its own "+" (add) and info
buttons. See module-level widgets below for the rest of the affordances
(user/missing badges, keyboard flow).
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping

from PySide6.QtCore import QEvent, QObject, QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
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


def _glyph_icon(kind: str, color: str, *, size: int = 14, dpr: float = 2.0) -> QIcon:
    """Paint a crisp '+' or info glyph icon, independent of platform style.

    The per-row buttons must render identically under every QStyle and the
    app stylesheet: theme standard icons and bare text glyphs both proved
    style-dependent (blank chips on Windows), so the two glyphs are drawn
    directly at ``dpr``x resolution.
    """
    pixmap = QPixmap(int(size * dpr), int(size * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)

    center = size / 2.0
    if kind == "plus":
        arm = size * 0.30
        painter.drawLine(QPointF(center - arm, center), QPointF(center + arm, center))
        painter.drawLine(QPointF(center, center - arm), QPointF(center, center + arm))
    elif kind == "info":
        radius = size * 0.40
        painter.drawEllipse(QPointF(center, center), radius, radius)
        painter.drawLine(
            QPointF(center, center - radius * 0.05),
            QPointF(center, center + radius * 0.55),
        )
        painter.drawPoint(QPointF(center, center - radius * 0.5))
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown glyph kind: {kind!r}")
    painter.end()
    return QIcon(pixmap)


#: Local button style: the global app stylesheet gives buttons padding and a
#: filled background that leave no content area at row-button sizes, so the
#: row buttons opt out explicitly and draw only their glyph icon.
_ROW_BUTTON_QSS = (
    "QToolButton { border: none; background: transparent; padding: 0px; margin: 0px; }"
    "QToolButton:hover { background: rgba(0, 0, 0, 28); border-radius: 4px; }"
    "QToolButton:pressed { background: rgba(0, 0, 0, 48); border-radius: 4px; }"
)


class _RowSpec:
    """Immutable render recipe for one component row's rich-text label."""

    __slots__ = ("name", "name_span", "annotation", "is_user", "is_missing")

    def __init__(
        self,
        name: str,
        *,
        name_span: tuple[int, int] | None,
        annotation: str | None,
        is_user: bool,
        is_missing: bool,
    ) -> None:
        self.name = name
        self.name_span = name_span
        self.annotation = annotation
        self.is_user = is_user
        self.is_missing = is_missing


class _RowLabel(QLabel):
    """Rich-text row label used inside the tree's per-item row widget.

    This is the *sole* text-painting mechanism for a component row: the
    owning ``QTreeWidgetItem`` keeps its own ``text(0)`` empty so Qt never
    draws a second, unstyled copy of the name underneath this label. Using a
    widget (rather than a delegate) keeps the highlight styling in plain Qt
    rich text while the tree itself still drives keyboard navigation,
    selection, and the standard selected-row background — the label paints
    on top of (and does not intercept) that background because it never sets
    an opaque one of its own, and it stays transparent to mouse events so
    clicks fall through to row selection.

    A genuinely-too-long name (wider than the row even with no annotation)
    is elided with an ellipsis rather than silently clipped by the parent's
    paint rect — the full name (and description) is always available in the
    tooltip. ``_row_spec`` carries what's needed to re-elide on resize (e.g.
    a splitter drag), since eliding rich text has to operate on the plain
    name substring rather than the rendered HTML.
    """

    def __init__(self, row_spec: _RowSpec, *, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row_spec = row_spec
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setToolTip(tooltip)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setContentsMargins(2, 0, 2, 0)
        self._apply_html(elide_width=None)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_html(elide_width=event.size().width())

    def _apply_html(self, *, elide_width: int | None) -> None:
        available = elide_width if elide_width is not None else self.width()
        self.setText(_row_html(self._row_spec, available_width=available, font=self.font()))


class _RowWidget(QWidget):
    """Per-item row: rich-text label (stretch) + small add/info buttons.

    The label is transparent to mouse events (see ``_RowLabel``), so clicks
    anywhere in the label area fall through to the tree and select the row
    as usual. The buttons are ordinary (non-transparent) ``QToolButton``s;
    the ``on_add``/``on_info`` callbacks passed in by the panel select the
    owning row before running their action, so the action always applies to
    the row the user just clicked rather than whatever was previously
    current.
    """

    def __init__(
        self,
        row_spec: _RowSpec,
        *,
        tooltip: str,
        on_add: Callable[[], None],
        on_info: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(2)

        self.label = _RowLabel(row_spec, tooltip=tooltip, parent=self)
        layout.addWidget(self.label, 1)

        self.add_button = self._glyph_button(
            _glyph_icon("plus", tokens.TEXT),
            tooltip="Add this function",
            on_click=on_add,
        )
        layout.addWidget(self.add_button, 0)

        self.info_button = self._glyph_button(
            _glyph_icon("info", tokens.ACCENT),
            tooltip="Show details for this function",
            on_click=on_info,
        )
        layout.addWidget(self.info_button, 0)

        self._row_spec = row_spec

    def _glyph_button(
        self,
        icon: QIcon,
        *,
        tooltip: str,
        on_click: Callable[[], None],
    ) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(icon)
        button.setIconSize(QSize(14, 14))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setFixedSize(20, 20)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button.setStyleSheet(_ROW_BUTTON_QSS)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(on_click)
        return button

    def natural_width(self) -> int:
        """Un-elided width: full name text + badges + buttons + margins.

        Used as the tree item's size hint. This is computed analytically
        from the row spec rather than read back from the live label, since
        the label may already have elided its text in response to an
        earlier (narrower) layout pass by the time a size hint is queried.
        """
        spec = self._row_spec
        fm = QFontMetrics(self.label.font())
        name_width = fm.horizontalAdvance(spec.name)
        badge_width = 0
        if spec.is_user:
            badge_width += fm.horizontalAdvance("  · user")
        if spec.annotation:
            badge_width += fm.horizontalAdvance(f"  · {spec.annotation}")
        if spec.is_missing:
            badge_width += fm.horizontalAdvance("  · missing")

        margins = self.layout().contentsMargins()
        spacing = self.layout().spacing()
        buttons_width = self.add_button.width() + self.info_button.width()
        return (
            margins.left()
            + margins.right()
            + self.label.contentsMargins().left()
            + self.label.contentsMargins().right()
            + name_width
            + badge_width
            + spacing * 2
            + buttons_width
        )


def _escape(text: str) -> str:
    return html.escape(text)


def _elide_name(
    name: str, name_span: tuple[int, int] | None, *, max_width: int, font
) -> tuple[str, tuple[int, int] | None]:
    """Elide ``name`` to fit ``max_width`` px, dropping the highlight span if cut.

    Returns ``(name, name_span)`` unchanged when it already fits (the common
    case) or there is no width budget to test against yet.
    """
    if max_width <= 0:
        return name, name_span
    fm = QFontMetrics(font)
    if fm.horizontalAdvance(name) <= max_width:
        return name, name_span
    elided = fm.elidedText(name, Qt.TextElideMode.ElideRight, max_width)
    if elided == name:
        return name, name_span
    # The matched span may no longer correspond to visible text once elided;
    # rather than track its position through the ellipsis, drop the
    # highlight for this (rare, only-when-too-long) case. The full name is
    # still available in the tooltip.
    return elided, None


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


#: Reserved pixel width for the "· user" / "· <annotation>" / "· missing"
#: badges appended after the name — used to decide how much width the name
#: itself gets before eliding. This is a deliberately generous estimate (the
#: badges are short, muted words) rather than an exact font measurement, so
#: eliding stays cheap on every resize/repaint.
_BADGE_RESERVE_PX = 90


def _row_html(spec: _RowSpec, *, available_width: int | None = None, font=None) -> str:
    name, name_span = spec.name, spec.name_span
    if available_width is not None and font is not None:
        has_badge = spec.is_user or spec.annotation or spec.is_missing
        name_budget = available_width - (_BADGE_RESERVE_PX if has_badge else 0)
        name, name_span = _elide_name(name, name_span, max_width=name_budget, font=font)

    parts = [_name_html(name, name_span, muted=spec.is_missing)]
    if spec.is_user:
        parts.append(f"<span style='color:{tokens.TEXT_MUTED};'>&#8194;&middot; user</span>")
    if spec.annotation:
        parts.append(
            f"<span style='color:{tokens.TEXT_DIM};'>&#8194;&middot; {_escape(spec.annotation)}</span>"
        )
    if spec.is_missing:
        parts.append(f"<span style='color:{tokens.WARN};'>&#8194;&middot; missing</span>")
    return "".join(parts)


class ComponentLibraryPanel(QWidget):
    """Search box over a ranked, category-grouped component tree."""

    component_activated = Signal(str)  # noqa: N815
    create_requested = Signal()  # noqa: N815

    def __init__(
        self,
        component_definitions: Mapping[str, object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._definitions: dict[str, object] = dict(component_definitions)
        self._creation_enabled = False
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
        self._tree.setColumnCount(1)
        self._tree.header().setStretchLastSection(True)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(False)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(False)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.itemActivated.connect(self._on_item_activated)
        self._tree.installEventFilter(self)
        layout.addWidget(self._tree, 1)

        self._empty_label = QLabel(self)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; padding: 8px;")
        self._empty_label.setVisible(False)
        layout.addWidget(self._empty_label)

        # A second invitation shown only in the no-matches empty state, so a
        # search that comes up empty still offers a one-click way to author
        # the missing function rather than a dead end.
        self._empty_create_button = QPushButton("New user function…", self)
        self._empty_create_button.setVisible(False)
        self._empty_create_button.clicked.connect(self.create_requested.emit)
        layout.addWidget(self._empty_create_button)

        # Footer button: always visible (once creation is enabled), independent
        # of search state.
        self._footer_create_button = QPushButton("New user function…", self)
        self._footer_create_button.setVisible(False)
        self._footer_create_button.clicked.connect(self.create_requested.emit)
        layout.addWidget(self._footer_create_button)

        self._refresh()

    # -- Public contract ---------------------------------------------------

    def set_components(self, component_definitions: Mapping[str, object]) -> None:
        """Replace the searchable pool, keeping the current query text."""
        self._definitions = dict(component_definitions)
        self._refresh()

    def set_creation_enabled(self, enabled: bool) -> None:
        """Show/hide the "New user function…" affordances (default: hidden).

        When enabled, the footer button is always visible and the no-matches
        empty state additionally offers the same action.
        """
        self._creation_enabled = enabled
        self._footer_create_button.setVisible(enabled)
        # The empty-state button is only shown alongside the "no matches"
        # label, which _refresh already tracks — re-run it so the two stay
        # in sync with the current search state.
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

    def _populate_grouped(self) -> None:
        self._empty_label.setVisible(False)
        self._empty_create_button.setVisible(False)
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
            message = "No matches — try 'KT', 'muonium', 'background'…"
            if self._creation_enabled:
                message += " Or author a new one below."
            self._empty_label.setText(message)
            self._empty_label.setVisible(True)
            self._empty_create_button.setVisible(self._creation_enabled)
            return

        self._empty_label.setVisible(False)
        self._empty_create_button.setVisible(False)
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
        # Always lead with the full (un-elided) name, so a row whose visible
        # label has been elided to fit a narrow panel still exposes its full
        # name via hover — not just the description.
        tooltip = f"{result.name}\n{description}" if description else result.name
        item.setToolTip(0, tooltip)

        is_user = bool(getattr(definition, "user", False))
        is_missing = bool(getattr(definition, "missing", False))
        annotation = None
        if result.matched_field != "name" and result.matched_field in _FIELD_ANNOTATIONS:
            annotation = _FIELD_ANNOTATIONS[result.matched_field]

        name = result.name
        row_spec = _RowSpec(
            name,
            name_span=result.name_span,
            annotation=annotation,
            is_user=is_user,
            is_missing=is_missing,
        )
        row = _RowWidget(
            row_spec,
            tooltip=tooltip,
            on_add=lambda: self._activate_row(item, name),
            on_info=lambda: self._show_info_for_row(item, name),
            parent=self._tree,
        )
        self._tree.setItemWidget(item, 0, row)
        # Column 0 has no painted text of its own: the row widget's _RowLabel
        # is the sole text-painting mechanism, so Qt never draws a second,
        # unstyled copy of the name underneath it (that double-draw produced
        # smeared/bold-looking rows before this fix).
        item.setText(0, "")

        # Size hint reflects the row's *natural* (unelided) width so the
        # column/tree can grow to fit most names; a name that genuinely
        # doesn't fit the available column width still elides gracefully
        # (see _RowLabel.resizeEvent) rather than being clipped, with the
        # full name always available in the tooltip set above. Computed
        # analytically (not read back from the live label) since the label
        # may have already elided in response to an earlier layout pass.
        height = max(row.sizeHint().height(), 22)
        item.setSizeHint(0, QSize(row.natural_width(), height))

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

    def _activate_row(self, item: QTreeWidgetItem, name: str) -> None:
        """Handle a per-row "+" click: select the row, then activate it."""
        self._tree.setCurrentItem(item)
        self.component_activated.emit(name)

    def _show_info_for_row(self, item: QTreeWidgetItem, name: str) -> None:
        """Handle a per-row info click: select the row, then show its info."""
        self._tree.setCurrentItem(item)
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
