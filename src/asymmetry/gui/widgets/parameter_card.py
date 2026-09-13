"""One card per trended parameter, and the stack that orders them.

The parameters panel plots each trended parameter on its own small figure with
the controls that act on it — Fit, log, the y lens — in the card's header rather
than in a table row somewhere else. A card is a pure view: it owns its canvas
and its chrome, renders what the panel hands it, and emits a signal (carrying
the parameter name) for every gesture. It imports nothing from
``asymmetry.core`` or from the panels, so it can be built and tested in
isolation. :class:`ParameterCardStack` owns the ordering, the focus mode, and
the drag-to-reorder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QMimeData, QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import metrics, tokens
from asymmetry.gui.styles.widgets import build_segmented_button_qss, make_context_chip
from asymmetry.gui.widgets.elided_label import ElidedLabel
from asymmetry.gui.widgets.mpl_canvas import create_canvas

__all__ = [
    "PARAMETER_CARD_MIME",
    "ParameterCard",
    "ParameterCardStack",
    "sparkline_pixmap",
]

#: Mime type carrying a dragged card's parameter name during a stack reorder.
PARAMETER_CARD_MIME = "application/x-asymmetry-parameter-card"

#: objectName so the focused/unfocused stylesheet targets only the surface frame
#: (a bare ``QFrame { … }`` rule would cascade onto the child controls).
_SURFACE_OBJECT_NAME = "parameterCardSurface"

#: Fixed size (px) of the colour-swatch square in the header.
_SWATCH_SIZE = 12

_GRIP_GLYPH = "⠿"
#: Focus button: "grow" while the card is one of many, "shrink" while focused.
_FOCUS_GLYPH = "⤢"
_UNFOCUS_GLYPH = "⤡"
#: Transform button's resting label — the identity lens has no formula to show.
_IDENTITY_TRANSFORM_LABEL = "ƒ"

#: Collapsed-card sparkline width, in average characters of the live application
#: font (its height is one table row), so the glyph tracks the UI zoom.
_SPARKLINE_CHARS = 14
_SPARKLINE_PAD = 2.0
_SPARKLINE_PEN_WIDTH = 1.2
_SPARKLINE_DOT_RADIUS = 1.4

#: Card body (canvas + summary + tools) minimum height, in table-row heights.
_BODY_ROWS = 6


def sparkline_pixmap(
    xs: Sequence[float],
    ys: Sequence[float],
    color: str,
    size: QSize,
    device_pixel_ratio: float = 1.0,
) -> QPixmap:
    """Return a transparent pixmap holding a tiny polyline-and-dots trend glyph.

    Painted with QPainter rather than matplotlib: a collapsed card shows this
    instead of its figure, so drawing it must cost nothing next to the figure it
    replaces. Non-finite points are dropped; a series with no spread in y draws
    along the middle of the pixmap rather than collapsing onto an edge.
    """
    width = max(1, size.width())
    height = max(1, size.height())
    pixmap = QPixmap(round(width * device_pixel_ratio), round(height * device_pixel_ratio))
    pixmap.setDevicePixelRatio(device_pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)

    points: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys, strict=True):
        x = float(raw_x)
        y = float(raw_y)
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if not points:
        return pixmap

    inner_w = width - 2 * _SPARKLINE_PAD
    inner_h = height - 2 * _SPARKLINE_PAD
    x_lo = min(x for x, _ in points)
    x_span = max(x for x, _ in points) - x_lo
    y_lo = min(y for _, y in points)
    y_span = max(y for _, y in points) - y_lo
    plotted = [
        QPointF(
            _SPARKLINE_PAD + (inner_w / 2 if x_span == 0 else inner_w * (x - x_lo) / x_span),
            _SPARKLINE_PAD
            + (inner_h / 2 if y_span == 0 else inner_h * (1.0 - (y - y_lo) / y_span)),
        )
        for x, y in points
    ]

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor(color), _SPARKLINE_PEN_WIDTH))
    painter.drawPolyline(plotted)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    for point in plotted:
        painter.drawEllipse(point, _SPARKLINE_DOT_RADIUS, _SPARKLINE_DOT_RADIUS)
    painter.end()
    return pixmap


class _CardHeader(QWidget):
    """The header strip; a left press on its background toggles the card open.

    Child buttons and the checkbox consume their own presses, so only a press on
    the strip itself (or on one of its plain labels) reaches here.
    """

    def __init__(self, on_pressed, parent: QWidget) -> None:
        super().__init__(parent)
        self._on_pressed = on_pressed
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_pressed()
        super().mousePressEvent(event)


class _DragGrip(QLabel):
    """The reorder handle; a left press starts the card's drag.

    Accepts the press (by not delegating to ``QLabel``) so grabbing the grip
    never also toggles the header underneath it. Right presses fall through so
    the header's context menu still opens over the grip.
    """

    def __init__(self, on_drag, parent: QWidget) -> None:
        super().__init__(_GRIP_GLYPH, parent)
        self._on_drag = on_drag
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip("Drag to reorder")
        self.setStyleSheet(f"color: {tokens.TEXT_DIM};")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_drag()
            return
        super().mousePressEvent(event)


class ParameterCard(QFrame):
    """Header (name + per-parameter controls) over a figure, summary and tools."""

    #: The card's Fit button was pressed.
    fit_requested = Signal(str)
    #: The card's log checkbox was toggled (name, checked).
    log_toggled = Signal(str, bool)
    #: The ƒ button was pressed; carries the global position to pop the menu at.
    transform_menu_requested = Signal(str, QPoint)
    #: The focus button was pressed — the stack decides what focus means.
    focus_toggled = Signal(str)
    #: The card was expanded or collapsed (name, expanded).
    expanded_changed = Signal(str, bool)
    #: The header was right-clicked; carries the global position.
    context_menu_requested = Signal(str, QPoint)

    def __init__(
        self,
        name: str,
        label: str,
        *,
        derived: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name = name
        self._expanded = True
        self._focused = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName(_SURFACE_OBJECT_NAME)
        outer.addWidget(self._surface)

        surface_layout = QVBoxLayout(self._surface)
        surface_layout.setContentsMargins(8, 6, 8, 6)
        surface_layout.setSpacing(4)

        # ── Header ──────────────────────────────────────────────────────────
        self._header = _CardHeader(self._toggle_expanded, self._surface)
        self._header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._header.customContextMenuRequested.connect(self._on_header_context_menu)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        surface_layout.addWidget(self._header)

        self._arrow = QToolButton(self._header)
        self._arrow.setAutoRaise(True)
        self._arrow.clicked.connect(self._toggle_expanded)
        header_layout.addWidget(self._arrow)

        self._swatch = QLabel(self._header)
        self._swatch.setFixedSize(_SWATCH_SIZE, _SWATCH_SIZE)
        header_layout.addWidget(self._swatch)

        self._name_label = ElidedLabel(label, self._header)
        # ElidedLabel paints its own text, so its colour comes from the pen, not
        # from a QSS `color:` rule; the weight still rides on the widget font.
        self._name_label.set_pen_color(tokens.TEXT)
        self._name_label.setStyleSheet("font-weight: 600;")
        header_layout.addWidget(self._name_label, 1)

        self.fit_button = QPushButton("Fit…", self._header)
        self.fit_button.setStyleSheet(build_segmented_button_qss())
        self.fit_button.clicked.connect(self._on_fit_clicked)
        header_layout.addWidget(self.fit_button)

        self.log_check = QCheckBox("log", self._header)
        self.log_check.toggled.connect(self._on_log_toggled)
        header_layout.addWidget(self.log_check)

        self.transform_button = QToolButton(self._header)
        self.transform_button.setText(_IDENTITY_TRANSFORM_LABEL)
        self.transform_button.setToolTip("Transform this parameter's y axis")
        self.transform_button.clicked.connect(self._on_transform_clicked)
        header_layout.addWidget(self.transform_button)

        if derived:
            header_layout.addWidget(make_context_chip("derived"))

        header_layout.addStretch(1)

        self._sparkline_label = QLabel(self._header)
        self._sparkline_label.setFixedSize(
            QSize(metrics.char_width(_SPARKLINE_CHARS), metrics.row_height())
        )
        header_layout.addWidget(self._sparkline_label)

        self.focus_button = QToolButton(self._header)
        self.focus_button.setText(_FOCUS_GLYPH)
        self.focus_button.setToolTip("Show this parameter on its own")
        self.focus_button.setAutoRaise(True)
        self.focus_button.clicked.connect(self._on_focus_clicked)
        header_layout.addWidget(self.focus_button)

        header_layout.addWidget(_DragGrip(self._start_drag, self._header))

        # ── Body ────────────────────────────────────────────────────────────
        self._body = QWidget(self._surface)
        self._body.setMinimumHeight(metrics.row_height() * _BODY_ROWS)
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(4)
        surface_layout.addWidget(self._body)

        self.figure, self.canvas = create_canvas(layout="constrained")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body_layout.addWidget(self.canvas, 1)

        self._summary_label = ElidedLabel("", self._body)
        self._summary_label.set_pen_color(tokens.TEXT_MUTED)
        body_layout.addWidget(self._summary_label)

        self._tools_row = QWidget(self._body)
        tools_layout = QHBoxLayout(self._tools_row)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(6)
        body_layout.addWidget(self._tools_row)

        self.add_label_button = QPushButton("Add label", self._tools_row)
        self.add_label_button.setCheckable(True)
        self.add_label_button.setStyleSheet(build_segmented_button_qss())
        tools_layout.addWidget(self.add_label_button)

        self.clear_labels_button = QPushButton("Clear labels", self._tools_row)
        tools_layout.addWidget(self.clear_labels_button)
        tools_layout.addStretch(1)
        self._tools_row.setVisible(False)

        self._apply_expanded()
        self.set_focused(False)

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """The parameter this card plots; fixed for the card's lifetime."""
        return self._name

    def set_swatch(self, color: str) -> None:
        """Tint the header swatch with the parameter's plot colour."""
        self._swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid {tokens.BORDER}; border-radius: 2px;"
        )

    def set_fit_label(self, text: str, tooltip: str = "") -> None:
        """Set the Fit button's label ("Fit…", "Fit ✓", "Global fit ×3…", …)."""
        self.fit_button.setText(text)
        self.fit_button.setToolTip(tooltip)

    def set_transform_label(self, text: str) -> None:
        """Show the active y lens on the ƒ button ("1/y"), or ƒ for identity."""
        self.transform_button.setText(text)

    def set_summary(self, text: str) -> None:
        """Set the one-line fit summary under the figure (elided; full on hover)."""
        self._summary_label.setText(text)

    def set_sparkline(self, xs: Sequence[float], ys: Sequence[float], color: str) -> None:
        """Render the collapsed-card trend glyph for this parameter's points."""
        self._sparkline_label.setPixmap(
            sparkline_pixmap(xs, ys, color, self._sparkline_label.size(), self.devicePixelRatioF())
        )

    def set_expanded(self, expanded: bool) -> None:
        """Show (or hide) the figure, summary and per-parameter controls."""
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._apply_expanded()
        self.expanded_changed.emit(self._name, expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_focused(self, focused: bool) -> None:
        """Apply (or drop) the focused card's accent border and flip the glyph.

        Purely the card's own chrome — the stack drives the expansion of this
        card and the collapse of its siblings.
        """
        self._focused = focused
        self.focus_button.setText(_UNFOCUS_GLYPH if focused else _FOCUS_GLYPH)
        if focused:
            self._surface.setStyleSheet(
                f"QFrame#{_SURFACE_OBJECT_NAME} {{ border: 2px solid {tokens.ACCENT};"
                f" background: {tokens.ACCENT_SOFT}; border-radius: 4px; }}"
            )
        else:
            self._surface.setStyleSheet(
                f"QFrame#{_SURFACE_OBJECT_NAME} {{ border: 1px solid {tokens.BORDER};"
                f" background: {tokens.SURFACE}; border-radius: 4px; }}"
            )

    def is_focused(self) -> bool:
        return self._focused

    def set_tools_visible(self, visible: bool) -> None:
        """Show the Add label / Clear labels row (the focused card's tools)."""
        self._tools_row.setVisible(visible)

    # ── Internals ───────────────────────────────────────────────────────────

    def _apply_expanded(self) -> None:
        self._arrow.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(self._expanded)
        self.fit_button.setVisible(self._expanded)
        self.log_check.setVisible(self._expanded)
        self.transform_button.setVisible(self._expanded)
        self._sparkline_label.setVisible(not self._expanded)

    def _toggle_expanded(self) -> None:
        self.set_expanded(not self._expanded)

    def _start_drag(self) -> None:
        data = QMimeData()
        data.setData(PARAMETER_CARD_MIME, self._name.encode())
        drag = QDrag(self)
        drag.setMimeData(data)
        drag.setPixmap(self._header.grab())
        drag.exec(Qt.DropAction.MoveAction)

    def _on_fit_clicked(self) -> None:
        self.fit_requested.emit(self._name)

    def _on_log_toggled(self, checked: bool) -> None:
        self.log_toggled.emit(self._name, checked)

    def _on_transform_clicked(self) -> None:
        below = QPoint(0, self.transform_button.height())
        self.transform_menu_requested.emit(self._name, self.transform_button.mapToGlobal(below))

    def _on_focus_clicked(self) -> None:
        self.focus_toggled.emit(self._name)

    def _on_header_context_menu(self, pos: QPoint) -> None:
        self.context_menu_requested.emit(self._name, self._header.mapToGlobal(pos))


class ParameterCardStack(QWidget):
    """A vertical stack of :class:`ParameterCard`s with focus and drag reorder."""

    #: The visual order changed; carries the parameter names top to bottom.
    order_changed = Signal(list)
    #: The focused parameter changed (its name, or ``None`` when focus is off).
    focus_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, ParameterCard] = {}
        self._focused: str | None = None
        #: Expansion of every card at the moment focus was entered, restored on exit.
        self._remembered: dict[str, bool] = {}

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        # Trailing spacer so a stack of collapsed cards sits at the top rather
        # than spreading; cards are always inserted in front of it.
        self._layout.addStretch(0)

        self.setAcceptDrops(True)

    # ── Membership ──────────────────────────────────────────────────────────

    def add_card(self, card: ParameterCard) -> None:
        """Append *card* to the bottom of the stack."""
        self._cards[card.name] = card
        if self._focused is not None:
            self._remembered[card.name] = card.is_expanded()
            card.set_expanded(False)
        card.focus_toggled.connect(self._on_card_focus_toggled)
        card.expanded_changed.connect(self._on_card_expanded_changed)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._apply_stretch()

    def remove_card(self, name: str) -> ParameterCard:
        """Detach the card for *name* and hand it back; the caller owns it."""
        if self._focused == name:
            self.set_focus(None)
        card = self._cards.pop(name)
        self._remembered.pop(name, None)
        card.focus_toggled.disconnect(self._on_card_focus_toggled)
        card.expanded_changed.disconnect(self._on_card_expanded_changed)
        self._layout.removeWidget(card)
        card.setParent(None)
        self._apply_stretch()
        return card

    def card(self, name: str) -> ParameterCard:
        return self._cards[name]

    def cards(self) -> list[ParameterCard]:
        """Every card, top to bottom."""
        widgets = (self._layout.itemAt(i).widget() for i in range(self._layout.count()))
        return [widget for widget in widgets if isinstance(widget, ParameterCard)]

    # ── Order ───────────────────────────────────────────────────────────────

    def set_order(self, names: list[str]) -> None:
        """Reorder to *names*; unknown names are ignored, unnamed cards trail."""
        wanted = [name for name in names if name in self._cards]
        ordered = [self._cards[name] for name in wanted]
        ordered += [card for card in self.cards() if card.name not in set(wanted)]
        for index, card in enumerate(ordered):
            self._layout.removeWidget(card)
            self._layout.insertWidget(index, card)
        self._apply_stretch()

    def move_card(self, name: str, index: int) -> None:
        """Move the card for *name* so it ends up at position *index*."""
        card = self._cards[name]
        self._layout.removeWidget(card)
        self._layout.insertWidget(min(max(index, 0), len(self._cards) - 1), card)
        self._apply_stretch()
        self.order_changed.emit([each.name for each in self.cards()])

    # ── Focus ───────────────────────────────────────────────────────────────

    def set_focus(self, name: str | None) -> None:
        """Give the whole stack to one card, or (``None``) restore every card.

        Entering focus remembers every card's expansion so leaving it — or
        focusing straight onto another card and then leaving — puts the stack
        back exactly as the user had it.
        """
        if name == self._focused:
            return
        if name is None:
            for card in self.cards():
                card.set_focused(False)
                card.set_tools_visible(False)
                card.set_expanded(self._remembered[card.name])
            self._remembered = {}
        else:
            if self._focused is None:
                self._remembered = {card.name: card.is_expanded() for card in self.cards()}
            for card in self.cards():
                focused = card.name == name
                card.set_focused(focused)
                card.set_tools_visible(focused)
                card.set_expanded(focused)
        self._focused = name
        self._apply_stretch()
        self.focus_changed.emit(name)

    def focused(self) -> str | None:
        return self._focused

    # ── Drag reorder ────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 — Qt override
        if self._dragged_name(event) in self._cards:
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 — Qt override
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 — Qt override
        name = self._dragged_name(event)
        order = [card.name for card in self.cards()]
        target = len(order)
        for index, card in enumerate(self.cards()):
            if event.position().y() < card.geometry().center().y():
                target = index
                break
        # The dragged card still occupies a slot above the target while the drop
        # index is computed, so dropping below its own position over-counts by one.
        if order.index(name) < target:
            target -= 1
        self.move_card(name, target)
        event.acceptProposedAction()

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _dragged_name(event: QDropEvent) -> str:
        return bytes(event.mimeData().data(PARAMETER_CARD_MIME)).decode()

    def _apply_stretch(self) -> None:
        """Give the height to the expanded cards; collapsed cards stay header-tall."""
        for card in self.cards():
            expanded = card.is_expanded()
            card.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Expanding if expanded else QSizePolicy.Policy.Fixed,
            )
            self._layout.setStretch(self._layout.indexOf(card), 1 if expanded else 0)

    def _on_card_focus_toggled(self, name: str) -> None:
        self.set_focus(None if self._focused == name else name)

    def _on_card_expanded_changed(self, _name: str, _expanded: bool) -> None:
        self._apply_stretch()
