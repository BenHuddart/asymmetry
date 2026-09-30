"""Horizontal step navigator for the guided fit wizards.

A row of step buttons (Scope → Screen → Compare → Phases → Apply in the Global
Fit Wizard), each a state disc, a bold title and a one-line summary. The step's
:class:`StepState` says how far its work has got; which step is *current* (on
screen) is held separately, because a done, stale or running step can each be
the one being viewed. Design: ``docs/plans/global-wizard-stepper.md`` (D1, D10).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.typography import SIZE_BODY, SIZE_FOOTER


class StepState(Enum):
    """How far a step's work has got; the value is its accessible-name word."""

    PENDING = "pending"  # its inputs do not exist yet
    READY = "ready"  # its inputs exist and it has not run
    RUNNING = "running"
    DONE = "done"
    STALE = "stale"  # done, but an input has changed since
    SKIPPED = "skipped"  # does not apply to this analysis


#: States a user can navigate to; the current step is always reachable too.
_CLICKABLE = frozenset({StepState.READY, StepState.RUNNING, StepState.DONE, StepState.STALE})


@dataclass(frozen=True)
class _Look:
    disc_fill: str | None  # None: an outline-only disc
    disc_border: str
    mark: str
    title: str
    summary: str
    dashed: bool = False


_LOOKS = {
    StepState.PENDING: _Look(
        None, tokens.BORDER_STRONG, tokens.TEXT_MUTED, tokens.TEXT_MUTED, tokens.TEXT_DIM
    ),
    StepState.READY: _Look(None, tokens.ACCENT, tokens.ACCENT, tokens.TEXT, tokens.TEXT_MUTED),
    StepState.RUNNING: _Look(
        tokens.ACCENT_SOFT, tokens.ACCENT, tokens.ACCENT, tokens.TEXT, tokens.ACCENT
    ),
    StepState.DONE: _Look(
        tokens.SUCCESS_SOFT, tokens.OK, tokens.OK, tokens.TEXT, tokens.TEXT_MUTED
    ),
    StepState.STALE: _Look(tokens.WARN_SOFT, tokens.WARN, tokens.WARN, tokens.TEXT, tokens.WARN),
    StepState.SKIPPED: _Look(
        None, tokens.BORDER_STRONG, tokens.TEXT_DIM, tokens.TEXT_DIM, tokens.TEXT_DIM, dashed=True
    ),
}

#: The disc shows the step number unless its state has a mark (running paints an arc).
_MARKS = {StepState.DONE: "✓", StepState.STALE: "!"}

_PADDING = 8
_GAP = 10
_RADIUS = 8


class _StepButton(QPushButton):
    """One step: a painted disc, title and elided summary on a real, focusable button."""

    def __init__(self, number: int, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._number = number
        self._title = title
        self._state = StepState.PENDING
        self._summary = ""
        self._current = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh()

    @property
    def state(self) -> StepState:
        return self._state

    @property
    def is_current(self) -> bool:
        return self._current

    def show_step(self, state: StepState, summary: str) -> None:
        self._state = state
        self._summary = summary
        self._refresh()

    def set_current(self, current: bool) -> None:
        self._current = current
        self._refresh()

    def _refresh(self) -> None:
        self.setEnabled(self._current or self._state in _CLICKABLE)
        name = f"Step {self._number}, {self._title}, {self._state.value}"
        self.setAccessibleName(f"{name}, current" if self._current else name)
        self.setAccessibleDescription(self._summary)
        self.setToolTip(self._summary)
        self.update()

    def _fonts(self) -> tuple[QFont, QFont]:
        title = QFont(self.font())
        title.setBold(True)
        summary = QFont(self.font())
        summary.setPointSizeF(self.font().pointSizeF() * SIZE_FOOTER / SIZE_BODY)
        return title, summary

    def _line_heights(self) -> tuple[int, int]:
        title, summary = self._fonts()
        return QFontMetrics(title).height(), QFontMetrics(summary).height()

    def _size_for(self, text_width: int) -> QSize:
        # The disc is as tall as the two text lines beside it (26 px at the default font).
        title_height, summary_height = self._line_heights()
        disc = title_height + summary_height
        return QSize(2 * _PADDING + disc + _GAP + text_width, 2 * _PADDING + disc)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        title, summary = self._fonts()
        return self._size_for(
            max(
                QFontMetrics(title).horizontalAdvance(self._title),
                QFontMetrics(summary).horizontalAdvance(self._summary),
            )
        )

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        title, _summary = self._fonts()
        return self._size_for(QFontMetrics(title).horizontalAdvance(self._title))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        del event
        look = _LOOKS[self._state]
        title_font, summary_font = self._fonts()
        title_height, summary_height = self._line_heights()
        disc_size = title_height + summary_height
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        frame = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._current:
            painter.setPen(QPen(QColor(tokens.ACCENT_SOFT2)))
            painter.setBrush(QColor(tokens.ACCENT_SOFT))
            painter.drawRoundedRect(frame, _RADIUS, _RADIUS)
        elif self.isEnabled() and self.underMouse():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tokens.SURFACE_ALT))
            painter.drawRoundedRect(frame, _RADIUS, _RADIUS)
        if self.hasFocus() and self.window().testAttribute(
            Qt.WidgetAttribute.WA_KeyboardFocusChange
        ):
            painter.setPen(QPen(QColor(tokens.ACCENT), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(frame.adjusted(1, 1, -1, -1), _RADIUS, _RADIUS)

        disc = QRectF(_PADDING, (self.height() - disc_size) / 2, disc_size, disc_size)
        if self._current:
            fill, border, mark = tokens.ACCENT, tokens.ACCENT, tokens.WHITE
        else:
            fill, border, mark = look.disc_fill, look.disc_border, look.mark
        pen = QPen(QColor(border), 1.2)
        if look.dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor(fill) if fill is not None else Qt.BrushStyle.NoBrush)
        painter.drawEllipse(disc.adjusted(0.6, 0.6, -0.6, -0.6))
        if self._state is StepState.RUNNING:
            # A three-quarter ring: the busy mark.
            ring = disc.adjusted(
                disc_size * 0.3, disc_size * 0.3, -disc_size * 0.3, -disc_size * 0.3
            )
            ring_pen = QPen(QColor(mark), 2.0)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(ring, 90 * 16, -270 * 16)
        else:
            painter.setPen(QColor(mark))
            painter.setFont(title_font)
            painter.drawText(
                disc, Qt.AlignmentFlag.AlignCenter, _MARKS.get(self._state, str(self._number))
            )

        text_left = disc.right() + _GAP
        text_width = max(0.0, self.width() - _PADDING - text_left)
        top = disc.top()
        painter.setPen(QColor(tokens.TEXT if self._current else look.title))
        painter.setFont(title_font)
        painter.drawText(
            QRectF(text_left, top, text_width, title_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            QFontMetrics(title_font).elidedText(
                self._title, Qt.TextElideMode.ElideRight, int(text_width)
            ),
        )
        summary_colour = (
            tokens.ACCENT if self._current and self._state is not StepState.STALE else look.summary
        )
        painter.setPen(QColor(summary_colour))
        painter.setFont(summary_font)
        painter.drawText(
            QRectF(text_left, top + title_height, text_width, summary_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            QFontMetrics(summary_font).elidedText(
                self._summary, Qt.TextElideMode.ElideRight, int(text_width)
            ),
        )


class WizardStepper(QWidget):
    """A row of wizard steps; emits :attr:`step_requested` when one is clicked.

    Starts with every step pending and the first step current. The host owns
    navigation: it answers :attr:`step_requested` by showing that step's view and
    calling :meth:`set_current`.
    """

    step_requested = Signal(str)

    def __init__(self, steps: Sequence[tuple[str, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[str, _StepButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for number, (key, title) in enumerate(steps, start=1):
            button = _StepButton(number, title, self)
            button.clicked.connect(partial(self.step_requested.emit, key))
            layout.addWidget(button, 1)
            self._buttons[key] = button
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_current(steps[0][0])

    def set_step(self, key: str, state: StepState, summary: str) -> None:
        """Show step ``key`` in ``state`` with its one-line ``summary``."""
        self._buttons[key].show_step(state, summary)

    def set_current(self, key: str) -> None:
        """Mark step ``key`` as the one on screen."""
        if key not in self._buttons:
            raise KeyError(f"WizardStepper has no step {key!r}")
        for step_key, button in self._buttons.items():
            button.set_current(step_key == key)

    def current_key(self) -> str:
        return next(key for key, button in self._buttons.items() if button.is_current)

    def state(self, key: str) -> StepState:
        return self._buttons[key].state
