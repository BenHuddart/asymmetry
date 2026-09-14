"""The shared ⓘ popover: a frameless card of ``name → text`` rows.

Several BENCH headers carry a ⓘ indicator whose answer is a short list rather
than a sentence — what the Data Browser knows about a phase
(:class:`~asymmetry.gui.widgets.phase_info_popover.PhaseInfoPopover`), what each
role in the Batch tab's Type column means. This is the frame they share: a
``Qt.Popup`` window, so it grabs the mouse and closes itself on the first click
outside — no outside-click filter and no "is it open" flag to keep in sync with
the host — holding one :class:`~asymmetry.gui.widgets.key_value_grid.KeyValueGrid`
in a bordered surface.

Subclasses put their own widgets around the grid through :attr:`body_layout`;
everything else (chrome, the width cap, placement, dismissal) lives here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.metrics import char_width
from asymmetry.gui.styles.typography import SIZE_BODY, SIZE_NUMERIC
from asymmetry.gui.widgets.key_value_grid import KeyValueGrid

#: Ceiling on the value column, in characters of the monospaced font the grid
#: renders values in. The frame otherwise sizes itself to its content
#: (``adjustSize`` after each populate): a fixed width truncates the model title
#: and the boundary estimates on a perfectly ordinary phase (the longest row a
#: real phase carries, "Found by", is 43 characters). The cap only bites on a
#: pathological model title, which the fit panel spells out in full anyway.
_POPOVER_MAX_VALUE_CHARS = 60
#: Horizontal chrome outside the two text columns: the frame's two 12 px content
#: margins, the grid's 12 px column gap, and its 1 px border either side.
_POPOVER_CHROME_PX = 12 + 12 + 12 + 2


class InfoPopover(QFrame):
    """A frameless ``Qt.Popup`` card listing ``(name, text)`` rows."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("infoPopover")
        self.setStyleSheet(
            f"QFrame#infoPopover {{ background: {tokens.SURFACE}; "
            f"border: 1px solid {tokens.BORDER}; }}"
        )

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(8)
        self._grid = KeyValueGrid(self, bold_labels=True)
        self._layout.addWidget(self._grid)

    @property
    def body_layout(self) -> QVBoxLayout:
        """The frame's own column, so a subclass can add rows around the grid."""
        return self._layout

    def set_rows(self, rows: list[tuple[str, str]]) -> None:
        """Fill the grid, eliding any value that would push past the width cap.

        ``KeyValueGrid``'s value labels do not wrap, so their text sets the
        layout's *minimum* width — which outranks ``setMaximumWidth`` and would
        stretch the frame across the window for a long model title. Eliding the
        text instead keeps the cap real, and every row short of it (all of them,
        for the wizard's own titles) is still shown in full.

        The frame's own maximum follows from the two columns it ends up with,
        measured in the grid's fonts rather than the proportional UI font — the
        value column is monospaced and around a third wider per character.
        """
        label_metrics = QFontMetrics(mono_font(SIZE_BODY))
        value_font = mono_font(SIZE_NUMERIC)
        value_metrics = QFontMetrics(value_font)
        label_column = max((label_metrics.horizontalAdvance(name) for name, _ in rows), default=0)
        budget = char_width(_POPOVER_MAX_VALUE_CHARS, value_font)
        self.setMaximumWidth(budget + label_column + _POPOVER_CHROME_PX)
        self._grid.set_rows(
            [
                (name, value_metrics.elidedText(value, Qt.TextElideMode.ElideRight, budget))
                for name, value in rows
            ]
        )
        self.adjustSize()

    def show_below(self, anchor: QWidget) -> None:
        """Pop the frame up under *anchor*, their left edges aligned."""
        self.move(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        self.show()


__all__ = ["InfoPopover"]
