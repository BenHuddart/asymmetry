"""Collapsible correction card for the grouping dialog's Corrections column.

Grouping-specific — deliberately richer than the shared
:class:`~asymmetry.gui.widgets.panel_section.PanelSection`: the header row
carries a *live status summary* (the same one-line text as the pipeline chip,
minus the redundant title prefix) and a *compare indicator* that lights the
card up while its stage's before/after ghost is focused in the shared preview.
Expansion state is plain widget state for the dialog's lifetime — no QSettings
persistence — because the dialog derives the default (expanded iff the stage is
active) from the draft on every open.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.elided_label import ElidedLabel

#: objectName on the outer card frame so its border rule never cascades onto
#: child widgets (the ID selector matches this frame only).
_CARD_OBJECT_NAME = "correctionCard"

#: objectName on the clickable header frame, for the same non-cascading reason.
_HEADER_OBJECT_NAME = "correctionCardHeader"


class _ClickableHeader(QFrame):
    """A header frame that emits :attr:`clicked` on a left mouse press."""

    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class CorrectionCard(QFrame):
    """A rounded-border card wrapping one correction's controls.

    The header row — disclosure arrow, title, live status summary, and a
    right-aligned compare indicator — is clickable anywhere to toggle the body.
    :meth:`set_comparing` accent-tints the header while the stage's compare is
    focused; :meth:`set_stale` warn-tints the title/status (the α card, when a
    calibrated α no longer matches the corrections it was measured under).
    """

    #: Emitted with the new expanded state whenever the card toggles.
    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        *,
        color: str = tokens.ACCENT,
        soft: str = tokens.ACCENT_SOFT,
        parent: QWidget | None = None,
    ) -> None:
        """*color*/*soft* are the stage's identity pair (see ``tokens.STAGE_*``).

        The header wears a *color* stripe at all times — the same colour the
        stage's pipeline chip wears as its outline — and fills with *soft*
        while the stage's compare is focused.
        """
        super().__init__(parent)
        self._title = str(title)
        self._color = str(color)
        self._soft = str(soft)
        self._expanded = True
        self._comparing_text: str | None = None
        self._stale = False

        self.setObjectName(_CARD_OBJECT_NAME)
        self.setStyleSheet(
            f"QFrame#{_CARD_OBJECT_NAME} {{"
            f" border: 1px solid {tokens.BORDER};"
            f" border-radius: 4px;"
            f" }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        # ── Header row (clickable anywhere) ────────────────────────────────
        self._header = _ClickableHeader(self)
        self._header.setObjectName(_HEADER_OBJECT_NAME)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.clicked.connect(self._on_header_clicked)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)

        self._arrow = QLabel(self._header)
        header_layout.addWidget(self._arrow)

        self._title_label = QLabel(self._title, self._header)
        title_font = self._title_label.font()
        title_font.setWeight(title_font.Weight.DemiBold)
        self._title_label.setFont(title_font)
        header_layout.addWidget(self._title_label)

        # Status stretches to fill the middle and elides when tight, so a long
        # summary (or the compare indicator) never widens the corrections column
        # into a horizontal scrollbar.
        self._status_label = ElidedLabel(parent=self._header)
        header_layout.addWidget(self._status_label, 1)

        self._comparing_label = ElidedLabel(parent=self._header)
        self._comparing_label.hide()
        header_layout.addWidget(self._comparing_label)

        outer.addWidget(self._header)

        # ── Body ───────────────────────────────────────────────────────────
        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(6, 4, 6, 6)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)

        self._sync_visuals()

    # ── Body access ─────────────────────────────────────────────────────────

    @property
    def body_layout(self) -> QVBoxLayout:
        """The card's content layout (a ``QVBoxLayout``)."""
        return self._body_layout

    def add_body_widget(self, widget: QWidget) -> None:
        """Add *widget* to the card body (below any previously added)."""
        self._body_layout.addWidget(widget)

    def set_body(self, widget: QWidget) -> None:
        """Add *widget* as the card body (alias for :meth:`add_body_widget`)."""
        self.add_body_widget(widget)

    # ── Header state ────────────────────────────────────────────────────────

    def title(self) -> str:
        """Return the card title."""
        return self._title

    def set_status(self, text: str) -> None:
        """Set the muted live status summary shown beside the title."""
        self._status_label.setText(str(text))

    def status_text(self) -> str:
        """Return the current status summary (test seam)."""
        return self._status_label.text()

    def set_stale(self, stale: bool) -> None:
        """Warn-tint the title/status when *stale* (the α card's staleness cue)."""
        stale = bool(stale)
        if stale == self._stale:
            return
        self._stale = stale
        self._sync_visuals()

    def set_comparing(self, text: str | None) -> None:
        """Show (or clear, with ``None``) the compare state on the header.

        While set, the header is accent-tinted (soft background + accent left
        border) and the right-aligned indicator shows *text* (e.g.
        ``"comparing: without deadtime"``).
        """
        text = str(text) if text else None
        if text == self._comparing_text:
            return
        self._comparing_text = text
        self._sync_visuals()

    def comparing_text(self) -> str | None:
        """Return the active compare indicator text, or ``None`` (test seam)."""
        return self._comparing_text

    # ── Expand / collapse ───────────────────────────────────────────────────

    def is_expanded(self) -> bool:
        """Return whether the body is currently shown."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the body; emits :attr:`toggled` on a change."""
        expanded = bool(expanded)
        changed = expanded != self._expanded
        self._expanded = expanded
        self._sync_visuals()
        if changed:
            self.toggled.emit(expanded)

    def _on_header_clicked(self) -> None:
        self.set_expanded(not self._expanded)

    # ── Visual sync ─────────────────────────────────────────────────────────

    def _sync_visuals(self) -> None:
        self._body.setVisible(self._expanded)
        self._arrow.setText("▾" if self._expanded else "▸")

        comparing = self._comparing_text is not None
        # The stage-colour stripe is the card's identity and stays on in every
        # state (matching its pipeline chip's outline); comparing only deepens
        # the header fill to the stage's soft tint.
        header_bg = self._soft if comparing else tokens.SURFACE_ALT
        self._header.setStyleSheet(
            f"QFrame#{_HEADER_OBJECT_NAME} {{"
            f" background-color: {header_bg};"
            f" border: none;"
            f" border-left: 3px solid {self._color};"
            f" border-radius: 3px;"
            f" }}"
        )

        title_color = tokens.WARN if self._stale else tokens.TEXT
        status_color = tokens.WARN if self._stale else tokens.TEXT_MUTED
        self._arrow.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._title_label.setStyleSheet(f"QLabel {{ color: {title_color}; }}")
        self._status_label.set_pen_color(status_color)
        self._comparing_label.set_pen_color(tokens.TEXT_MUTED)
        self._comparing_label.setText(self._comparing_text or "")
        self._comparing_label.setVisible(comparing)


__all__ = ["CorrectionCard"]
