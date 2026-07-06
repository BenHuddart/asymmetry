"""The single titled-section primitive for BENCH control panels.

``PanelSection`` is the one way a panel introduces a titled block of controls.
It replaces the two older idioms it is designed to fully cover:

- ``styles/widgets.make_section`` — a flat header over an always-expanded body
  (``PanelSection`` in its default *static* mode), and
- the former ``CollapsibleSection`` — a disclosure header that hides its body
  (``PanelSection`` in *collapsible* mode; that module has been removed).

Beyond that union it adds the pieces the panel refresh needs everywhere: a
one-line muted *hint* under the header, an optional right-aligned *title suffix*
chip for collapsed-section summaries ("3 exclusions"), and optional QSettings
persistence of the expanded/collapsed state.

The header always uses :func:`~asymmetry.gui.styles.widgets.make_section_header`
so its font, ``objectName``, colour and uppercasing match the flat sections
already in the app. In collapsible mode a disclosure arrow is prepended and the
*whole header row* is clickable.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.typography import footer_font
from asymmetry.gui.styles.widgets import make_section_header


class _ClickableRow(QWidget):
    """A header row that emits :attr:`clicked` on a left mouse press."""

    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PanelSection(QWidget):
    """A titled section of panel controls — static or collapsible.

    Parameters
    ----------
    title:
        Section title. Rendered through :func:`make_section_header` (uppercased,
        muted, BENCH header font).
    collapsible:
        When ``True`` the header shows a disclosure arrow, the whole header row
        is clickable, and the body starts collapsed unless ``expanded=True``.
        When ``False`` (default) the section is static: always expanded, no
        disclosure affordance.
    expanded:
        Initial expanded state for a collapsible section. Ignored for a static
        section (always expanded). Overridden by a persisted value when
        ``settings_key`` is set and present.
    hint:
        Optional one-line muted description shown under the header. Hidden when
        ``None``.
    settings_key:
        Optional ``QSettings`` key under which the expanded/collapsed state is
        persisted (collapsible sections only). A persisted value overrides
        *expanded*.
    settings:
        Optional ``QSettings`` instance for persistence — defaults to a plain
        ``QSettings()`` using the app-configured org/app scope. Injectable so
        tests can isolate a scratch scope (mirrors ``gui/fit_settings.py``).
    parent:
        Optional parent widget.
    """

    #: Emitted with the new expanded state whenever the section toggles.
    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        *,
        collapsible: bool = False,
        expanded: bool = False,
        hint: str | None = None,
        settings_key: str | None = None,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = str(title)
        self._collapsible = bool(collapsible)
        self._settings_key = settings_key
        self._settings = settings if settings is not None else QSettings()

        # A static section is always expanded; a collapsible one honours the
        # persisted value first, then the constructor default.
        if not self._collapsible:
            start_expanded = True
        elif self._settings_key is not None:
            start_expanded = self._settings.value(self._settings_key, bool(expanded), type=bool)
        else:
            start_expanded = bool(expanded)
        self._expanded = bool(start_expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Header row ──────────────────────────────────────────────────────
        self._header_row = _ClickableRow(self)
        header_layout = QHBoxLayout(self._header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self._arrow: QLabel | None = None
        if self._collapsible:
            self._arrow = QLabel(self._header_row)
            self._arrow.setFont(footer_font())
            self._arrow.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
            header_layout.addWidget(self._arrow)
            self._header_row.clicked.connect(self._on_header_clicked)
            self._header_row.setCursor(Qt.CursorShape.PointingHandCursor)

        self._header_label = make_section_header(self._title)
        header_layout.addWidget(self._header_label)
        header_layout.addStretch(1)

        self._suffix_label = QLabel("", self._header_row)
        self._suffix_label.setFont(footer_font())
        self._suffix_label.setTextFormat(Qt.TextFormat.RichText)
        self._suffix_label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._suffix_label.hide()
        header_layout.addWidget(self._suffix_label)

        outer.addWidget(self._header_row)

        # ── Optional hint ───────────────────────────────────────────────────
        self._hint_label = QLabel("", self)
        self._hint_label.setFont(footer_font())
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._hint_label.hide()
        outer.addWidget(self._hint_label)
        if hint is not None:
            self.set_hint(hint)

        # ── Body (matches make_section's margins/spacing) ───────────────────
        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(4)
        outer.addWidget(self._body)

        self._sync_expanded_visuals()

    # ── Body access ─────────────────────────────────────────────────────────

    @property
    def body_layout(self) -> QVBoxLayout:
        """The section's content layout (a ``QVBoxLayout``)."""
        return self._body_layout

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802 — Qt-style name
        """Add *widget* to the section body."""
        self._body_layout.addWidget(widget)

    def addLayout(self, layout) -> None:  # noqa: N802 — Qt-style name
        """Add a sub-*layout* to the section body."""
        self._body_layout.addLayout(layout)

    def contentLayout(self) -> QVBoxLayout:  # noqa: N802 — CollapsibleSection parity
        """Return the body layout (``CollapsibleSection`` API parity)."""
        return self._body_layout

    # ── Header extras ─────────────────────────────────────────────────────────

    def set_hint(self, text: str | None) -> None:
        """Set (or clear, with ``None``) the muted one-line hint under the header."""
        if text:
            self._hint_label.setText(str(text))
            self._hint_label.show()
        else:
            self._hint_label.clear()
            self._hint_label.hide()

    def set_title_suffix(self, html: str | None) -> None:
        """Set a small right-aligned rich-text suffix (chip/count) in the header.

        Pass ``None`` or an empty string to hide it.
        """
        if html:
            self._suffix_label.setText(str(html))
            self._suffix_label.show()
        else:
            self._suffix_label.clear()
            self._suffix_label.hide()

    def title(self) -> str:
        """Return the section title (``CollapsibleSection`` API parity)."""
        return self._title

    # ── Expand / collapse ─────────────────────────────────────────────────────

    def isExpanded(self) -> bool:  # noqa: N802 — CollapsibleSection parity
        """Return whether the body is currently expanded."""
        return self._expanded

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802 — CollapsibleSection parity
        """Expand or collapse the section (no-op for a static section).

        Persists the new state when ``settings_key`` was given, and emits
        :attr:`toggled` only when the state actually changes.
        """
        expanded = bool(expanded)
        if not self._collapsible:
            return
        changed = expanded != self._expanded
        self._expanded = expanded
        self._sync_expanded_visuals()
        if self._settings_key is not None:
            self._settings.setValue(self._settings_key, expanded)
        if changed:
            self.toggled.emit(expanded)

    def _on_header_clicked(self) -> None:
        self.setExpanded(not self._expanded)

    def _sync_expanded_visuals(self) -> None:
        self._body.setVisible(self._expanded)
        if self._arrow is not None:
            # ▼ when open, ▶ when closed — muted disclosure triangles.
            self._arrow.setText("▾" if self._expanded else "▸")
