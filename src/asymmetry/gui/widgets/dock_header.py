"""BENCH dock title bar: uppercase title left, mono meta right, float/close.

Installed via ``QDockWidget.setTitleBarWidget`` so it *replaces* Qt's native
title bar — the design-handoff header strip and the Qt title must never both
render (the "LOG twice" failure mode). The dock's ``windowTitle`` stays set
for tab labels, View-menu entries, and floating-window OS chrome; it just is
not painted while this widget occupies the title-bar slot.

Replacing the title bar also removes Qt's float/close buttons, so this widget
provides its own. Close goes through ``dock.close()`` — a real Close event —
so the main window's per-representation closed-tab memory keeps working.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QPushButton, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.typography import footer_font, header_font

# Font-size is intentionally omitted: the glyph buttons inherit the (UI-scaled)
# footer font set in __init__, so they track the UI-scale setting rather than
# pinning a fixed 11px.
_BUTTON_QSS = (
    "QPushButton { border: none; background: transparent; padding: 0 4px;"
    f" color: {tokens.TEXT_MUTED}; }}"
    f"QPushButton:hover {{ background-color: {tokens.SURFACE_HI};"
    " border-radius: 3px; }"
)


class DockHeader(QWidget):
    """Custom dock title bar matching the BENCH header-strip grammar."""

    def __init__(
        self,
        title: str,
        dock: QDockWidget,
        *,
        closable: bool = True,
        floatable: bool = True,
        title_when_floating_only: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Build the header.

        ``title_when_floating_only`` is for tabified deck docks: while docked,
        the tab bar already names the pane, so painting the title again right
        under it reads as a duplicate ("Fit" twice). The label is hidden while
        docked and shown — tracking ``windowTitle`` — once the dock floats and
        the tab bar is no longer there to identify it.
        """
        super().__init__(parent)
        self._dock = dock
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"DockHeader {{ background-color: {tokens.SURFACE_ALT};"
            f" border-bottom: 1px solid {tokens.BORDER}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 4, 3)
        layout.setSpacing(4)

        self._title_label = QLabel(title or dock.windowTitle().upper())
        self._title_label.setFont(header_font())
        self._title_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; background: transparent;")
        layout.addWidget(self._title_label)
        layout.addStretch()

        if title_when_floating_only:
            self._title_label.setVisible(dock.isFloating())
            dock.topLevelChanged.connect(self._title_label.setVisible)
            # Deck dock titles change with context (Fit → ALC scan, Spectrum →
            # Fourier/MaxEnt); mirror them so the floating header stays honest.
            dock.windowTitleChanged.connect(
                lambda text: self._title_label.setText(str(text).upper())
            )

        self._meta_label = QLabel("")
        self._meta_label.setFont(footer_font())
        self._meta_label.setStyleSheet(f"color: {tokens.TEXT_MUTED}; background: transparent;")
        layout.addWidget(self._meta_label)

        if floatable:
            float_btn = QPushButton("↗")
            float_btn.setToolTip("Float / dock this panel")
            float_btn.setFont(footer_font())
            float_btn.setStyleSheet(_BUTTON_QSS)
            float_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            float_btn.clicked.connect(self._toggle_floating)
            layout.addWidget(float_btn)
        if closable:
            close_btn = QPushButton("✕")
            close_btn.setToolTip("Close this panel (reopen from the View menu)")
            close_btn.setFont(footer_font())
            close_btn.setStyleSheet(_BUTTON_QSS)
            close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            close_btn.clicked.connect(self._dock.close)
            layout.addWidget(close_btn)

    def set_meta(self, text: str) -> None:
        """Update the right-aligned mono annotation (counts, selection, …)."""
        self._meta_label.setText(str(text))

    def _toggle_floating(self) -> None:
        self._dock.setFloating(not self._dock.isFloating())
