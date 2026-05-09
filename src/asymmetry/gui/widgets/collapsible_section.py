"""Reusable collapsible section widget for dense control panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Simple disclosure-style section with a toggle header and content area."""

    def __init__(self, title: str, *, expanded: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = str(title)

        self._toggle = QToolButton(self)
        self._toggle.setText(self._title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(bool(expanded))
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.toggled.connect(self.setExpanded)

        self._content = QWidget(self)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 2, 0, 0)
        self._content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

        self.setExpanded(bool(expanded))

    def addWidget(self, widget: QWidget) -> None:  # noqa: N802
        self._content_layout.addWidget(widget)

    def addLayout(self, layout) -> None:  # noqa: N802
        self._content_layout.addLayout(layout)

    def contentLayout(self) -> QVBoxLayout:  # noqa: N802
        return self._content_layout

    def isExpanded(self) -> bool:  # noqa: N802
        return self._toggle.isChecked()

    def setExpanded(self, expanded: bool) -> None:  # noqa: N802
        expanded = bool(expanded)
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)

    def title(self) -> str:
        return self._title
