"""Log / message panel — displays status messages and command history."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from asymmetry.gui.styles.fonts import mono_font


class LogPanel(QWidget):
    """Scrollable text log for status messages."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(5000)
        self._text.setFont(mono_font(10.5))
        layout.addWidget(self._text)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._text.appendPlainText(f"[{timestamp}] {message}")

    def clear(self) -> None:
        self._text.clear()

    def to_plain_text(self) -> str:
        return self._text.toPlainText()
