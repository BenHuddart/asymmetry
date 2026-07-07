"""Log / message panel — displays status messages with timestamp and category tags."""

from __future__ import annotations

import html as _html
from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font

# Tag → text colour mapping (from BENCH tokens)
_TAG_COLOURS: dict[str, str] = {
    "load": tokens.ACCENT,
    "group": tokens.ACCENT,
    "fit": tokens.OK,
    "mgfit": tokens.OK,
    "peak": tokens.OK,
    "trend": tokens.WARN,
    "warn": tokens.WARN,
}


#: Ring-buffer cap on retained log lines (``QTextDocument`` blocks). Once the
#: document holds this many blocks, appending further entries drops the
#: oldest ones automatically. ``to_plain_text()`` (and any "save log" /
#: copy-from-panel feature) only ever sees the most recent
#: ``_MAX_LOG_BLOCKS`` entries — earlier history is gone, not just hidden.
_MAX_LOG_BLOCKS = 5000


class LogPanel(QWidget):
    """Scrollable log with per-category tag colouring.

    Capped to the most recent :data:`_MAX_LOG_BLOCKS` entries via
    ``QTextDocument.setMaximumBlockCount`` so a long-running session's log
    cannot grow the widget (and the app's memory footprint) without bound.
    """

    #: Emitted with the running entry count (feeds the dock header's badge).
    entry_count_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entry_count = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(mono_font(10.5))
        self._text.document().setMaximumBlockCount(_MAX_LOG_BLOCKS)
        layout.addWidget(self._text)

    def log(self, message: str, *, tag: str = "") -> None:
        """Append a timestamped entry, optionally categorised by *tag*."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        ts_html = f'<span style="color:{tokens.TEXT_MUTED};">{timestamp}</span>'
        tag_html = self._tag_span(tag)
        msg_html = _html.escape(str(message))
        self._text.append(f"{ts_html}&nbsp;&nbsp;{tag_html}{msg_html}")
        self._entry_count += 1
        self.entry_count_changed.emit(self._entry_count)

    def entry_count(self) -> int:
        """Return the number of entries logged since the last clear."""
        return self._entry_count

    def clear(self) -> None:
        """Clear all log entries."""
        self._text.clear()
        self._entry_count = 0
        self.entry_count_changed.emit(0)

    def to_plain_text(self) -> str:
        return self._text.toPlainText()

    @staticmethod
    def _tag_span(tag: str) -> str:
        """Return a coloured HTML span for *tag*, or empty string if blank."""
        if not tag:
            return ""
        colour = _TAG_COLOURS.get(tag.lower(), tokens.TEXT_MUTED)
        label = _html.escape(f"[{tag.upper()}]")
        return f'<span style="color:{colour};font-weight:600;">{label}</span>&nbsp;&nbsp;'
