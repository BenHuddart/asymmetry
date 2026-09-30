"""The progress block a wizard step shows while its analysis runs.

A bold header, the streaming :class:`DecisionTrail`, a collapsed "Live log" and
a Cancel button, packaged so the step that is running can host it (plan D2).
The host drives the trail directly through :attr:`RunProgress.trail`.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from asymmetry.core.fitting.wizard_narrative import TrailStep
from asymmetry.gui.panels.log_panel import LogPanel
from asymmetry.gui.styles.metrics import row_height
from asymmetry.gui.widgets.decision_trail import DecisionTrail
from asymmetry.gui.widgets.panel_section import PanelSection

#: The open log shows about this many lines before it scrolls.
_LOG_VISIBLE_LINES = 10


class RunProgress(QWidget):
    """Header, streaming trail, live log and Cancel for one running analysis."""

    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._messages: list[str] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header_row = QHBoxLayout()
        self._header = QLabel("", self)
        header_font = self._header.font()
        header_font.setBold(True)
        self._header.setFont(header_font)
        header_row.addWidget(self._header, 1)
        self._cancel = QPushButton("Cancel", self)
        self._cancel.clicked.connect(self.cancel_requested)
        header_row.addWidget(self._cancel)
        layout.addLayout(header_row)

        self._trail = DecisionTrail(self)
        layout.addWidget(self._trail)

        self._log_section = PanelSection("Live log", collapsible=True, expanded=False, parent=self)
        self._log = LogPanel(self._log_section)
        self._log.setMinimumHeight(row_height() * _LOG_VISIBLE_LINES)
        self._log_section.addWidget(self._log)
        layout.addWidget(self._log_section)
        layout.addStretch(1)

    @property
    def trail(self) -> DecisionTrail:
        return self._trail

    def start(self, header: str, placeholder_steps: tuple[TrailStep, ...]) -> None:
        """Begin a run: set the header, stream the pending steps and clear the log."""
        self._header.setText(header)
        self._trail.stream_placeholders(placeholder_steps)
        self._trail.set_status("")
        self._log.clear()
        self._messages.clear()

    def append_log(self, text: str) -> None:
        self._log.log(text)
        self._messages.append(text)

    def log_text(self) -> str:
        """The run's log messages, one per line, without the panel's timestamps."""
        return "\n".join(self._messages)
