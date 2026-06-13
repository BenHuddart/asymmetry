"""A translucent "busy" overlay that covers one widget while work is in flight.

Long analysis recomputed on project open (FFT spectra, cross-group fit curves)
runs off the GUI thread, so the affected panel must show that its content is
provisional rather than render a half-populated view as if it were final. This
overlay mounts as a child of the panel it covers, paints a translucent scrim
plus an indeterminate progress bar and a message, and tracks the panel's
geometry so it stays covering on resize.

It owns no threads and does no work — call :meth:`show_message` from a task's
launch site and :meth:`hide` from its completion/error handler. ``hide`` is
idempotent so the "always clears" contract holds for success, empty and error
paths alike.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from asymmetry.gui.styles import tokens


class LoadingOverlay(QWidget):
    """Translucent busy overlay parented to (and covering) ``target``."""

    def __init__(self, target: QWidget) -> None:
        super().__init__(target)
        self._target = target
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Semi-transparent scrim so the (provisional) panel content stays dimly
        # visible underneath — it reads as "loading", not "blank".
        self.setStyleSheet("LoadingOverlay { background-color: rgba(250, 250, 249, 200); }")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        self._message = QLabel("", self)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setStyleSheet(
            f"color: {tokens.TEXT}; background: transparent; font-size: 13px;"
        )
        layout.addWidget(self._message)

        # Indeterminate (busy) bar — range(0, 0) is Qt's marquee mode, so we
        # need no timer, no image asset and no extra dependency.
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(180)
        self._bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {tokens.BORDER}; border-radius: 3px;"
            f" background: {tokens.SURFACE}; height: 6px; }}"
            f"QProgressBar::chunk {{ background-color: {tokens.ACCENT}; border-radius: 2px; }}"
        )
        layout.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignCenter)

        # Follow the target's geometry: resizes and moves both reach us through
        # the event filter so the scrim never lags behind the panel.
        target.installEventFilter(self)
        self._sync_geometry()
        self.hide()

    def show_message(self, text: str) -> None:
        """Show the overlay over the target with *text* as its caption."""
        self._message.setText(str(text))
        self._sync_geometry()
        self.show()
        self.raise_()

    def hide(self) -> None:
        """Hide the overlay. Idempotent — safe to call when already hidden."""
        super().hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if obj is self._target and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
        ):
            self._sync_geometry()
        return super().eventFilter(obj, event)

    def _sync_geometry(self) -> None:
        self.setGeometry(self._target.rect())
