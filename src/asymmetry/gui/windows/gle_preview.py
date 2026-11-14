"""Read-only, vector-backed GLE export preview."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QEvent, QIODevice, Qt
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout

from asymmetry.gui.styles.metrics import dialog_width, row_height
from asymmetry.gui.tasks import TaskRunner

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GleExportResult:
    """Completed worker output; the PDF is independent of its disk lifetime."""

    files: tuple[Path, ...]
    pdf: bytes | None
    summary: str
    generation_seconds: float
    compilation_seconds: float


class GlePlotPreview(QDialog):
    """Own a PDF snapshot independently of subsequent exports to its path."""

    def __init__(self, parent, *, destination: Path | None = None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("GLE Plot Preview")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(dialog_width(118), row_height() * 24)
        self.destination = destination
        self.completed_at: float | None = None
        self.first_paint_seconds: float | None = None
        self.tasks = TaskRunner(self)
        self.status = QLabel("Generating plot…")
        self.status.setWordWrap(True)
        self.view = QPdfView(self)
        self.document = QPdfDocument(self)
        self.document.statusChanged.connect(self.document_status_changed)
        self.buffer = QBuffer(self)
        self._pan_position = None
        self.view.viewport().installEventFilter(self)
        self.view.setDocument(self.document)
        self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
        self.details = QTextEdit(self)
        self.details.setReadOnly(True)
        self.details.hide()
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.details)
        buttons = QHBoxLayout()
        for label, callback in (
            ("Fit to window", self.fit),
            ("Zoom in", self.zoom_in),
            ("Zoom out", self.zoom_out),
            ("100%", self.actual_size),
            ("Details", self.toggle_details),
            ("Close", self.close),
        ):
            button = QPushButton(label, self)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.view.viewport():
            if (
                event.type() == QEvent.Type.Paint
                and self.completed_at is not None
                and self.first_paint_seconds is None
                and self.document.status() == QPdfDocument.Status.Ready
            ):
                self.first_paint_seconds = time.perf_counter() - self.completed_at
                _log.debug(
                    "GLE preview first paint %.3fs after completion", self.first_paint_seconds
                )
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.MiddleButton
            ):
                self._pan_position = event.position()
                return True
            if event.type() == QEvent.Type.MouseMove and self._pan_position is not None:
                delta = event.position() - self._pan_position
                self._pan_position = event.position()
                horizontal = self.view.horizontalScrollBar()
                vertical = self.view.verticalScrollBar()
                horizontal.setValue(horizontal.value() - round(delta.x()))
                vertical.setValue(vertical.value() - round(delta.y()))
                return True
            if (
                event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.MiddleButton
            ):
                self._pan_position = None
                return True
        return super().eventFilter(watched, event)

    def export_finished(self, result: GleExportResult | None) -> None:
        if result is None:
            self.close()
            return
        self.status.setText(result.summary)
        self.details.setPlainText("\n".join(str(path) for path in result.files))
        if result.pdf is not None:
            self.completed_at = time.perf_counter()
            self.set_pdf(result.pdf)

    def export_failed(self, message: str) -> None:
        self.status.setText(f"Export failed: {message}")

    def export_progress(self, current: int, total: int, message: str) -> None:
        self.status.setText(message)

    def document_status_changed(self, status) -> None:
        if status == QPdfDocument.Status.Error:
            self.status.setText("Preview unavailable: the generated PDF could not be loaded.")

    def set_pdf(self, data: bytes) -> None:
        self.document.close()
        self.buffer.close()
        self.buffer.setData(QByteArray(data))
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.document.load(self.buffer)
        if not data:
            self.status.setText("Preview unavailable: the generated PDF is empty.")

    def fit(self):
        self.view.setZoomMode(QPdfView.ZoomMode.FitInView)

    def zoom_in(self):
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(min(10.0, self.view.zoomFactor() * 1.25))

    def zoom_out(self):
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(max(0.1, self.view.zoomFactor() / 1.25))

    def actual_size(self):
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(1.0)

    def toggle_details(self):
        self.details.setVisible(not self.details.isVisible())

    def closeEvent(self, event):  # noqa: N802
        self.tasks.shutdown()
        super().closeEvent(event)

    def reject(self):
        self.tasks.shutdown()
        super().reject()


def close_gle_previews(parent):
    """Stop exports and close previews belonging to this project or shell."""
    for preview in parent.findChildren(GlePlotPreview):
        preview.close()
