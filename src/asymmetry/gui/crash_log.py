"""Persist desktop diagnostics and report uncaught Python errors on the GUI thread."""

from __future__ import annotations

import faulthandler
import logging
import platform
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from PySide6 import __version__ as pyside_version
from PySide6.QtCore import (
    QMessageLogContext,
    QObject,
    Qt,
    QtMsgType,
    QUrl,
    Signal,
    Slot,
    qFormatLogMessage,
    qInstallMessageHandler,
    qVersion,
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox

from asymmetry import __version__
from asymmetry.gui.utils import app_data

session_log = logging.getLogger(__name__)
session_log.propagate = False

_QT_LEVELS = {
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _log_directory() -> Path:
    return app_data.app_data_dir() / "logs"


def open_log_folder() -> None:
    """Open the desktop diagnostics folder in the system file manager."""
    directory = _log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))


class CrashLog(QObject):
    """Own the process-wide diagnostic hooks and native log for the GUI session."""

    _unexpected_error = Signal(str)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._notifications: set[QMessageBox] = set()
        self._unexpected_error.connect(self._notify, Qt.ConnectionType.QueuedConnection)

        directory = _log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        self._path = directory / "asymmetry.log"
        handler = RotatingFileHandler(
            self._path, maxBytes=2 * 1024**2, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        session_log.addHandler(handler)
        session_log.setLevel(logging.INFO)
        session_log.info(
            "Session started | Asymmetry %s | Python %s | Qt %s | PySide6 %s | %s",
            __version__,
            platform.python_version(),
            qVersion(),
            pyside_version,
            platform.platform(),
        )

        # Faulthandler writes to a file descriptor that must stay open and never rotate.
        self._native_stream = (directory / "native-crash.log").open("a", encoding="utf-8")
        self._native_stream.write(f"\n{datetime.now().astimezone().isoformat()} Session started\n")
        self._native_stream.flush()
        faulthandler.enable(file=self._native_stream, all_threads=True)

        self._previous_exception_hook = sys.excepthook
        self._previous_thread_hook = threading.excepthook
        sys.excepthook = self._handle_exception
        threading.excepthook = self._handle_thread_exception
        self._previous_qt_handler = qInstallMessageHandler(self._handle_qt_message)

    def _report_exception(
        self,
        exc_type: type[BaseException],
        error: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        session_log.error("Uncaught Python exception", exc_info=(exc_type, error, traceback))
        self._unexpected_error.emit(f"{exc_type.__name__}: {error}")

    def _handle_exception(
        self,
        exc_type: type[BaseException],
        error: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        self._report_exception(exc_type, error, traceback)
        self._previous_exception_hook(exc_type, error, traceback)

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        self._report_exception(args.exc_type, args.exc_value, args.exc_traceback)
        self._previous_thread_hook(args)

    def _handle_qt_message(
        self, kind: QtMsgType, context: QMessageLogContext, message: str
    ) -> None:
        level = _QT_LEVELS.get(kind)
        if level is not None:
            session_log.log(level, "Qt [%s] %s", context.category, message)
        if self._previous_qt_handler is not None:
            self._previous_qt_handler(kind, context, message)
        elif sys.stderr is not None:
            print(qFormatLogMessage(kind, context, message), file=sys.stderr)

    @Slot(str)
    def _notify(self, error: str) -> None:
        parent = QApplication.activeWindow()
        if isinstance(parent, QMessageBox):
            parent = parent.parentWidget()
        box = QMessageBox(parent)
        box.setWindowTitle("Unexpected Error")
        box.setIcon(QMessageBox.Icon.Critical)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(
            f"{error}\n\nThe traceback is in the application log.\n{self._path}\n\n"
            "Use Help → Open Log Folder to find the log."
        )
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._notifications.add(box)
        box.destroyed.connect(lambda: self._notifications.discard(box))
        box.show()
