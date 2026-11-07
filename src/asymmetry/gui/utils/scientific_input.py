"""Cancellable worker preparation for a synchronous scientific input boundary."""

from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import QProgressDialog

from asymmetry.gui.tasks import TaskRunner


class ScientificInputDialog(QProgressDialog):
    """Keep the interface responsive while a caller waits for prepared input."""

    def __init__(self, parent, label, error_context):
        super().__init__(label, "Cancel", 0, 0, parent)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumDuration(0)
        self._error_context = error_context
        self._loop = QEventLoop(self)
        self._tasks = TaskRunner(self)
        self._outcome = None

    def run(self, task):
        """Return the worker result, or None after cooperative cancellation."""
        self.show()
        self._worker = self._tasks.start(
            task,
            on_finished=self._on_finished,
            on_error=self._on_error,
            on_cancelled=self._loop.quit,
        )
        self.canceled.connect(self._on_cancel)
        self.rejected.connect(self._on_cancel)
        try:
            self._loop.exec()
        finally:
            self.canceled.disconnect(self._on_cancel)
            self.rejected.disconnect(self._on_cancel)
            self._tasks.shutdown()
            self.close()
        if self._outcome is None:
            return None
        success, result = self._outcome
        if not success:
            raise ValueError(f"{self._error_context} failed: {result}")
        return result

    def _on_cancel(self):
        self._worker.cancel()
        self.show()

    def _on_finished(self, result):
        if not self._worker.is_cancelled():
            self._outcome = (True, result)
        self._loop.quit()

    def _on_error(self, message):
        self._outcome = (False, message)
        self._loop.quit()
