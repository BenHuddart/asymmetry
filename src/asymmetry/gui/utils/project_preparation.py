"""Cancellable project preparation, before any active session is replaced."""

from PySide6.QtCore import QEventLoop, Qt, Slot
from PySide6.QtWidgets import QProgressDialog

from asymmetry.core.project.preparation import load_prepared_project
from asymmetry.gui.tasks import check_task_cancelled


def _read_project(worker, path):
    try:
        return load_prepared_project(path, checkpoint=lambda: check_task_cancelled(worker))
    except (ValueError, OSError) as exc:
        return exc


class ProjectPreparation(QProgressDialog):
    """One modal request; cancellation discards any late result."""

    def __init__(self, parent):
        super().__init__("Checking project…", "Cancel", 0, 0, parent)
        self.setWindowTitle("Opening Project")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumDuration(400)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self._loop = QEventLoop(self)
        self._result = None
        self._worker = None

    def run(self, tasks, path=None, *, operation=None):
        self._worker = tasks.start(
            lambda worker: _read_project(worker, path) if operation is None else operation(worker),
            on_finished=self._finished,
            on_error=self._error,
            on_cancelled=self._loop.quit,
            on_progress=self._progress,
        )
        self.canceled.connect(self.cancel_preparation)
        self.setValue(0)
        self._loop.exec()
        result = self._result
        self._result = None
        cancelled = self.wasCanceled()
        self.close()
        if cancelled:
            return None
        if isinstance(result, (ValueError, OSError)):
            raise result
        return result

    @Slot()
    def cancel_preparation(self):
        # The worker reads detached input; it can finish after the modal request
        # ends without publishing anything into this or a subsequent session.
        self._worker.cancel()
        self.cancel()
        self._loop.quit()

    @Slot(object)
    def _finished(self, result):
        if self._loop.isRunning():
            self._result = result
            self._loop.quit()

    @Slot(str)
    def _error(self, message):
        if self._loop.isRunning():
            self._result = ValueError(message)
            self._loop.quit()

    @Slot(int, int, str)
    def _progress(self, done, total, label):
        if self._loop.isRunning():
            self.setRange(0, total)
            self.setLabelText(label)
            self.setValue(done)
