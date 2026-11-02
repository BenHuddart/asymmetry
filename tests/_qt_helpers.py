from __future__ import annotations

import time

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication


def wait_for(predicate: object, qapp: QApplication, timeout_s: float = 5.0) -> None:
    """Poll *predicate* until it returns truthy or *timeout_s* elapses.

    Fits in tests are mocked, so the predicate normally flips within
    milliseconds and the deadline is never approached — it only bounds how
    long a genuinely broken test takes to fail. It must still be generous:
    under a fully loaded machine (full-tier xdist plus the wizard's process
    pools) QThread scheduling and queued-signal delivery alone can take
    longer than the 0.5 s this helper originally allowed, which made GUI
    window tests flake nondeterministically. Pass a larger timeout_s only
    when the test exercises genuinely slow real work.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out after {timeout_s}s waiting for UI state")


def flush_deferred_deletes(qapp: QApplication) -> None:
    """Dispatch queued deletion outside a running Qt event loop."""
    deferred_delete = QEvent.Type.DeferredDelete.value
    qapp.sendPostedEvents(None, deferred_delete)
    qapp.processEvents()
    qapp.sendPostedEvents(None, deferred_delete)


def dispose_qt_widgets(qapp: QApplication) -> None:
    """Close all test-owned widgets and enforce a clean Qt lifetime boundary."""
    import matplotlib.pyplot as plt
    from PySide6.QtCore import QThread

    from asymmetry.gui.mainwindow import MainWindow
    from asymmetry.gui.tasks import drain_orphan_threads
    from asymmetry.gui.windows.grouping_dialog import GroupingDialog

    for widget in list(qapp.topLevelWidgets()):
        if isinstance(widget, (MainWindow, GroupingDialog)):
            widget._clear_dirty()
        widget.close()
    for thread in qapp.findChildren(QThread):
        if thread.isRunning():
            thread.quit()
            assert thread.wait(5000), "Qt worker survived test teardown"
    for widget in list(qapp.topLevelWidgets()):
        widget.deleteLater()
    flush_deferred_deletes(qapp)
    assert not qapp.topLevelWidgets(), "Top-level widgets survived test teardown"
    qapp.setStyleSheet("")
    drain_orphan_threads()
    plt.close("all")
