"""Application-owned notifications for the shared recent-projects list."""

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal


class _RecentProjectsBroadcast(QObject):
    recent_projects_changed = Signal()


def recent_projects_broadcast() -> _RecentProjectsBroadcast:
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() != app.thread():
        raise RuntimeError("Recent Projects notifications require the Qt application thread")
    broadcast = app.findChild(_RecentProjectsBroadcast)
    if broadcast is None:
        broadcast = _RecentProjectsBroadcast(app)
    return broadcast
