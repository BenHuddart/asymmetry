"""Project-local plot presentation control outside the crowded plot rows."""

from PySide6.QtCore import QObject, Slot


class PlotDisplayControl(QObject):
    def __init__(self, menu, panel, workspace):
        super().__init__(menu)
        self._panel, self._workspace = panel, workspace
        menu.clear()
        self.action = menu.addAction("Connect data points")
        self.action.setCheckable(True)
        self.action.setToolTip(
            "Join neighbouring data points with straight segments as a guide to the eye. "
            "Zoom in to inspect individual samples."
        )
        self.action.triggered.connect(panel.set_connect_points_enabled)
        panel.connect_points_changed.connect(self.sync_state)
        workspace.active_view_changed.connect(self.sync_state)
        menu.aboutToShow.connect(self.sync_state)
        self.sync_state()

    @Slot()
    def sync_state(self, *_args):
        self.action.setChecked(self._panel.connect_points_enabled())
        self.action.setEnabled(self._workspace.active_view() in {"fb_asymmetry", "groups"})
