"""Fixed ordinary-fit source response controls shared by model authoring."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QWidget

from asymmetry.core.fitting.time_response import TimeResponse
from asymmetry.gui.widgets.no_scroll_spin import NoScrollComboBox, NoScrollDoubleSpinBox


class TimeResponseEditor(QWidget):
    """Author a fixed double-delta setting without adding fit parameters."""

    changed = Signal()

    def __init__(self, response: TimeResponse, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = NoScrollComboBox()
        self.mode.addItem("Off", "off")
        self.mode.addItem("Fixed double pulse", "double_delta")
        self.mode.setToolTip(
            "Two instantaneous pulses with equal incident strengths and muon-survival "
            "weighting. Times use the supplied pulse midpoint on the observation clock."
        )
        self.separation = NoScrollDoubleSpinBox()
        self.separation.setDecimals(6)
        self.separation.setRange(0, 1e6)
        self.separation.setSuffix(" µs")
        self.separation.setToolTip(
            "Supplied fixed separation in the double-delta approximation; never fitted. "
            "This setting does not move acquired time zero or model finite pulse width."
        )
        self.mode.setCurrentIndex(self.mode.findData(response.mode))
        self.separation.setValue(response.separation_us if response.mode != "off" else 0.324)
        self.separation.setEnabled(response.mode != "off")
        layout.addRow("Time response", self.mode)
        layout.addRow("Pulse separation", self.separation)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.separation.valueChanged.connect(self.changed)

    def _mode_changed(self) -> None:
        self.separation.setEnabled(self.mode.currentData() != "off")
        self.changed.emit()

    def value(self) -> TimeResponse:
        """Return the typed consumed settings for a new model snapshot."""
        return TimeResponse(self.mode.currentData(), self.separation.value())
