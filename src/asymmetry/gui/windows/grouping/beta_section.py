"""Manual beta value; explicit calibration lives in the shared calibration section."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from asymmetry.gui.styles.widgets import make_warning_banner
from asymmetry.gui.widgets.no_scroll_spin import NoScrollDoubleSpinBox


def beta_status_text(value: float) -> str:
    """Describe the applied amplitude balance."""
    return f"β = {float(value):.4f}"


class BetaSectionWidget(QWidget):
    """Manual coefficient entry and existing correction-staleness notice."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._beta_spin = NoScrollDoubleSpinBox()
        self._beta_spin.setDecimals(6)
        self._beta_spin.setRange(0.01, 1000.0)
        self._beta_spin.setValue(1.0)
        self._beta_spin.valueChanged.connect(self.changed)
        form = QFormLayout()
        form.addRow("β value", self._beta_spin)
        layout.addLayout(form)
        hint = QLabel(
            "Enter β manually or use Calibrate β with TF calibration fit in the calibration section."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._stale_banner = make_warning_banner("", severity="warn")
        self._stale_banner.hide()
        layout.addWidget(self._stale_banner)

    def value(self):
        return float(self._beta_spin.value())

    def set_value(self, value):
        self._beta_spin.setValue(float(value))

    def is_active(self):
        return abs(self.value() - 1.0) > 1e-9

    def set_stale_message(self, text):
        self._stale_banner.setText(text or "")
        self._stale_banner.setVisible(bool(text))
