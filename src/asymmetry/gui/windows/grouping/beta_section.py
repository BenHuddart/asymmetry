"""Manual beta value; explicit calibration lives in the shared calibration section."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from asymmetry.core.project.calibration import CalibratedCoefficient
from asymmetry.core.transform.grouping import grouping_balances
from asymmetry.gui.styles.widgets import make_warning_banner
from asymmetry.gui.widgets.no_scroll_spin import NoScrollDoubleSpinBox
from asymmetry.gui.windows.grouping.format import ALPHA_METHOD_ITEMS


def beta_provenance_text(policy) -> str:
    """Read the captured beta constraint and method for its coefficient view."""
    if isinstance(policy, CalibratedCoefficient) and policy.result.beta_fixed:
        return "fixed"
    return next(
        (label for label, key, _ in ALPHA_METHOD_ITEMS if key == policy.method),
        policy.method or "manual",
    )


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
        row = QWidget()
        fields = QHBoxLayout(row)
        fields.setContentsMargins(0, 0, 0, 0)
        self._provenance_label = QLabel("manual")
        fields.addWidget(self._beta_spin)
        fields.addWidget(self._provenance_label, 1)
        form.addRow("β value", row)
        layout.addLayout(form)
        self._stale_banner = make_warning_banner("", severity="warn")
        self._stale_banner.hide()
        layout.addWidget(self._stale_banner)

    def value(self):
        return float(self._beta_spin.value())

    def set_value(self, value):
        self._beta_spin.setValue(grouping_balances({"beta": value})[1])

    def is_active(self):
        return abs(self.value() - 1.0) > 1e-9

    def set_provenance(self, text):
        self._provenance_label.setText(text)

    def set_stale_message(self, text):
        self._stale_banner.setText(text or "")
        self._stale_banner.setVisible(bool(text))
