"""Dialog to configure a joint K(θ) fit with per-angle component assignment."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.angular_assignment import ANGULAR_MODELS
from asymmetry.gui.styles.metrics import dialog_width

_MODEL_LABELS = {
    "KnightAnisotropy": "Axial  K_iso + K_ax·(3cos²θ−1)/2",
    "AngularCos2": "Two-fold  K_avg + K_amp·cos2(θ−θ₀)",
}


class KnightJointFitDialog(QDialog):
    """Choose the K(θ) model and iteration cap for a joint multi-curve fit."""

    def __init__(self, *, n_curves: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Joint K(θ) Fit")
        self.setMinimumWidth(dialog_width(58))  # ~420px at default scale
        self._result: tuple[str, int] | None = None

        root = QVBoxLayout(self)
        intro = QLabel(
            f"Fit {n_curves} K(θ) curves jointly. At each angle the {n_curves} component "
            "points are assigned one-to-one to the curves they best fit, so each site is "
            "followed continuously through crossings."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._model_combo = QComboBox()
        for name in ANGULAR_MODELS:
            self._model_combo.addItem(_MODEL_LABELS.get(name, name), userData=name)
        form.addRow("Model:", self._model_combo)

        self._max_iter = QSpinBox()
        self._max_iter.setRange(1, 200)
        self._max_iter.setValue(25)
        form.addRow("Max iterations:", self._max_iter)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _on_accept(self) -> None:
        self._result = (str(self._model_combo.currentData()), int(self._max_iter.value()))
        self.accept()

    def joint_fit_config(self) -> tuple[str, int] | None:
        """Return ``(model_name, max_iter)`` when accepted, else None."""
        return self._result
