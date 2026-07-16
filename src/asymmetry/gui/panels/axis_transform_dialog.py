"""Dialog for entering a custom per-axis trend transform expression.

A deliberately slim sibling of :class:`~asymmetry.gui.panels.composite_parameter_dialog.
CompositeParameterDialog`: a single-variable transform is short, so there is one
expression field (no calculator keypad), live validation, and a preview on a
representative data value with the propagated uncertainty.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.axis_transforms import (
    AXIS_VARIABLE,
    AxisTransform,
    validate_axis_expression,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import dialog_width


class AxisTransformDialog(QDialog):
    """Enter a custom transform expression in the axis variable ``x``."""

    def __init__(
        self,
        *,
        axis_label: str,
        initial_expression: str = "",
        sample_value: float | None = None,
        sample_error: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Custom {axis_label} transform")
        self.setMinimumWidth(dialog_width(64))  # ~460px at default scale

        self._sample_value = sample_value
        self._sample_error = sample_error

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self._edit = QLineEdit(initial_expression)
        self._edit.setPlaceholderText("e.g. 1000/x")
        self._edit.textChanged.connect(self._revalidate)
        form.addRow(f"Expression in {AXIS_VARIABLE}:", self._edit)
        layout.addLayout(form)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._revalidate()

    def expression(self) -> str:
        """The validated expression text (call after :meth:`exec` accepts)."""
        return self._edit.text().strip()

    def _revalidate(self) -> None:
        text = self._edit.text().strip()
        ok, message = validate_axis_expression(text)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
        if not ok:
            self._status.setText(message or "Invalid expression")
            self._status.setStyleSheet(f"color: {tokens.ERROR};")
            return
        self._status.setStyleSheet(f"color: {tokens.OK};")
        self._status.setText(self._preview_text(text) or "Valid expression")

    def _preview_text(self, expression: str) -> str | None:
        """A 'x = 250 → 4 ± 0.12' preview on the representative sample value."""
        value = self._sample_value
        if value is None or not np.isfinite(value):
            return None
        transform = AxisTransform.custom(expression)
        error = self._sample_error if self._sample_error is not None else np.nan
        out_v, out_e = transform.apply([float(value)], [float(error)])
        if not np.isfinite(out_v[0]):
            return f"Preview: {AXIS_VARIABLE} = {value:.4g} → undefined here"
        rendered = f"Preview: {AXIS_VARIABLE} = {value:.4g} → {out_v[0]:.4g}"
        if np.isfinite(out_e[0]):
            rendered += f" ± {out_e[0]:.3g}"
        return rendered
