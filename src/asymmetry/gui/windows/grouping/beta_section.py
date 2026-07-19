"""Inline β (intrinsic-asymmetry balance) controls for the Corrections panel.

β = A₀,B/A₀,F corrects the asymmetry for the two detector groups' *intrinsic
asymmetries* differing (solid-angle / absorption effects that scale the
observable amplitude rather than the count rate), the musrfit asymmetry-fit
(fit type 2) companion to α. It enters the reduction as
``A = (F − αB)/(βF + αB)``; β = 1 is the standard formula.

The widget owns no reduction state: a fixed user-entered value plus an
explanation, emitting :attr:`changed` for the owning grouping dialog to
dirty-track and re-preview. There is no estimator — β cannot be measured from
count ratios (it is invisible to count totals), only from the two groups'
fitted asymmetry amplitudes; a data-driven estimate is deferred to the
count-domain F/B fit (see ``docs/porting/beta-correction/``).
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.no_scroll_spin import NoScrollDoubleSpinBox

__all__ = ["BetaSectionWidget", "beta_status_text"]


def beta_status_text(value: float) -> str:
    """One-line β summary for the pipeline chip / card header."""
    return f"β = {float(value):.4f}"


class BetaSectionWidget(QWidget):
    """Fixed-value β entry for the Corrections panel.

    The dialog reads :meth:`value` when building the grouping payload (emitting
    the ``beta`` key only when ≠ 1, so a default payload stays byte-identical
    to a pre-β one) and seeds it back with :meth:`set_value` on a profile/run
    switch.
    """

    #: Emitted when the β value changes.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the value row and the explanation label."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._beta_spin = NoScrollDoubleSpinBox()
        self._beta_spin.setDecimals(6)
        # Same positive range as the α spin: β is a ratio of asymmetry
        # amplitudes and is meaningless at or below zero.
        self._beta_spin.setRange(0.01, 1000.0)
        self._beta_spin.setValue(1.0)
        self._beta_spin.valueChanged.connect(self.changed)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(12)
        form.addRow("β value", self._beta_spin)
        layout.addLayout(form)

        hint = QLabel(
            "β = A₀,b/A₀,f corrects for the two groups' intrinsic asymmetries "
            "differing (musrfit asymmetry fit). Leave at 1 unless a calibration "
            "has measured it; there is no count-based estimate for β."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(hint)

    # -- value plumbing ---------------------------------------------------

    def value(self) -> float:
        """The current β value (always positive — the spin enforces the range)."""
        return float(self._beta_spin.value())

    def set_value(self, value: float) -> None:
        """Seed the spin, mapping degenerate values to the 1.0 default."""
        try:
            beta = float(value)
        except (TypeError, ValueError):
            beta = 1.0
        if not math.isfinite(beta) or beta <= 0.0:
            beta = 1.0
        self._beta_spin.setValue(beta)

    def is_active(self) -> bool:
        """Whether β departs from the do-nothing default."""
        return abs(self.value() - 1.0) > 1e-9
