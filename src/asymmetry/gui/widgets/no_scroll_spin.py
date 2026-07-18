"""Input widgets that ignore the mouse wheel unless they hold keyboard focus.

A plain ``QSpinBox``/``QDoubleSpinBox``/``QComboBox`` defaults to
``Qt.WheelFocus`` and consumes wheel events even when unfocused. When such a
control lives inside a scrolling panel (MaxEnt, Fourier, the fit-range block,
the grouping/reduction editor), scrolling *past* it silently mutates its value —
the audit's F20: scrolling the MaxEnt panel changed "Spectrum points" 1024 → 512
with no intent, and scrolling the grouping editor could nudge a correction combo.

:class:`NoScrollSpinBox` / :class:`NoScrollDoubleSpinBox` / :class:`NoScrollComboBox`
fix this by (a) using ``Qt.StrongFocus`` so a wheel event never focuses them, and
(b) ignoring a wheel event when they are not focused, so it propagates to the
enclosing scroll area instead of changing the value. A focused control still
wheels normally — the user has deliberately clicked/tabbed into it. Use them at
the construction site in place of the plain Qt classes.

(On macOS the native combo style already declines the unfocused wheel, so the
combo bug is largely invisible there but live on Linux/Windows — don't let a Mac
green convince you the combo case is theoretical.)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class _WheelGuardMixin:
    """Ignore wheel events unless focused; scroll passes to the parent instead."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # A wheel event must never *focus* the spin box (that is what lets an
        # unfocused wheel change the value in the first place).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            # Ignore (not accept) so the enclosing QScrollArea scrolls instead.
            event.ignore()


class NoScrollSpinBox(_WheelGuardMixin, QSpinBox):
    """A ``QSpinBox`` that only wheels when focused."""


class NoScrollDoubleSpinBox(_WheelGuardMixin, QDoubleSpinBox):
    """A ``QDoubleSpinBox`` that only wheels when focused."""


class NoScrollComboBox(_WheelGuardMixin, QComboBox):
    """A ``QComboBox`` that only wheels when focused.

    The guard fires on the combo itself (popup closed); an *open* popup gets its
    own wheel events on the popup view, so scrolling a dropdown list is unaffected.
    """
