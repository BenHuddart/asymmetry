"""Tests for the wheel-guarded spin boxes (F20).

A spin box in a scrolling dock panel must not change value when the user
scrolls *past* it (the audit's F20: scrolling the MaxEnt panel silently moved
"Spectrum points" 1024 → 512). The guard: ignore wheel events unless focused.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QSpinBox

from asymmetry.gui.widgets.no_scroll_spin import (
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wheel_event(widget) -> QWheelEvent:
    """A downward wheel notch delivered to *widget*'s centre."""
    pos = widget.rect().center()
    global_pos = widget.mapToGlobal(pos)
    return QWheelEvent(
        QPoint(pos),
        global_pos,
        QPoint(0, -120),  # pixelDelta
        QPoint(0, -120),  # angleDelta: one notch down
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _send_wheel(qapp, widget) -> None:
    qapp.sendEvent(widget, _wheel_event(widget))


@pytest.mark.parametrize(
    "make, start",
    [
        (NoScrollSpinBox, 10),
        (NoScrollDoubleSpinBox, 10.0),
    ],
)
def test_unfocused_wheel_does_not_change_value(qapp, make, start):
    spin = make()
    spin.setRange(0, 100)
    spin.setValue(start)
    assert not spin.hasFocus()

    _send_wheel(qapp, spin)

    assert spin.value() == start
    spin.deleteLater()


def test_focused_wheel_still_changes_value(qapp):
    spin = NoScrollSpinBox()
    spin.setRange(0, 100)
    spin.setValue(10)
    spin.show()
    spin.setFocus()
    qapp.processEvents()
    # Guard the assertion: offscreen focus can be flaky; force it if needed.
    if not spin.hasFocus():
        spin.activateWindow()
        spin.setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
    assert spin.hasFocus()

    _send_wheel(qapp, spin)

    # A focused spin box wheels normally (down one notch → value decreases).
    assert spin.value() != 10
    spin.deleteLater()


def test_focus_policy_is_strong(qapp):
    # StrongFocus (not the QSpinBox default WheelFocus) so a wheel never focuses
    # the box, which is what lets an unfocused wheel change the value.
    assert NoScrollSpinBox().focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert NoScrollDoubleSpinBox().focusPolicy() == Qt.FocusPolicy.StrongFocus


def test_plain_spinbox_is_the_thing_being_guarded_against(qapp):
    # Sanity check that a *plain* QSpinBox does change on an unfocused wheel —
    # otherwise the guard tests above would pass vacuously.
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(10)
    assert not spin.hasFocus()

    _send_wheel(qapp, spin)

    assert spin.value() != 10
    spin.deleteLater()
