"""Standalone tests for the WizardStepper widget."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from asymmetry.gui.widgets.wizard_stepper import StepState, WizardStepper

_STEPS = [
    ("scope", "Scope"),
    ("screen", "Screen"),
    ("compare", "Compare"),
    ("phases", "Phases"),
    ("apply", "Apply"),
]


def _buttons(stepper: WizardStepper) -> dict[str, QPushButton]:
    buttons = stepper.findChildren(QPushButton)
    return {key: button for (key, _title), button in zip(_STEPS, buttons, strict=True)}


def test_starts_pending_with_the_first_step_current(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    buttons = _buttons(stepper)
    assert stepper.current_key() == "scope"
    assert all(stepper.state(key) is StepState.PENDING for key, _ in _STEPS)
    assert buttons["scope"].accessibleName() == "Step 1, Scope, pending, current"
    assert buttons["screen"].accessibleName() == "Step 2, Screen, pending"


def test_set_step_names_the_state_and_carries_the_summary(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("screen", StepState.DONE, "LF Kubo–Toyabe leads")
    button = _buttons(stepper)["screen"]
    assert stepper.state("screen") is StepState.DONE
    assert button.accessibleName() == "Step 2, Screen, done"
    assert button.accessibleDescription() == "LF Kubo–Toyabe leads"
    assert button.toolTip() == "LF Kubo–Toyabe leads"


@pytest.mark.parametrize(
    ("state", "clickable"),
    [
        (StepState.PENDING, False),
        (StepState.SKIPPED, False),
        (StepState.READY, True),
        (StepState.RUNNING, True),
        (StepState.DONE, True),
        (StepState.STALE, True),
    ],
)
def test_clickability_follows_state(qapp: QApplication, state: StepState, clickable: bool) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("compare", state, "")
    assert _buttons(stepper)["compare"].isEnabled() is clickable


def test_the_current_step_is_clickable_whatever_its_state(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("phases", StepState.SKIPPED, "No transition found")
    stepper.set_current("phases")
    buttons = _buttons(stepper)
    assert buttons["phases"].isEnabled()
    assert not buttons["scope"].isEnabled()  # pending and no longer current


def test_set_current_moves_the_single_current_step(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("screen", StepState.STALE, "Scope changed")
    stepper.set_current("screen")
    names = [button.accessibleName() for button in _buttons(stepper).values()]
    assert stepper.current_key() == "screen"
    assert sum(name.endswith(", current") for name in names) == 1
    assert _buttons(stepper)["screen"].accessibleName() == "Step 2, Screen, stale, current"


def test_set_current_rejects_an_unknown_step(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    with pytest.raises(KeyError):
        stepper.set_current("results")


def test_clicking_a_step_requests_it(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("screen", StepState.DONE, "")
    requested: list[str] = []
    stepper.step_requested.connect(requested.append)
    _buttons(stepper)["screen"].click()
    assert requested == ["screen"]


def test_a_step_is_reachable_from_the_keyboard(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("screen", StepState.DONE, "")
    button = _buttons(stepper)["screen"]
    requested: list[str] = []
    stepper.step_requested.connect(requested.append)
    assert button.focusPolicy() & Qt.FocusPolicy.TabFocus
    QTest.keyClick(button, Qt.Key.Key_Space)
    assert requested == ["screen"]


def test_five_steps_fit_1180px_with_long_summaries_eliding(qapp: QApplication) -> None:
    stepper = WizardStepper(_STEPS)
    long_summary = "Stretched exponential × Gaussian Kubo–Toyabe + Const leads by 12.3"
    for key, _title in _STEPS:
        stepper.set_step(key, StepState.DONE, long_summary)
    assert stepper.minimumSizeHint().width() <= 1180
    stepper.resize(1180, stepper.sizeHint().height())
    stepper.show()
    button = _buttons(stepper)["screen"]
    assert button.sizeHint().width() > button.width()  # the summary is elided
    assert not stepper.grab().isNull()


@pytest.mark.parametrize("state", list(StepState))
@pytest.mark.parametrize("current", [False, True])
def test_every_state_paints(qapp: QApplication, state: StepState, current: bool) -> None:
    stepper = WizardStepper(_STEPS)
    stepper.set_step("compare", state, "Summary")
    if current:
        stepper.set_current("compare")
    stepper.resize(1180, stepper.sizeHint().height())
    assert not stepper.grab().isNull()
