"""Knight-shift configuration dialog (Phase 3d)."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
)
from asymmetry.gui.panels.knight_shift_dialog import KnightShiftDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_default_is_applied_field_and_ok_builds_enabled(qapp):
    dialog = KnightShiftDialog(
        available_components=["frequency", "frequency_2"],
        config=KnightShiftConfig(),
    )
    assert dialog._applied_radio.isChecked()
    dialog._on_accept()
    cfg = dialog.knight_shift_config()
    assert cfg.enabled is True
    assert cfg.reference_mode == REFERENCE_APPLIED_FIELD


def test_component_mode_carries_reference_and_checked_components(qapp):
    dialog = KnightShiftDialog(
        available_components=["frequency", "frequency_2", "frequency_3"],
        config=KnightShiftConfig(),
    )
    dialog._component_radio.setChecked(True)
    idx = dialog._reference_combo.findData("frequency_2")
    dialog._reference_combo.setCurrentIndex(idx)
    # Tick frequency_3 as the only component to convert.
    for i in range(dialog._component_list.count()):
        item = dialog._component_list.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == "frequency_3":
            item.setCheckState(Qt.CheckState.Checked)
    dialog._on_accept()
    cfg = dialog.knight_shift_config()
    assert cfg.reference_mode == REFERENCE_COMPONENT
    assert cfg.reference_component == "frequency_2"
    assert cfg.components == ("frequency_3",)


def test_disable_button_yields_disabled_config(qapp):
    dialog = KnightShiftDialog(
        available_components=["frequency"],
        config=KnightShiftConfig(enabled=True),
    )
    dialog._on_disable()
    cfg = dialog.knight_shift_config()
    assert cfg.enabled is False


def test_initial_config_round_trips_into_widgets(qapp):
    dialog = KnightShiftDialog(
        available_components=["frequency", "frequency_2"],
        config=KnightShiftConfig(
            enabled=True,
            reference_mode=REFERENCE_COMPONENT,
            reference_component="frequency",
            unit=KnightShiftUnit.PERCENT,
            components=("frequency_2",),
        ),
    )
    assert dialog._component_radio.isChecked()
    assert dialog._reference_combo.currentData() == "frequency"
    assert dialog._unit_combo.currentData() is KnightShiftUnit.PERCENT


def test_no_components_disables_ok(qapp):
    dialog = KnightShiftDialog(available_components=[], config=KnightShiftConfig())
    assert not dialog._buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
