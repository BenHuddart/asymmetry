"""Nonuniform width controls fit beneath the grouping window's mode selector."""

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from tests.gui.test_grouping_dialog import _dataset_with_histograms

pytestmark = pytest.mark.gui


@pytest.mark.parametrize("mode", ["variable", "constant_error"])
def test_width_fields_stay_below_mode_without_horizontal_scroll(qapp, monkeypatch, mode):
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Discard)
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog.resize(*dialog.preferred_window_size())
    dialog.show()
    QApplication.processEvents()
    width = dialog.width()
    dialog._set_binning_mode(mode)
    QApplication.processEvents()
    bottom = dialog._binning_mode_combo.mapTo(
        dialog, QPoint(0, dialog._binning_mode_combo.height())
    ).y()
    for spin in [dialog._bin0_spin] + ([dialog._bin10_spin] if mode == "variable" else []):
        assert spin.isVisible()
        assert spin.mapTo(dialog, QPoint(0, 0)).y() > bottom
    binning_label = next(
        label for label in dialog.findChildren(QLabel) if label.text() == "Binning"
    )
    parent_text_left = binning_label.mapTo(dialog, QPoint(0, 0)).x() + max(
        0, binning_label.indent()
    )
    for label in [dialog._bin0_label] + ([dialog._bin10_label] if mode == "variable" else []):
        text_left = label.mapTo(dialog, QPoint(0, 0)).x() + max(0, label.indent())
        assert text_left > parent_text_left
    assert dialog.width() == width
    assert dialog._grouping_scroll.horizontalScrollBar().maximum() == 0
    assert dialog._bin10_spin.isVisible() == (mode == "variable")
    dialog._set_binning_mode("fixed")
    QApplication.processEvents()
    assert not dialog._bin0_spin.isVisible()
    assert not dialog._bin10_spin.isVisible()
    assert dialog._bunch_spin.isEnabled()
    dialog.close()
