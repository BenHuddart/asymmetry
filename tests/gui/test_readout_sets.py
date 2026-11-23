"""Applied readout choices retain recorded groups until an explicit transition."""

from copy import deepcopy

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QDialog, QInputDialog, QMessageBox, QTableWidget

from asymmetry.core.instrument import get_instrument_layout
from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

pytestmark = pytest.mark.gui


def transition_button(monkeypatch, text):
    monkeypatch.setattr(QMessageBox, "exec", lambda _q: 0)
    monkeypatch.setattr(
        QMessageBox,
        "clickedButton",
        lambda question: next(b for b in question.buttons() if b.text().replace("&", "") == text),
    )


def editor(groups=None, *, choice=None):
    return DetectorLayoutDialog(
        get_instrument_layout("GPS-RD15"),
        groups or {1: [1], 2: [2], 3: [7, 8], 4: [9, 10]},
        readout_set=choice,
    )


def test_legacy_open_ok_has_no_new_default_or_hidden_assignment_loss(qapp):
    dialog = editor({1: [6], 2: [13, 14], 9: [3]})
    result = deepcopy(dialog.get_result())
    assert "readout_set" not in result
    assert dialog._readout_combo.currentIndex() == -1
    assert {s.detector_id for s in dialog._schematic._instrument.active_segments} == {
        1,
        2,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    }
    text = dialog._outside_readouts.toPlainText()
    assert "6 Left: groups 1" in text
    assert "3 Up: groups 9" in text
    dialog._on_ok()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.get_result() == result
    dialog.close()


@pytest.mark.parametrize("layout,label", [("GPS", "Hardware-combined"), ("GPS-RD", "Individual")])
def test_single_choice_is_static_and_unrelated_instruments_gain_no_control(qapp, layout, label):
    dialog = DetectorLayoutDialog(get_instrument_layout(layout), {1: [1], 2: [2]})
    assert dialog._readout_combo.isHidden()
    assert dialog._readout_name.text() == label
    assert not dialog._readout_name.isHidden()
    dialog.close()
    other = DetectorLayoutDialog(get_instrument_layout("EMU"), {1: [1], 2: [2]})
    assert other._readout_heading.isHidden()
    assert other._readout_name.isHidden()
    other.close()


def test_cancel_transition_preserves_every_setting_and_emits_no_channel_change(qapp, monkeypatch):
    dialog = editor(choice="individual")
    before = deepcopy(dialog.get_result())
    assignment = QSignalSpy(dialog._schematic.detector_toggled)
    exclusion = QSignalSpy(dialog._schematic.detector_exclusion_toggled)
    transition_button(monkeypatch, "Cancel")
    dialog._readout_combo.setCurrentIndex(dialog._readout_combo.findData("hardware_combined"))
    assert dialog.get_result() == before
    assert dialog._readout_combo.currentData() == "individual"
    assert assignment.count() == exclusion.count() == 0
    dialog.close()


def test_preset_transition_uses_explicit_real_combined_channels(qapp, monkeypatch):
    dialog = editor(choice="individual")
    transition_button(monkeypatch, "Apply preset")
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("WEP (combined)", True))
    dialog._readout_combo.setCurrentIndex(dialog._readout_combo.findData("hardware_combined"))
    result = dialog.get_result()
    assert result["readout_set"] == "hardware_combined"
    assert result["groups"] == {1: [1], 2: [2], 3: [3], 4: [4]}
    assert [(p["label"], p["alpha"]) for p in result["projections"]] == [("FB", 0.75), ("UD", 1.0)]
    assert set(dialog._schematic._patches) == {1, 2, 3, 4, 5, 6, 15}
    assert dialog._readout_notice.text() == "May include Mobile counts"
    assert dialog._outside_readouts.isHidden()
    dialog.close()


def test_custom_edit_stages_explicit_channels_and_preserves_other_groups(qapp, monkeypatch):
    dialog = editor({1: [7, 8], 2: [9, 10], 9: [15]}, choice="individual")
    dialog.show()
    transition_button(monkeypatch, "Edit grouping")

    def accept_edit():
        modal = qapp.activeModalWidget()
        assert isinstance(modal, QDialog)
        table = modal.findChild(QTableWidget)
        for row in range(table.rowCount()):
            gid = int(table.item(row, 0).text())
            if gid in {1, 2}:
                table.item(row, 1).setText(str(gid + 2))
        modal.accept()

    QTimer.singleShot(0, accept_edit)
    dialog._readout_combo.setCurrentIndex(dialog._readout_combo.findData("hardware_combined"))
    assert dialog.get_result()["groups"] == {1: [3], 2: [4], 9: [15]}
    assert dialog.get_result()["readout_set"] == "hardware_combined"
    dialog.close()


def test_shared_forward_backward_change_does_not_reassign_and_explicit_preset_establishes_choice(
    qapp,
):
    dialog = editor({1: [1], 2: [2]})
    dialog._readout_combo.setCurrentIndex(dialog._readout_combo.findData("hardware_combined"))
    assert dialog.get_result()["groups"] == {1: [1], 2: [2]}
    assert dialog.get_result()["readout_set"] == "hardware_combined"
    dialog.close()
    fresh = editor({1: [1], 2: [2]})
    fresh._preset_combo.setCurrentText("Longitudinal")
    fresh._on_apply_preset()
    assert fresh.get_result()["readout_set"] == "individual"
    fresh.close()


def grouping_dataset():
    """A real acquired GPS source with the Individual WEP projection defaults."""
    import numpy as np

    from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
    from asymmetry.gui.windows.grouping.profile_bridge import preset_payload

    grouping = preset_payload(
        get_instrument_layout("GPS-RD15"), "WEP (spin-rotated)", readout_set="individual"
    )
    grouping.update(instrument="GPS-RD15", first_good_bin=0, last_good_bin=39, alpha=1.0)
    run = Run(
        run_number=680,
        histograms=[
            Histogram(
                np.full(40, 100.0 + i), bin_width=0.1, t0_bin=0, good_bin_start=0, good_bin_end=39
            )
            for i in range(15)
        ],
        metadata={"instrument": "GPS", "facility": "PSI", "run_number": 680},
        grouping=grouping,
    )
    return MuonDataset(np.arange(40) * 0.1, np.zeros(40), np.ones(40), run=run)


@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_parent_actual_child_stages_changed_projection_without_alpha_transfer(
    qapp, monkeypatch, accept, captured
):
    from asymmetry.core.workflow.calibration import CalibrationOptions, calibrate_run
    from asymmetry.core.workflow.reduction import ReductionSettings
    from asymmetry.gui.windows.grouping.alpha_section import CalibrationReply
    from asymmetry.gui.windows.grouping_dialog import GroupingDialog

    dataset = grouping_dataset()
    source = deepcopy(dataset.run.grouping)
    dialog = GroupingDialog([dataset])
    if captured:
        outcome = calibrate_run(
            dataset, ReductionSettings(pair=("3", "4")), CalibrationOptions("ratio")
        )
        dialog._on_calibration_ready(CalibrationReply(0, "UD", outcome))
        assert "UD" in dialog._draft.projection_alpha_policies
        assert "UD" in dialog._alpha_estimate_state
    else:
        dialog._vector_alpha_spins["UD"].setValue(1.234)
    before = deepcopy(dialog._current_grouping_payload())
    old_alpha = dialog._vector_alpha_spins["UD"].value()
    transition_button(monkeypatch, "Apply preset")
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("WEP (combined)", True))

    def finish_child():
        child = qapp.activeModalWidget()
        assert isinstance(child, DetectorLayoutDialog)
        child._readout_combo.setCurrentIndex(child._readout_combo.findData("hardware_combined"))
        child.accept() if accept else child.reject()

    QTimer.singleShot(0, finish_child)
    dialog._on_detector_layout()
    after = dialog._current_grouping_payload()
    if accept:
        assert after["readout_set"] == "hardware_combined"
        assert after["groups"] == {1: [1], 2: [2], 3: [3], 4: [4]}
        assert {p["label"]: p["alpha"] for p in after["projections"]} == {"FB": 0.75, "UD": 1.0}
        assert "UD" not in dialog._draft.projection_alpha_policies
        assert "UD" not in dialog._alpha_estimate_state
    else:
        assert after == before
        assert dialog._vector_alpha_spins["UD"].value() == old_alpha
        assert ("UD" in dialog._draft.projection_alpha_policies) == captured
    assert dataset.run.grouping == source
    dialog._clear_dirty()
    dialog.close()


def test_parent_custom_readout_edit_keeps_explicit_manual_alpha(qapp, monkeypatch):
    from asymmetry.gui.windows.grouping_dialog import GroupingDialog

    dialog = GroupingDialog([grouping_dataset()])
    dialog._vector_alpha_spins["UD"].setValue(1.234)
    transition_button(monkeypatch, "Edit grouping")

    def accept_numeric_edit():
        modal = qapp.activeModalWidget()
        assert isinstance(modal, QDialog)
        table = modal.findChild(QTableWidget)
        for row in range(table.rowCount()):
            gid = int(table.item(row, 0).text())
            if gid in {3, 4}:
                table.item(row, 1).setText(str(gid))
        modal.accept()

    def finish_child():
        child = qapp.activeModalWidget()
        assert isinstance(child, DetectorLayoutDialog)
        QTimer.singleShot(0, accept_numeric_edit)
        child._readout_combo.setCurrentIndex(child._readout_combo.findData("hardware_combined"))
        child.accept()

    QTimer.singleShot(0, finish_child)
    dialog._on_detector_layout()
    payload = dialog._current_grouping_payload()
    assert payload["groups"][3] == [3]
    assert {p["label"]: p["alpha"] for p in payload["projections"]} == {"FB": 0.75, "UD": 1.234}
    assert "UD" not in dialog._draft.projection_alpha_policies
    dialog._clear_dirty()
    dialog.close()


def test_unavailable_saved_set_explicit_preset_repairs_diagram_and_preserves_exclusions(qapp):
    dialog = editor({1: [1], 2: [2]}, choice="retired")
    dialog._schematic.set_excluded_detectors({15})
    assert set(dialog._schematic._patches) == set(range(1, 16))
    dialog._preset_combo.setCurrentText("WEP (combined)")
    dialog._on_apply_preset()
    result = dialog.get_result()
    assert result["readout_set"] == "hardware_combined"
    assert result["groups"] == {1: [1], 2: [2], 3: [3], 4: [4]}
    assert result["excluded_detectors"] == [15]
    assert set(dialog._schematic._patches) == {1, 2, 3, 4, 5, 6, 15}
    assert dialog._readout_combo.currentData() == "hardware_combined"
    assert dialog._preset_combo.currentText() == "WEP (combined)"
    assert "WEP (spin-rotated)" not in {
        dialog._preset_combo.itemText(i) for i in range(dialog._preset_combo.count())
    }
    dialog.close()
