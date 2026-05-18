"""Tests for grouping dialog alpha-estimation workflow."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

import asymmetry.gui.windows.grouping_dialog as grouping_dialog_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _dataset_with_histograms() -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=4001,
        histograms=[h1, h2],
        metadata={"run_number": 4001, "title": "Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": 4001},
        run=run,
    )


def _vector_dataset_with_histograms(run_number: int = 4010) -> MuonDataset:
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "Vector Grouping Test"},
        grouping={
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "alpha_x": 1.1,
            "alpha_y": 1.2,
            "alpha_z": 1.3,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dataset_with_deadtime_profile(run_number: int, tau_us: float) -> MuonDataset:
    amplitude = 120.0
    bin_width = 0.01
    num_good_frames = 1000.0
    lifetime_us = 2.1969811
    times = (np.arange(12, dtype=float) + 1.0) * bin_width
    frame_scale = num_good_frames * bin_width
    true_counts = amplitude * np.exp(-times / lifetime_us)
    observed = true_counts * (
        1.0 - (true_counts / frame_scale) * lifetime_us * (1.0 - np.exp(-tau_us / lifetime_us))
    )
    histograms = [Histogram(observed.copy(), bin_width=bin_width) for _ in range(4)]
    run = Run(
        run_number=run_number,
        histograms=histograms,
        metadata={"run_number": run_number, "title": f"Deadtime Run {run_number}"},
        grouping={
            "groups": {1: [1, 2], 2: [3, 4]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 11,
            "good_frames": num_good_frames,
        },
    )
    return MuonDataset(
        time=np.arange(12, dtype=float) * bin_width,
        asymmetry=np.zeros(12, dtype=float),
        error=np.full(12, 0.01, dtype=float),
        metadata={"run_number": run_number},
        run=run,
    )


def _vector_dataset_with_ratio(run_number: int, ratio: float) -> MuonDataset:
    forward = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)
    backward = forward / ratio
    h1 = Histogram(counts=forward, bin_width=0.01)
    h2 = Histogram(counts=backward, bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": f"Vector Run {run_number}"},
        grouping={
            "groups": {
                1: [1],
                2: [2],
                3: [1],
                4: [2],
                5: [1],
                6: [2],
            },
            "group_names": {
                1: "Pz Forward",
                2: "Pz Backward",
                3: "Py Top",
                4: "Py Bottom",
                5: "Px Left",
                6: "Px Right",
            },
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _dataset_with_ratio(run_number: int, ratio: float) -> MuonDataset:
    forward = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)
    backward = forward / ratio
    h1 = Histogram(counts=forward, bin_width=0.01)
    h2 = Histogram(counts=backward, bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": f"Run {run_number}"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 3,
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number},
        run=run,
    )


def _two_period_dataset(run_number: int = 6001) -> MuonDataset:
    red = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    green = Histogram(counts=np.array([120.0, 120.0, 120.0, 120.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[red],
        metadata={"run_number": run_number, "period_count": 2},
        grouping={
            "groups": {1: [1], 2: [1]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "period_mode": str(PeriodMode.GREEN),
            "period_histograms": [[red], [green]],
        },
    )
    t = np.array([0.0, 0.01, 0.02, 0.03])
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "period_count": 2},
        run=run,
    )


def test_estimate_alpha_updates_spinbox(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    assert dialog._alpha_spin.value() == pytest.approx(2.0)


def test_get_grouping_result_contains_required_keys(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.metadata["facility"] = "PSI"
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["dead_time_us"] = [0.01, 0.01]
    dialog = GroupingDialog([dataset])
    dialog._deadtime_checkbox.setChecked(True)
    dialog._background_checkbox.setChecked(True)
    dialog._bunch_spin.setValue(1234)
    result = dialog.get_grouping_result()
    assert result is not None
    assert "forward_indices" in result
    assert "backward_indices" in result
    assert "alpha" in result
    assert "included_groups" in result
    assert result["included_groups"] == {1: True, 2: True}
    assert result["deadtime_correction"] is True
    assert result["background_correction"] is True
    assert result["bunching_factor"] == 1234


def test_grouping_result_respects_group_include_checkbox(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    include_item = dialog._group_table.item(1, 1)
    assert include_item is not None
    include_item.setCheckState(Qt.CheckState.Unchecked)

    result = dialog.get_grouping_result()

    assert result is not None
    assert result["included_groups"] == {1: True, 2: False}


def test_grouping_dialog_does_not_show_bunching_rules(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    labels = {label.text() for label in dialog.findChildren(QLabel)}

    assert "Bunching Rules" not in labels
    assert dialog._bunch_spin.toolTip() == "Set any bunching factor >= 1."


def test_deadtime_modes_available_without_file_deadtime(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    payload = dialog._current_grouping_payload()

    assert dialog._deadtime_checkbox.isEnabled()
    assert dialog._deadtime_checkbox.isChecked() is False
    assert dialog._deadtime_mode_buttons["file"].isChecked()
    assert "load" not in dialog._deadtime_mode_buttons
    assert payload["deadtime_correction"] is False
    dialog._deadtime_checkbox.setChecked(True)
    assert dialog._deadtime_mode_buttons["file"].isEnabled()
    assert dialog._deadtime_mode_buttons["file"].isChecked()
    assert dialog._deadtime_mode_buttons["manual"].isEnabled()
    assert dialog._deadtime_mode_buttons["estimate"].isEnabled()


def test_manual_deadtime_payload_resolves_uniform_values(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("manual")
    dialog._update_deadtime_controls()
    dialog._deadtime_value_combo.setCurrentIndex(0)
    dialog._deadtime_value_combo.setEditText("25.0")
    dialog._on_deadtime_value_edited()
    dialog._deadtime_value_combo.setCurrentIndex(1)
    dialog._deadtime_value_combo.setEditText("25.0")
    dialog._on_deadtime_value_edited()

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_correction"] is True
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "manual"
    assert payload["dead_time_us"] == pytest.approx([0.025, 0.025])


def test_file_deadtime_updates_detector_value_combo(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["dead_time_us"] = [0.011, 0.022]
    dialog = GroupingDialog([dataset])

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("file")
    dialog._update_deadtime_controls()

    assert dialog._deadtime_value_combo.count() == 2
    assert dialog._deadtime_value_combo.itemText(0) == "H1: 11.000 ns"
    assert dialog._deadtime_value_combo.itemText(1) == "H2: 22.000 ns"


def test_estimate_deadtime_uses_reference_run_only(qapp: QApplication) -> None:
    reference = _dataset_with_deadtime_profile(4101, 0.02)
    other = _dataset_with_deadtime_profile(4102, 0.04)
    dialog = GroupingDialog(
        [reference, other],
        selected_run_number=4101,
        selected_run_numbers=[4101, 4102],
    )

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("estimate")

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_mode"] == "estimate"
    assert payload["deadtime_method"] == "estimate"
    assert payload["deadtime_reference_run"] == 4101
    assert payload["run_numbers"] == [4101, 4102]
    assert payload["dead_time_us"] == pytest.approx([0.02, 0.02, 0.02, 0.02], rel=1e-2, abs=5e-4)
    assert dialog._deadtime_value_combo.itemText(0).startswith("H1: 20.000")


def test_calibrate_deadtime_populates_explicit_table(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    monkeypatch.setattr(
        grouping_dialog_module,
        "calibrate_deadtime_from_histograms",
        lambda *args, **kwargs: [0.011, 0.022],
    )

    dialog._deadtime_checkbox.setChecked(True)
    dialog._set_deadtime_mode("manual")
    dialog._update_deadtime_controls()
    assert dialog._deadtime_calibrate_btn.isEnabled()
    assert "Fit one deadtime value per detector" in dialog._deadtime_calibrate_btn.toolTip()
    dialog._calibrate_deadtime_from_reference()
    payload = dialog.get_grouping_result()

    assert payload is not None
    assert dialog._deadtime_mode_buttons["manual"].isChecked()
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "calibrate"
    assert payload["dead_time_us"] == pytest.approx([0.011, 0.022])
    assert payload["deadtime_reference_run"] == 4001
    assert dialog._deadtime_value_combo.itemText(0) == "H1: 11.000 ns"


def test_background_checkbox_disabled_for_non_psi_data(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    assert not dialog._background_checkbox.isEnabled()
    assert dialog._current_grouping_payload()["background_correction"] is False


def test_vector_mode_shows_per_axis_alpha_controls(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])

    assert dialog._vector_axis_pairs
    assert not dialog._vector_alpha_widget.isHidden()
    assert dialog._single_alpha_widget.isHidden()


def test_vector_payload_contains_per_axis_alpha_values(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._vector_alpha_spins["P_x"].setValue(1.11)
    dialog._vector_alpha_spins["P_y"].setValue(1.22)
    dialog._vector_alpha_spins["P_z"].setValue(1.33)

    payload = dialog._current_grouping_payload()

    assert payload["alpha_x"] == pytest.approx(1.11)
    assert payload["alpha_y"] == pytest.approx(1.22)
    assert payload["alpha_z"] == pytest.approx(1.33)
    assert payload["alpha"] == pytest.approx(1.33)


def test_vector_estimate_alpha_for_axis_updates_axis_spin(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("P_x")

    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(2.0)


def test_vector_estimate_alpha_uses_selected_reference_run(qapp: QApplication) -> None:
    ds_a = _vector_dataset_with_ratio(5201, ratio=2.0)
    ds_b = _vector_dataset_with_ratio(5202, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5202)
    dialog._estimate_alpha_for_axis("P_x")

    # Must use selected reference run (5202 => alpha=4), not an average of runs.
    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(4.0)


def test_estimate_alpha_uses_reference_run_only(qapp: QApplication) -> None:
    ds_a = _dataset_with_ratio(5001, ratio=2.0)
    ds_b = _dataset_with_ratio(5002, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5002)
    dialog._estimate_alpha()

    # Must use selected reference run (5002 => alpha=4), not an average of runs.
    assert dialog._alpha_spin.value() == pytest.approx(4.0)


def test_preselected_run_numbers_set_dataset_tickboxes(qapp: QApplication) -> None:
    ds_a = _dataset_with_ratio(5101, ratio=2.0)
    ds_b = _dataset_with_ratio(5102, ratio=3.0)

    dialog = GroupingDialog(
        [ds_a, ds_b],
        selected_run_number=5101,
        selected_run_numbers=[5102],
    )

    assert dialog._checked_run_numbers() == [5102]


def test_pressing_enter_on_bunch_factor_does_not_estimate_alpha(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    assert dialog._alpha_spin.value() == pytest.approx(1.0)

    dialog._bunch_spin.setFocus()
    dialog._bunch_spin.setValue(7)
    QTest.keyClick(dialog._bunch_spin, Qt.Key.Key_Return)

    # Enter on bunching should not trigger the Estimate alpha action.
    assert dialog._alpha_spin.value() == pytest.approx(1.0)


def test_pressing_enter_on_bunch_factor_does_not_trigger_save_grp(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    save_called = {"value": False}

    def _stub_get_save_file_name(*_args, **_kwargs):
        save_called["value"] = True
        return "", ""

    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping_dialog.QFileDialog.getSaveFileName",
        _stub_get_save_file_name,
    )

    dialog._bunch_spin.setFocus()
    dialog._bunch_spin.setValue(9)
    QTest.keyClick(dialog._bunch_spin, Qt.Key.Key_Return)

    assert save_called["value"] is False


def test_grp_round_trip_parser_and_serializer() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.2345,
        "t0_bin": 2,
        "t_good_offset": 7,
        "first_good_bin": 9,
        "last_good_bin": 2048,
        "bunching_factor": 5,
        "deadtime_correction": True,
        "background_correction": True,
        "period_mode": str(PeriodMode.GREEN_PLUS_RED),
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["groups"][1] == [1, 2]
    assert parsed["groups"][2] == [3, 4]
    assert parsed["forward_group"] == 1
    assert parsed["backward_group"] == 2
    assert parsed["alpha"] == pytest.approx(1.2345)
    assert parsed["t0_bin"] == 2
    assert parsed["t_good_offset"] == 7
    assert parsed["first_good_bin"] == 9
    assert parsed["last_good_bin"] == 2048
    assert parsed["bunching_factor"] == 5
    assert parsed["deadtime_correction"] is True
    assert parsed["background_correction"] is True
    assert parsed["period_mode"] == str(PeriodMode.GREEN_PLUS_RED)


def test_grp_round_trip_preserves_deadtime_mode_metadata() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "deadtime_correction": True,
        "deadtime_mode": "manual",
        "deadtime_method": "manual",
        "deadtime_manual_us": 0.025,
        "dead_time_us": [0.025, 0.025],
    }

    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["deadtime_correction"] is True
    assert parsed["deadtime_mode"] == "manual"
    assert parsed["deadtime_method"] == "manual"
    assert parsed["deadtime_manual_us"] == pytest.approx(0.025)
    assert parsed["dead_time_us"] == pytest.approx([0.025, 0.025])


def test_grp_round_trip_parser_and_serializer_with_vector_alphas() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.2345,
        "alpha_x": 1.1,
        "alpha_y": 1.2,
        "alpha_z": 1.3,
        "first_good_bin": 9,
        "last_good_bin": 2048,
        "bunching_factor": 5,
        "deadtime_correction": True,
        "period_mode": str(PeriodMode.GREEN_PLUS_RED),
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["alpha_x"] == pytest.approx(1.1)
    assert parsed["alpha_y"] == pytest.approx(1.2)
    assert parsed["alpha_z"] == pytest.approx(1.3)


def test_parse_grp_legacy_first_good_derives_t_good_offset() -> None:
    text = "\n".join(
        [
            "forward_group=1",
            "backward_group=2",
            "t0_bin=3",
            "first_good_bin=11",
            "last_good_bin=50",
            "group.1=1",
            "group.2=2",
        ]
    )
    parsed = GroupingDialog.parse_grp(text)
    assert parsed["t0_bin"] == 3
    assert parsed["first_good_bin"] == 11
    assert parsed["t_good_offset"] == 8


def test_current_payload_uses_t0_and_t_good_offset(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._t0_spin.setValue(1)
    dialog._t_good_offset_spin.setValue(2)
    dialog._last_good_spin.setValue(3)

    payload = dialog._current_grouping_payload()

    assert payload["t0_bin"] == 1
    assert payload["t_good_offset"] == 2
    assert payload["first_good_bin"] == 3
    assert payload["last_good_bin"] == 3


def test_one_based_bin_index_base_displays_file_facing_t0(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.histograms[0].t0_bin = 1
    dataset.run.grouping["t0_bin"] = 1
    dataset.run.grouping["first_good_bin"] = 3
    dataset.run.grouping["last_good_bin"] = 3
    dataset.run.grouping["bin_index_base"] = 1

    dialog = GroupingDialog([dataset])

    # Display should match one-based file metadata while payload stays internal.
    assert dialog._t0_spin.value() == 2
    payload = dialog._current_grouping_payload()
    assert payload["t0_bin"] == 1
    assert payload["bin_index_base"] == 1


def test_period_mode_row_visible_for_two_period_reference(qapp: QApplication) -> None:
    dialog = GroupingDialog([_two_period_dataset()])
    assert not dialog._period_mode_label.isHidden()
    assert not dialog._period_mode_widget.isHidden()
    assert dialog._current_period_mode() == str(PeriodMode.GREEN)


# ---------------------------------------------------------------------------
# group_names in payload
# ---------------------------------------------------------------------------


def test_current_grouping_payload_contains_group_names_key(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    payload = dialog._current_grouping_payload()
    assert "group_names" in payload


def test_current_grouping_payload_group_names_reflects_state(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._group_names = {1: "Forward", 2: "Backward"}
    payload = dialog._current_grouping_payload()
    assert payload["group_names"] == {1: "Forward", 2: "Backward"}


def test_current_grouping_payload_contains_grouping_preset(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._grouping_preset_name = "Longitudinal"
    payload = dialog._current_grouping_payload()
    assert payload["grouping_preset"] == "Longitudinal"


def test_current_grouping_payload_contains_instrument(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "MuSR"

    dialog = GroupingDialog([dataset])
    payload = dialog._current_grouping_payload()

    assert payload["instrument"] == "MuSR"


def test_psi_detector_convention_defaults_are_swapped_in_grouping_dropdowns(
    qapp: QApplication,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 1
    dataset.run.grouping["backward_group"] = 2

    dialog = GroupingDialog([dataset])

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


def test_psi_already_swapped_grouping_dropdowns_are_preserved(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 2
    dataset.run.grouping["backward_group"] = 1

    dialog = GroupingDialog([dataset])

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


def test_detector_layout_prefers_saved_instrument(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "MuSR"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping_dialog.detect_instrument",
        lambda *_args, **_kwargs: "EMU",
    )
    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping_dialog.get_instrument_layout",
        lambda name: type("_Layout", (), {"name": name})(),
    )
    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "MuSR"


def test_detector_layout_detects_flame_from_psi_metadata(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "FLAME"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "FLAME"


def test_detector_layout_retries_detection_when_saved_instrument_is_generic_psi(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "PSI"
    dataset.run.grouping["histogram_labels"] = [
        "Forw",
        "Back",
        "Righ",
        "Left",
        "R_F",
        "R_B",
        "L_F",
        "L_B",
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "PSI"
    dataset.run.histograms = dataset.run.histograms * 4

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 0

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert captured["instrument"] == "FLAME"


def test_psi_detector_layout_result_is_swapped_for_analysis_dropdowns(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "FLAME"
    dataset.run.grouping["instrument"] = "FLAME"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    dataset.run.grouping["forward_group"] = 1
    dataset.run.grouping["backward_group"] = 2

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return {
                "groups": {1: [1], 2: [2]},
                "group_names": {1: "Forward", 2: "Backward"},
                "forward_group": 1,
                "backward_group": 2,
                "instrument": "FLAME",
                "grouping_preset": "Longitudinal",
            }

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    assert dialog._forward_combo.currentData() == 2
    assert dialog._backward_combo.currentData() == 1


# ---------------------------------------------------------------------------
# .grp format: group_name round-trip
# ---------------------------------------------------------------------------


def test_grp_round_trip_with_group_names() -> None:
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "included_groups": {1: True, 2: False},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 2048,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "group_names": {1: "Forward", 2: "Backward"},
    }
    text = GroupingDialog.serialize_grp(payload)
    assert "group_name.1=Forward" in text
    assert "group_name.2=Backward" in text
    assert "group_include.1=1" in text
    assert "group_include.2=0" in text
    parsed = GroupingDialog.parse_grp(text)
    assert parsed.get("group_names", {}).get(1) == "Forward"
    assert parsed.get("group_names", {}).get(2) == "Backward"
    assert parsed.get("included_groups", {}).get(1) is True
    assert parsed.get("included_groups", {}).get(2) is False


def test_grp_round_trip_without_group_names_backwards_compat() -> None:
    """Old .grp files without group_name lines must still parse cleanly."""
    payload = {
        "groups": {1: [1, 2], 2: [3, 4]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 2048,
        "bunching_factor": 1,
        "deadtime_correction": False,
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)
    # Should produce empty dict, not raise
    assert parsed.get("group_names", {}) == {}


def test_serialize_grp_no_group_names_no_spurious_lines() -> None:
    """serialize_grp with empty group_names must not emit group_name lines."""
    payload = {
        "groups": {1: [1]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "first_good_bin": 0,
        "last_good_bin": 10,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "group_names": {},
    }
    text = GroupingDialog.serialize_grp(payload)
    assert "group_name" not in text


# ---------------------------------------------------------------------------
# "Detector Layout…" button exists in the GroupingDialog UI
# ---------------------------------------------------------------------------


def test_detector_layout_button_exists(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    # The button is connected to _on_detector_layout; verify it is present.
    from PySide6.QtWidgets import QPushButton

    buttons = dialog.findChildren(QPushButton)
    labels = [b.text() for b in buttons]
    assert any("Detector Layout" in lbl for lbl in labels)


def test_group_table_uses_scrollable_capped_height(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["groups"] = {idx: [idx] for idx in range(1, 11)}
    dialog = GroupingDialog([dataset])

    assert dialog._group_table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert dialog._group_table.maximumHeight() > 0
