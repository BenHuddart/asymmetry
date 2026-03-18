"""Tests for grouping dialog alpha-estimation workflow."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._deadtime_checkbox.setChecked(True)
    dialog._bunch_spin.setValue(1234)
    result = dialog.get_grouping_result()
    assert result is not None
    assert "forward_indices" in result
    assert "backward_indices" in result
    assert "alpha" in result
    assert result["deadtime_correction"] is True
    assert result["bunching_factor"] == 1234


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
        "first_good_bin": 9,
        "last_good_bin": 2048,
        "bunching_factor": 5,
        "deadtime_correction": True,
        "period_mode": str(PeriodMode.GREEN_PLUS_RED),
    }
    text = GroupingDialog.serialize_grp(payload)
    parsed = GroupingDialog.parse_grp(text)

    assert parsed["groups"][1] == [1, 2]
    assert parsed["groups"][2] == [3, 4]
    assert parsed["forward_group"] == 1
    assert parsed["backward_group"] == 2
    assert parsed["alpha"] == pytest.approx(1.2345)
    assert parsed["first_good_bin"] == 9
    assert parsed["last_good_bin"] == 2048
    assert parsed["bunching_factor"] == 5
    assert parsed["deadtime_correction"] is True
    assert parsed["period_mode"] == str(PeriodMode.GREEN_PLUS_RED)


def test_period_mode_row_visible_for_two_period_reference(qapp: QApplication) -> None:
    dialog = GroupingDialog([_two_period_dataset()])
    assert not dialog._period_mode_label.isHidden()
    assert not dialog._period_mode_widget.isHidden()
    assert dialog._current_period_mode() == str(PeriodMode.GREEN)
