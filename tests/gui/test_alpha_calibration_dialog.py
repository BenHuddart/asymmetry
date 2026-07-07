"""Tests for the dedicated Alpha calibration dialog."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.project.profiles import AlphaPolicy
from asymmetry.gui.windows.grouping.alpha_calibration_dialog import AlphaCalibrationDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _run(run_number: int, *, ratio: float, metadata: dict | None = None) -> MuonDataset:
    """Two-histogram run whose forward/backward count ratio is *ratio*."""
    forward = np.full(4, 100.0)
    backward = forward / ratio
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=forward, bin_width=0.01),
            Histogram(counts=backward, bin_width=0.01),
        ],
        metadata={"run_number": run_number, "title": f"Run {run_number}", **(metadata or {})},
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


def _make_dialog(datasets, **kwargs) -> AlphaCalibrationDialog:
    defaults = dict(
        groups={1: [0], 2: [1]},
        group_names={1: "Forward", 2: "Backward"},
        forward_group=1,
        backward_group=2,
    )
    defaults.update(kwargs)
    return AlphaCalibrationDialog(datasets, **defaults)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def test_dropdown_lists_all_runs(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(1, ratio=2.0), _run(2, ratio=3.0), _run(3, ratio=4.0)])
    run_numbers = {dialog._run_combo.itemData(i) for i in range(dialog._run_combo.count())}
    assert run_numbers == {1, 2, 3}


def test_run_summary_includes_title_temperature_and_field(qapp: QApplication) -> None:
    ds = _run(7, ratio=2.0, metadata={"title": "FeSe", "temperature": 5.0, "field": 100.0})
    summary = AlphaCalibrationDialog._run_summary(ds)
    assert "Run 7" in summary
    assert "FeSe" in summary
    assert "5 K" in summary
    assert "100 G" in summary


# ---------------------------------------------------------------------------
# TF highlighting + auto-select
# ---------------------------------------------------------------------------


def test_tf_run_is_highlighted_in_foreground(qapp: QApplication) -> None:
    tf = _run(2, ratio=2.0, metadata={"field_direction": "Transverse", "field": 100.0})
    plain = _run(1, ratio=2.0, metadata={"field_direction": "Longitudinal", "field": 3000.0})
    dialog = _make_dialog([plain, tf])

    tf_index = dialog._run_combo.findData(2)
    plain_index = dialog._run_combo.findData(1)
    tf_brush = dialog._run_combo.itemData(tf_index, Qt.ItemDataRole.ForegroundRole)
    plain_brush = dialog._run_combo.itemData(plain_index, Qt.ItemDataRole.ForegroundRole)
    assert tf_brush is not None  # highlighted
    assert plain_brush is None  # not highlighted


def test_best_tf_candidate_is_auto_selected(qapp: QApplication) -> None:
    plain = _run(1, ratio=2.0, metadata={"field_direction": "Longitudinal", "field": 3000.0})
    tf = _run(2, ratio=2.0, metadata={"field_direction": "Transverse", "field": 100.0})
    dialog = _make_dialog([plain, tf])
    assert dialog._run_combo.currentData() == 2


def test_policy_source_run_overrides_auto_select(qapp: QApplication) -> None:
    plain = _run(1, ratio=2.0)
    tf = _run(2, ratio=2.0, metadata={"field_direction": "Transverse", "field": 100.0})
    dialog = _make_dialog(
        [plain, tf],
        initial_policy=AlphaPolicy(mode="calibrated", value=1.0, source_run=1),
    )
    assert dialog._run_combo.currentData() == 1


def test_selected_run_number_seeds_initial_run(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(1, ratio=2.0), _run(2, ratio=3.0)], selected_run_number=2)
    assert dialog._run_combo.currentData() == 2


# ---------------------------------------------------------------------------
# Estimation wiring
# ---------------------------------------------------------------------------


def test_estimate_populates_result_and_estimate(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(5, ratio=2.0)])
    dialog._set_method("ratio")
    dialog._on_estimate()
    assert dialog._estimate is not None
    assert dialog._estimate.alpha == pytest.approx(2.0)
    assert "α =" in dialog._result_label.text()
    assert "run 5" in dialog._result_label.text()


def test_good_bin_window_seeds_from_run_facts(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(5, ratio=2.0)])
    assert dialog._first_good_spin.value() == 0
    assert dialog._last_good_spin.value() == 3


def test_switching_run_invalidates_estimate(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(1, ratio=2.0), _run(2, ratio=4.0)])
    dialog._on_estimate()
    assert dialog._estimate is not None
    other = dialog._run_combo.findData(2)
    dialog._run_combo.setCurrentIndex(other)
    assert dialog._estimate is None  # a run change clears the stale estimate


def test_changing_method_invalidates_estimate(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(1, ratio=2.0)])
    dialog._set_method("ratio")
    dialog._on_estimate()
    assert dialog._estimate is not None
    dialog._set_method("diamagnetic")
    assert dialog._estimate is None


# ---------------------------------------------------------------------------
# OK / Cancel result contract
# ---------------------------------------------------------------------------


def test_ok_returns_calibrated_policy(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(9, ratio=2.0)])
    dialog._set_method("ratio")
    dialog._on_estimate()
    dialog._on_accept()

    policy = dialog.result_policy()
    assert policy is not None
    assert policy.mode == "calibrated"
    assert policy.value == pytest.approx(2.0)
    assert policy.method == "ratio"
    assert policy.source_run == 9


def test_accept_without_estimate_is_blocked(qapp: QApplication, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.alpha_calibration_dialog.QMessageBox.warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dialog = _make_dialog([_run(9, ratio=2.0)])
    dialog._on_accept()
    assert warnings  # a warning is shown
    assert dialog.result_policy() is None  # nothing accepted


def test_cancel_returns_no_policy(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(9, ratio=2.0)])
    dialog._on_estimate()  # even after a successful estimate...
    dialog.reject()  # ...cancel returns nothing
    assert dialog.result_policy() is None


def test_empty_dataset_list_is_handled(qapp: QApplication) -> None:
    dialog = AlphaCalibrationDialog(
        [],
        groups={1: [0], 2: [1]},
        forward_group=1,
        backward_group=2,
    )
    assert dialog.result_policy() is None


# ---------------------------------------------------------------------------
# Preview curve
# ---------------------------------------------------------------------------


def test_preview_draws_before_and_after_curves(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(5, ratio=2.0)])
    if dialog._axes is None:
        pytest.skip("matplotlib not installed")
    # Before estimate: only the α = 1 (before) curve plus the zero line.
    labels_before = {line.get_label() for line in dialog._axes.get_lines()}
    assert any("before" in str(lbl) for lbl in labels_before)

    dialog._set_method("ratio")
    dialog._on_estimate()
    labels_after = {str(line.get_label()) for line in dialog._axes.get_lines()}
    assert any("before" in lbl for lbl in labels_after)
    assert any("after" in lbl for lbl in labels_after)
