"""Tests for grouping dialog alpha-estimation workflow."""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel

import asymmetry.gui.windows.grouping.dialog as grouping_dialog_dialog_module
import asymmetry.gui.windows.grouping_dialog as grouping_dialog_module
from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.windows.grouping.alpha_calibration_dialog import AlphaCalibrationDialog
from asymmetry.gui.windows.grouping_dialog import GroupingDialog


def _wait_until(predicate, timeout_ms: int = 30_000) -> None:
    """Pump a real nested event loop until *predicate* holds (queued signals).

    Both the alpha estimate (B4) and the background-configure preview grouping
    run on a ``TaskRunner`` worker thread now, so tests that trigger them must
    pump the event loop for the finished/error callback to land instead of
    reading state right after the triggering call returns.
    """
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(timeout_ms)
    loop.exec()
    check.stop()
    guard.stop()
    assert predicate(), "timed out waiting for the background worker"


def _autocalibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the launched Alpha calibration dialog run headlessly.

    The grouping dialog's ``Calibrate…`` (and the per-projection estimate
    buttons) now open :class:`AlphaCalibrationDialog` modally. Tests that used to
    call the old synchronous ``_estimate_alpha`` drive the *real* calibration
    flow instead: this patches ``exec`` to run the dialog's own estimate + accept
    path (so the estimator, provenance and returned policy are all exercised) and
    never blocks. ``result_policy()`` then returns the genuine calibrated policy.

    ``_on_estimate`` now runs the estimate on a TaskRunner worker thread (B4),
    so this pumps the event loop until it lands before reading ``_estimate``.

    This dialog is launched with ``parent=self`` (the GroupingDialog), so it is
    a *child* widget, not top-level — the ``tests/conftest.py`` teardown fixture
    only ``close()``s top-level widgets, and it is not reachable via
    ``QApplication.findChildren(QThread)`` either (its GroupingDialog parent is
    itself typically parentless in these tests). The only thing that shuts its
    ``_tasks`` runner down is going through ``done()`` here, exactly as a real
    modal ``exec()`` would on close — so both branches below must call
    ``reject()``/``accept()`` rather than just returning a result code.
    """

    def _fake_exec(self: AlphaCalibrationDialog) -> int:
        self._on_estimate()
        _wait_until(lambda: self._tasks.active_count == 0)
        if self._estimate is None:
            # Estimate failed (e.g. the General method on flat data): behave like
            # a user who cancels, so the caller leaves alpha untouched. reject()
            # (not a bare return) so done() -> _tasks.shutdown() still runs.
            self.reject()
            return QDialog.DialogCode.Rejected
        self._on_accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(AlphaCalibrationDialog, "exec", _fake_exec, raising=True)


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


def test_estimate_alpha_updates_spinbox(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
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
    dialog._set_deadtime_mode("file")
    dialog._background_mode = "range"
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
    assert result["background_mode"] == "range"
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
    """Deadtime state defaults to off; setting a mode does not require the
    (now dedicated-dialog) file/manual/estimate widgets to exist inline."""
    dialog = GroupingDialog([_dataset_with_histograms()])
    payload = dialog._current_grouping_payload()

    assert dialog._deadtime_mode == "off"
    assert payload["deadtime_correction"] is False
    dialog._set_deadtime_mode("file")
    assert dialog._current_deadtime_mode() == "file"
    dialog._set_deadtime_mode("manual")
    assert dialog._current_deadtime_mode() == "manual"
    dialog._set_deadtime_mode("estimate")
    assert dialog._current_deadtime_mode() == "estimate"


def test_manual_deadtime_payload_resolves_uniform_values(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    dialog._set_deadtime_mode("manual")
    dialog._deadtime_manual_values_us = [0.025, 0.025]
    dialog._deadtime_manual_method = "manual"

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_correction"] is True
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "manual"
    assert payload["dead_time_us"] == pytest.approx([0.025, 0.025])


def test_file_deadtime_status_reflects_reference_run(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["dead_time_us"] = [0.011, 0.022]
    dialog = GroupingDialog([dataset])

    dialog._set_deadtime_mode("file")
    dialog._update_deadtime_status()

    assert dialog._reference_file_deadtime_values() == pytest.approx([0.011, 0.022])
    assert dialog._deadtime_status_label.text() == "Deadtime: from file"


def test_estimate_deadtime_uses_reference_run_only(qapp: QApplication) -> None:
    reference = _dataset_with_deadtime_profile(4101, 0.02)
    other = _dataset_with_deadtime_profile(4102, 0.04)
    dialog = GroupingDialog(
        [reference, other],
        selected_run_number=4101,
        selected_run_numbers=[4101, 4102],
    )

    dialog._set_deadtime_mode("estimate")

    payload = dialog.get_grouping_result()

    assert payload is not None
    assert payload["deadtime_mode"] == "estimate"
    assert payload["deadtime_method"] == "estimate"
    assert payload["deadtime_reference_run"] == 4101
    assert payload["run_numbers"] == [4101, 4102]
    assert payload["dead_time_us"] == pytest.approx([0.02, 0.02, 0.02, 0.02], rel=1e-2, abs=5e-4)


def test_calibrate_deadtime_populates_explicit_table(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    monkeypatch.setattr(
        grouping_dialog_dialog_module,
        "calibrate_deadtime_from_histograms",
        lambda *args, **kwargs: [0.011, 0.022],
    )

    dialog._set_deadtime_mode("manual")
    dialog._calibrate_deadtime_from_reference()
    payload = dialog.get_grouping_result()

    assert payload is not None
    assert dialog._current_deadtime_mode() == "manual"
    assert payload["deadtime_mode"] == "manual"
    assert payload["deadtime_method"] == "calibrate"
    assert payload["dead_time_us"] == pytest.approx([0.011, 0.022])
    assert payload["deadtime_reference_run"] == 4001


def test_background_range_mode_disabled_for_non_psi_data(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])

    available = dialog._available_background_modes()
    assert "range" not in available
    # Tail fit and background-run modes stay available on pulsed data.
    assert "tail_fit" in available
    assert "reference_run" in available
    assert dialog._current_grouping_payload()["background_correction"] is False
    assert dialog._current_grouping_payload()["background_mode"] == "none"


def _tf_dataset_with_histograms(run_number: int = 4020) -> MuonDataset:
    """A dataset grouped with a transverse-field dual-grouping (non-canonical)
    preset declaring two projections — the MuSR ``Transverse (Vector)`` shape."""
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "TF Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
            "group_names": {
                1: "Top-Bottom Top",
                2: "Top-Bottom Bottom",
                3: "Fwd-Back Forward",
                4: "Fwd-Back Backward",
            },
            "projections": [
                {"label": "Top-Bottom", "forward_group": 1, "backward_group": 2},
                {"label": "Fwd-Back", "forward_group": 3, "backward_group": 4},
            ],
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


def _wep_dataset_with_histograms(run_number: int = 4030) -> MuonDataset:
    """A non-canonical preset whose projections declare their own alpha — the
    GPS ``WEP`` shape (FB alpha 0.75 / UD alpha 1.0)."""
    h1 = Histogram(counts=np.array([100.0, 100.0, 100.0, 100.0]), bin_width=0.01)
    h2 = Histogram(counts=np.array([50.0, 50.0, 50.0, 50.0]), bin_width=0.01)
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "title": "WEP Grouping Test"},
        grouping={
            "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
            "group_names": {
                1: "FB Forward",
                2: "FB Backward",
                3: "UD Up",
                4: "UD Down",
            },
            "projections": [
                {"label": "FB", "forward_group": 1, "backward_group": 2, "alpha": 0.75},
                {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.0},
            ],
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


def test_vector_mode_shows_per_axis_alpha_controls(qapp: QApplication) -> None:
    dialog = GroupingDialog([_vector_dataset_with_histograms()])

    assert dialog._vector_axis_pairs
    assert not dialog._vector_alpha_widget.isHidden()
    assert dialog._single_alpha_widget.isHidden()


def test_transverse_field_preset_shows_per_projection_alpha_controls(
    qapp: QApplication,
) -> None:
    """A non-canonical (transverse-field) projection set shows the generalized
    per-projection alpha table — one row per declared projection — instead of
    the single-alpha control, so users can recalibrate each projection's alpha."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])

    # The TF projections are detected (they drive the plot chip bar) ...
    assert set(dialog._vector_axis_pairs) == {"Top-Bottom", "Fwd-Back"}
    # ... and each gets a row in the per-projection alpha table.
    assert set(dialog._vector_alpha_spins) == {"Top-Bottom", "Fwd-Back"}
    assert not dialog._vector_alpha_widget.isHidden()
    assert dialog._single_alpha_widget.isHidden()
    # No canonical P_x/P_y/P_z rows leak into a non-canonical table.
    assert "P_z" not in dialog._vector_alpha_spins


def test_transverse_field_payload_persists_per_projection_alpha(
    qapp: QApplication,
) -> None:
    """Editing a non-canonical projection's α writes it into that projection's
    ``alpha`` in the projections payload, while the base alpha stays the single
    spin and no canonical per-axis alpha keys leak."""
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._alpha_spin.setValue(1.42)
    dialog._vector_alpha_spins["Top-Bottom"].setValue(0.83)
    dialog._vector_alpha_spins["Fwd-Back"].setValue(1.17)

    payload = dialog._current_grouping_payload()

    # Base alpha is the single spin (the projections' fallback), not a stale
    # canonical P_z spin.
    assert payload["alpha"] == pytest.approx(1.42)
    # No canonical per-axis alpha keys leak into a non-canonical grouping.
    assert "alpha_x" not in payload
    assert "alpha_z" not in payload
    # The edited per-projection alphas land inside the projections payload.
    by_label = {p["label"]: p for p in payload["projections"]}
    assert set(by_label) == {"Top-Bottom", "Fwd-Back"}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(0.83)
    assert by_label["Fwd-Back"]["alpha"] == pytest.approx(1.17)


def test_per_projection_alpha_seeds_from_declared_projection_alpha(
    qapp: QApplication,
) -> None:
    """Each non-canonical row seeds from its projection's declared alpha (e.g.
    GPS WEP's FB = 0.75 / UD = 1.0), so the table opens on the real values."""
    dialog = GroupingDialog([_wep_dataset_with_histograms()])

    assert set(dialog._vector_alpha_spins) == {"FB", "UD"}
    assert dialog._vector_alpha_spins["FB"].value() == pytest.approx(0.75)
    assert dialog._vector_alpha_spins["UD"].value() == pytest.approx(1.0)


def test_per_projection_alpha_estimate_updates_and_persists(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estimating a non-canonical projection updates its spin and the estimated
    value persists into the projection payload."""
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("Top-Bottom")
    estimated = dialog._vector_alpha_spins["Top-Bottom"].value()

    payload = dialog._current_grouping_payload()
    by_label = {p["label"]: p for p in payload["projections"]}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(estimated)


def test_per_projection_alpha_estimate_persists_provenance(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-canonical projection's estimate carries its reference-run provenance
    into the projection payload, and a manual edit invalidates it — mirroring the
    canonical per-axis provenance."""
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_tf_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("Top-Bottom")
    estimated = dialog._vector_alpha_spins["Top-Bottom"].value()

    payload = dialog._current_grouping_payload()
    by_label = {p["label"]: p for p in payload["projections"]}
    assert by_label["Top-Bottom"]["alpha"] == pytest.approx(estimated)
    assert "alpha_reference_run" in by_label["Top-Bottom"]
    # Fwd-Back was not estimated, so it carries a value but no provenance.
    assert "alpha_reference_run" not in by_label["Fwd-Back"]

    # A manual edit invalidates the provenance for that projection.
    dialog._vector_alpha_spins["Top-Bottom"].setValue(estimated + 0.5)
    payload2 = dialog._current_grouping_payload()
    by_label2 = {p["label"]: p for p in payload2["projections"]}
    assert "alpha_reference_run" not in by_label2["Top-Bottom"]


def test_per_projection_alpha_edit_survives_detector_layout_accept(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unsaved per-projection alpha edit is preserved when the detector-layout
    editor is accepted (its result carries the preset-declared alpha, which must
    not clobber the user's edit for a surviving projection label)."""
    dialog = GroupingDialog([_wep_dataset_with_histograms()])
    dialog._vector_alpha_spins["FB"].setValue(0.83)  # unsaved table edit

    result = {
        "groups": {1: [1], 2: [2], 3: [1], 4: [2]},
        "group_names": {1: "FB Forward", 2: "FB Backward", 3: "UD Up", 4: "UD Down"},
        "forward_group": 1,
        "backward_group": 2,
        "excluded_detectors": [],
        # The editor returns the preset-declared alphas (FB = 0.75), not the edit.
        "projections": [
            {"label": "FB", "forward_group": 1, "backward_group": 2, "alpha": 0.75},
            {"label": "UD", "forward_group": 3, "backward_group": 4, "alpha": 1.0},
        ],
        "grouping_preset": "WEP",
        "instrument": "GPS",
    }

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return result

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog._on_detector_layout()

    by_label = {p["label"]: p for p in dialog._projection_specs}
    assert by_label["FB"]["alpha"] == pytest.approx(0.83)  # edit preserved
    assert by_label["UD"]["alpha"] == pytest.approx(1.0)  # untouched projection intact
    # The rebuilt table reflects the preserved edit.
    assert dialog._vector_alpha_spins["FB"].value() == pytest.approx(0.83)


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


def test_vector_payload_records_per_axis_alpha_provenance(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After estimating an axis, the payload carries that axis's error and
    reference run (per-axis provenance, skipped in Phase 1)."""
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("P_x")
    estimated = dialog._vector_alpha_spins["P_x"].value()

    payload = dialog._current_grouping_payload()

    assert payload["alpha_x"] == pytest.approx(estimated)
    assert "alpha_x_reference_run" in payload
    # P_y was not estimated, so it carries a value but no provenance.
    assert "alpha_y_reference_run" not in payload
    # A manual edit invalidates the provenance for that axis.
    dialog._vector_alpha_spins["P_x"].setValue(estimated + 0.5)
    payload2 = dialog._current_grouping_payload()
    assert "alpha_x_reference_run" not in payload2


def test_vector_estimate_alpha_for_axis_updates_axis_spin(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_vector_dataset_with_histograms()])
    dialog._estimate_alpha_for_axis("P_x")

    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(2.0)


def test_vector_estimate_alpha_uses_selected_reference_run(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
    ds_a = _vector_dataset_with_ratio(5201, ratio=2.0)
    ds_b = _vector_dataset_with_ratio(5202, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5202)
    dialog._estimate_alpha_for_axis("P_x")

    # Must use selected reference run (5202 => alpha=4), not an average of runs.
    assert dialog._vector_alpha_spins["P_x"].value() == pytest.approx(4.0)


def test_estimate_alpha_uses_reference_run_only(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
    ds_a = _dataset_with_ratio(5001, ratio=2.0)
    ds_b = _dataset_with_ratio(5002, ratio=4.0)

    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=5002)
    dialog._estimate_alpha()

    # Must use selected reference run (5002 => alpha=4), not an average of runs.
    assert dialog._alpha_spin.value() == pytest.approx(4.0)


def test_selected_run_number_seeds_preview_run(qapp: QApplication) -> None:
    """selected_run_number chooses the initial preview run (no broadcast list)."""
    ds_a = _dataset_with_ratio(5101, ratio=2.0)
    ds_b = _dataset_with_ratio(5102, ratio=3.0)

    dialog = GroupingDialog(
        [ds_a, ds_b],
        selected_run_number=5102,
    )

    assert int(dialog._reference_dataset.run_number) == 5102
    # The scope panel lists every run of the fingerprint, inheriting by default.
    assert dialog._scope_panel.inheriting_run_numbers() == {5101, 5102}
    assert dialog._scope_panel.released_run_numbers() == set()


def test_pressing_enter_on_bunch_factor_does_not_estimate_alpha(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    assert dialog._alpha_spin.value() == pytest.approx(1.0)

    dialog._bunch_spin.setFocus()
    dialog._bunch_spin.setValue(7)
    QTest.keyClick(dialog._bunch_spin, Qt.Key.Key_Return)

    # Enter on bunching should not trigger the Estimate alpha action.
    assert dialog._alpha_spin.value() == pytest.approx(1.0)


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


def test_beam_direction_label_recognises_single_letter_and_compound_names() -> None:
    """Defence in depth: the swap classifier reads F/B and compound spin-rotator
    names, not only the spelled-out Forward/Backward group names."""
    fn = GroupingDialog._beam_direction_label
    assert fn("Forward") == "forward"
    assert fn("Backward") == "backward"
    assert fn("F") == "forward"
    assert fn("B") == "backward"
    assert fn("F+D") == "forward"  # compound: leading beam-axis letter wins
    assert fn("B+U") == "backward"
    # Non-beam directions and unrelated names never classify as F/B.
    assert fn("Up") is None
    assert fn("Down") is None
    assert fn("Left") is None
    assert fn("Right") is None
    assert fn("") is None
    # Names that merely START with f/b must not be swept up by the
    # single-letter rule (Copilot review, PR #173).
    assert fn("fit") is None
    assert fn("baseline") is None
    assert fn("F1") is None  # HAL per-detector groups are not beam pairs
    assert fn("B3") is None


def test_applying_fixed_gps_preset_is_not_re_swapped(qapp: QApplication) -> None:
    """Exactly-once: a GPS preset already declares its analysis slots (the
    Backward-named group in the analysis-forward slot), so the PSI beam->analysis
    swap in the dialog must NOT fire again when that preset is applied on a PSI
    reference. Applying the preset's declared pair leaves it unchanged."""
    from asymmetry.core.instrument import get_instrument_layout

    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"

    dialog = GroupingDialog([dataset])
    assert dialog._reference_is_psi()

    layout = get_instrument_layout("GPS")
    lon = layout.presets["Longitudinal"]
    # Mirror the detector-layout editor's group_names into the dialog so the swap
    # classifier sees the physical F/B names the preset carries.
    dialog._group_names = {gid: gdef.name for gid, gdef in lon.groups.items()}

    # The preset already declares analysis-forward = the Backward-named group.
    assert lon.groups[lon.forward_group].name == "Backward"

    # Applying the preset's declared pair (the payload path at _on_apply calls
    # _analysis_pair_for_reference on result["forward_group"]/["backward_group"]).
    result_fwd, result_bwd = lon.forward_group, lon.backward_group
    once_fwd, once_bwd = dialog._analysis_pair_for_reference(result_fwd, result_bwd)
    # No swap: the forward slot does not hold a forward-NAMED group.
    assert (once_fwd, once_bwd) == (result_fwd, result_bwd)

    # Idempotent under a second pass (proves it cannot drift on re-entry).
    twice = dialog._analysis_pair_for_reference(once_fwd, once_bwd)
    assert twice == (once_fwd, once_bwd)


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
        "asymmetry.gui.windows.grouping.dialog.detect_instrument",
        lambda *_args, **_kwargs: "EMU",
    )
    monkeypatch.setattr(
        "asymmetry.gui.windows.grouping.dialog.get_instrument_layout",
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


def test_detector_layout_corrects_gps_variant_to_histogram_count(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    # Stored as the 6-detector BIN variant, but the data is an 11-histogram ROOT
    # run: the layout editor must open on the matching GPS-RD variant, not the
    # stale stored "GPS".
    dataset.run.grouping["instrument"] = "GPS"
    dataset.run.grouping["histogram_labels"] = [
        "Forw",
        "Back",
        "Up_B",
        "Up_F",
        "Down_B",
        "Down_F",
        "Right_B",
        "Right_F",
        "Left_B",
        "Left_F",
        "Mob-RL",
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["instrument"] = "LMU_BULKMUSR_GPS"
    dataset.run.histograms = [dataset.run.histograms[0]] * 11

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

    assert captured["instrument"] == "GPS-RD"


def test_detector_layout_resolves_psi_hifi_to_hal(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The PSI loader stores instrument="HIFI" for HAL-9500 runs; this string
    # canonicalises to the unrelated ISIS HiFi layout, so it must be re-detected
    # to HAL for PSI data. Uses the real detector/layout functions.
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["psi_format"] = "psi-mdu"
    dataset.run.metadata["instrument"] = "HIFI"
    dataset.run.grouping["instrument"] = "HIFI"

    captured: dict[str, str] = {}

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, instrument, **kwargs):
            captured["instrument"] = instrument.name

        def exec(self):
            return 1  # Accepted

        def get_result(self):
            return {
                "instrument": captured["instrument"],
                "groups": {1: [1], 2: [2]},
                "group_names": {},
            }

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    dialog._on_detector_layout()

    # The editor is seeded with the HAL layout (not the unrelated ISIS HiFi),
    assert captured["instrument"] == "HAL"
    # and on Accept the corrected layout name is committed, so it does not revert
    # to the raw "HIFI" string on subsequent opens.
    assert dialog._detector_layout_instrument_name == "HAL"


def test_detector_layout_cancel_does_not_change_stored_instrument(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Cancelling the editor must be a no-op for the stored instrument selection.
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.metadata["facility"] = "PSI"
    dataset.run.metadata["psi_format"] = "psi-mdu"
    dataset.run.metadata["instrument"] = "HIFI"
    dataset.run.grouping["instrument"] = "HIFI"

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 0  # Rejected / cancelled

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog = GroupingDialog([dataset])
    before = dialog._detector_layout_instrument_name
    dialog._on_detector_layout()
    assert dialog._detector_layout_instrument_name == before


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


def test_detector_layout_custom_edit_refreshes_preset_chip_and_marks_dirty(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drifted (custom) result from the detector-layout editor must refresh the
    preset chip and arm the dialog's dirty/close-guard flag immediately, not just
    update the internal state silently (regression: _on_detector_layout updated
    ``_grouping_preset_name``/groups/combos but never called ``_mark_dirty`` or
    ``_refresh_preset_chip``, so the chip kept showing the stale preset name)."""
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "GPS"
    dataset.run.grouping["grouping_preset"] = "Longitudinal"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    # GPS's Longitudinal preset declares forward=2 / backward=1.
    dataset.run.grouping["forward_group"] = 2
    dataset.run.grouping["backward_group"] = 1

    dialog = GroupingDialog([dataset])
    assert dialog._preset_chip.text() == "Preset: Longitudinal"
    assert dialog._draft_dirty is False

    # Drifted from the preset: the layout editor itself detects the state no
    # longer matches any preset and reports grouping_preset=None (see
    # DetectorLayoutDialog._update_preset_status_label).
    result = {
        "groups": {1: [1, 3], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 2,
        "backward_group": 1,
        "instrument": "GPS",
        "grouping_preset": None,
        "excluded_detectors": [],
        "projections": [],
    }

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return result

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog._on_detector_layout()

    assert dialog._grouping_preset_name is None
    assert dialog._preset_chip.text() == "Custom (edited from Longitudinal)"
    assert dialog._draft_dirty is True


def test_detector_layout_unchanged_preset_does_not_clear_chip(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the layout editor reconfirms the same preset (no drift), the chip
    must keep reading "Preset: <name>" rather than being over-cleared to
    Custom."""
    dataset = _dataset_with_histograms()
    assert dataset.run is not None
    dataset.run.grouping["instrument"] = "GPS"
    dataset.run.grouping["grouping_preset"] = "Longitudinal"
    dataset.run.grouping["group_names"] = {1: "Forward", 2: "Backward"}
    # GPS's Longitudinal preset declares forward=2 / backward=1.
    dataset.run.grouping["forward_group"] = 2
    dataset.run.grouping["backward_group"] = 1

    dialog = GroupingDialog([dataset])
    assert dialog._preset_chip.text() == "Preset: Longitudinal"

    result = {
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 2,
        "backward_group": 1,
        "instrument": "GPS",
        "grouping_preset": "Longitudinal",
        "excluded_detectors": [],
        "projections": [],
    }

    class _FakeDialog:
        DialogCode = type("DialogCode", (), {"Accepted": 1})

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return 1

        def get_result(self):
            return result

    monkeypatch.setattr(
        "asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog",
        _FakeDialog,
    )

    dialog._on_detector_layout()

    assert dialog._grouping_preset_name == "Longitudinal"
    assert dialog._preset_chip.text() == "Preset: Longitudinal"


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


# ---------------------------------------------------------------------------
# Alpha estimation method picker + provenance (data-reduction-parity Phase 1)
# ---------------------------------------------------------------------------


def test_alpha_method_combo_defaults_to_diamagnetic(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    assert dialog._current_alpha_method() == "diamagnetic"
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "diamagnetic"


def test_alpha_method_round_trips_through_payload_and_reload(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("ratio")
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "ratio"

    dataset.run.grouping["alpha_method"] = "general"
    dialog2 = GroupingDialog([dataset])
    assert dialog2._current_alpha_method() == "general"


def test_estimate_records_provenance_in_payload(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    assert dialog._alpha_spin.value() == pytest.approx(2.0)
    assert dialog._alpha_result_label.text() != ""
    # The single-alpha provenance status reflects the calibration, not "manual".
    assert dialog._alpha_provenance_label.text() != "manual"
    result = dialog.get_grouping_result()
    assert result["alpha_method"] == "diamagnetic"
    assert result["alpha_reference_run"] == 4001
    # Bootstrap error from flat 100/50 counts is small but present.
    assert result.get("alpha_error") is None or result["alpha_error"] >= 0.0


def test_manual_alpha_edit_invalidates_estimate_provenance(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    _autocalibrate(monkeypatch)
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._estimate_alpha()
    dialog._alpha_spin.setValue(dialog._alpha_spin.value() + 0.5)
    result = dialog.get_grouping_result()
    assert "alpha_error" not in result
    assert "alpha_reference_run" not in result
    assert result["alpha_method"] == "diamagnetic"
    # A hand-edit flips the provenance status to "manual".
    assert dialog._alpha_provenance_label.text() == "manual"


def test_estimate_failure_leaves_alpha(qapp: QApplication, monkeypatch) -> None:
    """A failed estimate inside the calibration dialog leaves the alpha untouched
    (the dialog reports the failure; the caller does not write a value back)."""
    _autocalibrate(monkeypatch)
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("general")  # flat 4-bin data: no relaxation contrast
    before = dialog._alpha_spin.value()
    dialog._estimate_alpha()
    assert dialog._alpha_spin.value() == pytest.approx(before)


def test_format_value_with_uncertainty() -> None:
    fmt = grouping_dialog_module._format_value_with_uncertainty
    assert fmt(1.2345, 0.0067) == "1.2345(67)"
    assert fmt(1.37, None) == "1.3700"
    assert fmt(0.9876, 0.05) == "0.988(50)"
    assert fmt(1.3, 0.0995) == "1.30(10)"


def test_tail_fit_mode_shows_preview_status(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    # Long histograms so the tail fit has a usable window.
    rng = np.random.default_rng(0)
    counts = rng.poisson(np.full(400, 50.0)).astype(float)
    dataset.run.histograms = [
        Histogram(counts=counts, bin_width=0.016),
        Histogram(counts=counts * 0.8, bin_width=0.016),
    ]
    dataset.run.grouping["last_good_bin"] = 399
    dialog = GroupingDialog([dataset])
    dialog._background_mode = "tail_fit"
    dialog._update_background_status()
    assert "Tail-fit background" in dialog._background_status_label.text()
    result = dialog.get_grouping_result()
    assert result["background_mode"] == "tail_fit"
    assert result["background_correction"] is True


def test_background_run_payload_round_trips(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["background_correction"] = True
    dataset.run.grouping["background_mode"] = "reference_run"
    dataset.run.grouping["background_run"] = {"run_number": 9001, "source_file": "/tmp/x.nxs"}
    dialog = GroupingDialog([dataset])
    assert dialog._current_background_mode() == "reference_run"
    dialog._update_background_status()
    result = dialog.get_grouping_result()
    assert result["background_run"]["run_number"] == 9001
    assert "9001" in dialog._background_status_label.text()


# ---------------------------------------------------------------------------
# Background Configure… grouping runs off-thread (B4)
# ---------------------------------------------------------------------------


def test_configure_background_groups_off_thread_and_opens_with_arrays(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forward/backward preview arrays are grouped off the GUI thread, and
    BackgroundDialog opens seeded with exactly those arrays."""
    call_threads: list[int] = []
    real_apply_grouping = grouping_dialog_dialog_module.apply_grouping

    def spy(histograms, indices):
        call_threads.append(threading.get_ident())
        return real_apply_grouping(histograms, indices)

    monkeypatch.setattr(grouping_dialog_dialog_module, "apply_grouping", spy)

    captured: dict[str, object] = {}

    class _FakeBackgroundDialog:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def exec(self) -> int:
            return QDialog.DialogCode.Rejected  # cancel; only the inputs matter here

    monkeypatch.setattr(grouping_dialog_dialog_module, "BackgroundDialog", _FakeBackgroundDialog)

    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    gui_thread = threading.get_ident()

    assert dialog._background_configure_btn.isEnabled()
    dialog._on_configure_background()
    _wait_until(lambda: "preview" in captured)
    assert not any(t == gui_thread for t in call_threads), "apply_grouping ran on the GUI thread"
    assert call_threads  # apply_grouping was actually called (twice: forward + backward)

    _wait_until(lambda: dialog._tasks.active_count == 0)
    assert dialog._background_configure_btn.isEnabled()

    preview = captured["preview"]
    assert preview is not None
    forward_counts, backward_counts, bin_width, _t0_bin, _last_good = preview
    assert forward_counts.tolist() == [100.0, 100.0, 100.0, 100.0]
    assert backward_counts.tolist() == [50.0, 50.0, 50.0, 50.0]
    assert bin_width == pytest.approx(0.01)


def test_configure_background_mid_flight_close_does_not_crash(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing the grouping dialog while the background grouping is still
    running cancels and joins the worker instead of aborting the process."""
    real_apply_grouping = grouping_dialog_dialog_module.apply_grouping
    release = threading.Event()

    def blocked_apply_grouping(histograms, indices):
        release.wait(timeout=5.0)
        return real_apply_grouping(histograms, indices)

    monkeypatch.setattr(grouping_dialog_dialog_module, "apply_grouping", blocked_apply_grouping)

    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._on_configure_background()
    assert dialog._tasks.active_count == 1
    assert not dialog._background_configure_btn.isEnabled()

    # Let the blocked worker proceed shortly after, so the shutdown's bounded
    # wait does not have to wait out its full timeout.
    threading.Timer(0.2, release.set).start()

    # Close mid-flight: done() -> _teardown_workers() -> self._tasks.shutdown()
    # must cancel + join the worker cleanly.
    dialog.reject()

    for _ in range(50):
        qapp.processEvents()
    assert dialog._tasks.active_count == 0


# ---------------------------------------------------------------------------
# Binning modes, Find t0, detector exclusion (data-reduction-parity Phase 3)
# ---------------------------------------------------------------------------


def test_binning_mode_round_trips_through_payload(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    assert dialog._current_binning_mode() == "fixed"
    assert "binning_mode" not in dialog.get_grouping_result()

    dialog._set_binning_mode("variable")
    dialog._bin0_spin.setValue(0.1)
    dialog._bin10_spin.setValue(0.5)
    result = dialog.get_grouping_result()
    assert result["binning_mode"] == "variable"
    assert result["bin0_us"] == pytest.approx(0.1)
    assert result["bin10_us"] == pytest.approx(0.5)
    assert not dialog._bunch_spin.isEnabled()

    dataset.run.grouping.update(result)
    dialog2 = GroupingDialog([dataset])
    assert dialog2._current_binning_mode() == "variable"
    assert dialog2._bin0_spin.value() == pytest.approx(0.1)


def test_constant_error_mode_hides_bin10(qapp: QApplication) -> None:
    dialog = GroupingDialog([_dataset_with_histograms()])
    dialog._set_binning_mode("constant_error")
    result = dialog.get_grouping_result()
    assert result["binning_mode"] == "constant_error"
    assert "bin10_us" not in result
    assert not dialog._bin10_spin.isEnabled()


def test_find_t0_fills_spinner_without_applying(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    rng = np.random.default_rng(0)
    counts = np.full(200, 5.0)
    counts[37] = 5000.0
    counts[38:] = 100.0
    dataset.run.histograms = [
        Histogram(counts=rng.poisson(counts).astype(float), bin_width=0.00125),
        Histogram(counts=rng.poisson(counts).astype(float), bin_width=0.00125),
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.metadata["facility"] = "PSI"
    dataset.run.grouping["last_good_bin"] = 199
    dialog = GroupingDialog([dataset])
    dialog._on_find_t0()
    assert dialog._t0_spin.value() == 37 + dialog._bin_index_base()
    assert "t0" in dialog._alpha_result_label.text()


def test_exclusion_field_round_trips_and_validates(qapp: QApplication, monkeypatch) -> None:
    dataset = _dataset_with_histograms()
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    dataset.run.histograms = [Histogram(counts=np.full(4, 100.0), bin_width=0.01) for _ in range(4)]
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("2")
    result = dialog.get_grouping_result()
    assert result["excluded_detectors"] == [2]

    dialog._exclude_edit.setText("")
    # The key is always present: an empty list explicitly clears exclusions
    # (a missing key would leave the apply path falling back to stale state).
    assert dialog.get_grouping_result()["excluded_detectors"] == []

    warnings: list[str] = []
    monkeypatch.setattr(
        grouping_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dialog._exclude_edit.setText("nonsense")
    assert dialog._current_excluded_detectors() is None
    assert warnings


def test_apply_blocks_when_exclusion_empties_a_group(qapp: QApplication, monkeypatch) -> None:
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("1")  # forward group is exactly detector 1
    warnings: list[str] = []
    monkeypatch.setattr(
        grouping_dialog_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(str(args[2]) if len(args) > 2 else ""),
    )
    dialog._on_apply()
    assert any("no detectors left" in w for w in warnings)


def test_estimate_alpha_respects_detector_exclusion(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix: the estimate is computed on the same detector set the
    reduction will use."""
    _autocalibrate(monkeypatch)
    dataset = _dataset_with_histograms()
    dataset.run.histograms = [
        Histogram(counts=np.full(4, 100.0), bin_width=0.01),  # det 1 (forward)
        Histogram(counts=np.full(4, 900.0), bin_width=0.01),  # det 2 (forward, hot)
        Histogram(counts=np.full(4, 50.0), bin_width=0.01),  # det 3 (backward)
        Histogram(counts=np.full(4, 50.0), bin_width=0.01),  # det 4 (backward)
    ]
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3, 4]}
    dialog = GroupingDialog([dataset])
    dialog._set_alpha_method("ratio")

    dialog._estimate_alpha()
    with_hot = dialog._alpha_spin.value()
    dialog._exclude_edit.setText("2")
    dialog._estimate_alpha()
    without_hot = dialog._alpha_spin.value()

    assert with_hot == pytest.approx(1000.0 / 100.0)
    assert without_hot == pytest.approx(100.0 / 100.0)


def test_find_t0_skips_excluded_detectors(qapp: QApplication) -> None:
    dataset = _dataset_with_histograms()
    good = np.full(200, 5.0)
    good[40] = 5000.0
    bad = np.full(200, 5.0)
    bad[120] = 5000.0  # excluded detector with a bogus peak
    dataset.run.histograms = [
        Histogram(counts=good.copy(), bin_width=0.00125),
        Histogram(counts=bad.copy(), bin_width=0.00125),
        Histogram(counts=good.copy(), bin_width=0.00125),
    ]
    dataset.run.metadata["facility"] = "PSI"
    dataset.metadata["facility"] = "PSI"
    dataset.run.grouping["groups"] = {1: [1, 2], 2: [3]}
    dataset.run.grouping["last_good_bin"] = 199
    dialog = GroupingDialog([dataset])
    dialog._exclude_edit.setText("2")
    dialog._on_find_t0()
    assert dialog._t0_spin.value() == 40 + dialog._bin_index_base()


def test_format_value_with_uncertainty_large_errors() -> None:
    """Review fix: uncertainties >= ~100 must not crash the formatter."""
    fmt = grouping_dialog_module._format_value_with_uncertainty
    assert fmt(1.2345, 150.0) == "1(150)"
    assert fmt(1.2, 99.6) == "1(100)"
    assert fmt(2.4, 1234.0) == "2(1234)"


def _gps_dataset(field_direction: str | None) -> MuonDataset:
    """A six-histogram PSI GPS run defaulting to the Longitudinal grouping."""
    histograms = [Histogram(counts=np.full(4, 100.0), bin_width=0.01) for _ in range(6)]
    metadata = {"run_number": 5001, "facility": "PSI", "instrument": "GPS"}
    if field_direction is not None:
        metadata["field_direction"] = field_direction
    run = Run(
        run_number=5001,
        histograms=histograms,
        metadata=dict(metadata),
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
        metadata=dict(metadata),
        run=run,
    )


def test_transverse_field_gps_run_shows_grouping_nudge(qapp) -> None:
    """B8a: a TF GPS run on the longitudinal default nudges toward spin-rotated."""
    dialog = GroupingDialog([_gps_dataset("Transverse")])
    assert dialog._tf_hint_label.isVisibleTo(dialog)
    assert "Spin-rotated (B+U/F+D)" in dialog._tf_hint_label.text()


def test_gps_run_without_field_direction_does_not_nudge(qapp) -> None:
    dialog = GroupingDialog([_gps_dataset(None)])
    assert not dialog._tf_hint_label.isVisibleTo(dialog)


# ---------------------------------------------------------------------------
# M2 profile-editor semantics
# ---------------------------------------------------------------------------


from asymmetry.core.project.profiles import (  # noqa: E402
    GroupingProfile,
    profile_fingerprint_for_run,
    profile_from_payload,
)


def _profile_for(dataset: MuonDataset, name: str, **overrides) -> GroupingProfile:
    """Build a GroupingProfile from a dataset's payload, with field overrides."""
    fingerprint = profile_fingerprint_for_run(dataset.run)
    payload = dict(dataset.run.grouping or {})
    payload.update(overrides)
    return profile_from_payload(payload, name, fingerprint, active=True)


def test_preview_run_change_preserves_draft_edits(qapp: QApplication) -> None:
    """Changing the preview run must never discard in-progress draft edits."""
    ds_a = _dataset_with_ratio(6201, ratio=2.0)
    ds_b = _dataset_with_ratio(6202, ratio=3.0)
    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=6201)

    dialog._alpha_spin.setValue(1.2345)
    dialog._mark_dirty()
    # Selecting another (inheriting) run in the scope panel is the preview switch;
    # in-progress profile-draft edits must survive it.
    dialog._scope_panel.set_current_run(6202)

    assert dialog._current_grouping_payload()["alpha"] == pytest.approx(1.2345)
    assert int(dialog._reference_dataset.run_number) == 6202


def test_draft_seeds_from_active_profile_when_present(qapp: QApplication) -> None:
    """An active profile for the fingerprint seeds the draft (not the run payload)."""
    dataset = _dataset_with_histograms()
    profile = _profile_for(dataset, "My Profile", alpha=1.75)
    dialog = GroupingDialog([dataset], profiles=[profile])

    assert dialog._draft_name == "My Profile"
    assert dialog._current_grouping_payload()["alpha"] == pytest.approx(1.75)


def test_default_draft_synthesized_without_profile(qapp: QApplication) -> None:
    """A fingerprint with no profile synthesizes a Default (<instrument>) draft."""
    dialog = GroupingDialog([_gps_dataset(None)])
    assert dialog._draft_name.startswith("Default (")


def test_default_draft_name_is_neutral_without_positive_instrument(qapp: QApplication) -> None:
    """A generic 'PSI' instrument token never names a profile after an instrument.

    An unresolved PSI file (loader fallback instrument ``"PSI"``) must get the
    neutral ``Default (<N> detectors)`` name rather than masquerading as a
    specific spectrometer (the "Default (FLAME)" bug).
    """
    dataset = _gps_dataset(None)
    dataset.run.metadata["instrument"] = "PSI"
    dataset.metadata["instrument"] = "PSI"
    dataset.run.grouping.pop("instrument", None)
    dialog = GroupingDialog([dataset])
    assert dialog._draft_name == "Default (6 detectors)"


def test_payload_matches_preset_accepts_detector_t0_pair_entries(qapp: QApplication) -> None:
    """Group entries stored as (detector_id, t0_bin) pairs still match a preset.

    resolve_group_indices() accepts pair entries, so the drift check must
    decode them the same way or a pair-carrying payload looks falsely
    "drifted" (Copilot review, PR #174).
    """
    from asymmetry.core.instrument import get_instrument_layout
    from asymmetry.gui.windows.grouping.profile_bridge import payload_matches_preset

    layout = get_instrument_layout("GPS")
    preset = layout.presets["Longitudinal"]
    payload = {
        "groups": {1: [[1, 100]], 2: [[2, 100]]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": preset.forward_group,
        "backward_group": preset.backward_group,
    }
    assert payload_matches_preset(payload, layout, "Longitudinal")


def test_preset_chip_clears_stale_preset_on_drift(qapp: QApplication) -> None:
    """Editing groups away from a preset clears the stored grouping_preset."""
    dataset = _gps_dataset(None)
    dialog = GroupingDialog([dataset])
    # Apply a named preset via the dropdown.
    dialog._preset_combo.setCurrentIndex(0)
    dialog._on_preset_combo_activated(0)
    assert dialog._grouping_preset_name is not None
    assert dialog._preset_chip.text().startswith("Preset:")

    # Drift the groups so they no longer match the preset.
    dialog._groups = {1: [0], 2: [1, 2, 3, 4, 5]}
    dialog._populate_group_table()
    dialog._refresh_preset_chip(dialog._current_grouping_payload())
    assert dialog._grouping_preset_name is None
    assert "Custom (edited from" in dialog._preset_chip.text()
    # The drifted draft must not carry the stale preset.
    assert not dialog._current_grouping_payload().get("grouping_preset")


def test_release_from_profile_excludes_run_from_apply(qapp: QApplication) -> None:
    """A released run leaves the inheriting set (Apply targets only inheritors)."""
    ds_a = _dataset_with_ratio(6301, ratio=2.0)
    ds_b = _dataset_with_ratio(6302, ratio=3.0)
    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=6301)

    assert dialog._scope_panel.inheriting_run_numbers() == {6301, 6302}
    dialog._scope_panel._released[6302] = True

    result = dialog.get_grouping_result()
    assert set(result["run_numbers"]) == {6301}
    profile_result = dialog.get_profile_result()
    assert profile_result["inheriting"] == {6301}
    assert profile_result["released"] == {6302}
    assert profile_result["newly_released"] == {6302}


def test_reattach_run_returns_it_to_inheriting(qapp: QApplication) -> None:
    """Reattaching an initially-overridden run returns it to the profile."""
    ds_a = _dataset_with_ratio(6401, ratio=2.0)
    ds_b = _dataset_with_ratio(6402, ratio=3.0)
    dialog = GroupingDialog(
        [ds_a, ds_b],
        selected_run_number=6401,
        overridden_run_numbers=[6402],
    )
    assert dialog._scope_panel.released_run_numbers() == {6402}

    dialog._scope_panel._released[6402] = False
    profile_result = dialog.get_profile_result()
    assert profile_result["inheriting"] == {6401, 6402}
    assert profile_result["newly_reattached"] == {6402}


def test_apply_disabled_when_no_run_inherits(qapp: QApplication) -> None:
    """Apply is disabled (with a reason) when every run is released."""
    dataset = _dataset_with_ratio(6501, ratio=2.0)
    dialog = GroupingDialog([dataset])
    assert dialog._apply_btn.isEnabled()

    dialog._scope_panel._released[6501] = True
    dialog._on_scope_changed()
    assert not dialog._apply_btn.isEnabled()
    assert "released" in dialog._apply_btn.toolTip().lower()


def test_unsaved_draft_guard_blocks_reject_on_keep_editing(qapp: QApplication, monkeypatch) -> None:
    """Cancelling a dirty draft prompts; Keep-editing aborts the close."""
    from PySide6.QtWidgets import QMessageBox

    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._alpha_spin.setValue(2.5)
    dialog._mark_dirty()

    answers = iter([QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Discard])
    monkeypatch.setattr(
        grouping_dialog_dialog_module.QMessageBox,
        "question",
        lambda *args, **kwargs: next(answers),
    )
    results: list[int] = []
    monkeypatch.setattr(GroupingDialog, "done", lambda self, code: results.append(code))

    dialog.reject()  # first answer: Cancel -> no close
    assert results == []
    dialog.reject()  # second answer: Discard -> closes
    assert results  # done() was called


def test_apply_marks_draft_saved(qapp: QApplication, monkeypatch) -> None:
    """Applying a dirty draft clears the dirty flag so close does not prompt."""
    dataset = _dataset_with_histograms()
    dialog = GroupingDialog([dataset])
    dialog._alpha_spin.setValue(1.9)
    dialog._mark_dirty()
    monkeypatch.setattr(GroupingDialog, "accept", lambda self: None)

    dialog._on_apply()
    assert dialog._draft_dirty is False


def test_multi_instrument_fingerprint_separation(qapp: QApplication) -> None:
    """The scope panel and preview list only show the current fingerprint."""
    gps = _gps_dataset(None)  # 6 histograms, GPS
    other = _dataset_with_histograms()  # 2 histograms, no instrument
    dialog = GroupingDialog([gps, other], selected_run_number=5001)

    fingerprint_runs = {int(ds.run_number) for ds in dialog._fingerprint_datasets()}
    assert 5001 in fingerprint_runs
    assert 4001 not in fingerprint_runs
    assert dialog._scope_panel.inheriting_run_numbers() == {5001}


# --------------------------------------------------------------------------- #
# t0 mode selector (T0Policy)
# --------------------------------------------------------------------------- #


def _detector_t0_dataset(run_number: int = 4600, detector_t0: tuple[int, int] = (2, 3)):
    """A dataset whose two detectors carry distinct file-derived t0 bins."""
    h1 = Histogram(
        counts=np.array([10.0, 20.0, 100.0, 40.0, 30.0, 20.0]),
        bin_width=0.01,
        t0_bin=detector_t0[0],
    )
    h2 = Histogram(
        counts=np.array([12.0, 22.0, 60.0, 90.0, 30.0, 20.0]),
        bin_width=0.01,
        t0_bin=detector_t0[1],
    )
    run = Run(
        run_number=run_number,
        histograms=[h1, h2],
        metadata={"run_number": run_number, "facility": "PSI"},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": max(detector_t0),
            "first_good_bin": max(detector_t0),
            "last_good_bin": 5,
            "detector_t0_bins": list(detector_t0),
        },
    )
    t = np.arange(6, dtype=float) * 0.01
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "facility": "PSI"},
        run=run,
    )


def test_t0_mode_defaults_to_from_file_with_readonly_spin(qapp: QApplication) -> None:
    dialog = GroupingDialog([_detector_t0_dataset()])
    assert dialog._current_t0_mode() == "from_file"
    assert dialog._t0_spin.isReadOnly() is True
    assert dialog._find_t0_btn.isEnabled() is False
    # Read-only spin shows the preview run's file common t0 (max over groups = 3).
    assert dialog._t0_spin.value() == 3
    assert "each run's file" in dialog._t0_mode_label.text()
    assert dialog._draft.t0_policy.mode == "from_file"


def test_t0_manual_mode_enables_spin_and_dirties_draft(qapp: QApplication) -> None:
    dialog = GroupingDialog([_detector_t0_dataset()])
    dialog._draft_dirty = False
    dialog._set_t0_mode_combo("manual")
    dialog._on_t0_mode_changed()
    assert dialog._t0_spin.isReadOnly() is False
    assert dialog._find_t0_btn.isEnabled() is True

    dialog._draft_dirty = False
    dialog._t0_spin.setValue(5)
    assert dialog._draft_dirty is True
    dialog._sync_draft_from_form()
    assert dialog._draft.t0_policy.mode == "manual"
    assert dialog._draft.t0_policy.value == 5


def test_t0_from_file_shows_file_value_not_stored_override(qapp: QApplication) -> None:
    """From-file display derives from histograms, not a stored/shifted payload t0.

    An overridden run's payload can carry a manual t0 shift; "From file" must
    still show (and on Apply, restore) the genuine file value so the shift can
    be cleared (Copilot review, PR #177).
    """
    ds = _detector_t0_dataset()  # file t0 (max over groups) = 3
    ds.run.grouping["t0_bin"] = 25  # stored manual/override shift
    ds.run.grouping["effective_detector_t0_bins"] = [24, 25]
    dialog = GroupingDialog([ds])
    assert dialog._current_t0_mode() != "from_file" or dialog._t0_spin.value() == 3
    dialog._set_t0_mode_combo("from_file")
    dialog._apply_t0_mode_to_controls()
    assert dialog._t0_spin.value() == 3


def test_t0_from_file_spin_follows_preview_run(qapp: QApplication) -> None:
    """Switching the preview run refreshes the read-only from_file t0."""
    ds_a = _detector_t0_dataset(run_number=4601, detector_t0=(2, 3))  # file t0 = 3
    ds_b = _detector_t0_dataset(run_number=4602, detector_t0=(1, 1))  # file t0 = 1
    dialog = GroupingDialog([ds_a, ds_b], selected_run_number=4601)
    assert dialog._current_t0_mode() == "from_file"
    assert dialog._t0_spin.value() == 3
    dialog._scope_panel.set_current_run(4602)
    assert dialog._t0_spin.value() == 1


def test_t0_auto_detect_shows_detected_value_and_provenance(qapp: QApplication) -> None:
    dialog = GroupingDialog([_detector_t0_dataset()])
    dialog._set_t0_mode_combo("auto_detect")
    dialog._on_t0_mode_changed()
    assert dialog._t0_spin.isReadOnly() is True
    # PSI (continuous) → prompt-peak argmax. h1 peak at bin 2, h2 peak at bin 3.
    # median consensus rounds to 2 or 3; provenance text names the strategy.
    assert dialog._t0_spin.value() in (2, 3)
    assert "prompt peak" in dialog._t0_mode_label.text()
    dialog._sync_draft_from_form()
    assert dialog._draft.t0_policy.mode == "auto_detect"


def test_t0_find_button_fills_manual_spin(qapp: QApplication) -> None:
    dialog = GroupingDialog([_detector_t0_dataset()])
    dialog._set_t0_mode_combo("manual")
    dialog._on_t0_mode_changed()
    dialog._on_find_t0()
    # Find fills the spin with the detected consensus (prompt peak).
    assert dialog._t0_spin.value() in (2, 3)
