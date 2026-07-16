"""Tests for the dedicated Alpha calibration dialog.

The Estimate action groups the full forward/backward histograms and runs
``estimate_alpha_detailed`` on a :class:`~asymmetry.gui.tasks.TaskRunner`
worker thread (see ``alpha_calibration_dialog.py``'s module docstring and
``gui/tasks.py``). Tests that call :meth:`AlphaCalibrationDialog._on_estimate`
must pump a real event loop until the worker lands — ``_wait_until`` below,
the same idiom ``test_grouping_preview_pane.py`` uses — before reading
``dialog._estimate``. Skipping that pump risks two failure modes: the assert
races the still-in-flight worker (flaky failure), or, worse, the dialog goes
out of scope while its worker thread is still running, which aborts the
process (``gui/tasks.py``'s no-GC-of-a-live-QThread invariant).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.project.profiles import AlphaPolicy
from asymmetry.gui.windows.grouping import (
    alpha_calibration_dialog as alpha_calibration_dialog_module,
)
from asymmetry.gui.windows.grouping.alpha_calibration_dialog import AlphaCalibrationDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_until(predicate, timeout_ms: int = 30_000) -> None:
    """Pump a real nested event loop until *predicate* holds (queued signals)."""
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
    assert predicate(), "timed out waiting for the alpha estimate"


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
    _wait_until(lambda: dialog._tasks.active_count == 0)
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
    _wait_until(lambda: dialog._tasks.active_count == 0)
    assert dialog._estimate is not None
    other = dialog._run_combo.findData(2)
    dialog._run_combo.setCurrentIndex(other)
    assert dialog._estimate is None  # a run change clears the stale estimate


def test_changing_method_invalidates_estimate(qapp: QApplication) -> None:
    dialog = _make_dialog([_run(1, ratio=2.0)])
    dialog._set_method("ratio")
    dialog._on_estimate()
    _wait_until(lambda: dialog._tasks.active_count == 0)
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
    _wait_until(lambda: dialog._tasks.active_count == 0)
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
    _wait_until(lambda: dialog._tasks.active_count == 0)
    assert dialog._estimate is not None
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
    _wait_until(lambda: dialog._tasks.active_count == 0)
    labels_after = {str(line.get_label()) for line in dialog._axes.get_lines()}
    assert any("before" in lbl for lbl in labels_after)
    assert any("after" in lbl for lbl in labels_after)


# ---------------------------------------------------------------------------
# Off-thread estimate (B4: TaskRunner-backed Estimate action)
# ---------------------------------------------------------------------------


def test_estimate_runs_off_gui_thread_and_toggles_button(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate runs on a worker thread; the button is busy for its duration."""
    call_threads: list[int] = []
    release = threading.Event()
    real_estimate = alpha_calibration_dialog_module.estimate_alpha_detailed

    def spy(*args, **kwargs):
        call_threads.append(threading.get_ident())
        release.wait(timeout=5.0)  # held open until the test has checked the button
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(alpha_calibration_dialog_module, "estimate_alpha_detailed", spy)

    dialog = _make_dialog([_run(5, ratio=2.0)])
    dialog._set_method("ratio")
    gui_thread = threading.get_ident()

    assert dialog._estimate_btn.isEnabled()
    dialog._on_estimate()
    _wait_until(lambda: len(call_threads) == 1)
    assert call_threads[0] != gui_thread, "estimate_alpha_detailed ran on the GUI thread"
    assert not dialog._estimate_btn.isEnabled(), "button should be disabled mid-flight"

    release.set()
    _wait_until(lambda: dialog._tasks.active_count == 0)
    assert dialog._estimate_btn.isEnabled(), "button should re-enable once finished"
    assert dialog._estimate is not None
    assert dialog._estimate.alpha == pytest.approx(2.0)


def test_mid_flight_close_does_not_crash(qapp: QApplication) -> None:
    """Closing the dialog while an estimate is in flight cancels cleanly."""
    dialog = _make_dialog([_run(5, ratio=2.0)])
    dialog._set_method("ratio")
    dialog._on_estimate()
    # Close immediately — the worker may still be running (or, for this tiny
    # synthetic dataset, may have already finished). Either way this must not
    # crash: ``done()`` shuts the TaskRunner down before the dialog is dropped.
    dialog.reject()
    for _ in range(20):
        qapp.processEvents()
    assert dialog._tasks.active_count == 0
    assert dialog.result_policy() is None


def _run_with_pedestal(run_number: int, *, ratio: float, bg: float) -> MuonDataset:
    """Two-histogram run with a flat pedestal: F = ratio·base + bg, B = base + bg."""
    base = np.array([100.0, 80.0, 60.0, 40.0])
    forward = ratio * base + bg
    backward = base + bg
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=forward, bin_width=0.01),
            Histogram(counts=backward, bin_width=0.01),
        ],
        metadata={"run_number": run_number},
        grouping={
            "groups": {1: [1], 2: [2]},
            "forward_group": 1,
            "backward_group": 2,
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


def test_estimate_uses_corrected_counts_when_provider_supplies_background(
    qapp: QApplication,
) -> None:
    """With a background-supplying provider, α recovers the true ratio, not the raw one."""
    ds = _run_with_pedestal(5, ratio=2.0, bg=50.0)

    def provider(_dataset):
        return {
            "background_correction": True,
            "background_mode": "fixed",
            "background_fixed_values": [50.0, 50.0],
        }

    dialog = _make_dialog([ds], correction_provider=provider)
    dialog._set_method("ratio")
    dialog._on_estimate()
    _wait_until(lambda: dialog._tasks.active_count == 0)
    assert dialog._estimate is not None
    # Raw ratio would be Σ(2b+50)/Σ(b+50) = 760/480 ≈ 1.583; subtraction gives 2.0.
    assert dialog._estimate.alpha == pytest.approx(2.0)
    assert "background" in dialog._correction_note.text().lower()


def test_correction_note_warns_when_reference_unresolved(qapp: QApplication) -> None:
    """A requested reference_run background that cannot resolve must be flagged."""
    ds = _run_with_pedestal(5, ratio=2.0, bg=50.0)

    def provider(_dataset):
        return {
            "background_correction": True,
            "background_mode": "reference_run",
            "background_run": {"run_number": 999},
        }

    # No resolver → the reference cannot be resolved → subtraction cannot happen.
    dialog = _make_dialog([ds], correction_provider=provider, reference_resolver=None)
    note = dialog._correction_note.text().lower()
    assert "not applied" in note
    assert "background" in note


def test_parent_destruction_mid_estimate_does_not_crash(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Destroying the dialog's *parent* mid-estimate must not abort the process.

    In production the dialog is launched as a *child* of the grouping dialog, so
    real teardown (and the ``tests/conftest.py`` cleanup fixture) reaps it
    through its parent's destruction — never via ``done()``/``closeEvent``. With
    the worker gated mid-flight, that destruction would drop the dialog's live
    estimate ``QThread`` while it runs, which qFatal-aborts the process. The
    ``TaskRunner`` destroyed-safety-net must park it in the reaper instead. This
    is the exact race a previous B4 attempt introduced.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QWidget

    from asymmetry.gui import tasks as tasks_mod

    release = threading.Event()
    entered = threading.Event()
    real_estimate = alpha_calibration_dialog_module.estimate_alpha_detailed

    def gated(*args, **kwargs):
        entered.set()
        release.wait(timeout=5.0)  # hold the worker in-flight
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(alpha_calibration_dialog_module, "estimate_alpha_detailed", gated)

    parent = QWidget()
    dialog = _make_dialog([_run(5, ratio=2.0)], parent=parent)
    dialog._set_method("ratio")
    dialog._on_estimate()
    _wait_until(lambda: entered.is_set())  # worker is inside the gate -> thread running

    reaper_before = tasks_mod._orphan_reaper
    parked_before = len(reaper_before._threads) if reaper_before is not None else 0

    # Destroy the PARENT without close()/done() on the child. This reaps the
    # child dialog and its TaskRunner through C++ destruction alone.
    del dialog
    parent.deleteLater()
    del parent
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()

    reaper = tasks_mod._orphan_reaper
    assert reaper is not None
    # The still-running estimate thread was parked, not destroyed with the dialog.
    assert len(reaper._threads) == parked_before + 1

    # Release the gate; the reaper prunes the finished thread back to baseline.
    release.set()
    _wait_until(lambda: len(reaper._threads) == parked_before)
