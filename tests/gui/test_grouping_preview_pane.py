"""Live asymmetry preview pane for the grouping editor.

Covers the pane's behaviour end-to-end: it populates for a synthetic run, an
edit (and specifically an alpha change) recomputes and visibly changes the curve
data, the reduction error path is muted rather than crashing, and a
histogram-less dataset hides the pane with a note. Also pins that the core
reduction the pane shares with MainWindow (:func:`reduce_grouped_asymmetry`) is
bit-identical to a from-scratch replay of the documented counts-then-ratio
pipeline, so the extraction never forks the numerics.

The TaskRunner-based flow is driven deterministically the same way
``test_gui_tasks.py`` does: a real nested event loop is pumped until the
background reduction lands (``QTest.qWait`` / a ``QEventLoop`` guarded by a
timeout, both fine offscreen). The pane's ``flush()`` bypasses the 300 ms
debounce so tests do not wait on wall-clock.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.project.profiles import GroupingProfile, ProfileFingerprint
from asymmetry.core.transform import (
    apply_grouped_background_correction,
    apply_grouping_aligned,
    binned_fb_asymmetry,
    common_t0_for_groups,
    prepare_histograms_with_deadtime,
    reduce_grouped_asymmetry,
)
from asymmetry.gui.windows.grouping import preview_pane as preview_pane_module
from asymmetry.gui.windows.grouping.dialog import GroupingDialog
from asymmetry.gui.windows.grouping.preview_pane import (
    _MAX_PREVIEW_POINTS,
    GroupingPreviewPane,
    _decimate_for_preview,
)


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
    assert predicate(), "timed out waiting for the preview reduction"


def _histogram_dataset(
    *,
    run_number: int = 5001,
    forward: np.ndarray | None = None,
    backward: np.ndarray | None = None,
    grouping_extra: dict | None = None,
) -> MuonDataset:
    """A run with two raw histograms and a minimal recorded grouping."""
    if forward is None:
        forward = np.array([120.0, 90.0, 70.0, 55.0, 44.0, 36.0], dtype=float)
    if backward is None:
        backward = np.array([80.0, 62.0, 50.0, 40.0, 33.0, 27.0], dtype=float)
    n = forward.size
    grouping = {
        "groups": {1: [1], 2: [2]},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "instrument": "TESTINST",
        "t0_bin": 0,
        "first_good_bin": 0,
        "last_good_bin": n - 1,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if grouping_extra:
        grouping.update(grouping_extra)
    run = Run(
        run_number=run_number,
        histograms=[
            Histogram(counts=forward.copy(), bin_width=0.016, t0_bin=0),
            Histogram(counts=backward.copy(), bin_width=0.016, t0_bin=0),
        ],
        metadata={"run_number": run_number, "instrument": "TESTINST"},
        grouping=grouping,
    )
    t = np.arange(n, dtype=float) * 0.016
    return MuonDataset(
        time=t,
        asymmetry=np.zeros_like(t),
        error=np.full_like(t, 0.01),
        metadata={"run_number": run_number, "instrument": "TESTINST"},
        run=run,
    )


# --------------------------------------------------------------------------- #
# Standalone pane behaviour
# --------------------------------------------------------------------------- #


def _last_curve(pane: GroupingPreviewPane) -> np.ndarray:
    """Return the y-data of the errorbar the pane last drew."""
    axes = pane._axes
    assert axes is not None
    lines = axes.get_lines()
    assert lines, "expected a plotted curve"
    return np.asarray(lines[0].get_ydata(), dtype=float)


def test_pane_populates_for_synthetic_run(qapp: QApplication) -> None:
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    pane.request_preview(
        histograms=dataset.run.histograms,
        grouping=dataset.run.grouping,
        run_number=int(dataset.run_number),
    )
    pane.flush()
    _wait_until(lambda: pane._tasks.active_count == 0 and bool(pane._axes.get_lines()))
    assert pane.isVisible()
    assert "Preview: run 5001" in pane._status.text()
    curve = _last_curve(pane)
    assert curve.size > 0 and np.all(np.isfinite(curve))
    pane.shutdown()


def test_alpha_change_visibly_changes_curve(qapp: QApplication) -> None:
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()

    grouping_a = dict(dataset.run.grouping)
    grouping_a["alpha"] = 1.0
    pane.request_preview(histograms=dataset.run.histograms, grouping=grouping_a, run_number=5001)
    pane.flush()
    _wait_until(lambda: pane._tasks.active_count == 0 and bool(pane._axes.get_lines()))
    curve_alpha_1 = _last_curve(pane).copy()

    grouping_b = dict(dataset.run.grouping)
    grouping_b["alpha"] = 2.5
    pane.request_preview(histograms=dataset.run.histograms, grouping=grouping_b, run_number=5001)
    pane.flush()
    _wait_until(
        lambda: pane._tasks.active_count == 0 and not np.allclose(_last_curve(pane), curve_alpha_1)
    )
    curve_alpha_25 = _last_curve(pane)
    assert not np.allclose(curve_alpha_1, curve_alpha_25)
    pane.shutdown()


def test_rapid_edits_coalesce_to_one_inflight_reduction(qapp: QApplication) -> None:
    """Rapid requests never spawn concurrent workers: latest-pending wins.

    Copilot review (PR #174): each edit used to start its own TaskRunner
    thread; only the *result* was generation-gated. The pane now keeps at most
    one reduction in flight and dispatches only the newest pending request
    when it completes.
    """
    import threading

    from asymmetry.gui.windows.grouping import preview_pane as pane_module

    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()

    lock = threading.Lock()
    concurrency = {"now": 0, "max": 0, "calls": 0}
    original_reduction = pane_module._run_reduction

    def tracking_reduction(worker, request):
        with lock:
            concurrency["now"] += 1
            concurrency["calls"] += 1
            concurrency["max"] = max(concurrency["max"], concurrency["now"])
        try:
            return original_reduction(worker, request)
        finally:
            with lock:
                concurrency["now"] -= 1

    pane_module._run_reduction = tracking_reduction
    try:
        for alpha in (1.0, 1.5, 2.0, 2.5, 3.0):
            grouping = dict(dataset.run.grouping)
            grouping["alpha"] = alpha
            pane.request_preview(
                histograms=dataset.run.histograms, grouping=grouping, run_number=5001
            )
            pane.flush()
        _wait_until(lambda: pane._tasks.active_count == 0 and pane._pending is None)
        _wait_until(lambda: bool(pane._axes.get_lines()))
    finally:
        pane_module._run_reduction = original_reduction
    # Reductions never overlapped, and intermediate requests coalesced (at
    # most first + latest-pending per completion — never one thread per edit).
    assert concurrency["max"] == 1
    assert concurrency["calls"] <= 5
    assert "Preview: run 5001" in pane._status.text()
    pane.shutdown()


def test_error_path_is_muted_not_crashing(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    # Force a genuine reduction failure on the worker thread and assert it is
    # surfaced as a muted status message — never a popup or a crash.
    import asymmetry.gui.windows.grouping.preview_pane as pane_module

    def _boom(**_kwargs):
        raise RuntimeError("synthetic reduction failure")

    monkeypatch.setattr(pane_module, "reduce_grouped_asymmetry", _boom)
    pane.request_preview(
        histograms=dataset.run.histograms,
        grouping=dataset.run.grouping,
        run_number=5001,
    )
    pane.flush()
    _wait_until(lambda: pane._tasks.active_count == 0 and "unavailable" in pane._status.text())
    assert "unavailable" in pane._status.text().lower()
    assert "synthetic reduction failure" in pane._status.text()
    pane.shutdown()


def test_histogramless_dataset_hides_pane(qapp: QApplication) -> None:
    pane = GroupingPreviewPane()
    pane.request_preview(histograms=None, grouping={}, run_number=5002)
    assert not pane.isVisible()
    assert "histogram" in pane._status.text().lower()
    pane.shutdown()


# --------------------------------------------------------------------------- #
# Dialog integration
# --------------------------------------------------------------------------- #


def test_dialog_preview_populates_and_recomputes_on_edit(qapp: QApplication) -> None:
    dataset = _histogram_dataset()
    dialog = GroupingDialog([dataset])
    pane = dialog._preview_pane
    pane.flush()
    _wait_until(lambda: pane._tasks.active_count == 0 and bool(pane._axes.get_lines()))
    first = _last_curve(pane).copy()

    # Editing alpha drives a debounced recompute; flush + wait for a new curve.
    dialog._alpha_spin.setValue(3.0)
    pane.flush()
    _wait_until(lambda: pane._tasks.active_count == 0 and not np.allclose(_last_curve(pane), first))
    assert not np.allclose(_last_curve(pane), first)

    dialog._clear_dirty()
    dialog.close()


def test_dialog_hides_preview_for_histogramless_dataset(qapp: QApplication) -> None:
    # A dataset with a run but no histograms cannot be previewed. The grouping
    # dialog drops run-less datasets, so use a run whose histograms are empty.
    dataset = _histogram_dataset()
    dataset.run.histograms = []
    # The dialog needs at least one usable dataset; pair the empty one with a
    # normal run so the dialog opens, then point the preview at the empty run.
    good = _histogram_dataset(run_number=5003)
    dialog = GroupingDialog([good])
    dialog._preview_pane.request_preview(histograms=None, grouping={}, run_number=5001)
    assert not dialog._preview_pane.isVisible()
    dialog._clear_dirty()
    dialog.close()


# --------------------------------------------------------------------------- #
# Core-reduction pin: MainWindow reduction unchanged by the extraction
# --------------------------------------------------------------------------- #


def _replay_reduction(
    histograms,
    grouping,
    forward_idx,
    backward_idx,
    alpha,
    *,
    use_deadtime,
    use_background,
    facility,
):
    """From-scratch replay of the documented counts-then-ratio pipeline."""
    working = list(histograms)
    if use_deadtime:
        working, _ = prepare_histograms_with_deadtime(histograms, grouping, True)
    common_t0 = common_t0_for_groups(working, forward_idx, backward_idx)
    forward = apply_grouping_aligned(working, forward_idx, common_t0_bin=common_t0)
    backward = apply_grouping_aligned(working, backward_idx, common_t0_bin=common_t0)
    n = min(len(forward), len(backward))
    forward, backward = forward[:n], backward[:n]
    f_err = b_err = None
    if use_background:
        bw = float(working[0].bin_width)
        last_good = int(grouping.get("last_good_bin", n - 1))
        res = apply_grouped_background_correction(
            forward,
            backward,
            grouping=grouping,
            t0_bin=common_t0,
            bin_width_us=bw,
            facility=facility,
            last_good_bin=last_good,
        )
        forward, backward = res.forward, res.backward
        if res.applied and res.forward_error is not None and res.backward_error is not None:
            f_err, b_err = res.forward_error, res.backward_error
    bw = float(working[0].bin_width)
    first_good = max(0, int(grouping.get("first_good_bin", 0)))
    last_good = int(grouping.get("last_good_bin", n - 1))
    t, a, e = binned_fb_asymmetry(
        forward,
        backward,
        grouping=grouping,
        common_t0=common_t0,
        bin_width_us=bw,
        alpha=alpha,
        first_good_bin=first_good,
        last_good_bin=last_good,
        forward_error=f_err,
        backward_error=b_err,
    )
    return t, a * 100.0, e * 100.0


@pytest.mark.parametrize("bunch", [1, 2])
@pytest.mark.parametrize("alpha", [1.0, 1.7])
def test_reduce_grouped_asymmetry_matches_replay(bunch: float, alpha: float) -> None:
    dataset = _histogram_dataset(grouping_extra={"bunching_factor": bunch})
    grouping = dataset.run.grouping
    forward_idx = [0]
    backward_idx = [1]
    result = reduce_grouped_asymmetry(
        histograms=dataset.run.histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        alpha=alpha,
        use_deadtime=False,
        deadtime_mode="off",
        use_background=False,
        facility="TESTINST",
    )
    exp_t, exp_a, exp_e = _replay_reduction(
        dataset.run.histograms,
        grouping,
        forward_idx,
        backward_idx,
        alpha,
        use_deadtime=False,
        use_background=False,
        facility="TESTINST",
    )
    np.testing.assert_array_equal(result.time, exp_t)
    np.testing.assert_array_equal(result.asymmetry, exp_a)
    np.testing.assert_array_equal(result.error, exp_e)
    assert result.deadtime_applied is False


def test_reduce_grouped_asymmetry_pins_mainwindow_delegation(qapp: QApplication) -> None:
    """The MainWindow method now delegates; pin that it returns core's arrays."""
    from PySide6.QtCore import QSettings

    import asymmetry.gui.mainwindow as mw_module
    from asymmetry.gui.mainwindow import MainWindow

    settings = QSettings()
    settings.setValue(mw_module._UI_SCALE_SETTINGS_KEY, 1.0)
    window = MainWindow()
    try:
        dataset = _histogram_dataset()
        grouping = dataset.run.grouping
        mw_time, mw_asym, mw_err, mw_dt, _bkg = window._reduce_grouped_histograms_to_asymmetry(
            histograms=dataset.run.histograms,
            grouping=grouping,
            dataset=dataset,
            run=dataset.run,
            forward_idx=[0],
            backward_idx=[1],
            alpha=1.3,
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
        )
        core = reduce_grouped_asymmetry(
            histograms=dataset.run.histograms,
            grouping=grouping,
            forward_idx=[0],
            backward_idx=[1],
            alpha=1.3,
            use_deadtime=False,
            deadtime_mode="off",
            use_background=False,
            facility="TESTINST",
        )
        np.testing.assert_array_equal(mw_time, core.time)
        np.testing.assert_array_equal(mw_asym, core.asymmetry)
        np.testing.assert_array_equal(mw_err, core.error)
        assert mw_dt == core.deadtime_applied
    finally:
        window.close()


# --------------------------------------------------------------------------- #
# Profile-mode requests: resolution stays off the GUI thread (audit finding #1)
# --------------------------------------------------------------------------- #


def _draft_profile(*, groups: dict[int, list[int]] | None = None) -> GroupingProfile:
    """A minimal draft matching :func:`_histogram_dataset`'s two-detector run."""
    return GroupingProfile(
        name="draft",
        fingerprint=ProfileFingerprint("TESTINST", 2),
        groups={1: [1], 2: [2]} if groups is None else groups,
        forward_group=1,
        backward_group=2,
    )


def test_profile_request_resolves_on_worker_thread_only(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """request_preview_from_profile never resolves synchronously on the GUI thread.

    resolve_effective_grouping can scan every detector's full histogram (auto-t0)
    or sum whole groups (per-run alpha); the pane must defer it to the worker.
    """
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    try:
        gui_thread = threading.get_ident()
        call_threads: list[int] = []
        real_resolve = preview_pane_module.resolve_effective_grouping

        def spy(profile: GroupingProfile, run: Run) -> dict:
            call_threads.append(threading.get_ident())
            return real_resolve(profile, run)

        monkeypatch.setattr(preview_pane_module, "resolve_effective_grouping", spy)
        pane.request_preview_from_profile(
            profile=_draft_profile(), run=dataset.run, run_number=5001
        )
        assert call_threads == [], "resolve ran synchronously during the request call"
        pane.flush()
        _wait_until(lambda: len(call_threads) == 1)
        assert call_threads[0] != gui_thread, "resolve ran on the GUI thread"
        _wait_until(lambda: pane._status.text().startswith("Preview: run"))
        assert _last_curve(pane).size > 0
    finally:
        pane.shutdown()


def test_profile_is_snapshotted_against_later_edits(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pane deep-copies the draft so form edits cannot race the worker."""
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    try:
        seen_profiles: list[GroupingProfile] = []
        real_resolve = preview_pane_module.resolve_effective_grouping

        def spy(profile: GroupingProfile, run: Run) -> dict:
            seen_profiles.append(profile)
            return real_resolve(profile, run)

        monkeypatch.setattr(preview_pane_module, "resolve_effective_grouping", spy)
        draft = _draft_profile()
        pane.request_preview_from_profile(profile=draft, run=dataset.run, run_number=5001)
        # Simulate the user editing the form while the request is pending.
        draft.groups[1] = [2]
        draft.forward_group = 99
        pane.flush()
        _wait_until(lambda: len(seen_profiles) == 1)
        assert seen_profiles[0] is not draft
        assert seen_profiles[0].groups == {1: [1], 2: [2]}
        assert seen_profiles[0].forward_group == 1
    finally:
        pane.shutdown()


def test_profile_request_burst_coalesces_resolves(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keystroke-burst of profile requests coalesces to at most two resolves."""
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    try:
        call_count = 0
        real_resolve = preview_pane_module.resolve_effective_grouping

        def spy(profile: GroupingProfile, run: Run) -> dict:
            nonlocal call_count
            call_count += 1
            return real_resolve(profile, run)

        monkeypatch.setattr(preview_pane_module, "resolve_effective_grouping", spy)
        for _ in range(10):
            pane.request_preview_from_profile(
                profile=_draft_profile(), run=dataset.run, run_number=5001
            )
        pane.flush()
        _wait_until(lambda: pane._status.text().startswith("Preview: run"))
        # Latest-wins pending slot + single-flight dispatch bound the resolves.
        assert call_count <= 2, f"expected coalescing, saw {call_count} resolves"
    finally:
        pane.shutdown()


def test_profile_with_empty_groups_surfaces_muted_error(qapp: QApplication) -> None:
    """A draft whose F/B groups resolve to no detectors errors via the status strip."""
    dataset = _histogram_dataset()
    pane = GroupingPreviewPane()
    try:
        pane.request_preview_from_profile(
            profile=_draft_profile(groups={1: [], 2: []}),
            run=dataset.run,
            run_number=5001,
        )
        pane.flush()
        _wait_until(lambda: pane._status.text().startswith("Preview unavailable"))
        assert "no detectors" in pane._status.text()
    finally:
        pane.shutdown()


# --------------------------------------------------------------------------- #
# Preview decimation: bounds the GUI-thread errorbar draw on huge runs
# --------------------------------------------------------------------------- #


def test_decimate_for_preview_passes_through_small_curves() -> None:
    """A curve at/below the cap is returned unchanged (no copy needed)."""
    n = _MAX_PREVIEW_POINTS
    time = np.arange(n, dtype=float)
    asymmetry = np.linspace(0.0, 1.0, n)
    error = np.full(n, 0.01)
    out_t, out_a, out_e = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    assert out_t is time
    assert out_a is asymmetry
    assert out_e is error


def test_decimate_for_preview_strides_large_curve_to_cap() -> None:
    """A ~1M-point curve (the pathological case that froze the GUI) is bounded."""
    n = 1_000_000
    time = np.arange(n, dtype=float)
    asymmetry = np.sin(time)
    error = np.full(n, 0.01)
    out_t, out_a, out_e = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    assert out_t.size <= _MAX_PREVIEW_POINTS
    assert out_a.size == out_t.size
    assert out_e.size == out_t.size
    # Uniform stride: the decimated time values stay monotonically increasing
    # and are a subsequence of the original.
    assert np.all(np.diff(out_t) > 0)
    assert np.isin(out_t, time).all()


def test_decimate_for_preview_nonpositive_cap_returns_input_unchanged() -> None:
    time = np.arange(10, dtype=float)
    asymmetry = np.zeros(10)
    error = np.zeros(10)
    out_t, out_a, out_e = _decimate_for_preview(time, asymmetry, error, 0)
    assert out_t is time
    assert out_a is asymmetry
    assert out_e is error


def test_decimate_for_preview_handles_empty_arrays() -> None:
    time = np.array([], dtype=float)
    asymmetry = np.array([], dtype=float)
    error = np.array([], dtype=float)
    out_t, out_a, out_e = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    assert out_t.size == 0
    assert out_a.size == 0
    assert out_e.size == 0


def test_run_reduction_result_is_bounded_for_large_curve(qapp: QApplication) -> None:
    """End-to-end: a preview request over a huge run yields a bounded curve.

    Builds a run with 1M-bin histograms (the size that produced the 12 s
    GUI-thread freeze) and asserts the marshalled ``_PreviewResult`` — and thus
    the arrays ``_draw`` hands to matplotlib — never exceeds the preview cap.
    """
    n = 1_000_000
    rng = np.random.default_rng(0)
    forward = 100.0 + rng.normal(size=n)
    backward = 80.0 + rng.normal(size=n)
    dataset = _histogram_dataset(forward=forward, backward=backward)
    pane = GroupingPreviewPane()
    try:
        pane.request_preview(
            histograms=dataset.run.histograms,
            grouping=dataset.run.grouping,
            run_number=int(dataset.run_number),
        )
        pane.flush()
        _wait_until(lambda: pane._tasks.active_count == 0 and bool(pane._axes.get_lines()))
        curve = _last_curve(pane)
        assert curve.size <= _MAX_PREVIEW_POINTS
    finally:
        pane.shutdown()
