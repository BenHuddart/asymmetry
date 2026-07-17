"""Live forward/backward asymmetry preview for the grouping editor.

The grouping dialog edits an in-memory *draft* grouping profile; without a
preview, the effect of a grouping / alpha / binning / exclusion / deadtime /
background change is only visible after Apply. :class:`GroupingPreviewPane`
closes that loop: it reduces the *preview run* under the *current draft* and
draws the resulting asymmetry curve, updating (debounced) as the user edits.

Threading contract (see ``AGENTS.md`` and ``gui/tasks.py``): the reduction — the
only expensive step — runs on a :class:`~asymmetry.gui.tasks.TaskRunner` worker
thread and never on the GUI thread. Edits are debounced with a ~300 ms timer;
each computation carries a generation counter so a result from a superseded edit
is dropped on arrival. Results cross back as plain numpy arrays through the
TaskRunner's GUI-thread relay (never a bare lambda touching widgets). The runner
is shut down in :meth:`shutdown`, called from the dialog's ``closeEvent``.

Resolution happens on the worker thread too: :meth:`request_preview_from_profile`
takes the *unresolved* draft profile plus the preview run, and the worker calls
:func:`resolve_effective_grouping` before reducing. That call can be expensive —
an ``auto_detect`` t0 policy scans every detector's full histogram and a
``per_run_estimate`` alpha policy sums whole groups — so it must never run on
the GUI thread per edit. The profile is deep-copied at request time so later
form edits cannot race the worker; the run is shared read-only (the dialog does
not mutate it while open). Vector-mode drafts are previewed on their primary
forward/backward pair (the resolved ``forward_group`` / ``backward_group``; for
canonical EMU that is the P_z axis).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.project.profiles import GroupingProfile, resolve_effective_grouping
from asymmetry.core.transform import (
    binned_fb_asymmetry,
    corrected_grouped_counts,
    correction_flags_from_grouping,
    effective_group_indices,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker
from asymmetry.gui.utils.plot_decimation import decimate_for_preview as _decimate_for_preview

#: Debounce window for coalescing rapid edits before a recompute.
_DEBOUNCE_MS = 300

#: Cap on points drawn in the preview. The pane is only a few hundred pixels
#: wide, so plotting the full reduced curve (which can be ~1M points for a
#: long high-resolution run) is wasted work — and matplotlib ``errorbar`` over
#: ~1M points freezes the GUI thread for ~12 s (rendering cannot leave the GUI
#: thread). Uniformly striding down to this many points before drawing keeps
#: the draw ~30 ms; the preview is advisory so exact sampling does not matter.
_MAX_PREVIEW_POINTS = 2000

#: Fixed height of the preview section so it never fights the form for space.
_PANE_HEIGHT = 200


@dataclass(frozen=True)
class _PreviewRequest:
    """An immutable snapshot of what to resolve/reduce, built on the GUI thread.

    Everything here is a plain object, so the worker function can run entirely
    off the GUI thread. Exactly one of two shapes is populated: a pre-resolved
    ``grouping`` payload, or an unresolved ``profile`` + ``run`` pair that the
    worker resolves first (the expensive path — auto t0 / per-run alpha scans).
    Index/alpha/policy extraction from the resolved payload happens in the
    worker for both shapes.
    """

    generation: int
    histograms: list[Histogram]
    facility: str
    run_number: int | None
    grouping: dict[str, Any] | None = None
    profile: GroupingProfile | None = None
    run: Run | None = None
    #: Calibrate view: also draw the α=1 curve (ghosted) behind the draft-α curve
    #: and report the residual baseline ⟨A⟩ over the good window, so a calibrated
    #: α that centres the corrected asymmetry is self-evident in the one preview.
    #: Retained as the legacy switch for the α compare; ``overlay=True`` is
    #: equivalent to ``compare_stage="alpha"``.
    overlay: bool = False
    #: Stage-generic before/after compare (one focused stage at a time). The solid
    #: curve is always the reduction the request describes; the *ghost* removes one
    #: stage: ``"alpha"`` ghosts α=1 (and reports the residual baseline), while
    #: ``"deadtime"``/``"background"`` ghost a *second* corrected pass with that one
    #: stage dropped. ``None`` draws only the solid curve. Preview-only, like the
    #: overrides — it never touches the persisted reduction.
    compare_stage: str | None = None
    #: Diagnostic per-stage view (preview only, never the persisted reduction):
    #: when ``False``, that correction is dropped from *this preview* so its
    #: incremental effect is visible. They can only *subtract* a configured stage,
    #: never add one — ``use_x = flags.use_x and override_use_x`` — so an off
    #: stage stays off. The default ``True`` is a strict no-op.
    override_use_deadtime: bool = True
    override_use_background: bool = True


@dataclass(frozen=True)
class _PreviewResult:
    """Plain-array reduction result marshalled back to the GUI thread."""

    generation: int
    time: np.ndarray
    asymmetry: np.ndarray
    error: np.ndarray
    run_number: int | None
    #: Compare extras (all ``None``/``False`` unless the request asked for a
    #: compare). ``baseline`` is the ghost curve (aligned to ``time``);
    #: ``baseline_label`` names it ("α = 1", "without deadtime", …); the
    #: ``centre_*`` residual baseline is populated only for the α compare.
    baseline: np.ndarray | None = None
    baseline_label: str | None = None
    compare_stage: str | None = None
    centre_mean: float | None = None
    centre_err: float | None = None
    alpha: float | None = None
    overlay: bool = False


class GroupingPreviewPane(QWidget):
    """Embedded canvas + status strip showing the draft's F/B asymmetry.

    The owning dialog calls :meth:`request_preview` (from any of its refresh
    seams) with the resolved effective grouping and the preview run; the pane
    debounces, reduces off-thread, and redraws. Reduction failures show a muted
    message in the status strip — never a popup or a crash.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the canvas, status strip, and debounce/worker plumbing."""
        super().__init__(parent)
        self.setFixedHeight(_PANE_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._status.setWordWrap(True)

        self._tasks = TaskRunner(self)
        self._generation = 0
        self._pending: _PreviewRequest | None = None
        #: Coalescing guard: at most one reduction runs at a time. While a task
        #: is in flight, newer requests wait in ``_pending`` (latest wins) and
        #: are dispatched from the finished/error callback instead of spawning
        #: concurrent worker threads.
        self._in_flight = False

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._dispatch_pending)

        self._figure = None
        self._canvas = None
        self._axes = None
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas = create_canvas(layout="tight")
            self._axes = self._figure.add_subplot(111)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            layout.addWidget(self._canvas, stretch=1)
        except ImportError:
            fallback = QLabel("matplotlib is not installed — preview unavailable.")
            fallback.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
            fallback.setWordWrap(True)
            layout.addWidget(fallback, stretch=1)

        layout.addWidget(self._status)

    # -- public API ------------------------------------------------------

    def request_preview(
        self,
        *,
        histograms: list[Histogram] | None,
        grouping: dict[str, Any],
        facility: str = "",
        run_number: int | None = None,
        overlay: bool = False,
        compare_stage: str | None = None,
        override_use_deadtime: bool = True,
        override_use_background: bool = True,
    ) -> None:
        """Queue a (debounced) recompute of the preview for the current draft.

        *grouping* is the draft resolved against the preview run — i.e. exactly
        the ``run.grouping`` shape the reduction consumes. When the dataset has no
        histograms (co-added curves) the pane hides itself with a note; nothing is
        scheduled. *overlay* additionally draws the α=1 curve and the residual
        baseline (the calibrate view). *compare_stage* is the stage-generic form of
        the same idea (``"deadtime"``/``"background"``/``"alpha"``), overriding
        *overlay*. ``override_use_deadtime`` / ``override_use_background`` drop a
        configured stage from *this preview only* (the diagnostic view); none of
        these touch the persisted reduction.
        """
        if not histograms:
            self._show_unavailable("Preview needs raw detector histograms (none loaded).")
            return
        if self._canvas is None:
            return  # matplotlib missing; fallback label already shown

        self.setVisible(True)
        self._queue_request(
            _PreviewRequest(
                generation=self._next_generation(),
                histograms=list(histograms),
                grouping=dict(grouping),
                facility=str(facility or grouping.get("instrument", "") or ""),
                run_number=run_number,
                overlay=bool(overlay),
                compare_stage=compare_stage,
                override_use_deadtime=bool(override_use_deadtime),
                override_use_background=bool(override_use_background),
            )
        )

    def request_preview_from_profile(
        self,
        *,
        profile: GroupingProfile,
        run: Run | None,
        facility: str = "",
        run_number: int | None = None,
        overlay: bool = False,
        compare_stage: str | None = None,
        override_use_deadtime: bool = True,
        override_use_background: bool = True,
    ) -> None:
        """Queue a (debounced) resolve + recompute for an unresolved draft.

        Unlike :meth:`request_preview`, resolution against the run —
        :func:`resolve_effective_grouping`, which may scan every detector for an
        ``auto_detect`` t0 policy or sum whole groups for a per-run alpha
        estimate — happens on the worker thread. *profile* is deep-copied here
        so subsequent form edits cannot race the in-flight worker; *run* is
        shared read-only. *compare_stage* draws a stage-generic before/after ghost
        (overriding *overlay*). ``override_use_deadtime`` / ``override_use_background``
        drop a configured stage from *this preview only* (the diagnostic view);
        none of these touch the persisted reduction.
        """
        histograms = list(run.histograms) if run is not None and run.histograms else []
        if not histograms:
            self._show_unavailable("Preview needs raw detector histograms (none loaded).")
            return
        if self._canvas is None:
            return  # matplotlib missing; fallback label already shown

        self.setVisible(True)
        self._queue_request(
            _PreviewRequest(
                generation=self._next_generation(),
                histograms=histograms,
                profile=copy.deepcopy(profile),
                run=run,
                facility=str(facility or ""),
                run_number=run_number,
                overlay=bool(overlay),
                compare_stage=compare_stage,
                override_use_deadtime=bool(override_use_deadtime),
                override_use_background=bool(override_use_background),
            )
        )

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _queue_request(self, request: _PreviewRequest) -> None:
        self._pending = request
        self._status.setText("Computing preview…")
        self._debounce.start()

    def flush(self) -> None:
        """Dispatch any pending request immediately (used by tests)."""
        if self._debounce.isActive():
            self._debounce.stop()
            self._dispatch_pending()

    def shutdown(self) -> None:
        """Stop the debounce and tear down the runner (call from closeEvent)."""
        self._debounce.stop()
        self._pending = None
        self._tasks.shutdown()

    # -- dispatch + worker ----------------------------------------------

    def _dispatch_pending(self) -> None:
        if self._pending is None:
            return
        if self._in_flight:
            # Coalesce: keep the latest request pending; the finished/error
            # callback of the running task dispatches it. This bounds the
            # worker-thread count at one no matter how fast edits arrive.
            return
        request = self._pending
        self._pending = None
        self._in_flight = True
        # A ``reference_run`` background needs the loaded-dataset registry the
        # dialog does not own; the preview simply skips that subtraction (the
        # resolver returns None) so the curve still renders.
        self._tasks.start(
            lambda worker: _run_reduction(worker, request),
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _dispatch_next_after_completion(self) -> None:
        self._in_flight = False
        if self._pending is not None and not self._debounce.isActive():
            self._dispatch_pending()

    def _on_finished(self, result: object) -> None:
        self._dispatch_next_after_completion()
        if not isinstance(result, _PreviewResult):
            return
        if result.generation != self._generation:
            return  # superseded by a newer edit
        self._draw(result)

    def _on_error(self, message: str) -> None:
        self._dispatch_next_after_completion()
        self._set_error(f"Preview unavailable: {message}")

    # -- drawing ---------------------------------------------------------

    def _draw(self, result: _PreviewResult) -> None:
        if self._axes is None or self._canvas is None:
            return
        self._axes.clear()
        if result.time.size == 0:
            self._axes.text(
                0.5,
                0.5,
                "No data in the good-bin window.",
                ha="center",
                va="center",
                transform=self._axes.transAxes,
                color=tokens.TEXT_MUTED,
            )
        else:
            # Compare view: ghost the "before" (α=1, or the stage removed) behind
            # the solid "as reduced" curve so the effect of that stage is
            # self-evident. The ghost's label names it ("α = 1" / "without …").
            if result.overlay and result.baseline is not None:
                self._axes.plot(
                    result.time,
                    result.baseline,
                    color=tokens.TEXT_DIM,
                    linewidth=1.0,
                    alpha=0.7,
                    label=result.baseline_label or "before",
                    zorder=2,
                )
            # Solid curve label: the α value for the α compare, "as reduced" when a
            # count-stage removal is ghosted, else unlabelled.
            if result.compare_stage == "alpha" and result.alpha is not None:
                main_label: str | None = f"α = {result.alpha:.4f}"
            elif result.overlay and result.baseline is not None:
                main_label = "as reduced"
            else:
                main_label = None
            self._axes.errorbar(
                result.time,
                result.asymmetry,
                yerr=result.error,
                fmt="o",
                markersize=2.0,
                linewidth=0.0,
                elinewidth=0.5,
                capsize=0.0,
                color=tokens.ACCENT,
                ecolor=tokens.TEXT_MUTED,
                label=main_label,
                zorder=3,
            )
            self._axes.axhline(0.0, color=tokens.TEXT_MUTED, linewidth=0.5, alpha=0.5)
            if result.overlay and result.baseline is not None:
                self._axes.legend(loc="best", fontsize=7, framealpha=0.85)
        self._axes.set_xlabel("Time (µs)", fontsize=8)
        self._axes.set_ylabel("Asymmetry (%)", fontsize=8)
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()
        run_label = f"run {result.run_number}" if result.run_number is not None else "preview run"
        status = f"Preview: {run_label}"
        if result.overlay and result.centre_mean is not None and result.centre_err is not None:
            status += (
                f"  ·  residual baseline ⟨A⟩ = {result.centre_mean:.3f} ± {result.centre_err:.3f} %"
            )
        self._status.setText(status)

    def _set_error(self, message: str) -> None:
        self._status.setText(message)
        if self._axes is not None and self._canvas is not None:
            self._axes.clear()
            self._axes.tick_params(labelsize=7)
            self._canvas.draw_idle()

    def _show_unavailable(self, message: str) -> None:
        self.setVisible(False)
        self._status.setText(message)


def _run_reduction(worker: TaskWorker, request: _PreviewRequest) -> _PreviewResult:
    """Resolve (if needed) and reduce one preview request off the GUI thread.

    Pure numpy work: it touches no widgets and returns plain arrays, so the
    TaskRunner relay can marshal the result back safely. Cooperative cancellation
    is honoured up front (``worker.is_cancelled()``) so a shutdown mid-flight
    stops promptly; a merely *superseded* result is dropped later by generation
    in :meth:`GroupingPreviewPane._on_finished`. Raised errors (including the
    no-detectors case) surface through the pane's error status strip.
    """
    if worker.is_cancelled():
        raise TaskCancelledError
    if request.profile is not None:
        # The expensive step for auto-t0 / per-run-alpha policies; must stay
        # off the GUI thread. The profile is the pane's private deep copy.
        grouping = resolve_effective_grouping(request.profile, request.run)
    else:
        grouping = request.grouping or {}

    n_hist = len(request.histograms)
    forward_gid = _as_int(grouping.get("forward_group"), 1)
    backward_gid = _as_int(grouping.get("backward_group"), 2)
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
    if not forward_idx or not backward_idx:
        raise ValueError("forward/backward groups have no detectors in this run")

    if worker.is_cancelled():
        raise TaskCancelledError
    # Correct once, form the asymmetry per α. corrected_grouped_counts runs the
    # deadtime → grouping → background stages (the expensive part).
    flags = correction_flags_from_grouping(grouping)
    # Diagnostic per-stage view: a toggle can only *drop* a configured stage from
    # this preview (``and``), never add one — so the persisted reduction (which
    # never sees these overrides) is unaffected and an off stage stays off.
    use_deadtime = flags.use_deadtime and request.override_use_deadtime
    use_background = flags.use_background and request.override_use_background
    facility = request.facility or str(grouping.get("instrument", "") or "")

    def _reduce(dt: bool, bg: bool):
        return corrected_grouped_counts(
            histograms=request.histograms,
            grouping=grouping,
            forward_idx=forward_idx,
            backward_idx=backward_idx,
            use_deadtime=dt,
            deadtime_mode=flags.deadtime_mode,
            use_background=bg,
            facility=facility,
            reference_resolver=None,
        )

    corrected = _reduce(use_deadtime, use_background)
    n_grouped = min(len(corrected.forward), len(corrected.backward))
    try:
        first_good = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first_good = 0
    try:
        last_good = int(grouping.get("last_good_bin", n_grouped - 1))
    except (TypeError, ValueError):
        last_good = n_grouped - 1
    alpha = _as_float(grouping.get("alpha"), 1.0)

    time, asymmetry, error = _form_asymmetry(corrected, grouping, alpha, first_good, last_good)

    # Stage-generic before/after compare: the solid curve is the reduction above;
    # the *ghost* removes one stage. "alpha" ghosts α=1 from the SAME corrected
    # counts (one reduction, two curves) and reports the residual baseline;
    # "deadtime"/"background" ghost a SECOND corrected pass with that one stage
    # dropped — CorrectedGroupedCounts keeps only post-background arrays, so the
    # pedestal cannot be added back and the extra pass is unavoidable, but it runs
    # behind the pane's debounce + single-flight. All preview-only (`_reduce` reads
    # the same grouping; nothing here touches the persisted reduction). An
    # un-applied stage has nothing to remove, so it draws no ghost.
    compare = request.compare_stage or ("alpha" if request.overlay else None)
    baseline = None
    baseline_label: str | None = None
    centre_mean: float | None = None
    centre_err: float | None = None
    if compare == "alpha":
        # Residual baseline (inverse-variance weighted ⟨A⟩) on the full-res curve.
        centre_mean, centre_err = _weighted_centre(asymmetry, error)
        if abs(alpha - 1.0) > 1e-12:
            _bt, base_asym, _be = _form_asymmetry(corrected, grouping, 1.0, first_good, last_good)
            _dt, baseline, _de = _decimate_for_preview(time, base_asym, error, _MAX_PREVIEW_POINTS)
            baseline_label = "α = 1"
    elif compare == "deadtime" and use_deadtime:
        # The ghost is a second full reduction — honour cancellation before it, as
        # the first pass does, so a shutdown mid-flight stops promptly on big runs.
        if worker.is_cancelled():
            raise TaskCancelledError
        ghost = _form_asymmetry(
            _reduce(False, use_background), grouping, alpha, first_good, last_good
        )
        _dt, baseline, _de = _decimate_for_preview(time, ghost[1], error, _MAX_PREVIEW_POINTS)
        baseline_label = "without deadtime"
    elif compare == "background" and use_background:
        if worker.is_cancelled():
            raise TaskCancelledError
        ghost = _form_asymmetry(
            _reduce(use_deadtime, False), grouping, alpha, first_good, last_good
        )
        _dt, baseline, _de = _decimate_for_preview(time, ghost[1], error, _MAX_PREVIEW_POINTS)
        baseline_label = "without background"

    # Decimate here, off the GUI thread: bounds both the marshalled payload and
    # the GUI-thread errorbar draw (which is O(points) and the real hang on large
    # runs — see _MAX_PREVIEW_POINTS).
    time, asymmetry, error = _decimate_for_preview(time, asymmetry, error, _MAX_PREVIEW_POINTS)
    return _PreviewResult(
        generation=request.generation,
        time=time,
        asymmetry=asymmetry,
        error=error,
        run_number=request.run_number,
        baseline=baseline,
        baseline_label=baseline_label,
        compare_stage=compare,
        centre_mean=centre_mean,
        centre_err=centre_err,
        alpha=alpha,
        overlay=compare is not None,
    )


def _form_asymmetry(
    corrected: Any,
    grouping: dict[str, Any],
    alpha: float,
    first_good: int,
    last_good: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin the corrected counts into a percent asymmetry for one α.

    Mirrors :func:`reduce_grouped_asymmetry`'s final step (counts → binned
    asymmetry, scaled to percent) so the preview matches the reduction exactly.
    """
    time, asym, err = binned_fb_asymmetry(
        corrected.forward,
        corrected.backward,
        grouping=grouping,
        common_t0=corrected.common_t0,
        bin_width_us=corrected.bin_width,
        alpha=alpha,
        first_good_bin=first_good,
        last_good_bin=last_good,
        forward_error=corrected.forward_error,
        backward_error=corrected.backward_error,
    )
    return (
        np.asarray(time, dtype=np.float64),
        np.asarray(asym, dtype=np.float64) * 100.0,
        np.asarray(err, dtype=np.float64) * 100.0,
    )


def _weighted_centre(asymmetry: np.ndarray, error: np.ndarray) -> tuple[float | None, float | None]:
    """Inverse-variance weighted mean of the asymmetry and its error.

    This is the residual baseline the calibrate view reports: for a weak-TF
    calibration run a balanced α drives it to zero, so it is the honest numeric
    replacement for eyeballing whether the oscillation sits on zero.
    """
    a = np.asarray(asymmetry, dtype=np.float64)
    e = np.asarray(error, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(e) & (e > 0.0)
    if not mask.any():
        return None, None
    weights = 1.0 / np.square(e[mask])
    total = float(np.sum(weights))
    if total <= 0.0:
        return None, None
    mean = float(np.sum(a[mask] * weights) / total)
    return mean, float(np.sqrt(1.0 / total))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["GroupingPreviewPane"]
