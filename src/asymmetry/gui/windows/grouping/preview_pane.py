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
    effective_group_indices,
    reduce_grouped_asymmetry,
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


@dataclass(frozen=True)
class _PreviewResult:
    """Plain-array reduction result marshalled back to the GUI thread."""

    generation: int
    time: np.ndarray
    asymmetry: np.ndarray
    error: np.ndarray
    run_number: int | None


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
    ) -> None:
        """Queue a (debounced) recompute of the preview for the current draft.

        *grouping* is the draft resolved against the preview run — i.e. exactly
        the ``run.grouping`` shape :func:`reduce_grouped_asymmetry` consumes. When
        the dataset has no histograms (co-added curves) the pane hides itself with
        a note; nothing is scheduled.
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
            )
        )

    def request_preview_from_profile(
        self,
        *,
        profile: GroupingProfile,
        run: Run | None,
        facility: str = "",
        run_number: int | None = None,
    ) -> None:
        """Queue a (debounced) resolve + recompute for an unresolved draft.

        Unlike :meth:`request_preview`, resolution against the run —
        :func:`resolve_effective_grouping`, which may scan every detector for an
        ``auto_detect`` t0 policy or sum whole groups for a per-run alpha
        estimate — happens on the worker thread. *profile* is deep-copied here
        so subsequent form edits cannot race the in-flight worker; *run* is
        shared read-only.
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
            )
            self._axes.axhline(0.0, color=tokens.TEXT_MUTED, linewidth=0.5, alpha=0.5)
        self._axes.set_xlabel("Time (µs)", fontsize=8)
        self._axes.set_ylabel("Asymmetry (%)", fontsize=8)
        self._axes.tick_params(labelsize=7)
        self._canvas.draw_idle()
        run_label = f"run {result.run_number}" if result.run_number is not None else "preview run"
        self._status.setText(f"Preview: {run_label}")

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
    deadtime_mode = str(grouping.get("deadtime_mode", "off")).strip().lower() or "off"
    reduction = reduce_grouped_asymmetry(
        histograms=request.histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        alpha=_as_float(grouping.get("alpha"), 1.0),
        use_deadtime=bool(grouping.get("deadtime_correction", False)),
        deadtime_mode=deadtime_mode,
        use_background=bool(grouping.get("background_correction", False)),
        facility=request.facility or str(grouping.get("instrument", "") or ""),
        reference_resolver=None,
    )
    # Decimate here, off the GUI thread: bounds both the marshalled payload and
    # the GUI-thread errorbar draw (which is O(points) and the real hang on large
    # runs — see _MAX_PREVIEW_POINTS).
    time, asymmetry, error = _decimate_for_preview(
        reduction.time, reduction.asymmetry, reduction.error, _MAX_PREVIEW_POINTS
    )
    return _PreviewResult(
        generation=request.generation,
        time=time,
        asymmetry=asymmetry,
        error=error,
        run_number=request.run_number,
    )


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
