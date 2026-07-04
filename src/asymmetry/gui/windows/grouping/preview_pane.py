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

The pane is deliberately dumb about *how* the draft resolves: the dialog hands it
a fully-resolved effective grouping dict plus the preview run's histograms (both
cheap to build on the GUI thread), and the pane resolves the forward/backward
detector indices and runs :func:`reduce_grouped_asymmetry`. Vector-mode drafts
are previewed on their primary forward/backward pair (the resolved
``forward_group`` / ``backward_group``; for canonical EMU that is the P_z axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform import (
    effective_group_indices,
    reduce_grouped_asymmetry,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker

#: Debounce window for coalescing rapid edits before a recompute.
_DEBOUNCE_MS = 300

#: Fixed height of the preview section so it never fights the form for space.
_PANE_HEIGHT = 200


@dataclass(frozen=True)
class _PreviewRequest:
    """An immutable snapshot of what to reduce, built on the GUI thread.

    Everything here is a plain object (histograms + a resolved grouping dict), so
    the worker function can run entirely off the GUI thread.
    """

    generation: int
    histograms: list[Histogram]
    grouping: dict[str, Any]
    forward_idx: list[int]
    backward_idx: list[int]
    alpha: float
    use_deadtime: bool
    deadtime_mode: str
    use_background: bool
    facility: str
    run_number: int | None


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
        forward_gid = _as_int(grouping.get("forward_group"), 1)
        backward_gid = _as_int(grouping.get("backward_group"), 2)
        n_hist = len(histograms)
        forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n_hist)
        backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n_hist)
        if not forward_idx or not backward_idx:
            self._set_error("Forward/backward groups have no detectors in this run.")
            return

        self._generation += 1
        use_deadtime = bool(grouping.get("deadtime_correction", False))
        deadtime_mode = str(grouping.get("deadtime_mode", "off")).strip().lower() or "off"
        use_background = bool(grouping.get("background_correction", False))
        self._pending = _PreviewRequest(
            generation=self._generation,
            histograms=list(histograms),
            grouping=dict(grouping),
            forward_idx=list(forward_idx),
            backward_idx=list(backward_idx),
            alpha=_as_float(grouping.get("alpha"), 1.0),
            use_deadtime=use_deadtime,
            deadtime_mode=deadtime_mode,
            use_background=use_background,
            facility=str(facility or grouping.get("instrument", "") or ""),
            run_number=run_number,
        )
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
        request = self._pending
        self._pending = None
        if request is None:
            return
        # A ``reference_run`` background needs the loaded-dataset registry the
        # dialog does not own; the preview simply skips that subtraction (the
        # resolver returns None) so the curve still renders.
        self._tasks.start(
            lambda worker: _run_reduction(worker, request),
            on_finished=self._on_finished,
            on_error=self._on_error,
        )

    def _on_finished(self, result: object) -> None:
        if not isinstance(result, _PreviewResult):
            return
        if result.generation != self._generation:
            return  # superseded by a newer edit
        self._draw(result)

    def _on_error(self, message: str) -> None:
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
    """Reduce one preview request off the GUI thread.

    Pure numpy work: it touches no widgets and returns plain arrays, so the
    TaskRunner relay can marshal the result back safely. Cooperative cancellation
    is honoured up front (``worker.is_cancelled()``) so a shutdown mid-flight
    stops promptly; a merely *superseded* result is dropped later by generation
    in :meth:`GroupingPreviewPane._on_finished`.
    """
    if worker.is_cancelled():
        raise TaskCancelledError
    grouping = request.grouping
    reduction = reduce_grouped_asymmetry(
        histograms=request.histograms,
        grouping=grouping,
        forward_idx=request.forward_idx,
        backward_idx=request.backward_idx,
        alpha=request.alpha,
        use_deadtime=request.use_deadtime,
        deadtime_mode=request.deadtime_mode,
        use_background=request.use_background,
        facility=request.facility,
        reference_resolver=None,
    )
    return _PreviewResult(
        generation=request.generation,
        time=reduction.time,
        asymmetry=reduction.asymmetry,
        error=reduction.error,
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
