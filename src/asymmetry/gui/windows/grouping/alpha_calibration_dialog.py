"""Dedicated alpha-calibration dialog for the grouping profile editor.

Alpha (the detector-balance parameter) is measured once, on a transverse-field
*calibration run*, then applied to every run. This dialog makes that measurement
concrete: the user picks a calibration run, an estimation method, and the good-bin
window, presses **Estimate**, and *sees* the effect — the calibration run's
forward/backward asymmetry is drawn twice, once with alpha = 1 (a muted grey
"before") and once with the estimated alpha (an accent-coloured "after"). A
balanced alpha flattens the "after" curve about zero, so the plot makes a good
calibration self-evident.

The dialog edits nothing directly: it opens on the draft profile's current
:class:`~asymmetry.core.project.profiles.AlphaPolicy` and, on **OK**, returns a
new ``AlphaPolicy(mode="calibrated", …)`` carrying the estimate's value, error,
method, and source run. **Cancel** returns ``None`` and the caller keeps its
policy. The grouping window's ``Calibrate…`` button launches it (see
:mod:`asymmetry.gui.windows.grouping.dialog`).

The two preview curves (the α=1 "before" and the α̂ "after") are fast, pure-core
reductions and stay synchronous. The **Estimate** action's own work —
:func:`~asymmetry.core.transform.grouping.group_forward_backward` over the full
forward/backward groups (a full-histogram scan for large detector counts) plus
:func:`~asymmetry.core.transform.asymmetry.estimate_alpha_detailed` — runs on a
:class:`~asymmetry.gui.tasks.TaskRunner` worker thread instead (see
``gui/tasks.py`` and ``AGENTS.md``'s no-GUI-thread-blocking rule). Inputs are
snapshotted into a plain :class:`_AlphaEstimateRequest` on the GUI thread before
the worker starts, so the worker never touches widgets; the button is disabled
and the result label shows a busy hint for the duration. The runner is shut down
on every dismissal (``done``/``closeEvent``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.calibration import (
    best_calibration_run_index,
    classify_tf_calibration_run,
)
from asymmetry.core.data.dataset import Histogram, MuonDataset
from asymmetry.core.project.profiles import AlphaPolicy
from asymmetry.core.transform.asymmetry import AlphaEstimate, estimate_alpha_detailed
from asymmetry.core.transform.grouping import group_forward_backward
from asymmetry.core.transform.rebin import binned_fb_asymmetry
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker
from asymmetry.gui.windows.grouping.format import (
    ALPHA_METHOD_ITEMS,
    format_value_with_uncertainty,
)


@dataclass(frozen=True)
class _AlphaEstimateRequest:
    """An immutable snapshot of what to estimate, built on the GUI thread.

    Everything here is a plain object, so the worker function can run entirely
    off the GUI thread — it must never read widgets.
    """

    token: int
    histograms: list[Histogram]
    grouping: dict[str, Any]
    method: str
    first_good_bin: int
    last_good_bin: int
    run_label: str


@dataclass(frozen=True)
class _AlphaEstimateResult:
    """The estimate marshalled back to the GUI thread, tagged with its token."""

    token: int
    estimate: AlphaEstimate
    run_label: str


def _run_alpha_estimate(worker: TaskWorker, request: _AlphaEstimateRequest) -> _AlphaEstimateResult:
    """Group the full forward/backward histograms and estimate alpha off-thread.

    ``group_forward_backward`` raises :class:`ValueError` when the groups
    reference no present detectors; :class:`~asymmetry.gui.tasks.TaskWorker`
    turns that into the worker's ``error`` signal, which the dialog surfaces
    with the same warning dialog the synchronous path used.
    """
    if worker.is_cancelled():
        raise TaskCancelledError
    grouped = group_forward_backward(request.histograms, request.grouping)
    forward, backward, common_t0 = grouped.forward, grouped.backward, int(grouped.common_t0)
    bin_width = float(request.histograms[0].bin_width)

    time_us = None
    if request.method == "general":
        time_us = (np.arange(forward.size, dtype=np.float64) - float(common_t0)) * bin_width

    if worker.is_cancelled():
        raise TaskCancelledError
    estimate = estimate_alpha_detailed(
        forward,
        backward,
        method=request.method,
        time_us=time_us,
        first_good_bin=request.first_good_bin,
        last_good_bin=request.last_good_bin,
    )
    return _AlphaEstimateResult(token=request.token, estimate=estimate, run_label=request.run_label)


class AlphaCalibrationDialog(QDialog):
    """Estimate and preview the balance ``alpha`` from a calibration run.

    Parameters
    ----------
    datasets
        Loaded datasets of the current fingerprint (the calibration-run
        dropdown lists all of them; datasets without a run are ignored).
    groups
        The draft's detector groups as ``gid -> 0-based detector indices`` (the
        dialog's internal grouping representation).
    group_names
        ``gid -> display name`` for the groups (used in the group summary).
    forward_group, backward_group
        Analysis group ids the calibration integrates over.
    excluded_detectors
        1-based detector ids to drop from every group (the exclusion field).
    initial_policy
        The draft's current alpha policy; seeds the method combo and, when it
        carries a ``source_run``, the initial calibration-run selection.
    slot_label
        Optional projection label shown in the title bar (e.g. ``"P_x"`` or
        ``"Top-Bottom"``) when calibrating one axis of a vector grouping.
    selected_run_number
        Run to prefer as the initial calibration run when the policy carries no
        source run (typically the grouping window's preview run).
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        groups: dict[int, list[int]],
        group_names: dict[int, str] | None = None,
        forward_group: int,
        backward_group: int,
        excluded_detectors: list[int] | None = None,
        initial_policy: AlphaPolicy | None = None,
        slot_label: str | None = None,
        selected_run_number: int | None = None,
        parent=None,
    ) -> None:
        """Build the dialog; see the class docstring for parameter semantics."""
        super().__init__(parent)
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._groups = {int(gid): [int(i) for i in idxs] for gid, idxs in groups.items()}
        self._group_names = {int(k): str(v) for k, v in (group_names or {}).items()}
        self._forward_group = int(forward_group)
        self._backward_group = int(backward_group)
        self._excluded_detectors = [int(d) for d in (excluded_detectors or [])]
        self._slot_label = slot_label
        self._result_policy: AlphaPolicy | None = None
        #: Latest successful estimate, used to draw the "after" curve and build
        #: the returned policy. Cleared whenever the run/window/method changes.
        self._estimate: AlphaEstimate | None = None

        # Off-thread estimate worker. Created before the "no runs" early return
        # below so ``done``/``closeEvent`` can unconditionally shut it down.
        self._tasks = TaskRunner(self)
        #: Bumped on every Estimate click; a finished/errored result whose
        #: token no longer matches the current one is stale (the run or inputs
        #: changed while the worker was in flight) and is discarded.
        self._estimate_token = 0
        self._estimate_prior_text = ""

        title = "Alpha Calibration"
        if slot_label:
            title = f"Alpha Calibration — {slot_label}"
        self.setWindowTitle(title)
        self.resize(720, 620)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        if not self._datasets:
            root.addWidget(QLabel("No runs with histograms are available to calibrate against."))
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)
            return

        # -- controls --------------------------------------------------------
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._run_combo = QComboBox()
        self._populate_run_combo()
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        form.addRow("Calibration run", self._run_combo)

        self._group_summary = QLabel("")
        self._group_summary.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._group_summary.setWordWrap(True)
        form.addRow("Groups", self._group_summary)

        self._method_combo = QComboBox()
        for label, key, explanation in ALPHA_METHOD_ITEMS:
            self._method_combo.addItem(label, key)
            self._method_combo.setItemData(
                self._method_combo.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole
            )
        initial_method = (initial_policy.method if initial_policy else "") or "diamagnetic"
        self._set_method(initial_method)
        self._method_combo.currentIndexChanged.connect(self._on_inputs_changed)
        form.addRow("Method", self._method_combo)

        self._first_good_spin = QSpinBox()
        self._first_good_spin.setRange(0, 1)
        self._last_good_spin = QSpinBox()
        self._last_good_spin.setRange(0, 1)
        self._first_good_spin.valueChanged.connect(self._on_inputs_changed)
        self._last_good_spin.valueChanged.connect(self._on_inputs_changed)
        window_row = QWidget()
        window_layout = QHBoxLayout(window_row)
        window_layout.setContentsMargins(0, 0, 0, 0)
        window_layout.addWidget(QLabel("First"))
        window_layout.addWidget(self._first_good_spin)
        window_layout.addWidget(QLabel("Last"))
        window_layout.addWidget(self._last_good_spin)
        window_layout.addStretch()
        form.addRow("Good-bin window", window_row)

        root.addLayout(form)

        # -- estimate row ----------------------------------------------------
        estimate_row = QHBoxLayout()
        self._estimate_btn = QPushButton("Estimate")
        self._estimate_btn.setDefault(True)
        self._estimate_btn.clicked.connect(self._on_estimate)
        estimate_row.addWidget(self._estimate_btn)
        self._result_label = QLabel("Press Estimate to measure α from this run.")
        self._result_label.setWordWrap(True)
        estimate_row.addWidget(self._result_label, stretch=1)
        root.addLayout(estimate_row)

        # -- preview canvas --------------------------------------------------
        self._figure = None
        self._axes = None
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas = create_canvas(layout="tight", parent=self)
            self._axes = self._figure.add_subplot(111)
            root.addWidget(self._canvas, stretch=1)
        except ImportError:
            root.addWidget(QLabel("matplotlib is not installed — preview unavailable."))
            self._canvas = None

        # -- OK / Cancel -----------------------------------------------------
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        # Seed the initial run selection: the policy's source run wins, then the
        # caller's selected run, then the auto-picked calibration candidate.
        self._select_initial_run(initial_policy, selected_run_number)
        self._on_run_changed()

    # ------------------------------------------------------------------
    # Run dropdown
    # ------------------------------------------------------------------

    def _populate_run_combo(self) -> None:
        """List every fingerprint run, highlighting weak-TF calibration runs."""
        combo = self._run_combo
        combo.blockSignals(True)
        combo.clear()
        candidate_brush = QBrush(QColor(tokens.ACCENT))
        for ds in self._datasets:
            verdict = classify_tf_calibration_run(self._run_metadata(ds))
            combo.addItem(self._run_summary(ds), int(ds.run_number))
            index = combo.count() - 1
            if verdict.is_candidate:
                combo.setItemData(index, candidate_brush, Qt.ItemDataRole.ForegroundRole)
                combo.setItemData(index, verdict.reason, Qt.ItemDataRole.ToolTipRole)
        combo.blockSignals(False)

    @staticmethod
    def _run_metadata(dataset: MuonDataset) -> dict[str, Any]:
        """Merged run + dataset metadata (the run's is the loaders' richer copy)."""
        metadata: dict[str, Any] = dict(dataset.metadata or {})
        run = dataset.run
        if run is not None and isinstance(run.metadata, dict):
            metadata.update(run.metadata)
        return metadata

    @classmethod
    def _run_summary(cls, dataset: MuonDataset) -> str:
        """One-line ``run — title · T · B`` summary for the dropdown."""
        metadata = cls._run_metadata(dataset)
        parts: list[str] = [f"Run {dataset.run_label}"]
        title = str(metadata.get("title", "")).strip()
        if title:
            parts.append(title)
        temperature = metadata.get("temperature")
        if temperature is not None:
            try:
                parts.append(f"{float(temperature):g} K")
            except (TypeError, ValueError):
                pass
        field = metadata.get("field")
        if field is not None:
            try:
                parts.append(f"{float(field):g} G")
            except (TypeError, ValueError):
                pass
        return "  ·  ".join(parts)

    def _select_initial_run(
        self, initial_policy: AlphaPolicy | None, selected_run_number: int | None
    ) -> None:
        """Choose the opening calibration run (policy source > selected > auto)."""
        preferred: int | None = None
        if initial_policy is not None and initial_policy.source_run is not None:
            preferred = int(initial_policy.source_run)
        elif selected_run_number is not None:
            preferred = int(selected_run_number)

        if preferred is not None:
            index = self._run_combo.findData(preferred)
            if index >= 0:
                self._run_combo.setCurrentIndex(index)
                return

        auto = best_calibration_run_index([self._run_metadata(ds) for ds in self._datasets])
        self._run_combo.setCurrentIndex(auto if auto is not None else 0)

    def _current_dataset(self) -> MuonDataset | None:
        run_number = self._run_combo.currentData()
        if run_number is None:
            return None
        return next((ds for ds in self._datasets if int(ds.run_number) == int(run_number)), None)

    # ------------------------------------------------------------------
    # Reactions
    # ------------------------------------------------------------------

    def _on_run_changed(self) -> None:
        """Re-seed the good-bin window from the new run and redraw."""
        dataset = self._current_dataset()
        self._group_summary.setText(self._group_summary_text())
        # Invalidate any in-flight estimate: its result would no longer match
        # the selected run when the worker finishes.
        self._estimate_token += 1
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            self._estimate = None
            self._clear_plot("Selected run has no histograms.")
            return
        n_bins = int(dataset.run.histograms[0].n_bins)
        grouping = dataset.run.grouping if isinstance(dataset.run.grouping, dict) else {}
        first_default, last_default = self._resolved_good_window(grouping, n_bins)
        for spin, value in (
            (self._first_good_spin, first_default),
            (self._last_good_spin, last_default),
        ):
            spin.blockSignals(True)
            spin.setRange(0, max(0, n_bins - 1))
            spin.setValue(value)
            spin.blockSignals(False)
        # A new run invalidates any prior estimate.
        self._estimate = None
        self._result_label.setStyleSheet("")
        self._result_label.setText("Press Estimate to measure α from this run.")
        self._redraw_preview()

    def _on_inputs_changed(self) -> None:
        """A method/window change invalidates the estimate; redraw the before."""
        # Invalidate any in-flight estimate — it was snapshotted from the
        # inputs before this change.
        self._estimate_token += 1
        self._estimate = None
        self._result_label.setStyleSheet("")
        self._result_label.setText("Inputs changed — press Estimate to re-measure α.")
        self._redraw_preview()

    @staticmethod
    def _resolved_good_window(grouping: dict[str, Any], n_bins: int) -> tuple[int, int]:
        """Default (first, last) good bins from the run's resolved facts."""
        try:
            first = int(grouping.get("first_good_bin", 0))
        except (TypeError, ValueError):
            first = 0
        try:
            last = int(grouping.get("last_good_bin", n_bins - 1))
        except (TypeError, ValueError):
            last = n_bins - 1
        first = max(0, min(first, n_bins - 1))
        last = max(first, min(last, n_bins - 1))
        return first, last

    def _group_summary_text(self) -> str:
        """Human-readable ``Forward <n> vs Backward <m>`` group description."""
        fwd = self._group_display(self._forward_group)
        bwd = self._group_display(self._backward_group)
        return f"Forward {fwd}  vs  Backward {bwd}"

    def _group_display(self, gid: int) -> str:
        name = self._group_names.get(int(gid), "").strip()
        return f"{gid} ({name})" if name else str(gid)

    # ------------------------------------------------------------------
    # Estimation + preview
    # ------------------------------------------------------------------

    def _set_method(self, method_key: str) -> None:
        index = self._method_combo.findData(str(method_key))
        self._method_combo.setCurrentIndex(index if index >= 0 else 0)

    def _current_method(self) -> str:
        return str(self._method_combo.currentData() or "diamagnetic")

    def _grouping_for_reduction(self, dataset: MuonDataset) -> dict[str, Any]:
        """A minimal grouping dict for the F/B reduction of *dataset*.

        Uses the dialog's draft groups (1-based ids, as the reduction expects)
        and the current forward/backward selection plus exclusions, so the
        preview matches exactly what the estimate integrates.
        """
        return {
            "groups": {int(gid): [int(i) + 1 for i in idxs] for gid, idxs in self._groups.items()},
            "forward_group": self._forward_group,
            "backward_group": self._backward_group,
            "excluded_detectors": list(self._excluded_detectors),
        }

    def _grouped_counts(
        self, dataset: MuonDataset
    ) -> tuple[np.ndarray, np.ndarray, int, float] | None:
        """Aligned forward/backward counts (+ common t0, bin width) for *dataset*.

        Returns ``None`` (and warns) when the groups reference no present
        detectors, mirroring the grouping dialog's guard.
        """
        run = dataset.run
        assert run is not None
        try:
            grouped = group_forward_backward(run.histograms, self._grouping_for_reduction(dataset))
        except ValueError as exc:
            QMessageBox.warning(self, "Alpha Calibration", str(exc))
            return None
        bin_width = float(run.histograms[0].bin_width)
        return grouped.forward, grouped.backward, int(grouped.common_t0), bin_width

    def _on_estimate(self) -> None:
        """Snapshot the current inputs and estimate alpha off the GUI thread."""
        dataset = self._current_dataset()
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            QMessageBox.warning(self, "Alpha Calibration", "Selected run has no histograms.")
            return

        self._estimate_token += 1
        request = _AlphaEstimateRequest(
            token=self._estimate_token,
            histograms=list(dataset.run.histograms),
            grouping=self._grouping_for_reduction(dataset),
            method=self._current_method(),
            first_good_bin=int(self._first_good_spin.value()),
            last_good_bin=int(self._last_good_spin.value()),
            run_label=str(dataset.run_label),
        )

        self._estimate_prior_text = self._result_label.text()
        self._estimate_btn.setEnabled(False)
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._result_label.setText("Computing estimate…")
        self._tasks.start(
            lambda worker: _run_alpha_estimate(worker, request),
            on_finished=self._on_estimate_finished,
            on_error=self._on_estimate_error,
        )

    def _on_estimate_finished(self, result: object) -> None:
        """GUI thread: apply the estimate unless a newer request supersedes it."""
        self._estimate_btn.setEnabled(True)
        self._result_label.setStyleSheet("")
        if not isinstance(result, _AlphaEstimateResult) or result.token != self._estimate_token:
            return  # superseded by a later Estimate click / input change
        estimate = result.estimate

        if not estimate.ok:
            self._estimate = None
            self._result_label.setText(f"Estimate failed: {estimate.message}")
            self._redraw_preview()
            return

        self._estimate = estimate
        method_label = next(
            (label for label, key, _ in ALPHA_METHOD_ITEMS if key == estimate.method),
            estimate.method,
        )
        formatted = format_value_with_uncertainty(estimate.alpha, estimate.alpha_error)
        self._result_label.setText(f"α = {formatted}  ·  {method_label}  ·  run {result.run_label}")
        self._redraw_preview()

    def _on_estimate_error(self, message: str) -> None:
        """GUI thread: restore the prior status and surface the same warning."""
        self._estimate_btn.setEnabled(True)
        self._result_label.setStyleSheet("")
        self._result_label.setText(self._estimate_prior_text)
        QMessageBox.warning(self, "Alpha Calibration", message)

    def _redraw_preview(self) -> None:
        """Draw the before (α=1) and, when available, after (α̂) asymmetry."""
        if self._axes is None or self._canvas is None:
            return
        axes = self._axes
        axes.clear()
        dataset = self._current_dataset()
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            self._canvas.draw_idle()
            return
        counts = self._grouped_counts(dataset)
        if counts is None:
            self._canvas.draw_idle()
            return
        forward, backward, common_t0, bin_width = counts
        first_good = int(self._first_good_spin.value())
        last_good = int(self._last_good_spin.value())

        # A modest display bunching keeps the two overlaid curves legible on a
        # fine-binned continuous-source run without changing the estimate.
        n_window = max(1, last_good - first_good + 1)
        bunch = max(1, n_window // 400)
        grouping = {"bunching_factor": int(bunch)}

        axes.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=1.0, zorder=1)

        drew = False
        time_before, asym_before = self._binned_curve(
            forward, backward, grouping, common_t0, bin_width, 1.0, first_good, last_good
        )
        if time_before is not None:
            axes.plot(
                time_before,
                asym_before * 100.0,
                color=tokens.TEXT_DIM,
                linewidth=1.2,
                label="α = 1 (before)",
                zorder=2,
            )
            drew = True

        if self._estimate is not None:
            time_after, asym_after = self._binned_curve(
                forward,
                backward,
                grouping,
                common_t0,
                bin_width,
                float(self._estimate.alpha),
                first_good,
                last_good,
            )
            if time_after is not None:
                axes.plot(
                    time_after,
                    asym_after * 100.0,
                    color=tokens.ACCENT,
                    linewidth=1.4,
                    label=f"α = {self._estimate.alpha:.4f} (after)",
                    zorder=3,
                )
                drew = True

        axes.set_xlabel("Time (µs)")
        axes.set_ylabel("Asymmetry (%)")
        if drew:
            axes.legend(loc="best", fontsize="small", framealpha=0.9)
        self._canvas.draw_idle()

    @staticmethod
    def _binned_curve(
        forward: np.ndarray,
        backward: np.ndarray,
        grouping: dict[str, Any],
        common_t0: int,
        bin_width: float,
        alpha: float,
        first_good: int,
        last_good: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Reduce F/B counts to a display asymmetry curve for one alpha."""
        time, asym, _error = binned_fb_asymmetry(
            forward,
            backward,
            grouping=grouping,
            common_t0=common_t0,
            bin_width_us=bin_width,
            alpha=alpha,
            first_good_bin=first_good,
            last_good_bin=last_good,
        )
        if time.size == 0:
            return None, None
        return time, asym

    def _clear_plot(self, message: str) -> None:
        self._result_label.setText(message)
        if self._axes is not None and self._canvas is not None:
            self._axes.clear()
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        """Build the calibrated policy from the last estimate and accept."""
        if self._estimate is None:
            QMessageBox.warning(
                self,
                "Alpha Calibration",
                "Press Estimate to measure α before accepting the calibration.",
            )
            return
        dataset = self._current_dataset()
        source_run = int(dataset.run_number) if dataset is not None else None
        self._result_policy = AlphaPolicy(
            mode="calibrated",
            value=float(self._estimate.alpha),
            error=self._estimate.alpha_error,
            method=str(self._estimate.method),
            source_run=source_run,
        )
        self.accept()

    def result_policy(self) -> AlphaPolicy | None:
        """The calibrated :class:`AlphaPolicy` on OK, or ``None`` on Cancel."""
        return self._result_policy

    # ------------------------------------------------------------------
    # Teardown (TaskRunner contract, gui/tasks.py)
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Cancel and join the estimate worker on window close (the ✕ button)."""
        self._tasks.shutdown()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """Cancel and join the estimate worker on every dismissal (accept/reject).

        ``QDialog.done`` only ``hide()``s the window rather than routing through
        ``closeEvent``, so accept()/reject() need their own teardown call;
        ``TaskRunner.shutdown`` is idempotent (an empty ``_live`` list on a
        second call), so this and ``closeEvent`` both firing for the ✕-button
        path is harmless.
        """
        self._tasks.shutdown()
        super().done(result)


__all__ = ["AlphaCalibrationDialog"]
