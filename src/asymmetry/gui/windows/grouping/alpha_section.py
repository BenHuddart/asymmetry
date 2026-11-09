"""Shared explicit calibration controls and cancellable workflow lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.calibration import best_calibration_run_index, classify_tf_calibration_run
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.io import background_reference_for_grouping
from asymmetry.core.project.profiles import AlphaPolicy, resolve_effective_grouping
from asymmetry.core.transform.t0 import find_t0_for_run
from asymmetry.core.workflow.calibration import (
    CalibrationOptions,
    calibrate_source,
    calibration_source,
    explicit_calibration_options,
    supports_joint_calibration,
)
from asymmetry.core.workflow.calibration_results import CalibrationOutcome
from asymmetry.core.workflow.reduction import ReductionSettings
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskRunner, TaskWorker, check_task_cancelled
from asymmetry.gui.widgets.no_scroll_spin import NoScrollComboBox
from asymmetry.gui.windows.grouping.format import ALPHA_METHOD_ITEMS, format_value_with_uncertainty


@dataclass(frozen=True)
class CalibrationReference:
    """Reference selection and loaded-source handles, resolved on the worker."""

    payload: dict
    datasets: tuple


@dataclass(frozen=True)
class CalibrationRequest:
    """Bounded request facts; acquired arrays are held by the captured source."""

    generation: int
    dataset: MuonDataset
    profile: Any
    options: CalibrationOptions
    reference: Any = None
    slot: str = "single"


@dataclass(frozen=True)
class CalibrationReply:
    generation: int
    slot: str
    outcome: CalibrationOutcome


def run_calibration(worker: TaskWorker, request: CalibrationRequest) -> CalibrationReply:
    """Resolve the source profile and run the shared numerical operation off-thread."""
    check_task_cancelled(worker)
    from asymmetry.core.io.periods import period_count

    run = request.dataset.run
    parent = run is not None and (
        "period_histograms" in run.grouping or "period_reduced" in run.grouping
    )
    options = replace(
        request.options,
        period=request.profile.period_mode
        if parent and period_count(request.dataset) > 1
        else None,
    )
    source = calibration_source(request.dataset, ReductionSettings(), options)
    profile = replace(request.profile, alpha_policy=AlphaPolicy(value=1.0), period_mode=None)
    reference = request.reference
    if isinstance(reference, CalibrationReference):
        from asymmetry.core.io import resolve_background_reference
        from asymmetry.core.transform.reference_scale import ReferenceExposure

        reference = resolve_background_reference(
            reference.payload,
            sample_exposure=ReferenceExposure.from_grouping(source.run.grouping),
            datasets=list(reference.datasets),
        )

    def resolve_reference(grouping):
        resolved = background_reference_for_grouping(reference, grouping)
        if resolved is None:
            return None
        from asymmetry.core.transform.grouping import independent_pair_indices
        from asymmetry.gui.utils.reference_timing import _resolve_reference_timing

        pair = independent_pair_indices(grouping, n_histograms=len(resolved.histograms))
        return _resolve_reference_timing(resolved, pair, worker)

    def search(histograms, metadata, **_options):
        return find_t0_for_run(
            histograms, metadata, cancel_check=lambda: check_task_cancelled(worker)
        )

    grouping = resolve_effective_grouping(
        profile, source.run, reference_resolver=resolve_reference, timing_search=search
    )
    source = replace(source, run=replace(source.run, grouping=grouping))
    outcome = calibrate_source(
        source,
        request.options,
        reference_resolver=resolve_reference,
        cancel_callback=worker.is_cancelled,
    )
    check_task_cancelled(worker)
    return CalibrationReply(request.generation, request.slot, outcome)


def run_metadata(dataset: MuonDataset) -> dict[str, Any]:
    """Merged run + dataset metadata (the run's is the loaders' richer copy)."""
    metadata: dict[str, Any] = dict(dataset.metadata or {})
    run = dataset.run
    if run is not None and isinstance(run.metadata, dict):
        metadata.update(run.metadata)
    return metadata


def run_summary(dataset: MuonDataset) -> str:
    """One-line ``Run — title · T · B`` summary for a calibration-run dropdown."""
    metadata = run_metadata(dataset)
    parts: list[str] = [f"Run {dataset.run_label}"]
    title = str(metadata.get("title", "")).strip()
    if title:
        parts.append(title)
    for key, unit in (("temperature", "K"), ("field", "G")):
        value = metadata.get(key)
        if value is not None:
            try:
                parts.append(f"{float(value):g} {unit}")
            except (TypeError, ValueError):
                pass
    return "  ·  ".join(parts)


def populate_calibration_run_combo(
    combo: QComboBox,
    datasets: list[MuonDataset],
    selected_run_number: int | None,
) -> None:
    """Fill *combo* with the fingerprint runs, highlighting weak-TF candidates.

    Weak-TF calibration candidates are drawn in the accent colour with the
    classifier's reason as a tooltip. The selection prefers *selected_run_number*,
    then the auto-picked best calibration run, then the first entry.
    """
    combo.clear()
    candidate_brush = QBrush(QColor(tokens.ACCENT))
    for ds in datasets:
        verdict = classify_tf_calibration_run(run_metadata(ds))
        combo.addItem(run_summary(ds), int(ds.run_number))
        index = combo.count() - 1
        if verdict.is_candidate:
            combo.setItemData(index, candidate_brush, Qt.ItemDataRole.ForegroundRole)
            combo.setItemData(index, verdict.reason, Qt.ItemDataRole.ToolTipRole)
    if not datasets:
        return
    if selected_run_number is not None:
        found = combo.findData(int(selected_run_number))
        if found >= 0:
            combo.setCurrentIndex(found)
            return
    auto = best_calibration_run_index([run_metadata(ds) for ds in datasets])
    combo.setCurrentIndex(auto if auto is not None else 0)


def source_identity(dataset):
    """Bounded acquired facts; count arrays are replaced rather than edited by the GUI."""
    run = dataset.run
    if run is None:
        return (id(dataset), None)
    grouping = run.grouping
    facts = (
        "period_mode",
        "t0_bin",
        "t0_time_us",
        "first_good_bin",
        "last_good_bin",
        "good_frames",
        "dead_time_us",
        "detector_t0_bins",
        "detector_first_good_bins",
        "detector_last_good_bins",
    )
    return (
        id(dataset),
        id(run),
        id(grouping),
        id(run.histograms),
        run.run_number,
        run.source_file,
        tuple(id(h.counts) for h in run.histograms),
        run.metadata.get("source_run_number"),
        run.metadata.get("period_number"),
        run.metadata.get("raw_period_histograms_available"),
        tuple((grouping.get("period_mapping", run.metadata.get("period_mapping")) or {}).items()),
        tuple((key, repr(grouping.get(key))) for key in facts),
        id(grouping.get("period_histograms")),
        id(grouping.get("period_reduced")),
    )


def source_snapshot(dataset):
    """Capture source containers while retaining owned, immutable acquisition arrays."""
    run = dataset.run
    if run is None:
        return replace(dataset, metadata=dict(dataset.metadata))
    grouping = {
        key: value.copy() if isinstance(value, (dict, list)) else value
        for key, value in run.grouping.items()
    }
    return replace(
        dataset,
        metadata=dict(dataset.metadata),
        run=replace(
            run, histograms=list(run.histograms), grouping=grouping, metadata=dict(run.metadata)
        ),
    )


class CalibrationDelivery(QObject):
    """Bind every terminal signal to the request generation on the GUI thread."""

    def __init__(self, section, generation):
        super().__init__(section)
        self.section = section
        self.generation = generation

    @Slot(str)
    def error(self, message):
        if self.generation == self.section._generation:
            self.section._error(message)
        self.deleteLater()

    @Slot()
    def cancelled(self):
        if self.generation == self.section._generation:
            self.section._cancelled()
        self.deleteLater()

    @Slot(object)
    def finished(self, reply):
        self.section._finished(reply)
        self.deleteLater()


class AlphaSectionWidget(QWidget):
    """One lifecycle for scalar and per-projection calibration, staged until Apply."""

    result_ready = Signal(object)
    invalidated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._datasets = []
        self._context_provider = None
        self._tasks = TaskRunner(self)
        self._worker = None
        self._generation = 0
        self._pending: dict[str, CalibrationOutcome] = {}
        self._queue: list[tuple[str, MuonDataset, Any, Any]] = []
        self._source = None
        self._source_key = None
        self._references = ()
        self._vector = False
        self._slots = []
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._run_combo = NoScrollComboBox()
        self._run_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._run_combo.setMinimumContentsLength(18)
        root.addWidget(QLabel("Calibration run"))
        root.addWidget(self._run_combo)
        self._method_combo = NoScrollComboBox()
        for label, key, tip in ALPHA_METHOD_ITEMS:
            self._method_combo.addItem(label, key)
            self._method_combo.setItemData(
                self._method_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        self._method_combo.setCurrentIndex(
            self._method_combo.findData(explicit_calibration_options().method)
        )
        root.addWidget(self._method_combo)
        self._fit_beta = QCheckBox("Calibrate β")
        root.addWidget(self._fit_beta)
        row = QHBoxLayout()
        self._estimate_btn = QPushButton("Estimate calibration")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        for button in (self._estimate_btn, self._stop_btn):
            button.setAutoDefault(False)
            row.addWidget(button)
        root.addLayout(row)
        self._result_label = QLabel(
            "Choose a source and estimate; Apply commits the displayed result."
        )
        self._result_label.setWordWrap(True)
        root.addWidget(self._result_label)
        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        root.addWidget(self._note_label)
        self._estimate_btn.clicked.connect(self._on_estimate)
        self._stop_btn.clicked.connect(self.invalidate)
        self._run_combo.currentIndexChanged.connect(self.invalidate)
        self._method_combo.currentIndexChanged.connect(self._method_changed)
        self._fit_beta.toggled.connect(self.invalidate)

    def configure(self, *, datasets, selected_run_number, context_provider):
        self.invalidate()
        self._datasets = list(datasets)
        self._context_provider = context_provider
        with_block = self._run_combo.blockSignals(True)
        populate_calibration_run_combo(self._run_combo, self._datasets, selected_run_number)
        self._run_combo.blockSignals(with_block)
        self._estimate_btn.setEnabled(bool(datasets))

    def set_vector(self, vector, slots=()):
        self._slots = list(slots)
        if vector != self._vector:
            self.invalidate()
        self._vector = vector
        self._method_changed()

    @Slot()
    def _method_changed(self):
        enabled = supports_joint_calibration(
            self._method_combo.currentData(), projection_layout=self._vector
        )
        self._fit_beta.setEnabled(enabled)
        if not enabled:
            self._fit_beta.setChecked(False)
        self.invalidate()

    @Slot()
    def invalidate(self):
        self._generation += 1
        if self._worker is not None:
            self._worker.cancel()
        self._worker = None
        self._pending.clear()
        self._queue.clear()
        self._source = None
        self._source_key = None
        self._references = ()
        self._stop_btn.setEnabled(False)
        self._estimate_btn.setEnabled(bool(self._datasets))
        self._result_label.setText("Estimate calibration for the current inputs.")
        self._note_label.clear()
        self.invalidated.emit()

    def _current_dataset(self):
        index = self._run_combo.currentIndex()
        return self._datasets[index] if 0 <= index < len(self._datasets) else None

    @Slot()
    def _on_estimate(self):
        self.estimate_slots(self._slots if self._vector else ["single"])

    def estimate_slots(self, slots):
        self.invalidate()
        dataset = self._current_dataset()
        if dataset is None or self._context_provider is None:
            return
        self._source = dataset
        self._source_key = source_identity(dataset)
        dataset = source_snapshot(dataset)
        try:
            context = self._context_provider()
            self._queue = []
            references = {}
            for slot in slots:
                profile, reference = context(slot, dataset)
                if isinstance(reference, CalibrationReference):
                    for source in reference.datasets:
                        references[id(source)] = (source, source_identity(source))
                    reference = replace(
                        reference, datasets=tuple(source_snapshot(ds) for ds in reference.datasets)
                    )
                self._queue.append((slot, dataset, profile, reference))
            self._references = tuple(references.values())
        except (ValueError, OSError) as exc:
            self._result_label.setText(f"Calibration unavailable: {exc}")
            return
        self._start_next()

    def _start_next(self):
        if not self._queue:
            self._worker = None
            self._estimate_btn.setEnabled(bool(self._datasets))
            self._stop_btn.setEnabled(False)
            return
        slot, dataset, profile, reference = self._queue.pop(0)
        options = CalibrationOptions(
            str(self._method_combo.currentData()), self._fit_beta.isChecked(), period=None
        )
        request = CalibrationRequest(self._generation, dataset, profile, options, reference, slot)
        self._estimate_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._result_label.setText("Computing calibration…")
        delivery = CalibrationDelivery(self, request.generation)
        self._worker = self._tasks.start(
            lambda worker: run_calibration(worker, request),
            on_finished=delivery.finished,
            on_error=delivery.error,
            on_cancelled=delivery.cancelled,
            cancel_exceptions=(FitCancelledError,),
        )

    @Slot(object)
    def _finished(self, reply):
        if reply.generation != self._generation:
            return
        if not self._sources_current():
            self.invalidate()
            return
        outcome = reply.outcome
        selected = outcome.selected
        if outcome.alpha is None:
            self._error(outcome.message)
            return
        self._pending[reply.slot] = outcome
        lines = []
        for slot, result in self._pending.items():
            fitted = result.selected
            alpha = format_value_with_uncertainty(fitted.alpha, fitted.alpha_error)
            if result.record is not None and fitted.alpha_error is None:
                alpha += " (error unavailable)"
            beta = "1 (fixed)"
            if result.record is not None and not fitted.beta_fixed:
                beta = format_value_with_uncertainty(fitted.beta, fitted.beta_error)
                if fitted.beta_error is None:
                    beta += " (error unavailable)"
            model = (
                result.selected.envelope.value.title()
                if result.record
                else next(label for label, key, _ in ALPHA_METHOD_ITEMS if key == result.method)
            )
            prefix = "" if slot == "single" else f"{slot}: "
            lines.append(f"{prefix}α = {alpha} · β = {beta} · {model}")
        self._result_label.setText("\n".join(lines))
        self._note_label.setText(
            f"TF errors are conditional on the selected count model and corrections (covariance: {selected.fit.uncertainty_status}). Apply commits these values."
            if outcome.record
            else "Quick calibration; Apply commits these values."
        )
        source = outcome.source
        period = (
            f"acquired period {outcome.period_number}"
            if outcome.period_number is not None
            else (
                f"mapped set {source.mapped_period_number}"
                if outcome.record is None and source.mapped_period_number is not None
                else "period unidentified"
            )
        )
        self._note_label.setText(
            f"Source run {outcome.run_number}, {period}. " + self._note_label.text()
        )
        self.result_ready.emit(reply)
        self._start_next()

    @Slot(str)
    def _error(self, message):
        if self._worker is None:
            return
        self._queue.clear()
        self._pending.clear()
        self._result_label.setText(f"Calibration failed: {message}")
        self._start_next()

    @Slot()
    def _cancelled(self):
        if self._worker is not None:
            self.invalidate()

    def _sources_current(self):
        return (
            self._source is not None
            and source_identity(self._source) == self._source_key
            and all(source_identity(source) == key for source, key in self._references)
        )

    def pending(self):
        if self._source is not None and not self._sources_current():
            self.invalidate()
        return dict(self._pending)

    def shutdown(self):
        self.invalidate()
        self._tasks.shutdown()
