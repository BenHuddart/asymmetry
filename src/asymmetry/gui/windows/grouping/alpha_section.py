"""Inline α (detector-balance) calibration for the grouping Corrections panel.

Hosts the single-α estimate controls — a calibration-run picker (weak-TF
candidates highlighted), a method combo, and an **Estimate** button — that used
to live in the standalone ``AlphaCalibrationDialog``. The estimate runs on a
:class:`~asymmetry.gui.tasks.TaskRunner` worker thread over the *corrected*
forward/backward counts (deadtime + background applied, see PR 1); on success
the section emits :attr:`alpha_estimated` with a calibrated
:class:`~asymmetry.core.project.profiles.AlphaPolicy`, and the owning grouping
dialog applies it to the α spin and the shared preview (which already draws the
α=1↔α̂ overlay and the residual baseline). There is no preview of its own — the
unified grouping preview is the single source of truth.

The off-thread estimate worker (:func:`run_alpha_estimate`), its corrected-F/B
builder (:func:`build_corrected`), the request builder (:func:`build_alpha_request`)
and the run-combo helpers (:func:`populate_calibration_run_combo`) are shared with
the grouping dialog's inline *vector* per-projection α estimate (the standalone
``AlphaCalibrationDialog`` it used to launch has been retired).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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
from asymmetry.core.transform.background import resolve_background_mode
from asymmetry.core.transform.grouping import effective_group_indices
from asymmetry.core.transform.reduce import (
    CorrectedGroupedCounts,
    ReferenceResolver,
    corrected_grouped_counts,
    correction_flags_from_grouping,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker
from asymmetry.gui.widgets.no_scroll_spin import NoScrollComboBox
from asymmetry.gui.windows.grouping.format import (
    ALPHA_METHOD_ITEMS,
    format_value_with_uncertainty,
)

__all__ = [
    "AlphaEstimateRequest",
    "AlphaEstimateResult",
    "AlphaSectionWidget",
    "build_alpha_request",
    "build_corrected",
    "correction_note",
    "good_window",
    "grouping_for_reduction",
    "populate_calibration_run_combo",
    "resolve_reference",
    "run_alpha_estimate",
    "run_metadata",
    "run_summary",
]

CorrectionProvider = Callable[[MuonDataset], "dict[str, Any]"]

#: Background ``method`` labels that mean *no subtraction happened* — a requested
#: background correction that could not be applied to this run.
_BACKGROUND_NOT_APPLIED = frozenset({"", "none", "missing_reference", "missing_fixed_values"})


@dataclass(frozen=True)
class AlphaEstimateRequest:
    """An immutable snapshot of what to estimate, built on the GUI thread.

    Everything here is a plain object, so the worker runs entirely off the GUI
    thread. ``resolved_reference`` is the ``reference_run`` background's
    ``(histograms, scale)`` pre-resolved on the GUI thread (or ``None``).
    """

    token: int
    histograms: list[Histogram]
    grouping: dict[str, Any]
    method: str
    first_good_bin: int
    last_good_bin: int
    run_label: str
    resolved_reference: tuple[list[Histogram], float] | None
    facility: str


@dataclass(frozen=True)
class AlphaEstimateResult:
    """The estimate marshalled back to the GUI thread, tagged with its token.

    ``note_text``/``note_warn`` describe which corrections the estimate reflected
    (the modal recomputes its own note for its preview and ignores these).
    """

    token: int
    estimate: AlphaEstimate
    run_label: str
    note_text: str = ""
    note_warn: bool = False


def build_corrected(
    histograms: list[Histogram],
    grouping: dict[str, Any],
    resolved_reference: tuple[list[Histogram], float] | None,
    facility: str,
) -> CorrectedGroupedCounts:
    """Deadtime-correct, group and background-subtract F/B for the estimate.

    Pure (no widgets). Raises :class:`ValueError` when the groups reference no
    present detectors, matching the grouping dialog's guard.
    """
    n = len(histograms)
    forward_gid = int(grouping.get("forward_group", 1))
    backward_gid = int(grouping.get("backward_group", 2))
    forward_idx = effective_group_indices(grouping, forward_gid, n_histograms=n)
    backward_idx = effective_group_indices(grouping, backward_gid, n_histograms=n)
    if not forward_idx or not backward_idx:
        raise ValueError(
            "Forward/backward groups do not reference any detectors present in this run "
            "(after detector exclusion)."
        )
    flags = correction_flags_from_grouping(grouping)
    resolver = None if resolved_reference is None else (lambda _g: resolved_reference)
    return corrected_grouped_counts(
        histograms=histograms,
        grouping=grouping,
        forward_idx=forward_idx,
        backward_idx=backward_idx,
        use_deadtime=flags.use_deadtime,
        deadtime_mode=flags.deadtime_mode,
        use_background=flags.use_background,
        facility=facility,
        reference_resolver=resolver,
    )


def correction_note(
    grouping: dict[str, Any], corrected: CorrectedGroupedCounts
) -> tuple[str, bool]:
    """Describe which corrections the estimate reflects (text, warn?).

    ``warn`` is ``True`` when a *requested* correction could not be applied to
    this run — the anti-mislead guardrail.
    """
    flags = correction_flags_from_grouping(grouping)
    applied: list[str] = []
    missing: list[str] = []
    if flags.use_deadtime:
        (applied if corrected.deadtime_applied else missing).append("deadtime")
    if flags.use_background:
        method = str((corrected.background_state or {}).get("method", ""))
        if method and method not in _BACKGROUND_NOT_APPLIED:
            applied.append(f"background ({method})")
        else:
            missing.append("background")
    if not flags.use_deadtime and not flags.use_background:
        return (
            "No deadtime or background correction is configured — α is estimated on raw counts.",
            False,
        )
    if missing:
        pronoun = "it" if len(missing) == 1 else "them"
        note = ""
        if applied:
            note = "Applied: " + ", ".join(applied) + ".  "
        note += (
            "Not applied to this run: " + ", ".join(missing) + f" — α does not reflect {pronoun}."
        )
        return (note, True)
    return ("α is estimated on corrected counts (" + ", ".join(applied) + ").", False)


def run_alpha_estimate(worker: TaskWorker, request: AlphaEstimateRequest) -> AlphaEstimateResult:
    """Build corrected F/B and estimate α off the GUI thread (+ correction note).

    ``build_corrected`` raises :class:`ValueError` when the groups reference no
    present detectors; the TaskWorker turns that into the worker's ``error``
    signal.
    """
    if worker.is_cancelled():
        raise TaskCancelledError
    corrected = build_corrected(
        request.histograms, request.grouping, request.resolved_reference, request.facility
    )
    forward, backward, common_t0 = corrected.forward, corrected.backward, int(corrected.common_t0)
    bin_width = float(corrected.bin_width)

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
    note_text, note_warn = correction_note(request.grouping, corrected)
    return AlphaEstimateResult(
        token=request.token,
        estimate=estimate,
        run_label=request.run_label,
        note_text=note_text,
        note_warn=note_warn,
    )


# ----------------------------------------------------------------------------
# Shared run-combo + request builders (used by the inline single-α section here
# and by the grouping dialog's inline per-projection vector estimate). One
# implementation, so the two estimate call sites agree by construction.
# ----------------------------------------------------------------------------


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


def grouping_for_reduction(
    dataset: MuonDataset,
    *,
    groups: dict[int, list[int]],
    forward_gid: int,
    backward_gid: int,
    excluded_detectors: list[int],
    correction_provider: CorrectionProvider | None,
) -> dict[str, Any]:
    """Resolved correction grouping for *dataset* with a draft group pair.

    Starts from the dataset's resolved correction payload (deadtime + background
    config plus per-run t0 / good-window facts) so α is estimated on the same
    corrected counts the reduction applies it to, then overrides the group
    selection, forward/backward pair and exclusions. Without a
    ``correction_provider`` this degrades to the minimal raw-count dict. ``groups``
    are the dialog's 0-based detector indices; they are stored 1-based here.
    """
    grouping: dict[str, Any] = {}
    if correction_provider is not None:
        try:
            grouping = dict(correction_provider(dataset) or {})
        except Exception:  # noqa: BLE001 — a broken provider degrades to raw counts
            grouping = {}
    grouping["groups"] = {int(gid): [int(i) + 1 for i in idxs] for gid, idxs in groups.items()}
    grouping["forward_group"] = int(forward_gid)
    grouping["backward_group"] = int(backward_gid)
    grouping["excluded_detectors"] = list(excluded_detectors)
    return grouping


def resolve_reference(
    grouping: dict[str, Any], reference_resolver: ReferenceResolver | None
) -> tuple[list[Histogram], float] | None:
    """Resolve the ``reference_run`` background on the GUI thread, or ``None``.

    Done on the GUI thread (not in the worker) because the resolver reaches into
    the loaded-dataset registry; the result is plain data snapshotted into the
    worker request. Degrades to ``None`` for non-reference modes or any failure.
    """
    if reference_resolver is None:
        return None
    if resolve_background_mode(grouping) != "reference_run":
        return None
    try:
        return reference_resolver(grouping)
    except Exception:  # noqa: BLE001 — an unresolvable reference degrades to no subtraction
        return None


def good_window(grouping: dict[str, Any], n_bins: int) -> tuple[int, int]:
    """Good-bin ``(first, last)`` from the resolved grouping, clamped to *n_bins*."""
    try:
        first = max(0, int(grouping.get("first_good_bin", 0)))
    except (TypeError, ValueError):
        first = 0
    try:
        last = int(grouping.get("last_good_bin", n_bins - 1))
    except (TypeError, ValueError):
        last = n_bins - 1
    return first, last


def build_alpha_request(
    *,
    token: int,
    dataset: MuonDataset,
    groups: dict[int, list[int]],
    forward_gid: int,
    backward_gid: int,
    excluded_detectors: list[int],
    method: str,
    correction_provider: CorrectionProvider | None,
    reference_resolver: ReferenceResolver | None,
    facility: str,
) -> AlphaEstimateRequest:
    """Snapshot the current inputs into an :class:`AlphaEstimateRequest`.

    Built on the GUI thread — resolves the correction grouping, the good-bin
    window and the ``reference_run`` background to plain data so the worker never
    touches widgets or the loader. *dataset* must carry a run with histograms.
    """
    run = dataset.run
    assert run is not None
    grouping = grouping_for_reduction(
        dataset,
        groups=groups,
        forward_gid=forward_gid,
        backward_gid=backward_gid,
        excluded_detectors=excluded_detectors,
        correction_provider=correction_provider,
    )
    n_bins = int(run.histograms[0].n_bins)
    first_good, last_good = good_window(grouping, n_bins)
    return AlphaEstimateRequest(
        token=token,
        histograms=list(run.histograms),
        grouping=grouping,
        method=method,
        first_good_bin=first_good,
        last_good_bin=last_good,
        run_label=str(dataset.run_label),
        resolved_reference=resolve_reference(grouping, reference_resolver),
        facility=str(facility or ""),
    )


class AlphaSectionWidget(QWidget):
    """Inline single-α calibration controls (run picker + method + Estimate).

    :meth:`configure` (re)seeds it from the grouping draft; a successful Estimate
    emits :attr:`alpha_estimated` with a calibrated :class:`AlphaPolicy`. The
    runner is shut down via :meth:`shutdown` from the dialog's teardown.
    """

    #: Emitted with a calibrated ``AlphaPolicy`` when an estimate succeeds.
    alpha_estimated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the run combo, method combo, Estimate button and note."""
        super().__init__(parent)
        self._datasets: list[MuonDataset] = []
        #: Supplies the current {groups, forward_group, backward_group,
        #: excluded_detectors, correction_provider, reference_resolver, facility}
        #: fresh at Estimate time, so a group/pair edit is never stale and the
        #: run-combo selection is preserved across edits.
        self._context_provider: Callable[[], dict[str, Any]] | None = None
        self._estimate_source_run: int | None = None

        self._tasks = TaskRunner(self)
        self._estimate_token = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.addWidget(QLabel("Calibration run"))
        self._run_combo = NoScrollComboBox()
        # Long run labels ("Run 7101 · YBCO TF 200G 100K (Knight shift) · …")
        # must not set the combo's *minimum* width — that forces the whole
        # corrections column into a horizontal scrollbar on narrow panes. Size
        # to a modest minimum instead; the stretch grants whatever width the
        # column actually has, and the popup still shows full labels.
        self._run_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._run_combo.setMinimumContentsLength(18)
        run_row.addWidget(self._run_combo, stretch=1)
        root.addLayout(run_row)

        method_row = QHBoxLayout()
        method_row.setContentsMargins(0, 0, 0, 0)
        method_row.addWidget(QLabel("Method"))
        self._method_combo = NoScrollComboBox()
        for label, key, explanation in ALPHA_METHOD_ITEMS:
            self._method_combo.addItem(label, key)
            self._method_combo.setItemData(
                self._method_combo.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole
            )
        method_row.addWidget(self._method_combo, stretch=1)
        self._estimate_btn = QPushButton("Estimate α")
        self._estimate_btn.setAutoDefault(False)
        self._estimate_btn.setDefault(False)
        self._estimate_btn.clicked.connect(self._on_estimate)
        method_row.addWidget(self._estimate_btn)
        root.addLayout(method_row)

        self._result_label = QLabel("Pick a calibration run and press Estimate α.")
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        root.addWidget(self._result_label)

        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        root.addWidget(self._note_label)

    # -- configuration ---------------------------------------------------

    def configure(
        self,
        *,
        datasets: list[MuonDataset],
        method: str,
        selected_run_number: int | None,
        context_provider: Callable[[], dict[str, Any]],
    ) -> None:
        """(Re)seed the run list and method; the context is pulled at Estimate time.

        ``context_provider`` returns the current ``{groups, forward_group,
        backward_group, excluded_detectors, correction_provider,
        reference_resolver, facility}`` — read fresh on each Estimate, so a later
        group/pair edit is honoured without resetting the run selection.
        """
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._context_provider = context_provider
        self._run_combo.blockSignals(True)
        self._populate_run_combo(selected_run_number)
        self._run_combo.blockSignals(False)
        idx = self._method_combo.findData(str(method or "diamagnetic"))
        self._method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._estimate_btn.setEnabled(bool(self._datasets))

    def shutdown(self) -> None:
        """Tear down the estimate runner (call from the dialog's teardown)."""
        self._tasks.shutdown()

    # -- run combo -------------------------------------------------------

    def _populate_run_combo(self, selected_run_number: int | None) -> None:
        populate_calibration_run_combo(self._run_combo, self._datasets, selected_run_number)

    def _current_dataset(self) -> MuonDataset | None:
        run_number = self._run_combo.currentData()
        if run_number is None:
            return None
        return next((ds for ds in self._datasets if int(ds.run_number) == int(run_number)), None)

    def _current_method(self) -> str:
        return str(self._method_combo.currentData() or "diamagnetic")

    # -- estimate --------------------------------------------------------

    def _on_estimate(self) -> None:
        dataset = self._current_dataset()
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            QMessageBox.warning(self, "Alpha Calibration", "Selected run has no histograms.")
            return
        context = self._context_provider() if self._context_provider is not None else {}
        self._estimate_source_run = int(dataset.run_number)
        self._estimate_token += 1
        request = build_alpha_request(
            token=self._estimate_token,
            dataset=dataset,
            groups=context.get("groups") or {},
            forward_gid=int(context.get("forward_group", 1)),
            backward_gid=int(context.get("backward_group", 2)),
            excluded_detectors=context.get("excluded_detectors") or [],
            method=self._current_method(),
            correction_provider=context.get("correction_provider"),
            reference_resolver=context.get("reference_resolver"),
            facility=str(context.get("facility", "")),
        )
        self._estimate_btn.setEnabled(False)
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._result_label.setText("Computing estimate…")
        self._tasks.start(
            lambda worker: run_alpha_estimate(worker, request),
            on_finished=self._on_estimate_finished,
            on_error=self._on_estimate_error,
        )

    def _on_estimate_finished(self, result: object) -> None:
        self._estimate_btn.setEnabled(True)
        self._result_label.setStyleSheet("")
        if not isinstance(result, AlphaEstimateResult) or result.token != self._estimate_token:
            return  # superseded by a later Estimate click
        estimate = result.estimate
        self._note_label.setStyleSheet(
            f"color: {tokens.WARN if result.note_warn else tokens.TEXT_MUTED};"
        )
        self._note_label.setText(result.note_text)
        if not estimate.ok:
            self._result_label.setText(f"Estimate failed: {estimate.message}")
            return
        method_label = next(
            (label for label, key, _ in ALPHA_METHOD_ITEMS if key == estimate.method),
            estimate.method,
        )
        formatted = format_value_with_uncertainty(estimate.alpha, estimate.alpha_error)
        self._result_label.setText(f"α = {formatted}  ·  {method_label}  ·  run {result.run_label}")
        self.alpha_estimated.emit(
            AlphaPolicy(
                mode="calibrated",
                value=float(estimate.alpha),
                error=estimate.alpha_error,
                method=estimate.method,
                source_run=self._estimate_source_run,
            )
        )

    def _on_estimate_error(self, message: str) -> None:
        self._estimate_btn.setEnabled(True)
        self._result_label.setStyleSheet("")
        self._result_label.setText("Press Estimate α to measure α from this run.")
        QMessageBox.warning(self, "Alpha Calibration", message)
