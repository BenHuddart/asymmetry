"""Inline β (intrinsic-asymmetry balance) controls for the Corrections panel.

β = A₀,B/A₀,F corrects the asymmetry for the two detector groups' *intrinsic
asymmetries* differing (solid-angle / absorption effects that scale the
observable amplitude rather than the count rate), the musrfit asymmetry-fit
(fit type 2) companion to α. It enters the reduction as
``A = (F − αB)/(βF + αB)``; β = 1 is the standard formula.

The widget hosts a fixed-value β entry **and** a "Measure from run" area that
mirrors the α card's Flow A: a calibration-run picker (weak-TF candidates
highlighted), a protocol combo and an **Estimate** button. The estimate runs on
a :class:`~asymmetry.gui.tasks.TaskRunner` worker thread over the *count-domain*
forward/backward fit (:func:`~asymmetry.core.fitting.beta_calibration.estimate_beta_detailed`);
on success the section shows β with the fitted α as a consistency readout, and
**Apply β** emits :attr:`beta_estimated` with a calibrated
:class:`~asymmetry.core.project.profiles.BetaPolicy`. β cannot be measured from
count ratios (it is invisible to count totals) — only from the two groups'
fitted asymmetry amplitudes, which is exactly what the count-domain fit recovers
(see ``docs/porting/beta-correction/``).

Because β is a count fit that also floats α, an **Also update α** affordance can
additionally emit an :class:`~asymmetry.core.project.profiles.AlphaPolicy`
through the owning dialog's existing α apply path; that policy is stamped
``method="count_fit"`` for *both* protocols (Protocol B is still a count-fit
pair, and ``"single_histogram"`` is not part of α's method vocabulary).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.beta_calibration import BetaEstimate, estimate_beta_detailed
from asymmetry.core.project.profiles import AlphaPolicy, BetaPolicy
from asymmetry.core.utils.constants import GAUSS_TO_TESLA
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import make_warning_banner
from asymmetry.gui.tasks import TaskCancelledError, TaskRunner, TaskWorker
from asymmetry.gui.widgets.no_scroll_spin import NoScrollComboBox, NoScrollDoubleSpinBox
from asymmetry.gui.windows.grouping.alpha_section import (
    grouping_for_reduction,
    populate_calibration_run_combo,
)

__all__ = [
    "BETA_PROTOCOL_ITEMS",
    "BetaEstimateRequest",
    "BetaEstimateResult",
    "BetaSectionWidget",
    "beta_status_text",
    "build_beta_request",
    "format_value_with_pm",
    "run_beta_estimate",
]

#: β-estimation protocols offered by the Estimate control: combo label, method
#: id (as persisted in ``beta_method``), and a one-line tooltip explanation.
BETA_PROTOCOL_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "Count fit (recommended)",
        "count_fit",
        "One simultaneous forward+backward count fit that shares the physics "
        "and floats a single β on the backward amplitude — the endorsed protocol "
        "and the only one that reports an α–β correlation.",
    ),
    (
        "Single-histogram ratio",
        "single_histogram",
        "Two independent single-histogram fits; β = Â₀,b/Â₀,f by ratio. "
        "Statistically weaker (the physics is not shared) but a useful "
        "independent cross-check of the count fit.",
    ),
)

#: β̂ outside this range is flagged as suspicious in the result row — typical
#: instruments sit at 0.8–1.0 (docs/porting/beta-correction/estimator-plan.md,
#: Decision 3).
_BETA_SANITY_RANGE = (0.5, 1.5)

#: α consistency: warn when the fitted α disagrees with the applied α by more
#: than this many of the estimate's α errors (Decision 1).
_ALPHA_CONSISTENCY_SIGMA = 3.0


def beta_status_text(value: float) -> str:
    """One-line β summary for the pipeline chip / card header."""
    return f"β = {float(value):.4f}"


def format_value_with_pm(value: float, error: float | None) -> str:
    """Format ``0.8732 ± 0.0041`` (or just ``0.8732`` when no usable error)."""
    if error is None or not math.isfinite(error) or error <= 0.0:
        return f"{value:.4f}"
    return f"{value:.4f} ± {error:.4f}"


@dataclass(frozen=True)
class BetaEstimateRequest:
    """An immutable snapshot of what to estimate, built on the GUI thread.

    ``dataset`` is the calibration run with the *draft* grouping (groups, F/B
    pair, exclusions and the resolved deadtime/background corrections) merged
    onto its ``run.grouping``, so β is measured under the same corrections the
    reduction applies — the basis the owning dialog stamps into
    ``beta_correction_digest``. Everything here is plain (non-Qt) data so the
    worker runs entirely off the GUI thread.
    """

    token: int
    dataset: MuonDataset
    forward_group: int
    backward_group: int
    method: str
    field_tesla: float | None
    run_label: str


@dataclass(frozen=True)
class BetaEstimateResult:
    """The estimate marshalled back to the GUI thread, tagged with its token."""

    token: int
    estimate: BetaEstimate
    run_label: str
    method: str


def build_beta_request(
    *,
    token: int,
    dataset: MuonDataset,
    groups: dict[int, list[int]],
    forward_gid: int,
    backward_gid: int,
    excluded_detectors: list[int],
    method: str,
    correction_provider: Callable[[MuonDataset], dict[str, Any]] | None,
) -> BetaEstimateRequest:
    """Snapshot the current inputs into a :class:`BetaEstimateRequest`.

    Built on the GUI thread: the draft grouping (groups, F/B pair, exclusions and
    the resolved deadtime/background corrections) is merged onto the calibration
    run so :func:`estimate_beta_detailed` resolves the draft's forward/backward
    groups and fits under the draft's corrections — not the run's stored
    grouping. *dataset* must carry a run with histograms.
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
    merged_dataset = replace(dataset, run=replace(run, grouping=grouping))
    field_gauss = float(run.field)
    field_tesla = (
        field_gauss * GAUSS_TO_TESLA if math.isfinite(field_gauss) and field_gauss != 0.0 else None
    )
    return BetaEstimateRequest(
        token=token,
        dataset=merged_dataset,
        forward_group=int(forward_gid),
        backward_group=int(backward_gid),
        method=method,
        field_tesla=field_tesla,
        run_label=str(dataset.run_label),
    )


def run_beta_estimate(worker: TaskWorker, request: BetaEstimateRequest) -> BetaEstimateResult:
    """Estimate β off the GUI thread via the count-domain calibration fit."""
    if worker.is_cancelled():
        raise TaskCancelledError
    estimate = estimate_beta_detailed(
        request.dataset,
        request.forward_group,
        request.backward_group,
        method=request.method,
        field_tesla=request.field_tesla,
        cancel_callback=worker.is_cancelled,
    )
    return BetaEstimateResult(
        token=request.token,
        estimate=estimate,
        run_label=request.run_label,
        method=request.method,
    )


class BetaSectionWidget(QWidget):
    """Fixed-value β entry plus an inline "Measure from run" calibration area.

    The dialog reads :meth:`value` when building the grouping payload (emitting
    the ``beta`` key only when ≠ 1, so a default payload stays byte-identical to
    a pre-β one) and seeds it back with :meth:`set_value`. :meth:`configure`
    (re)seeds the calibration-run list; a successful Estimate followed by
    **Apply β** emits :attr:`beta_estimated` with a calibrated
    :class:`BetaPolicy`, and — when **Also update α** is ticked — :attr:`alpha_estimated`
    with a ``count_fit`` :class:`AlphaPolicy`. The runner is shut down via
    :meth:`shutdown` from the dialog's teardown.
    """

    #: Emitted when the β value changes (spin edit or programmatic set).
    changed = Signal()
    #: Emitted with a calibrated ``BetaPolicy`` when **Apply β** is pressed.
    beta_estimated = Signal(object)
    #: Emitted with a ``count_fit`` ``AlphaPolicy`` when **Apply β** is pressed
    #: with **Also update α** ticked (routed through the dialog's α apply path).
    alpha_estimated = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the value row, the explanation, and the measure-from-run area."""
        super().__init__(parent)
        self._datasets: list[MuonDataset] = []
        #: Supplies the current {groups, forward_group, backward_group,
        #: excluded_detectors, correction_provider, …} fresh at Estimate time.
        self._context_provider: Callable[[], dict[str, Any]] | None = None
        #: Supplies the card's currently applied α for the consistency readout.
        self._current_alpha_provider: Callable[[], float] | None = None

        self._tasks = TaskRunner(self)
        self._estimate_token = 0
        self._estimate_source_run: int | None = None
        self._last_estimate: BetaEstimate | None = None
        self._last_source_run: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._beta_spin = NoScrollDoubleSpinBox()
        self._beta_spin.setDecimals(6)
        # Same positive range as the α spin: β is a ratio of asymmetry
        # amplitudes and is meaningless at or below zero.
        self._beta_spin.setRange(0.01, 1000.0)
        self._beta_spin.setValue(1.0)
        self._beta_spin.valueChanged.connect(self.changed)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(12)
        form.addRow("β value", self._beta_spin)
        layout.addLayout(form)

        hint = QLabel(
            "β = A₀,b/A₀,f corrects for the two groups' intrinsic asymmetries "
            "differing (musrfit asymmetry fit). Measure it from a weak-TF "
            "calibration run below, or leave at 1."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(hint)

        # ── Measure from run ────────────────────────────────────────────────
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.addWidget(QLabel("Calibration run"))
        self._run_combo = NoScrollComboBox()
        # A long run label must not set the combo's *minimum* width — that forces
        # the whole corrections column into a horizontal scrollbar on narrow
        # panes (PR #274). Size to a modest minimum instead; the stretch grants
        # whatever width the column has, and the popup still shows full labels.
        self._run_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._run_combo.setMinimumContentsLength(18)
        run_row.addWidget(self._run_combo, stretch=1)
        layout.addLayout(run_row)

        protocol_row = QHBoxLayout()
        protocol_row.setContentsMargins(0, 0, 0, 0)
        protocol_row.addWidget(QLabel("Protocol"))
        self._protocol_combo = NoScrollComboBox()
        for label, key, explanation in BETA_PROTOCOL_ITEMS:
            self._protocol_combo.addItem(label, key)
            self._protocol_combo.setItemData(
                self._protocol_combo.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole
            )
        protocol_row.addWidget(self._protocol_combo, stretch=1)
        self._estimate_btn = QPushButton("Estimate β")
        self._estimate_btn.setAutoDefault(False)
        self._estimate_btn.setDefault(False)
        self._estimate_btn.clicked.connect(self._on_estimate)
        protocol_row.addWidget(self._estimate_btn)
        layout.addLayout(protocol_row)

        self._result_label = QLabel("Pick a calibration run and press Estimate β.")
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(self._result_label)

        self._note_label = QLabel("")
        self._note_label.setWordWrap(True)
        self._note_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(self._note_label)

        apply_row = QHBoxLayout()
        apply_row.setContentsMargins(0, 0, 0, 0)
        self._apply_btn = QPushButton("Apply β")
        self._apply_btn.setAutoDefault(False)
        self._apply_btn.setDefault(False)
        self._apply_btn.clicked.connect(self._on_apply)
        apply_row.addWidget(self._apply_btn)
        self._also_alpha_check = QCheckBox("Also update α")
        self._also_alpha_check.setToolTip(
            "Additionally apply the α the count fit measured alongside β."
        )
        apply_row.addWidget(self._also_alpha_check)
        apply_row.addStretch()
        layout.addLayout(apply_row)

        self._stale_banner = make_warning_banner("", severity="warn")
        self._stale_banner.setVisible(False)
        layout.addWidget(self._stale_banner)

        self._set_apply_enabled(False)

    # -- configuration ---------------------------------------------------

    def configure(
        self,
        *,
        datasets: list[MuonDataset],
        selected_run_number: int | None,
        context_provider: Callable[[], dict[str, Any]],
        current_alpha_provider: Callable[[], float],
    ) -> None:
        """(Re)seed the calibration-run list; the context is pulled at Estimate time.

        ``context_provider`` returns the current group pair + correction context
        (as the α section's does); ``current_alpha_provider`` returns the card's
        currently applied α for the consistency readout — both read fresh on each
        Estimate, so a later group/pair edit is honoured without resetting the
        run selection.
        """
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._context_provider = context_provider
        self._current_alpha_provider = current_alpha_provider
        self._run_combo.blockSignals(True)
        populate_calibration_run_combo(self._run_combo, self._datasets, selected_run_number)
        self._run_combo.blockSignals(False)
        self._estimate_btn.setEnabled(bool(self._datasets))

    def shutdown(self) -> None:
        """Tear down the estimate runner (call from the dialog's teardown)."""
        self._tasks.shutdown()

    # -- value plumbing ---------------------------------------------------

    def value(self) -> float:
        """The current β value (always positive — the spin enforces the range)."""
        return float(self._beta_spin.value())

    def set_value(self, value: float) -> None:
        """Seed the spin, mapping degenerate values to the 1.0 default."""
        try:
            beta = float(value)
        except (TypeError, ValueError):
            beta = 1.0
        if not math.isfinite(beta) or beta <= 0.0:
            beta = 1.0
        self._beta_spin.setValue(beta)

    def is_active(self) -> bool:
        """Whether β departs from the do-nothing default."""
        return abs(self.value() - 1.0) > 1e-9

    def set_stale_message(self, text: str | None) -> None:
        """Show (``text``) or hide (``None``) the staleness banner in the card."""
        if text:
            self._stale_banner.setText(str(text))
            self._stale_banner.setVisible(True)
        else:
            self._stale_banner.setVisible(False)

    # -- estimate --------------------------------------------------------

    def _current_dataset(self) -> MuonDataset | None:
        run_number = self._run_combo.currentData()
        if run_number is None:
            return None
        return next((ds for ds in self._datasets if int(ds.run_number) == int(run_number)), None)

    def _current_method(self) -> str:
        return str(self._protocol_combo.currentData() or "count_fit")

    def _set_apply_enabled(self, enabled: bool) -> None:
        self._apply_btn.setEnabled(enabled)
        self._also_alpha_check.setEnabled(enabled)

    def _on_estimate(self) -> None:
        dataset = self._current_dataset()
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            QMessageBox.warning(self, "Beta Calibration", "Selected run has no histograms.")
            return
        context = self._context_provider() if self._context_provider is not None else {}
        self._estimate_source_run = int(dataset.run_number)
        self._estimate_token += 1
        request = build_beta_request(
            token=self._estimate_token,
            dataset=dataset,
            groups=context.get("groups") or {},
            forward_gid=int(context.get("forward_group", 1)),
            backward_gid=int(context.get("backward_group", 2)),
            excluded_detectors=context.get("excluded_detectors") or [],
            method=self._current_method(),
            correction_provider=context.get("correction_provider"),
        )
        self._last_estimate = None
        self._set_apply_enabled(False)
        self._estimate_btn.setEnabled(False)
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._result_label.setText("Computing estimate…")
        self._note_label.setText("")
        self._tasks.start(
            lambda worker: run_beta_estimate(worker, request),
            on_finished=self._on_estimate_finished,
            on_error=self._on_estimate_error,
        )

    def _on_estimate_finished(self, result: object) -> None:
        self._estimate_btn.setEnabled(True)
        if not isinstance(result, BetaEstimateResult) or result.token != self._estimate_token:
            return  # superseded by a later Estimate click
        estimate = result.estimate
        if not estimate.ok:
            self._last_estimate = None
            self._set_apply_enabled(False)
            self._result_label.setStyleSheet(f"color: {tokens.WARN};")
            self._result_label.setTextFormat(Qt.TextFormat.PlainText)
            self._result_label.setText(f"Estimate failed: {estimate.message}")
            self._note_label.setText("")
            return
        self._last_estimate = estimate
        self._last_source_run = self._estimate_source_run
        self._show_success(estimate, result)
        self._set_apply_enabled(True)

    def _show_success(self, estimate: BetaEstimate, result: BetaEstimateResult) -> None:
        """Render the β + fitted-α readout, warn-tinting the flagged parts."""
        lo, hi = _BETA_SANITY_RANGE
        beta_suspicious = not (lo <= float(estimate.beta) <= hi)
        alpha_disagrees = self._alpha_disagrees(estimate)

        beta_color = tokens.WARN if beta_suspicious else tokens.TEXT
        alpha_color = tokens.WARN if alpha_disagrees else tokens.TEXT_MUTED
        method_label = next(
            (label for label, key, _ in BETA_PROTOCOL_ITEMS if key == estimate.method),
            estimate.method,
        )
        run_text = f" · run {result.run_label}"
        self._result_label.setStyleSheet("")
        self._result_label.setTextFormat(Qt.TextFormat.RichText)
        self._result_label.setText(
            f'β = <span style="color: {beta_color};">'
            f"{format_value_with_pm(estimate.beta, estimate.beta_error)}</span>"
            f' · fitted α = <span style="color: {alpha_color};">'
            f"{format_value_with_pm(estimate.alpha, estimate.alpha_error)}</span>"
            f" · {method_label}{run_text}"
        )

        hints: list[str] = []
        if beta_suspicious:
            hints.append("β is outside the typical 0.5–1.5 range — check the fit.")
        if alpha_disagrees:
            hints.append("Fitted α disagrees with the applied α (> 3σ).")
        self._note_label.setStyleSheet(f"color: {tokens.WARN if hints else tokens.TEXT_MUTED};")
        self._note_label.setText("  ".join(hints))

    def _alpha_disagrees(self, estimate: BetaEstimate) -> bool:
        """Whether the fitted α is > 3σ from the card's currently applied α."""
        if self._current_alpha_provider is None:
            return False
        error = estimate.alpha_error
        if error is None or not math.isfinite(error) or error <= 0.0:
            return False
        applied = float(self._current_alpha_provider())
        return abs(float(estimate.alpha) - applied) > _ALPHA_CONSISTENCY_SIGMA * float(error)

    def _on_apply(self) -> None:
        """Emit the calibrated β (and, if ticked, the count-fit α)."""
        estimate = self._last_estimate
        if estimate is None:
            return
        self.beta_estimated.emit(
            BetaPolicy(
                mode="calibrated",
                value=float(estimate.beta),
                error=estimate.beta_error,
                method=str(estimate.method),
                source_run=self._last_source_run,
            )
        )
        if self._also_alpha_check.isChecked():
            # Protocol B is still a count-fit pair, and "single_histogram" is not
            # in α's method vocabulary, so both protocols stamp "count_fit".
            self.alpha_estimated.emit(
                AlphaPolicy(
                    mode="calibrated",
                    value=float(estimate.alpha),
                    error=estimate.alpha_error,
                    method="count_fit",
                    source_run=self._last_source_run,
                )
            )

    def _on_estimate_error(self, message: str) -> None:
        self._estimate_btn.setEnabled(True)
        self._set_apply_enabled(False)
        self._last_estimate = None
        self._result_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._result_label.setTextFormat(Qt.TextFormat.PlainText)
        self._result_label.setText("Press Estimate β to measure β from this run.")
        QMessageBox.warning(self, "Beta Calibration", message)
