"""Compare a run's saved single fits: ranked, A against B (single-fit plan D2, D7–D9).

The shared :class:`~asymmetry.gui.widgets.model_compare_panel.ModelComparePanel`
with N = 1 run, fed by :func:`~asymmetry.core.fitting.model_comparison.summarise_saved_fits`.
The window holds plain data only — the run's record and its saved fits with
their names — and the host refreshes it whenever those change. Dense curves are
built on demand on a worker (D9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from functools import partial

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.fit_wizard import SelectionMetric
from asymmetry.core.fitting.model_comparison import (
    CandidateSummary,
    RunCurves,
    saved_fit_curves,
    saved_fit_model,
    summarise_saved_fits,
)
from asymmetry.core.representation.base import FitSlot
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import char_width, row_height
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.model_compare_panel import ModelComparePanel

#: The footer button: A becomes the run's open fit, in the Single tab.
OPEN_A_TEXT = "Open A in the Single tab"
#: Under the ranking: why fits over another window carry no Δ.
CAPTION = (
    "Fits with verified matching observations, objectives, and computation revisions are ranked together. "
    "Unknown or different inputs remain listed without a shared Δ; "
    "residuals require a match to the current data."
)


class SavedFitCompareWindow(QDialog):
    """A run's saved single fits, ranked within each window, with A against B."""

    #: The fit id the user wants opened in the Single tab (the panel's A).
    open_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare saved fits")
        self._dataset: MuonDataset | None = None
        self._fits: tuple[tuple[FitSlot, str], ...] = ()
        self._open_id: str | None = None
        #: Curves or a display message, with the slot they belong to.
        self._curves: dict[str, tuple[FitSlot, RunCurves | str]] = {}
        self._building: set[str] = set()
        self._tasks = TaskRunner(self)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        caption = QLabel(CAPTION)
        caption.setWordWrap(True)
        caption.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        header.addWidget(caption, 1)
        header.addWidget(QLabel("Ranking metric:"))
        self._metric = QComboBox()
        self._metric.addItems([metric.value for metric in SelectionMetric])
        self._metric.setCurrentText(SelectionMetric.AICC.value)
        self._metric.currentTextChanged.connect(self._rank)
        header.addWidget(self._metric)
        layout.addLayout(header)

        self._panel = ModelComparePanel(continue_text=OPEN_A_TEXT)
        self._panel.curves_required.connect(self._build_curves)
        self._panel.a_changed.connect(self._update_curve_status)
        self._panel.b_changed.connect(self._update_curve_status)
        self._panel.continue_requested.connect(self.open_requested)
        layout.addWidget(self._panel, 1)
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(self._status)
        self.resize(char_width(150), row_height() * 30)

    @property
    def panel(self) -> ModelComparePanel:
        return self._panel

    def set_fits(
        self,
        dataset: MuonDataset,
        fits: Sequence[tuple[FitSlot, str]],
        open_id: str | None,
    ) -> None:
        """Show *fits* (slot, name) of *dataset*'s run, *open_id* leading as A."""
        # Workers and cached residuals belong to one detached binding snapshot.
        dataset = replace(
            dataset.fit_record(),
            time=dataset.time.copy(),
            asymmetry=dataset.asymmetry.copy(),
            error=dataset.error.copy(),
        )
        self._dataset = dataset
        self._fits = tuple(fits)
        self._open_id = open_id
        self._curves.clear()
        self._building.clear()
        self._panel.set_series([dataset], [dataset.run_label], [math.nan], "")
        self.setWindowTitle(f"Compare saved fits — run {dataset.run_label}")
        self._rank()

    def _summaries(self) -> tuple[CandidateSummary, ...]:
        live = {slot.fit_id: slot for slot, _name in self._fits}
        curves = {
            fit_id: run_curves
            for fit_id, (slot, run_curves) in self._curves.items()
            if live.get(fit_id) is slot and isinstance(run_curves, RunCurves)
        }
        return summarise_saved_fits(
            self._fits,
            self._dataset,
            SelectionMetric.from_value(self._metric.currentText()),
            curves,
            first=self._open_id or "",
        )

    def _rank(self, *_args) -> None:
        """List the fits on the chosen metric, keeping A while it is still listed."""
        summaries = self._summaries()
        keys = [summary.key for summary in summaries]
        a_key = self._panel.a_key()
        if a_key not in keys:
            a_key = self._open_id if self._open_id in keys else None
        self._panel.set_candidates(summaries, self._metric.currentText(), a_key=a_key)
        self._update_curve_status()

    def _update_curve_status(self, *_args) -> None:
        for fit_id in (self._panel.a_key(), self._panel.b_key()):
            entry = self._curves.get(fit_id)
            if entry is not None and isinstance(entry[1], str):
                self._status.setText(entry[1])
                return
        self._status.clear()

    def _build_curves(self, fit_id: str) -> None:
        """Build one fit's dense curve and residuals on a worker (D9)."""
        slot = next((slot for slot, _name in self._fits if slot.fit_id == fit_id), None)
        if slot is None or fit_id in self._building or fit_id in self._curves:
            return
        self._building.add(fit_id)
        dataset = self._dataset

        def _materialise(_worker: object) -> RunCurves | str:
            if saved_fit_model(slot) is None:
                return "That fit was saved without fitted values to draw; its data is shown alone."
            curves = saved_fit_curves(slot, dataset)
            if curves is None:
                return "The saved fit window is outside the model's supported range."
            return curves

        self._tasks.start(
            _materialise,
            on_finished=partial(self._on_curves_built, dataset, slot),
            on_error=partial(self._on_curves_failed, dataset, slot),
        )

    def _on_curves_built(
        self, dataset: MuonDataset, slot: FitSlot, result: RunCurves | str
    ) -> None:
        """Publish one outcome for the binding that requested it."""
        if dataset is not self._dataset:
            return
        self._building.remove(slot.fit_id)
        self._curves[slot.fit_id] = (slot, result)
        if isinstance(result, RunCurves):
            self._panel.refresh_curves(self._summaries())
        self._update_curve_status()

    def _on_curves_failed(self, dataset: MuonDataset, slot: FitSlot, message: str) -> None:
        self._on_curves_built(dataset, slot, f"Could not draw that fit: {message}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._tasks.shutdown()
        super().closeEvent(event)

    def reject(self) -> None:
        self._tasks.shutdown()
        super().reject()
