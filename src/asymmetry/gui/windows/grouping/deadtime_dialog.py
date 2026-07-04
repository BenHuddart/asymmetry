"""Dedicated deadtime-configuration dialog for the grouping profile editor.

Moved out of the main grouping form (:mod:`asymmetry.gui.windows.grouping.dialog`)
so the main window can show a compact status row + "Configure…" button instead of
the full mode/table/summary cluster. The dialog edits nothing but its own local
state; the caller (:class:`~asymmetry.gui.windows.grouping.dialog.GroupingDialog`)
seeds it from the draft profile / preview run and, on Accept, lifts the result
back into the draft via :meth:`get_policy`.

The four modes mirror the historical inline controls exactly:

* ``off`` — no correction.
* ``file`` — use the reference run's own file deadtime values (read-only table);
  only offered when the reference run actually carries them
  (:func:`asymmetry.core.transform.deadtime.has_resolved_deadtime`-style gating,
  reused from the original dialog's ``_reference_has_file_deadtime`` logic).
* ``manual`` — an editable per-detector table, with a "Fill all" convenience and
  a "Cal" button that calibrates the table from the reference run
  (:func:`calibrate_deadtime_from_histograms`).
* ``estimate`` — a source-run combo (runs of the fingerprint) + "Estimate" button
  that fills the table with one broadcast value from
  :func:`estimate_deadtime_from_histograms`.

The summary line reports the peak per-detector correction implied by the current
table, using the same non-paralyzable formula the reduction applies
(:func:`asymmetry.core.transform.deadtime.apply_deadtime_correction`): the
fractional correction at the highest observed early-time rate is
``N_corr/N - 1 = 1/(1 - N·τ/(Δt·n_frames)) - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.project.profiles import DeadtimePolicy
from asymmetry.core.transform import (
    calibrate_deadtime_from_histograms,
    estimate_deadtime_from_histograms,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import apply_param_table_style

__all__ = ["DeadtimeDialog", "DeadtimeSourceRun", "deadtime_status_text"]


@dataclass(frozen=True)
class DeadtimeSourceRun:
    """One candidate source run for calibrate/estimate (the fingerprint's runs)."""

    run_number: int
    label: str
    histograms: list[Histogram]
    good_frames: float = 1.0


def deadtime_status_text(policy: DeadtimePolicy) -> str:
    """Return the main window's compact status-row text for *policy*."""
    if policy.mode == "off":
        return "Deadtime: off"
    if policy.mode == "from_file":
        return "Deadtime: from file"
    if policy.mode == "manual":
        if policy.values:
            mean_us = float(np.mean(policy.values))
            return f"Deadtime: manual ({mean_us * 1000.0:.3f} ns avg)"
        return "Deadtime: manual"
    if policy.mode == "estimate":
        run = f" · run {policy.source_run}" if policy.source_run is not None else ""
        return f"Deadtime: estimated{run}"
    return "Deadtime: off"


class DeadtimeDialog(QDialog):
    """Edit the deadtime-correction policy for the current grouping draft.

    Parameters
    ----------
    n_detectors
        Number of detector histograms in the current (preview) run.
    mode
        Initial mode: ``"off"``/``"file"``/``"manual"``/``"estimate"``.
    file_values_us
        The reference run's own file deadtime values (µs), or ``[]`` when the
        run does not provide them — gates the "from file" mode.
    manual_values_us
        Initial per-detector manual table (µs), one entry per detector.
    manual_method
        Provenance of the manual table: ``"manual"`` (hand-typed) or
        ``"calibrate"`` (from the Cal button).
    estimated_us
        Last estimated single value (µs), if any.
    source_run
        Run number the manual/estimate table was last calibrated/estimated from.
    source_runs
        Candidate runs (of the fingerprint) offered in the estimate-mode combo.
    reference_run_number
        The run number used for the "Cal" button and initial estimate-combo
        selection (the grouping editor's current preview/reference run).
    peak_rates_per_us
        Per-detector peak early-time count rate (counts/µs), used with
        ``bin_width_us``/``good_frames`` to compute the summary line. Empty when
        unavailable (the summary line is then omitted).
    bin_width_us, good_frames
        Passed to the non-paralyzable deadtime formula for the summary line.
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        *,
        n_detectors: int,
        mode: str,
        file_values_us: list[float],
        manual_values_us: list[float],
        manual_method: str,
        estimated_us: float | None,
        source_run: int | None,
        source_runs: list[DeadtimeSourceRun],
        reference_run_number: int | None,
        peak_rates_per_us: list[float],
        bin_width_us: float,
        good_frames: float,
        parent=None,
    ) -> None:
        """Build the dialog; see the class docstring for parameter semantics."""
        super().__init__(parent)
        self.setWindowTitle("Deadtime Correction")
        self.resize(520, 480)

        self._n_detectors = max(0, int(n_detectors))
        self._file_values_us = list(file_values_us)
        self._manual_values_us = (
            list(manual_values_us) if manual_values_us else [0.01] * self._n_detectors
        )
        self._manual_method = manual_method or "manual"
        self._estimated_us = estimated_us
        self._source_run = source_run
        self._source_runs = list(source_runs)
        self._reference_run_number = reference_run_number
        self._peak_rates_per_us = list(peak_rates_per_us)
        self._bin_width_us = float(bin_width_us)
        self._good_frames = float(good_frames) if good_frames else 1.0

        root = QVBoxLayout(self)

        self._enable_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        self._mode_buttons: dict[str, QRadioButton] = {}
        specs = [
            ("off", "Off"),
            ("file", "From file"),
            ("manual", "Manual"),
            ("estimate", "Estimate from run"),
        ]
        for key, label in specs:
            btn = QRadioButton(label)
            self._enable_group.addButton(btn)
            self._mode_buttons[key] = btn
            mode_row.addWidget(btn)
        mode_row.addStretch()
        root.addLayout(mode_row)
        for btn in self._mode_buttons.values():
            btn.toggled.connect(self._on_mode_or_state_changed)

        self._file_hint = QLabel("")
        self._file_hint.setWordWrap(True)
        self._file_hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        root.addWidget(self._file_hint)

        # -- estimate-mode source-run combo -----------------------------------
        estimate_row = QHBoxLayout()
        estimate_row.addWidget(QLabel("Source run"))
        self._source_run_combo = QComboBox()
        for run in self._source_runs:
            self._source_run_combo.addItem(run.label, run.run_number)
        estimate_row.addWidget(self._source_run_combo)
        self._estimate_btn = QPushButton("Estimate")
        self._estimate_btn.setAutoDefault(False)
        self._estimate_btn.setDefault(False)
        self._estimate_btn.clicked.connect(self._on_estimate_clicked)
        estimate_row.addWidget(self._estimate_btn)
        estimate_row.addStretch()
        self._estimate_row_widget = QWidget()
        self._estimate_row_widget.setLayout(estimate_row)
        root.addWidget(self._estimate_row_widget)

        # -- per-detector table -------------------------------------------------
        table_controls = QHBoxLayout()
        table_controls.addWidget(QLabel("Per-detector values (ns)"))
        table_controls.addStretch()
        self._fill_all_spin = QDoubleSpinBox()
        self._fill_all_spin.setDecimals(3)
        self._fill_all_spin.setRange(0.0, 1.0e6)
        self._fill_all_spin.setSuffix(" ns")
        table_controls.addWidget(self._fill_all_spin)
        self._fill_all_btn = QPushButton("Fill all")
        self._fill_all_btn.setAutoDefault(False)
        self._fill_all_btn.setDefault(False)
        self._fill_all_btn.clicked.connect(self._on_fill_all_clicked)
        table_controls.addWidget(self._fill_all_btn)
        self._calibrate_btn = QPushButton("Cal")
        self._calibrate_btn.setAutoDefault(False)
        self._calibrate_btn.setDefault(False)
        self._calibrate_btn.setToolTip(
            "Fit one deadtime value per detector from the reference run and populate the table."
        )
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        table_controls.addWidget(self._calibrate_btn)
        root.addLayout(table_controls)

        self._table = QTableWidget(self._n_detectors, 2)
        self._table.setHorizontalHeaderLabels(["Detector", "Deadtime (ns)"])
        apply_param_table_style(self._table)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemChanged.connect(self._on_table_item_changed)
        root.addWidget(self._table)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        root.addWidget(self._summary_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._set_mode(mode)
        self._refresh_table()
        self._on_mode_or_state_changed()

    # ------------------------------------------------------------------
    # Mode plumbing
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        button = self._mode_buttons.get(str(mode).strip().lower())
        if button is None:
            button = self._mode_buttons["off"]
        button.setChecked(True)

    def _current_mode(self) -> str:
        for mode, button in self._mode_buttons.items():
            if button.isChecked():
                return mode
        return "off"

    def _file_available(self) -> bool:
        return bool(self._file_values_us) and len(self._file_values_us) >= max(1, self._n_detectors)

    def _display_values(self) -> list[float]:
        """Values the table should currently show for the active mode."""
        mode = self._current_mode()
        if mode == "file" and self._file_available():
            return list(self._file_values_us)
        return list(self._manual_values_us)

    def _on_mode_or_state_changed(self) -> None:
        mode = self._current_mode()
        self._mode_buttons["file"].setEnabled(self._file_available())
        editable = mode == "manual"
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers
            if editable
            else QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._fill_all_spin.setEnabled(editable)
        self._fill_all_btn.setEnabled(editable)
        self._calibrate_btn.setEnabled(editable and bool(self._source_runs))
        self._estimate_row_widget.setEnabled(mode == "estimate" and bool(self._source_runs))
        if mode == "file" and not self._file_available():
            self._file_hint.setText("The reference run does not provide file deadtime values.")
            self._file_hint.setVisible(True)
        elif mode == "off":
            self._file_hint.setText("Deadtime correction is disabled.")
            self._file_hint.setVisible(True)
        else:
            self._file_hint.setVisible(False)
        self._refresh_table()
        self._refresh_summary()

    def _refresh_table(self) -> None:
        values = self._display_values()
        self._table.blockSignals(True)
        self._table.setRowCount(max(self._n_detectors, len(values)))
        for row in range(self._table.rowCount()):
            label_item = QTableWidgetItem(f"H{row + 1}")
            label_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 0, label_item)
            value_ns = values[row] * 1000.0 if row < len(values) else 0.0
            value_item = QTableWidgetItem(f"{value_ns:.3f}")
            if self._current_mode() != "manual":
                value_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, 1, value_item)
        self._table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1 or self._current_mode() != "manual":
            return
        row = item.row()
        try:
            value_ns = float(item.text())
        except ValueError:
            self._refresh_table()
            return
        while len(self._manual_values_us) <= row:
            self._manual_values_us.append(0.0)
        self._manual_values_us[row] = max(0.0, value_ns) / 1000.0
        self._manual_method = "manual"
        self._refresh_summary()

    def _on_fill_all_clicked(self) -> None:
        value_us = float(self._fill_all_spin.value()) / 1000.0
        self._manual_values_us = [value_us] * max(self._n_detectors, len(self._manual_values_us))
        self._manual_method = "manual"
        self._refresh_table()
        self._refresh_summary()

    def _on_calibrate_clicked(self) -> None:
        run = self._resolve_reference_source_run()
        if run is None or not run.histograms:
            QMessageBox.warning(
                self,
                "Deadtime Calibration Failed",
                "No reference run with histograms is available to calibrate from.",
            )
            return
        values = calibrate_deadtime_from_histograms(
            run.histograms,
            num_good_frames=run.good_frames,
        )
        if not values:
            QMessageBox.warning(
                self,
                "Deadtime Calibration Failed",
                "The reference run did not provide enough valid early-time "
                "counts to calibrate per-detector deadtime values.",
            )
            return
        self._manual_values_us = list(values)
        self._manual_method = "calibrate"
        self._source_run = run.run_number
        self._set_mode("manual")
        self._refresh_table()
        self._refresh_summary()

    def _resolve_reference_source_run(self) -> DeadtimeSourceRun | None:
        for run in self._source_runs:
            if run.run_number == self._reference_run_number:
                return run
        return self._source_runs[0] if self._source_runs else None

    def _on_estimate_clicked(self) -> None:
        run_number = self._source_run_combo.currentData()
        run = next((r for r in self._source_runs if r.run_number == run_number), None)
        if run is None or not run.histograms:
            QMessageBox.warning(
                self,
                "Deadtime Estimate Failed",
                "No source run with histograms is available to estimate from.",
            )
            return
        tau_us = estimate_deadtime_from_histograms(run.histograms, num_good_frames=run.good_frames)
        if tau_us is None or tau_us <= 0.0:
            QMessageBox.warning(
                self,
                "Deadtime Estimate Failed",
                "The source run did not provide enough valid early-time counts "
                "to estimate deadtime.",
            )
            return
        self._estimated_us = tau_us
        self._source_run = int(run_number)
        self._manual_values_us = [tau_us] * max(self._n_detectors, 1)
        self._refresh_table()
        self._refresh_summary()

    # ------------------------------------------------------------------
    # Summary line
    # ------------------------------------------------------------------

    def _refresh_summary(self) -> None:
        mode = self._current_mode()
        if mode == "off":
            self._summary_label.setText("")
            return
        values = self._display_values()
        if not values or not self._peak_rates_per_us or self._bin_width_us <= 0.0:
            self._summary_label.setText("")
            return
        n = min(len(values), len(self._peak_rates_per_us))
        if n == 0:
            self._summary_label.setText("")
            return
        max_fraction = 0.0
        for i in range(n):
            tau_us = values[i]
            rate = self._peak_rates_per_us[i]
            counts_per_frame = rate * self._bin_width_us
            denom = 1.0 - (counts_per_frame * tau_us / (self._bin_width_us * self._good_frames))
            denom = max(denom, 1.0e-6)
            fraction = 1.0 / denom - 1.0
            max_fraction = max(max_fraction, fraction)
        self._summary_label.setText(f"Max correction at t=0: {max_fraction * 100.0:.1f}%")

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        mode = self._current_mode()
        if mode == "manual" and (
            not self._manual_values_us or any(v <= 0.0 for v in self._manual_values_us)
        ):
            QMessageBox.warning(
                self,
                "Invalid Deadtime",
                "Manual deadtime values must be greater than zero for every detector.",
            )
            return
        if mode == "file" and not self._file_available():
            QMessageBox.warning(
                self,
                "Deadtime Unavailable",
                "The reference run does not provide file deadtime values.",
            )
            return
        if mode == "estimate" and self._estimated_us is None:
            QMessageBox.warning(
                self,
                "Deadtime Estimate Missing",
                "Press Estimate to compute a deadtime value before accepting.",
            )
            return
        self.accept()

    def get_policy(self) -> DeadtimePolicy:
        """Return the edited :class:`DeadtimePolicy`."""
        mode = self._current_mode()
        if mode == "off":
            return DeadtimePolicy(mode="off")
        if mode == "file":
            return DeadtimePolicy(mode="from_file")
        if mode == "manual":
            return DeadtimePolicy(
                mode="manual",
                values=list(self._manual_values_us),
                manual_us=self._manual_values_us[0] if self._manual_values_us else None,
                method=self._manual_method,
                source_run=self._source_run,
            )
        # estimate
        n = max(self._n_detectors, 1)
        return DeadtimePolicy(
            mode="estimate",
            estimated_us=self._estimated_us,
            values=[float(self._estimated_us or 0.0)] * n,
            source_run=self._source_run,
        )
