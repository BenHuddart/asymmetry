"""Inline deadtime-configuration section for the grouping Corrections panel.

The embeddable body of the retired ``DeadtimeDialog`` — mode radios (off / from
file / manual / estimate), the estimate source-run combo, the per-detector table
with Fill-all / Cal, and the max-correction summary — minus the modal shell and
its OK/Cancel. Edits apply live and the unified grouping preview shows the
effect; the widget emits :attr:`changed` and the owning dialog reads
:meth:`get_policy` and re-previews.

Modes match the historical inline controls exactly (see the deadtime study);
Cal calibrates the table from the reference run
(:func:`calibrate_deadtime_from_histograms`), Estimate broadcasts one value from
:func:`estimate_deadtime_from_histograms`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
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
from asymmetry.gui.widgets.no_scroll_spin import NoScrollComboBox, NoScrollDoubleSpinBox

__all__ = ["DeadtimeSectionWidget", "DeadtimeSourceRun", "deadtime_status_text"]

# Above this fraction (100%) the t=0 correction has saturated — the raw
# percentage is no longer a physically meaningful number to show.
_MAX_SANE_CORRECTION_FRACTION = 1.0


@dataclass(frozen=True)
class DeadtimeSourceRun:
    """One candidate source run for calibrate/estimate (the fingerprint's runs)."""

    run_number: int
    label: str
    histograms: list[Histogram]
    good_frames: float = 1.0


def deadtime_status_text(policy: DeadtimePolicy) -> str:
    """Return a compact status-line description for *policy*."""
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


class DeadtimeSectionWidget(QWidget):
    """Mode radios + per-detector table + Cal/Estimate for the Corrections panel.

    Emits :attr:`changed` when the user edits the mode, table, or runs
    Cal/Estimate. :meth:`configure` (re)seeds it from the draft state without
    emitting; :meth:`get_policy` returns the edited :class:`DeadtimePolicy`.
    """

    #: Emitted on any user edit that changes the resulting policy.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the mode radios, estimate row, table and summary."""
        super().__init__(parent)
        self._seeding = False
        self._n_detectors = 0
        self._file_values_us: list[float] = []
        self._manual_values_us: list[float] = []
        self._manual_method = "manual"
        self._estimated_us: float | None = None
        self._source_run: int | None = None
        self._source_runs: list[DeadtimeSourceRun] = []
        self._reference_run_number: int | None = None
        self._peak_rates_per_us: list[float] = []
        self._bin_width_us = 0.0
        self._good_frames = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self._enable_group = QButtonGroup(self)
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for key, label in (
            ("off", "Off"),
            ("file", "From file"),
            ("manual", "Manual"),
            ("estimate", "Estimate from run"),
        ):
            btn = QRadioButton(label)
            self._enable_group.addButton(btn)
            self._mode_buttons[key] = btn
            mode_row.addWidget(btn)
        mode_row.addStretch()
        root.addLayout(mode_row)
        self._enable_group.buttonClicked.connect(self._on_mode_clicked)

        self._file_hint = QLabel("")
        self._file_hint.setWordWrap(True)
        self._file_hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        root.addWidget(self._file_hint)

        estimate_row = QHBoxLayout()
        estimate_row.setContentsMargins(0, 0, 0, 0)
        estimate_row.addWidget(QLabel("Source run"))
        self._source_run_combo = NoScrollComboBox()
        estimate_row.addWidget(self._source_run_combo, stretch=1)
        self._estimate_btn = QPushButton("Estimate")
        self._estimate_btn.setAutoDefault(False)
        self._estimate_btn.setDefault(False)
        self._estimate_btn.clicked.connect(self._on_estimate_clicked)
        estimate_row.addWidget(self._estimate_btn)
        self._estimate_row_widget = QWidget()
        self._estimate_row_widget.setLayout(estimate_row)
        root.addWidget(self._estimate_row_widget)

        table_controls = QHBoxLayout()
        table_controls.setContentsMargins(0, 0, 0, 0)
        table_controls.addWidget(QLabel("Per-detector values (ns)"))
        table_controls.addStretch()
        self._fill_all_spin = NoScrollDoubleSpinBox()
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
        self._table_controls_widget = QWidget()
        self._table_controls_widget.setLayout(table_controls)
        root.addWidget(self._table_controls_widget)

        # file/estimate summary + disclosure share one row so the collapsed
        # states stay short enough to fit the Corrections tab without scrolling.
        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        self._summary_label = QLabel("")
        # One-line summary — word-wrap would inflate the row's height hint and
        # push the collapsed estimate mode past the Corrections-tab viewport.
        self._summary_label.setWordWrap(False)
        summary_row.addWidget(self._summary_label, stretch=1)
        self._disclosure_btn = QToolButton()
        self._disclosure_btn.setText("Show per-detector values")
        self._disclosure_btn.setCheckable(True)
        self._disclosure_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._disclosure_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._disclosure_btn.setStyleSheet("QToolButton { border: none; }")
        self._disclosure_btn.toggled.connect(self._on_disclosure_toggled)
        summary_row.addWidget(self._disclosure_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self._summary_row_widget = QWidget()
        self._summary_row_widget.setLayout(summary_row)
        root.addWidget(self._summary_row_widget)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Detector", "Deadtime (ns)"])
        apply_param_table_style(self._table)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemChanged.connect(self._on_table_item_changed)
        root.addWidget(self._table)

    # -- configuration ---------------------------------------------------

    def configure(
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
    ) -> None:
        """(Re)seed the section from the draft state, without emitting."""
        self._seeding = True
        try:
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

            self._source_run_combo.blockSignals(True)
            self._source_run_combo.clear()
            for run in self._source_runs:
                self._source_run_combo.addItem(run.label, run.run_number)
            self._source_run_combo.blockSignals(False)

            self._set_mode(mode)
            self._on_mode_or_state_changed()
        finally:
            self._seeding = False

    # -- queries ---------------------------------------------------------

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
        n = max(self._n_detectors, 1)
        return DeadtimePolicy(
            mode="estimate",
            estimated_us=self._estimated_us,
            values=[float(self._estimated_us or 0.0)] * n,
            source_run=self._source_run,
        )

    # -- mode plumbing ---------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        button = self._mode_buttons.get(str(mode).strip().lower()) or self._mode_buttons["off"]
        button.setChecked(True)

    def _current_mode(self) -> str:
        for mode, button in self._mode_buttons.items():
            if button.isChecked():
                return mode
        return "off"

    def _file_available(self) -> bool:
        return bool(self._file_values_us) and len(self._file_values_us) >= max(1, self._n_detectors)

    def _display_values(self) -> list[float]:
        if self._current_mode() == "file" and self._file_available():
            return list(self._file_values_us)
        return list(self._manual_values_us)

    def _on_mode_clicked(self) -> None:
        self._on_mode_or_state_changed()
        self._notify()

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
        self._estimate_btn.setEnabled(bool(self._source_runs))
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
        self._apply_row_visibility()

    def _summary_available(self) -> bool:
        """Whether the summary/disclosure have per-detector values to show."""
        mode = self._current_mode()
        if mode == "file":
            return self._file_available()
        if mode == "estimate":
            return self._n_detectors > 0
        return False

    def _apply_row_visibility(self) -> None:
        """Show only the rows the current mode uses; collapse the rest.

        Whole-row ``setVisible`` (not ``setEnabled``) so the scroll-area content
        actually shrinks — off/file/estimate collapsed must fit the tab without
        outer scrolling; only manual's capped table may reach its scroll cap.
        """
        mode = self._current_mode()
        # Estimate carries the extra source-run row, so it alone needs the tight
        # spacing to keep its collapsed height inside the tab viewport.
        self.layout().setSpacing(0 if mode == "estimate" else 4)
        show_summary = mode in ("file", "estimate", "manual")
        self._estimate_row_widget.setVisible(mode == "estimate")
        self._table_controls_widget.setVisible(mode == "manual")
        disclose = self._summary_available()
        self._disclosure_btn.setVisible(disclose)
        self._summary_row_widget.setVisible(show_summary and bool(self._summary_label.text()))
        if mode == "manual":
            self._table.setVisible(True)
        elif disclose:
            self._table.setVisible(self._disclosure_btn.isChecked())
        else:
            self._table.setVisible(False)
        self.updateGeometry()

    def _on_disclosure_toggled(self, checked: bool) -> None:
        # View-only reveal of the read-only table; never touches the policy.
        self._disclosure_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._apply_row_visibility()

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
        self._apply_table_height_cap()

    def _apply_table_height_cap(self) -> None:
        # Cap the table at ~6 data rows; further rows scroll inside the table so
        # a large detector count never forces the Corrections tab to scroll.
        # Fewer rows than the cap shrink to fit rather than reserving blank rows.
        rows = min(max(self._table.rowCount(), 1), 6)
        row_h = self._table.verticalHeader().defaultSectionSize()
        header_h = self._table.horizontalHeader().height()
        frame = self._table.frameWidth()
        self._table.setMaximumHeight(row_h * rows + header_h + 2 * frame)

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
        self._notify()

    def _on_fill_all_clicked(self) -> None:
        value_us = float(self._fill_all_spin.value()) / 1000.0
        self._manual_values_us = [value_us] * max(self._n_detectors, len(self._manual_values_us))
        self._manual_method = "manual"
        self._refresh_table()
        self._refresh_summary()
        self._notify()

    def _on_calibrate_clicked(self) -> None:
        run = self._resolve_reference_source_run()
        if run is None or not run.histograms:
            QMessageBox.warning(
                self,
                "Deadtime Calibration Failed",
                "No reference run with histograms is available to calibrate from.",
            )
            return
        values = calibrate_deadtime_from_histograms(run.histograms, num_good_frames=run.good_frames)
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
        self._on_mode_or_state_changed()
        self._notify()

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
                "The source run did not provide enough valid early-time counts to estimate deadtime.",
            )
            return
        self._estimated_us = tau_us
        self._source_run = int(run_number)
        self._manual_values_us = [tau_us] * max(self._n_detectors, 1)
        self._refresh_table()
        self._refresh_summary()
        self._apply_row_visibility()
        self._notify()

    def _max_correction_fraction(self, values: list[float]) -> float | None:
        """Largest fractional correction any detector receives at the first bin."""
        if not values or not self._peak_rates_per_us or self._bin_width_us <= 0.0:
            return None
        n = min(len(values), len(self._peak_rates_per_us))
        if n == 0:
            return None
        max_fraction = 0.0
        for i in range(n):
            tau_us = values[i]
            rate = self._peak_rates_per_us[i]
            counts_per_frame = rate * self._bin_width_us
            denom = 1.0 - (counts_per_frame * tau_us / (self._bin_width_us * self._good_frames))
            denom = max(denom, 1.0e-6)
            fraction = 1.0 / denom - 1.0
            max_fraction = max(max_fraction, fraction)
        return max_fraction

    def _refresh_summary(self) -> None:
        mode = self._current_mode()
        if mode == "off" or (mode == "file" and not self._file_available()):
            self._summary_label.setText("")
            return
        values = self._display_values()
        fraction = self._max_correction_fraction(values)
        saturated = fraction is not None and fraction > _MAX_SANE_CORRECTION_FRACTION
        if mode == "manual":
            # The manual table already lists every value; report the peak only.
            if saturated:
                self._summary_label.setText(
                    "Deadtime saturates the t=0 correction — value too large."
                )
            else:
                self._summary_label.setText(
                    f"Max correction at t=0: {fraction * 100.0:.1f}%"
                    if fraction is not None
                    else ""
                )
            return
        # file / estimate collapse to one line: mean value × N + peak correction.
        if not values:
            self._summary_label.setText("")
            return
        n = len(values)
        mean_ns = float(np.mean(values)) * 1000.0
        detail = f"{mean_ns:.3f} ns × {n} detector{'' if n == 1 else 's'}"
        if saturated:
            detail += " · deadtime saturates the t=0 correction — value too large"
        elif fraction is not None:
            detail += f" · max correction at t=0: {fraction * 100.0:.1f}%"
        self._summary_label.setText(detail)

    def _notify(self) -> None:
        if not self._seeding:
            self.changed.emit()
