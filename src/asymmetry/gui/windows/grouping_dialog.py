"""WiMDA-style shared grouping dialog with alpha estimation and .grp I/O.

The dialog edits detector grouping once and applies it across multiple datasets
in the active project. Grouping definitions can be saved to and loaded from
``.grp`` files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.instrument import detect_instrument, get_instrument_layout
from asymmetry.core.transform import apply_grouping
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.utils.constants import PeriodMode


class GroupingDialog(QDialog):
    """Edit forward/backward grouping for multiple datasets.

    Parameters
    ----------
    datasets
        Datasets available in the active project. Datasets without raw
        histograms are ignored for grouping operations.
    selected_run_number
        Optional run number used as initial reference dataset.
    selected_run_numbers
        Optional run numbers to pre-select in the dataset tick-list. When not
        provided, all datasets are selected by default.
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        selected_run_number: int | None = None,
        selected_run_numbers: list[int] | None = None,
        parent=None,
    ) -> None:
        """Create a shared grouping dialog for project datasets."""
        super().__init__(parent)
        self._datasets = [
            ds for ds in datasets if ds.run is not None
        ]
        self._selected_run_number = selected_run_number
        self._selected_run_numbers = (
            {int(v) for v in selected_run_numbers}
            if selected_run_numbers is not None
            else None
        )

        self.setWindowTitle("Grouping")
        self.resize(860, 560)

        if not self._datasets:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("No runs are available in the active project."))
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)
            return

        self._reference_dataset = self._choose_reference_dataset()
        self._run = self._reference_dataset.run
        assert self._run is not None

        self._groups = self._load_groups(self._run)
        self._group_names: dict[int, str] = self._load_group_names(self._run)

        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(f"Datasets: {len(self._datasets)}"))

        self._reference_combo = QComboBox()
        for ds in self._datasets:
            self._reference_combo.addItem(ds.run_label, int(ds.run_number))
        self._set_combo_to_run(self._reference_combo, int(self._reference_dataset.run_number))
        self._reference_combo.currentIndexChanged.connect(self._on_reference_dataset_changed)
        top_row.addWidget(QLabel("Reference run"))
        top_row.addWidget(self._reference_combo)
        top_row.addStretch()
        root.addLayout(top_row)

        self._dataset_list = QListWidget()
        for ds in self._datasets:
            item = QListWidgetItem(ds.run_label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            run_number = int(ds.run_number)
            item.setData(Qt.ItemDataRole.UserRole, run_number)
            if self._selected_run_numbers is None:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                state = (
                    Qt.CheckState.Checked
                    if run_number in self._selected_run_numbers
                    else Qt.CheckState.Unchecked
                )
                item.setCheckState(state)
            self._dataset_list.addItem(item)
        self._dataset_list.setMaximumHeight(150)

        dataset_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.setAutoDefault(False)
        select_all_btn.setDefault(False)
        select_all_btn.clicked.connect(self._select_all_datasets)
        select_reference_btn = QPushButton("Select Reference Run")
        select_reference_btn.setAutoDefault(False)
        select_reference_btn.setDefault(False)
        select_reference_btn.clicked.connect(self._select_reference_dataset)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.setAutoDefault(False)
        deselect_all_btn.setDefault(False)
        deselect_all_btn.clicked.connect(self._deselect_all_datasets)
        dataset_buttons.addWidget(select_all_btn)
        dataset_buttons.addWidget(select_reference_btn)
        dataset_buttons.addWidget(deselect_all_btn)
        dataset_buttons.addStretch()
        root.addLayout(dataset_buttons)
        root.addWidget(self._dataset_list)

        self._group_table = QTableWidget(0, 3)
        self._group_table.setHorizontalHeaderLabels(["Group", "Name", "Detector Indices (1-based)"])
        self._group_table.setMaximumHeight(140)
        root.addWidget(self._group_table)
        self._populate_group_table()

        detector_layout_btn = QPushButton("Detector Layout\u2026")
        detector_layout_btn.setAutoDefault(False)
        detector_layout_btn.setDefault(False)
        detector_layout_btn.setToolTip(
            "Open the interactive detector schematic editor to assign "
            "detectors to groups visually."
        )
        detector_layout_btn.clicked.connect(self._on_detector_layout)
        root.addWidget(detector_layout_btn)

        form = QFormLayout()
        self._forward_combo = QComboBox()
        self._backward_combo = QComboBox()
        self._forward_combo.setMinimumWidth(220)
        self._backward_combo.setMinimumWidth(220)
        self._forward_combo.setMinimumContentsLength(18)
        self._backward_combo.setMinimumContentsLength(18)
        for gid in sorted(self._groups):
            text = str(gid)
            self._forward_combo.addItem(text, gid)
            self._backward_combo.addItem(text, gid)

        grouping = self._run.grouping or {}
        self._grouping_preset_name: str | None = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._detector_layout_instrument_name: str | None = (
            str(grouping.get("instrument")).strip()
            if grouping.get("instrument")
            else None
        )
        self._enforce_source_bunching = self._reference_is_wim_run()
        self._source_bunching_factor = self._read_source_bunching_factor(grouping)
        self._set_combo_to_group(self._forward_combo, int(grouping.get("forward_group", 1)))
        self._set_combo_to_group(self._backward_combo, int(grouping.get("backward_group", 2)))

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setDecimals(6)
        self._alpha_spin.setRange(0.01, 1000.0)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))

        self._first_good_spin = QSpinBox()
        max_bin = self._max_bin_index_for_reference_dataset()
        self._first_good_spin.setRange(0, max_bin)
        self._first_good_spin.setValue(int(grouping.get("first_good_bin", 0)))

        self._last_good_spin = QSpinBox()
        self._last_good_spin.setRange(0, max_bin)
        self._last_good_spin.setValue(int(grouping.get("last_good_bin", max_bin)))

        self._bunch_spin = QSpinBox()
        self._bunch_spin.setRange(1, 10000)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        if self._enforce_source_bunching and requested_bunching < self._source_bunching_factor:
            requested_bunching = self._source_bunching_factor
        self._bunch_spin.setValue(requested_bunching)
        self._bunch_spin.setMaximumWidth(100)
        self._bunch_source_hint = QLabel()
        self._update_bunching_ui_hints()

        self._deadtime_checkbox = QCheckBox("Enable Deadtime Correction")
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))

        self._period_mode_label = QLabel("RG Mode")
        self._period_mode_group = QButtonGroup(self)
        self._period_mode_buttons: dict[str, QRadioButton] = {}
        self._period_mode_widget = QWidget()
        period_layout = QHBoxLayout(self._period_mode_widget)
        period_layout.setContentsMargins(0, 0, 0, 0)
        period_layout.setSpacing(10)

        period_specs = [
            ("Red", str(PeriodMode.RED), "#c00000"),
            ("Green", str(PeriodMode.GREEN), "#008000"),
            ("G minus R", str(PeriodMode.GREEN_MINUS_RED), "#0000c0"),
            ("G plus R", str(PeriodMode.GREEN_PLUS_RED), "#800080"),
        ]
        for idx, (label, mode_key, color) in enumerate(period_specs):
            btn = QRadioButton(label)
            btn.setStyleSheet(f"color: {color};")
            self._period_mode_group.addButton(btn, idx)
            self._period_mode_buttons[mode_key] = btn
            period_layout.addWidget(btn)
        period_layout.addStretch()
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))

        estimate_btn = QPushButton("Estimate α")
        estimate_btn.setAutoDefault(False)
        estimate_btn.setDefault(False)
        estimate_btn.clicked.connect(self._estimate_alpha)

        form.addRow("Forward Group", self._forward_combo)
        form.addRow("Backward Group", self._backward_combo)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self._alpha_spin)
        alpha_row.addWidget(estimate_btn)
        form.addRow("Alpha", alpha_row)

        form.addRow("First Good Bin", self._first_good_spin)
        form.addRow("Last Good Bin", self._last_good_spin)
        form.addRow("Bunching Factor", self._bunch_spin)
        form.addRow("Bunching Rules", self._bunch_source_hint)
        form.addRow("Deadtime", self._deadtime_checkbox)
        form.addRow(self._period_mode_label, self._period_mode_widget)
        root.addLayout(form)

        file_buttons = QHBoxLayout()
        load_btn = QPushButton("Load .grp")
        load_btn.setAutoDefault(False)
        load_btn.setDefault(False)
        load_btn.clicked.connect(self._load_grp_file)
        save_btn = QPushButton("Save .grp")
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(self._save_grp_file)
        file_buttons.addWidget(load_btn)
        file_buttons.addWidget(save_btn)
        file_buttons.addStretch()
        root.addLayout(file_buttons)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setAutoDefault(False)
        apply_btn.setDefault(False)
        apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        root.addLayout(buttons)
        self._update_period_mode_visibility()

    def _choose_reference_dataset(self) -> MuonDataset:
        """Return preferred reference dataset for initial grouping values."""
        if self._selected_run_number is not None:
            for ds in self._datasets:
                if int(ds.run_number) == int(self._selected_run_number):
                    return ds
        return self._datasets[0]

    def _load_group_names(self, run) -> dict[int, str]:
        """Load group names from run metadata, returning an empty dict if absent."""
        grouping = run.grouping or {}
        raw = grouping.get("group_names")
        if not isinstance(raw, dict):
            return {}
        result: dict[int, str] = {}
        for key, val in raw.items():
            try:
                gid = int(key)
                result[gid] = str(val)
            except (TypeError, ValueError):
                continue
        return result

    def _load_groups(self, run) -> dict[int, list[int]]:
        """Load detector groups from run metadata or default half/half groups."""
        grouping = run.grouping or {}
        groups_raw = grouping.get("groups")

        groups: dict[int, list[int]] = {}
        if isinstance(groups_raw, dict):
            for key, values in groups_raw.items():
                try:
                    gid = int(key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(values, list):
                    continue
                idxs = []
                for val in values:
                    try:
                        # Persisted groups are 1-based in project/run metadata.
                        idxs.append(max(0, int(val) - 1))
                    except (TypeError, ValueError):
                        continue
                if idxs:
                    groups[gid] = sorted(set(idxs))

        if len(groups) >= 2:
            return groups

        n = len(run.histograms)
        if n <= 0:
            try:
                n = int(run.metadata.get("histograms_count", 0))
            except (TypeError, ValueError):
                n = 0
        if n <= 0:
            n = 2
        split = max(1, n // 2)
        return {1: list(range(0, split)), 2: list(range(split, n)) or list(range(0, n))}

    def _max_bin_index_for_reference_dataset(self) -> int:
        """Return max index used by good-bin controls for the reference dataset."""
        if self._run is not None and self._run.histograms:
            return max(0, self._run.histograms[0].n_bins - 1)
        return max(0, int(self._reference_dataset.n_points) - 1)

    def _set_combo_to_run(self, combo: QComboBox, run_number: int) -> None:
        """Set reference-run combo to the provided run number if it exists."""
        idx = combo.findData(run_number)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_reference_dataset_changed(self) -> None:
        """Reload grouping defaults when the reference dataset changes."""
        run_number = int(self._reference_combo.currentData())
        dataset = next((ds for ds in self._datasets if int(ds.run_number) == run_number), None)
        if dataset is None or dataset.run is None:
            return

        self._reference_dataset = dataset
        self._run = dataset.run
        self._groups = self._load_groups(self._run)
        self._populate_group_table()

        grouping = self._run.grouping or {}
        self._grouping_preset_name = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._enforce_source_bunching = self._reference_is_wim_run()
        self._source_bunching_factor = self._read_source_bunching_factor(grouping)
        self._group_names = self._load_group_names(self._run)
        self._set_combo_to_group(self._forward_combo, int(grouping.get("forward_group", 1)))
        self._set_combo_to_group(self._backward_combo, int(grouping.get("backward_group", 2)))
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))
        max_bin = self._max_bin_index_for_reference_dataset()
        self._first_good_spin.setRange(0, max_bin)
        self._last_good_spin.setRange(0, max_bin)
        self._first_good_spin.setValue(int(grouping.get("first_good_bin", 0)))
        self._last_good_spin.setValue(int(grouping.get("last_good_bin", max_bin)))
        requested_bunching = int(grouping.get("bunching_factor", 1))
        if self._enforce_source_bunching and requested_bunching < self._source_bunching_factor:
            requested_bunching = self._source_bunching_factor
        self._bunch_spin.setValue(requested_bunching)
        self._update_bunching_ui_hints()
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))
        self._update_period_mode_visibility()

    def _reference_is_wim_run(self) -> bool:
        """Return ``True`` when the reference dataset was loaded from a ``.wim`` file."""
        source_file = str(getattr(self._run, "source_file", "") or "")
        return source_file.lower().endswith(".wim")

    def _update_bunching_ui_hints(self) -> None:
        """Refresh explanatory text shown next to bunching controls."""
        if self._enforce_source_bunching:
            base = int(self._source_bunching_factor)
            self._bunch_source_hint.setText(
                f"WIM baseline: {base}. Allowed values: {base}, {2 * base}, {3 * base}, ..."
            )
            self._bunch_spin.setToolTip(
                "For this .wim dataset, bunching must be an integer multiple of "
                f"{base}."
            )
            return

        self._bunch_source_hint.setText("No file-imposed bunching constraint for this dataset type.")
        self._bunch_spin.setToolTip("Set any bunching factor >= 1.")

    def _read_source_bunching_factor(self, grouping: dict[str, Any]) -> int:
        """Return the immutable source bunching baseline for the reference run."""
        raw_value = grouping.get("source_bunching_factor", grouping.get("bunching_factor", 1))
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    def _validate_bunching_factor(self, requested: int) -> str | None:
        """Return validation error text for requested bunching, or ``None``."""
        if not self._enforce_source_bunching:
            return None
        base = self._source_bunching_factor
        if requested < base:
            return (
                "This run was loaded with bunching factor "
                f"{base}, so you cannot decrease below {base}."
            )
        if requested % base != 0:
            return (
                "Bunching factor must be an integer multiple of the file value "
                f"{base}. For example: {base}, {2 * base}, {3 * base}, ..."
            )
        return None

    def _on_apply(self) -> None:
        """Validate form values before accepting the dialog."""
        requested = int(self._bunch_spin.value())
        err = self._validate_bunching_factor(requested)
        if err is not None:
            QMessageBox.warning(self, "Invalid Bunching Factor", err)
            return
        self.accept()

    def _checked_run_numbers(self) -> list[int]:
        """Return run numbers selected for grouping application."""
        selected: list[int] = []
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def _set_all_dataset_checkstates(self, state: Qt.CheckState) -> None:
        """Set check state for all runs in the dataset list."""
        for i in range(self._dataset_list.count()):
            self._dataset_list.item(i).setCheckState(state)

    def _select_all_datasets(self) -> None:
        """Mark all datasets for grouping application."""
        self._set_all_dataset_checkstates(Qt.CheckState.Checked)

    def _select_reference_dataset(self) -> None:
        """Select only the currently chosen reference run."""
        reference_run = self._reference_combo.currentData()
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            run_number = item.data(Qt.ItemDataRole.UserRole)
            state = Qt.CheckState.Checked if run_number == reference_run else Qt.CheckState.Unchecked
            item.setCheckState(state)

    def _deselect_all_datasets(self) -> None:
        """Unmark all datasets for grouping application."""
        self._set_all_dataset_checkstates(Qt.CheckState.Unchecked)

    def _populate_group_table(self) -> None:
        """Render the detector-group table used as grouping context."""
        self._group_table.setRowCount(len(self._groups))
        for row, gid in enumerate(sorted(self._groups)):
            self._group_table.setItem(row, 0, QTableWidgetItem(str(gid)))
            name = self._group_names.get(gid, "")
            self._group_table.setItem(row, 1, QTableWidgetItem(name))
            detectors = [str(idx + 1) for idx in self._groups[gid]]
            self._group_table.setItem(row, 2, QTableWidgetItem(", ".join(detectors)))
        self._group_table.resizeColumnsToContents()

    def _on_detector_layout(self) -> None:
        """Open the interactive detector layout editor as a sub-dialog."""
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        # Determine number of histograms for instrument auto-detection
        n_histo = len(self._run.histograms) if self._run and self._run.histograms else 0
        try:
            instrument_name = self._detector_layout_instrument_name
            if not instrument_name:
                instrument_name = detect_instrument(
                    n_histo,
                    metadata=self._run.metadata if self._run else None,
                    source_file=self._run.source_file if self._run else None,
                )
            if instrument_name is None:
                instrument_name = "HiFi"  # safe fallback
            instrument = get_instrument_layout(instrument_name)
        except KeyError:
            instrument = get_instrument_layout("HiFi")

        grouping = self._run.grouping or {} if self._run else {}
        forward_gid = int(grouping.get("forward_group", self._forward_combo.currentData() or 1))
        backward_gid = int(grouping.get("backward_group", self._backward_combo.currentData() or 2))

        # Convert internal 0-based indices to 1-based for the layout editor
        groups_1based = {gid: [idx + 1 for idx in idxs] for gid, idxs in self._groups.items()}

        dlg = DetectorLayoutDialog(
            instrument=instrument,
            groups=groups_1based,
            group_names=dict(self._group_names),
            initial_preset_name=self._grouping_preset_name,
            forward_group=forward_gid,
            backward_group=backward_gid,
            parent=self,
        )
        if dlg.exec() != DetectorLayoutDialog.DialogCode.Accepted:
            return

        result = dlg.get_result()

        # Write back: convert 1-based IDs back to 0-based internal indices
        new_groups_0based: dict[int, list[int]] = {}
        for gid, det_ids in result["groups"].items():
            new_groups_0based[gid] = sorted(max(0, d - 1) for d in det_ids)

        # Keep any previously defined groups that are not in the result
        # (preserves groups beyond the 8 shown in the editor)
        self._groups = new_groups_0based
        self._group_names = result.get("group_names", {})
        preset_name = result.get("grouping_preset")
        self._grouping_preset_name = str(preset_name) if preset_name else None
        instrument_name = result.get("instrument")
        self._detector_layout_instrument_name = str(instrument_name) if instrument_name else None

        # Update forward/backward combos
        new_fwd = result.get("forward_group", forward_gid)
        new_bwd = result.get("backward_group", backward_gid)

        self._forward_combo.blockSignals(True)
        self._backward_combo.blockSignals(True)
        self._forward_combo.clear()
        self._backward_combo.clear()
        for gid in sorted(self._groups):
            label = self._group_names.get(gid, str(gid))
            display = f"{gid}: {label}" if label and label != str(gid) else str(gid)
            self._forward_combo.addItem(display, gid)
            self._backward_combo.addItem(display, gid)
        self._set_combo_to_group(self._forward_combo, new_fwd)
        self._set_combo_to_group(self._backward_combo, new_bwd)
        self._forward_combo.blockSignals(False)
        self._backward_combo.blockSignals(False)

        self._populate_group_table()

    def _set_combo_to_group(self, combo: QComboBox, group_id: int) -> None:
        """Set combo box to a group ID if present, preserving defaults otherwise."""
        idx = combo.findData(group_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _estimate_alpha(self) -> None:
        """Estimate alpha using only the current reference run.

        This mirrors the intended workflow where alpha is determined from a
        single representative run, then optionally applied to multiple runs.
        """
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        if forward_gid == backward_gid:
            QMessageBox.warning(self, "Invalid Grouping", "Forward and backward groups must differ.")
            return

        forward_indices = self._groups.get(forward_gid, [])
        backward_indices = self._groups.get(backward_gid, [])
        if not forward_indices or not backward_indices:
            QMessageBox.warning(self, "Invalid Grouping", "Selected groups are empty.")
            return

        if self._run is None or not self._run.histograms:
            QMessageBox.warning(self, "Estimate Failed", "Reference run has no histograms.")
            return

        if max(forward_indices, default=-1) >= len(self._run.histograms):
            QMessageBox.warning(self, "Estimate Failed", "Forward group exceeds detector count for reference run.")
            return
        if max(backward_indices, default=-1) >= len(self._run.histograms):
            QMessageBox.warning(self, "Estimate Failed", "Backward group exceeds detector count for reference run.")
            return

        forward_counts = apply_grouping(self._run.histograms, forward_indices)
        backward_counts = apply_grouping(self._run.histograms, backward_indices)

        alpha = estimate_alpha(
            forward_counts,
            backward_counts,
            first_good_bin=int(self._first_good_spin.value()),
            last_good_bin=int(self._last_good_spin.value()),
        )
        self._alpha_spin.setValue(float(alpha))

    def _reference_has_two_period_data(self) -> bool:
        """Return True when the reference run contains two-period histograms."""
        if self._run is None:
            return False
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        period_histograms = grouping.get("period_histograms")
        if isinstance(period_histograms, list) and len(period_histograms) == 2:
            return True
        try:
            return int(self._run.metadata.get("period_count", 1)) == 2
        except (TypeError, ValueError):
            return False

    def _update_period_mode_visibility(self) -> None:
        """Show RG controls only for two-period reference datasets."""
        has_two_period = self._reference_has_two_period_data()
        self._period_mode_label.setVisible(has_two_period)
        self._period_mode_widget.setVisible(has_two_period)
        self._period_mode_widget.setEnabled(has_two_period)

    def _set_period_mode(self, mode_key: str) -> None:
        """Select a period-mode radio button, defaulting to RED."""
        btn = self._period_mode_buttons.get(str(mode_key))
        if btn is None:
            btn = self._period_mode_buttons[str(PeriodMode.RED)]
        btn.setChecked(True)

    def _current_period_mode(self) -> str:
        """Return key for the selected period mode."""
        for key, btn in self._period_mode_buttons.items():
            if btn.isChecked():
                return key
        return str(PeriodMode.RED)

    def _current_grouping_payload(self) -> dict[str, Any]:
        """Build the current grouping payload from UI controls."""
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        return {
            "groups": {gid: [idx + 1 for idx in values] for gid, values in self._groups.items()},
            "group_names": dict(self._group_names),
            "grouping_preset": self._grouping_preset_name,
            "instrument": self._detector_layout_instrument_name,
            "forward_group": forward_gid,
            "backward_group": backward_gid,
            "forward_indices": list(self._groups.get(forward_gid, [])),
            "backward_indices": list(self._groups.get(backward_gid, [])),
            "alpha": float(self._alpha_spin.value()),
            "first_good_bin": int(self._first_good_spin.value()),
            "last_good_bin": int(self._last_good_spin.value()),
            "bunching_factor": int(self._bunch_spin.value()),
            "source_bunching_factor": int(self._source_bunching_factor),
            "enforce_source_bunching": bool(self._enforce_source_bunching),
            "deadtime_correction": bool(self._deadtime_checkbox.isChecked()),
            "period_mode": self._current_period_mode(),
        }

    def _save_grp_file(self) -> None:
        """Save current grouping configuration to a ``.grp`` file."""
        payload = self._current_grouping_payload()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Grouping",
            "grouping.grp",
            "Grouping files (*.grp);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".grp"):
            path += ".grp"

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.serialize_grp(payload))

    def _load_grp_file(self) -> None:
        """Load grouping configuration from a ``.grp`` file into the dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Grouping",
            "",
            "Grouping files (*.grp);;All files (*)",
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as handle:
            payload = self.parse_grp(handle.read())

        loaded_groups = payload.get("groups", {})
        if not isinstance(loaded_groups, dict) or len(loaded_groups) < 2:
            QMessageBox.warning(self, "Invalid Grouping", "Loaded .grp file does not define at least two groups.")
            return

        groups: dict[int, list[int]] = {}
        for key, dets in loaded_groups.items():
            gid = int(key)
            idxs = [max(0, int(v) - 1) for v in dets]
            groups[gid] = sorted(set(idxs))

        self._groups = groups
        self._populate_group_table()

        self._forward_combo.blockSignals(True)
        self._backward_combo.blockSignals(True)
        self._forward_combo.clear()
        self._backward_combo.clear()
        for gid in sorted(self._groups):
            text = str(gid)
            self._forward_combo.addItem(text, gid)
            self._backward_combo.addItem(text, gid)
        self._set_combo_to_group(self._forward_combo, int(payload.get("forward_group", 1)))
        self._set_combo_to_group(self._backward_combo, int(payload.get("backward_group", 2)))
        self._forward_combo.blockSignals(False)
        self._backward_combo.blockSignals(False)

        loaded_group_names = payload.get("group_names", {})
        if isinstance(loaded_group_names, dict):
            self._group_names = {int(k): str(v) for k, v in loaded_group_names.items()}
        self._alpha_spin.setValue(float(payload.get("alpha", 1.0)))
        self._first_good_spin.setValue(int(payload.get("first_good_bin", self._first_good_spin.value())))
        self._last_good_spin.setValue(int(payload.get("last_good_bin", self._last_good_spin.value())))
        self._bunch_spin.setValue(int(payload.get("bunching_factor", self._bunch_spin.value())))
        self._deadtime_checkbox.setChecked(bool(payload.get("deadtime_correction", False)))
        self._set_period_mode(str(payload.get("period_mode", PeriodMode.RED)))
        self._populate_group_table()

    @staticmethod
    def serialize_grp(payload: dict[str, Any]) -> str:
        """Serialize grouping payload to text ``.grp`` format.

        The generated file is intentionally simple and line-based:

        ``key=value`` for scalars and ``group.<id>=csv`` for detector lists.
        Detector indices are stored 1-based for compatibility with existing
        μSR tooling conventions.
        """
        lines = [
            "# Asymmetry grouping file v1",
            f"forward_group={int(payload.get('forward_group', 1))}",
            f"backward_group={int(payload.get('backward_group', 2))}",
            f"alpha={float(payload.get('alpha', 1.0)):.12g}",
            f"first_good_bin={int(payload.get('first_good_bin', 0))}",
            f"last_good_bin={int(payload.get('last_good_bin', 0))}",
            f"bunching_factor={int(payload.get('bunching_factor', 1))}",
            f"deadtime_correction={1 if bool(payload.get('deadtime_correction', False)) else 0}",
            f"period_mode={str(payload.get('period_mode', PeriodMode.RED))}",
        ]

        groups = payload.get("groups", {})
        group_names = payload.get("group_names", {})
        if isinstance(groups, dict):
            for gid in sorted(int(k) for k in groups.keys()):
                detectors = [str(int(v)) for v in groups.get(gid, [])]
                lines.append(f"group.{gid}={','.join(detectors)}")
        if isinstance(group_names, dict):
            for gid in sorted(int(k) for k in group_names.keys()):
                name = str(group_names.get(gid, "")).strip()
                if name:
                    lines.append(f"group_name.{gid}={name}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_grp(text: str) -> dict[str, Any]:
        """Parse line-based ``.grp`` text into a grouping payload dictionary."""
        payload: dict[str, Any] = {
            "groups": {},
            "group_names": {},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 0,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "period_mode": str(PeriodMode.RED),
        }

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key.startswith("group_name."):
                gid = int(key.split(".", 1)[1])
                payload["group_names"][gid] = value
                continue

            if key.startswith("group."):
                gid = int(key.split(".", 1)[1])
                dets = [int(v.strip()) for v in value.split(",") if v.strip()]
                payload["groups"][gid] = dets
                continue

            if key in {"forward_group", "backward_group", "first_good_bin", "last_good_bin", "bunching_factor"}:
                payload[key] = int(float(value))
            elif key == "alpha":
                payload[key] = float(value)
            elif key == "deadtime_correction":
                payload[key] = value.strip().lower() in {"1", "true", "yes", "on"}
            elif key == "period_mode":
                if value in {
                    str(PeriodMode.RED),
                    str(PeriodMode.GREEN),
                    str(PeriodMode.GREEN_MINUS_RED),
                    str(PeriodMode.GREEN_PLUS_RED),
                }:
                    payload[key] = value

        return payload

    def get_grouping_result(self) -> dict[str, Any] | None:
        """Return grouping settings selected in the dialog for all checked runs.

        Returns
        -------
        dict or None
            Grouping payload used by the main window to recompute asymmetry.
            Returns ``None`` when no run histograms are available.
        """
        if self._run is None or not self._run.histograms:
            return None
        payload = self._current_grouping_payload()
        payload["run_numbers"] = self._checked_run_numbers()
        return payload


class WimGroupingDialog(QDialog):
    """WIM-specific grouping dialog.

    This dialog is intentionally limited to bunching-factor edits. Other
    grouping parameters parsed from ``.wim`` files are displayed as read-only
    values for transparency.
    """

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        selected_run_number: int | None = None,
        selected_run_numbers: list[int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._datasets = [
            ds
            for ds in datasets
            if ds.run is not None and str(getattr(ds.run, "source_file", "")).lower().endswith(".wim")
        ]
        self._selected_run_number = selected_run_number
        self._selected_run_numbers = (
            {int(v) for v in selected_run_numbers}
            if selected_run_numbers is not None
            else None
        )

        self.setWindowTitle("WIM Grouping")
        self.resize(760, 520)

        if not self._datasets:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("No .wim runs are available in the active selection."))
            close_btn = QPushButton("Close")
            close_btn.setAutoDefault(False)
            close_btn.setDefault(False)
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)
            return

        self._reference_dataset = self._choose_reference_dataset()
        self._run = self._reference_dataset.run
        assert self._run is not None

        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel(f"WIM datasets: {len(self._datasets)}"))
        self._reference_combo = QComboBox()
        for ds in self._datasets:
            self._reference_combo.addItem(ds.run_label, int(ds.run_number))
        self._set_combo_to_run(self._reference_combo, int(self._reference_dataset.run_number))
        self._reference_combo.currentIndexChanged.connect(self._on_reference_dataset_changed)
        top_row.addWidget(QLabel("Reference run"))
        top_row.addWidget(self._reference_combo)
        top_row.addStretch()
        root.addLayout(top_row)

        self._dataset_list = QListWidget()
        self._dataset_list.setMaximumHeight(150)
        for ds in self._datasets:
            item = QListWidgetItem(ds.run_label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            run_number = int(ds.run_number)
            item.setData(Qt.ItemDataRole.UserRole, run_number)
            if self._selected_run_numbers is None:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                state = (
                    Qt.CheckState.Checked
                    if run_number in self._selected_run_numbers
                    else Qt.CheckState.Unchecked
                )
                item.setCheckState(state)
            self._dataset_list.addItem(item)
        root.addWidget(self._dataset_list)

        selection_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.setAutoDefault(False)
        select_all_btn.setDefault(False)
        select_all_btn.clicked.connect(self._select_all_datasets)
        select_reference_btn = QPushButton("Select Reference Run")
        select_reference_btn.setAutoDefault(False)
        select_reference_btn.setDefault(False)
        select_reference_btn.clicked.connect(self._select_reference_dataset)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.setAutoDefault(False)
        deselect_all_btn.setDefault(False)
        deselect_all_btn.clicked.connect(self._deselect_all_datasets)
        selection_buttons.addWidget(select_all_btn)
        selection_buttons.addWidget(select_reference_btn)
        selection_buttons.addWidget(deselect_all_btn)
        selection_buttons.addStretch()
        root.addLayout(selection_buttons)

        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        self._source_bunching_factor = self._read_source_bunching_factor(grouping)

        read_only_form = QFormLayout()
        self._forward_label = QLabel()
        self._backward_label = QLabel()
        self._alpha_label = QLabel()
        self._first_good_label = QLabel()
        self._last_good_label = QLabel()
        self._deadtime_label = QLabel()
        read_only_form.addRow("Forward Group", self._forward_label)
        read_only_form.addRow("Backward Group", self._backward_label)
        read_only_form.addRow("Alpha", self._alpha_label)
        read_only_form.addRow("First Good Bin", self._first_good_label)
        read_only_form.addRow("Last Good Bin", self._last_good_label)
        read_only_form.addRow("Deadtime", self._deadtime_label)

        self._bunch_spin = QSpinBox()
        self._bunch_spin.setRange(1, 10000)
        requested_bunching = int(grouping.get("bunching_factor", self._source_bunching_factor))
        if requested_bunching < self._source_bunching_factor:
            requested_bunching = self._source_bunching_factor
        self._bunch_spin.setValue(requested_bunching)
        self._bunch_spin.setMaximumWidth(120)
        self._bunch_hint_label = QLabel()

        bunch_form = QFormLayout()
        bunch_form.addRow("Bunching Factor", self._bunch_spin)
        bunch_form.addRow("Bunching Rules", self._bunch_hint_label)

        root.addLayout(read_only_form)
        root.addLayout(bunch_form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setAutoDefault(False)
        apply_btn.setDefault(False)
        apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        root.addLayout(buttons)

        self._refresh_read_only_fields()
        self._update_bunching_ui_hints()

    def _choose_reference_dataset(self) -> MuonDataset:
        if self._selected_run_number is not None:
            for ds in self._datasets:
                if int(ds.run_number) == int(self._selected_run_number):
                    return ds
        return self._datasets[0]

    def _set_combo_to_run(self, combo: QComboBox, run_number: int) -> None:
        idx = combo.findData(run_number)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_reference_dataset_changed(self) -> None:
        run_number = int(self._reference_combo.currentData())
        dataset = next((ds for ds in self._datasets if int(ds.run_number) == run_number), None)
        if dataset is None or dataset.run is None:
            return
        self._reference_dataset = dataset
        self._run = dataset.run
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        self._source_bunching_factor = self._read_source_bunching_factor(grouping)
        requested_bunching = int(grouping.get("bunching_factor", self._source_bunching_factor))
        if requested_bunching < self._source_bunching_factor:
            requested_bunching = self._source_bunching_factor
        self._bunch_spin.setValue(requested_bunching)
        self._refresh_read_only_fields()
        self._update_bunching_ui_hints()

    def _refresh_read_only_fields(self) -> None:
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}

        def _fmt(key: str, default: str = "N/A") -> str:
            value = grouping.get(key)
            return default if value is None else str(value)

        self._forward_label.setText(_fmt("forward_group"))
        self._backward_label.setText(_fmt("backward_group"))
        self._alpha_label.setText(_fmt("alpha"))
        self._first_good_label.setText(_fmt("first_good_bin"))
        self._last_good_label.setText(_fmt("last_good_bin"))
        self._deadtime_label.setText("on" if bool(grouping.get("deadtime_correction", False)) else "off")

    def _update_bunching_ui_hints(self) -> None:
        base = int(self._source_bunching_factor)
        self._bunch_hint_label.setText(
            f"WIM baseline: {base}. Allowed values: {base}, {2 * base}, {3 * base}, ..."
        )
        self._bunch_spin.setToolTip(
            "For this .wim dataset, bunching must be an integer multiple of "
            f"{base}."
        )

    def _read_source_bunching_factor(self, grouping: dict[str, Any]) -> int:
        raw_value = grouping.get("source_bunching_factor", grouping.get("bunching_factor", 1))
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 1

    def _validate_bunching_factor(self, requested: int) -> str | None:
        base = self._source_bunching_factor
        if requested < base:
            return (
                "This run was loaded with bunching factor "
                f"{base}, so you cannot decrease below {base}."
            )
        if requested % base != 0:
            return (
                "Bunching factor must be an integer multiple of the file value "
                f"{base}. For example: {base}, {2 * base}, {3 * base}, ..."
            )
        return None

    def _on_apply(self) -> None:
        requested = int(self._bunch_spin.value())
        err = self._validate_bunching_factor(requested)
        if err is not None:
            QMessageBox.warning(self, "Invalid Bunching Factor", err)
            return
        self.accept()

    def _checked_run_numbers(self) -> list[int]:
        selected: list[int] = []
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def _set_all_dataset_checkstates(self, state: Qt.CheckState) -> None:
        for i in range(self._dataset_list.count()):
            self._dataset_list.item(i).setCheckState(state)

    def _select_all_datasets(self) -> None:
        self._set_all_dataset_checkstates(Qt.CheckState.Checked)

    def _select_reference_dataset(self) -> None:
        reference_run = self._reference_combo.currentData()
        for i in range(self._dataset_list.count()):
            item = self._dataset_list.item(i)
            run_number = item.data(Qt.ItemDataRole.UserRole)
            state = Qt.CheckState.Checked if run_number == reference_run else Qt.CheckState.Unchecked
            item.setCheckState(state)

    def _deselect_all_datasets(self) -> None:
        self._set_all_dataset_checkstates(Qt.CheckState.Unchecked)

    def _current_grouping_payload(self) -> dict[str, Any]:
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        payload: dict[str, Any] = {
            "forward_group": int(grouping.get("forward_group", 1)),
            "backward_group": int(grouping.get("backward_group", 2)),
            "alpha": float(grouping.get("alpha", 1.0)),
            "first_good_bin": int(grouping.get("first_good_bin", 0)),
            "last_good_bin": int(grouping.get("last_good_bin", max(0, self._reference_dataset.n_points - 1))),
            "bunching_factor": int(self._bunch_spin.value()),
            "source_bunching_factor": int(self._source_bunching_factor),
            "enforce_source_bunching": True,
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "period_mode": str(grouping.get("period_mode", PeriodMode.RED)),
        }

        groups_raw = grouping.get("groups")
        if isinstance(groups_raw, dict):
            groups: dict[int, list[int]] = {}
            for key, values in groups_raw.items():
                try:
                    gid = int(key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(values, list):
                    continue
                detectors: list[int] = []
                for value in values:
                    try:
                        detectors.append(int(value))
                    except (TypeError, ValueError):
                        continue
                if detectors:
                    groups[gid] = detectors
            if groups:
                payload["groups"] = groups

        if "dead_time_us" in grouping and isinstance(grouping.get("dead_time_us"), list):
            payload["dead_time_us"] = list(grouping.get("dead_time_us", []))
        if "good_frames" in grouping:
            payload["good_frames"] = grouping.get("good_frames")
        instrument_name = grouping.get("instrument")
        if instrument_name:
            payload["instrument"] = str(instrument_name)
        return payload

    def get_grouping_result(self) -> dict[str, Any] | None:
        if self._run is None:
            return None
        payload = self._current_grouping_payload()
        payload["run_numbers"] = self._checked_run_numbers()
        return payload
