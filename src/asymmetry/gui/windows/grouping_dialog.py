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
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.transform import apply_grouping
from asymmetry.core.transform.asymmetry import estimate_alpha


class GroupingDialog(QDialog):
    """Edit forward/backward grouping for multiple datasets.

    Parameters
    ----------
    datasets
        Datasets available in the active project. Datasets without raw
        histograms are ignored for grouping operations.
    selected_run_number
        Optional run number used as initial reference dataset.
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        selected_run_number: int | None = None,
        parent=None,
    ) -> None:
        """Create a shared grouping dialog for project datasets."""
        super().__init__(parent)
        self._datasets = [
            ds for ds in datasets if ds.run is not None and bool(ds.run.histograms)
        ]
        self._selected_run_number = selected_run_number

        self.setWindowTitle("Grouping")
        self.resize(860, 560)

        if not self._datasets:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("No raw histograms are available in the active project."))
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)
            return

        self._reference_dataset = self._choose_reference_dataset()
        self._run = self._reference_dataset.run
        assert self._run is not None

        self._groups = self._load_groups(self._run)

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
            item.setData(Qt.ItemDataRole.UserRole, int(ds.run_number))
            item.setCheckState(Qt.CheckState.Checked)
            self._dataset_list.addItem(item)

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

        self._group_table = QTableWidget(0, 2)
        self._group_table.setHorizontalHeaderLabels(["Group", "Detector Indices (1-based)"])
        root.addWidget(self._group_table)
        self._populate_group_table()

        form = QFormLayout()
        self._forward_combo = QComboBox()
        self._backward_combo = QComboBox()
        for gid in sorted(self._groups):
            text = str(gid)
            self._forward_combo.addItem(text, gid)
            self._backward_combo.addItem(text, gid)

        grouping = self._run.grouping or {}
        self._set_combo_to_group(self._forward_combo, int(grouping.get("forward_group", 1)))
        self._set_combo_to_group(self._backward_combo, int(grouping.get("backward_group", 2)))

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setDecimals(6)
        self._alpha_spin.setRange(0.01, 1000.0)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))

        self._first_good_spin = QSpinBox()
        self._first_good_spin.setRange(0, max(0, self._run.histograms[0].n_bins - 1))
        self._first_good_spin.setValue(int(grouping.get("first_good_bin", 0)))

        self._last_good_spin = QSpinBox()
        self._last_good_spin.setRange(0, max(0, self._run.histograms[0].n_bins - 1))
        self._last_good_spin.setValue(int(grouping.get("last_good_bin", self._run.histograms[0].n_bins - 1)))

        self._bunch_spin = QSpinBox()
        self._bunch_spin.setRange(1, 10000)
        self._bunch_spin.setValue(int(grouping.get("bunching_factor", 1)))
        self._bunch_spin.setMaximumWidth(100)

        self._deadtime_checkbox = QCheckBox("Enable Deadtime Correction")
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))

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
        form.addRow("Deadtime", self._deadtime_checkbox)
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
        apply_btn.clicked.connect(self.accept)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(apply_btn)
        root.addLayout(buttons)

    def _choose_reference_dataset(self) -> MuonDataset:
        """Return preferred reference dataset for initial grouping values."""
        if self._selected_run_number is not None:
            for ds in self._datasets:
                if int(ds.run_number) == int(self._selected_run_number):
                    return ds
        return self._datasets[0]

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
        split = max(1, n // 2)
        return {1: list(range(0, split)), 2: list(range(split, n)) or list(range(0, n))}

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
        self._set_combo_to_group(self._forward_combo, int(grouping.get("forward_group", 1)))
        self._set_combo_to_group(self._backward_combo, int(grouping.get("backward_group", 2)))
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))

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
            detectors = [str(idx + 1) for idx in self._groups[gid]]
            self._group_table.setItem(row, 1, QTableWidgetItem(", ".join(detectors)))
        self._group_table.resizeColumnsToContents()

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

    def _current_grouping_payload(self) -> dict[str, Any]:
        """Build the current grouping payload from UI controls."""
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        return {
            "groups": {gid: [idx + 1 for idx in values] for gid, values in self._groups.items()},
            "forward_group": forward_gid,
            "backward_group": backward_gid,
            "forward_indices": list(self._groups.get(forward_gid, [])),
            "backward_indices": list(self._groups.get(backward_gid, [])),
            "alpha": float(self._alpha_spin.value()),
            "first_good_bin": int(self._first_good_spin.value()),
            "last_good_bin": int(self._last_good_spin.value()),
            "bunching_factor": int(self._bunch_spin.value()),
            "deadtime_correction": bool(self._deadtime_checkbox.isChecked()),
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

        self._alpha_spin.setValue(float(payload.get("alpha", 1.0)))
        self._first_good_spin.setValue(int(payload.get("first_good_bin", self._first_good_spin.value())))
        self._last_good_spin.setValue(int(payload.get("last_good_bin", self._last_good_spin.value())))
        self._bunch_spin.setValue(int(payload.get("bunching_factor", self._bunch_spin.value())))
        self._deadtime_checkbox.setChecked(bool(payload.get("deadtime_correction", False)))

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
        ]

        groups = payload.get("groups", {})
        if isinstance(groups, dict):
            for gid in sorted(int(k) for k in groups.keys()):
                detectors = [str(int(v)) for v in groups.get(gid, [])]
                lines.append(f"group.{gid}={','.join(detectors)}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_grp(text: str) -> dict[str, Any]:
        """Parse line-based ``.grp`` text into a grouping payload dictionary."""
        payload: dict[str, Any] = {
            "groups": {},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "first_good_bin": 0,
            "last_good_bin": 0,
            "bunching_factor": 1,
            "deadtime_correction": False,
        }

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

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
