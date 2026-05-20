"""WiMDA-style shared grouping dialog with alpha estimation and .grp I/O.

The dialog edits detector grouping once and applies it across multiple datasets
in the active project. Grouping definitions can be saved to and loaded from
``.grp`` files.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.instrument import detect_instrument, get_instrument_layout
from asymmetry.core.transform import (
    apply_grouping,
    calibrate_deadtime_from_histograms,
    estimate_deadtime_from_histograms,
    supports_background_correction,
)
from asymmetry.core.transform.asymmetry import estimate_alpha
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.styles.widgets import apply_param_table_style


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
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._selected_run_number = selected_run_number
        self._selected_run_numbers = (
            {int(v) for v in selected_run_numbers} if selected_run_numbers is not None else None
        )

        self.setWindowTitle("Grouping")
        self.resize(860, 500)

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
        self._included_groups: dict[int, bool] = self._load_included_groups(self._run)
        self._vector_axis_pairs: dict[str, tuple[int, int]] = {}
        self._vector_alpha_spins: dict[str, QDoubleSpinBox] = {}
        self._vector_forward_labels: dict[str, QLabel] = {}
        self._vector_backward_labels: dict[str, QLabel] = {}
        self._vector_estimate_buttons: dict[str, QPushButton] = {}
        self._updating_deadtime_value_combo = False

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # \u2500\u2500 Top bar: dataset count + reference run selector \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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

        # \u2500\u2500 Main split: left pane (datasets + groups) | right pane (form) \u2500
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)

        # Left pane
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

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
        left_layout.addLayout(dataset_buttons)

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
        left_layout.addWidget(self._dataset_list)

        self._group_table = QTableWidget(0, 4)
        self._group_table.setHorizontalHeaderLabels(
            ["Group", "Include", "Name", "Detector Indices (1-based)"]
        )
        apply_param_table_style(self._group_table)
        self._group_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._group_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._group_table.setMinimumHeight(0)
        left_layout.addWidget(self._group_table)
        self._populate_group_table()

        detector_layout_btn = QPushButton("Detector Layout\u2026")
        detector_layout_btn.setAutoDefault(False)
        detector_layout_btn.setDefault(False)
        detector_layout_btn.setToolTip(
            "Open the interactive detector schematic editor to assign detectors to groups visually."
        )
        detector_layout_btn.clicked.connect(self._on_detector_layout)
        left_layout.addWidget(detector_layout_btn)
        left_layout.addStretch()
        splitter.addWidget(left_pane)

        # Right pane
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(4, 4, 0, 0)
        right_layout.setSpacing(0)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(12)
        self._forward_combo = QComboBox()
        self._backward_combo = QComboBox()
        self._forward_combo.setMinimumWidth(220)
        self._backward_combo.setMinimumWidth(220)
        self._forward_combo.setMinimumContentsLength(18)
        self._backward_combo.setMinimumContentsLength(18)
        self._refresh_group_combo_items()

        grouping = self._run.grouping or {}
        self._grouping_preset_name: str | None = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._detector_layout_instrument_name: str | None = (
            str(grouping.get("instrument")).strip() if grouping.get("instrument") else None
        )
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(grouping.get("forward_group", 1)),
            int(grouping.get("backward_group", 2)),
        )
        self._set_combo_to_group(self._forward_combo, forward_gid)
        self._set_combo_to_group(self._backward_combo, backward_gid)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setDecimals(6)
        self._alpha_spin.setRange(0.01, 1000.0)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))

        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = self._bin_index_base(grouping)
        default_t0_internal = self._default_t0_bin(grouping, max_bin)
        default_t_good = self._default_t_good_offset(grouping, default_t0_internal, max_bin)

        self._t0_spin = QSpinBox()
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._t0_spin.setValue(default_t0_internal + index_base)

        self._t_good_offset_spin = QSpinBox()
        self._t_good_offset_spin.setRange(0, max_bin)
        self._t_good_offset_spin.setValue(default_t_good)
        self._t0_spin.valueChanged.connect(self._on_t0_changed)
        self._on_t0_changed()

        self._last_good_spin = QSpinBox()
        self._last_good_spin.setRange(index_base, max_bin + index_base)
        default_first_good = min(max_bin, default_t0_internal + default_t_good)
        default_last_good = int(grouping.get("last_good_bin", max_bin))
        if default_last_good < default_first_good:
            default_last_good = default_first_good
        self._last_good_spin.setValue(default_last_good + index_base)

        self._bunch_spin = QSpinBox()
        self._bunch_spin.setRange(1, 10000)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        self._bunch_spin.setValue(requested_bunching)
        self._bunch_spin.setMaximumWidth(100)
        self._bunch_spin.setToolTip("Set any bunching factor >= 1.")

        self._deadtime_checkbox = QCheckBox("Enable Deadtime Correction")
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))
        self._deadtime_checkbox.toggled.connect(self._update_deadtime_controls)

        self._deadtime_mode_group = QButtonGroup(self)
        self._deadtime_mode_buttons: dict[str, QRadioButton] = {}
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(grouping)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(grouping)

        self._deadtime_value_combo = QComboBox()
        self._deadtime_value_combo.setEditable(True)
        self._deadtime_value_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._deadtime_value_combo.activated.connect(self._on_deadtime_value_index_changed)
        line_edit = self._deadtime_value_combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self._on_deadtime_value_edited)

        self._deadtime_mode_widget = QWidget()
        deadtime_mode_layout = QGridLayout(self._deadtime_mode_widget)
        deadtime_mode_layout.setContentsMargins(0, 0, 0, 0)
        deadtime_mode_layout.setHorizontalSpacing(8)
        deadtime_mode_layout.setVerticalSpacing(4)
        deadtime_mode_layout.setColumnStretch(1, 1)

        deadtime_mode_specs = [
            ("file", "File"),
            ("manual", "Manual"),
            ("estimate", "Estimate"),
        ]
        for col, (mode, label) in enumerate(deadtime_mode_specs):
            button = QRadioButton(label)
            button.toggled.connect(self._update_deadtime_controls)
            self._deadtime_mode_group.addButton(button)
            self._deadtime_mode_buttons[mode] = button
            deadtime_mode_layout.addWidget(button, 0, col)

        deadtime_mode_layout.addWidget(QLabel("Detector Values"), 1, 0)
        deadtime_mode_layout.addWidget(self._deadtime_value_combo, 1, 1)

        self._deadtime_calibrate_btn = QPushButton("Cal")
        self._deadtime_calibrate_btn.setAutoDefault(False)
        self._deadtime_calibrate_btn.setDefault(False)
        self._deadtime_calibrate_btn.clicked.connect(self._calibrate_deadtime_from_reference)
        self._deadtime_calibrate_btn.setToolTip(
            "Fit one deadtime value per detector from the selected reference run and populate the manual detector table."
        )
        deadtime_mode_layout.addWidget(self._deadtime_calibrate_btn, 1, 2)

        self._deadtime_status_label = QLabel("")
        self._deadtime_status_label.setWordWrap(True)
        self._deadtime_status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._deadtime_status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        self._deadtime_status_label.setMinimumHeight(self.fontMetrics().lineSpacing() * 3)
        deadtime_mode_layout.addWidget(self._deadtime_status_label, 2, 0, 1, 3)

        self._set_deadtime_mode(self._default_deadtime_mode(grouping))
        self._update_deadtime_controls(grouping)
        self._background_checkbox = QCheckBox("Enable Background Correction")
        self._background_checkbox.setChecked(bool(grouping.get("background_correction", False)))
        self._update_background_checkbox_state()

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

        self._forward_row_label = QLabel("Forward Group")
        self._backward_row_label = QLabel("Backward Group")
        self._alpha_row_label = QLabel("Alpha")

        form.addRow(self._forward_row_label, self._forward_combo)
        form.addRow(self._backward_row_label, self._backward_combo)

        self._single_alpha_widget = QWidget()
        alpha_row = QHBoxLayout(self._single_alpha_widget)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.addWidget(self._alpha_spin)
        alpha_row.addWidget(estimate_btn)
        form.addRow(self._alpha_row_label, self._single_alpha_widget)

        # Vector alpha widget: one row per axis (P_z primary, then P_y, P_x)
        # Columns: axis label | Forward group | Backward group | α spin | Estimate button
        self._vector_alpha_widget = QWidget()
        vector_layout = QGridLayout(self._vector_alpha_widget)
        vector_layout.setContentsMargins(0, 0, 0, 0)
        vector_layout.setHorizontalSpacing(12)
        vector_layout.setVerticalSpacing(8)

        vector_layout.addWidget(QLabel("Forward"), 0, 1)
        vector_layout.addWidget(QLabel("Backward"), 0, 2)
        vector_layout.addWidget(QLabel("α"), 0, 3)

        grouping_alpha = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        for row_idx, axis in enumerate(("P_z", "P_y", "P_x"), start=1):
            vector_layout.addWidget(QLabel(axis), row_idx, 0)
            fwd_label = QLabel("-")
            bwd_label = QLabel("-")
            vector_layout.addWidget(fwd_label, row_idx, 1)
            vector_layout.addWidget(bwd_label, row_idx, 2)
            self._vector_forward_labels[axis] = fwd_label
            self._vector_backward_labels[axis] = bwd_label

            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0.01, 1000.0)
            spin.setValue(
                self._alpha_value_for_axis(grouping_alpha, axis, float(self._alpha_spin.value()))
            )
            vector_layout.addWidget(spin, row_idx, 3)
            self._vector_alpha_spins[axis] = spin

            btn = QPushButton("Estimate α")
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.clicked.connect(
                lambda _checked=False, axis_key=axis: self._estimate_alpha_for_axis(axis_key)
            )
            vector_layout.addWidget(btn, row_idx, 4)
            self._vector_estimate_buttons[axis] = btn

        self._estimate_all_btn = QPushButton("Estimate All α")
        self._estimate_all_btn.setAutoDefault(False)
        self._estimate_all_btn.setDefault(False)
        self._estimate_all_btn.clicked.connect(self._estimate_all_alpha)
        vector_layout.addWidget(self._estimate_all_btn, 4, 0, 1, 5)

        form.addRow(self._vector_alpha_widget)

        form.addRow("t0 Bin", self._t0_spin)
        form.addRow("t_good Offset", self._t_good_offset_spin)
        form.addRow("Last Good Bin", self._last_good_spin)
        form.addRow("Bunching Factor", self._bunch_spin)
        form.addRow("Deadtime", self._deadtime_checkbox)
        form.addRow("Deadtime Mode", self._deadtime_mode_widget)
        form.addRow("Background", self._background_checkbox)
        form.addRow(self._period_mode_label, self._period_mode_widget)

        right_layout.addLayout(form)
        right_layout.addStretch()
        splitter.addWidget(right_pane)

        splitter.setSizes([330, 520])
        root.addWidget(splitter, stretch=1)

        self._update_vector_mode_controls(grouping)

        # ── Bottom bar: file I/O + action buttons ────────────────────────
        load_btn = QPushButton("Load .grp")
        load_btn.setAutoDefault(False)
        load_btn.setDefault(False)
        load_btn.clicked.connect(self._load_grp_file)
        save_btn = QPushButton("Save .grp")
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(self._save_grp_file)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setAutoDefault(False)
        apply_btn.setDefault(False)
        apply_btn.clicked.connect(self._on_apply)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(load_btn)
        bottom_bar.addWidget(save_btn)
        bottom_bar.addStretch()
        bottom_bar.addWidget(cancel_btn)
        bottom_bar.addWidget(apply_btn)
        root.addLayout(bottom_bar)

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

    def _load_included_groups(self, run) -> dict[int, bool]:
        """Load per-group include flags from run metadata, defaulting missing rows to True."""
        grouping = run.grouping or {}
        raw = grouping.get("included_groups")
        result: dict[int, bool] = {}
        if isinstance(raw, dict):
            for key, val in raw.items():
                try:
                    gid = int(key)
                except (TypeError, ValueError):
                    continue
                result[gid] = bool(val)
        for gid in self._groups:
            result.setdefault(int(gid), True)
        return result

    def _reference_is_psi(self) -> bool:
        """Return True when the current reference run uses PSI detector conventions."""
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        if self._run is not None:
            metadata.update(getattr(self._run, "metadata", {}) or {})
            grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
            if grouping.get("psi_format"):
                return True
        facility = str(metadata.get("facility", "")).strip().lower()
        return facility == "psi" or bool(metadata.get("psi_format"))

    @staticmethod
    def _beam_direction_label(label: object) -> str | None:
        """Return beam-direction label for explicit forward/backward group names."""
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if token.startswith(("forw", "fwd")) or "forward" in token:
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token:
            return "backward"
        return None

    def _analysis_pair_for_reference(self, forward_gid: int, backward_gid: int) -> tuple[int, int]:
        """Map PSI beam-forward/backward selections to analysis spin-forward/backward."""
        if not self._reference_is_psi():
            return int(forward_gid), int(backward_gid)
        forward_name = self._group_names.get(int(forward_gid), "")
        backward_name = self._group_names.get(int(backward_gid), "")
        if (
            self._beam_direction_label(forward_name) == "forward"
            and self._beam_direction_label(backward_name) == "backward"
        ):
            return int(backward_gid), int(forward_gid)
        return int(forward_gid), int(backward_gid)

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

    def _bin_index_base(self, grouping: dict[str, Any] | None = None) -> int:
        """Return displayed index base (0- or 1-based) for bin controls."""
        if grouping is None:
            grouping = (
                self._run.grouping
                if self._run is not None and isinstance(self._run.grouping, dict)
                else {}
            )
        raw_base = grouping.get("bin_index_base", 0)
        try:
            return 1 if int(raw_base) == 1 else 0
        except (TypeError, ValueError):
            return 0

    def _default_t0_bin(self, grouping: dict[str, Any], max_bin: int) -> int:
        """Return initial internal t0 bin from grouping metadata or histograms."""
        raw_t0 = grouping.get("t0_bin")
        if raw_t0 is None and self._run is not None and self._run.histograms:
            raw_t0 = self._run.histograms[0].t0_bin
        try:
            return max(0, min(max_bin, int(raw_t0)))
        except (TypeError, ValueError):
            return 0

    def _default_t_good_offset(self, grouping: dict[str, Any], t0_bin: int, max_bin: int) -> int:
        """Return initial t_good offset from grouping metadata."""
        raw_offset = grouping.get("t_good_offset")
        if raw_offset is None:
            raw_first = grouping.get("first_good_bin", t0_bin)
            try:
                raw_offset = int(raw_first) - int(t0_bin)
            except (TypeError, ValueError):
                raw_offset = 0
        try:
            return max(0, min(max_bin, int(raw_offset)))
        except (TypeError, ValueError):
            return 0

    def _on_t0_changed(self) -> None:
        """Constrain t_good offset so t0 + offset remains in the histogram range."""
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_bin = max(0, min(max_bin, int(self._t0_spin.value()) - base))
        max_offset = max(0, max_bin - t0_bin)
        self._t_good_offset_spin.setMaximum(max_offset)
        if int(self._t_good_offset_spin.value()) > max_offset:
            self._t_good_offset_spin.setValue(max_offset)

    def _resolve_good_bin_limits_from_controls(self) -> tuple[int, int, int, int]:
        """Return validated ``(t0_bin, t_good_offset, first_good_bin, last_good_bin)``."""
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_bin = max(0, min(max_bin, int(self._t0_spin.value()) - base))
        t_good_offset = max(0, int(self._t_good_offset_spin.value()))
        first_good_bin = t0_bin + t_good_offset
        last_good_bin = max(0, min(max_bin, int(self._last_good_spin.value()) - base))
        return t0_bin, t_good_offset, first_good_bin, last_good_bin

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
        self._group_names = self._load_group_names(self._run)
        self._included_groups = self._load_included_groups(self._run)
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(grouping.get("forward_group", 1)),
            int(grouping.get("backward_group", 2)),
        )
        self._refresh_group_combo_items(forward_gid=forward_gid, backward_gid=backward_gid)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))
        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = self._bin_index_base(grouping)
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._t_good_offset_spin.setRange(0, max_bin)
        self._last_good_spin.setRange(index_base, max_bin + index_base)
        default_t0_internal = self._default_t0_bin(grouping, max_bin)
        default_t_good = self._default_t_good_offset(grouping, default_t0_internal, max_bin)
        self._t0_spin.setValue(default_t0_internal + index_base)
        self._t_good_offset_spin.setValue(default_t_good)
        self._on_t0_changed()
        default_first_good = min(max_bin, default_t0_internal + default_t_good)
        default_last_good = int(grouping.get("last_good_bin", max_bin))
        if default_last_good < default_first_good:
            default_last_good = default_first_good
        self._last_good_spin.setValue(default_last_good + index_base)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        self._bunch_spin.setValue(requested_bunching)
        self._deadtime_checkbox.setChecked(bool(grouping.get("deadtime_correction", False)))
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(grouping)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(grouping)
        self._set_deadtime_mode(self._default_deadtime_mode(grouping))
        self._update_deadtime_controls(grouping)
        self._background_checkbox.setChecked(bool(grouping.get("background_correction", False)))
        self._update_background_checkbox_state()
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))
        self._update_vector_mode_controls(grouping)
        self._update_period_mode_visibility()

    def _reference_has_file_deadtime(self, grouping: dict[str, Any]) -> bool:
        """Return whether the reference run provides file deadtime values."""
        values = grouping.get("dead_time_us") if isinstance(grouping, dict) else None
        if not isinstance(values, list):
            return False
        n_histograms = len(self._run.histograms) if self._run is not None else 0
        required = max(1, n_histograms)
        if len(values) < required:
            return False
        for value in values:
            try:
                if float(value) != 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _default_deadtime_mode(self, grouping: dict[str, Any]) -> str:
        """Return preferred deadtime mode for the current grouping state."""
        mode = (
            str(grouping.get("deadtime_mode", grouping.get("deadtime_method", ""))).strip().lower()
        )
        if mode in {"file", "manual", "estimate"}:
            return mode
        if mode == "load":
            return "manual"
        return "file"

    def _set_deadtime_mode(self, mode: str) -> None:
        """Select the requested deadtime mode button when available."""
        button = self._deadtime_mode_buttons.get(str(mode).strip().lower())
        if button is None:
            button = self._deadtime_mode_buttons["file"]
        button.setChecked(True)

    def _current_deadtime_mode(self) -> str:
        """Return the active deadtime mode, or ``off`` when disabled."""
        if not self._deadtime_checkbox.isChecked():
            return "off"
        for mode, button in self._deadtime_mode_buttons.items():
            if button.isChecked():
                return mode
        return "file"

    def _initial_manual_deadtime_method(self, grouping: dict[str, Any]) -> str:
        """Return provenance for explicit manual deadtime values."""
        method = str(grouping.get("deadtime_method", "manual")).strip().lower()
        if method == "calibrate":
            return "calibrate"
        return "manual"

    def _histogram_deadtime_count(self) -> int:
        """Return the detector count for the current reference run."""
        if self._run is None:
            return 0
        return len(self._run.histograms)

    def _normalize_deadtime_values(self, values: object) -> list[float]:
        """Return finite deadtime values matching the reference histogram count."""
        if not isinstance(values, list):
            return []
        expected = self._histogram_deadtime_count()
        if expected <= 0 or len(values) != expected:
            return []
        normalized: list[float] = []
        for value in values:
            try:
                tau_us = float(value)
            except (TypeError, ValueError):
                return []
            if not np.isfinite(tau_us):
                return []
            normalized.append(max(0.0, tau_us))
        return normalized

    def _reference_file_deadtime_values(
        self, grouping: dict[str, Any] | None = None
    ) -> list[float]:
        """Return file deadtime values for the current reference run, if any."""
        if grouping is None:
            grouping = self._run.grouping if self._run is not None else {}
        if not self._reference_has_file_deadtime(grouping or {}):
            return []
        return self._normalize_deadtime_values((grouping or {}).get("dead_time_us"))

    def _initial_manual_deadtime_values(self, grouping: dict[str, Any]) -> list[float]:
        """Return initial explicit deadtime values for manual editing."""
        values = self._normalize_deadtime_values(grouping.get("dead_time_us"))
        mode = (
            str(grouping.get("deadtime_mode", grouping.get("deadtime_method", ""))).strip().lower()
        )
        method = str(grouping.get("deadtime_method", "")).strip().lower()
        if values and (mode in {"manual", "load"} or method in {"manual", "calibrate", "load"}):
            return values
        file_values = self._reference_file_deadtime_values(grouping)
        if file_values:
            return file_values
        n_histograms = self._histogram_deadtime_count()
        default_tau = float(grouping.get("deadtime_manual_us", 0.01) or 0.01)
        return [default_tau] * n_histograms

    def _formatted_deadtime_value_text(self, detector_index: int, tau_us: float) -> str:
        """Return combo label for a detector deadtime value."""
        return f"H{detector_index + 1}: {tau_us * 1000.0:.3f} ns"

    def _set_deadtime_combo_values(self, values_us: list[float]) -> None:
        """Populate the detector deadtime combo from values in microseconds."""
        current_index = self._deadtime_value_combo.currentIndex()
        self._updating_deadtime_value_combo = True
        self._deadtime_value_combo.blockSignals(True)
        self._deadtime_value_combo.clear()
        for index, tau_us in enumerate(values_us):
            self._deadtime_value_combo.addItem(
                self._formatted_deadtime_value_text(index, tau_us),
                index,
            )
        if values_us:
            self._deadtime_value_combo.setCurrentIndex(
                min(max(current_index, 0), len(values_us) - 1)
            )
        self._deadtime_value_combo.blockSignals(False)
        self._updating_deadtime_value_combo = False

    def _parse_deadtime_value_text(self, text: str) -> float | None:
        """Parse a detector deadtime edit string as nanoseconds."""
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match is None:
            return None
        try:
            value_ns = float(match.group(0))
        except ValueError:
            return None
        if not np.isfinite(value_ns) or value_ns <= 0.0:
            return None
        return value_ns

    def _current_deadtime_display_values(
        self, grouping: dict[str, Any] | None = None
    ) -> list[float]:
        """Return detector deadtime values that should be shown in the combo."""
        if grouping is None:
            grouping = self._run.grouping if self._run is not None else {}
        mode = self._current_deadtime_mode()
        if mode == "file":
            file_values = self._reference_file_deadtime_values(grouping)
            if file_values:
                return file_values
        if mode == "estimate":
            tau_us = self._estimate_deadtime_us_from_reference()
            if tau_us is not None and tau_us > 0.0:
                return [tau_us] * self._histogram_deadtime_count()
        if self._deadtime_manual_values_us:
            return list(self._deadtime_manual_values_us)
        return self._initial_manual_deadtime_values(grouping or {})

    def _on_deadtime_value_index_changed(self, index: int) -> None:
        """Normalize combo text when the current detector selection changes."""
        if self._updating_deadtime_value_combo or index < 0:
            return
        values = self._current_deadtime_display_values()
        if 0 <= index < len(values):
            self._deadtime_value_combo.setItemText(
                index,
                self._formatted_deadtime_value_text(index, values[index]),
            )

    def _on_deadtime_value_edited(self) -> None:
        """Commit manual deadtime edits from the current combo entry."""
        if self._updating_deadtime_value_combo:
            return
        if self._current_deadtime_mode() != "manual":
            return
        index = self._deadtime_value_combo.currentIndex()
        if index < 0 or index >= len(self._deadtime_manual_values_us):
            return
        value_ns = self._parse_deadtime_value_text(self._deadtime_value_combo.currentText())
        if value_ns is None:
            self._deadtime_value_combo.setItemText(
                index,
                self._formatted_deadtime_value_text(index, self._deadtime_manual_values_us[index]),
            )
            return
        self._deadtime_manual_values_us[index] = value_ns / 1000.0
        self._deadtime_manual_method = "manual"
        self._deadtime_value_combo.setItemText(
            index,
            self._formatted_deadtime_value_text(index, self._deadtime_manual_values_us[index]),
        )
        self._set_deadtime_status(
            f"Manual deadtime table ready for {len(self._deadtime_manual_values_us)} detectors."
        )

    def _estimate_deadtime_us_from_reference(self) -> float | None:
        """Estimate a uniform deadtime value from the current reference run."""
        if self._run is None or not self._run.histograms:
            return None
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        _t0_bin, t_good_offset, _first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        try:
            good_frames = float(grouping.get("good_frames", 1.0))
        except (TypeError, ValueError):
            good_frames = 1.0
        return estimate_deadtime_from_histograms(
            self._run.histograms,
            t_good_offset=t_good_offset,
            last_good_bin=last_good_bin,
            num_good_frames=good_frames,
        )

    def _resolve_deadtime_payload(self, *, show_warnings: bool = False) -> dict[str, Any] | None:
        """Return resolved deadtime payload for the current dialog state."""
        mode = self._current_deadtime_mode()
        if mode == "off":
            return {"deadtime_correction": False, "deadtime_mode": "off"}

        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        n_histograms = len(self._run.histograms) if self._run is not None else 0
        payload: dict[str, Any] = {"deadtime_correction": True, "deadtime_mode": mode}

        if mode == "file":
            if not self._reference_has_file_deadtime(grouping):
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Deadtime Unavailable",
                        "The reference run does not provide file deadtime values.",
                    )
                return None
            payload["deadtime_method"] = "file"
            return payload

        if mode == "manual":
            values_us = list(self._deadtime_manual_values_us)
            if not values_us or any(value <= 0.0 for value in values_us):
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Invalid Deadtime",
                        "Manual deadtime values must be greater than zero for every detector.",
                    )
                return None
            payload.update(
                {
                    "deadtime_method": self._deadtime_manual_method,
                    "deadtime_manual_us": float(values_us[0]),
                    "dead_time_us": values_us,
                }
            )
            if self._deadtime_manual_method == "calibrate":
                payload["deadtime_reference_run"] = int(self._reference_dataset.run_number)
            return payload

        if mode == "estimate":
            tau_us = self._estimate_deadtime_us_from_reference()
            if tau_us is None or tau_us <= 0.0:
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Deadtime Estimate Failed",
                        "The reference run did not provide enough valid early-time counts to estimate deadtime.",
                    )
                return None
            payload.update(
                {
                    "deadtime_method": "estimate",
                    "deadtime_estimated_us": tau_us,
                    "deadtime_reference_run": int(self._reference_dataset.run_number),
                    "dead_time_us": [tau_us] * n_histograms,
                }
            )
            return payload

        return None

    def _calibrate_deadtime_from_reference(self) -> None:
        """Calibrate a per-detector deadtime table from the reference run."""
        if self._run is None or not self._run.histograms:
            return
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        _t0_bin, t_good_offset, _first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        try:
            good_frames = float(grouping.get("good_frames", 1.0))
        except (TypeError, ValueError):
            good_frames = 1.0
        values = calibrate_deadtime_from_histograms(
            self._run.histograms,
            t_good_offset=t_good_offset,
            last_good_bin=last_good_bin,
            num_good_frames=good_frames,
        )
        if not values:
            QMessageBox.warning(
                self,
                "Deadtime Calibration Failed",
                "The reference run did not provide enough valid early-time counts to calibrate per-detector deadtime values.",
            )
            return
        self._deadtime_manual_values_us = list(values)
        self._deadtime_manual_method = "calibrate"
        self._set_deadtime_mode("manual")
        self._deadtime_checkbox.setChecked(True)
        self._update_deadtime_controls()

    def _set_deadtime_status(self, text: str) -> None:
        """Update helper text and keep enough height for wrapped content."""
        self._deadtime_status_label.setText(text)
        self._deadtime_status_label.updateGeometry()
        self._deadtime_mode_widget.updateGeometry()

    def _update_deadtime_controls(self, grouping: dict[str, Any] | None = None) -> None:
        """Refresh deadtime mode availability, editor state, and status text."""
        if grouping is None:
            grouping = self._run.grouping if self._run is not None else {}
        enabled = bool(self._deadtime_checkbox.isChecked())
        file_available = self._reference_has_file_deadtime(grouping or {})

        self._deadtime_checkbox.setToolTip(
            "Apply deadtime correction using the selected deadtime mode."
        )
        self._deadtime_mode_buttons["file"].setEnabled(enabled)
        self._deadtime_mode_buttons["manual"].setEnabled(enabled)
        self._deadtime_mode_buttons["estimate"].setEnabled(enabled and bool(self._run.histograms))

        mode = self._current_deadtime_mode()
        self._deadtime_value_combo.setEnabled(enabled)
        self._deadtime_value_combo.setEditable(enabled and mode == "manual")
        line_edit = self._deadtime_value_combo.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(not (enabled and mode == "manual"))
        self._deadtime_calibrate_btn.setEnabled(
            enabled and mode == "manual" and bool(self._run.histograms)
        )
        self._set_deadtime_combo_values(self._current_deadtime_display_values(grouping))

        if not enabled:
            self._set_deadtime_status("Deadtime disabled.")
            return
        if mode == "file":
            if file_available:
                self._set_deadtime_status("Use this run's file deadtime values.")
            else:
                self._set_deadtime_status(
                    "Use file deadtime values when the selected run provides them."
                )
            return
        if mode == "manual":
            self._set_deadtime_status(
                f"Edit the detector table directly, or use Cal to fit one deadtime value per detector from reference run {self._reference_dataset.run_number}."
            )
            return
        if mode == "estimate":
            self._set_deadtime_status(
                f"Estimate one value from reference run {self._reference_dataset.run_number} and apply it to all selected runs."
            )
            return

    def _update_background_checkbox_state(self) -> None:
        """Enable background correction for PSI-style grouped raw data."""
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        if self._run is not None:
            metadata.update(getattr(self._run, "metadata", {}) or {})
        source_file = str(getattr(self._run, "source_file", "") or metadata.get("source_file", ""))
        enabled = supports_background_correction(metadata=metadata, source_file=source_file)
        self._background_checkbox.setEnabled(enabled)
        if not enabled:
            self._background_checkbox.setChecked(False)
            self._background_checkbox.setToolTip(
                "Background correction is available for PSI BIN/MDU and PSI/LEM ROOT data."
            )
            return
        self._background_checkbox.setToolTip(
            "Apply grouped background subtraction before asymmetry."
        )

    def _on_apply(self) -> None:
        """Validate form values before accepting the dialog."""
        t0_bin, t_good_offset, first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        max_bin = self._max_bin_index_for_reference_dataset()
        if first_good_bin > max_bin:
            QMessageBox.warning(
                self,
                "Invalid Good-Data Window",
                "t_good offset places first good bin beyond the histogram range.",
            )
            return
        if last_good_bin < first_good_bin:
            QMessageBox.warning(
                self,
                "Invalid Good-Data Window",
                "Last good bin must be greater than or equal to t0 + t_good offset.",
            )
            return
        if (
            self._deadtime_checkbox.isChecked()
            and self._resolve_deadtime_payload(show_warnings=True) is None
        ):
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
            state = (
                Qt.CheckState.Checked if run_number == reference_run else Qt.CheckState.Unchecked
            )
            item.setCheckState(state)

    def _deselect_all_datasets(self) -> None:
        """Unmark all datasets for grouping application."""
        self._set_all_dataset_checkstates(Qt.CheckState.Unchecked)

    def _populate_group_table(self) -> None:
        """Render the detector-group table used as grouping context."""
        self._group_table.setRowCount(len(self._groups))
        for row, gid in enumerate(sorted(self._groups)):
            self._group_table.setItem(row, 0, QTableWidgetItem(str(gid)))
            include_item = QTableWidgetItem()
            include_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            include_item.setCheckState(
                Qt.CheckState.Checked
                if self._included_groups.get(int(gid), True)
                else Qt.CheckState.Unchecked
            )
            self._group_table.setItem(row, 1, include_item)
            name = self._group_names.get(gid, "")
            self._group_table.setItem(row, 2, QTableWidgetItem(name))
            detectors = [str(idx + 1) for idx in self._groups[gid]]
            self._group_table.setItem(row, 3, QTableWidgetItem(", ".join(detectors)))
        self._group_table.resizeColumnsToContents()
        visible_rows = min(max(len(self._groups), 3), 5)
        row_height = max(24, self._group_table.verticalHeader().defaultSectionSize())
        header_height = self._group_table.horizontalHeader().height()
        frame = 2 * self._group_table.frameWidth()
        height = header_height + visible_rows * row_height + frame + 8
        self._group_table.setMinimumHeight(0)
        self._group_table.setMaximumHeight(height)

    def _current_included_groups(self) -> dict[int, bool]:
        """Return the include-checkbox state from the group table."""
        included: dict[int, bool] = {}
        for row in range(self._group_table.rowCount()):
            gid_item = self._group_table.item(row, 0)
            include_item = self._group_table.item(row, 1)
            if gid_item is None or include_item is None:
                continue
            try:
                gid = int(gid_item.text())
            except (TypeError, ValueError):
                continue
            included[gid] = include_item.checkState() == Qt.CheckState.Checked
        for gid in self._groups:
            included.setdefault(int(gid), True)
        self._included_groups = included
        return dict(included)

    def _detect_vector_axis_pairs(self) -> dict[str, tuple[int, int]]:
        """Return vector-axis group pairs if canonical vector group names exist."""
        if not self._groups or not isinstance(self._group_names, dict):
            return {}

        by_name: dict[str, int] = {}
        for gid, name in self._group_names.items():
            try:
                gid_int = int(gid)
            except (TypeError, ValueError):
                continue
            by_name[str(name).strip().lower()] = gid_int

        def _find(*candidates: str) -> int | None:
            for cand in candidates:
                gid = by_name.get(cand)
                if gid in self._groups and self._groups.get(gid):
                    return gid
            return None

        pz_f = _find("pz forward")
        pz_b = _find("pz backward")
        py_t = _find("py top", "py up")
        py_b = _find("py bottom", "py down")
        px_l = _find("px left")
        px_r = _find("px right")
        if None in {pz_f, pz_b, py_t, py_b, px_l, px_r}:
            return {}
        return {
            "P_x": (int(px_l), int(px_r)),
            "P_y": (int(py_t), int(py_b)),
            "P_z": (int(pz_f), int(pz_b)),
        }

    def _vector_alpha_key(self, axis: str) -> str:
        """Return grouping payload key for a canonical vector axis."""
        return {
            "P_x": "alpha_x",
            "P_y": "alpha_y",
            "P_z": "alpha_z",
        }.get(axis, "alpha")

    def _legacy_vector_alpha_key(self, axis: str) -> str:
        """Return legacy payload key for backward compatibility."""
        return {
            "P_x": "alpha_px",
            "P_y": "alpha_py",
            "P_z": "alpha_pz",
        }.get(axis, "alpha")

    def _alpha_value_for_axis(self, grouping: dict[str, Any], axis: str, fallback: float) -> float:
        """Return best alpha value for *axis* from grouping payload/state."""
        key = self._vector_alpha_key(axis)
        legacy_key = self._legacy_vector_alpha_key(axis)
        try:
            return float(
                grouping.get(
                    key,
                    grouping.get(legacy_key, grouping.get("alpha", fallback)),
                )
            )
        except (TypeError, ValueError):
            return float(fallback)

    def _group_display_name(self, gid: int) -> str:
        """Return display text for a group ID with optional group name."""
        label = str(self._group_names.get(gid, "")).strip()
        if label and label != str(gid):
            return f"{gid}: {label}"
        return str(gid)

    def _refresh_group_combo_items(
        self,
        *,
        forward_gid: int | None = None,
        backward_gid: int | None = None,
    ) -> None:
        """Rebuild forward/backward group combo entries from current group state."""
        if forward_gid is None:
            try:
                forward_gid = int(self._forward_combo.currentData())
            except (TypeError, ValueError):
                forward_gid = 1
        if backward_gid is None:
            try:
                backward_gid = int(self._backward_combo.currentData())
            except (TypeError, ValueError):
                backward_gid = 2

        self._forward_combo.blockSignals(True)
        self._backward_combo.blockSignals(True)
        self._forward_combo.clear()
        self._backward_combo.clear()
        for gid in sorted(self._groups):
            display = self._group_display_name(gid)
            self._forward_combo.addItem(display, gid)
            self._backward_combo.addItem(display, gid)
        self._set_combo_to_group(self._forward_combo, int(forward_gid))
        self._set_combo_to_group(self._backward_combo, int(backward_gid))
        self._forward_combo.blockSignals(False)
        self._backward_combo.blockSignals(False)

    def _update_vector_mode_controls(self, grouping_values: dict[str, Any] | None = None) -> None:
        """Toggle between single-alpha and vector-alpha controls."""
        if grouping_values is None:
            grouping_values = {}

        pairs = self._detect_vector_axis_pairs()
        self._vector_axis_pairs = pairs
        vector_mode = bool(pairs)

        self._forward_row_label.setVisible(not vector_mode)
        self._forward_combo.setVisible(not vector_mode)
        self._backward_row_label.setVisible(not vector_mode)
        self._backward_combo.setVisible(not vector_mode)
        self._alpha_row_label.setVisible(not vector_mode)
        self._single_alpha_widget.setVisible(not vector_mode)
        self._vector_alpha_widget.setVisible(vector_mode)

        if not vector_mode:
            if "alpha" in grouping_values:
                try:
                    self._alpha_spin.setValue(float(grouping_values.get("alpha", 1.0)))
                except (TypeError, ValueError):
                    pass
            return

        fallback = float(self._alpha_spin.value())
        for axis in ("P_x", "P_y", "P_z"):
            fwd_gid, bwd_gid = pairs.get(axis, (None, None))
            if fwd_gid is not None:
                self._vector_forward_labels[axis].setText(self._group_display_name(int(fwd_gid)))
            else:
                self._vector_forward_labels[axis].setText("-")
            if bwd_gid is not None:
                self._vector_backward_labels[axis].setText(self._group_display_name(int(bwd_gid)))
            else:
                self._vector_backward_labels[axis].setText("-")

            spin = self._vector_alpha_spins[axis]
            try:
                value = float(
                    grouping_values.get(
                        self._vector_alpha_key(axis), grouping_values.get("alpha", spin.value())
                    )
                )
            except (TypeError, ValueError):
                value = self._alpha_value_for_axis(grouping_values, axis, fallback)
            spin.setValue(value)

        if "P_z" in pairs:
            pz_fwd, pz_bwd = pairs["P_z"]
            self._set_combo_to_group(self._forward_combo, pz_fwd)
            self._set_combo_to_group(self._backward_combo, pz_bwd)
            self._alpha_spin.setValue(float(self._vector_alpha_spins["P_z"].value()))

    def _on_detector_layout(self) -> None:
        """Open the interactive detector layout editor as a sub-dialog."""
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        # Determine number of histograms for instrument auto-detection
        n_histo = len(self._run.histograms) if self._run and self._run.histograms else 0
        grouping = self._run.grouping or {} if self._run else {}
        metadata = (
            dict(self._run.metadata) if self._run and isinstance(self._run.metadata, dict) else {}
        )
        if isinstance(grouping, dict):
            for key in ("histogram_labels", "group_names"):
                if key in grouping and key not in metadata:
                    metadata[key] = grouping[key]

        instrument_name = self._detector_layout_instrument_name
        instrument = None
        if instrument_name:
            try:
                instrument = get_instrument_layout(instrument_name)
            except KeyError:
                instrument_name = None

        if instrument is None:
            instrument_name = detect_instrument(
                n_histo,
                metadata=metadata,
                source_file=self._run.source_file if self._run else None,
            )
            try:
                instrument = get_instrument_layout(instrument_name) if instrument_name else None
            except KeyError:
                instrument = None

        if instrument is None:
            instrument = get_instrument_layout("HiFi")

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
        previous_included = dict(self._included_groups)
        self._included_groups = {
            int(gid): bool(previous_included.get(int(gid), True)) for gid in self._groups
        }
        self._group_names = result.get("group_names", {})
        preset_name = result.get("grouping_preset")
        self._grouping_preset_name = str(preset_name) if preset_name else None
        instrument_name = result.get("instrument")
        self._detector_layout_instrument_name = str(instrument_name) if instrument_name else None

        # Update forward/backward combos
        new_fwd = result.get("forward_group", forward_gid)
        new_bwd = result.get("backward_group", backward_gid)
        new_fwd, new_bwd = self._analysis_pair_for_reference(int(new_fwd), int(new_bwd))
        self._refresh_group_combo_items(forward_gid=int(new_fwd), backward_gid=int(new_bwd))

        self._populate_group_table()
        self._update_vector_mode_controls()

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
        alpha = self._estimate_alpha_for_group_ids(forward_gid, backward_gid)
        if alpha is not None:
            self._alpha_spin.setValue(float(alpha))

    def _estimate_alpha_for_group_ids(self, forward_gid: int, backward_gid: int) -> float | None:
        """Estimate alpha for the provided group IDs using the reference run."""
        if forward_gid == backward_gid:
            QMessageBox.warning(
                self, "Invalid Grouping", "Forward and backward groups must differ."
            )
            return None

        forward_indices = self._groups.get(forward_gid, [])
        backward_indices = self._groups.get(backward_gid, [])
        if not forward_indices or not backward_indices:
            QMessageBox.warning(self, "Invalid Grouping", "Selected groups are empty.")
            return None

        if self._run is None or not self._run.histograms:
            QMessageBox.warning(self, "Estimate Failed", "Reference run has no histograms.")
            return None

        if max(forward_indices, default=-1) >= len(self._run.histograms):
            QMessageBox.warning(
                self, "Estimate Failed", "Forward group exceeds detector count for reference run."
            )
            return None
        if max(backward_indices, default=-1) >= len(self._run.histograms):
            QMessageBox.warning(
                self, "Estimate Failed", "Backward group exceeds detector count for reference run."
            )
            return None

        forward_counts = apply_grouping(self._run.histograms, forward_indices)
        backward_counts = apply_grouping(self._run.histograms, backward_indices)

        _t0_bin, _t_good_offset, first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        return float(
            estimate_alpha(
                forward_counts,
                backward_counts,
                first_good_bin=first_good_bin,
                last_good_bin=last_good_bin,
            )
        )

    def _estimate_alpha_for_axis(self, axis: str) -> None:
        """Estimate alpha for one vector polarization axis pair."""
        pair = self._vector_axis_pairs.get(axis)
        if pair is None:
            QMessageBox.warning(
                self, "Estimate Failed", f"No vector grouping pair is available for {axis}."
            )
            return
        alpha = self._estimate_alpha_for_group_ids(int(pair[0]), int(pair[1]))
        if alpha is not None:
            self._vector_alpha_spins[axis].setValue(alpha)
            if axis == "P_z":
                self._alpha_spin.setValue(alpha)

    def _estimate_all_alpha(self) -> None:
        """Estimate alpha for all vector polarization axes."""
        for axis in ("P_x", "P_y", "P_z"):
            if axis in self._vector_axis_pairs:
                self._estimate_alpha_for_axis(axis)

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
        t0_bin, t_good_offset, first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        deadtime_payload = self._resolve_deadtime_payload() or {
            "deadtime_correction": False,
            "deadtime_mode": "off",
        }
        vector_mode = bool(self._vector_axis_pairs)
        if vector_mode and "P_z" in self._vector_axis_pairs:
            forward_gid, backward_gid = self._vector_axis_pairs["P_z"]
            self._set_combo_to_group(self._forward_combo, int(forward_gid))
            self._set_combo_to_group(self._backward_combo, int(backward_gid))

        alpha_value = float(self._alpha_spin.value())
        if vector_mode:
            alpha_value = float(self._vector_alpha_spins["P_z"].value())

        return (
            {
                "groups": {
                    gid: [idx + 1 for idx in values] for gid, values in self._groups.items()
                },
                "included_groups": self._current_included_groups(),
                "group_names": dict(self._group_names),
                "grouping_preset": self._grouping_preset_name,
                "instrument": self._detector_layout_instrument_name,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "forward_indices": list(self._groups.get(forward_gid, [])),
                "backward_indices": list(self._groups.get(backward_gid, [])),
                "alpha": alpha_value,
                "t0_bin": int(t0_bin),
                "t_good_offset": int(t_good_offset),
                "first_good_bin": int(first_good_bin),
                "last_good_bin": int(last_good_bin),
                "bunching_factor": int(self._bunch_spin.value()),
                "background_correction": bool(
                    self._background_checkbox.isEnabled() and self._background_checkbox.isChecked()
                ),
                "period_mode": self._current_period_mode(),
                "bin_index_base": self._bin_index_base(),
            }
            | (
                {
                    "alpha_x": float(self._vector_alpha_spins["P_x"].value()),
                    "alpha_y": float(self._vector_alpha_spins["P_y"].value()),
                    "alpha_z": float(self._vector_alpha_spins["P_z"].value()),
                }
                if vector_mode
                else {}
            )
            | deadtime_payload
        )

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

        with open(path, encoding="utf-8") as handle:
            payload = self.parse_grp(handle.read())

        loaded_index_base = (
            1 if int(payload.get("bin_index_base", self._bin_index_base())) == 1 else 0
        )
        if isinstance(self._run.grouping, dict):
            self._run.grouping["bin_index_base"] = loaded_index_base

        loaded_groups = payload.get("groups", {})
        if not isinstance(loaded_groups, dict) or len(loaded_groups) < 2:
            QMessageBox.warning(
                self, "Invalid Grouping", "Loaded .grp file does not define at least two groups."
            )
            return

        groups: dict[int, list[int]] = {}
        for key, dets in loaded_groups.items():
            gid = int(key)
            idxs = [max(0, int(v) - 1) for v in dets]
            groups[gid] = sorted(set(idxs))

        self._groups = groups
        loaded_included_groups = payload.get("included_groups", {})
        if isinstance(loaded_included_groups, dict):
            self._included_groups = {
                int(k): bool(v) for k, v in loaded_included_groups.items() if str(k).strip()
            }
        else:
            self._included_groups = {int(gid): True for gid in groups}
        self._populate_group_table()
        self._refresh_group_combo_items(
            forward_gid=int(payload.get("forward_group", 1)),
            backward_gid=int(payload.get("backward_group", 2)),
        )

        loaded_group_names = payload.get("group_names", {})
        if isinstance(loaded_group_names, dict):
            self._group_names = {int(k): str(v) for k, v in loaded_group_names.items()}
            self._refresh_group_combo_items(
                forward_gid=int(payload.get("forward_group", 1)),
                backward_gid=int(payload.get("backward_group", 2)),
            )
        self._alpha_spin.setValue(float(payload.get("alpha", 1.0)))
        if "alpha_x" in payload or "alpha_px" in payload:
            self._vector_alpha_spins["P_x"].setValue(
                float(payload.get("alpha_x", payload.get("alpha_px", payload.get("alpha", 1.0))))
            )
        if "alpha_y" in payload or "alpha_py" in payload:
            self._vector_alpha_spins["P_y"].setValue(
                float(payload.get("alpha_y", payload.get("alpha_py", payload.get("alpha", 1.0))))
            )
        if "alpha_z" in payload or "alpha_pz" in payload:
            self._vector_alpha_spins["P_z"].setValue(
                float(payload.get("alpha_z", payload.get("alpha_pz", payload.get("alpha", 1.0))))
            )

        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = loaded_index_base
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._last_good_spin.setRange(index_base, max_bin + index_base)

        t0_bin = int(payload.get("t0_bin", self._t0_spin.value() - index_base))
        t0_bin = max(0, min(max_bin, t0_bin))
        self._t0_spin.setValue(t0_bin + index_base)
        self._on_t0_changed()

        raw_offset = payload.get("t_good_offset")
        if raw_offset is None:
            try:
                raw_offset = int(payload.get("first_good_bin", t0_bin)) - t0_bin
            except (TypeError, ValueError):
                raw_offset = self._t_good_offset_spin.value()
        t_good_offset = max(0, int(raw_offset))
        t_good_offset = min(t_good_offset, int(self._t_good_offset_spin.maximum()))
        self._t_good_offset_spin.setValue(t_good_offset)

        last_good_bin = int(payload.get("last_good_bin", self._last_good_spin.value() - index_base))
        last_good_bin = max(0, min(max_bin, last_good_bin))
        self._last_good_spin.setValue(last_good_bin + index_base)
        self._bunch_spin.setValue(int(payload.get("bunching_factor", self._bunch_spin.value())))
        self._deadtime_checkbox.setChecked(bool(payload.get("deadtime_correction", False)))
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(payload)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(payload)
        self._set_deadtime_mode(self._default_deadtime_mode(payload))
        self._update_deadtime_controls(payload)
        self._background_checkbox.setChecked(bool(payload.get("background_correction", False)))
        self._update_background_checkbox_state()
        self._set_period_mode(str(payload.get("period_mode", PeriodMode.RED)))
        self._populate_group_table()
        self._update_vector_mode_controls(payload)

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
            f"alpha_x={float(payload.get('alpha_x', payload.get('alpha', 1.0))):.12g}",
            f"alpha_y={float(payload.get('alpha_y', payload.get('alpha', 1.0))):.12g}",
            f"alpha_z={float(payload.get('alpha_z', payload.get('alpha', 1.0))):.12g}",
            f"t0_bin={int(payload.get('t0_bin', 0))}",
            f"t_good_offset={int(payload.get('t_good_offset', int(payload.get('first_good_bin', 0)) - int(payload.get('t0_bin', 0))))}",
            f"first_good_bin={int(payload.get('first_good_bin', 0))}",
            f"last_good_bin={int(payload.get('last_good_bin', 0))}",
            f"bunching_factor={int(payload.get('bunching_factor', 1))}",
            f"bin_index_base={1 if int(payload.get('bin_index_base', 0)) == 1 else 0}",
            f"deadtime_correction={1 if bool(payload.get('deadtime_correction', False)) else 0}",
            f"deadtime_mode={str(payload.get('deadtime_mode', 'off'))}",
            f"deadtime_method={str(payload.get('deadtime_method', ''))}",
            f"deadtime_manual_us={float(payload.get('deadtime_manual_us', 0.0)):.12g}",
            f"deadtime_estimated_us={float(payload.get('deadtime_estimated_us', 0.0)):.12g}",
            f"background_correction={1 if bool(payload.get('background_correction', False)) else 0}",
            f"period_mode={str(payload.get('period_mode', PeriodMode.RED))}",
        ]

        if "deadtime_reference_run" in payload:
            lines.append(f"deadtime_reference_run={int(payload.get('deadtime_reference_run', 0))}")
        dead_time_us = payload.get("dead_time_us")
        if isinstance(dead_time_us, list) and dead_time_us:
            values = ",".join(f"{float(value):.12g}" for value in dead_time_us)
            lines.append(f"dead_time_us={values}")

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
        included_groups = payload.get("included_groups", {})
        if isinstance(included_groups, dict):
            for gid in sorted(int(k) for k in included_groups.keys()):
                include = 1 if bool(included_groups.get(gid, True)) else 0
                lines.append(f"group_include.{gid}={include}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_grp(text: str) -> dict[str, Any]:
        """Parse line-based ``.grp`` text into a grouping payload dictionary."""
        payload: dict[str, Any] = {
            "groups": {},
            "included_groups": {},
            "group_names": {},
            "forward_group": 1,
            "backward_group": 2,
            "alpha": 1.0,
            "t0_bin": 0,
            "t_good_offset": 0,
            "first_good_bin": 0,
            "last_good_bin": 0,
            "bunching_factor": 1,
            "deadtime_correction": False,
            "deadtime_mode": "off",
            "background_correction": False,
            "period_mode": str(PeriodMode.RED),
            "bin_index_base": 0,
        }
        saw_t_good_offset = False

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

            if key.startswith("group_include."):
                gid = int(key.split(".", 1)[1])
                payload["included_groups"][gid] = value.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                continue

            if key.startswith("group."):
                gid = int(key.split(".", 1)[1])
                dets = [int(v.strip()) for v in value.split(",") if v.strip()]
                payload["groups"][gid] = dets
                continue

            if key in {
                "forward_group",
                "backward_group",
                "t0_bin",
                "t_good_offset",
                "first_good_bin",
                "last_good_bin",
                "bunching_factor",
                "bin_index_base",
            }:
                payload[key] = int(float(value))
                if key == "t_good_offset":
                    saw_t_good_offset = True
            elif key in {
                "alpha",
                "alpha_x",
                "alpha_y",
                "alpha_z",
                "alpha_px",
                "alpha_py",
                "alpha_pz",
                "deadtime_manual_us",
                "deadtime_estimated_us",
            }:
                payload[key] = float(value)
            elif key in {"deadtime_mode", "deadtime_method", "deadtime_source_path"}:
                payload[key] = value
            elif key == "deadtime_reference_run":
                payload[key] = int(float(value))
            elif key in {"dead_time_us", "deadtime_loaded_us"}:
                payload[key] = [float(v.strip()) for v in value.split(",") if v.strip()]
            elif key == "deadtime_correction":
                payload[key] = value.strip().lower() in {"1", "true", "yes", "on"}
            elif key == "background_correction":
                payload[key] = value.strip().lower() in {"1", "true", "yes", "on"}
            elif key == "period_mode":
                if value in {
                    str(PeriodMode.RED),
                    str(PeriodMode.GREEN),
                    str(PeriodMode.GREEN_MINUS_RED),
                    str(PeriodMode.GREEN_PLUS_RED),
                }:
                    payload[key] = value

        alpha_scalar = float(payload.get("alpha", 1.0))
        payload.setdefault("alpha_x", float(payload.get("alpha_px", alpha_scalar)))
        payload.setdefault("alpha_y", float(payload.get("alpha_py", alpha_scalar)))
        payload.setdefault("alpha_z", float(payload.get("alpha_pz", alpha_scalar)))
        t0_bin = int(payload.get("t0_bin", 0))
        if saw_t_good_offset:
            payload["t_good_offset"] = max(0, int(payload.get("t_good_offset", 0)))
        else:
            payload["t_good_offset"] = max(0, int(payload.get("first_good_bin", 0)) - t0_bin)
        payload["first_good_bin"] = max(0, t0_bin + int(payload.get("t_good_offset", 0)))
        payload["bin_index_base"] = 1 if int(payload.get("bin_index_base", 0)) == 1 else 0
        if isinstance(payload.get("groups"), dict):
            included_groups = payload.get("included_groups")
            if not isinstance(included_groups, dict):
                included_groups = {}
            payload["included_groups"] = {
                int(gid): bool(included_groups.get(int(gid), True)) for gid in payload["groups"]
            }

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
