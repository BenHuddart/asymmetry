"""Data browser / logbook panel with dataset grouping support."""

from __future__ import annotations

import copy
import csv
import json
import uuid
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset, Run

_GROUP_TEMP_ABS_TOL_K = 5e-3
_GROUP_TEMP_REL_TOL = 2e-3
_GROUP_FIELD_ABS_TOL_G = 1e-3
_LOG_TEMPERATURE_FOREGROUND = QColor(176, 36, 36)
_GROUP_FIELD_REL_TOL = 1e-4


def _is_effectively_constant(values: list[float], *, abs_tol: float, rel_tol: float) -> bool:
    """Return True when finite values vary only within tolerance.

    Uses a combined absolute/relative tolerance so low-value groups (e.g.
    0.1 K) and higher-value groups (e.g. 20 K with small drift) are both
    handled robustly.
    """
    if not values:
        return False
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return False

    center = float(np.nanmedian(arr))
    span = float(np.nanmax(arr) - np.nanmin(arr))
    tolerance = max(float(abs_tol), float(rel_tol) * max(abs(center), 1.0))
    return span <= tolerance


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically."""

    def __init__(self, value: float | int | str):
        super().__init__(str(value))
        try:
            self._numeric_value = float(value)
        except (ValueError, TypeError):
            self._numeric_value = 0.0

    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self._numeric_value < other._numeric_value

        other_text = other.text() if isinstance(other, QTableWidgetItem) else str(other)
        try:
            other_numeric = float(other_text)
            return self._numeric_value < other_numeric
        except (ValueError, TypeError):
            return self.text() < other_text


@dataclass
class DataGroup:
    group_id: str
    name: str
    member_run_numbers: list[int]
    collapsed: bool = False


class FilterDialog(QDialog):
    """Excel-style filter dialog with checkboxes for unique values."""

    def __init__(
        self,
        column_name: str,
        unique_values: list[str],
        current_selection: set[str] | None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Filter - {column_name}")
        self.setMinimumWidth(300)
        self.setMinimumHeight(400)

        self._checkboxes: list[QCheckBox] = []

        layout = QVBoxLayout(self)

        self._all_checkbox = QCheckBox("(Select All)")
        self._all_checkbox.setChecked(current_selection is None)
        self._all_checkbox.stateChanged.connect(self._on_all_changed)
        layout.addWidget(self._all_checkbox)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        for value in unique_values:
            checkbox = QCheckBox(value)
            checkbox.setChecked(current_selection is None or value in current_selection)
            checkbox.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        clear_btn = QPushButton("Clear Filter")
        clear_btn.clicked.connect(self._clear_filter)
        button_layout.addWidget(clear_btn)

        layout.addLayout(button_layout)

    def _on_all_changed(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _on_checkbox_changed(self) -> None:
        all_checked = all(cb.isChecked() for cb in self._checkboxes)
        none_checked = not any(cb.isChecked() for cb in self._checkboxes)

        self._all_checkbox.blockSignals(True)
        if all_checked:
            self._all_checkbox.setCheckState(Qt.CheckState.Checked)
        elif none_checked:
            self._all_checkbox.setCheckState(Qt.CheckState.Unchecked)
        else:
            self._all_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self._all_checkbox.blockSignals(False)

    def _clear_filter(self) -> None:
        self._all_checkbox.setChecked(True)
        self.done(QDialog.DialogCode.Accepted)

    def get_selected_values(self) -> set[str] | None:
        if all(cb.isChecked() for cb in self._checkboxes):
            return None
        return {checkbox.text() for checkbox in self._checkboxes if checkbox.isChecked()}


class DataBrowserPanel(QWidget):
    """Logbook-style run table with grouping, sorting, filtering and co-add."""

    dataset_selected = Signal(int)
    selection_changed = Signal()
    group_selected = Signal(str)
    get_info_requested = Signal(int)
    grouping_requested = Signal(int)

    _COLUMNS = ["Run", "Title", "𝑇 (K)", "𝐵 (G)", "Comment"]
    _RUN_INFO_FIELD_LABELS = {
        "instrument": "Instrument",
        "run_label": "Run",
        "title": "Title",
        "comment": "Comment",
        "started": "Start",
        "stopped": "End",
        "temperature": "Temperature (K)",
        "field": "Magnetic Field (G)",
        "field_direction": "Field Direction",
        "period_count": "Periods",
        "run_info.points": "Points",
        "run_info.histograms": "Histograms",
        "run_info.bins": "Bins",
        "run_info.bin_width_us": "Bin Width (us)",
        "run_info.counts_mev": "Counts (MEv)",
        "run_info.counts_per_detector": "Counts per Detector",
        "nexus_fields.sample.shape": "Orientation",
    }
    _BASE_COLUMN_OVERRIDE_KEYS = {"temperature"}
    _GROUP_ROLE = Qt.ItemDataRole.UserRole
    _GROUP_SENTINEL_PREFIX = "group:"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: dict[int, MuonDataset] = {}
        self._combined_datasets: dict[int, list[int]] = {}
        self._combined_source_datasets: dict[int, list[MuonDataset]] = {}
        self._next_combined_id = -1

        self._groups: dict[str, DataGroup] = {}
        self._run_to_group: dict[int, str] = {}
        self._display_order: list[int | str] = []

        self._column_filters: dict[int, set[str]] = {}
        self._extra_columns: list[str] = []
        self._use_temperature_from_log = False
        self._temperature_from_log_overrides: dict[int, bool] = {}
        self._current_sort_column: int = -1
        self._current_sort_order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
        self._selection_anchor_row: int | None = None
        self._updating_table = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._refresh_column_headers()
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.SelectedClicked
        )

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 110)
        header.resizeSection(1, 145)
        header.resizeSection(2, 60)
        header.resizeSection(3, 60)
        header.resizeSection(4, 155)
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setSortIndicatorShown(True)
        self._table.horizontalHeader().setSectionsClickable(False)
        self._table.horizontalHeader().viewport().installEventFilter(self)
        self._table.viewport().installEventFilter(self)
        self._table.installEventFilter(self)

        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.viewport().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_table_context_menu)
        self._table.viewport().customContextMenuRequested.connect(self._show_table_context_menu)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self._table)
        self.setMinimumWidth(250)

    # ------------------------------------------------------------------
    # Dataset and grouping CRUD
    # ------------------------------------------------------------------

    def add_dataset(self, dataset: MuonDataset) -> None:
        rn = int(dataset.run_number)
        self._datasets[rn] = dataset
        if rn not in self._display_order and rn not in self._run_to_group:
            self._display_order.append(rn)
        if self._current_sort_column >= 0 and not self._groups:
            self._sort_table(rebuild=False)
        self._rebuild_table()
        self._resize_columns_to_content()

    def create_data_group(
        self,
        run_numbers: list[int],
        name: str | None = None,
        group_id: str | None = None,
        collapsed: bool = False,
    ) -> str | None:
        valid_runs = [rn for rn in run_numbers if rn in self._datasets]
        if len(valid_runs) < 2:
            return None

        gid = group_id or str(uuid.uuid4())
        if gid in self._groups:
            return None

        for rn in valid_runs:
            old_gid = self._run_to_group.get(rn)
            if old_gid is not None:
                self._remove_run_from_group(rn, old_gid)

        if not name:
            name = self._default_group_name(valid_runs)

        first_index = min(self._display_index_for_run(rn) for rn in valid_runs)
        for rn in valid_runs:
            if rn in self._display_order:
                self._display_order.remove(rn)

        self._display_order.insert(first_index, gid)
        self._groups[gid] = DataGroup(
            group_id=gid,
            name=name,
            member_run_numbers=list(valid_runs),
            collapsed=collapsed,
        )
        for rn in valid_runs:
            self._run_to_group[rn] = gid

        self._move_groups_to_top()

        self._rebuild_table()
        return gid

    def ungroup(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return

        insert_index = (
            self._display_order.index(group_id)
            if group_id in self._display_order
            else len(self._display_order)
        )
        if group_id in self._display_order:
            self._display_order.remove(group_id)

        for offset, rn in enumerate(group.member_run_numbers):
            self._run_to_group.pop(rn, None)
            if rn in self._datasets:
                self._display_order.insert(insert_index + offset, rn)

        self._groups.pop(group_id, None)
        self._move_groups_to_top()
        self._rebuild_table()

    def _move_groups_to_top(self) -> None:
        """Keep all group headers above non-grouped rows in display order."""
        groups = [
            entry
            for entry in self._display_order
            if isinstance(entry, str) and entry in self._groups
        ]
        runs = [entry for entry in self._display_order if isinstance(entry, int)]
        self._display_order = groups + runs

    def _remove_run_from_group(self, run_number: int, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        group.member_run_numbers = [rn for rn in group.member_run_numbers if rn != run_number]
        self._run_to_group.pop(run_number, None)
        if len(group.member_run_numbers) == 0:
            self.ungroup(group_id)

    def add_runs_to_group(self, run_numbers: list[int], group_id: str) -> bool:
        """Add existing dataset run rows into an existing group.

        Parameters
        ----------
        run_numbers
            Dataset run numbers to move.
        group_id
            Target group identifier.

        Returns
        -------
        bool
            ``True`` if at least one run was moved.
        """
        group = self._groups.get(group_id)
        if group is None:
            return False

        moved_any = False
        for rn in run_numbers:
            if rn not in self._datasets:
                continue
            if self._run_to_group.get(rn) == group_id:
                continue

            old_gid = self._run_to_group.get(rn)
            if old_gid is not None:
                self._remove_run_from_group(rn, old_gid)

            if rn in self._display_order:
                self._display_order.remove(rn)
            self._run_to_group[rn] = group_id
            group.member_run_numbers.append(rn)
            moved_any = True

        if moved_any:
            self._move_groups_to_top()
            self._rebuild_table()
        return moved_any

    def remove_runs_from_group(self, run_numbers: list[int]) -> bool:
        """Remove selected runs from their current groups and move to top-level list."""
        moved_any = False
        insert_at = len(self._display_order)
        for rn in run_numbers:
            gid = self._run_to_group.get(rn)
            if gid is None or rn not in self._datasets:
                continue
            group_index = (
                self._display_order.index(gid)
                if gid in self._display_order
                else len(self._display_order)
            )
            insert_at = min(insert_at, group_index + 1)
            self._remove_run_from_group(rn, gid)
            if rn not in self._display_order:
                self._display_order.insert(insert_at, rn)
                insert_at += 1
            moved_any = True

        if moved_any:
            self._move_groups_to_top()
            self._rebuild_table()
        return moved_any

    def _default_group_name(self, run_numbers: list[int]) -> str:
        datasets = [self._datasets[rn] for rn in run_numbers if rn in self._datasets]
        if not datasets:
            return f"Group {len(self._groups) + 1}"

        temps = [float(ds.metadata.get("temperature", np.nan)) for ds in datasets]
        fields = [float(ds.metadata.get("field", np.nan)) for ds in datasets]
        if _is_effectively_constant(
            temps,
            abs_tol=_GROUP_TEMP_ABS_TOL_K,
            rel_tol=_GROUP_TEMP_REL_TOL,
        ):
            return f"T = {float(np.nanmedian(temps)):.6g} K"
        if _is_effectively_constant(
            fields,
            abs_tol=_GROUP_FIELD_ABS_TOL_G,
            rel_tol=_GROUP_FIELD_REL_TOL,
        ):
            return f"B = {float(np.nanmedian(fields)):.6g} G"
        return f"Group {len(self._groups) + 1}"

    def _display_index_for_run(self, run_number: int) -> int:
        gid = self._run_to_group.get(run_number)
        if gid is not None and gid in self._display_order:
            return self._display_order.index(gid)
        if run_number in self._display_order:
            return self._display_order.index(run_number)
        return len(self._display_order)

    # ------------------------------------------------------------------
    # Table building
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        selected_keys = self._selected_keys()

        self._updating_table = True
        self._table.setRowCount(0)

        for entry in self._display_order:
            if isinstance(entry, str):
                self._add_group_header_row(entry)
                group = self._groups.get(entry)
                if group is None:
                    continue
                if not group.collapsed:
                    for rn in group.member_run_numbers:
                        if rn in self._datasets:
                            self._add_dataset_row(self._datasets[rn], indent=True)
            else:
                dataset = self._datasets.get(entry)
                if dataset is not None:
                    self._add_dataset_row(dataset, indent=False)

        self._updating_table = False
        self._apply_row_visibility()
        self._restore_selection_by_keys(selected_keys)

    def _add_group_header_row(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return

        row = self._table.rowCount()
        self._table.insertRow(row)

        prefix = "▸" if group.collapsed else "▾"
        run_item = QTableWidgetItem(f"{prefix} {group.name}")
        run_item.setData(self._GROUP_ROLE, f"{self._GROUP_SENTINEL_PREFIX}{group.group_id}")
        run_item.setFlags(
            (run_item.flags() & ~Qt.ItemFlag.ItemIsEditable) | Qt.ItemFlag.ItemIsSelectable
        )
        font = run_item.font()
        font.setBold(True)
        run_item.setFont(font)
        shade = QColor(230, 236, 245)
        run_item.setBackground(shade)
        self._table.setItem(row, 0, run_item)

        count_item = QTableWidgetItem(f"({len(group.member_run_numbers)} datasets)")
        count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        count_item.setFont(font)
        count_item.setBackground(shade)
        self._table.setItem(row, 1, count_item)

        for col in range(2, self._table.columnCount()):
            blank = QTableWidgetItem("")
            blank.setFlags(blank.flags() & ~Qt.ItemFlag.ItemIsEditable)
            blank.setFont(font)
            blank.setBackground(shade)
            self._table.setItem(row, col, blank)

    def _add_dataset_row(self, dataset: MuonDataset, *, indent: bool) -> None:
        rn = int(dataset.run_number)
        meta = dataset.metadata
        run_display = str(dataset.run_label)
        if rn in self._combined_datasets:
            run_display = " + ".join(map(str, self._combined_datasets[rn]))
        if indent:
            run_display = f"    {run_display}"

        row = self._table.rowCount()
        self._table.insertRow(row)

        if rn in self._combined_datasets or dataset.run_label != str(rn):
            run_item = QTableWidgetItem(run_display)
        else:
            run_item = NumericTableWidgetItem(run_display)
        run_item.setData(self._GROUP_ROLE, rn)
        run_item.setFlags(run_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, run_item)

        title = str(meta.get("title", ""))
        title_item = QTableWidgetItem(title)
        title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 1, title_item)

        temp = self._temperature_for_display(dataset)
        temp_item = NumericTableWidgetItem(f"{temp:.2f}")
        temp_item.setFlags(temp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if self._temperature_uses_log_for_display(dataset):
            temp_item.setForeground(_LOG_TEMPERATURE_FOREGROUND)
        self._table.setItem(row, 2, temp_item)

        field = float(meta.get("field", 0.0))
        field_item = NumericTableWidgetItem(f"{field:.1f}")
        field_item.setFlags(field_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 3, field_item)

        comment = str(meta.get("comment", ""))
        comment_item = QTableWidgetItem(comment)
        comment_item.setFlags(comment_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 4, comment_item)
        for i, field_key in enumerate(self._visible_extra_columns(), start=len(self._COLUMNS)):
            value = self._value_for_extra_column(dataset, field_key)
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, i, item)

    def _selected_keys(self) -> list[int | str]:
        selected: list[int | str] = []
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return selected
        for idx in selection_model.selectedRows():
            item = self._table.item(idx.row(), 0)
            if item is None:
                continue
            key = item.data(self._GROUP_ROLE)
            if isinstance(key, (int, str)):
                selected.append(key)
        return selected

    def _restore_selection_by_keys(self, keys: list[int | str]) -> None:
        if not keys:
            return
        wanted = set(keys)
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return
        selected_any = False
        with QSignalBlocker(self._table):
            self._table.clearSelection()
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item is None:
                    continue
                key = item.data(self._GROUP_ROLE)
                if key in wanted:
                    idx = self._table.model().index(row, 0)
                    selection_model.select(
                        idx,
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    selected_any = True
        if selected_any:
            self._on_selection_changed()

    def _is_row_visible_for_selection(self, row: int) -> bool:
        return 0 <= row < self._table.rowCount() and not self._table.isRowHidden(row)

    def _next_visible_row(self, start_row: int, direction: int) -> int | None:
        if direction == 0:
            return start_row if self._is_row_visible_for_selection(start_row) else None

        row = start_row + direction
        while 0 <= row < self._table.rowCount():
            if self._is_row_visible_for_selection(row):
                return row
            row += direction
        return None

    def _selection_anchor_for_row(self, fallback_row: int) -> int:
        anchor_row = self._selection_anchor_row
        if anchor_row is not None and self._is_row_visible_for_selection(anchor_row):
            return anchor_row

        current_row = self._table.currentRow()
        if self._is_row_visible_for_selection(current_row):
            return current_row

        return fallback_row

    def _select_visible_row_range(
        self,
        anchor_row: int,
        target_row: int,
        *,
        add_to_selection: bool,
    ) -> bool:
        if not (
            self._is_row_visible_for_selection(anchor_row)
            and self._is_row_visible_for_selection(target_row)
        ):
            return False

        selection_model = self._table.selectionModel()
        if selection_model is None:
            return False

        start_row = min(anchor_row, target_row)
        end_row = max(anchor_row, target_row)
        visible_rows = [
            row for row in range(start_row, end_row + 1) if self._is_row_visible_for_selection(row)
        ]
        if not visible_rows:
            return False

        with QSignalBlocker(self._table):
            for index, row in enumerate(visible_rows):
                row_index = self._table.model().index(row, 0)
                flags = QItemSelectionModel.SelectionFlag.Rows
                if add_to_selection or index > 0:
                    flags |= QItemSelectionModel.SelectionFlag.Select
                else:
                    flags |= QItemSelectionModel.SelectionFlag.ClearAndSelect
                selection_model.select(row_index, flags)

            current_index = self._table.model().index(target_row, 0)
            selection_model.setCurrentIndex(
                current_index,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        self._on_selection_changed()
        return True

    def _resize_columns_to_content(self) -> None:
        self._table.resizeColumnsToContents()
        header = self._table.horizontalHeader()
        minimums = {0: 90, 1: 145, 2: 60, 3: 60, 4: 155}
        maximums = {0: 180, 1: 260, 2: 90, 3: 90, 4: 320}
        for col, min_width in minimums.items():
            size = header.sectionSize(col)
            if size < min_width:
                header.resizeSection(col, min_width)
            elif size > maximums[col]:
                header.resizeSection(col, maximums[col])

        for col in range(len(self._COLUMNS), self._table.columnCount()):
            size = header.sectionSize(col)
            if size < 120:
                header.resizeSection(col, 120)
            elif size > 320:
                header.resizeSection(col, 320)

    def _refresh_column_headers(self) -> None:
        """Apply base and dynamic column labels to the table header."""
        labels = list(self._COLUMNS) + [
            self._extra_column_header(key) for key in self._visible_extra_columns()
        ]
        self._table.setColumnCount(len(labels))
        self._table.setHorizontalHeaderLabels(labels)

    def _visible_extra_columns(self) -> list[str]:
        """Return extra columns that should appear beyond the fixed browser columns."""
        return [key for key in self._extra_columns if key not in self._BASE_COLUMN_OVERRIDE_KEYS]

    def _extra_column_header(self, field_key: str) -> str:
        """Return display header for an extra metadata-backed column."""
        key = str(field_key).strip()
        if not key:
            return ""
        return self._RUN_INFO_FIELD_LABELS.get(key, key)

    def add_extra_column(self, field_key: str) -> None:
        """Add a metadata-backed dynamic column to the browser table."""
        key = str(field_key).strip()
        if key == "temperature":
            self.set_use_temperature_from_log(True)
            return
        if not key or key in self._extra_columns:
            return
        self._extra_columns.append(key)
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()

    def remove_extra_column(self, field_key: str) -> None:
        """Remove a dynamic metadata column from the browser table."""
        if field_key == "temperature":
            self.set_use_temperature_from_log(False)
            return
        if field_key not in self._extra_columns:
            return
        self._extra_columns = [key for key in self._extra_columns if key != field_key]
        self._refresh_column_headers()
        self._rebuild_table()
        self._resize_columns_to_content()

    def get_extra_columns(self) -> list[str]:
        """Return the current metadata-backed extra columns."""
        columns = list(self._extra_columns)
        if self._use_temperature_from_log:
            columns.append("temperature")
        return columns

    def set_use_temperature_from_log(self, enabled: bool) -> None:
        """Set the global temperature-from-log display option."""
        enabled = bool(enabled)
        changed = self._use_temperature_from_log != enabled or bool(
            self._temperature_from_log_overrides
        )
        self._use_temperature_from_log = enabled
        self._temperature_from_log_overrides.clear()
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def use_temperature_from_log(self) -> bool:
        """Return the global temperature-from-log display option."""
        return bool(self._use_temperature_from_log)

    def set_dataset_temperature_from_log(self, run_number: int, enabled: bool) -> None:
        """Override temperature-from-log display for a single dataset."""
        rn = int(run_number)
        enabled = bool(enabled)
        if enabled == self._use_temperature_from_log:
            changed = rn in self._temperature_from_log_overrides
            self._temperature_from_log_overrides.pop(rn, None)
        else:
            changed = self._temperature_from_log_overrides.get(rn) != enabled
            self._temperature_from_log_overrides[rn] = enabled
        if changed:
            self._rebuild_table()
            self._resize_columns_to_content()

    def dataset_uses_temperature_from_log(self, run_number: int) -> bool:
        """Return whether one dataset is configured to show log temperature."""
        rn = int(run_number)
        return bool(self._temperature_from_log_overrides.get(rn, self._use_temperature_from_log))

    def _resolve_metadata_path(self, dataset: MuonDataset, field_key: str):
        """Resolve a metadata/synthetic key to a value for dynamic columns."""
        if field_key.startswith("run_info."):
            return self._resolve_run_info_value(dataset, field_key)

        if field_key == "temperature":
            return self._temperature_for_display(dataset)

        metadata = dataset.metadata
        current = metadata
        for part in field_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _temperature_for_display(self, dataset: MuonDataset) -> float:
        """Return the temperature shown in the fixed browser temperature column."""
        if self.dataset_uses_temperature_from_log(int(dataset.run_number)):
            series_mean = self._series_mean_for_field(dataset, "temperature")
            if series_mean is not None:
                return float(series_mean)
        try:
            return float(dataset.metadata.get("temperature", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _temperature_uses_log_for_display(self, dataset: MuonDataset) -> bool:
        """Return whether the displayed temperature value came from a log."""
        return (
            self.dataset_uses_temperature_from_log(int(dataset.run_number))
            and self._series_mean_for_field(dataset, "temperature") is not None
        )

    def _series_mean_for_field(self, dataset: MuonDataset, field_key: str) -> float | None:
        """Return the mean from the time-series log associated with a summary field."""
        series = dataset.metadata.get("nexus_time_series", {})
        if not isinstance(series, dict):
            return None
        scored = [
            (score, series_path)
            for series_path in series
            if (score := self._series_path_score(field_key, series_path, series.get(series_path)))
            > 0
        ]
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        for _, series_path in scored:
            info = series.get(series_path, {})
            if not isinstance(info, dict) or "mean" not in info:
                continue
            try:
                return float(info.get("mean"))
            except (TypeError, ValueError):
                continue
        return None

    def _series_path_score(self, field_key: str, series_path: str, info) -> int:
        """Score how well a log series matches a browser summary field."""
        if not isinstance(info, dict):
            info = {}
        role = str(info.get("role", "")).strip().lower()
        if field_key == "temperature" and role == "sample_temperature":
            return 100 if bool(info.get("primary", False)) else 70

        normalized = " ".join(str(series_path).replace("_", " ").replace("/", " ").lower().split())
        compact = normalized.replace(" ", "")
        if field_key == "temperature":
            if not (
                "temp" in compact
                or "sampletemp" in compact
                or "samtsvalue" in compact
                or "dilt" in compact
                or "variox" in compact
                or "(k)" in str(series_path).lower()
            ):
                return 0
            score = 10
            if "sample" in normalized:
                score += 20
            if "sam ts value" in normalized:
                score += 30
            return score
        return 0

    def _resolve_run_info_value(self, dataset: MuonDataset, field_key: str):
        """Resolve synthetic ``run_info.*`` keys used by Run Info summary rows."""
        key = field_key[len("run_info.") :]
        if key == "points":
            return dataset.n_points

        run = dataset.run
        if run is None or not run.histograms:
            return None

        if key == "histograms":
            return len(run.histograms)

        h0 = run.histograms[0]
        if key == "bins":
            return h0.n_bins
        if key == "bin_width_us":
            return h0.bin_width

        total_counts = float(np.sum([np.sum(h.counts) for h in run.histograms]))
        if key == "counts_mev":
            return total_counts / 1.0e6
        if key == "counts_per_detector":
            return total_counts / max(len(run.histograms), 1)
        return None

    def _format_extra_value(self, value) -> str:
        """Format dynamic-column values into compact table text."""
        if value is None:
            return "—"

        if isinstance(value, dict):
            if "mean" in value:
                try:
                    return f"{float(value['mean']):.6g}"
                except (TypeError, ValueError):
                    return str(value.get("mean", "—"))
            text = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
            return text if len(text) <= 48 else f"{text[:45]}..."

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                return "—"
            if np.issubdtype(arr.dtype, np.number):
                return f"{float(np.nanmean(arr.astype(np.float64))):.6g}"
            text = str(list(value))
            return text if len(text) <= 48 else f"{text[:45]}..."

        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"

        text = str(value)
        return text if text else "—"

    def _value_for_extra_column(self, dataset: MuonDataset, field_key: str) -> str:
        """Return rendered text for a metadata-backed extra column cell."""
        value = self._resolve_metadata_path(dataset, field_key)
        return self._format_extra_value(value)

    # ------------------------------------------------------------------
    # Row and selection helpers
    # ------------------------------------------------------------------

    def _is_group_key(self, key: object) -> bool:
        return isinstance(key, str) and key.startswith(self._GROUP_SENTINEL_PREFIX)

    def _group_id_from_key(self, key: object) -> str | None:
        if not self._is_group_key(key):
            return None
        return str(key)[len(self._GROUP_SENTINEL_PREFIX) :]

    def _dataset_run_numbers_from_keys(self, keys: list[int | str]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for key in keys:
            if isinstance(key, int) and key in self._datasets and key not in seen:
                out.append(key)
                seen.add(key)
                continue
            gid = self._group_id_from_key(key)
            if gid is None:
                continue
            group = self._groups.get(gid)
            if group is None:
                continue
            for rn in group.member_run_numbers:
                if rn in self._datasets and rn not in seen:
                    out.append(rn)
                    seen.add(rn)
        return out

    def _get_selected_run_numbers(self) -> list[int]:
        return self._dataset_run_numbers_from_keys(self._selected_keys())

    def _get_selected_group_ids(self) -> list[str]:
        ids: list[str] = []
        for key in self._selected_keys():
            gid = self._group_id_from_key(key)
            if gid is not None:
                ids.append(gid)
        return ids

    def get_selected_group_ids(self) -> list[str]:
        return self._get_selected_group_ids()

    def get_current_selection_key(self) -> int | str | None:
        """Return the key for the current row, if any."""
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, 0)
        if item is None:
            return None
        key = item.data(self._GROUP_ROLE)
        return key if isinstance(key, (int, str)) else None

    def get_current_dataset(self) -> MuonDataset | None:
        """Return dataset on the current table row when that row is a run."""
        key = self.get_current_selection_key()
        if not isinstance(key, int):
            return None
        return self._datasets.get(key)

    def is_single_group_selected(self) -> bool:
        """Return True when the selection contains exactly one group header row."""
        return len(self._selected_keys()) == 1 and len(self._get_selected_group_ids()) == 1

    def get_group_name(self, group_id: str) -> str | None:
        group = self._groups.get(group_id)
        return None if group is None else group.name

    def get_group_id_for_run(self, run_number: int) -> str | None:
        """Return data-group id containing *run_number*, if any."""
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return None
        return self._run_to_group.get(run_key)

    def get_group_member_run_numbers(self, group_id: str) -> list[int]:
        """Return run numbers currently belonging to *group_id*."""
        group = self._groups.get(group_id)
        if group is None:
            return []
        return [int(rn) for rn in group.member_run_numbers]

    def get_dataset(self, run_number: int) -> MuonDataset | None:
        return self._datasets.get(run_number)

    def get_selected_datasets(self) -> list[MuonDataset]:
        selected: list[MuonDataset] = []
        for run_number in self._get_selected_run_numbers():
            dataset = self._datasets.get(run_number)
            if dataset is not None:
                selected.append(dataset)
        return selected

    def get_all_datasets(self) -> list[MuonDataset]:
        """Return all datasets currently present in the browser."""
        return list(self._datasets.values())

    def is_combined_dataset(self, run_number: int) -> bool:
        """Return ``True`` when *run_number* refers to a combined row."""
        try:
            return int(run_number) in self._combined_datasets
        except (TypeError, ValueError):
            return False

    def get_combined_source_datasets(self, run_number: int) -> list[MuonDataset]:
        """Return hidden source datasets for a combined row."""
        try:
            combined_rn = int(run_number)
        except (TypeError, ValueError):
            return []
        return list(self._combined_source_datasets.get(combined_rn, []))

    def rebuild_combined_dataset(self, run_number: int) -> MuonDataset | None:
        """Recompute one combined dataset from its hidden source datasets."""
        try:
            combined_rn = int(run_number)
        except (TypeError, ValueError):
            return None

        source_datasets = self._combined_source_datasets.get(combined_rn, [])
        if len(source_datasets) < 2:
            return None

        source_run_numbers = self._combined_datasets.get(
            combined_rn,
            [int(ds.run_number) for ds in source_datasets],
        )
        rebuilt = self._coadd_datasets(
            source_datasets,
            source_run_numbers,
            combined_run_number=combined_rn,
            existing_dataset=self._datasets.get(combined_rn),
        )
        self._datasets[combined_rn] = rebuilt
        return rebuilt

    def _normalize_grouping_value(self, value):
        """Return a deterministic representation for grouping comparisons."""
        if isinstance(value, dict):
            normalized: dict[str, object] = {}
            for key in sorted(value, key=lambda item: str(item)):
                try:
                    norm_key = str(int(key))
                except (TypeError, ValueError):
                    norm_key = str(key)
                normalized[norm_key] = self._normalize_grouping_value(value[key])
            return normalized
        if isinstance(value, (list, tuple)):
            return [self._normalize_grouping_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return [self._normalize_grouping_value(v) for v in value.tolist()]
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (np.floating, float)):
            val = float(value)
            if not np.isfinite(val):
                return str(val)
            return round(val, 12)
        if isinstance(value, str):
            return value.strip()
        return value

    def _grouping_signature(self, dataset: MuonDataset):
        """Return normalized grouping payload for co-add compatibility checks."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if run is None or not isinstance(grouping, dict):
            return None

        groups = grouping.get("groups")
        if not isinstance(groups, dict) or not groups:
            return None

        histograms = getattr(run, "histograms", None) or []
        t0_default = 0
        last_good_default = max(0, dataset.n_points - 1)
        if histograms:
            try:
                t0_default = int(histograms[0].t0_bin)
            except (TypeError, ValueError, IndexError):
                t0_default = 0
            try:
                last_good_default = max(0, len(histograms[0].counts) - 1)
            except (TypeError, ValueError, IndexError):
                last_good_default = max(0, dataset.n_points - 1)

        try:
            t0_bin = int(grouping.get("t0_bin", t0_default))
        except (TypeError, ValueError):
            t0_bin = t0_default

        raw_t_good = grouping.get("t_good_offset")
        if raw_t_good is None:
            try:
                raw_t_good = int(grouping.get("first_good_bin", t0_bin)) - t0_bin
            except (TypeError, ValueError):
                raw_t_good = 0
        try:
            t_good_offset = max(0, int(raw_t_good))
        except (TypeError, ValueError):
            t_good_offset = 0

        first_good_bin = max(0, t0_bin + t_good_offset)
        try:
            last_good_bin = int(grouping.get("last_good_bin", last_good_default))
        except (TypeError, ValueError):
            last_good_bin = last_good_default

        try:
            bin_index_base = 1 if int(grouping.get("bin_index_base", 0)) == 1 else 0
        except (TypeError, ValueError):
            bin_index_base = 0

        try:
            bunching_factor = int(grouping.get("bunching_factor", 1))
        except (TypeError, ValueError):
            bunching_factor = 1
        signature = {
            "groups": groups,
            "forward_group": int(grouping.get("forward_group", 1)),
            "backward_group": int(grouping.get("backward_group", 2)),
            "alpha": float(grouping.get("alpha", 1.0)),
            "alpha_x": grouping.get("alpha_x"),
            "alpha_y": grouping.get("alpha_y"),
            "alpha_z": grouping.get("alpha_z"),
            "vector_axis": grouping.get("vector_axis"),
            "group_names": grouping.get("group_names", {}),
            "t0_bin": t0_bin,
            "t_good_offset": t_good_offset,
            "first_good_bin": first_good_bin,
            "last_good_bin": last_good_bin,
            "bin_index_base": bin_index_base,
            "bunching_factor": bunching_factor,
            "deadtime_correction": bool(grouping.get("deadtime_correction", False)),
            "dead_time_us": grouping.get("dead_time_us"),
            "good_frames": grouping.get("good_frames"),
            "period_mode": grouping.get("period_mode"),
            "period_dead_time_us": grouping.get("period_dead_time_us"),
            "period_good_frames": grouping.get("period_good_frames"),
        }
        return self._normalize_grouping_value(signature)

    def _coadd_compatibility_error(self, datasets: list[MuonDataset]) -> str | None:
        """Return a user-facing error when selected datasets cannot be co-added."""
        if len(datasets) < 2:
            return "Select at least two grouped datasets to co-add."

        signatures = [self._grouping_signature(ds) for ds in datasets]
        if any(signature is None for signature in signatures):
            return (
                "Co-add requires identical grouping on every selected dataset. "
                "Apply grouping to each source run before combining them."
            )

        first_signature = signatures[0]
        if any(signature != first_signature for signature in signatures[1:]):
            return (
                "Co-add requires identical grouping on every selected dataset. "
                "Align groups, alpha, good-bin limits, bunching, and deadtime settings first."
            )
        return None

    def _mirrored_grouping_for_combined_dataset(self, dataset: MuonDataset) -> dict:
        """Return grouping metadata mirrored onto a combined dataset."""
        run = getattr(dataset, "run", None)
        grouping = getattr(run, "grouping", None)
        if not isinstance(grouping, dict):
            return {}
        return copy.deepcopy(grouping)

    def export_logbook_tsv(self, path: str) -> int:
        """Export all runs to tab-separated text using current columns/grouping.

        This export includes rows hidden by filters or collapsed groups.
        """
        headers = self._active_column_headers()
        sections = self._export_sections()
        exported_rows = 0

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            for section_index, (section_name, run_numbers) in enumerate(sections):
                writer.writerow(self._group_header_values(len(headers), section_name))
                writer.writerow(headers)

                for run_number in run_numbers:
                    dataset = self._datasets.get(run_number)
                    if dataset is None:
                        continue
                    writer.writerow(self._export_row_values(run_number, dataset))
                    exported_rows += 1

                if section_index < len(sections) - 1:
                    writer.writerow([])

        return exported_rows

    def export_logbook_rtf(self, path: str) -> int:
        """Export all runs to tab-separated RTF using current columns/grouping.

        The export includes rows hidden in the table by filters or collapsed
        groups so the file always reflects the full data browser contents.
        """
        headers = self._active_column_headers()
        sections = self._export_sections()
        exported_rows = 0
        header_cells = [self._rtf_header_cell(header) for header in headers]
        header_line = self._rtf_tabbed_line(header_cells, preescaped=True)

        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write(r"{\rtf1\ansi\deff0\n")

            for section_index, (section_name, run_numbers) in enumerate(sections):
                group_header = self._group_header_values(len(headers), section_name)
                group_cells = [self._rtf_escape(value) for value in group_header]
                handle.write(self._rtf_tabbed_line(group_cells, preescaped=True))
                handle.write("\n")
                handle.write(header_line)
                handle.write("\n")

                for run_number in run_numbers:
                    dataset = self._datasets.get(run_number)
                    if dataset is None:
                        continue
                    handle.write(
                        self._rtf_tabbed_line(self._export_row_values(run_number, dataset))
                    )
                    handle.write("\n")
                    exported_rows += 1

                if section_index < len(sections) - 1:
                    handle.write(r"\par\n")

            handle.write("}")

        return exported_rows

    def _rtf_tabbed_line(self, values: list[str], *, preescaped: bool = False) -> str:
        escaped = values if preescaped else [self._rtf_escape(str(value)) for value in values]
        return r"\tab ".join(escaped) + r"\par"

    def _group_header_values(self, column_count: int, section_name: str) -> list[str]:
        """Return a section-header row with the same width as the table."""
        if column_count <= 0:
            return [f"Data Group: {section_name}"]

        row = [""] * column_count
        row[0] = "Data Group"
        if column_count >= 2:
            row[1] = section_name
        else:
            row[0] = f"Data Group: {section_name}"
        return row

    def _rtf_header_cell(self, header: str) -> str:
        """Return RTF-formatted header text for export table cells."""
        if header == "𝑇 (K)":
            return r"\i T\i0 (K)"
        if header == "𝐵 (G)":
            return r"\i B\i0 (G)"
        return self._rtf_escape(header)

    def _rtf_signed16(self, value: int) -> int:
        return value if value < 0x8000 else value - 0x10000

    def _rtf_escape(self, text: str) -> str:
        sanitized = text.replace("\r", " ").replace("\n", " ")
        if sanitized.isascii():
            return sanitized.translate(
                {
                    ord("\\"): r"\\",
                    ord("{"): r"\{",
                    ord("}"): r"\}",
                    ord("\t"): r"\tab ",
                }
            )

        escaped: list[str] = []
        for ch in sanitized:
            if ch == "\\":
                escaped.append(r"\\")
                continue
            if ch == "{":
                escaped.append(r"\{")
                continue
            if ch == "}":
                escaped.append(r"\}")
                continue
            if ch == "\t":
                escaped.append(r"\tab ")
                continue
            if ch in ("\r", "\n"):
                escaped.append(" ")
                continue

            codepoint = ord(ch)
            if codepoint <= 0x7F:
                escaped.append(ch)
                continue

            if codepoint <= 0xFFFF:
                escaped.append(f"\\u{self._rtf_signed16(codepoint)}?")
                continue

            encoded = codepoint - 0x10000
            high_surrogate = 0xD800 + (encoded >> 10)
            low_surrogate = 0xDC00 + (encoded & 0x3FF)
            escaped.append(f"\\u{self._rtf_signed16(high_surrogate)}?")
            escaped.append(f"\\u{self._rtf_signed16(low_surrogate)}?")

        return "".join(escaped)

    def _active_column_headers(self) -> list[str]:
        """Return visible/active data-browser column headers."""
        headers: list[str] = []
        for col in range(self._table.columnCount()):
            header_item = self._table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item is not None else "")
        return headers

    def _export_sections(self) -> list[tuple[str, list[int]]]:
        """Build export sections in display order with group headers."""
        sections: list[tuple[str, list[int]]] = []
        ungrouped_runs: list[int] = []

        for entry in self._display_order:
            if isinstance(entry, str):
                group = self._groups.get(entry)
                if group is None:
                    continue
                members = [int(rn) for rn in group.member_run_numbers if int(rn) in self._datasets]
                sections.append((group.name, members))
                continue

            if entry in self._datasets:
                ungrouped_runs.append(int(entry))

        if ungrouped_runs or not sections:
            sections.append(("Ungrouped", ungrouped_runs))

        return sections

    def _export_row_values(self, run_number: int, dataset: MuonDataset) -> list[str]:
        """Return exported row values for one dataset in active-column order."""
        meta = dataset.metadata
        run_display = str(dataset.run_label)
        if run_number in self._combined_datasets:
            run_display = " + ".join(map(str, self._combined_datasets[run_number]))

        row = [
            run_display,
            str(meta.get("title", "")),
            f"{self._temperature_for_display(dataset):.2f}",
            f"{float(meta.get('field', 0.0)):.1f}",
            str(meta.get("comment", "")),
        ]
        for field_key in self._visible_extra_columns():
            row.append(self._value_for_extra_column(dataset, field_key))
        return row

    # ------------------------------------------------------------------
    # Editing and removal
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table:
            return
        if item.column() != 3:
            return

        row = item.row()
        run_item = self._table.item(row, 0)
        if run_item is None:
            return

        run_number = run_item.data(self._GROUP_ROLE)
        if not isinstance(run_number, int):
            return

        dataset = self._datasets.get(run_number)
        if dataset is None:
            return

        text = item.text().strip()
        try:
            field_value = float(text.split()[0]) if text else 0.0
        except ValueError:
            self._updating_table = True
            item.setText(f"{float(dataset.metadata.get('field', 0.0)):.1f}")
            self._updating_table = False
            return

        dataset.metadata["field"] = field_value
        if dataset.run is not None:
            dataset.run.metadata["field"] = field_value

        self._updating_table = True
        item.setText(f"{field_value:.1f}")
        self._updating_table = False

    def _remove_run_number(self, run_number: int) -> None:
        self._datasets.pop(run_number, None)
        self._combined_datasets.pop(run_number, None)
        self._combined_source_datasets.pop(run_number, None)
        self._temperature_from_log_overrides.pop(int(run_number), None)

        gid = self._run_to_group.get(run_number)
        if gid is not None:
            self._remove_run_from_group(run_number, gid)
        if run_number in self._display_order:
            self._display_order.remove(run_number)

    def _remove_selected_entries(self) -> None:
        keys = self._selected_keys()
        if not keys:
            return

        selected_group_ids = [
            gid for gid in (self._group_id_from_key(k) for k in keys) if gid is not None
        ]
        for gid in selected_group_ids:
            self.ungroup(gid)

        for run_number in self._dataset_run_numbers_from_keys(keys):
            self._remove_run_number(run_number)

        self._rebuild_table()
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _create_table_context_menu(self) -> QMenu | None:
        keys = self._selected_keys()
        if not keys:
            return None

        menu = QMenu(self)
        selected_runs = [k for k in keys if isinstance(k, int)]
        selected_group_ids = [
            gid for gid in (self._group_id_from_key(k) for k in keys) if gid is not None
        ]
        expanded_selected_runs = self._dataset_run_numbers_from_keys(keys)
        grouped_selected_runs = [
            rn for rn in selected_runs if self._run_to_group.get(rn) is not None
        ]

        regular_runs = [rn for rn in expanded_selected_runs if rn not in self._combined_datasets]
        combined_runs = [rn for rn in expanded_selected_runs if rn in self._combined_datasets]

        if len(regular_runs) >= 2 and not combined_runs:
            menu.addAction("Co-add Selected", self._coadd_selected)
        if len(expanded_selected_runs) >= 2 and not selected_group_ids:
            menu.addAction("Form Data Group", self._form_data_group)
        if len(selected_runs) == 1 and not selected_group_ids:
            selected_run = selected_runs[0]
            menu.addAction("Get Info", lambda rn=selected_run: self.get_info_requested.emit(rn))

        if combined_runs:
            menu.addAction("Separate Combined", self._separate_combined)

        if selected_runs and self._groups:
            send_menu = menu.addMenu("Send to Group")
            self._populate_send_to_group_menu(send_menu, selected_runs)

        if grouped_selected_runs:
            label = "Remove from Group" if len(grouped_selected_runs) == 1 else "Remove from Groups"
            menu.addAction(label, self._remove_selected_from_group)

        if len(selected_group_ids) == 1 and len(keys) == 1:
            gid = selected_group_ids[0]
            group = self._groups.get(gid)
            if group is not None:
                collapse_text = "Expand Group" if group.collapsed else "Collapse Group"
                menu.addAction(collapse_text, lambda gid=gid: self._toggle_group_collapsed(gid))
                menu.addAction("Rename Group", lambda gid=gid: self._rename_group(gid))
                menu.addAction("Ungroup", lambda gid=gid: self.ungroup(gid))
                menu.addSeparator()

        label = "Remove Entry" if len(keys) == 1 else "Remove Selected Entries"
        menu.addAction(label, self._remove_selected_entries)
        return menu

    def _populate_send_to_group_menu(self, send_menu: QMenu, selected_runs: list[int]) -> None:
        """Populate Send-to-Group submenu with current groups."""
        groups = sorted(self._groups.values(), key=lambda g: g.name.lower())
        if not groups:
            action = send_menu.addAction("(No groups)")
            action.setEnabled(False)
            return

        for group in groups:
            action = send_menu.addAction(group.name)
            action.triggered.connect(
                lambda _checked=False, gid=group.group_id, runs=list(selected_runs): (
                    self.add_runs_to_group(runs, gid)
                )
            )

    def _remove_selected_from_group(self) -> None:
        run_numbers = [
            rn for rn in self._get_selected_run_numbers() if self._run_to_group.get(rn) is not None
        ]
        if not run_numbers:
            return
        self.remove_runs_from_group(run_numbers)

    def _show_table_context_menu(self, position: QPoint) -> None:
        viewport_pos = position
        item = self._table.itemAt(viewport_pos)
        if item is None:
            return

        row = item.row()
        selected_rows = {idx.row() for idx in self._table.selectedIndexes()}
        if row not in selected_rows:
            self._table.selectRow(row)

        menu = self._create_table_context_menu()
        if menu is None:
            return

        global_pos = self._table.viewport().mapToGlobal(viewport_pos)
        menu.popup(global_pos)

    def _form_data_group(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        if len(run_numbers) < 2:
            return

        default_name = self._default_group_name(run_numbers)
        name, ok = QInputDialog.getText(self, "Form Data Group", "Group name:", text=default_name)
        if not ok:
            return
        group_name = name.strip() or default_name
        self.create_data_group(run_numbers, name=group_name)

    def _toggle_group_collapsed(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        group.collapsed = not group.collapsed
        self._rebuild_table()

    def _rename_group(self, group_id: str) -> None:
        group = self._groups.get(group_id)
        if group is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Data Group", "Group name:", text=group.name)
        if not ok:
            return
        new_name = name.strip()
        if not new_name:
            return
        group.name = new_name
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Event filter, selection, sorting, filtering
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):  # noqa: N802
        header = self._table.horizontalHeader()

        if watched is self._table and event.type() == QEvent.Type.KeyPress:
            if (
                event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and self._table.state() != QAbstractItemView.State.EditingState
            ):
                self._remove_selected_entries()
                return True

            if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and bool(
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            ):
                current_row = self._table.currentRow()
                if current_row < 0:
                    return False

                direction = -1 if event.key() == Qt.Key.Key_Up else 1
                target_row = self._next_visible_row(current_row, direction)
                if target_row is None:
                    return True

                add_to_selection = bool(
                    event.modifiers()
                    & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
                )
                anchor_row = self._selection_anchor_for_row(current_row)
                if self._select_visible_row_range(
                    anchor_row,
                    target_row,
                    add_to_selection=add_to_selection,
                ):
                    self._selection_anchor_row = anchor_row
                    return True

        if watched is self._table.viewport():
            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                item = self._table.itemAt(event.position().toPoint())
                if item is not None:
                    gid = self._group_id_from_key(item.data(self._GROUP_ROLE))
                    if gid is not None:
                        self._toggle_group_collapsed(gid)
                        return True

            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                row = self._table.rowAt(event.position().toPoint().y())
                if row >= 0:
                    modifiers = event.modifiers()
                    index = self._table.model().index(row, 0)
                    selection_model = self._table.selectionModel()

                    if bool(modifiers & Qt.KeyboardModifier.ShiftModifier):
                        anchor_row = self._selection_anchor_for_row(row)

                        if selection_model is not None:
                            add_to_selection = bool(
                                modifiers
                                & (
                                    Qt.KeyboardModifier.ControlModifier
                                    | Qt.KeyboardModifier.MetaModifier
                                )
                            )
                            if self._select_visible_row_range(
                                anchor_row,
                                row,
                                add_to_selection=add_to_selection,
                            ):
                                self._selection_anchor_row = anchor_row
                                return True
                    else:
                        self._selection_anchor_row = row
                        if selection_model is not None:
                            selection_model.setCurrentIndex(
                                index, QItemSelectionModel.SelectionFlag.NoUpdate
                            )

        if watched is header.viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                pos = event.position().toPoint()
                logical_index = header.logicalIndexAt(pos)
                if logical_index < 0:
                    return False
                if event.button() == Qt.MouseButton.LeftButton:
                    self._on_header_clicked(logical_index)
                    return True
                if event.button() == Qt.MouseButton.RightButton:
                    QTimer.singleShot(
                        0, lambda ci=logical_index: self._open_header_context_menu(ci)
                    )
                    return True
        return super().eventFilter(watched, event)

    def _open_header_context_menu(self, col_idx: int) -> None:
        """Open right-click header menu for filtering or dynamic-column removal."""
        if col_idx < 0:
            return

        if col_idx < len(self._COLUMNS):
            self._open_filter_dialog(col_idx)
            return

        extra_index = col_idx - len(self._COLUMNS)
        visible_extra_columns = self._visible_extra_columns()
        if extra_index < 0 or extra_index >= len(visible_extra_columns):
            return

        field_key = visible_extra_columns[extra_index]
        menu = QMenu(self)
        menu.addAction(
            "Remove from Data Browser",
            lambda fk=field_key: self.remove_extra_column(fk),
        )
        menu.exec(self.cursor().pos())

    def _on_header_clicked(self, logical_index: int) -> None:
        if logical_index == self._current_sort_column:
            self._current_sort_order = (
                Qt.SortOrder.DescendingOrder
                if self._current_sort_order == Qt.SortOrder.AscendingOrder
                else Qt.SortOrder.AscendingOrder
            )
        else:
            self._current_sort_column = logical_index
            self._current_sort_order = Qt.SortOrder.AscendingOrder
        self._sort_table()

    def _sort_table(self, *, rebuild: bool = True) -> None:
        if self._current_sort_column < 0:
            return

        reverse = self._current_sort_order == Qt.SortOrder.DescendingOrder

        def _sort_key(run_number: int):
            dataset = self._datasets.get(run_number)
            if dataset is None:
                return ""
            meta = dataset.metadata
            if self._current_sort_column == 0:
                return run_number
            if self._current_sort_column == 1:
                return str(meta.get("title", ""))
            if self._current_sort_column == 2:
                return self._temperature_for_display(dataset)
            if self._current_sort_column == 3:
                return float(meta.get("field", 0.0))
            if self._current_sort_column >= len(self._COLUMNS):
                idx = self._current_sort_column - len(self._COLUMNS)
                visible_extra_columns = self._visible_extra_columns()
                if idx < 0 or idx >= len(visible_extra_columns):
                    return ""
                value = self._resolve_metadata_path(dataset, visible_extra_columns[idx])
                if isinstance(value, (int, float, np.integer, np.floating)):
                    return float(value)
                if isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value)
                    if arr.size and np.issubdtype(arr.dtype, np.number):
                        return float(np.nanmean(arr.astype(np.float64)))
                return "" if value is None else str(value)
            return str(meta.get("comment", ""))

        runs = [entry for entry in self._display_order if isinstance(entry, int)]
        sorted_runs = sorted(runs, key=_sort_key, reverse=reverse)

        if self._groups:
            groups = [
                entry
                for entry in self._display_order
                if isinstance(entry, str) and entry in self._groups
            ]
            self._display_order = groups + sorted_runs
        else:
            self._display_order = sorted_runs

        self._table.horizontalHeader().setSortIndicator(
            self._current_sort_column, self._current_sort_order
        )
        if rebuild:
            self._rebuild_table()

    def _open_filter_dialog(self, col_idx: int) -> None:
        unique_values = set()
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            if self._is_group_key(run_item.data(self._GROUP_ROLE)):
                continue
            item = self._table.item(row, col_idx)
            if item:
                unique_values.add(item.text().strip())

        header_item = self._table.horizontalHeaderItem(col_idx)
        column_name = header_item.text() if header_item is not None else str(col_idx)

        dialog = FilterDialog(
            column_name,
            sorted(unique_values),
            self._column_filters.get(col_idx),
            self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_values = dialog.get_selected_values()
            if selected_values is None:
                self._column_filters.pop(col_idx, None)
            else:
                self._column_filters[col_idx] = selected_values
            self._apply_row_visibility()

    def _row_visible_by_filters(self, row: int) -> bool:
        if not self._column_filters:
            return True
        for col_idx, allowed in self._column_filters.items():
            item = self._table.item(row, col_idx)
            if item and item.text().strip() not in allowed:
                return False
        return True

    def _apply_row_visibility(self) -> None:
        for row in range(self._table.rowCount()):
            self._table.setRowHidden(row, False)

        if not self._column_filters:
            # still need to apply collapsed state
            for row in range(self._table.rowCount()):
                run_item = self._table.item(row, 0)
                if run_item is None:
                    continue
                key = run_item.data(self._GROUP_ROLE)
                if isinstance(key, int):
                    gid = self._run_to_group.get(key)
                    if (
                        gid is not None
                        and self._groups.get(gid) is not None
                        and self._groups[gid].collapsed
                    ):
                        self._table.setRowHidden(row, True)
            self.selection_changed.emit()
            return

        # First pass: hide dataset rows not matching filter or collapsed by group.
        group_has_visible: dict[str, bool] = {gid: False for gid in self._groups}
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            key = run_item.data(self._GROUP_ROLE)
            if self._is_group_key(key):
                continue

            visible = self._row_visible_by_filters(row)
            if isinstance(key, int):
                gid = self._run_to_group.get(key)
                if (
                    gid is not None
                    and self._groups.get(gid) is not None
                    and self._groups[gid].collapsed
                ):
                    visible = False
                if gid is not None and visible:
                    group_has_visible[gid] = True
            self._table.setRowHidden(row, not visible)

        # Second pass: hide group rows when all children filtered out.
        for row in range(self._table.rowCount()):
            run_item = self._table.item(row, 0)
            if run_item is None:
                continue
            gid = self._group_id_from_key(run_item.data(self._GROUP_ROLE))
            if gid is None:
                continue
            self._table.setRowHidden(row, not group_has_visible.get(gid, False))

        if not self._is_row_visible_for_selection(self._selection_anchor_row or -1):
            self._selection_anchor_row = None

        self.selection_changed.emit()

    def _on_selection_changed(self) -> None:
        selected_datasets = self.get_selected_datasets()
        selected_group_ids = self._get_selected_group_ids()

        self.selection_changed.emit()

        if len(selected_group_ids) == 1 and len(self._selected_keys()) == 1:
            self.group_selected.emit(selected_group_ids[0])
            return

        if len(selected_datasets) == 1:
            self.dataset_selected.emit(selected_datasets[0].run_number)

    # ------------------------------------------------------------------
    # Co-add and separate
    # ------------------------------------------------------------------

    def _coadd_selected(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        if len(run_numbers) < 2:
            return

        datasets_to_combine: list[MuonDataset] = []
        for rn in run_numbers:
            if rn in self._combined_datasets:
                return
            dataset = self._datasets.get(rn)
            if dataset:
                datasets_to_combine.append(dataset)

        if len(datasets_to_combine) < 2:
            return

        incompatibility = self._coadd_compatibility_error(datasets_to_combine)
        if incompatibility is not None:
            QMessageBox.warning(self, "Cannot Co-add Selected Datasets", incompatibility)
            return

        insert_index = min(self._display_index_for_run(rn) for rn in run_numbers)
        combined_rn = self._next_combined_id
        self._next_combined_id -= 1
        source_datasets = [self._datasets[rn] for rn in run_numbers if rn in self._datasets]
        combined_dataset = self._coadd_datasets(
            source_datasets,
            run_numbers,
            combined_run_number=combined_rn,
        )

        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = list(run_numbers)
        self._combined_source_datasets[combined_rn] = source_datasets

        for rn in run_numbers:
            self._remove_run_number(rn)

        self._display_order.insert(insert_index, combined_rn)
        self._rebuild_table()

    def _coadd_datasets(
        self,
        datasets: list[MuonDataset],
        run_numbers: list[int],
        *,
        combined_run_number: int,
        existing_dataset: MuonDataset | None = None,
    ) -> MuonDataset:
        time_grid = datasets[0].time

        asymmetries = []
        errors_squared = []

        for dataset in datasets:
            if np.array_equal(dataset.time, time_grid):
                asymmetries.append(dataset.asymmetry)
                errors_squared.append(dataset.error**2)
            else:
                interp_asymmetry = np.interp(time_grid, dataset.time, dataset.asymmetry)
                interp_error = np.interp(time_grid, dataset.time, dataset.error)
                asymmetries.append(interp_asymmetry)
                errors_squared.append(interp_error**2)

        combined_asymmetry = np.mean(asymmetries, axis=0)
        combined_error = np.sqrt(np.sum(errors_squared, axis=0)) / len(datasets)

        titles = [str(d.metadata.get("title", "")).strip() for d in datasets]
        non_empty_titles = [t for t in titles if t]
        if non_empty_titles and all(t == non_empty_titles[0] for t in non_empty_titles):
            combined_title = non_empty_titles[0]
        else:
            combined_title = f"Combined {len(datasets)} runs"

        metadata = {
            "title": combined_title,
            "temperature": np.mean([d.metadata.get("temperature", 0) for d in datasets]),
            "field": np.mean([d.metadata.get("field", 0) for d in datasets]),
            "run_number": combined_run_number,
            "run_label": " + ".join(map(str, run_numbers)),
            "combined_from": list(run_numbers),
        }
        mirrored_grouping = self._mirrored_grouping_for_combined_dataset(datasets[0])
        run = Run(
            run_number=combined_run_number,
            histograms=[],
            metadata=dict(metadata),
            grouping=mirrored_grouping,
            source_file="",
        )

        if existing_dataset is not None:
            existing_dataset.time = time_grid.copy()
            existing_dataset.asymmetry = combined_asymmetry
            existing_dataset.error = combined_error
            existing_dataset.metadata = metadata
            existing_dataset.run = run
            if hasattr(existing_dataset, "_grouping_source_arrays_cache"):
                delattr(existing_dataset, "_grouping_source_arrays_cache")
            return existing_dataset

        return MuonDataset(
            time=time_grid.copy(),
            asymmetry=combined_asymmetry,
            error=combined_error,
            metadata=metadata,
            run=run,
        )

    def _separate_combined(self) -> None:
        run_numbers = self._get_selected_run_numbers()
        combined_items = [rn for rn in run_numbers if rn in self._combined_datasets]
        if not combined_items:
            return

        for rn in combined_items:
            insert_index = self._display_index_for_run(rn)
            source_datasets = self._combined_source_datasets.get(rn, [])
            group_id = self._run_to_group.get(rn)
            group = self._groups.get(group_id) if group_id is not None else None

            self._datasets.pop(rn, None)
            self._combined_datasets.pop(rn, None)
            self._combined_source_datasets.pop(rn, None)
            if group is not None:
                try:
                    member_index = group.member_run_numbers.index(rn)
                except ValueError:
                    member_index = len(group.member_run_numbers)
                group.member_run_numbers = [
                    member for member in group.member_run_numbers if member != rn
                ]
                self._run_to_group.pop(rn, None)

                for offset, dataset in enumerate(source_datasets):
                    source_rn = int(dataset.run_number)
                    self._datasets[source_rn] = dataset
                    group.member_run_numbers.insert(member_index + offset, source_rn)
                    self._run_to_group[source_rn] = group.group_id
            else:
                self._run_to_group.pop(rn, None)
            if rn in self._display_order:
                self._display_order.remove(rn)

            if group is None:
                for offset, dataset in enumerate(source_datasets):
                    source_rn = int(dataset.run_number)
                    self._datasets[source_rn] = dataset
                    if source_rn not in self._display_order:
                        self._display_order.insert(insert_index + offset, source_rn)

        self._rebuild_table()

    # ------------------------------------------------------------------
    # Project state
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._datasets.clear()
        self._combined_datasets.clear()
        self._combined_source_datasets.clear()
        self._next_combined_id = -1
        self._groups.clear()
        self._run_to_group.clear()
        self._display_order.clear()
        self._column_filters.clear()
        self._extra_columns.clear()
        self._use_temperature_from_log = False
        self._temperature_from_log_overrides.clear()
        self._current_sort_column = -1
        self._current_sort_order = Qt.SortOrder.AscendingOrder
        self._refresh_column_headers()
        self._table.setRowCount(0)

    def add_combined_dataset(self, source_run_numbers: list[int]) -> int | None:
        datasets_to_combine = []
        for rn in source_run_numbers:
            ds = self._datasets.get(rn)
            if ds is None:
                return None
            datasets_to_combine.append(ds)

        if len(datasets_to_combine) < 2:
            return None

        incompatibility = self._coadd_compatibility_error(datasets_to_combine)
        if incompatibility is not None:
            return None

        combined_rn = self._next_combined_id
        self._next_combined_id -= 1
        source_datasets = [self._datasets[rn] for rn in source_run_numbers if rn in self._datasets]
        combined_dataset = self._coadd_datasets(
            source_datasets,
            source_run_numbers,
            combined_run_number=combined_rn,
        )

        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = source_run_numbers
        self._combined_source_datasets[combined_rn] = source_datasets

        insert_index = min(self._display_index_for_run(rn) for rn in source_run_numbers)
        for rn in source_run_numbers:
            self._remove_run_number(rn)
        self._display_order.insert(insert_index, combined_rn)

        self._rebuild_table()
        return combined_rn

    def get_state(self) -> dict:
        filters = {str(col): sorted(values) for col, values in self._column_filters.items()}
        data_groups = [
            {
                "group_id": group.group_id,
                "name": group.name,
                "member_run_numbers": [int(rn) for rn in group.member_run_numbers],
                "collapsed": bool(group.collapsed),
            }
            for group in self._groups.values()
        ]
        selected_group_ids = self._get_selected_group_ids()
        return {
            "sort_column": self._current_sort_column,
            "sort_order": "ascending"
            if self._current_sort_order == Qt.SortOrder.AscendingOrder
            else "descending",
            "filters": filters,
            "selected_run_numbers": self._get_selected_run_numbers(),
            "selected_group_ids": selected_group_ids,
            "data_groups": data_groups,
            "extra_columns": list(self._extra_columns),
            "use_temperature_from_log": bool(self._use_temperature_from_log),
            "temperature_from_log_overrides": {
                str(rn): bool(enabled)
                for rn, enabled in sorted(self._temperature_from_log_overrides.items())
            },
        }

    def restore_state(self, state: dict) -> None:
        self._column_filters = {}
        for col_str, values in state.get("filters", {}).items():
            self._column_filters[int(col_str)] = set(values)

        self._current_sort_column = int(state.get("sort_column", -1))
        sort_order_str = state.get("sort_order", "ascending")
        self._current_sort_order = (
            Qt.SortOrder.AscendingOrder
            if sort_order_str == "ascending"
            else Qt.SortOrder.DescendingOrder
        )
        saved_extra_columns = [str(v) for v in state.get("extra_columns", []) if str(v).strip()]
        self._use_temperature_from_log = bool(
            state.get("use_temperature_from_log", "temperature" in saved_extra_columns)
        )
        self._extra_columns = [key for key in saved_extra_columns if key != "temperature"]
        self._temperature_from_log_overrides = {}
        for run_number, enabled in state.get("temperature_from_log_overrides", {}).items():
            try:
                rn = int(run_number)
            except (TypeError, ValueError):
                continue
            if rn in self._datasets:
                self._temperature_from_log_overrides[rn] = bool(enabled)
        self._refresh_column_headers()

        for group_entry in state.get("data_groups", []):
            if not isinstance(group_entry, dict):
                continue
            group_id = str(group_entry.get("group_id") or "")
            if not group_id:
                continue
            run_numbers = [
                int(v)
                for v in group_entry.get("member_run_numbers", [])
                if int(v) in self._datasets
            ]
            if len(run_numbers) < 2:
                continue
            self.create_data_group(
                run_numbers,
                name=str(group_entry.get("name") or "").strip() or None,
                group_id=group_id,
                collapsed=bool(group_entry.get("collapsed", False)),
            )

        self._sort_table(rebuild=False)
        self._move_groups_to_top()
        self._rebuild_table()

        selected_runs = set(state.get("selected_run_numbers", []))
        selected_group_ids = {str(v) for v in state.get("selected_group_ids", [])}

        keys: list[int | str] = list(selected_runs)
        keys.extend(f"{self._GROUP_SENTINEL_PREFIX}{gid}" for gid in selected_group_ids)
        self._restore_selection_by_keys(keys)
