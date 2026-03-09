"""Data browser / logbook panel.

Displays a table of loaded runs (like WiMDA's run browser) with columns for
run number, title, temperature, field, etc.  Clicking a row selects it for
plotting and analysis.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset


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

        # Mixed item types can happen in the Run column when combined rows are
        # displayed as labels like "3031 + 3034". Avoid calling super().__lt__
        # here because Qt may bounce back into Python overrides recursively.
        other_text = other.text() if isinstance(other, QTableWidgetItem) else str(other)
        try:
            other_numeric = float(other_text)
            return self._numeric_value < other_numeric
        except (ValueError, TypeError):
            return self.text() < other_text


class FilterDialog(QDialog):
    """Excel-style filter dialog with checkboxes for unique values."""

    def __init__(
        self,
        column_name: str,
        unique_values: list[str],
        current_selection: set[str] | None,
        parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Filter - {column_name}")
        self.setMinimumWidth(300)
        self.setMinimumHeight(400)

        self._checkboxes: list[QCheckBox] = []
        self._unique_values = unique_values

        layout = QVBoxLayout(self)

        # "All" checkbox
        self._all_checkbox = QCheckBox("(Select All)")
        self._all_checkbox.setChecked(current_selection is None)
        self._all_checkbox.stateChanged.connect(self._on_all_changed)
        layout.addWidget(self._all_checkbox)

        # Scrollable area for value checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        # Create checkbox for each unique value
        for value in unique_values:
            checkbox = QCheckBox(value)
            if current_selection is None:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(value in current_selection)
            checkbox.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons
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
        """Handle 'Select All' checkbox change."""
        checked = state == Qt.CheckState.Checked.value
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)

    def _on_checkbox_changed(self) -> None:
        """Handle individual checkbox change - update 'All' checkbox state."""
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
        """Clear the filter for this column and close."""
        self._all_checkbox.setChecked(True)
        self.done(QDialog.DialogCode.Accepted)

    def get_selected_values(self) -> set[str] | None:
        """Return set of selected values, or None if all are selected (no filter)."""
        # Check if all checkboxes are checked (not just the Select All checkbox)
        all_checked = all(cb.isChecked() for cb in self._checkboxes)

        if all_checked:
            # All selected = no filter
            return None

        selected = {
            checkbox.text() for checkbox in self._checkboxes
            if checkbox.isChecked()
        }

        # Return the set of selected values (could be empty)
        return selected


class DataBrowserPanel(QWidget):
    """Logbook-style run table with sorting, filtering, and co-add capabilities.

    Features
    --------
    * **Sorting**: Left-click column headers to sort (toggle asc/desc)
    * **Excel-style filtering**: Right-click column headers to open filter dialog
      with checkboxes for all unique values in that column
    * **Multi-selection**: Ctrl+Click and Shift+Click for range selection
    * **Co-adding**: Average multiple selected datasets with proper error propagation

    Signals
    -------
    dataset_selected : Signal(int)
        Emitted with run_number when a dataset is selected for viewing/analysis
    selection_changed : Signal()
        Emitted when the selection of datasets changes in the table

    Implementation Notes
    --------------------
    Uses manual sorting via sortItems() and custom event filter on header viewport
    to avoid Qt's broken header state machine when modal dialogs are opened.
    """

    dataset_selected = Signal(int)  # emits run_number
    selection_changed = Signal()    # emits when table selection changes

    _COLUMNS = ["Run", "Title", "𝑇 (K)", "𝐵 (G)", "Comment"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: dict[int, MuonDataset] = {}
        self._combined_datasets: dict[int, list[int]] = {}  # combined_run -> [original_runs]
        self._combined_source_datasets: dict[int, list[MuonDataset]] = {}
        self._next_combined_id = -1  # Use negative IDs for combined datasets
        self._column_filters: dict[int, set[str]] = {}  # column_index -> set of selected values
        self._current_sort_column: int = -1
        self._current_sort_order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
        self._updating_table = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Table – sorting and filtering are handled entirely through an event
        # filter on the header viewport.  We never use setSortingEnabled or
        # sectionClicked/customContextMenuRequested because Qt's internal
        # header state machine gets corrupted by the modal filter dialog,
        # causing subsequent left-clicks to hang.
        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.SelectedClicked
        )
        # Start with widths that fill the default browser width (~480px).
        # After data is loaded, columns are resized to fit content.
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 70)   # Run
        header.resizeSection(1, 145)  # Title
        header.resizeSection(2, 60)   # 𝑇 (K)
        header.resizeSection(3, 60)   # 𝐵 (G)
        header.resizeSection(4, 155)  # Comment
        self._table.setSortingEnabled(False)
        self._table.horizontalHeader().setSortIndicatorShown(True)
        self._table.horizontalHeader().setSectionsClickable(False)
        self._table.horizontalHeader().viewport().installEventFilter(self)
        self._table.viewport().installEventFilter(self)
        # Removed: self._table.cellClicked.connect(self._on_cell_clicked)
        # Using itemSelectionChanged instead to avoid interfering with multi-selection
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self._table)

        # Co-add controls
        self._create_coadd_controls()
        layout.addLayout(self._coadd_layout)

        # Set minimum width very small to allow shrinking, but default is user-resizable
        self.setMinimumWidth(250)

    def _create_coadd_controls(self) -> None:
        """Create co-add controls."""
        self._coadd_layout = QHBoxLayout()

        coadd_btn = QPushButton("Co-add Selected")
        coadd_btn.clicked.connect(self._coadd_selected)
        self._coadd_layout.addWidget(coadd_btn)

        separate_btn = QPushButton("Separate Combined")
        separate_btn.clicked.connect(self._separate_combined)
        self._coadd_layout.addWidget(separate_btn)

        self._coadd_layout.addStretch()

    def add_dataset(self, dataset: MuonDataset) -> None:
        rn = dataset.run_number
        self._datasets[rn] = dataset
        self._add_table_row(dataset)
        self._resize_columns_to_content()

    def _resize_columns_to_content(self) -> None:
        """Expand columns to fit loaded content while keeping sensible minima."""
        self._table.resizeColumnsToContents()
        header = self._table.horizontalHeader()
        minimums = {
            0: 70,   # Run
            1: 145,  # Title
            2: 60,   # 𝑇 (K)
            3: 60,   # 𝐵 (G)
            4: 155,  # Comment
        }
        maximums = {
            0: 110,  # Run
            1: 260,  # Title
            2: 90,   # 𝑇 (K)
            3: 90,   # 𝐵 (G)
            4: 320,  # Comment
        }
        for col, min_width in minimums.items():
            size = header.sectionSize(col)
            if size < min_width:
                header.resizeSection(col, min_width)
            elif size > maximums[col]:
                header.resizeSection(col, maximums[col])

    def _add_table_row(self, dataset: MuonDataset) -> None:
        """Add a row to the table for the given dataset."""
        rn = dataset.run_number
        meta = dataset.metadata

        # Get title from metadata (for combined datasets, this already contains
        # the shared title if applicable, or a generic "Combined N runs" label)
        title = meta.get("title", "")

        # Display run numbers as "3077 + 3076" for combined datasets
        run_display = str(rn)
        if rn in self._combined_datasets:
            run_display = " + ".join(map(str, self._combined_datasets[rn]))

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._updating_table = True

        # Column 0: Run number (display composite label for combined entries)
        if rn in self._combined_datasets:
            item = QTableWidgetItem(run_display)
        else:
            item = NumericTableWidgetItem(rn)
        item.setData(Qt.ItemDataRole.UserRole, rn)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 0, item)

        # Column 1: Title (text)
        title_item = QTableWidgetItem(title)
        title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 1, title_item)

        # Column 2: Temperature (numeric)
        temp = float(meta.get('temperature', 0))
        temp_item = NumericTableWidgetItem(f"{temp:.2f}")
        temp_item.setFlags(temp_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 2, temp_item)

        # Column 3: Field (numeric)
        field = float(meta.get('field', 0))
        field_item = NumericTableWidgetItem(f"{field:.1f}")
        # Field is editable to support manual correction.
        field_item.setFlags(field_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 3, field_item)

        # Column 4: Comment (read-only)
        comment = str(meta.get("comment", ""))
        comment_item = QTableWidgetItem(comment)
        comment_item.setFlags(comment_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, 4, comment_item)
        self._updating_table = False

        # Maintain current sort order after inserting
        self._sort_table()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Persist manual edits from table back to dataset metadata."""
        if self._updating_table:
            return

        # Only field column is editable/persisted.
        if item.column() != 3:
            return

        row = item.row()
        run_item = self._table.item(row, 0)
        if run_item is None:
            return
        run_number = run_item.data(Qt.ItemDataRole.UserRole)
        dataset = self._datasets.get(run_number)
        if dataset is None:
            return

        text = item.text().strip()
        try:
            field_value = float(text.split()[0]) if text else 0.0
        except ValueError:
            # Revert invalid edit.
            self._updating_table = True
            item.setText(f"{float(dataset.metadata.get('field', 0.0)):.1f}")
            self._updating_table = False
            return

        dataset.metadata["field"] = field_value
        if dataset.run is not None:
            dataset.run.metadata["field"] = field_value

        # Normalize display text for consistent sorting/filtering.
        self._updating_table = True
        item.setText(f"{field_value:.1f}")
        self._updating_table = False

    def _remove_row_for_run_number(self, run_number: int) -> None:
        """Remove a table row by its internal run number ID."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == run_number:
                self._table.removeRow(row)
                return

    # ------------------------------------------------------------------
    # Event filter – handles header clicks for sorting & filtering
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):  # noqa: N802
        """Intercept mouse clicks on the header viewport.

        Left-click  → sort by clicked column.
        Right-click → open filter dialog for clicked column.

        By handling clicks at this level we bypass QHeaderView's internal
        mouse-tracking state machine, which was getting corrupted by the
        modal filter dialog and causing subsequent left-clicks to hang.
        """
        header = self._table.horizontalHeader()

        # Custom selection behavior for table rows:
        # Shift+Click adds just the clicked row to selection instead of a range.
        if watched is self._table.viewport():
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                row = self._table.rowAt(event.position().toPoint().y())
                if row >= 0:
                    index = self._table.model().index(row, 0)
                    selection_model = self._table.selectionModel()
                    if selection_model is not None:
                        selection_model.select(
                            index,
                            QItemSelectionModel.SelectionFlag.Select
                            | QItemSelectionModel.SelectionFlag.Rows,
                        )
                        selection_model.setCurrentIndex(
                            index,
                            QItemSelectionModel.SelectionFlag.NoUpdate,
                        )
                        return True

        if watched is header.viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                pos = event.position().toPoint()
                logical_index = header.logicalIndexAt(pos)
                if logical_index < 0:
                    return False
                if event.button() == Qt.MouseButton.LeftButton:
                    self._on_header_clicked(logical_index)
                    return True  # consumed – don't let header process it
                if event.button() == Qt.MouseButton.RightButton:
                    QTimer.singleShot(
                        0, lambda ci=logical_index: self._open_filter_dialog(ci)
                    )
                    return True
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------
    # Sorting – fully manual, never uses setSortingEnabled
    # ------------------------------------------------------------------

    def _on_header_clicked(self, logical_index: int) -> None:
        """Sort by the given column, toggling order on repeated clicks."""
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

    def _sort_table(self) -> None:
        """Perform a one-shot sort and re-apply row visibility."""
        if self._current_sort_column < 0:
            return
        self._table.sortItems(self._current_sort_column, self._current_sort_order)
        self._table.horizontalHeader().setSortIndicator(
            self._current_sort_column, self._current_sort_order
        )
        # Row visibility flags don't move with items, so re-apply after sort.
        self._apply_row_visibility()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _open_filter_dialog(self, col_idx: int) -> None:
        """Open the filter dialog for *col_idx*."""
        unique_values = set()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, col_idx)
            if item:
                unique_values.add(item.text())

        dialog = FilterDialog(
            self._COLUMNS[col_idx],
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

    def _apply_row_visibility(self) -> None:
        """Show or hide rows according to the active column filters."""
        if not self._column_filters:
            for row in range(self._table.rowCount()):
                self._table.setRowHidden(row, False)
            # Filtering state changed; refresh any dependent selection displays.
            self.selection_changed.emit()
            return

        for row in range(self._table.rowCount()):
            visible = True
            for col_idx, allowed in self._column_filters.items():
                item = self._table.item(row, col_idx)
                if item and item.text() not in allowed:
                    visible = False
                    break
            self._table.setRowHidden(row, not visible)

        # Filtering state changed; refresh any dependent selection displays.
        self.selection_changed.emit()

    def _coadd_selected(self) -> None:
        """Co-add (average) selected datasets."""
        selected_rows = set(item.row() for item in self._table.selectedItems())
        if len(selected_rows) < 2:
            return  # Need at least 2 datasets to co-add

        # Get run numbers of selected datasets
        run_numbers = []
        for row in sorted(selected_rows):
            item = self._table.item(row, 0)
            if item:
                rn = item.data(Qt.ItemDataRole.UserRole)
                run_numbers.append(rn)

        # Do not allow nested combined datasets.
        datasets_to_combine = []
        for rn in run_numbers:
            if rn in self._combined_datasets:
                return
            dataset = self._datasets.get(rn)
            if dataset:
                datasets_to_combine.append(dataset)

        if len(datasets_to_combine) < 2:
            return

        # Perform co-add
        combined_dataset = self._coadd_datasets(datasets_to_combine, run_numbers)

        # Store the combined dataset
        combined_rn = self._next_combined_id
        self._next_combined_id -= 1

        self._datasets[combined_rn] = combined_dataset
        self._combined_datasets[combined_rn] = run_numbers
        self._combined_source_datasets[combined_rn] = [
            self._datasets[rn] for rn in run_numbers if rn in self._datasets
        ]

        # Remove original runs from workspace/table.
        for rn in run_numbers:
            self._datasets.pop(rn, None)
            self._remove_row_for_run_number(rn)

        # Add to table
        self._add_table_row(combined_dataset)

    def _coadd_datasets(
        self,
        datasets: list[MuonDataset],
        run_numbers: list[int],
    ) -> MuonDataset:
        """Average multiple datasets with proper error propagation.

        Interpolates all datasets to a common time grid (the first dataset's grid).
        """
        # Use the first dataset's time grid
        time_grid = datasets[0].time

        asymmetries = []
        errors_squared = []

        for dataset in datasets:
            if np.array_equal(dataset.time, time_grid):
                asymmetries.append(dataset.asymmetry)
                errors_squared.append(dataset.error ** 2)
            else:
                # Interpolate to common grid using numpy
                interp_asymmetry = np.interp(time_grid, dataset.time, dataset.asymmetry)
                interp_error = np.interp(time_grid, dataset.time, dataset.error)
                asymmetries.append(interp_asymmetry)
                errors_squared.append(interp_error ** 2)

        # Average asymmetries
        combined_asymmetry = np.mean(asymmetries, axis=0)

        # Propagate errors: σ_avg = sqrt(sum(σ²)) / N
        combined_error = np.sqrt(np.sum(errors_squared, axis=0)) / len(datasets)

        # Determine combined title: use shared title if all non-empty titles match
        titles = [str(d.metadata.get("title", "")).strip() for d in datasets]
        non_empty_titles = [t for t in titles if t]
        if non_empty_titles and all(t == non_empty_titles[0] for t in non_empty_titles):
            # All non-empty titles match, use it
            combined_title = non_empty_titles[0]
        else:
            # Mixed or all-empty titles, use generic
            combined_title = f"Combined {len(datasets)} runs"

        # Create combined metadata
        metadata = {
            "title": combined_title,
            "temperature": np.mean([d.metadata.get("temperature", 0) for d in datasets]),
            "field": np.mean([d.metadata.get("field", 0) for d in datasets]),
            "run_number": self._next_combined_id,
            "run_label": " + ".join(map(str, run_numbers)),
            "combined_from": run_numbers,
        }

        return MuonDataset(
            time=time_grid.copy(),
            asymmetry=combined_asymmetry,
            error=combined_error,
            metadata=metadata,
        )

    def _separate_combined(self) -> None:
        """Separate selected combined datasets back to originals."""
        selected_rows = set(item.row() for item in self._table.selectedItems())

        selected_combined_runs: list[int] = []
        for row in sorted(selected_rows, reverse=True):
            item = self._table.item(row, 0)
            if not item:
                continue
            rn = item.data(Qt.ItemDataRole.UserRole)
            if rn in self._combined_datasets:
                selected_combined_runs.append(rn)

        for rn in selected_combined_runs:
            # Restore source datasets first.
            for dataset in self._combined_source_datasets.get(rn, []):
                source_rn = dataset.run_number
                self._datasets[source_rn] = dataset
                self._add_table_row(dataset)

            # Remove the combined dataset
            self._datasets.pop(rn, None)
            self._combined_datasets.pop(rn, None)
            self._combined_source_datasets.pop(rn, None)
            self._remove_row_for_run_number(rn)

        self._sort_table()

    def get_dataset(self, run_number: int) -> MuonDataset | None:
        """Retrieve a dataset by run number."""
        return self._datasets.get(run_number)

    def get_selected_datasets(self) -> list[MuonDataset]:
        """Get all currently selected datasets.

        Returns
        -------
        list[MuonDataset]
            List of datasets that are currently selected (highlighted) in the table,
            in table order.
        """
        selected = []
        # Use selection model to get all selected rows (more reliable than selectedItems)
        selection_model = self._table.selectionModel()

        if selection_model:
            selected_indices = selection_model.selectedRows()
            for idx in selected_indices:
                row = idx.row()
                # Ignore filtered-out rows that may remain selected internally.
                if self._table.isRowHidden(row):
                    continue
                item = self._table.item(row, 0)
                if item:
                    run_number = item.data(Qt.ItemDataRole.UserRole)
                    dataset = self._datasets.get(run_number)
                    if dataset:
                        selected.append(dataset)

        return selected

    def _on_selection_changed(self) -> None:
        """Handle selection changes - emit both single and multi-selection signals."""
        # Get all selected datasets
        selected = self.get_selected_datasets()

        # Emit signal for multi-selection (global fitting)
        self.selection_changed.emit()

        # Only emit dataset_selected if exactly ONE dataset is selected
        # (for plotting that dataset). If multiple, user is doing global fit.
        if len(selected) == 1:
            self.dataset_selected.emit(selected[0].run_number)
