"""Run-information dialog for loaded datasets.

The dialog has two layers:
* A prominent summary section with key run information.
* An advanced table exposing all NeXus-derived fields and log summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.gui.windows.log_plot_dialog import LogPlotDialog


class RunInfoDialog(QDialog):
    """Display key and advanced metadata for a selected run."""

    add_to_browser_requested = Signal(str)
    set_browser_field_inclusion_requested = Signal(str, bool)

    _TABLE_HEADERS = ["Include in Data Browser", "Field", "Value", "Log Plot"]

    def __init__(self, dataset: MuonDataset, parent=None, included_fields: set[str] | None = None) -> None:
        """Create a run-information dialog for ``dataset``.

        Parameters
        ----------
        dataset
            Dataset whose metadata should be displayed.
        parent
            Parent Qt widget.
        """
        super().__init__(parent)
        self._dataset = dataset
        self._series_cache: dict[str, dict[str, Any]] = dataset.metadata.get("nexus_time_series", {})
        self._included_fields: set[str] = {str(v) for v in (included_fields or set()) if str(v).strip()}
        self._advanced_dialog: AdvancedRunInfoDialog | None = None

        self.setWindowTitle(f"Run Info - {dataset.run_label}")
        self.resize(760, 460)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Run Parameters"))
        self._summary_table = QTableWidget(0, len(self._TABLE_HEADERS))
        self._summary_table.setHorizontalHeaderLabels(self._TABLE_HEADERS)
        self._summary_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._summary_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        root.addWidget(self._summary_table)

        controls = QHBoxLayout()
        controls.addStretch()

        self._advanced_button = QPushButton("Advanced")
        self._advanced_button.clicked.connect(self._open_advanced_dialog)
        controls.addWidget(self._advanced_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        controls.addWidget(close_button)

        root.addLayout(controls)

        self._populate_summary_table()

    def _populate_summary_table(self) -> None:
        """Populate the primary run-info table with include/log controls."""
        meta = self._dataset.metadata
        run = self._dataset.run
        orientation = self._stringify(
            ((meta.get("nexus_fields") or {}).get("sample") or {}).get("shape", "")
        )

        rows: list[tuple[str, str, str | None, str | None]] = [
            ("Instrument", str(meta.get("instrument", "")), "instrument", None),
            ("Run", self._dataset.run_label, "run_label", None),
            ("Title", str(meta.get("title", "")), "title", None),
            ("Comment", str(meta.get("comment", "")), "comment", None),
            ("Start", str(meta.get("started", "")), "started", None),
            ("End", str(meta.get("stopped", "")), "stopped", None),
            (
                "Temperature (K)",
                self._fmt_float(meta.get("temperature")),
                "temperature",
                self._series_path_for_field("temperature"),
            ),
            (
                "Magnetic Field (G)",
                self._fmt_float(meta.get("field")),
                "field",
                self._series_path_for_field("field"),
            ),
            (
                "Field Direction",
                str(meta.get("field_direction", "")),
                "field_direction",
                self._series_path_for_field("field_direction"),
            ),
            ("Orientation", orientation, "nexus_fields.sample.shape", None),
            ("Periods", str(meta.get("period_count", 1)), "period_count", None),
            ("Points", str(self._dataset.n_points), "run_info.points", None),
        ]

        if run is not None and run.histograms:
            h0 = run.histograms[0]
            n_hist = len(run.histograms)
            rows.extend(
                [
                    ("Histograms", str(n_hist), "run_info.histograms", None),
                    ("Bins", str(h0.n_bins), "run_info.bins", None),
                    ("Bin Width (us)", self._fmt_float(h0.bin_width), "run_info.bin_width_us", None),
                ]
            )

            total_counts = float(np.sum([np.sum(h.counts) for h in run.histograms]))
            rows.append(("Counts (MEv)", self._fmt_float(total_counts / 1.0e6), "run_info.counts_mev", None))
            rows.append(
                (
                    "Counts per Detector",
                    self._fmt_float(total_counts / max(n_hist, 1)),
                    "run_info.counts_per_detector",
                    None,
                )
            )

        self._fill_table(self._summary_table, rows)

    def _advanced_rows(self) -> list[tuple[str, str, str | None, str | None]]:
        """Build rows for the advanced metadata subwindow."""
        nexus_fields = self._dataset.metadata.get("nexus_fields", {})
        flat = self._flatten_fields("nexus_fields", nexus_fields)
        rows: list[tuple[str, str, str | None, str | None]] = [
            (key, value, key, series_path) for key, value, series_path in flat
        ]

        for series_path, info in sorted(self._series_cache.items()):
            summary_key = f"nexus_time_series.{series_path}.mean"
            units = str(info.get("units", "")).strip()
            units_suffix = f" {units}" if units else ""
            summary_val = (
                f"mean={self._fmt_float(info.get('mean'))}{units_suffix}, "
                f"min={self._fmt_float(info.get('min'))}{units_suffix}, "
                f"max={self._fmt_float(info.get('max'))}{units_suffix}"
            )
            rows.append((summary_key, summary_val, summary_key, series_path))
        return rows

    def _fill_table(self, table: QTableWidget, rows: list[tuple[str, str, str | None, str | None]]) -> None:
        """Populate a run-info table with include checkboxes and log-plot actions."""
        table.setRowCount(len(rows))
        for row, (label, value, field_key, series_path) in enumerate(rows):
            include_box = QCheckBox()
            include_box.setEnabled(bool(field_key))
            if field_key:
                include_box.setChecked(field_key in self._included_fields)
                include_box.toggled.connect(
                    lambda checked, fk=field_key: self._on_field_inclusion_toggled(fk, checked)
                )
            table.setCellWidget(row, 0, include_box)

            table.setItem(row, 1, QTableWidgetItem(label))
            table.setItem(row, 2, QTableWidgetItem(value))

            if series_path:
                plot_button = QPushButton("Plot")
                plot_button.clicked.connect(lambda _=False, sp=series_path: self._show_series_plot(sp))
                table.setCellWidget(row, 3, plot_button)
            else:
                table.setItem(row, 3, QTableWidgetItem(""))

        table.resizeColumnsToContents()

    def _on_field_inclusion_toggled(self, field_key: str, checked: bool) -> None:
        """Update local include state and emit browser inclusion requests."""
        if checked:
            self._included_fields.add(field_key)
            self.add_to_browser_requested.emit(field_key)
        else:
            self._included_fields.discard(field_key)
        self.set_browser_field_inclusion_requested.emit(field_key, checked)

    def _open_advanced_dialog(self) -> None:
        """Open the advanced run-info subwindow."""
        self._advanced_dialog = AdvancedRunInfoDialog(
            rows=self._advanced_rows(),
            included_fields=self._included_fields,
            parent=self,
        )
        self._advanced_dialog.set_browser_field_inclusion_requested.connect(
            self._on_field_inclusion_toggled
        )
        self._advanced_dialog.show_log_plot_requested.connect(self._show_series_plot)
        self._advanced_dialog.show()

    def _series_path_for_field(self, field_key: str) -> str | None:
        """Return a best-effort time-series path for a summary field."""
        token_map = {
            "temperature": ["temp", "sampletemp"],
            "field": ["field", "magnet"],
            "field_direction": ["field", "direction"],
        }
        tokens = token_map.get(field_key, [])
        if not tokens:
            return None
        for series_path in sorted(self._series_cache.keys()):
            lowered = series_path.lower()
            if any(token in lowered for token in tokens):
                return series_path
        return None

    def _show_series_plot(self, series_path: str) -> None:
        """Open a dedicated plot dialog for a selected time-series log."""
        info = self._series_cache.get(series_path)
        if not info:
            return
        dlg = LogPlotDialog(
            title=series_path,
            time_values=list(info.get("time", [])),
            data_values=list(info.get("values", [])),
            units=str(info.get("units", "")),
            parent=self,
        )
        dlg.exec()

    def _flatten_fields(self, prefix: str, value: Any) -> list[tuple[str, str, str | None]]:
        """Flatten nested metadata into ``(key, value, series_path)`` table rows."""
        rows: list[tuple[str, str, str | None]] = []
        if isinstance(value, Mapping):
            for key in sorted(value.keys(), key=str):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                rows.extend(self._flatten_fields(child_prefix, value[key]))
            return rows

        rows.append((prefix, self._stringify(value), None))
        return rows

    def _stringify(self, value: Any) -> str:
        """Format arbitrary metadata values for compact table display."""
        if isinstance(value, float):
            return self._fmt_float(value)
        if isinstance(value, list):
            if len(value) > 8:
                return f"[{', '.join(map(str, value[:5]))}, ...] ({len(value)} items)"
            return str(value)
        return str(value)

    def _fmt_float(self, value: Any) -> str:
        """Format numeric values consistently for read-only text display."""
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)


class AdvancedRunInfoDialog(QDialog):
    """Advanced metadata table with include/log actions."""

    set_browser_field_inclusion_requested = Signal(str, bool)
    show_log_plot_requested = Signal(str)

    def __init__(
        self,
        rows: list[tuple[str, str, str | None, str | None]],
        included_fields: set[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Run Info")
        self.resize(980, 560)

        self._all_rows = rows

        root = QVBoxLayout(self)

        self._search_bar = QLineEdit()
        self._search_bar.setPlaceholderText("Search fields...")
        self._search_bar.setClearButtonEnabled(True)
        self._search_bar.setEnabled(True)
        self._search_bar.setReadOnly(False)
        self._search_bar.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._search_bar.textChanged.connect(self._filter_rows)
        root.addWidget(self._search_bar)

        self._table = QTableWidget(0, len(RunInfoDialog._TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(RunInfoDialog._TABLE_HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        root.addWidget(self._table)

        self._table.setRowCount(len(rows))
        self._row_search_tokens: list[str] = []
        for row, (label, value, field_key, series_path) in enumerate(rows):
            include_box = QCheckBox()
            include_box.setEnabled(bool(field_key))
            if field_key:
                include_box.setChecked(field_key in included_fields)
                include_box.toggled.connect(
                    lambda checked, fk=field_key: self.set_browser_field_inclusion_requested.emit(fk, checked)
                )
            self._table.setCellWidget(row, 0, include_box)

            self._table.setItem(row, 1, QTableWidgetItem(label))
            self._table.setItem(row, 2, QTableWidgetItem(value))

            if series_path:
                plot_button = QPushButton("Plot")
                plot_button.clicked.connect(
                    lambda _=False, sp=series_path: self.show_log_plot_requested.emit(sp)
                )
                self._table.setCellWidget(row, 3, plot_button)
            else:
                self._table.setItem(row, 3, QTableWidgetItem(""))

            search_blob = " ".join(
                token for token in (
                    str(label),
                    str(value),
                    str(field_key or ""),
                    str(series_path or ""),
                ) if token
            ).lower()
            self._row_search_tokens.append(search_blob)

        self._table.resizeColumnsToContents()
        self._search_bar.setFocus()

        controls = QHBoxLayout()
        controls.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        controls.addWidget(close_button)
        root.addLayout(controls)

    def _filter_rows(self, text: str) -> None:
        """Show only rows whose Field or Value contains the search text."""
        query = text.strip().lower()
        for row in range(self._table.rowCount()):
            haystack = self._row_search_tokens[row] if row < len(self._row_search_tokens) else ""
            self._table.setRowHidden(row, bool(query) and query not in haystack)
