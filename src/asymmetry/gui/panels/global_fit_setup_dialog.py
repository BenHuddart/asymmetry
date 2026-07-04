"""Explicit setup dialog for a cross-group global parameter fit (Phase 4).

Before Phase 4 the only way to reach the cross-group fit dialog was to
multi-select trend group buttons and click a per-parameter Model Fit button; the
group-variable coordinate (the orthogonal x each local parameter is plotted
against — temperature for a field fit, etc.) was inferred with a hardcoded
temperature↔field complement and could not be edited or set to a custom column.

:class:`GlobalFitSetupDialog` makes the setup explicit: which parameter, which
trend series, which x-axis, and — crucially — which *group variable* and its
per-group values (prefilled from the same inference, but editable and able to
draw from any data-browser custom column). Its :class:`GlobalFitSetupResult`
feeds the panel's assembly path (``group_variable_overrides``/``group_order``)
so the chosen values reach every :class:`ParameterGroupData` and, via the study,
the results window's local-parameter axis labels.

The dialog is deliberately constructed from **plain data**
(:class:`GlobalFitSetupData`) rather than a live panel, so it is unit-testable
without a fully wired :class:`FitParametersPanel`. The panel builds that plain
data (:meth:`FitParametersPanel.global_fit_setup_data`) and consumes the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

__all__ = [
    "GlobalFitSetupData",
    "GlobalFitSetupResult",
    "GlobalFitSetupSeries",
    "GlobalFitSetupDialog",
]


@dataclass(frozen=True)
class GlobalFitSetupSeries:
    """One trend series offered as a cross-group member.

    ``param_names`` is the set of fitted parameters this series carries; the
    dialog intersects the checked series' parameter sets to populate the
    parameter combo (only parameters present in *all* checked series are
    fittable jointly).
    """

    group_id: str
    group_name: str
    member_count: int
    param_names: tuple[str, ...]


@dataclass(frozen=True)
class GlobalFitSetupData:
    """Everything the setup dialog needs, extracted from the panel as plain data.

    ``x_key_options`` mirrors the trend panel's X-axis combo: ``(display_label,
    x_key)`` pairs (``field``/``temperature``/``run``/``angle``/``custom:<id>``/
    ``param:<name>``). ``group_variable_options`` is ``(display_label,
    gv_key)`` for the group-variable combo (temperature/field/run + every custom
    column). ``default_group_variable_key(x_key)`` returns the inference default
    for an x-key; ``group_variable_value(group_id, gv_key)`` returns the prefill
    per-group value; ``group_variable_label(gv_key)`` the axis-label prefill.
    """

    series: tuple[GlobalFitSetupSeries, ...]
    x_key_options: tuple[tuple[str, str], ...]
    group_variable_options: tuple[tuple[str, str], ...]
    default_group_variable_key: Callable[[str], str]
    group_variable_value: Callable[[str, str], float]
    group_variable_label: Callable[[str], str]


@dataclass
class GlobalFitSetupResult:
    """The user's explicit setup choices, consumed by the panel assembly path."""

    parameter_name: str
    x_key: str
    x_label: str
    group_ids: list[str]
    group_variable_key: str
    group_variable_label: str
    group_variable_values: dict[str, float] = field(default_factory=dict)


class GlobalFitSetupDialog(QDialog):
    """Choose parameter, series, x-axis and group variable for a global fit."""

    def __init__(
        self,
        data: GlobalFitSetupData,
        *,
        preselected_group_ids: list[str] | None = None,
        preselected_parameter: str | None = None,
        preselected_x_key: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New global parameter fit")
        self._data = data
        self._series_by_id = {s.group_id: s for s in data.series}
        # Suppress the per-group-variable-table rebuild while we are ourselves
        # repopulating it (setItem fires cellChanged).
        self._suppress_gv_table = False
        # The group-variable key the value table was last built for; a rebuild
        # under the *same* key preserves user edits, a rebuild under a *different*
        # key (the user switched group variable) discards them and re-prefills.
        self._gv_table_key: str | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Fit one model jointly across several trend series, sharing some "
            "parameters (Global) and letting others vary per series (Local)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        # Parameter combo — populated from the intersection of checked series.
        self._param_combo = QComboBox()
        form.addRow("Parameter:", self._param_combo)

        # X-axis combo.
        self._x_combo = QComboBox()
        for label, key in data.x_key_options:
            self._x_combo.addItem(label, userData=key)
        if preselected_x_key is not None:
            idx = self._x_combo.findData(preselected_x_key)
            if idx >= 0:
                self._x_combo.setCurrentIndex(idx)
        form.addRow("X axis:", self._x_combo)

        # Group-variable combo + axis label.
        self._gv_combo = QComboBox()
        for label, key in data.group_variable_options:
            self._gv_combo.addItem(label, userData=key)
        form.addRow("Group variable:", self._gv_combo)

        self._gv_label_edit = QLineEdit()
        form.addRow("Group-variable label:", self._gv_label_edit)

        layout.addLayout(form)

        # Series checklist.
        layout.addWidget(QLabel("Series (check at least two):"))
        self._series_list = QListWidget()
        preselected = set(preselected_group_ids or [])
        for series in data.series:
            item = QListWidgetItem(f"{series.group_name} ({series.member_count} members)")
            item.setData(Qt.ItemDataRole.UserRole, series.group_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = series.group_id in preselected if preselected else False
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self._series_list.addItem(item)
        # Nothing preselected (launched from the menu) → check all so a first-time
        # user has a valid ≥2 selection to start from.
        if not preselected:
            for row in range(self._series_list.count()):
                self._series_list.item(row).setCheckState(Qt.CheckState.Checked)
        layout.addWidget(self._series_list)

        # Per-group values table.
        layout.addWidget(QLabel("Group-variable value per series (editable):"))
        self._gv_table = QTableWidget(0, 2)
        self._gv_table.setHorizontalHeaderLabels(["Series", "Value"])
        self._gv_table.verticalHeader().setVisible(False)
        layout.addWidget(self._gv_table)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        button_row.addWidget(self._buttons)
        layout.addLayout(button_row)

        # Wiring — connect after the initial widgets exist.
        self._series_list.itemChanged.connect(self._on_series_checked_changed)
        self._x_combo.currentIndexChanged.connect(self._on_x_changed)
        self._gv_combo.currentIndexChanged.connect(self._on_gv_changed)
        self._gv_table.cellChanged.connect(self._on_gv_cell_changed)

        # Initial state: default the group variable from the chosen x-key, refresh
        # the parameter combo from the checked series, and build the value table.
        self._sync_group_variable_default_for_x()
        self._refresh_parameter_combo(preselect=preselected_parameter)
        self._rebuild_gv_table()
        self._refresh_ok_enabled()

    # ── selection helpers ────────────────────────────────────────────────────

    def _checked_group_ids(self) -> list[str]:
        ids: list[str] = []
        for row in range(self._series_list.count()):
            item = self._series_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    def _current_x_key(self) -> str:
        data = self._x_combo.currentData()
        return str(data) if data is not None else "run"

    def _current_gv_key(self) -> str:
        data = self._gv_combo.currentData()
        return str(data) if data is not None else "run"

    # ── event handlers ───────────────────────────────────────────────────────

    def _on_series_checked_changed(self, _item: QListWidgetItem) -> None:
        self._refresh_parameter_combo(preselect=self._param_combo.currentText() or None)
        self._rebuild_gv_table()
        self._refresh_ok_enabled()

    def _on_x_changed(self, _idx: int) -> None:
        # Changing the x-axis re-infers the default group variable (field→temp,
        # temp→field, …); this in turn rebuilds the value table + label.
        self._sync_group_variable_default_for_x()

    def _on_gv_changed(self, _idx: int) -> None:
        self._gv_label_edit.setText(self._data.group_variable_label(self._current_gv_key()))
        self._rebuild_gv_table()

    def _on_gv_cell_changed(self, _row: int, _col: int) -> None:
        # User edits land straight in the widget; nothing to recompute. The guard
        # only matters so a programmatic repopulate does not recurse.
        return

    # ── population ───────────────────────────────────────────────────────────

    def _sync_group_variable_default_for_x(self) -> None:
        """Point the group-variable combo at the inference default for the x-key."""
        default_key = self._data.default_group_variable_key(self._current_x_key())
        idx = self._gv_combo.findData(default_key)
        self._gv_combo.blockSignals(True)
        if idx >= 0:
            self._gv_combo.setCurrentIndex(idx)
        self._gv_combo.blockSignals(False)
        self._gv_label_edit.setText(self._data.group_variable_label(self._current_gv_key()))
        self._rebuild_gv_table()

    def _refresh_parameter_combo(self, *, preselect: str | None) -> None:
        """Populate the parameter combo with params common to all checked series."""
        checked = self._checked_group_ids()
        common: set[str] | None = None
        for gid in checked:
            series = self._series_by_id.get(gid)
            if series is None:
                continue
            names = set(series.param_names)
            common = names if common is None else (common & names)
        params = sorted(common) if common else []

        current = self._param_combo.currentText()
        self._param_combo.blockSignals(True)
        self._param_combo.clear()
        self._param_combo.addItems(params)
        target = preselect if preselect in params else (current if current in params else None)
        if target is not None:
            self._param_combo.setCurrentText(target)
        elif params:
            self._param_combo.setCurrentIndex(0)
        self._param_combo.blockSignals(False)
        self._refresh_ok_enabled()

    def _rebuild_gv_table(self) -> None:
        """Rebuild the per-series value table for the currently checked series.

        Each row's value is prefilled from the inference (median member value),
        but a value the user already edited for that series is preserved across
        rebuilds so an x-axis toggle does not wipe manual edits for series that
        stay checked under the same group variable.
        """
        checked = self._checked_group_ids()
        gv_key = self._current_gv_key()
        # Preserve any user-edited values keyed by group_id so a rebuild (e.g. a
        # series check/uncheck) keeps them — but only while the group variable is
        # unchanged. Switching the group variable re-prefills from the inference.
        if not self._suppress_gv_table and gv_key == self._gv_table_key:
            preserved = self._current_table_values()
        else:
            preserved = {}
        self._gv_table_key = gv_key

        self._suppress_gv_table = True
        try:
            self._gv_table.setRowCount(len(checked))
            for row, gid in enumerate(checked):
                series = self._series_by_id.get(gid)
                name = series.group_name if series is not None else gid
                name_item = QTableWidgetItem(name)
                name_item.setData(Qt.ItemDataRole.UserRole, gid)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._gv_table.setItem(row, 0, name_item)

                spin = QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                if gid in preserved:
                    spin.setValue(preserved[gid])
                else:
                    spin.setValue(float(self._data.group_variable_value(gid, gv_key)))
                self._gv_table.setCellWidget(row, 1, spin)
            self._gv_table.resizeColumnsToContents()
        finally:
            self._suppress_gv_table = False

    def _current_table_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for row in range(self._gv_table.rowCount()):
            name_item = self._gv_table.item(row, 0)
            spin = self._gv_table.cellWidget(row, 1)
            if name_item is None or spin is None:
                continue
            gid = str(name_item.data(Qt.ItemDataRole.UserRole))
            try:
                values[gid] = float(spin.value())
            except (TypeError, ValueError):
                continue
        return values

    def _refresh_ok_enabled(self) -> None:
        ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        enough_series = len(self._checked_group_ids()) >= 2
        has_param = self._param_combo.count() > 0 and bool(self._param_combo.currentText())
        ok.setEnabled(enough_series and has_param)

    # ── result ───────────────────────────────────────────────────────────────

    def result_data(self) -> GlobalFitSetupResult | None:
        """Return the user's setup choices, or ``None`` if not acceptable."""
        group_ids = self._checked_group_ids()
        if len(group_ids) < 2:
            return None
        parameter = self._param_combo.currentText()
        if not parameter:
            return None
        x_key = self._current_x_key()
        gv_key = self._current_gv_key()
        return GlobalFitSetupResult(
            parameter_name=parameter,
            x_key=x_key,
            x_label=self._x_combo.currentText(),
            group_ids=group_ids,
            group_variable_key=gv_key,
            group_variable_label=self._gv_label_edit.text().strip(),
            group_variable_values=self._current_table_values(),
        )
