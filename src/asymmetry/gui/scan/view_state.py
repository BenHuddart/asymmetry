"""Integral scan form state, separate from captured numerical history."""

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QTableWidgetItem


def analysis_state(self) -> dict:
    """Serialise the view options, baseline regions, and peaks to a dict."""
    # Persist only valid (normalised lo<hi) regions — the ones the fit uses —
    # so a reopened scan can't show a phantom region with no effect.
    regions = [[lo, hi] for lo, hi in self.baseline_regions()]
    peaks: list[list] = []
    for row in range(self._peaks_table.rowCount()):
        type_item = self._peaks_table.item(row, 0)
        vals = [self._cell_float(self._peaks_table, row, col) for col in (1, 2, 3)]
        if type_item is None or any(v is None for v in vals):
            continue
        peaks.append([type_item.text(), *vals])
    return {
        "x_key": self.x_key(),
        "derivative": self.derivative_enabled(),
        "baseline_model": self.baseline_model(),
        "regions": regions,
        "peaks": peaks,
        "rf_nu": self.rf_nu(),
        "rf_a_mu": self.rf_a_mu_seed(),
        "rf_a_p": self.rf_a_p_seed(),
    }


def restore_analysis_state(self, state: dict) -> None:
    """Restore view options + region/peak tables from :meth:`analysis_state`.

    Does not re-render or re-fit (the caller drives that); signals are
    blocked so setting the x-axis does not wipe the tables it just filled.
    """
    x_key = state.get("x_key", "field")
    with QSignalBlocker(self._x_combo):
        if x_key in self._X_KEYS:
            self._x_combo.setCurrentIndex(self._X_KEYS.index(x_key))
    self._update_derivative_label()
    with QSignalBlocker(self._derivative_check):
        self._derivative_check.setChecked(bool(state.get("derivative", False)))
    model_idx = self._baseline_model_combo.findText(str(state.get("baseline_model", "Linear")))
    if model_idx >= 0:
        self._baseline_model_combo.setCurrentIndex(model_idx)

    for key, spin in (
        ("rf_nu", self._rf_nu_spin),
        ("rf_a_mu", self._rf_a_mu_spin),
        ("rf_a_p", self._rf_a_p_spin),
    ):
        if key in state:
            try:
                spin.setValue(float(state[key]))
            except (TypeError, ValueError):
                pass

    with QSignalBlocker(self._regions_table):
        self._regions_table.setRowCount(0)
        for region in self._coerce_rows(state.get("regions"), 2):
            row = self._regions_table.rowCount()
            self._regions_table.insertRow(row)
            for col, val in enumerate(region):
                self._regions_table.setItem(row, col, QTableWidgetItem(repr(val)))

    with QSignalBlocker(self._peaks_table):
        self._peaks_table.setRowCount(0)
        for peak in state.get("peaks", []):
            if not isinstance(peak, (list, tuple)) or len(peak) != 4:
                continue
            if str(peak[0]) not in self._PEAK_COMPONENTS:
                continue
            vals = self._coerce_rows([peak[1:]], 3)
            if not vals:
                continue
            row = self._peaks_table.rowCount()
            self._peaks_table.insertRow(row)
            type_item = QTableWidgetItem(str(peak[0]))
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._peaks_table.setItem(row, 0, type_item)
            for col, val in zip((1, 2, 3), vals[0], strict=True):
                self._peaks_table.setItem(row, col, QTableWidgetItem(repr(val)))
