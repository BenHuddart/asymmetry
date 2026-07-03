"""Fit-panel shared foundations: parameter table, delegates, tie dialog, and
module-level helpers shared by the single and global fit tabs.

Split out of ``fit_panel.py`` (Phase 2 mechanical split).
"""

import copy
import html
import re
from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.parameters import (
    AffineTie,
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
)
from asymmetry.gui.fit_settings import fit_quality_confidence
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.widgets import (
    apply_param_table_style,
    configure_formula_label,
    fit_quality_chip_html,
    make_formula_box,
    success_html,
    warning_html,
)
from asymmetry.gui.utils.formatting import format_param_label
from asymmetry.gui.widgets.axis_limits import FloatLimitField
from asymmetry.gui.widgets.no_scroll_spin import NoScrollDoubleSpinBox

from .seeding import _field_value_overrides


def _grouped_formula_string(model: CompositeModel) -> str:
    """Return grouped-fit formula text with fit-function amplitudes suppressed."""
    formula = model.formula_string()
    formula = re.sub(r"\bA(?:_\d+)?\*\(", "(", formula)
    formula = re.sub(r"\bA(?:_\d+)?\*", "", formula)
    return formula


def _refresh_field_defaults_in_table(
    table: QTableWidget,
    model: CompositeModel,
    *,
    previous_field_gauss: float,
    current_field_gauss: float,
) -> None:
    """Update field-like parameter rows when they still hold the prior auto-default."""
    if np.isclose(previous_field_gauss, current_field_gauss):
        return

    previous_overrides = _field_value_overrides(model, previous_field_gauss)
    current_overrides = _field_value_overrides(model, current_field_gauss)
    if not previous_overrides and not current_overrides:
        return

    row_by_name = _param_table_rows_by_name(table)
    previous_signal_state = table.blockSignals(True)
    try:
        for pname in set(previous_overrides) | set(current_overrides):
            row = row_by_name.get(pname)
            if row is None:
                continue
            value_item = table.item(row, 1)
            if value_item is None:
                continue
            previous_value = previous_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            current_value = current_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            try:
                existing_value = float(value_item.text())
            except (TypeError, ValueError):
                existing_value = previous_value
            if value_item.text().strip() and not np.isclose(existing_value, previous_value):
                continue
            value_item.setText(f"{float(current_value):.6g}")
    finally:
        table.blockSignals(previous_signal_state)


def _normalized_model_param_values(
    model: CompositeModel,
    param_values: dict[str, float],
) -> dict[str, float]:
    """Return display-ready values for one composite-model parameter set."""
    return model.normalized_parameter_values(param_values)


def _param_table_rows_by_name(table: QTableWidget) -> dict[str, int]:
    rows: dict[str, int] = {}
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        name = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(name, str):
            rows[name] = row
    return rows


def _parse_param_table_float(table: QTableWidget, row: int, default: float = 0.0) -> float:
    item = table.item(row, 1)
    if item is None:
        return float(default)
    try:
        value = float(item.text())
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _set_param_table_value(table: QTableWidget, row: int, value: float) -> None:
    item = table.item(row, 1)
    if item is not None:
        item.setText(f"{float(value):.6g}")


def _synchronize_fraction_group_values_in_table(
    table: QTableWidget,
    model: CompositeModel,
    *,
    edited_param_name: str | None = None,
) -> None:
    row_by_name = _param_table_rows_by_name(table)
    for group in model.fraction_parameter_groups():
        if len(group) < 2:
            continue
        if edited_param_name is not None and edited_param_name not in group:
            continue

        editable_names = group[:-1]
        final_name = group[-1]
        values: dict[str, float] = {}
        for name in editable_names:
            row = row_by_name.get(name)
            if row is None:
                continue
            values[name] = min(max(_parse_param_table_float(table, row, 0.0), 0.0), 1.0)

        if edited_param_name in editable_names:
            remaining = 1.0 - sum(
                values[name] for name in editable_names if name != edited_param_name
            )
            values[edited_param_name] = min(values[edited_param_name], max(0.0, remaining))
        else:
            running_sum = 0.0
            for name in editable_names:
                values[name] = min(values.get(name, 0.0), max(0.0, 1.0 - running_sum))
                running_sum += values[name]

        for name, value in values.items():
            row = row_by_name.get(name)
            if row is not None:
                _set_param_table_value(table, row, value)

        final_row = row_by_name.get(final_name)
        if final_row is not None:
            final_value = max(0.0, 1.0 - sum(values.get(name, 0.0) for name in editable_names))
            _set_param_table_value(table, final_row, final_value)


def _configure_fraction_rows_in_table(
    table: QTableWidget,
    model: CompositeModel,
    *,
    min_column: int | None = None,
    max_column: int | None = None,
    bounds_column: int | None = None,
    type_column: int | None = None,
) -> None:
    row_by_name = _param_table_rows_by_name(table)
    final_fraction_names = {group[-1] for group in model.fraction_parameter_groups() if group}
    all_fraction_names = {name for group in model.fraction_parameter_groups() for name in group}

    for name in all_fraction_names:
        row = row_by_name.get(name)
        if row is None:
            continue
        value_item = table.item(row, 1)
        if value_item is not None:
            tooltip = (
                "Final fraction is computed automatically."
                if name in final_fraction_names
                else "Edit the first n-1 fractions; the final fraction is the remainder to 1."
            )
            value_item.setToolTip(tooltip)
            if name in final_fraction_names:
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        if type_column is not None:
            type_combo = table.cellWidget(row, type_column)
            if isinstance(type_combo, QComboBox):
                type_combo.setToolTip(tooltip)
                if name in final_fraction_names:
                    fixed_index = type_combo.findText("Fixed")
                    if fixed_index >= 0:
                        type_combo.setCurrentIndex(fixed_index)
                    type_combo.setEnabled(False)
                else:
                    type_combo.setEnabled(True)

        if min_column is not None:
            min_item = table.item(row, min_column)
            if min_item is not None:
                min_item.setText("0.0")
        if max_column is not None:
            max_item = table.item(row, max_column)
            if max_item is not None:
                max_item.setText("1.0")
        if bounds_column is not None:
            bounds_item = table.item(row, bounds_column)
            if bounds_item is not None:
                bounds_item.setText("0, 1")


def _get_file_value_for_parameter(
    dataset: MuonDataset | None,
    param_base_name: str,
) -> float | None:
    """Get the file-specific value for a parameter from dataset metadata.

    Returns the value in Gauss for field-like parameters, or None if not available.
    """
    if dataset is None:
        return None
    if param_base_name in {"field", "B_L"}:
        if dataset.run is not None and hasattr(dataset.run, "field"):
            return float(dataset.run.field)
        # Fall back to dataset-level metadata (e.g. datasets without a Run object).
        raw = dataset.metadata.get("field")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
    return None


def _fit_summary(result) -> dict:
    """Shared fit summary (χ² verdict + params-at-bound) for *result*, once.

    Uses the user-configured quality-band confidence so the live fit-panel chip
    matches the verdict persisted onto the fit record. Computing the whole summary
    once (rather than re-deriving the verdict and the bound list separately) keeps
    the chip and its tooltip from each re-running the scipy band math and the
    parameter scan. Returns ``{}`` on any failure so callers degrade gracefully.
    """
    try:
        return fit_result_summary(result, confidence=fit_quality_confidence())
    except Exception:
        return {}


def _fit_range_provenance_text(min_spin, max_spin, unit_label) -> str | None:
    """Format the fit range as a provenance string, or ``None`` if degenerate.

    Shared by the Single and Batch tabs (identical fit-range widgets) so the
    ``fit_range`` stamped onto persisted records has one formatting source.
    Reads the spin values, decimals, and the domain unit label (µs/MHz);
    returns ``None`` when the spinboxes are disabled or the range is empty.
    """
    if not min_spin.isEnabled():
        return None
    lo = float(min_spin.value())
    hi = float(max_spin.value())
    if not hi > lo:
        return None
    decimals = min_spin.decimals()
    return f"{lo:.{decimals}f}–{hi:.{decimals}f} {unit_label.text()}"


def _apply_fit_range_display(
    domain: str,
    min_spin: FloatLimitField,
    max_spin: FloatLimitField,
    x_min: float | None,
    x_max: float | None,
) -> None:
    """Shared fit-range spinbox update for :class:`SingleFitTab`/:class:`GlobalFitTab`.

    In the time domain a plot always supplies a range (seeded to the full
    dataset extent), so the spins simply mirror it, disabled when the plot
    genuinely has none (no dataset). In the frequency domain there is no
    draggable range selector, so the spins stay editable regardless — an
    unset range shows a "full spectrum" placeholder instead of a leftover
    value from a previous domain/run (D6/F15).
    """
    have_range = x_min is not None and x_max is not None
    if domain == "frequency":
        min_spin.setEnabled(True)
        max_spin.setEnabled(True)
        if not have_range:
            min_spin.set_unset("full spectrum")
            max_spin.set_unset("full spectrum")
            return
    else:
        min_spin.setEnabled(have_range)
        max_spin.setEnabled(have_range)
        if not have_range:
            return
    with QSignalBlocker(min_spin):
        min_spin.setValue(float(x_min))
    with QSignalBlocker(max_spin):
        max_spin.setValue(float(x_max))


def _fit_success_html(result) -> str:
    """Return compact success HTML for the result label, with a χ² verdict chip."""
    npar = len(result.parameters.free_parameters)
    ndof = (
        round(result.chi_squared / result.reduced_chi_squared)
        if result.reduced_chi_squared > 0
        else 0
    )
    stats = f"χ²/ν = {result.reduced_chi_squared:.4f} · npar = {npar} · ndof = {ndof}"
    if result.edm is not None:
        stats += f" · Δ‖p‖ = {result.edm:.2e}"
    summary = _fit_summary(result)
    stats += fit_quality_chip_html(summary.get("quality"), summary.get("params_at_bound"))
    return success_html("Fit converged", detail=stats)


def _fit_warnings_html(result) -> str:
    """Return one warning row per advisory the engine emitted during the fit.

    #100/#101 emit :class:`AsymmetryScaleWarning` /
    :class:`FixedFrequencyFieldMismatchWarning` from the engine; the engine now
    carries their messages on ``FitResult.warnings`` so the panel can surface them
    here (they also still reach the Python log). Returns an empty string when the
    fit warned about nothing, so callers can append unconditionally. Messages are
    HTML-escaped — they interpolate user/run values (field, frequency, repr'd
    parameter names) that must not be treated as markup.
    """
    messages = list(getattr(result, "warnings", None) or [])
    if not messages:
        return ""
    return "".join(f"<br>{warning_html('⚠ ' + html.escape(str(m)))}" for m in messages)


_apply_param_table_style = apply_param_table_style


#: Data role storing a parameter's batch role (global/local/fixed) on its
#: read-only "Batch" cell, so a piped-back single fit shows how each parameter
#: was classified in the batch fit and keeps that across selection switches / save.
_PARAM_BATCH_ROLE_DATA = Qt.ItemDataRole.UserRole + 1

#: Column of the single-fit parameter table holding the batch-role read-out.
_SINGLE_PARAM_BATCH_COLUMN = 5

#: Column of the single-fit parameter table holding the equality link-group
#: selector (WiMDA "Ties"). Index 0 of the combo means "unlinked".
_SINGLE_PARAM_LINK_COLUMN = 6

#: Number of equality link groups offered in the single-fit table, matching
#: WiMDA's four groups.
_LINK_GROUP_COUNT = 4

#: Column of the single-fit parameter table holding the affine-tie editor button
#: (offset / equal-spacing ties; beyond WiMDA's equality links).
_SINGLE_PARAM_TIE_COLUMN = 7

_PARAM_ROLE_LABELS = {"global": "Global", "local": "Local", "fixed": "Fixed", "file": "File"}

#: Width (px) of the "Name"/"Parameter" column shared by every parameter table.
#: Kept narrow so the Batch tab does not outgrow the Single tab; "name (unit)"
#: labels that overflow it are still readable via the per-cell tooltip
#: (:func:`_make_param_name_item`). One constant so the tables stay aligned.
PARAM_NAME_COL_WIDTH = 92


def _format_tie_formula(name: str, tie: AffineTie | None) -> str:
    """Render an affine tie as a compact human-readable equation."""
    if tie is None:
        return f"{name} is free"
    terms: list[str] = []
    scale = tie.scale
    if scale == 1.0:
        terms.append(tie.main)
    elif scale == -1.0:
        terms.append(f"-{tie.main}")
    else:
        terms.append(f"{scale:g}·{tie.main}")
    if tie.offset is not None and tie.offset_scale != 0.0:
        sign = "-" if tie.offset_scale < 0 else "+"
        mag = abs(tie.offset_scale)
        terms.append(f"{sign} {tie.offset}" if mag == 1.0 else f"{sign} {mag:g}·{tie.offset}")
    if tie.const:
        terms.append(f"{'-' if tie.const < 0 else '+'} {abs(tie.const):g}")
    return f"{name} = {' '.join(terms)}"


class AffineTieDialog(QDialog):
    """Edit an affine tie for one parameter: ``scale·main + offset_scale·offset + const``.

    Offers the other (non-tied) parameters as the ``main``/``offset`` references,
    so the resulting tie always references valid, untied parameters (the engine
    rejects unknown refs and tie-to-tie chains). Equal spacing is expressed
    against existing parameters, e.g. a lower satellite ``f_lo = 2·f_c − f_hi``.
    The free-auxiliary-``delta`` form is authored via the API; this dialog edits
    ties between parameters already in the table.
    """

    def __init__(
        self,
        param_name: str,
        candidates: list[str],
        current: AffineTie | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Affine tie — {format_param_label(param_name)}")
        self._param_name = param_name
        self._candidates = candidates

        layout = QVBoxLayout(self)
        self._enable = QCheckBox("Tie this parameter to others")
        layout.addWidget(self._enable)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self._main = QComboBox()
        self._main.addItems(candidates)
        self._scale = self._make_coeff_spin(1.0)
        self._offset = QComboBox()
        self._offset.addItem("(none)", None)
        for name in candidates:
            self._offset.addItem(name, name)
        self._offset_scale = self._make_coeff_spin(1.0)
        self._const = self._make_coeff_spin(0.0)
        form.addRow("Main (× scale):", self._main)
        form.addRow("Scale:", self._scale)
        form.addRow("Offset param:", self._offset)
        form.addRow("Offset scale:", self._offset_scale)
        form.addRow("Constant:", self._const)
        layout.addWidget(form_widget)

        self._formula = QLabel()
        self._formula.setWordWrap(True)
        layout.addWidget(self._formula)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Seed from the current tie (if any) and wire live updates.
        if current is not None and current.main in candidates:
            self._enable.setChecked(True)
            self._main.setCurrentText(current.main)
            self._scale.setValue(current.scale)
            if current.offset is not None and current.offset in candidates:
                self._offset.setCurrentIndex(self._offset.findData(current.offset))
            self._offset_scale.setValue(current.offset_scale)
            self._const.setValue(current.const)

        self._enable.toggled.connect(self._refresh)
        self._main.currentIndexChanged.connect(self._refresh)
        self._scale.valueChanged.connect(self._refresh)
        self._offset.currentIndexChanged.connect(self._refresh)
        self._offset_scale.valueChanged.connect(self._refresh)
        self._const.valueChanged.connect(self._refresh)
        self._form_widget = form_widget
        self._refresh()

    @staticmethod
    def _make_coeff_spin(default: float) -> QDoubleSpinBox:
        spin = NoScrollDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(4)
        spin.setValue(default)
        return spin

    def _refresh(self) -> None:
        enabled = self._enable.isChecked() and bool(self._candidates)
        self._form_widget.setEnabled(enabled)
        self._offset_scale.setEnabled(enabled and self._offset.currentData() is not None)
        self._formula.setText(_format_tie_formula(self._param_name, self.tie()))

    def tie(self) -> AffineTie | None:
        if not self._enable.isChecked() or not self._candidates:
            return None
        offset = self._offset.currentData()
        return AffineTie(
            main=self._main.currentText(),
            scale=float(self._scale.value()),
            offset=offset,
            offset_scale=float(self._offset_scale.value()),
            const=float(self._const.value()),
        )


def _make_tie_button() -> QPushButton:
    """Per-row affine-tie editor button for the single-fit table.

    The button text shows ``—`` when untied and ``ƒ`` when a tie is set; the
    current :class:`AffineTie` (or ``None``) is stashed on the button so the
    table read-out and project round-trip can recover it.
    """
    button = QPushButton("—")
    button.setFlat(True)
    button.setMaximumWidth(36)
    button._affine_tie = None  # type: ignore[attr-defined]
    button.setToolTip("Affine tie: derive this parameter from others (offset / equal spacing).")
    return button


def _tie_button_value(button: QPushButton | None) -> AffineTie | None:
    if not isinstance(button, QPushButton):
        return None
    return getattr(button, "_affine_tie", None)


def _set_tie_button_value(button: QPushButton | None, tie: AffineTie | None) -> None:
    if not isinstance(button, QPushButton):
        return
    button._affine_tie = tie  # type: ignore[attr-defined]
    button.setText("ƒ" if tie is not None else "—")
    if tie is not None:
        button.setToolTip(_format_tie_formula(_param_name_from_tie_button(button), tie))
    else:
        button.setToolTip("Affine tie: derive this parameter from others (offset / equal spacing).")


def _param_name_from_tie_button(button: QPushButton) -> str:
    """Best-effort parameter name for a tie button's tooltip (set by the panel)."""
    return getattr(button, "_param_name", "")


def _coerce_bound(text, default: float) -> float:
    """Parse a saved min/max value (number or '-inf'/'inf' text) to a float."""
    if text is None:
        return default
    s = str(text).strip()
    if s in ("", "-inf", "inf", "+inf"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _parameter_from_state_dict(entry: dict) -> Parameter:
    """Reconstruct a :class:`Parameter` from a saved single-fit param entry."""
    raw_tie = entry.get("tie")
    tie = AffineTie.from_dict(raw_tie) if isinstance(raw_tie, dict) else None
    raw_link = entry.get("link_group")
    link_group = int(raw_link) if isinstance(raw_link, (int, float)) else None
    return Parameter(
        name=str(entry.get("name", "")),
        value=float(entry.get("value", 0.0) or 0.0),
        min=_coerce_bound(entry.get("min"), -float("inf")),
        max=_coerce_bound(entry.get("max"), float("inf")),
        fixed=bool(entry.get("fixed", False)),
        link_group=link_group,
        tie=tie,
    )


def _make_link_group_combo() -> QComboBox:
    """Build the per-row equality link-group selector for the single-fit table.

    Item 0 is "—" (unlinked); items 1..N select link groups 1..N. Parameters
    sharing a non-zero group are constrained equal during the fit, and every
    non-main member drops out of the free-fit set.
    """
    combo = QComboBox()
    combo.addItem("—", None)
    for gid in range(1, _LINK_GROUP_COUNT + 1):
        combo.addItem(str(gid), gid)
    combo.setToolTip(
        "Link group (equality tie): parameters sharing a group are fit as one "
        "value; only the group's main parameter is free."
    )
    return combo


def _link_group_combo_value(combo: QComboBox | None) -> int | None:
    """Return the selected link-group id for a link-column combo (None if unlinked)."""
    if not isinstance(combo, QComboBox):
        return None
    data = combo.currentData()
    return int(data) if data is not None else None


def _set_link_group_combo_value(combo: QComboBox | None, group: int | None) -> None:
    """Select ``group`` in a link-column combo (index 0 / "—" when None)."""
    if not isinstance(combo, QComboBox):
        return
    if group is None:
        combo.setCurrentIndex(0)
        return
    idx = combo.findData(group)
    if idx < 0:
        # A group id outside the default 1..N range (e.g. from a hand-edited
        # project, or one written by a build whose core exposes more groups):
        # add it rather than silently dropping the link assignment.
        combo.addItem(str(group), group)
        idx = combo.findData(group)
    combo.setCurrentIndex(idx)


def _make_param_name_item(label: str, raw_name: str) -> QTableWidgetItem:
    """Return a read-only, bold-mono name item for a parameter table.

    The full formatted label is also set as the tooltip: the Name column is kept
    narrow so the Batch tab does not outgrow the Single tab, so names+units can
    clip (``f (MHz)`` → ``f (MH…``); the tooltip guarantees the full name/unit is
    always readable on hover, regardless of column width.
    """
    item = QTableWidgetItem(label)
    item.setData(Qt.ItemDataRole.UserRole, raw_name)
    item.setToolTip(label)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    nf = mono_font(11.0)
    nf.setBold(True)
    item.setFont(nf)
    return item


def _set_param_batch_role_cell(table: QTableWidget, row: int, role: str | None) -> None:
    """Set the read-only 'Batch' role cell for a single-fit parameter row.

    A ``None``/unknown role blanks the cell. The raw role is stored on the cell so
    :meth:`SingleFitTab.get_state` can round-trip it.
    """
    item = table.item(row, _SINGLE_PARAM_BATCH_COLUMN)
    if item is None:
        item = QTableWidgetItem()
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table.setItem(row, _SINGLE_PARAM_BATCH_COLUMN, item)
    if isinstance(role, str) and role in _PARAM_ROLE_LABELS:
        item.setText(_PARAM_ROLE_LABELS[role])
        item.setToolTip(f"Batch fit role: {role}")
        item.setData(_PARAM_BATCH_ROLE_DATA, role)
    else:
        item.setText("")
        item.setToolTip("")
        item.setData(_PARAM_BATCH_ROLE_DATA, None)


_configure_formula_label = configure_formula_label
_make_formula_box = make_formula_box


def _set_formula_label_text(label: QLabel, formula: str, **_kwargs) -> None:
    """Set formula text; tooltip preserves the raw string for reference.

    When the label lives in a FormulaBox, route through it so the expression is
    break-marked (wraps only at top-level operators) and the box re-measures.
    """
    box = getattr(label, "_formula_box", None)
    if box is not None:
        box.set_formula(formula)
        return
    raw_text = str(formula)
    label.setText(raw_text)
    label.setToolTip(raw_text)


def _dataset_representation_domain(dataset: MuonDataset | None) -> str:
    """Return the analysis domain a dataset's samples live in.

    Frequency spectra produced by the Fourier panel carry
    ``metadata["plot_domain"] == "frequency"``; every other dataset is a
    time-domain asymmetry/counts representation. Used to refuse a fit whose
    declared domain does not match the data actually loaded — fitting a
    time-domain model against an FFT spectrum (or vice versa) silently produces
    a meaningless result.
    """
    if dataset is None:
        return "time"
    metadata = getattr(dataset, "metadata", None)
    if isinstance(metadata, dict):
        if str(metadata.get("plot_domain", "")).strip().lower() == "frequency":
            return "frequency"
    return "time"


def _fit_domain_mismatch_message(fit_domain: str, dataset: MuonDataset | None) -> str | None:
    """Return a user-facing refusal when the fit domain and data disagree.

    ``None`` means the data matches the fit domain and the fit may proceed.
    """
    data_domain = _dataset_representation_domain(dataset)
    if data_domain == fit_domain:
        return None
    if data_domain == "frequency":
        return (
            "the workspace is showing the frequency-domain spectrum, but this is a "
            "time-domain fit. Switch the central plot to a time-domain view "
            "(Time domain ▸ F-B asymmetry) before fitting — otherwise the "
            "time-domain model is fitted against the FFT spectrum and the result "
            "is meaningless."
        )
    return (
        "this is a frequency-domain fit, but the selected data is a time-domain "
        "representation. Switch the central plot to the Frequency-domain (FFT) "
        "view before fitting."
    )


def _apply_domain_mismatch_warning(label: QLabel, model: CompositeModel, domain: str) -> None:
    """Flag a model containing components from the wrong analysis domain.

    Such models can only come from projects saved before domain filtering (or
    hand-edited files); they are kept loaded and fittable so nothing the user
    saved is destroyed, but the formula label is marked so the mismatch is
    visible, and Edit Function explains which component is foreign.
    """
    foreign = {d for d in model.domains() if d != domain}
    if not foreign:
        label.setStyleSheet("")
        return
    foreign_names = sorted(
        component.name for component in model.components if component.domain != domain
    )
    label.setStyleSheet(f"color: {tokens.WARN}; font-weight: bold;")
    label.setText("\u26a0 " + label.text())
    label.setToolTip(
        f"This model contains {'/'.join(sorted(foreign))}-domain component(s) "
        f"({', '.join(foreign_names)}) but the representation is fitted in the "
        f"{domain} domain. The model is kept as saved and can still be fitted; "
        "use Edit Function to repair it."
    )
    box = getattr(label, "_formula_box", None)
    if box is not None:
        box.refresh_height()


def _model_without_trailing_background(model: CompositeModel | None) -> CompositeModel | None:
    """Return *model* with a trailing additive ``Constant`` removed, or ``None``.

    Only the unambiguous case is handled — a final ``+ Constant`` term outside
    any parentheses (e.g. ``Exponential + Constant`` or
    ``Oscillatory*Exponential + Constant``). A free constant background absorbs
    part of the signal during amplitude calibration, splitting the fitted
    amplitude; dropping it lets the relaxation term capture the full initial
    asymmetry (A₀). Returns ``None`` when there is no such removable background.
    """
    if model is None:
        return None
    names = list(model.component_names)
    operators = list(model.operators)
    if len(names) < 2 or names[-1] != "Constant":
        return None
    if not operators or operators[-1] != "+":
        return None
    if any(model.open_parentheses) or any(model.close_parentheses):
        return None
    try:
        return CompositeModel(names[:-1], operators=operators[:-1])
    except ValueError:
        return None


def _format_bounds_pair(min_val: float, max_val: float) -> str:
    def _format(value: float) -> str:
        if value == float("inf"):
            return "inf"
        if value == -float("inf"):
            return "-inf"
        return f"{float(value):.6g}"

    return f"{_format(min_val)}, {_format(max_val)}"


def _format_fit_worker_exception(exc: Exception) -> str:
    """Return a clearer user-facing error message for fit worker failures."""
    if isinstance(exc, KeyError):
        missing = exc.args[0] if exc.args else "unknown"
        return f"Missing fit parameter mapping for dataset/group key: {missing!r}"

    exc_name = type(exc).__name__
    text = str(exc).strip()
    if not text:
        return exc_name
    if text == exc_name:
        return text
    return f"{exc_name}: {text}"


_GLOBAL_FIT_PARAMETER_CLASSIFICATION_HELP_TEXT = (
    "Specify how each parameter behaves across datasets:\n\n"
    "Global: Same value for all datasets. Use this for shared physical parameters "
    "that should be fitted once across the full selection.\n\n"
    "Local: Different value for each dataset. Use this when the parameter is "
    "expected to vary from run to run.\n\n"
    "Fixed: Held constant at the specified value for every dataset. Use this for "
    "known values or parameters you want excluded from optimization.\n\n"
    "File: Use the value from dataset metadata where available. This is offered for "
    "field-like parameters such as B_L when the run file already stores the relevant value.\n\n"
    "The Seed column is the shared initial value applied to every run in the batch — "
    "it is not a per-run fitted result and does not change when you select different "
    "runs. Per-run fitted values appear in the Parameters tab after the batch fit completes."
)


def _fit_curve_sample_count(
    model: CompositeModel,
    param_values: dict[str, float],
    t_min: float,
    t_max: float,
    *,
    base_points: int = 500,
    points_per_cycle: int = 40,
    max_points: int = 20000,
) -> int:
    """Return a dense-enough sample count for plotting oscillatory models."""
    duration = max(float(t_max) - float(t_min), 0.0)
    if duration <= 0.0:
        return base_points

    max_frequency_mhz = 0.0
    for name, value in param_values.items():
        base_name, _index = split_parameter_name(name)
        try:
            numeric_value = abs(float(value))
        except (TypeError, ValueError):
            continue

        if base_name == "frequency":
            max_frequency_mhz = max(max_frequency_mhz, numeric_value)
        elif base_name == "field":
            field_frequency = MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA * numeric_value
            max_frequency_mhz = max(max_frequency_mhz, field_frequency)
        elif base_name in {"A_hf", "D_mu", "f_dip", "f_quad"}:
            # Hyperfine/dipolar couplings set the oscillation scale of the
            # muonium and spin-J components (lines up to ~A_hf in MHz).
            max_frequency_mhz = max(max_frequency_mhz, numeric_value)

    if max_frequency_mhz <= 0.0:
        return base_points

    cycles = max_frequency_mhz * duration
    required_points = int(np.ceil(cycles * points_per_cycle)) + 1
    return int(max(base_points, min(max_points, required_points)))


def _fit_work_pending(panel) -> bool:
    """True while *panel* has a worker fit in flight on its TaskRunner."""
    runner = getattr(panel, "_fit_call_runner", None)
    return runner is not None and runner.active_count > 0


def _wait_for_fit_thread(panel, timeout_ms: int = 30_000) -> bool:
    """Run a nested event loop until *panel*'s worker fits fully complete.

    Completion covers every worker fit (single, global, grouped, grouped-series
    and count-domain), all of which run on the panel's TaskRunner. Used by
    tests and synchronous callers; returns ``False`` on timeout.
    """
    if not _fit_work_pending(panel):
        return True
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: None if _fit_work_pending(panel) else loop.quit())
    check.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    check.stop()
    return not _fit_work_pending(panel)


def _start_fit_call(
    panel,
    call,
    *,
    on_finished,
    on_error,
    on_cancelled,
):
    """Start one prepared fit call on *panel*'s TaskRunner and return its worker.

    The call is built on the GUI thread with every argument already bound
    (e.g. ``functools.partial(engine.fit, dataset, fn, params, minos=...)``);
    the engine's ``cancel_callback`` kwarg is supplied from the worker's own
    cooperative flag. TaskRunner owns the whole thread lifecycle — including
    the GUI-thread relay for the callbacks and a bounded, Windows-safe
    shutdown — so the panel holds no thread state of its own. Engine errors
    are reformatted via :func:`_format_fit_worker_exception` before reaching
    ``on_error``.
    """

    def task(worker, call=call):
        try:
            return call(cancel_callback=worker.is_cancelled)
        except FitCancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(_format_fit_worker_exception(exc)) from exc

    return panel._fit_call_runner.start(
        task,
        on_finished=on_finished,
        on_error=on_error,
        on_cancelled=on_cancelled,
        cancel_exceptions=(FitCancelledError,),
    )


class _CommitOnTabDelegate(QStyledItemDelegate):
    """Item delegate that commits the open editor when Tab/Backtab is pressed.

    The parameter tables interleave focusable cell widgets (the Fix checkbox,
    the Type/Link combos) between the editable item cells. Pressing Tab in an
    open cell editor makes Qt's focus traversal jump to the adjacent cell
    widget *without* routing through the item-view's editor-commit path, so the
    value the user just typed is silently discarded (Return and clicking away
    commit normally — only Tab loses the edit). Intercepting Tab/Backtab here
    and committing explicitly closes that gap: Tab now commits the typed value
    and advances to the next editable cell, matching Return.
    """

    def eventFilter(self, editor: QWidget, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            key = event.key()
            if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                hint = (
                    QAbstractItemDelegate.EndEditHint.EditPreviousItem
                    if key == Qt.Key.Key_Backtab
                    else QAbstractItemDelegate.EndEditHint.EditNextItem
                )
                # Commit directly (not via the commitData signal): emitting
                # commitData lets a focus change to the adjacent cell widget reset
                # the editor to the model value before setModelData reads it, so
                # the typed value is lost. Calling setModelData here reads the
                # editor while it still holds the typed text.
                view = self.parent()
                if isinstance(view, QAbstractItemView):
                    index = view.currentIndex()
                    if index.isValid():
                        self.setModelData(editor, view.model(), index)
                self.closeEditor.emit(editor, hint)
                return True
        return super().eventFilter(editor, event)


class _ValueUncertaintyDelegate(_CommitOnTabDelegate):
    """Paints 'value  ±σ' in a table cell; editing shows only the bare value.

    Uncertainty is stored in UserRole+1 alongside the item text. When an opt-in
    MINOS asymmetric interval is present (UserRole+2, a ``(lower, upper)`` pair with
    ``lower < 0 < upper``), the cell instead shows ``value  +upper / lower`` — the
    display-only asymmetric overlay. Both roles are cleared when the user edits.
    """

    _UNC_ROLE = Qt.ItemDataRole.UserRole + 1
    _MINOS_ROLE = Qt.ItemDataRole.UserRole + 2
    _MUTED = QColor(tokens.TEXT_MUTED)

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        minos = index.data(self._MINOS_ROLE)
        unc = index.data(self._UNC_ROLE)
        if minos is not None and len(minos) == 2:
            lower, upper = float(minos[0]), float(minos[1])
            unc_str = f"  +{upper:.4f} / {lower:.4f}"
        elif unc is not None:
            unc_str = f"  ±{float(unc):.4f}"
        else:
            return
        val_text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        style = option.widget.style() if option.widget else QApplication.style()
        text_opt = QStyleOptionViewItem(option)
        self.initStyleOption(text_opt, index)
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, text_opt, option.widget
        )
        val_w = painter.fontMetrics().horizontalAdvance(val_text)
        unc_rect = text_rect.adjusted(val_w, 0, 0, 0)
        painter.save()
        painter.setPen(self._MUTED)
        painter.drawText(
            unc_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, unc_str
        )
        painter.restore()

    def setModelData(self, editor, model, index) -> None:  # noqa: N802
        # Write the edited value BEFORE clearing the ±σ overlay roles. Clearing a
        # role calls model.setData, which emits dataChanged; while the editor is
        # still open the view responds by re-pushing the model value back into the
        # editor (setEditorData), so clearing first would overwrite the freshly
        # typed text and super().setModelData would then read (and commit) the
        # stale value. Committing first reads the real edit; the overlay clear
        # follows (the fitted uncertainty no longer applies to a hand-edited value).
        super().setModelData(editor, model, index)
        model.setData(index, None, self._UNC_ROLE)
        model.setData(index, None, self._MINOS_ROLE)


def _size_param_table_to_content(table: QTableWidget) -> None:
    """Fix a parameter table's height to exactly its rows.

    The inspector dock scrolls vertically as a whole, so the table does not need
    to grow and scroll internally; sizing it to its content means a few-parameter
    model leaves no empty rows, while a many-parameter model simply makes the
    panel taller (and the dock scrolls — the natural axis). The horizontal
    scrollbar's height is reserved so wide column sets never clip the last row.
    """
    table.resizeRowsToContents()
    rows_height = table.verticalHeader().length()
    header_height = table.horizontalHeader().sizeHint().height()
    frame = 2 * table.frameWidth()
    scrollbar = table.horizontalScrollBar().sizeHint().height()
    table.setFixedHeight(rows_height + header_height + frame + scrollbar)


def _shift_rrf_parameters(
    parameters: ParameterSet, offsets: dict[str, float], *, sign: int
) -> ParameterSet:
    """Shift the rotation parameters of a set by ±ν₀ (value and bounds together).

    ``sign=-1`` maps a lab-frame seed to the rotating frame (δν = ν − ν₀) for
    the engine, which fits the small offset; ``sign=+1`` maps a rotating-frame
    fit result back to the lab frame for display/recording. The parameter table
    stays entirely lab-frame — what the user reads and edits — while the engine
    works in the better-conditioned δν, and a refit round-trips exactly.
    """
    shifted = ParameterSet()
    for p in parameters:
        delta = offsets.get(p.name, 0.0) * sign
        shifted.add(
            Parameter(
                name=p.name,
                value=float(p.value) + delta,
                min=p.min + delta,  # ±inf + finite stays ±inf
                max=p.max + delta,
                fixed=getattr(p, "fixed", False),
                expr=getattr(p, "expr", None),  # preserve constraints faithfully
                link_group=getattr(p, "link_group", None),
            )
        )
    return shifted


class FitParameterTable(QTableWidget):
    """Reusable fit-parameter table: Name·Value·Fix·Min·Max·Batch·Link·Tie.

    Shared by the single-fit panel (:class:`SingleFitTab`) and the single
    grouped (individual-groups) fit. It owns the per-row Fix checkbox, equality
    Link-group selector and affine Tie button (with their mutual-exclusion
    wiring), the value±uncertainty / commit-on-Tab delegates, fraction-row
    synchronisation, and the parameter read / seed / state round-trip. Hosts
    wrap it for the composite-model, result-text and wizard concerns that are
    not part of the table itself.
    """

    COL_NAME = 0
    COL_VALUE = 1
    COL_FIX = 2
    COL_MIN = 3
    COL_MAX = 4
    COL_BATCH = _SINGLE_PARAM_BATCH_COLUMN  # 5
    COL_LINK = _SINGLE_PARAM_LINK_COLUMN  # 6
    COL_TIE = _SINGLE_PARAM_TIE_COLUMN  # 7

    #: Emitted with the parameter name whose Value cell the user just edited.
    value_edited = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 8, parent)
        self.setHorizontalHeaderLabels(
            ["Name", "Value", "Fix", "Min", "Max", "Batch", "Link", "Tie"]
        )
        self.horizontalHeader().setStretchLastSection(False)
        # Name (col 0) holds formatted "name (unit)" labels; the old 72 px clipped
        # common cases like "f (MHz)" / "A_bg (%)" (tooltip backs the rest).
        name_w = PARAM_NAME_COL_WIDTH
        for col, width in (
            (0, name_w),
            (1, 88),
            (2, 30),
            (3, 52),
            (4, 52),
            (5, 50),
            (6, 40),
            (7, 40),
        ):
            self.setColumnWidth(col, width)
        _apply_param_table_style(self)
        # Tab commits the open editor on every editable column; the Value column
        # additionally paints the ±σ overlay.
        self.setItemDelegate(_CommitOnTabDelegate(self))
        self.setItemDelegateForColumn(self.COL_VALUE, _ValueUncertaintyDelegate(self))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWordWrap(False)

        #: Guards programmatic cell writes so they don't fire fraction-sync /
        #: value_edited (the host sets this around bulk updates via ``suspend``).
        self._updating = False
        self._composite_model: CompositeModel | None = None
        #: Non-model parameters carried from a loaded project (no table row) that
        #: a tie/seed may still reference; preserved across read/state.
        self._auxiliary_param_state: list[dict] = []
        self.itemChanged.connect(self._on_item_changed)

    @contextmanager
    def suspend(self):
        """Suspend value-edit reactions (fraction sync / ``value_edited``).

        Wrap programmatic cell writes (populate, restore, fit-result write-back)
        so they don't trip the user-edit handler.
        """
        prev = self._updating
        self._updating = True
        try:
            yield
        finally:
            self._updating = prev

    @property
    def is_updating(self) -> bool:
        return self._updating

    def set_batch_column_visible(self, visible: bool) -> None:
        """Show/hide the read-only Batch-role column (hidden for grouped fits)."""
        self.setColumnHidden(self.COL_BATCH, not visible)

    # ── populate ────────────────────────────────────────────────────────────

    def populate(
        self,
        model: CompositeModel,
        *,
        value_overrides: dict[str, float] | None = None,
        fixed_names: frozenset[str] | set[str] = frozenset(),
        param_names: list[str] | None = None,
    ) -> None:
        """Build one row per parameter, seeding values and the Fix state.

        ``param_names`` restricts the rows to a subset of the model's parameters
        (e.g. the grouped physics table, which omits per-group nuisance
        amplitudes); fraction-row helpers locate rows by name, so a subset is
        safe. Defaults to every model parameter.
        """
        self._composite_model = model
        # A fresh model build owns the parameter namespace: drop auxiliaries from a
        # previous (different-model) restore so they can't resurrect as ghost
        # parameters in read_parameter_set()/parameters_state(). restore_parameters
        # re-establishes them for the matching model.
        self._auxiliary_param_state = []
        overrides = value_overrides or {}
        names = list(model.param_names) if param_names is None else list(param_names)
        with self.suspend():
            self.setRowCount(len(names))
            for i, pname in enumerate(names):
                self.setItem(
                    i, self.COL_NAME, _make_param_name_item(format_param_label(pname), pname)
                )

                default_val = overrides.get(pname, model.param_defaults.get(pname, 0.0))
                value_item = QTableWidgetItem(str(default_val))
                value_item.setFont(mono_font(11.0))
                self.setItem(i, self.COL_VALUE, value_item)

                fix_widget = QWidget()
                fix_layout = QHBoxLayout(fix_widget)
                fix_layout.setContentsMargins(0, 0, 0, 0)
                fix_checkbox = QCheckBox()
                if pname in fixed_names:
                    fix_checkbox.setChecked(True)
                fix_layout.addWidget(fix_checkbox)
                fix_layout.setAlignment(fix_checkbox, Qt.AlignmentFlag.AlignCenter)
                self.setCellWidget(i, self.COL_FIX, fix_widget)

                default_min = get_param_info(pname).default_min
                min_text = str(default_min) if default_min is not None else "-inf"
                min_item = QTableWidgetItem(min_text)
                min_item.setFont(mono_font(11.0))
                self.setItem(i, self.COL_MIN, min_item)

                max_item = QTableWidgetItem("inf")
                max_item.setFont(mono_font(11.0))
                self.setItem(i, self.COL_MAX, max_item)

                _set_param_batch_role_cell(self, i, None)

                link_combo = _make_link_group_combo()
                self.setCellWidget(i, self.COL_LINK, link_combo)
                # Fix and Link are mutually exclusive (linking wins in the engine,
                # so allowing both would silently discard the fixed value).
                self._wire_fix_link_exclusion(fix_checkbox, link_combo)

                tie_button = _make_tie_button()
                tie_button._param_name = pname  # type: ignore[attr-defined]
                self.setCellWidget(i, self.COL_TIE, tie_button)
                self._wire_tie_button(i, tie_button, fix_checkbox, link_combo)

            _configure_fraction_rows_in_table(
                self, model, min_column=self.COL_MIN, max_column=self.COL_MAX
            )
        self.synchronize_fractions()
        _size_param_table_to_content(self)

    # ── fraction-row sync ───────────────────────────────────────────────────

    def synchronize_fractions(self, edited_param_name: str | None = None) -> None:
        if self._composite_model is None:
            return
        with self.suspend():
            _synchronize_fraction_group_values_in_table(
                self, self._composite_model, edited_param_name=edited_param_name
            )

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != self.COL_VALUE:
            return
        name_item = self.item(item.row(), self.COL_NAME)
        name = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
        if isinstance(name, str):
            self.synchronize_fractions(name)
            self.value_edited.emit(name)

    # ── Fix / Link / Tie wiring ─────────────────────────────────────────────

    def _wire_fix_link_exclusion(self, fix_checkbox: QCheckBox, link_combo: QComboBox) -> None:
        """Keep a row's Fix checkbox and Link-group combo mutually exclusive."""

        def on_fix_toggled(checked: bool) -> None:
            if self._updating:
                return
            if checked and _link_group_combo_value(link_combo) is not None:
                with self.suspend():
                    _set_link_group_combo_value(link_combo, None)
            link_combo.setEnabled(not checked)

        def on_link_changed(_index: int) -> None:
            if self._updating:
                return
            linked = _link_group_combo_value(link_combo) is not None
            if linked and fix_checkbox.isChecked():
                with self.suspend():
                    fix_checkbox.setChecked(False)
            fix_checkbox.setEnabled(not linked)

        fix_checkbox.toggled.connect(on_fix_toggled)
        link_combo.currentIndexChanged.connect(on_link_changed)

    def _tie_candidate_names(self, row: int) -> list[str]:
        """Names a row may reference in a tie: other, *untied* rows + auxiliaries."""
        names: list[str] = []
        for i in range(self.rowCount()):
            if i == row:
                continue
            name_item = self.item(i, self.COL_NAME)
            name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(name, str):
                continue
            if _tie_button_value(self.cellWidget(i, self.COL_TIE)) is not None:
                continue
            names.append(name)
        for entry in self._auxiliary_param_state:
            aux_name = entry.get("name")
            if isinstance(aux_name, str) and aux_name not in names:
                names.append(aux_name)
        return names

    def _wire_tie_button(
        self, row: int, tie_button: QPushButton, fix_checkbox: QCheckBox, link_combo: QComboBox
    ) -> None:
        """Open the affine-tie editor and keep Tie exclusive with Fix/Link."""

        def on_clicked() -> None:
            candidates = self._tie_candidate_names(row)
            if not candidates:
                QMessageBox.information(
                    self,
                    "Affine tie",
                    "An affine tie needs at least one other free parameter to "
                    "reference. Add or untie another parameter first.",
                )
                return
            name = getattr(tie_button, "_param_name", "")
            dialog = AffineTieDialog(name, candidates, _tie_button_value(tie_button), self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            tie = dialog.tie()
            _set_tie_button_value(tie_button, tie)
            with self.suspend():
                if tie is not None:
                    fix_checkbox.setChecked(False)
                    _set_link_group_combo_value(link_combo, None)
            fix_checkbox.setEnabled(tie is None)
            link_combo.setEnabled(tie is None)

        tie_button.clicked.connect(on_clicked)

    # ── read / seed / state ─────────────────────────────────────────────────

    def read_parameter_set(self) -> ParameterSet:
        """Build a :class:`ParameterSet` from the table (raises on a bad value)."""
        parameters = ParameterSet()
        for i in range(self.rowCount()):
            name_item = self.item(i, self.COL_NAME)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            try:
                value = float(self.item(i, self.COL_VALUE).text())
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid value for {format_param_label(param_name)}") from exc

            fix_widget = self.cellWidget(i, self.COL_FIX)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            try:
                min_text = self.item(i, self.COL_MIN).text()
                min_val = float(min_text) if min_text and min_text != "-inf" else -float("inf")
            except (ValueError, AttributeError):
                min_val = -float("inf")
            try:
                max_text = self.item(i, self.COL_MAX).text()
                max_val = float(max_text) if max_text and max_text != "inf" else float("inf")
            except (ValueError, AttributeError):
                max_val = float("inf")

            link_group = _link_group_combo_value(self.cellWidget(i, self.COL_LINK))
            tie = _tie_button_value(self.cellWidget(i, self.COL_TIE))

            parameters.add(
                Parameter(
                    name=param_name,
                    value=value,
                    min=min_val,
                    max=max_val,
                    fixed=fixed,
                    link_group=link_group,
                    tie=tie,
                )
            )
        for entry in self._auxiliary_param_state:
            parameters.add(_parameter_from_state_dict(entry))
        return parameters

    def current_seed_values(self) -> dict[str, str]:
        """Return the live seed text per parameter name (skips non-finite cells)."""
        seeds: dict[str, str] = {}
        for row in range(self.rowCount()):
            name_item = self.item(row, self.COL_NAME)
            if name_item is None:
                continue
            name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(name, str):
                continue
            value_item = self.item(row, self.COL_VALUE)
            if value_item is None:
                continue
            text = value_item.text().strip()
            try:
                float(text)
            except ValueError:
                continue
            seeds[name] = text
        return seeds

    def current_bounds(self) -> dict[str, str]:
        """Return the live ``"min, max"`` bounds text per parameter name.

        Used to carry parameter bounds (not just seed values) when sending a
        single-fit model to the Batch tab, whose classification table stores
        bounds as one combined ``"min, max"`` string per row. Blank/absent cells
        fall back to the open ``-inf``/``inf`` bound.
        """
        bounds: dict[str, str] = {}
        for row in range(self.rowCount()):
            name_item = self.item(row, self.COL_NAME)
            if name_item is None:
                continue
            name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(name, str):
                continue
            min_item = self.item(row, self.COL_MIN)
            max_item = self.item(row, self.COL_MAX)
            min_text = (min_item.text().strip() if min_item else "") or "-inf"
            max_text = (max_item.text().strip() if max_item else "") or "inf"
            bounds[name] = f"{min_text}, {max_text}"
        return bounds

    def parameters_state(self) -> list[dict]:
        """Serialise the table rows (+ auxiliaries) as parameter-state dicts."""
        params: list[dict] = []
        for i in range(self.rowCount()):
            name_item = self.item(i, self.COL_NAME)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else f"param_{i}"
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            value_item = self.item(i, self.COL_VALUE)
            try:
                value = float(value_item.text()) if value_item else 0.0
            except ValueError:
                value = 0.0
            unc = value_item.data(_ValueUncertaintyDelegate._UNC_ROLE) if value_item else None
            unc_asym = (
                value_item.data(_ValueUncertaintyDelegate._MINOS_ROLE) if value_item else None
            )

            fix_widget = self.cellWidget(i, self.COL_FIX)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            min_item = self.item(i, self.COL_MIN)
            max_item = self.item(i, self.COL_MAX)
            role_item = self.item(i, self.COL_BATCH)
            role = role_item.data(_PARAM_BATCH_ROLE_DATA) if role_item is not None else None
            tie = _tie_button_value(self.cellWidget(i, self.COL_TIE))
            params.append(
                {
                    "name": param_name,
                    "value": value,
                    "fixed": fixed,
                    "min": min_item.text() if min_item else "-inf",
                    "max": max_item.text() if max_item else "inf",
                    "uncertainty": unc,
                    "uncertainty_asymmetric": list(unc_asym) if unc_asym is not None else None,
                    "role": role if isinstance(role, str) else None,
                    "link_group": _link_group_combo_value(self.cellWidget(i, self.COL_LINK)),
                    "tie": tie.to_dict() if tie is not None else None,
                }
            )
        params.extend(copy.deepcopy(entry) for entry in self._auxiliary_param_state)
        if self._composite_model is not None:
            normalized = _normalized_model_param_values(
                self._composite_model,
                {str(entry["name"]): float(entry.get("value", 0.0)) for entry in params},
            )
            params = [
                {**entry, "value": normalized.get(str(entry["name"]), entry["value"])}
                for entry in params
            ]
        return params

    def restore_parameters(self, params_data: dict[str, dict]) -> None:
        """Apply saved parameter-state dicts onto the current rows (+ auxiliaries).

        ``populate`` must have been called for the matching model first.
        """
        model = self._composite_model
        normalized_state_values = (
            _normalized_model_param_values(
                model,
                {
                    str(name): float(entry.get("value", 0.0))
                    for name, entry in params_data.items()
                    if entry.get("value") is not None
                },
            )
            if model is not None
            else {}
        )
        with self.suspend():
            for i in range(self.rowCount()):
                name_item = self.item(i, self.COL_NAME)
                param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(param_name, str) and name_item:
                    param_name = name_item.text()
                if param_name not in params_data:
                    continue
                p_data = params_data[param_name]

                value_item = self.item(i, self.COL_VALUE)
                if value_item:
                    value_item.setText(
                        str(normalized_state_values.get(param_name, p_data.get("value", 0.0)))
                    )
                    value_item.setData(
                        _ValueUncertaintyDelegate._UNC_ROLE, p_data.get("uncertainty")
                    )
                    value_item.setData(
                        _ValueUncertaintyDelegate._MINOS_ROLE,
                        p_data.get("uncertainty_asymmetric"),
                    )

                fix_widget = self.cellWidget(i, self.COL_FIX)
                fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
                if fix_checkbox:
                    fix_checkbox.setChecked(bool(p_data.get("fixed", False)))

                min_item = self.item(i, self.COL_MIN)
                if min_item:
                    min_item.setText(str(p_data.get("min", "-inf")))
                max_item = self.item(i, self.COL_MAX)
                if max_item:
                    max_item.setText(str(p_data.get("max", "inf")))

                _set_param_batch_role_cell(self, i, p_data.get("role"))

                link_combo = self.cellWidget(i, self.COL_LINK)
                raw_link = p_data.get("link_group")
                _set_link_group_combo_value(
                    link_combo, int(raw_link) if isinstance(raw_link, (int, float)) else None
                )

                tie_button = self.cellWidget(i, self.COL_TIE)
                raw_tie = p_data.get("tie")
                tie = AffineTie.from_dict(raw_tie) if isinstance(raw_tie, dict) else None
                _set_tie_button_value(tie_button, tie)
                if tie is not None:
                    if fix_checkbox is not None:
                        fix_checkbox.setChecked(False)
                    _set_link_group_combo_value(link_combo, None)
                if fix_checkbox is not None:
                    fix_checkbox.setEnabled(tie is None)
                if link_combo is not None:
                    link_combo.setEnabled(tie is None)

            row_names: set[str] = set()
            for i in range(self.rowCount()):
                ni = self.item(i, self.COL_NAME)
                nm = ni.data(Qt.ItemDataRole.UserRole) if ni else None
                if isinstance(nm, str):
                    row_names.add(nm)
            self._auxiliary_param_state = [
                copy.deepcopy(entry)
                for entry in params_data.values()
                if isinstance(entry, dict) and entry.get("name") not in row_names
            ]
        self.synchronize_fractions()


class FitTabBase(QWidget):
    """Shared base for :class:`SingleFitTab` and :class:`GlobalFitTab`.

    Introduced by the Phase 2 skeleton checkpoint with **no behaviour of its
    own yet**: it exists so the two fit tabs share a common ancestor. The
    machinery duplicated across the tabs — model-formula box, fit-range field
    pair, parameter-table construction, the Stop button, wizard-result caching,
    and shared fit-precondition validation — is hoisted into this base one
    cluster at a time in the following checkpoints. Until then subclasses
    behave exactly as before (``super().__init__`` resolves through here to
    ``QWidget``).
    """
