"""Fit panel — model selection, parameter table, and fit controls.

Mirrors WiMDA's Analyse → Fit dialog: choose a model, set initial
parameters, run the fit, and inspect results.
"""

from __future__ import annotations

import copy
import re

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.domain_library import coerce_domain
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    FitWizardRecommendation,
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_fit_wizard import (
    GlobalCandidateAssessment,
    GlobalFitWizardRecommendation,
    deserialize_global_fit_wizard_recommendation,
    serialize_global_fit_wizard_recommendation,
)
from asymmetry.core.fitting.global_search.heuristics import (
    is_amplitude_parameter,
    is_background_parameter,
)
from asymmetry.core.fitting.grouped_time_domain import (
    GROUP_NUISANCE_PARAMS,
    _group_dataset_run_number,
    build_grouped_count_model,
    build_grouped_time_domain_datasets,
    build_grouped_time_domain_groups,
    fit_grouped_series,
    fit_grouped_time_domain,
    validate_grouped_model_contract,
)
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.spectral import (
    append_frequency_field_derived_parameters,
    default_frequency_model,
    seed_peak_parameters_from_dataset,
)
from asymmetry.core.fourier.fft import estimate_fft_phase, fft_complex_asymmetry
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from asymmetry.gui.panels.initial_values_dialog import InitialValuesDialog
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.widgets import (
    RESULTS_GROUP_SUCCESS_STYLE,
    apply_param_table_style,
    configure_formula_label,
    success_html,
)
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow
from asymmetry.gui.windows.global_fit_wizard_window import GlobalFitWizardWindow


def _format_param_label(name: str) -> str:
    """Return a display label with Greek symbols and units where applicable."""
    return get_param_info(name).unicode_label()


def _field_value_overrides(model: CompositeModel, field_gauss: float) -> dict[str, float]:
    """Return a dict overriding field-like defaults with *field_gauss*.

    Only overrides parameters whose base name is ``"field"`` or ``"B_L"``
    and only when *field_gauss* is non-zero.
    """
    if field_gauss == 0.0:
        return {}
    overrides: dict[str, float] = {}
    for pname in model.param_names:
        base_name, _index = split_parameter_name(pname)
        if base_name in {"field", "B_L"}:
            overrides[pname] = field_gauss
    return overrides


def _seed_group_background_and_n0(
    counts: np.ndarray,
    *,
    time: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Return heuristic grouped-count seeds for background, N0, and amplitude."""
    count_arr = np.asarray(counts, dtype=float)
    time_arr = np.asarray(time, dtype=float) if time is not None else None
    if time_arr is not None and time_arr.shape == count_arr.shape:
        finite_mask = np.isfinite(count_arr) & np.isfinite(time_arr)
        count_arr = count_arr[finite_mask]
        time_arr = time_arr[finite_mask]
    else:
        time_arr = None
        count_arr = count_arr[np.isfinite(count_arr)]

    if count_arr.size == 0:
        return 0.0, 100.0, 0.2

    if time_arr is None:
        time_arr = np.arange(count_arr.size, dtype=float)
        background_scale = np.ones_like(time_arr)
    else:
        background_scale = np.exp(time_arr / float(MUON_LIFETIME_US))

    def _window_mask(sample_time: np.ndarray, *, tail: bool) -> np.ndarray:
        if sample_time.size <= 1:
            return np.ones(sample_time.size, dtype=bool)

        start = float(np.min(sample_time))
        stop = float(np.max(sample_time))
        span = max(0.0, stop - start)
        width = min(1.0, span * 0.25) if span > 0.0 else 0.0
        if width > 0.0:
            mask = sample_time >= (stop - width) if tail else sample_time <= (start + width)
            if np.count_nonzero(mask) >= min(5, sample_time.size):
                return mask

        window_size = min(sample_time.size, max(1, int(np.ceil(sample_time.size * 0.2))))
        mask = np.zeros(sample_time.size, dtype=bool)
        if tail:
            mask[-window_size:] = True
        else:
            mask[:window_size] = True
        return mask

    late_mask = _window_mask(time_arr, tail=True)
    early_mask = _window_mask(time_arr, tail=False)

    raw_like_counts = count_arr / background_scale
    if time is not None and np.count_nonzero(late_mask) >= 2:
        late_time = np.asarray(time_arr[late_mask], dtype=float)
        late_raw_like = np.asarray(raw_like_counts[late_mask], dtype=float)
        design = np.column_stack(
            [np.exp(-late_time / float(MUON_LIFETIME_US)), np.ones_like(late_time)]
        )
        coeffs, *_ = np.linalg.lstsq(design, late_raw_like, rcond=None)
        background = max(float(coeffs[1]), 0.0)
    else:
        background = float(np.mean(raw_like_counts[late_mask]))
    residual = count_arr - background * background_scale
    if not np.any(np.isfinite(residual)):
        return float(background), 100.0, 0.2

    core_mask = (
        early_mask if np.count_nonzero(early_mask) >= 3 else np.ones_like(residual, dtype=bool)
    )
    core_residual = np.asarray(residual[core_mask], dtype=float)
    core_residual = core_residual[np.isfinite(core_residual)]
    if core_residual.size == 0:
        core_residual = np.asarray(residual[np.isfinite(residual)], dtype=float)

    n0 = max(float(np.median(core_residual)), 1.0)
    centered = core_residual - n0
    if centered.size >= 2:
        lower = float(np.percentile(centered, 10.0))
        upper = float(np.percentile(centered, 90.0))
        amplitude_scale = 0.5 * max(upper - lower, 0.0)
    elif centered.size == 1:
        amplitude_scale = abs(float(centered[0]))
    else:
        amplitude_scale = 0.0

    amplitude = amplitude_scale / n0 if n0 > 0.0 else 0.0
    amplitude = float(np.clip(amplitude, 0.01, 1.0))

    return float(background), n0, amplitude


def _group_phase_window_mhz(
    metadata: dict[str, object] | None,
    freqs: np.ndarray,
) -> tuple[float, float | None]:
    """Return a field-guided FFT phase-estimation window for one grouped trace."""
    frequencies = np.asarray(freqs, dtype=float)
    positive = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
    if positive.size == 0:
        return 0.0, None

    field_value = None if metadata is None else metadata.get("field")
    try:
        field_gauss = abs(float(field_value))
    except (TypeError, ValueError):
        return 0.0, None
    if not np.isfinite(field_gauss) or np.isclose(field_gauss, 0.0):
        return 0.0, None

    center = field_gauss * MUON_GYROMAGNETIC_RATIO_MHZ_PER_T * GAUSS_TO_TESLA
    half_width = 10.0
    lo = max(0.0, center - half_width)
    hi = center + half_width
    if np.any((positive >= lo) & (positive <= hi)):
        return lo, hi
    return 0.0, None


#: Upper bound on the padded FFT length used for group-phase *seeding*.
#: Zero-padding only interpolates the spectrum, so capping it for very large
#: (high-resolution) histograms leaves the peak-phase seed essentially
#: unchanged while keeping the per-selection cost bounded (avoids multi-second
#: hangs when seeding is refreshed on every dataset/selection change).
_MAX_PHASE_SEED_FFT_POINTS = 1 << 17  # 131072


def _bounded_phase_seed_padding(n_points: int, *, desired: int = 8) -> int:
    """Return a padding factor capped so the seed FFT stays bounded in size."""
    if n_points <= 0:
        return 1
    max_factor = max(1, _MAX_PHASE_SEED_FFT_POINTS // int(n_points))
    return max(1, min(int(desired), int(max_factor)))


def _seed_group_phase_estimates(grouped_groups: list[object]) -> tuple[float, dict[str, float]]:
    """Return the first-group absolute phase and per-group relative phases in radians."""
    phase_degrees_by_group: dict[str, float] = {}
    for group in grouped_groups:
        group_id = str(getattr(group, "group_id", ""))
        time = np.asarray(getattr(group, "time", []), dtype=float)
        counts = np.asarray(getattr(group, "counts", []), dtype=float)
        if time.size < 4 or counts.size != time.size:
            phase_degrees_by_group[group_id] = 0.0
            continue

        finite_mask = np.isfinite(time) & np.isfinite(counts)
        time = time[finite_mask]
        counts = counts[finite_mask]
        if time.size < 4:
            phase_degrees_by_group[group_id] = 0.0
            continue

        error = np.asarray(getattr(group, "error", np.ones_like(counts)), dtype=float)
        if error.shape != counts.shape:
            error = np.ones_like(counts, dtype=float)
        else:
            error = error[finite_mask]

        metadata = dict(getattr(group, "metadata", {}) or {})
        background_seed, n0_seed, _amplitude_seed = _seed_group_background_and_n0(
            counts,
            time=time,
        )
        residual = counts - (background_seed * np.exp(time / float(MUON_LIFETIME_US))) - n0_seed
        dataset = MuonDataset(
            time=time.copy(),
            asymmetry=np.asarray(residual, dtype=float),
            error=error.copy(),
            metadata=metadata,
            run=None,
        )
        freqs, spectrum = fft_complex_asymmetry(
            dataset,
            window="none",
            padding_factor=_bounded_phase_seed_padding(time.size),
            subtract_average_signal=True,
        )
        min_frequency, max_frequency = _group_phase_window_mhz(metadata, freqs)
        phase_degrees_by_group[group_id] = estimate_fft_phase(
            freqs,
            spectrum,
            method="peak",
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    if not phase_degrees_by_group:
        return 0.0, {}

    reference_group_id = str(getattr(grouped_groups[0], "group_id", ""))
    reference_phase = phase_degrees_by_group.get(reference_group_id, 0.0)
    reference_phase_rad = float(np.angle(np.exp(1j * np.deg2rad(reference_phase))))
    relative_phases = {
        group_id: float(np.angle(np.exp(1j * np.deg2rad(phase_deg - reference_phase))))
        for group_id, phase_deg in phase_degrees_by_group.items()
    }
    return reference_phase_rad, relative_phases


def _seed_group_relative_phases(grouped_groups: list[object]) -> dict[str, float]:
    """Return per-group relative phase seeds in radians using FFT auto-phase estimation."""
    _reference_phase, relative_phases = _seed_group_phase_estimates(grouped_groups)
    return relative_phases


def _grouped_model_phase_defaults(
    grouped_model: CompositeModel,
    grouped_groups: list[object],
) -> dict[str, float]:
    """Return grouped-model phase defaults seeded from the first group."""
    reference_phase, _relative_phases = _seed_group_phase_estimates(grouped_groups)
    phase_defaults: dict[str, float] = {}
    for pname in grouped_model.param_names:
        base_name, _index = split_parameter_name(pname)
        if base_name == "phase":
            phase_defaults[pname] = reference_phase
    return phase_defaults


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


def _fit_success_html(result) -> str:
    """Return compact success HTML for the result label."""
    npar = len(result.parameters.free_parameters)
    ndof = (
        round(result.chi_squared / result.reduced_chi_squared)
        if result.reduced_chi_squared > 0
        else 0
    )
    stats = f"χ²/ν = {result.reduced_chi_squared:.4f} · npar = {npar} · ndof = {ndof}"
    if result.edm is not None:
        stats += f" · Δ‖p‖ = {result.edm:.2e}"
    return success_html("Fit converged", detail=stats)


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

_PARAM_ROLE_LABELS = {"global": "Global", "local": "Local", "fixed": "Fixed", "file": "File"}


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
    """Return a read-only, bold-mono name item for a parameter table."""
    item = QTableWidgetItem(label)
    item.setData(Qt.ItemDataRole.UserRole, raw_name)
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


def _set_formula_label_text(label: QLabel, formula: str, **_kwargs) -> None:
    """Set formula text; tooltip preserves the raw string for reference."""
    raw_text = str(formula)
    label.setText(raw_text)
    label.setToolTip(raw_text)


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
    label.setStyleSheet("color: #A44A00; font-weight: bold;")
    label.setText("\u26a0 " + label.text())
    label.setToolTip(
        f"This model contains {'/'.join(sorted(foreign))}-domain component(s) "
        f"({', '.join(foreign_names)}) but the representation is fitted in the "
        f"{domain} domain. The model is kept as saved and can still be fitted; "
        "use Edit Function to repair it."
    )


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
    "field-like parameters such as B_L when the run file already stores the relevant value."
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


class GlobalFitWorker(QObject):
    """Worker for running global fits in a background thread.

    Signals
    -------
    finished : Signal(object, object)
        Emitted with (results_dict, fitted_global) when fit completes successfully.
    error : Signal(str)
        Emitted with error message if fit fails.
    """

    # Use object/object for cross-thread payloads containing Python objects
    # (FitResult, ParameterSet, numpy arrays). Typed Qt containers can trigger
    # conversion errors when queued between threads.
    finished = Signal(object, object)  # results_dict, fitted_global
    error = Signal(str)

    def __init__(self, fit_engine, datasets, model_fn, global_params, local_params, initial_params):
        super().__init__()
        self.fit_engine = fit_engine
        self.datasets = datasets
        self.model_fn = model_fn
        self.global_params = global_params
        self.local_params = local_params
        self.initial_params = initial_params

    def run(self):
        """Execute the global fit."""
        try:
            results_dict, fitted_global = self.fit_engine.global_fit(
                self.datasets,
                self.model_fn,
                self.global_params,
                self.local_params,
                self.initial_params,
            )
            self.finished.emit(results_dict, fitted_global)
        except Exception as e:
            self.error.emit(_format_fit_worker_exception(e))


class GroupedTimeDomainFitWorker(QObject):
    """Worker for grouped time-domain fitting in a background thread."""

    finished = Signal(object, object)  # grouped_datasets, fit_result_bundle
    error = Signal(str)

    def __init__(
        self,
        grouped_groups,
        grouped_datasets,
        model_fn,
        global_params,
        local_params,
        initial_params,
    ):
        super().__init__()
        self.grouped_groups = grouped_groups
        self.grouped_datasets = grouped_datasets
        self.model_fn = model_fn
        self.global_params = global_params
        self.local_params = local_params
        self.initial_params = initial_params

    def run(self):
        """Execute the grouped time-domain fit."""
        try:
            result = fit_grouped_time_domain(
                self.grouped_groups,
                self.model_fn,
                self.global_params,
                self.local_params,
                self.initial_params,
            )
            self.finished.emit(self.grouped_datasets, result)
        except Exception as e:
            self.error.emit(_format_fit_worker_exception(e))


class GroupedSeriesFitWorker(QObject):
    """Worker for a multi-run grouped-*series* fit in a background thread.

    Mirrors :class:`GroupedTimeDomainFitWorker` but calls
    :func:`fit_grouped_series` over a ``members`` mapping (run -> groups), so a
    grouped fit can span several runs (batch / global) instead of one.
    """

    finished = Signal(object, object)  # grouped_datasets, GroupedSeriesFitResult
    error = Signal(str)

    def __init__(
        self,
        relationship,
        members,
        grouped_datasets,
        model_fn,
        global_params,
        local_params,
        initial_params,
    ):
        super().__init__()
        self.relationship = relationship
        self.members = members
        self.grouped_datasets = grouped_datasets
        self.model_fn = model_fn
        self.global_params = global_params
        self.local_params = local_params
        self.initial_params = initial_params

    def run(self):
        """Execute the grouped-series fit."""
        try:
            result = fit_grouped_series(
                self.relationship,
                self.members,
                self.model_fn,
                self.global_params,
                self.local_params,
                self.initial_params,
            )
            self.finished.emit(self.grouped_datasets, result)
        except Exception as e:
            self.error.emit(_format_fit_worker_exception(e))


class _ValueUncertaintyDelegate(QStyledItemDelegate):
    """Paints 'value  ±σ' in a table cell; editing shows only the bare value.

    Uncertainty is stored in UserRole+1 alongside the item text. Cleared
    automatically when the user edits the cell.
    """

    _UNC_ROLE = Qt.ItemDataRole.UserRole + 1
    _MUTED = QColor(tokens.TEXT_MUTED)

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        unc = index.data(self._UNC_ROLE)
        if unc is None:
            return
        val_text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        unc_str = f"  ±{float(unc):.4f}"
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
        model.setData(index, None, self._UNC_ROLE)
        super().setModelData(editor, model, index)


class SingleFitTab(QWidget):
    """Single dataset fitting interface.

    Provides model selection, parameter configuration, and fit execution for a
    single dataset. Emits signals when fit completes successfully.

    Attributes
    ----------
    fit_completed : Signal
        Emitted with (FitResult, tuple, list) when fit finishes successfully.
        The tuple contains (t_fit, y_fit) arrays for plotting the fit curve,
        and the list contains per-component additive curves as
        (component_name, y_component).

    Methods
    -------
    set_dataset(dataset)
        Set the current dataset to fit.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    preview_requested = Signal(
        object, object, object
    )  # (FitResult, fitted_curve, component_curves)
    share_function_with_group_requested = Signal(int)
    send_model_to_batch_requested = Signal()
    add_to_series_requested = Signal()
    fit_range_edit_committed = Signal(float, float)  # (x_min, x_max) from spinbox commit

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._current_dataset: MuonDataset | None = None
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._fit_engine = FitEngine()
        self._domain = "time"
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
        self._fit_wizard_window: FitWizardWindow | None = None
        self._cached_wizard_recommendation: FitWizardRecommendation | None = None
        self._cached_wizard_signature: dict[str, object] | None = None
        self._cached_wizard_log_text = ""
        self._updating_fraction_values = False
        self._last_fit_result: FitResult | None = None
        self._last_fit_parameters: ParameterSet | None = None
        self._pull_diagnostic_btn: QPushButton | None = None
        self._pull_diagnostic_window: QWidget | None = None

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        self._share_group_btn = QPushButton("Share Function With Data Group")
        self._share_group_btn.clicked.connect(self._on_share_function_with_group)
        self._share_group_btn.setEnabled(False)
        self._send_to_batch_btn = QPushButton("Send Model to Batch")
        self._send_to_batch_btn.setToolTip(
            "Copy this fit function into the Batch tab to seed a batch fit over the selected runs."
        )
        self._send_to_batch_btn.clicked.connect(self.send_model_to_batch_requested.emit)
        self._add_to_series_btn = QPushButton("Add to Series...")
        self._add_to_series_btn.setToolTip(
            "Add this run's single fit to an existing batch series with a matching model."
        )
        self._add_to_series_btn.clicked.connect(self.add_to_series_requested.emit)

        model_button_layout = QGridLayout()
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setHorizontalSpacing(6)
        model_button_layout.setVerticalSpacing(6)
        self._edit_model_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fit_wizard_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._share_group_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._send_to_batch_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._add_to_series_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        model_button_layout.addWidget(self._edit_model_btn, 0, 0)
        model_button_layout.addWidget(self._fit_wizard_btn, 0, 1)
        model_button_layout.addWidget(self._share_group_btn, 1, 0, 1, 2)
        model_button_layout.addWidget(self._send_to_batch_btn, 2, 0)
        model_button_layout.addWidget(self._add_to_series_btn, 2, 1)
        model_button_layout.setColumnStretch(0, 1)
        model_button_layout.setColumnStretch(1, 1)

        self._formula_row_label = QLabel("A(t):")
        model_layout.addRow(self._formula_row_label, self._formula_label)
        model_layout.addRow("", model_button_layout)
        layout.addWidget(model_group)

        # Fit range section
        fit_range_group = QGroupBox("Fit range")
        fit_range_layout = QHBoxLayout(fit_range_group)
        fit_range_layout.setContentsMargins(6, 4, 6, 4)
        fit_range_layout.setSpacing(4)

        self._fit_range_min_spin = QDoubleSpinBox()
        self._fit_range_min_spin.setDecimals(3)
        self._fit_range_min_spin.setRange(-1000.0, 1000.0)
        self._fit_range_min_spin.setSingleStep(0.1)
        self._fit_range_min_spin.setMinimumWidth(90)
        self._fit_range_min_spin.setFont(mono_font(11.0))

        self._fit_range_mid_label = QLabel("≤ <i>t</i> ≤")
        self._fit_range_mid_label.setTextFormat(Qt.TextFormat.RichText)

        self._fit_range_max_spin = QDoubleSpinBox()
        self._fit_range_max_spin.setDecimals(3)
        self._fit_range_max_spin.setRange(-1000.0, 1000.0)
        self._fit_range_max_spin.setSingleStep(0.1)
        self._fit_range_max_spin.setMinimumWidth(90)
        self._fit_range_max_spin.setFont(mono_font(11.0))

        self._fit_range_unit_label = QLabel("μs")

        fit_range_layout.addWidget(self._fit_range_min_spin)
        fit_range_layout.addWidget(self._fit_range_mid_label)
        fit_range_layout.addWidget(self._fit_range_max_spin)
        fit_range_layout.addWidget(self._fit_range_unit_label)
        fit_range_layout.addStretch()
        layout.addWidget(fit_range_group)

        self._fit_range_min_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)
        self._fit_range_max_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)

        # Parameter table
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._param_table = QTableWidget(0, 7)
        self._param_table.setHorizontalHeaderLabels(
            ["Name", "Value", "Fix", "Min", "Max", "Batch", "Link"]
        )
        self._param_table.horizontalHeader().setStretchLastSection(False)
        self._param_table.setColumnWidth(0, 80)  # Name
        self._param_table.setColumnWidth(1, 100)  # Value
        self._param_table.setColumnWidth(2, 40)  # Fix
        self._param_table.setColumnWidth(3, 80)  # Min
        self._param_table.setColumnWidth(4, 80)  # Max
        self._param_table.setColumnWidth(5, 70)  # Batch role (read-only)
        self._param_table.setColumnWidth(6, 60)  # Link group (equality tie)

        _apply_param_table_style(self._param_table)
        self._param_table.setItemDelegateForColumn(1, _ValueUncertaintyDelegate(self._param_table))

        # Let the table grow with the dock and scroll when it can't show every
        # row. A many-parameter model (e.g. the 13-param CdS three-line fit) must
        # keep all rows reachable; without this the table collapses to a handful
        # of rows with no scrollbar and the lower parameters become unreachable.
        self._param_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._param_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._param_table.setMinimumHeight(160)

        self._param_table.itemChanged.connect(self._on_param_table_item_changed)
        param_layout.addWidget(self._param_table)
        # Stretch factor 1 lets the Parameters group claim the dock's free
        # vertical space ahead of the fixed-height Results group below it.
        layout.addWidget(param_group, 1)

        # Buttons
        btn_layout = QGridLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setHorizontalSpacing(6)
        btn_layout.setVerticalSpacing(6)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.clicked.connect(self._run_fit)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._reset_parameters)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_btn.setEnabled(False)
        self._pull_diagnostic_btn = QPushButton("Pull diagnostic…")
        self._pull_diagnostic_btn.setToolTip(
            "Re-simulate this fit at matched statistics, refit each copy, and "
            "check that the parameter pulls are standard normal (honest errors)."
        )
        self._pull_diagnostic_btn.clicked.connect(self._on_pull_diagnostic)
        self._pull_diagnostic_btn.setEnabled(False)
        btn_layout.addWidget(self._fit_btn, 0, 0)
        btn_layout.addWidget(self._reset_btn, 0, 1)
        btn_layout.addWidget(self._preview_btn, 0, 2)
        btn_layout.addWidget(self._pull_diagnostic_btn, 1, 0, 1, 3)
        btn_layout.setColumnStretch(3, 1)
        layout.addLayout(btn_layout)

        # Results
        self._results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(self._results_group)
        self._result_label = QLabel("No fit performed yet")
        self._result_label.setWordWrap(True)
        results_layout.addWidget(self._result_label)
        layout.addWidget(self._results_group)

        self._set_composite_model(self._composite_model)

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_domain(self, domain: str) -> None:
        """Switch labels and default model for time or frequency fitting."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            return
        self._domain = normalized
        if self._domain == "frequency":
            self._fit_wizard_btn.setEnabled(False)
            self._fit_wizard_btn.setToolTip(
                "Fit Wizard is currently available for time-domain fits."
            )
            self._share_group_btn.setEnabled(False)
            self._formula_row_label.setText("S(ν):")
            self._fit_range_mid_label.setText("≤ <i>ν</i> ≤")
            self._fit_range_unit_label.setText("MHz")
            self._fit_range_min_spin.setDecimals(4)
            self._fit_range_max_spin.setDecimals(4)
            self._fit_range_min_spin.setRange(-1_000_000.0, 1_000_000.0)
            self._fit_range_max_spin.setRange(-1_000_000.0, 1_000_000.0)
            self._set_composite_model(default_frequency_model())
        else:
            self._fit_wizard_btn.setToolTip("")
            self._formula_row_label.setText("A(t):")
            self._fit_range_mid_label.setText("≤ <i>t</i> ≤")
            self._fit_range_unit_label.setText("μs")
            self._fit_range_min_spin.setDecimals(3)
            self._fit_range_max_spin.setDecimals(3)
            self._fit_range_min_spin.setRange(-1000.0, 1000.0)
            self._fit_range_max_spin.setRange(-1000.0, 1000.0)
            self._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
        self.set_dataset(self._current_dataset)

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset to fit."""
        self._current_dataset = dataset
        # A fit result belongs to the dataset it was fit on; drop it on change.
        self._last_fit_result = None
        self._last_fit_parameters = None
        if self._pull_diagnostic_btn is not None:
            self._pull_diagnostic_btn.setEnabled(False)
        enabled = dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled and self._domain == "time")
        self._share_group_btn.setEnabled(dataset is not None and self._domain == "time")

    def _can_run_pull_diagnostic(self) -> bool:
        """A successful time-domain fit on a run with histograms is required."""
        return (
            self._domain == "time"
            and self._last_fit_result is not None
            and self._last_fit_result.success
            and self._last_fit_parameters is not None
            and self._current_dataset is not None
            and self._current_dataset.run is not None
            and bool(self._current_dataset.run.histograms)
        )

    def _on_pull_diagnostic(self) -> None:
        """Open the pull-distribution diagnostic for the last converged fit."""
        if not self._can_run_pull_diagnostic():
            return
        from asymmetry.core.simulate import matched_statistics
        from asymmetry.gui.windows.pull_diagnostic_window import (
            PullDiagnosticWindow,
            make_engine_refit,
        )

        dataset = self._current_dataset
        run = dataset.run
        # The generating "truth" is the CONVERGED fit, not the pre-fit guesses:
        # FitEngine.fit does not mutate its input ParameterSet, so
        # _last_fit_parameters still holds the start values. Seed the refit
        # template from result.parameters while keeping the fit's
        # bounds/fixed/link metadata.
        fitted_values = {p.name: float(p.value) for p in self._last_fit_result.parameters}
        refit_template = copy.deepcopy(self._last_fit_parameters)
        for parameter in refit_template:
            if parameter.name in fitted_values:
                parameter.value = fitted_values[parameter.name]
        truth = {p.name: float(p.value) for p in refit_template}
        free = [p.name for p in refit_template if not getattr(p, "fixed", False)]
        time_range = (float(dataset.time.min()), float(dataset.time.max()))
        refit = make_engine_refit(
            self._composite_model, refit_template, t_min=time_range[0], t_max=time_range[1]
        )
        # Matched statistics: split the run's flat background off the gross
        # count so background is regenerated as background, not as extra signal.
        signal_events, background_per_bin = matched_statistics(run)
        window = PullDiagnosticWindow(
            template=run,
            model=self._composite_model,
            parameters=truth,
            refit=refit,
            track=free or list(truth),
            total_events=signal_events,
            background_per_bin=background_per_bin,
            time_range=time_range,
            parent=self,
        )
        self._pull_diagnostic_window = window
        window.show()

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Enable/disable single-fit actions while preserving the active dataset."""
        self._fit_blocked = bool(blocked)
        self._fit_block_reason = str(reason)
        enabled = self._current_dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled and self._domain == "time")
        tooltip = self._fit_block_reason if self._fit_blocked else ""
        self._fit_btn.setToolTip(tooltip)
        self._preview_btn.setToolTip(tooltip)
        self._fit_wizard_btn.setToolTip(tooltip)

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update fit-range spinboxes from the plot without re-emitting."""
        have_range = x_min is not None and x_max is not None
        self._fit_range_min_spin.setEnabled(have_range)
        self._fit_range_max_spin.setEnabled(have_range)
        if not have_range:
            return
        with QSignalBlocker(self._fit_range_min_spin):
            self._fit_range_min_spin.setValue(float(x_min))
        with QSignalBlocker(self._fit_range_max_spin):
            self._fit_range_max_spin.setValue(float(x_max))

    def _on_fit_range_spinbox_committed(self) -> None:
        """Emit fit_range_edit_committed when the user finishes editing a spinbox."""
        self.fit_range_edit_committed.emit(
            self._fit_range_min_spin.value(),
            self._fit_range_max_spin.value(),
        )

    def _wizard_context_signature(self) -> dict[str, object]:
        return {
            "run_number": (
                int(self._current_dataset.run_number)
                if self._current_dataset is not None
                and getattr(self._current_dataset, "run_number", None) is not None
                else None
            ),
            "model": self._composite_model.to_dict(),
        }

    def _wizard_base_signature_matches(
        self,
        cached_signature: dict[str, object] | None,
        current_signature: dict[str, object],
    ) -> bool:
        if not isinstance(cached_signature, dict):
            return False
        cached_model = cached_signature.get("model")
        return cached_signature.get("run_number") == current_signature.get("run_number") and (
            cached_model is None or cached_model == current_signature.get("model")
        )

    def _cache_wizard_analysis(
        self,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object],
        log_text: str = "",
    ) -> None:
        self._cached_wizard_recommendation = recommendation
        self._cached_wizard_signature = copy.deepcopy(signature)
        self._cached_wizard_log_text = str(log_text)

    def _on_fit_wizard_analysis_cached(
        self,
        recommendation: object,
        log_text: str,
        signature: object,
    ) -> None:
        if not isinstance(recommendation, FitWizardRecommendation) or not isinstance(
            signature, dict
        ):
            return
        self._cache_wizard_analysis(recommendation, signature=signature, log_text=log_text)

    def _on_share_function_with_group(self) -> None:
        """Request sharing the active single-fit function with the current data group."""
        if self._current_dataset is None:
            return
        try:
            run_number = int(self._current_dataset.run_number)
        except (TypeError, ValueError):
            return
        self.share_function_with_group_requested.emit(run_number)

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild the parameter table."""
        self._updating_fraction_values = True
        self._composite_model = model
        _set_formula_label_text(self._formula_label, model.formula_string())
        _apply_domain_mismatch_warning(self._formula_label, model, self._domain)

        dataset_field = (
            self._current_dataset.run.field
            if self._current_dataset is not None and self._current_dataset.run is not None
            else 0.0
        )
        field_overrides = _field_value_overrides(model, dataset_field)
        frequency_overrides = (
            seed_peak_parameters_from_dataset(self._current_dataset, model)
            if self._domain == "frequency" and self._current_dataset is not None
            else {}
        )

        fixed_default_params = model.fixed_by_default_params()
        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            # Name column (read-only, bold mono)
            name_item = _make_param_name_item(_format_param_label(pname), pname)
            self._param_table.setItem(i, 0, name_item)

            # Value column — use dataset field for 'field' parameters if available
            default_val = frequency_overrides.get(
                pname, field_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            )
            value_item = QTableWidgetItem(str(default_val))
            value_item.setFont(mono_font(11.0))
            self._param_table.setItem(i, 1, value_item)

            # Fix checkbox column
            fix_widget = QWidget()
            fix_layout = QHBoxLayout(fix_widget)
            fix_layout.setContentsMargins(0, 0, 0, 0)
            fix_checkbox = QCheckBox()
            if pname == "shape_factor_a" or pname in fixed_default_params:
                fix_checkbox.setChecked(True)
            fix_layout.addWidget(fix_checkbox)
            fix_layout.setAlignment(fix_checkbox, Qt.AlignmentFlag.AlignCenter)
            self._param_table.setCellWidget(i, 2, fix_widget)

            # Min column — default to 0 for physically positive-definite parameters
            default_min = get_param_info(pname).default_min
            min_text = str(default_min) if default_min is not None else "-inf"
            min_item = QTableWidgetItem(min_text)
            min_item.setFont(mono_font(11.0))
            self._param_table.setItem(i, 3, min_item)

            # Max column
            max_item = QTableWidgetItem("inf")
            max_item.setFont(mono_font(11.0))
            self._param_table.setItem(i, 4, max_item)

            # Batch role column — read-only; filled when a batch result is piped back.
            _set_param_batch_role_cell(self._param_table, i, None)

            # Link column — equality link-group selector (WiMDA "Ties").
            link_combo = _make_link_group_combo()
            self._param_table.setCellWidget(i, _SINGLE_PARAM_LINK_COLUMN, link_combo)
            # Fix and Link are mutually exclusive: a linked follower tracks its
            # group main (linking wins over fix in the engine), so allowing both
            # would silently discard the fixed value. Couple the two controls so
            # the ambiguous combination can't be created.
            self._wire_fix_link_exclusion(fix_checkbox, link_combo)

        _configure_fraction_rows_in_table(
            self._param_table,
            model,
            min_column=3,
            max_column=4,
        )
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

    def _synchronize_fraction_value_rows(self, edited_param_name: str | None = None) -> None:
        self._updating_fraction_values = True
        try:
            _synchronize_fraction_group_values_in_table(
                self._param_table,
                self._composite_model,
                edited_param_name=edited_param_name,
            )
        finally:
            self._updating_fraction_values = False

    def _on_param_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_fraction_values or item.column() != 1:
            return
        name_item = self._param_table.item(item.row(), 0)
        param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
        if isinstance(param_name, str):
            self._synchronize_fraction_value_rows(param_name)

    def _wire_fix_link_exclusion(self, fix_checkbox: QCheckBox, link_combo: QComboBox) -> None:
        """Keep a row's Fix checkbox and Link-group combo mutually exclusive.

        Selecting a link group disables (and clears) Fix; ticking Fix disables
        (and clears) the link group. Programmatic table updates set
        ``_updating_fraction_values`` and are ignored, so restoring saved state
        never fights itself.
        """

        def on_fix_toggled(checked: bool) -> None:
            if self._updating_fraction_values:
                return
            if checked and _link_group_combo_value(link_combo) is not None:
                self._updating_fraction_values = True
                try:
                    _set_link_group_combo_value(link_combo, None)
                finally:
                    self._updating_fraction_values = False
            link_combo.setEnabled(not checked)

        def on_link_changed(_index: int) -> None:
            if self._updating_fraction_values:
                return
            linked = _link_group_combo_value(link_combo) is not None
            if linked and fix_checkbox.isChecked():
                self._updating_fraction_values = True
                try:
                    fix_checkbox.setChecked(False)
                finally:
                    self._updating_fraction_values = False
            fix_checkbox.setEnabled(not linked)

        fix_checkbox.toggled.connect(on_fix_toggled)
        link_combo.currentIndexChanged.connect(on_link_changed)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(
            self, initial_model=self._composite_model, domain=self._domain
        )
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _open_fit_wizard(self) -> None:
        """Launch or refresh the non-modal fit wizard window."""
        if self._current_dataset is None:
            QMessageBox.information(
                self, "Fit Wizard", "Select a dataset before opening the fit wizard."
            )
            return
        if self._domain == "frequency":
            QMessageBox.information(
                self,
                "Fit Wizard",
                "Fit Wizard is currently available for time-domain fits.",
            )
            return
        if self._fit_blocked:
            message = (
                self._fit_block_reason or "Fit actions are unavailable for the current selection."
            )
            QMessageBox.information(self, "Fit Wizard", message)
            return

        if self._fit_wizard_window is None:
            self._fit_wizard_window = FitWizardWindow(self)
            self._fit_wizard_window.apply_assessment_requested.connect(
                self._apply_fit_wizard_assessment
            )
            self._fit_wizard_window.analysis_cached.connect(self._on_fit_wizard_analysis_cached)

        signature = self._wizard_context_signature()

        self._fit_wizard_window.set_analysis_context(
            self._current_dataset,
            current_model=self._composite_model,
        )
        if self._cached_wizard_recommendation is not None and self._wizard_base_signature_matches(
            self._cached_wizard_signature, signature
        ):
            self._fit_wizard_window.set_cached_recommendation(
                self._cached_wizard_recommendation,
                signature=self._cached_wizard_signature,
                log_text=self._cached_wizard_log_text,
            )
        self._fit_wizard_window.show()
        self._fit_wizard_window.raise_()
        self._fit_wizard_window.activateWindow()

    def _reset_parameters(self) -> None:
        """Reset parameters to model defaults."""
        self._set_composite_model(self._composite_model)

    def _apply_fit_wizard_assessment(
        self,
        assessment: CandidateAssessment,
        recommendation: FitWizardRecommendation,
    ) -> None:
        """Apply a fit-wizard assessment back into the single-fit tab."""
        if self._current_dataset is None:
            return
        if not isinstance(assessment, CandidateAssessment):
            return

        result = assessment.fit_result
        if not result.success:
            self._result_label.setText(f"<b>Fit Wizard failed:</b> {result.message}")
            return

        self._set_composite_model(assessment.template.model)
        fitted_by_name = {parameter.name: parameter for parameter in result.parameters}
        display_values = _normalized_model_param_values(
            self._composite_model,
            {parameter.name: parameter.value for parameter in result.parameters},
        )
        self._updating_fraction_values = True

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                continue
            fitted = fitted_by_name.get(param_name)
            if fitted is None:
                continue

            value_item = self._param_table.item(row, 1)
            if value_item is not None:
                value_item.setText(f"{display_values.get(param_name, fitted.value):.6f}")
                unc = result.uncertainties.get(param_name, None)
                value_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)

            min_item = self._param_table.item(row, 3)
            if min_item is not None:
                min_item.setText("-inf" if not np.isfinite(fitted.min) else f"{fitted.min:g}")

            max_item = self._param_table.item(row, 4)
            if max_item is not None:
                max_item.setText("inf" if not np.isfinite(fitted.max) else f"{fitted.max:g}")

            fix_widget = self._param_table.cellWidget(row, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            if fix_checkbox is not None:
                fix_checkbox.setChecked(bool(fitted.fixed))
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        wizard_note = f"Fit Wizard — {assessment.template.title}"
        if assessment.residual_gate_reasons:
            wizard_note += " ⚠"
        self._results_group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)
        detail = _fit_success_html(result).split("<br>", 1)[1]
        self._result_label.setText(success_html(wizard_note, detail=detail))

        param_dict = {parameter.name: parameter.value for parameter in result.parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)
        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )
        self.fit_completed.emit(result, (t_fit, y_fit), component_curves)

    def _on_preview(self) -> None:
        """Generate and emit a preview fit curve with current parameters."""
        if self._fit_blocked:
            return

        if self._current_dataset is None:
            return

        if self._composite_model is None:
            return

        # Build parameter set from table
        parameters = ParameterSet()
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            # Parse value
            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                return

            # Check if fixed
            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox)
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            # Parse bounds
            try:
                min_text = self._param_table.item(i, 3).text()
                min_val = float(min_text) if min_text and min_text != "-inf" else -float("inf")
            except (ValueError, AttributeError):
                min_val = -float("inf")

            try:
                max_text = self._param_table.item(i, 4).text()
                max_val = float(max_text) if max_text and max_text != "inf" else float("inf")
            except (ValueError, AttributeError):
                max_val = float("inf")

            param = Parameter(
                name=param_name,
                value=value,
                min=min_val,
                max=max_val,
                fixed=fixed,
            )
            parameters.add(param)

        param_dict = {p.name: p.value for p in parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        # Generate fitted curve for plotting
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)

        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )

        # Create a dummy result object for preview (not a real fit)
        preview_result = object()
        self.preview_requested.emit(preview_result, (t_fit, y_fit), component_curves)

    def _parameter_set_from_table(self) -> ParameterSet:
        """Build a :class:`ParameterSet` from the parameter table.

        Raises :class:`ValueError` with a user-facing message on a malformed
        value (the only hard error; bad bounds fall back to ±inf). Shared by
        the fit run and the pull-distribution diagnostic.
        """
        parameters = ParameterSet()
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid value for {_format_param_label(param_name)}") from exc

            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox)
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            try:
                min_text = self._param_table.item(i, 3).text()
                min_val = float(min_text) if min_text and min_text != "-inf" else -float("inf")
            except (ValueError, AttributeError):
                min_val = -float("inf")

            try:
                max_text = self._param_table.item(i, 4).text()
                max_val = float(max_text) if max_text and max_text != "inf" else float("inf")
            except (ValueError, AttributeError):
                max_val = float("inf")

            link_combo = self._param_table.cellWidget(i, _SINGLE_PARAM_LINK_COLUMN)
            link_group = _link_group_combo_value(link_combo)

            parameters.add(
                Parameter(
                    name=param_name,
                    value=value,
                    min=min_val,
                    max=max_val,
                    fixed=fixed,
                    link_group=link_group,
                )
            )
        return parameters

    def _run_fit(self) -> None:
        """Execute the fit."""
        if self._fit_blocked:
            message = self._fit_block_reason or "Fit is unavailable for the current selection."
            self._result_label.setText(f"ERROR: {message}")
            return

        if self._current_dataset is None:
            self._result_label.setText("ERROR: No dataset selected")
            return

        if self._composite_model is None:
            self._result_label.setText("ERROR: No function defined")
            return

        # Build parameter set from table
        try:
            parameters = self._parameter_set_from_table()
        except ValueError as exc:
            self._result_label.setText(f"ERROR: {exc}")
            return

        # Run the fit
        self._results_group.setStyleSheet("")
        self._result_label.setText("Fitting...")
        try:
            result = self._fit_engine.fit(
                self._current_dataset,
                self._composite_model.function,
                parameters,
            )
        except Exception as e:
            self._result_label.setText(f"<b>Error during fit:</b><br>{str(e)}")
            return

        # Update results display
        if result.success:
            # Remember the converged fit so the pull-distribution diagnostic can
            # re-simulate and refit it (model, generating values and run).
            self._last_fit_result = result
            self._last_fit_parameters = parameters
            if self._pull_diagnostic_btn is not None:
                self._pull_diagnostic_btn.setEnabled(self._can_run_pull_diagnostic())
            display_values = _normalized_model_param_values(
                self._composite_model,
                {parameter.name: parameter.value for parameter in result.parameters},
            )
            self._results_group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)
            self._result_label.setText(_fit_success_html(result))

            # Update table with fit results
            self._updating_fraction_values = True
            for i in range(self._param_table.rowCount()):
                name_item = self._param_table.item(i, 0)
                param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(param_name, str):
                    param_name = name_item.text() if name_item else ""
                if param_name in result.parameters:
                    fitted_value = display_values.get(
                        param_name, result.parameters[param_name].value
                    )
                    val_item = self._param_table.item(i, 1)
                    val_item.setText(f"{fitted_value:.6f}")
                    unc = result.uncertainties.get(param_name, None)
                    val_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)
                    # A fresh single fit supersedes any piped-back batch role.
                    _set_param_batch_role_cell(self._param_table, i, None)
            self._updating_fraction_values = False
            self._synchronize_fraction_value_rows()

            param_dict = {p.name: p.value for p in result.parameters}
            n_samples = _fit_curve_sample_count(
                self._composite_model,
                param_dict,
                float(self._current_dataset.time.min()),
                float(self._current_dataset.time.max()),
            )

            # Generate fitted curve for plotting
            t_fit = np.linspace(
                self._current_dataset.time.min(),
                self._current_dataset.time.max(),
                n_samples,
            )
            y_fit = self._composite_model.function(t_fit, **param_dict)

            component_curves = self._composite_model.evaluate_components(
                t_fit,
                additive_only=True,
                **param_dict,
            )
            self.fit_completed.emit(result, (t_fit, y_fit), component_curves)
        else:
            self._results_group.setStyleSheet("")
            self._result_label.setText(f"<b>Fit failed:</b> {result.message}")

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the single-fit tab state."""
        if self._fit_wizard_window is not None:
            recommendation = self._fit_wizard_window.current_recommendation()
            if recommendation is not None:
                signature = self._cached_wizard_signature
                if not isinstance(signature, dict):
                    signature = self._wizard_context_signature()
                self._cache_wizard_analysis(
                    recommendation,
                    signature=signature,
                    log_text=self._fit_wizard_window.current_log_text(),
                )

        params = []
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else f"param_{i}"
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            value_item = self._param_table.item(i, 1)
            try:
                value = float(value_item.text()) if value_item else 0.0
            except ValueError:
                value = 0.0

            unc = (
                value_item.data(_ValueUncertaintyDelegate._UNC_ROLE)
                if value_item is not None
                else None
            )

            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            min_item = self._param_table.item(i, 3)
            max_item = self._param_table.item(i, 4)
            role_item = self._param_table.item(i, _SINGLE_PARAM_BATCH_COLUMN)
            role = role_item.data(_PARAM_BATCH_ROLE_DATA) if role_item is not None else None
            link_combo = self._param_table.cellWidget(i, _SINGLE_PARAM_LINK_COLUMN)
            params.append(
                {
                    "name": param_name,
                    "value": value,
                    "fixed": fixed,
                    "min": min_item.text() if min_item else "-inf",
                    "max": max_item.text() if max_item else "inf",
                    "uncertainty": unc,
                    "role": role if isinstance(role, str) else None,
                    "link_group": _link_group_combo_value(link_combo),
                }
            )

        normalized_values = _normalized_model_param_values(
            self._composite_model,
            {str(entry["name"]): float(entry.get("value", 0.0)) for entry in params},
        )

        state = {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            "parameters": [
                {**entry, "value": normalized_values.get(str(entry["name"]), entry["value"])}
                for entry in params
            ],
            "result_html": self._result_label.text(),
        }
        if (
            self._cached_wizard_recommendation is not None
            and self._cached_wizard_signature is not None
        ):
            state["wizard_state"] = {
                "signature": copy.deepcopy(self._cached_wizard_signature),
                "recommendation": serialize_fit_wizard_recommendation(
                    self._cached_wizard_recommendation
                ),
                "log_text": self._cached_wizard_log_text,
            }
        return state

    def restore_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        self._cached_wizard_recommendation = None
        self._cached_wizard_signature = None
        self._cached_wizard_log_text = ""

        composite_data = state.get("composite_model")
        if isinstance(composite_data, dict):
            try:
                self._set_composite_model(CompositeModel.from_dict(composite_data))
            except ValueError:
                self._set_composite_model(
                    CompositeModel(["Exponential", "Constant"], operators=["+"])
                )

        params_data = {p["name"]: p for p in state.get("parameters", [])}
        normalized_state_values = _normalized_model_param_values(
            self._composite_model,
            {
                str(name): float(entry.get("value", 0.0))
                for name, entry in params_data.items()
                if entry.get("value") is not None
            },
        )
        self._updating_fraction_values = True
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()
            if param_name not in params_data:
                continue

            p_data = params_data[param_name]

            value_item = self._param_table.item(i, 1)
            if value_item:
                value_item.setText(
                    str(normalized_state_values.get(param_name, p_data.get("value", 0.0)))
                )
                unc = p_data.get("uncertainty")
                value_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)

            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            if fix_checkbox:
                fix_checkbox.setChecked(bool(p_data.get("fixed", False)))

            min_item = self._param_table.item(i, 3)
            if min_item:
                min_item.setText(str(p_data.get("min", "-inf")))

            max_item = self._param_table.item(i, 4)
            if max_item:
                max_item.setText(str(p_data.get("max", "inf")))

            _set_param_batch_role_cell(self._param_table, i, p_data.get("role"))

            link_combo = self._param_table.cellWidget(i, _SINGLE_PARAM_LINK_COLUMN)
            raw_link = p_data.get("link_group")
            _set_link_group_combo_value(
                link_combo, int(raw_link) if isinstance(raw_link, (int, float)) else None
            )
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        result_html = state.get("result_html")
        if isinstance(result_html, str) and result_html:
            self._result_label.setText(result_html)

        wizard_state = state.get("wizard_state")
        if isinstance(wizard_state, dict):
            recommendation = deserialize_fit_wizard_recommendation(
                wizard_state.get("recommendation")
            )
            signature = wizard_state.get("signature")
            if recommendation is not None and isinstance(signature, dict):
                self._cached_wizard_recommendation = recommendation
                self._cached_wizard_signature = copy.deepcopy(signature)
                self._cached_wizard_log_text = str(wizard_state.get("log_text", ""))


class GlobalFitTab(QWidget):
    """Global fitting interface for simultaneous multi-dataset fitting.

    Allows user to specify which parameters are global (shared), local (vary per dataset),
    or fixed across all datasets in the workspace.

    Signals
    -------
    global_fit_completed : Signal(dict, ParameterSet)
        Emitted with (results_dict, global_params) when global fit completes.
        results_dict maps run_number -> (FitResult, fitted_curve_tuple).
    """

    # Use object/object to avoid Qt container coercion (which can alter key types).
    global_fit_started = Signal()  # emitted just before the worker launches
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)
    grouped_fit_completed = Signal(object, object)  # (grouped_datasets, results_dict)
    grouped_preview_requested = Signal(object, object)  # (grouped_datasets, preview_curves)
    fit_range_edit_committed = Signal(float, float)  # (x_min, x_max) from spinbox commit

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        member_kind: str = "runs",
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        # Member kind is fixed per instance and follows the active representation:
        # the groups surface (Individual-groups representation) is group-membered,
        # every other surface is run-membered. (Phase 3: scope is derived, not selected.)
        self._member_kind = member_kind if member_kind in ("runs", "groups") else "runs"

        self._fit_engine = FitEngine()
        self._domain = "time"
        self._datasets = []  # Will be set by parent
        self._current_dataset: MuonDataset | None = None
        # Last grouped fit's per-group simulate seed, keyed by source run number
        # (shared normalised model + base values + per-group amplitude/phase),
        # for the multi-group Generate Synthetic Run dialog.
        self._grouped_simulate_seed: dict[int, dict] = {}
        # Member runs for a grouped *series* fit (empty → fall back to the active
        # dataset, i.e. the single-run grouped "single fit").
        self._member_datasets: list[MuonDataset] = []
        # Populated by the grouped-context builder: run_number -> list[groups].
        self._grouped_members: dict[int, list[object]] = {}
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._composite_model = self._default_composite_model()
        self._applied_field_default_gauss = 0.0
        self._applied_group_phase_default_rad = 0.0
        # Successful single-fit seeds keyed by run number.
        self._single_fit_seed_by_run: dict[int, dict[str, object]] = {}
        # Inherited seed cache for current dataset selection.
        self._inherited_seed_by_run: dict[int, dict[str, float]] = {}
        self._inherited_model_dict: dict[str, object] | None = None
        # Per-run initial values set explicitly via the Initial-values dialog
        # (highest precedence over inherited single-fit seeds).
        self._user_initial_values_by_run: dict[int, dict[str, float]] = {}
        # Per-(run, group) nuisance initial values for grouped fits, keyed by the
        # synthetic group-member key.
        self._user_grouped_initial_values: dict[int, dict[str, float]] = {}
        self._fit_wizard_window: GlobalFitWizardWindow | None = None
        self._wizard_cache_by_run_set: dict[tuple[int, ...], dict[str, object]] = {}
        self._cached_wizard_recommendation: GlobalFitWizardRecommendation | None = None
        self._cached_wizard_signature: dict[str, object] | None = None
        self._cached_wizard_log_text = ""
        self._updating_fraction_values = False
        self._updating_group_model_fraction_values = False
        self._updating_group_param_values = False
        self._group_param_group_specs: list[tuple[object, str]] = []

        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)
        self._formula_label = QLabel()
        _configure_formula_label(self._formula_label)
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Global Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        model_button_layout = QGridLayout()
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setHorizontalSpacing(6)
        model_button_layout.setVerticalSpacing(6)
        self._edit_model_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fit_wizard_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        model_button_layout.addWidget(self._edit_model_btn, 0, 0)
        model_button_layout.addWidget(self._fit_wizard_btn, 0, 1)
        model_button_layout.setColumnStretch(0, 1)
        model_button_layout.setColumnStretch(1, 1)
        self._formula_row_label = QLabel("A(t):")
        model_layout.addRow(self._formula_row_label, self._formula_label)
        model_layout.addRow("", model_button_layout)
        layout.addWidget(model_group)

        # Fit range section
        _fr_group = QGroupBox("Fit range")
        _fr_layout = QHBoxLayout(_fr_group)
        _fr_layout.setContentsMargins(6, 4, 6, 4)
        _fr_layout.setSpacing(4)

        self._fit_range_min_spin = QDoubleSpinBox()
        self._fit_range_min_spin.setDecimals(3)
        self._fit_range_min_spin.setRange(-1000.0, 1000.0)
        self._fit_range_min_spin.setSingleStep(0.1)
        self._fit_range_min_spin.setMinimumWidth(90)
        self._fit_range_min_spin.setFont(mono_font(11.0))

        self._fit_range_mid_label = QLabel("≤ <i>t</i> ≤")
        self._fit_range_mid_label.setTextFormat(Qt.TextFormat.RichText)

        self._fit_range_max_spin = QDoubleSpinBox()
        self._fit_range_max_spin.setDecimals(3)
        self._fit_range_max_spin.setRange(-1000.0, 1000.0)
        self._fit_range_max_spin.setSingleStep(0.1)
        self._fit_range_max_spin.setMinimumWidth(90)
        self._fit_range_max_spin.setFont(mono_font(11.0))

        _fr_layout.addWidget(self._fit_range_min_spin)
        _fr_layout.addWidget(self._fit_range_mid_label)
        _fr_layout.addWidget(self._fit_range_max_spin)
        self._fit_range_unit_label = QLabel("μs")
        _fr_layout.addWidget(self._fit_range_unit_label)
        _fr_layout.addStretch()
        layout.addWidget(_fr_group)

        self._fit_range_min_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)
        self._fit_range_max_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)

        # Parameter classification table
        self._param_group = QGroupBox("Parameter Classification")
        param_layout = QVBoxLayout(self._param_group)

        param_header_layout = QHBoxLayout()
        param_header_layout.addStretch()
        self._param_help_btn = QPushButton("?")
        self._param_help_btn.setFixedWidth(28)
        self._param_help_btn.setToolTip("Explain Global, Local, Fixed, and File parameter roles")
        self._param_help_btn.clicked.connect(self._show_parameter_classification_help)
        param_header_layout.addWidget(self._param_help_btn)
        param_layout.addLayout(param_header_layout)

        self._param_table = QTableWidget(0, 4)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Bounds"])
        self._param_table.horizontalHeader().setStretchLastSection(False)
        self._param_table.setColumnWidth(0, 80)  # Parameter name
        self._param_table.setColumnWidth(1, 80)  # Initial value
        self._param_table.setColumnWidth(2, 100)  # Type (dropdown)
        self._param_table.setColumnWidth(3, 150)  # Bounds
        _apply_param_table_style(self._param_table)
        self._param_table.itemChanged.connect(self._on_param_table_item_changed)
        param_layout.addWidget(self._param_table)
        layout.addWidget(self._param_group)

        self._grouped_context_label = QLabel()
        self._grouped_context_label.setWordWrap(True)
        self._grouped_context_label.hide()
        layout.addWidget(self._grouped_context_label)

        self._group_param_group = QGroupBox("Per-Group Parameters")
        group_param_layout = QVBoxLayout(self._group_param_group)
        self._group_param_table = QTableWidget(0, 4)
        self._group_param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Bounds"])
        self._group_param_table.horizontalHeader().setStretchLastSection(False)
        self._group_param_table.setColumnWidth(0, 110)
        self._group_param_table.setColumnWidth(1, 90)
        self._group_param_table.setColumnWidth(2, 100)
        self._group_param_table.setColumnWidth(3, 150)
        _apply_param_table_style(self._group_param_table)
        self._group_param_table.itemChanged.connect(self._on_group_param_item_changed)
        group_param_layout.addWidget(self._group_param_table)
        group_param_button_layout = QHBoxLayout()
        group_param_button_layout.setContentsMargins(0, 0, 0, 0)
        group_param_button_layout.addStretch()
        self._group_param_reset_btn = QPushButton("Reset to Estimates")
        self._group_param_reset_btn.clicked.connect(self._reset_group_parameter_estimates)
        group_param_button_layout.addWidget(self._group_param_reset_btn)
        group_param_layout.addLayout(group_param_button_layout)
        layout.addWidget(self._group_param_group)

        self._group_model_group = QGroupBox("Fit-Function Parameters")
        group_model_layout = QVBoxLayout(self._group_model_group)
        self._group_model_table = QTableWidget(0, 4)
        self._group_model_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Bounds"])
        self._group_model_table.horizontalHeader().setStretchLastSection(False)
        self._group_model_table.setColumnWidth(0, 110)
        self._group_model_table.setColumnWidth(1, 90)
        self._group_model_table.setColumnWidth(2, 100)
        self._group_model_table.setColumnWidth(3, 150)
        _apply_param_table_style(self._group_model_table)
        self._group_model_table.itemChanged.connect(self._on_group_model_table_item_changed)
        group_model_layout.addWidget(self._group_model_table)
        layout.addWidget(self._group_model_group)

        # Fit button
        btn_layout = QHBoxLayout()
        self._fit_btn = QPushButton("Run Batch Fit")
        self._fit_btn.clicked.connect(self._run_global_fit)
        self._fit_btn.setEnabled(False)
        btn_layout.addWidget(self._fit_btn)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_requested)
        self._preview_btn.setEnabled(False)
        btn_layout.addWidget(self._preview_btn)
        self._initial_values_btn = QPushButton("Initial Values...")
        self._initial_values_btn.setToolTip(
            "Edit per-member initial parameter values (per run, or per run/group for grouped fits)."
        )
        self._initial_values_btn.clicked.connect(self._open_initial_values_dialog)
        btn_layout.addWidget(self._initial_values_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Results display
        self._results_group = QGroupBox("Batch Fit Results")
        results_layout = QVBoxLayout(self._results_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(200)
        self._result_text.setText("No fit performed yet")
        results_layout.addWidget(self._result_text)
        layout.addWidget(self._results_group)

        layout.addStretch()

        # Thread management for non-blocking fits
        self._fit_thread: QThread | None = None
        self._fit_worker: GlobalFitWorker | None = None

        self._setup_group_nuisance_table()
        self._set_composite_model(self._composite_model)
        self._update_mode_ui(preserve_result=False)

    def _default_composite_model(self) -> CompositeModel:
        """Return the initial composite model for the allowed fitting modes."""
        if self._domain == "frequency":
            return default_frequency_model()
        if self._member_kind == "groups":
            return CompositeModel(["OscillatoryField"])
        return CompositeModel(["Exponential", "Constant"], operators=["+"])

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_domain(self, domain: str) -> None:
        """Switch labels and default model for time or frequency global fitting."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            return
        self._domain = normalized
        if self._domain == "frequency":
            self._formula_row_label.setText("S(ν):")
            self._fit_range_mid_label.setText("≤ <i>ν</i> ≤")
            self._fit_range_unit_label.setText("MHz")
            self._fit_range_min_spin.setDecimals(4)
            self._fit_range_max_spin.setDecimals(4)
            self._fit_range_min_spin.setRange(-1_000_000.0, 1_000_000.0)
            self._fit_range_max_spin.setRange(-1_000_000.0, 1_000_000.0)
            self._fit_wizard_btn.setEnabled(False)
            self._fit_wizard_btn.setToolTip(
                "Global Fit Wizard is currently available for time-domain fits."
            )
        else:
            self._formula_row_label.setText("A(t):")
            self._fit_range_mid_label.setText("≤ <i>t</i> ≤")
            self._fit_range_unit_label.setText("μs")
            self._fit_range_min_spin.setDecimals(3)
            self._fit_range_max_spin.setDecimals(3)
            self._fit_range_min_spin.setRange(-1000.0, 1000.0)
            self._fit_range_max_spin.setRange(-1000.0, 1000.0)
            self._fit_wizard_btn.setToolTip("")
        self._set_composite_model(self._default_composite_model())
        self._update_mode_ui(preserve_result=False)

    def _show_parameter_classification_help(self) -> None:
        QMessageBox.information(
            self,
            "Parameter Classification Help",
            _GLOBAL_FIT_PARAMETER_CLASSIFICATION_HELP_TEXT,
        )

    def register_single_fit_seed(
        self, run_number: int, model: CompositeModel, fit_result: object
    ) -> None:
        """Store successful single-fit results for later global-fit initialisation."""
        if getattr(fit_result, "success", False) is not True:
            return

        values_by_name: dict[str, float] = {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            value = getattr(param, "value", None)
            if isinstance(name, str):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    values_by_name[name] = numeric_value

        if not values_by_name:
            return

        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return

        self._single_fit_seed_by_run[run_key] = {
            "model": model.to_dict(),
            "values": values_by_name,
        }
        self._refresh_inherited_single_fit_defaults()

    def remove_single_fit_seeds(self, run_numbers: list[int] | set[int]) -> set[int]:
        """Remove stored single-fit seeds for the given runs."""
        removed: set[int] = set()
        for run_number in run_numbers:
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            if self._single_fit_seed_by_run.pop(run_key, None) is not None:
                removed.add(run_key)
        if removed:
            self._refresh_inherited_single_fit_defaults()
        return removed

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting."""
        self._datasets = datasets
        self._invalidate_wizard_cache_if_stale()
        self._update_mode_ui(preserve_result=False)
        self._refresh_inherited_single_fit_defaults()

    def set_member_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the member runs for a grouped *series* fit.

        Each member contributes its detector groups; an empty/one-element set
        reduces to the single-run grouped fit (the groups "single fit").
        """
        self._member_datasets = [ds for ds in (datasets or []) if ds is not None]
        self._grouped_context_cache = None
        self._update_mode_ui(preserve_result=False)

    def _grouped_member_datasets(self) -> list[MuonDataset]:
        """Resolve the member runs for grouped fitting (active dataset fallback)."""
        if self._member_datasets:
            return list(self._member_datasets)
        return [self._current_dataset] if self._current_dataset is not None else []

    def set_frequency_missing_spectra_status(
        self, missing_run_numbers: list[int], cached_count: int
    ) -> None:
        """Show an actionable status for selected runs without cached spectra."""
        if self._domain != "frequency" or not missing_run_numbers:
            return
        preview = ", ".join(str(run_number) for run_number in missing_run_numbers[:8])
        if len(missing_run_numbers) > 8:
            preview += f", +{len(missing_run_numbers) - 8} more"
        prefix = (
            f"{cached_count} cached frequency spectra selected.\n"
            if cached_count > 0
            else "No cached frequency spectra selected.\n"
        )
        self._result_text.setText(
            f"{prefix}Compute a Fourier spectrum for run(s) {preview} before global frequency fitting."
        )

    def set_current_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the active dataset used by grouped time-domain mode."""
        self._current_dataset = dataset
        # Invalidate the grouped-context memo whenever the active dataset
        # changes (its grouped groups depend only on this dataset).
        self._grouped_context_cache = None
        self._refresh_field_parameter_defaults_for_current_dataset()
        self._refresh_group_phase_defaults_for_current_dataset()
        self._update_group_parameter_defaults()
        self._update_mode_ui(preserve_result=False)

    def _refresh_field_parameter_defaults_for_current_dataset(self) -> None:
        """Refresh auto-seeded field values when the active dataset changes."""
        field_gauss = _get_file_value_for_parameter(self._current_dataset, "field")
        target_field = float(field_gauss) if field_gauss is not None else 0.0
        _refresh_field_defaults_in_table(
            self._param_table,
            self._composite_model,
            previous_field_gauss=self._applied_field_default_gauss,
            current_field_gauss=target_field,
        )
        _refresh_field_defaults_in_table(
            self._group_model_table,
            self._grouped_fit_model(),
            previous_field_gauss=self._applied_field_default_gauss,
            current_field_gauss=target_field,
        )
        self._applied_field_default_gauss = target_field

    def _refresh_group_phase_defaults_for_current_dataset(self) -> None:
        """Refresh grouped-model phase defaults when the active dataset changes."""
        if self._group_model_table.rowCount() == 0:
            self._applied_group_phase_default_rad = 0.0
            return

        grouped_model = self._grouped_fit_model()
        grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        phase_defaults = _grouped_model_phase_defaults(grouped_model, grouped_groups or [])
        if not phase_defaults:
            self._applied_group_phase_default_rad = 0.0
            return

        previous_phase = float(self._applied_group_phase_default_rad)
        current_phase = float(next(iter(phase_defaults.values())))
        row_by_name = _param_table_rows_by_name(self._group_model_table)
        previous_signal_state = self._group_model_table.blockSignals(True)
        try:
            for pname, default_value in phase_defaults.items():
                row = row_by_name.get(pname)
                if row is None:
                    continue
                value_item = self._group_model_table.item(row, 1)
                if value_item is None:
                    continue
                try:
                    existing_value = float(value_item.text())
                except (TypeError, ValueError):
                    existing_value = previous_phase
                if value_item.text().strip() and not np.isclose(existing_value, previous_phase):
                    continue
                value_item.setText(f"{float(default_value):.6g}")
        finally:
            self._group_model_table.blockSignals(previous_signal_state)
        self._applied_group_phase_default_rad = current_phase

    def _invalidate_wizard_cache_if_stale(self) -> None:
        self._sync_active_wizard_cache_from_selection()

    def _normalized_wizard_run_set(
        self,
        run_numbers: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        normalized: list[int] = []
        source = run_numbers
        if source is None:
            source = [
                int(dataset.run_number)
                for dataset in self._datasets
                if getattr(dataset, "run_number", None) is not None
            ]
        for run_number in source:
            try:
                normalized.append(int(run_number))
            except (TypeError, ValueError):
                continue
        return tuple(sorted(normalized))

    def _normalized_wizard_signature(
        self,
        signature: dict[str, object],
    ) -> dict[str, object]:
        normalized = copy.deepcopy(signature)
        normalized.pop("search_strategy", None)
        run_numbers = normalized.get("run_numbers")
        if isinstance(run_numbers, tuple | list):
            normalized["run_numbers"] = list(self._normalized_wizard_run_set(tuple(run_numbers)))
        return normalized

    def _set_active_wizard_cache(
        self,
        recommendation: GlobalFitWizardRecommendation | None,
        *,
        signature: dict[str, object] | None,
        log_text: str = "",
    ) -> None:
        self._cached_wizard_recommendation = recommendation
        self._cached_wizard_signature = (
            self._normalized_wizard_signature(signature) if isinstance(signature, dict) else None
        )
        self._cached_wizard_log_text = str(log_text)

    def _wizard_cache_entry_for_run_set(
        self,
        run_set: tuple[int, ...] | None = None,
    ) -> dict[str, object] | None:
        key = self._normalized_wizard_run_set(run_set)
        if not key:
            return None
        entry = self._wizard_cache_by_run_set.get(key)
        return entry if isinstance(entry, dict) else None

    def _sync_active_wizard_cache_from_selection(self) -> None:
        entry = self._wizard_cache_entry_for_run_set()
        if entry is None:
            self._set_active_wizard_cache(None, signature=None, log_text="")
            return
        recommendation = entry.get("recommendation")
        signature = entry.get("signature")
        log_text = entry.get("log_text", "")
        if not isinstance(recommendation, GlobalFitWizardRecommendation) or not isinstance(
            signature, dict
        ):
            self._set_active_wizard_cache(None, signature=None, log_text="")
            return
        self._set_active_wizard_cache(
            recommendation,
            signature=signature,
            log_text=str(log_text),
        )

    def _wizard_context_signature(self, parsed: dict[str, object]) -> dict[str, object]:
        return {
            "run_numbers": list(self._normalized_wizard_run_set()),
            "model": self._composite_model.to_dict(),
            "types": {str(key): str(value) for key, value in dict(parsed["types"]).items()},
            "values": {str(key): float(value) for key, value in dict(parsed["values"]).items()},
            "bounds": {
                str(key): [float(bounds[0]), float(bounds[1])]
                for key, bounds in dict(parsed["bounds"]).items()
            },
        }

    def _cache_wizard_analysis(
        self,
        recommendation: GlobalFitWizardRecommendation,
        *,
        signature: dict[str, object],
        log_text: str = "",
    ) -> None:
        normalized_signature = self._normalized_wizard_signature(signature)
        run_set = self._normalized_wizard_run_set(normalized_signature.get("run_numbers"))
        if run_set:
            self._wizard_cache_by_run_set[run_set] = {
                "signature": normalized_signature,
                "recommendation": recommendation,
                "log_text": str(log_text),
            }
        self._set_active_wizard_cache(
            recommendation,
            signature=normalized_signature,
            log_text=log_text,
        )

    def _serialize_wizard_cache_store(self) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for run_set in sorted(self._wizard_cache_by_run_set):
            entry = self._wizard_cache_by_run_set.get(run_set)
            if not isinstance(entry, dict):
                continue
            recommendation = entry.get("recommendation")
            signature = entry.get("signature")
            if not isinstance(recommendation, GlobalFitWizardRecommendation) or not isinstance(
                signature, dict
            ):
                continue
            serialized.append(
                {
                    "run_numbers": list(run_set),
                    "signature": copy.deepcopy(signature),
                    "recommendation": serialize_global_fit_wizard_recommendation(recommendation),
                    "log_text": str(entry.get("log_text", "")),
                }
            )
        return serialized

    def _restore_wizard_cache_store(self, payload: object) -> None:
        self._wizard_cache_by_run_set = {}
        if not isinstance(payload, list):
            self._sync_active_wizard_cache_from_selection()
            return
        for raw_entry in payload:
            if not isinstance(raw_entry, dict):
                continue
            recommendation = deserialize_global_fit_wizard_recommendation(
                raw_entry.get("recommendation")
            )
            signature = raw_entry.get("signature")
            raw_run_numbers = raw_entry.get("run_numbers")
            if recommendation is None or not isinstance(signature, dict):
                continue
            if not isinstance(raw_run_numbers, tuple | list):
                raw_run_numbers = signature.get("run_numbers")
            run_set = self._normalized_wizard_run_set(raw_run_numbers)
            if not run_set:
                continue
            self._wizard_cache_by_run_set[run_set] = {
                "signature": self._normalized_wizard_signature(signature),
                "recommendation": recommendation,
                "log_text": str(raw_entry.get("log_text", "")),
            }
        self._sync_active_wizard_cache_from_selection()

    def _single_fit_wizard_cache_for_run(
        self,
        run_number: int,
    ) -> tuple[FitWizardRecommendation | None, dict[str, object] | None, str]:
        parent = self._fit_panel_host()
        getter = getattr(parent, "get_single_fit_wizard_cache_for_run", None)
        if not callable(getter):
            return None, None, ""
        payload = getter(run_number)
        if not isinstance(payload, tuple) or len(payload) != 3:
            return None, None, ""
        recommendation, signature, log_text = payload
        if not isinstance(recommendation, FitWizardRecommendation):
            return None, None, ""
        return recommendation, signature if isinstance(signature, dict) else None, str(log_text)

    def _existing_single_fit_recommendations_for_selected_runs(
        self,
    ) -> dict[int, FitWizardRecommendation]:
        recommendations: dict[int, FitWizardRecommendation] = {}
        for dataset in self._datasets:
            recommendation, _signature, _log_text = self._single_fit_wizard_cache_for_run(
                int(dataset.run_number)
            )
            if recommendation is not None:
                recommendations[int(dataset.run_number)] = recommendation
        return recommendations

    def _on_single_fit_recommendations_generated(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        parent = self._fit_panel_host()
        persist = getattr(parent, "persist_single_fit_wizard_cache_for_run", None)
        if not callable(persist):
            return
        for run_number, recommendation in payload.items():
            if not isinstance(recommendation, FitWizardRecommendation):
                continue
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            persist(
                run_key,
                recommendation,
                signature={"run_number": run_key, "model": None},
                log_text="Updated by Global Fit Wizard screening.",
            )

    def _fit_panel_host(self) -> object | None:
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "get_single_fit_wizard_cache_for_run") and hasattr(
                parent,
                "persist_single_fit_wizard_cache_for_run",
            ):
                return parent
            next_parent = getattr(parent, "parent", None)
            parent = next_parent() if callable(next_parent) else None
        return None

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Enable/disable global-fit execution while preserving selected datasets."""
        self._fit_blocked = bool(blocked)
        self._fit_block_reason = str(reason)
        self._update_mode_ui(preserve_result=True)

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update fit-range spinboxes from the plot without re-emitting."""
        have_range = x_min is not None and x_max is not None
        self._fit_range_min_spin.setEnabled(have_range)
        self._fit_range_max_spin.setEnabled(have_range)
        if not have_range:
            return
        with QSignalBlocker(self._fit_range_min_spin):
            self._fit_range_min_spin.setValue(float(x_min))
        with QSignalBlocker(self._fit_range_max_spin):
            self._fit_range_max_spin.setValue(float(x_max))

    def _on_fit_range_spinbox_committed(self) -> None:
        """Emit fit_range_edit_committed when the user finishes editing a spinbox."""
        self.fit_range_edit_committed.emit(
            self._fit_range_min_spin.value(),
            self._fit_range_max_spin.value(),
        )

    def _refresh_inherited_single_fit_defaults(self) -> None:
        """Apply single-fit seeds when every selected dataset shares one model."""
        self._inherited_seed_by_run = {}
        self._inherited_model_dict = None

        if len(self._datasets) < 2:
            return

        run_numbers: list[int] = []
        for ds in self._datasets:
            try:
                run_numbers.append(int(ds.run_number))
            except (TypeError, ValueError):
                return

        seeds: list[dict[str, object]] = []
        for run_number in run_numbers:
            seed = self._single_fit_seed_by_run.get(run_number)
            if not isinstance(seed, dict):
                return
            seeds.append(seed)

        first_model = seeds[0].get("model")
        if not isinstance(first_model, dict):
            return
        for seed in seeds[1:]:
            if seed.get("model") != first_model:
                return

        try:
            inherited_model = CompositeModel.from_dict(first_model)
        except ValueError:
            return

        inherited_values_by_run: dict[int, dict[str, float]] = {}
        for run_number, seed in zip(run_numbers, seeds, strict=False):
            values = seed.get("values")
            if not isinstance(values, dict):
                return
            typed_values: dict[str, float] = {}
            for key, value in values.items():
                if not isinstance(key, str):
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric_value):
                    typed_values[key] = numeric_value
            if not typed_values:
                return
            inherited_values_by_run[run_number] = typed_values

        self._set_composite_model(inherited_model)

        averages = self._inherited_param_averages(
            inherited_values_by_run,
            inherited_model.param_names,
        )
        if averages:
            self._updating_fraction_values = True
            for row in range(self._param_table.rowCount()):
                name_item = self._param_table.item(row, 0)
                pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(pname, str):
                    pname = name_item.text() if name_item else ""
                if pname not in averages:
                    continue
                value_item = self._param_table.item(row, 1)
                if value_item is not None:
                    value_item.setText(f"{averages[pname]:.6g}")
            self._updating_fraction_values = False
            self._synchronize_fraction_value_rows()

        self._inherited_seed_by_run = inherited_values_by_run
        self._inherited_model_dict = inherited_model.to_dict()

    def _inherited_param_averages(
        self,
        values_by_run: dict[int, dict[str, float]],
        param_names: list[str],
    ) -> dict[str, float]:
        """Return finite means per parameter from inherited per-run seeds."""
        averages: dict[str, float] = {}
        for pname in param_names:
            vals: list[float] = []
            for values in values_by_run.values():
                value = values.get(pname)
                if value is None:
                    continue
                if np.isfinite(value):
                    vals.append(float(value))
            if vals:
                averages[pname] = float(np.mean(vals))
        return averages

    def _current_parameter_row_state(self) -> dict[str, dict[str, str]]:
        """Capture current parameter-table edits before rebuilding rows."""
        state: dict[str, dict[str, str]] = {}
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            if name_item is None:
                continue
            param_name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(param_name, str):
                continue
            value_item = self._param_table.item(row, 1)
            bounds_item = self._param_table.item(row, 3)
            type_combo = self._param_table.cellWidget(row, 2)
            state[param_name] = {
                "value": value_item.text() if value_item is not None else "",
                "bounds": bounds_item.text() if bounds_item is not None else "-inf, inf",
                "type": type_combo.currentText() if isinstance(type_combo, QComboBox) else "",
            }
        return state

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild classification rows."""
        preserved_state = self._current_parameter_row_state()
        grouped_model_state = self._current_grouped_model_row_state()
        # A new model invalidates any per-run initial values keyed by old names.
        self._user_initial_values_by_run = {}
        self._user_grouped_initial_values = {}
        self._updating_fraction_values = True
        self._composite_model = model
        _set_formula_label_text(self._formula_label, model.formula_string())
        _apply_domain_mismatch_warning(self._formula_label, model, self._domain)

        # Use the mean field across loaded datasets (if non-zero) as the default
        # for any 'field' parameters.
        dataset_fields = [
            ds.run.field for ds in self._datasets if ds.run is not None and ds.run.field != 0.0
        ]
        mean_field = float(np.mean(dataset_fields)) if dataset_fields else 0.0
        field_overrides = _field_value_overrides(model, mean_field)
        frequency_seed_values: dict[str, list[float]] = {}
        if self._domain == "frequency":
            for dataset in self._datasets:
                for key, value in seed_peak_parameters_from_dataset(dataset, model).items():
                    frequency_seed_values.setdefault(key, []).append(float(value))
        frequency_overrides = {
            key: float(np.mean(values)) for key, values in frequency_seed_values.items() if values
        }
        self._applied_field_default_gauss = mean_field

        fixed_default_params = model.fixed_by_default_params()
        self._param_table.setRowCount(len(model.param_names))
        for i, pname in enumerate(model.param_names):
            previous = preserved_state.get(pname, {})
            # Parameter name
            name_item = _make_param_name_item(_format_param_label(pname), pname)
            self._param_table.setItem(i, 0, name_item)

            # Initial value — use dataset field for 'field' parameters if available
            default_val = frequency_overrides.get(
                pname, field_overrides.get(pname, model.param_defaults.get(pname, 0.0))
            )
            value_item = QTableWidgetItem(previous.get("value") or str(default_val))
            self._param_table.setItem(i, 1, value_item)

            # Type selection (Global/Local/Fixed/File dropdown)
            type_combo = QComboBox()
            type_combo.addItems(["Global", "Local", "Fixed"])
            # Check if this parameter has file-specific defaults
            base_name, _index = split_parameter_name(pname)
            if base_name in {"field", "B_L"}:
                type_combo.addItem("File")
            # Set default: first parameter (usually amplitude) as Global, others
            # as Local; component-declared fixed-by-default parameters as Fixed.
            if pname in fixed_default_params:
                type_combo.setCurrentText("Fixed")
            else:
                type_combo.setCurrentText("Global" if i == 0 else "Local")
            previous_type = previous.get("type")
            if previous_type:
                previous_index = type_combo.findText(previous_type)
                if previous_index >= 0:
                    type_combo.setCurrentIndex(previous_index)
            self._param_table.setCellWidget(i, 2, type_combo)

            # Bounds (min, max) — default lower bound to 0 for positive-definite parameters
            default_min = get_param_info(pname).default_min
            min_text = str(default_min) if default_min is not None else "-inf"
            bounds_item = QTableWidgetItem(previous.get("bounds") or f"{min_text}, inf")
            self._param_table.setItem(i, 3, bounds_item)

        _configure_fraction_rows_in_table(
            self._param_table,
            model,
            bounds_column=3,
            type_column=2,
        )
        self._rebuild_grouped_model_table(grouped_model_state)
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()
        if self.is_grouped_time_domain_mode():
            _set_formula_label_text(
                self._formula_label, _grouped_formula_string(self._grouped_fit_model())
            )

    def _grouped_fit_model(self) -> CompositeModel:
        """Return the grouped-mode model with default fraction semantics applied."""
        return self._composite_model.with_default_fraction_groups()

    def get_grouped_state(self) -> dict:
        """Return the grouped-fit classification for persisting a group FitSeries.

        Splits the two-tier parameter block into the *physics* roles
        (``param_roles``: the fit-function parameters on the global/local/fixed
        ladder) and the always-per-group ``nuisance_params`` block.  Returns an
        empty dict when the grouped tables cannot currently be parsed.
        """
        try:
            config = self._parse_grouped_parameter_configuration()
        except ValueError:
            return {}
        # Physics (fit-function) roles are tracked directly by the parser, where
        # "global" = shared across runs and "local" = per run (but still shared
        # across that run's groups).
        physics_roles = {
            str(name): str(role) for name, role in config.get("physics_roles", {}).items()
        }
        # Nuisance block = group-nuisance params that appear anywhere in the
        # classification (always estimated per (run, group)).
        seen: set[str] = set()
        nuisance_params: list[str] = []
        for name in (*config.get("global", []), *config.get("local", []), *config.get("fixed", [])):
            if name in GROUP_NUISANCE_PARAMS and name not in seen:
                seen.add(name)
                nuisance_params.append(name)
        return {
            "composite_model": self._grouped_fit_model().to_dict(),
            "param_roles": physics_roles,
            "nuisance_params": nuisance_params,
        }

    def param_role_map(self) -> dict[str, str]:
        """Return the run-batch parameter roles (``{name: global/local/fixed/file}``).

        Read from the classification table so a completed batch can annotate each
        member's piped-back single fit with how the parameter was treated.
        """
        roles: dict[str, str] = {}
        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            if name_item is None:
                continue
            name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(name, str):
                continue
            combo = self._param_table.cellWidget(row, 2)
            role = combo.currentText().strip().lower() if isinstance(combo, QComboBox) else ""
            if role in ("global", "local", "fixed", "file"):
                roles[name] = role
        return roles

    def _synchronize_fraction_value_rows(self, edited_param_name: str | None = None) -> None:
        self._updating_fraction_values = True
        try:
            _synchronize_fraction_group_values_in_table(
                self._param_table,
                self._composite_model,
                edited_param_name=edited_param_name,
            )
        finally:
            self._updating_fraction_values = False

    def _on_param_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_fraction_values or item.column() != 1:
            return
        name_item = self._param_table.item(item.row(), 0)
        param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
        if isinstance(param_name, str):
            self._synchronize_fraction_value_rows(param_name)

    def _synchronize_grouped_model_fraction_rows(
        self,
        edited_param_name: str | None = None,
    ) -> None:
        self._updating_group_model_fraction_values = True
        try:
            _synchronize_fraction_group_values_in_table(
                self._group_model_table,
                self._grouped_fit_model(),
                edited_param_name=edited_param_name,
            )
        finally:
            self._updating_group_model_fraction_values = False

    def _on_group_model_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_group_model_fraction_values or item.column() != 1:
            return
        name_item = self._group_model_table.item(item.row(), 0)
        param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
        if isinstance(param_name, str):
            self._synchronize_grouped_model_fraction_rows(param_name)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(
            self, initial_model=self._composite_model, domain=self._domain
        )
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _open_fit_wizard(self) -> None:
        """Launch or refresh the non-modal global fit wizard window."""
        if self.is_grouped_time_domain_mode():
            self._result_text.setText(
                "Grouped time-domain mode uses its own parameter blocks. "
                "The Global Fit Wizard is unavailable in this mode."
            )
            return
        if self._domain == "frequency":
            self._result_text.setText(
                "Global Fit Wizard is currently available for time-domain fits."
            )
            return
        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason or "Global fit is unavailable for the current selection."
            )
            return
        if len(self._datasets) < 2:
            self._result_text.setText("Global fit wizard requires at least 2 datasets.")
            return

        try:
            parsed = self._parse_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        if self._fit_wizard_window is None:
            self._fit_wizard_window = GlobalFitWizardWindow(self)
            self._fit_wizard_window.apply_assessment_requested.connect(
                self._apply_fit_wizard_assessment
            )
            self._fit_wizard_window.analysis_cached.connect(self._on_fit_wizard_analysis_cached)
            self._fit_wizard_window.single_fit_recommendations_generated.connect(
                self._on_single_fit_recommendations_generated
            )
            self._fit_wizard_window.parameter_setup_applied.connect(
                self._on_fit_wizard_parameter_setup_applied
            )
        signature = self._wizard_context_signature(parsed)

        self._fit_wizard_window.set_analysis_context(
            self._datasets,
            current_model=self._composite_model,
            current_parameter_types=parsed["types"],
            current_values=parsed["values"],
            parameter_bounds=parsed["bounds"],
            existing_single_fit_recommendations_by_run=self._existing_single_fit_recommendations_for_selected_runs(),
        )
        cached_entry = self._wizard_cache_entry_for_run_set()
        cached_recommendation = None
        cached_signature = None
        cached_log_text = ""
        if isinstance(cached_entry, dict):
            candidate = cached_entry.get("recommendation")
            candidate_signature = cached_entry.get("signature")
            if isinstance(candidate, GlobalFitWizardRecommendation) and isinstance(
                candidate_signature, dict
            ):
                cached_recommendation = candidate
                cached_signature = candidate_signature
                cached_log_text = str(cached_entry.get("log_text", ""))
        if cached_recommendation is not None and self._wizard_base_signature_matches(
            cached_signature,
            signature,
        ):
            self._fit_wizard_window.set_cached_recommendation(
                cached_recommendation,
                signature=cached_signature,
                log_text=cached_log_text,
            )
        elif cached_recommendation is not None:
            self._fit_wizard_window.set_cached_recommendation(
                cached_recommendation,
                signature=cached_signature,
                log_text=cached_log_text,
                status_text=(
                    "Showing previously cached Global Fit Wizard results for these runs. "
                    "Rebuild screening to refresh them for the current parameter setup."
                ),
            )
        self._fit_wizard_window.show()
        self._fit_wizard_window.raise_()
        self._fit_wizard_window.activateWindow()

    def _on_fit_wizard_analysis_cached(
        self,
        recommendation: GlobalFitWizardRecommendation,
        log_text: str,
        signature: object,
    ) -> None:
        if not isinstance(signature, dict):
            return
        self._cache_wizard_analysis(
            recommendation,
            signature=signature,
            log_text=log_text,
        )

    def _on_fit_wizard_parameter_setup_applied(self, config: object) -> None:
        if not isinstance(config, dict):
            return
        types = config.get("types")
        bounds = config.get("bounds")
        if not isinstance(types, dict) or not isinstance(bounds, dict):
            return

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else ""

            if pname in types:
                type_combo = self._param_table.cellWidget(row, 2)
                if isinstance(type_combo, QComboBox):
                    idx = type_combo.findText(str(types[pname]))
                    if idx >= 0:
                        type_combo.setCurrentIndex(idx)

            raw_bounds = bounds.get(pname)
            if not isinstance(raw_bounds, tuple | list) or len(raw_bounds) != 2:
                continue
            try:
                min_val = float(raw_bounds[0])
                max_val = float(raw_bounds[1])
            except (TypeError, ValueError):
                continue

            bounds_item = self._param_table.item(row, 4)
            if bounds_item is not None:
                bounds_item.setText(_format_bounds_pair(min_val, max_val))

            value_item = self._param_table.item(row, 1)
            if value_item is None:
                continue
            try:
                value = float(value_item.text())
            except (TypeError, ValueError):
                continue
            clipped = float(np.clip(value, min_val, max_val))
            if clipped != value:
                value_item.setText(f"{clipped:.6g}")

    def _parse_parameter_configuration(self) -> dict[str, object]:
        """Return validated parameter values, roles, and bounds from the table."""
        global_params: list[str] = []
        local_params: list[str] = []
        fixed_params: dict[str, float] = {}
        file_params: list[str] = []
        param_values: dict[str, float] = {}
        param_bounds: dict[str, tuple[float, float]] = {}
        param_types: dict[str, str] = {}

        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else f"param_{i}"

            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                raise ValueError(f"Error: Invalid value for {_format_param_label(pname)}") from None

            # Get the Type selection (Global/Local/Fixed/File)
            type_combo = self._param_table.cellWidget(i, 2)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Local"
            param_types[pname] = type_text

            # Only validate value if type is not "File"
            if type_text != "File":
                if not np.isfinite(value):
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                    )
            param_values[pname] = value

            bounds_text = self._param_table.item(i, 3).text()
            try:
                parts = bounds_text.split(",")
                lo = parts[0].strip()
                hi = parts[1].strip()
                min_val = float(lo) if lo != "-inf" else -float("inf")
                max_val = float(hi) if hi != "inf" else float("inf")
            except (ValueError, IndexError):
                min_val, max_val = -float("inf"), float("inf")

            if np.isfinite(min_val) and np.isfinite(max_val) and min_val > max_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} has invalid bounds: {min_val} > {max_val}"
                )
            if type_text != "File":
                if np.isfinite(min_val) and value < min_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} is below minimum {min_val}"
                    )
                if np.isfinite(max_val) and value > max_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} is above maximum {max_val}"
                    )

            param_bounds[pname] = (min_val, max_val)

            if type_text == "Global":
                global_params.append(pname)
            elif type_text == "Local":
                local_params.append(pname)
            elif type_text == "File":
                file_params.append(pname)
            else:  # Fixed
                fixed_params[pname] = value

        return {
            "global": global_params,
            "local": local_params,
            "fixed": fixed_params,
            "file": file_params,
            "values": param_values,
            "bounds": param_bounds,
            "types": param_types,
        }

    def _wizard_base_signature_matches(
        self,
        cached_signature: dict[str, object] | None,
        base_signature: dict[str, object],
    ) -> bool:
        if not isinstance(cached_signature, dict):
            return False
        for key in ("run_numbers", "model", "values"):
            if cached_signature.get(key) != base_signature.get(key):
                return False
        cached_types = cached_signature.get("types")
        base_types = base_signature.get("types")
        if not isinstance(cached_types, dict) or cached_types != base_types:
            return False
        cached_bounds = cached_signature.get("bounds")
        base_bounds = base_signature.get("bounds")
        if not isinstance(cached_bounds, dict):
            return False
        for name, bounds in base_bounds.items():
            if cached_bounds.get(name) != bounds:
                return False
        return True

    def _effective_initial_values_by_run(self, parsed: dict) -> dict[int, dict[str, float]]:
        """Per-run initial values used by the batch fit.

        Precedence: parameter-table value < inherited single-fit seed (per run for
        Local, average for Global/Fixed) < explicit Initial-values dialog entry.
        """
        model = self._composite_model
        if model is None:
            return {}
        param_values = dict(parsed.get("values", {}))
        local_params = set(parsed.get("local", []))
        global_params = set(parsed.get("global", []))
        fixed_params = set(parsed.get("fixed", {}))

        inherited_seed_by_run: dict[int, dict[str, float]] = {}
        inherited_averages: dict[str, float] = {}
        if self._inherited_model_dict == model.to_dict() and self._inherited_seed_by_run:
            selected_runs = {int(ds.run_number) for ds in self._datasets}
            if selected_runs.issubset(self._inherited_seed_by_run):
                inherited_seed_by_run = {
                    run_number: self._inherited_seed_by_run[run_number]
                    for run_number in selected_runs
                }
                inherited_averages = self._inherited_param_averages(
                    inherited_seed_by_run, model.param_names
                )

        result: dict[int, dict[str, float]] = {}
        for ds in self._datasets:
            run_number = int(ds.run_number)
            local_seed_values = inherited_seed_by_run.get(run_number, {})
            user_values = self._user_initial_values_by_run.get(run_number, {})
            run_values: dict[str, float] = {}
            for pname in model.param_names:
                value = float(param_values.get(pname, 0.0))
                if inherited_seed_by_run:
                    if pname in local_params and pname in local_seed_values:
                        value = float(local_seed_values[pname])
                    elif pname in inherited_averages and (
                        pname in global_params or pname in fixed_params
                    ):
                        value = float(inherited_averages[pname])
                if pname in user_values:
                    value = float(user_values[pname])
                run_values[pname] = value
            result[run_number] = run_values
        return result

    def _grouped_member_specs(self) -> list[tuple[int, str, int, object]]:
        """Return ``(synthetic_key, label, run, group_id)`` for every grouped member."""
        specs: list[tuple[int, str, int, object]] = []
        for run, groups in self._grouped_members.items():
            for index, group in enumerate(groups, start=1):
                key = _group_dataset_run_number(int(run), index)
                group_name = str(getattr(group, "group_name", getattr(group, "group_id", index)))
                specs.append((key, f"{run} · {group_name}", int(run), group.group_id))
        return specs

    def _open_grouped_initial_values_dialog(self) -> None:
        """Open the per-(run, group) nuisance initial-value editor for grouped fits."""
        # Ensure the grouped context (and ``_grouped_members``) is current.
        self._grouped_mode_context()
        if not self._grouped_members:
            return
        try:
            config = self._parse_grouped_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return
        group_values = dict(config.get("group_values", {}))  # param -> {group_id: value}

        params = [(name, _format_param_label(name), "local") for name in GROUP_NUISANCE_PARAMS]
        members: list[tuple[int, str]] = []
        values: dict[int, dict[str, float]] = {}
        for key, label, _run, group_id in self._grouped_member_specs():
            members.append((key, label))
            user = self._user_grouped_initial_values.get(key, {})
            values[key] = {
                name: float(user.get(name, group_values.get(name, {}).get(group_id, 0.0)))
                for name in GROUP_NUISANCE_PARAMS
            }
        if not members:
            return
        dialog = InitialValuesDialog(
            members, params, values, parent=self, title="Grouped nuisance initial values"
        )
        if dialog.exec():
            self._user_grouped_initial_values = dialog.edited_values()

    def _open_initial_values_dialog(self) -> None:
        """Open the members × parameters initial-value editor for the batch fit."""
        if self._member_kind == "groups":
            self._open_grouped_initial_values_dialog()
            return
        if not self._datasets or self._composite_model is None:
            return
        try:
            parsed = self._parse_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return
        types = dict(parsed.get("types", {}))
        params: list[tuple[str, str, str]] = []
        for pname in self._composite_model.param_names:
            raw_type = str(types.get(pname, "Local")).lower()
            if raw_type == "local":
                role = "local"
            elif raw_type == "global":
                role = "global"
            else:  # Fixed / File
                role = "fixed"
            params.append((pname, _format_param_label(pname), role))
        members = [
            (int(ds.run_number), str(getattr(ds, "run_label", ds.run_number)))
            for ds in self._datasets
        ]
        values = self._effective_initial_values_by_run(parsed)
        dialog = InitialValuesDialog(
            members, params, values, parent=self, title="Batch initial values"
        )
        if dialog.exec():
            self._user_initial_values_by_run = dialog.edited_values()

    def batch_datasets(self) -> list[MuonDataset]:
        """Return the datasets currently configured for the batch/scan."""
        return list(self._datasets)

    def _run_global_fit(self) -> None:
        """Execute global fit on all datasets."""
        if self.is_grouped_time_domain_mode():
            self._run_grouped_time_domain_fit()
            return

        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason or "Global fit is unavailable for the current selection."
            )
            return

        if len(self._datasets) < 2:
            self._result_text.setText("Error: Need at least 2 datasets for global fitting")
            return

        if self._composite_model is None:
            self._result_text.setText("Error: No function defined")
            return
        model = self._composite_model

        try:
            parsed = self._parse_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        global_params = list(parsed["global"])
        local_params = list(parsed["local"])
        fixed_params = dict(parsed["fixed"])
        param_values = dict(parsed["values"])
        param_bounds = dict(parsed["bounds"])
        file_params = list(parsed.get("file", []))

        # Per-run initial values: parameter table → inherited single-fit seeds →
        # explicit Initial-values dialog entries (highest precedence).
        effective_values = self._effective_initial_values_by_run(parsed)

        # Build initial parameter sets for each dataset
        initial_params = {}
        for ds in self._datasets:
            run_number = int(ds.run_number)
            run_effective = effective_values.get(run_number, {})
            params = ParameterSet()
            for pname in model.param_names:
                min_val, max_val = param_bounds[pname]
                value = run_effective.get(pname, param_values[pname])

                # File-type parameters are pinned to the dataset's file value.
                if pname in file_params:
                    base_name, _index = split_parameter_name(pname)
                    file_value = _get_file_value_for_parameter(ds, base_name)
                    if file_value is not None:
                        value = file_value

                fixed = pname in fixed_params or pname in file_params
                params.add(
                    Parameter(
                        name=pname,
                        value=value,
                        min=min_val,
                        max=max_val,
                        fixed=fixed,
                    )
                )
            initial_params[run_number] = params

        # Run global fit in background thread
        self._result_text.setText("Fitting... This may take a moment for many datasets...")
        self._fit_btn.setEnabled(False)  # Disable button during fit

        # Clean up any existing thread
        if self._fit_thread is not None:
            self._fit_thread.quit()
            self._fit_thread.wait()

        # Create worker and thread
        self._fit_thread = QThread()
        self._fit_worker = GlobalFitWorker(
            self._fit_engine,
            self._datasets,
            self._composite_model.function,
            global_params,
            local_params,
            initial_params,
        )
        self._fit_worker.moveToThread(self._fit_thread)

        # Store model for later use in callbacks
        self._current_model = self._composite_model
        self._current_global_params = global_params

        # Connect signals
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._fit_thread.quit)
        self._fit_worker.error.connect(self._fit_thread.quit)
        self._fit_thread.finished.connect(self._cleanup_thread)

        # Start the thread. The started signal lets listeners snapshot
        # launch-time context (e.g. which frequency representation the
        # datasets came from) before any UI refresh can change it.
        self.global_fit_started.emit()
        self._fit_thread.start()

    def _run_grouped_time_domain_fit(self) -> None:
        """Execute grouped time-domain fitting for the active dataset."""
        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason
                or "Grouped time-domain fit is unavailable for the current selection."
            )
            return

        grouped_groups, grouped_datasets, message = self._grouped_mode_context()
        if grouped_groups is None or grouped_datasets is None:
            self._result_text.setText(message)
            return

        try:
            grouped_config = self._parse_grouped_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return
        grouped_model = self._grouped_fit_model()
        try:
            validate_grouped_model_contract(
                grouped_model.param_names,
                model_values=dict(grouped_config["model_values"]),
                fixed_params=set(grouped_config["fixed"]),
            )
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        global_params = list(grouped_config["global"])
        local_params = list(grouped_config["local"])

        # Multiple member runs → grouped *series* fit (batch across runs).
        if len(self._grouped_members) > 1:
            self._run_grouped_series_fit(
                grouped_datasets, grouped_model, global_params, local_params, grouped_config
            )
            return

        single_run = int(self._current_dataset.run_number) if self._current_dataset else None
        initial_params = self._build_grouped_initial_params(
            grouped_groups, grouped_config, run_number=single_run
        )

        self._result_text.setText("Fitting grouped time-domain data...")
        self._fit_btn.setEnabled(False)

        if self._fit_thread is not None:
            self._fit_thread.quit()
            self._fit_thread.wait()

        self._fit_thread = QThread()
        self._fit_worker = GroupedTimeDomainFitWorker(
            grouped_groups,
            grouped_datasets,
            grouped_model.function,
            global_params,
            local_params,
            initial_params,
        )
        self._fit_worker.moveToThread(self._fit_thread)
        self._current_model = grouped_model
        self._current_global_params = global_params

        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.finished.connect(self._on_grouped_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._fit_thread.quit)
        self._fit_worker.error.connect(self._fit_thread.quit)
        self._fit_thread.finished.connect(self._cleanup_thread)
        self._fit_thread.start()

    @staticmethod
    def _derive_grouped_relationship(
        physics_roles: dict[str, str],
        n_members: int,
    ) -> tuple[str | None, str | None]:
        """Derive the grouped-series relationship from the physics roles.

        Returns ``(relationship, error)``. ``relationship`` is ``individual`` (one
        member), ``global`` (≥1 physics param shared across runs), or ``batch``
        (physics independent per run). ``error`` is non-``None`` when the physics
        classification mixes Global and Local — the simultaneous engine can't
        express that (A1), so the caller must reject the fit.
        """
        physics_global = [name for name, role in physics_roles.items() if role == "global"]
        physics_local = [name for name, role in physics_roles.items() if role == "local"]
        if physics_global and physics_local:
            return None, (
                "Grouped series fits can't mix Global and Local fit-function parameters. "
                "Set them all Global (shared across runs) or all Local (independent per run). "
                f"Global: {', '.join(physics_global)} · Local: {', '.join(physics_local)}."
            )
        if n_members <= 1:
            return "individual", None
        return ("global" if physics_global else "batch"), None

    def _run_grouped_series_fit(
        self,
        grouped_datasets: list[MuonDataset],
        grouped_model: CompositeModel,
        global_params: list[str],
        local_params: list[str],
        grouped_config: dict[str, object],
    ) -> None:
        """Launch a multi-run grouped *series* fit via :func:`fit_grouped_series`.

        Relationship is derived: one member → ``individual``; several members →
        ``batch`` (physics independent per run; cross-run ``global`` arrives with
        the unified physics-role table in S3). Per-group seeds are replicated to
        every run (per-member seeds come from the seed dialog in S6).
        """
        members = dict(self._grouped_members)
        physics_roles = dict(grouped_config.get("physics_roles", {}))
        relationship, mixing_error = self._derive_grouped_relationship(physics_roles, len(members))
        if mixing_error:
            self._result_text.setText(mixing_error)
            self._fit_btn.setEnabled(True)
            return
        initial_params = {
            run: self._build_grouped_initial_params(groups, grouped_config, run_number=run)
            for run, groups in members.items()
        }

        self._result_text.setText("Fitting grouped time-domain series...")
        self._fit_btn.setEnabled(False)

        if self._fit_thread is not None:
            self._fit_thread.quit()
            self._fit_thread.wait()

        self._fit_thread = QThread()
        self._fit_worker = GroupedSeriesFitWorker(
            relationship,
            members,
            grouped_datasets,
            grouped_model.function,
            global_params,
            local_params,
            initial_params,
        )
        self._fit_worker.moveToThread(self._fit_thread)
        self._current_model = grouped_model
        self._current_global_params = global_params

        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.finished.connect(self._on_grouped_series_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._fit_thread.quit)
        self._fit_worker.error.connect(self._fit_thread.quit)
        self._fit_thread.finished.connect(self._cleanup_thread)
        self._fit_thread.start()

    def _on_grouped_series_fit_finished(self, grouped_datasets, series_result) -> None:
        """Handle a completed multi-run grouped-series fit (persist + plot).

        Builds per-(run,group) fit curves keyed by the synthetic member key and
        emits ``grouped_fit_completed`` → ``MainWindow._record_grouped_fit_series``
        persists the ``FitSeries(member_kind="groups")``. (Reflecting fitted values
        back into the per-group tables is deferred; the seeds remain shown.)
        """
        self._update_mode_ui(preserve_result=True)
        member_results = dict(getattr(series_result, "member_results", {}))
        source_run = dict(getattr(series_result, "member_source_run", {}))
        grouped_model = build_grouped_count_model(self._current_model.function)

        results_with_curves: dict[int, tuple] = {}
        for dataset in grouped_datasets:
            try:
                key = int(dataset.metadata.get("run_number"))
            except (TypeError, ValueError):
                continue
            fit_result = member_results.get(key)
            if fit_result is None:
                continue
            param_dict = {parameter.name: parameter.value for parameter in fit_result.parameters}
            for pname in self._grouped_fit_model().param_names:
                if is_amplitude_parameter(pname):
                    param_dict.setdefault(pname, 1.0)
            time_values = np.asarray(dataset.time, dtype=float)
            finite_mask = np.isfinite(time_values)
            if np.any(finite_mask):
                fit_t_min = float(np.min(time_values[finite_mask]))
                fit_t_max = float(np.max(time_values[finite_mask]))
            else:
                fit_t_min, fit_t_max = float(dataset.time.min()), float(dataset.time.max())
            n_samples = _fit_curve_sample_count(
                self._current_model, param_dict, fit_t_min, fit_t_max
            )
            t_fit = np.linspace(fit_t_min, fit_t_max, n_samples)
            y_fit = grouped_model(t_fit, **param_dict)
            results_with_curves[key] = (fit_result, (t_fit, y_fit), tuple())

        n_members = len(member_results)
        n_runs = len(set(source_run.values())) if source_run else 0
        reduced = [r.reduced_chi_squared for r in member_results.values() if r.reduced_chi_squared]
        avg_red_chi2 = sum(reduced) / len(reduced) if reduced else 0.0
        stats = f"{n_runs} runs · {n_members} group fits · avg χ²/ν = {avg_red_chi2:.4f}"
        self._results_group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)
        self._result_text.setHtml(success_html("Grouped series fit converged", detail=stats))
        self.grouped_fit_completed.emit(grouped_datasets, results_with_curves)

    def _on_preview_requested(self) -> None:
        """Preview grouped time-domain curves using the current parameter values."""
        if not self.is_grouped_time_domain_mode():
            self._result_text.setText(
                "Preview is currently available only in grouped time-domain mode."
            )
            return

        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason
                or "Grouped time-domain preview is unavailable for the current selection."
            )
            return

        grouped_groups, grouped_datasets, message = self._grouped_mode_context()
        if grouped_groups is None or grouped_datasets is None:
            self._result_text.setText(message)
            return

        try:
            grouped_config = self._parse_grouped_parameter_configuration()
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return
        grouped_model = self._grouped_fit_model()
        try:
            validate_grouped_model_contract(
                grouped_model.param_names,
                model_values=dict(grouped_config["model_values"]),
                fixed_params=set(grouped_config["fixed"]),
            )
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        preview_curves = self._build_grouped_preview_curves(
            grouped_groups=grouped_groups,
            grouped_datasets=grouped_datasets,
            grouped_config=grouped_config,
        )
        self._result_text.setText(
            f"Previewing grouped time-domain curves for {len(grouped_datasets)} groups."
        )
        self.grouped_preview_requested.emit(grouped_datasets, preview_curves)

    def _build_grouped_initial_params(
        self,
        grouped_groups: list[object],
        grouped_config: dict[str, object],
        run_number: int | None = None,
    ) -> dict[object, ParameterSet]:
        """Build grouped parameter seeds from the current UI state.

        When *run_number* is given, per-(run, group) nuisance overrides from the
        grouped Initial-values dialog take precedence over the per-group table.
        """
        initial_params: dict[object, ParameterSet] = {}
        nuisance_group_values = dict(grouped_config["group_values"])
        model_values = dict(grouped_config["model_values"])
        bounds = dict(grouped_config["bounds"])
        fixed = set(grouped_config["fixed"])

        for index, group in enumerate(grouped_groups, start=1):
            user_values: dict[str, float] = {}
            if run_number is not None:
                member_key = _group_dataset_run_number(int(run_number), index)
                user_values = self._user_grouped_initial_values.get(member_key, {})
            params = ParameterSet()
            for name in GROUP_NUISANCE_PARAMS:
                per_group_values = nuisance_group_values.get(name, {})
                value = float(per_group_values.get(group.group_id, 0.0))
                if name in user_values:
                    value = float(user_values[name])
                min_val, max_val = bounds[name]
                params.add(
                    Parameter(
                        name=name,
                        value=value,
                        min=min_val,
                        max=max_val,
                        fixed=name in fixed,
                    )
                )
            for name, value in model_values.items():
                min_val, max_val = bounds[name]
                params.add(
                    Parameter(
                        name=name,
                        value=value,
                        min=min_val,
                        max=max_val,
                        fixed=name in fixed,
                    )
                )
            initial_params[group.group_id] = params

        return initial_params

    def _build_grouped_preview_curves(
        self,
        *,
        grouped_groups: list[object],
        grouped_datasets: list[MuonDataset],
        grouped_config: dict[str, object],
    ) -> dict[int, tuple[object, tuple[np.ndarray, np.ndarray], tuple]]:
        """Build preview overlays for grouped time-domain mode."""
        initial_params = self._build_grouped_initial_params(grouped_groups, grouped_config)
        fit_model = self._grouped_fit_model()
        grouped_model = build_grouped_count_model(fit_model.function)
        preview_curves: dict[int, tuple[object, tuple[np.ndarray, np.ndarray], tuple]] = {}

        for group, dataset in zip(grouped_groups, grouped_datasets, strict=False):
            params = initial_params.get(group.group_id)
            if params is None:
                continue
            param_dict = {parameter.name: parameter.value for parameter in params}
            fit_time = np.asarray(getattr(group, "time", dataset.time), dtype=float)
            finite_mask = np.isfinite(fit_time)
            if not np.any(finite_mask):
                continue
            fit_t_min = float(np.min(fit_time[finite_mask]))
            fit_t_max = float(np.max(fit_time[finite_mask]))
            n_samples = _fit_curve_sample_count(
                fit_model,
                param_dict,
                fit_t_min,
                fit_t_max,
            )
            t_fit = np.linspace(fit_t_min, fit_t_max, n_samples)
            y_fit = grouped_model(t_fit, **param_dict)
            preview_curves[int(dataset.run_number)] = (object(), (t_fit, y_fit), tuple())

        return preview_curves

    def _render_global_fit_success(
        self,
        *,
        results_dict: dict[int, FitResult],
        fitted_global: ParameterSet,
        global_param_names: list[str],
    ) -> None:
        n_datasets = len(results_dict)
        avg_red_chi2 = sum(r.reduced_chi_squared for r in results_dict.values()) / n_datasets
        npar = len(global_param_names)
        stats = f"avg χ²/ν = {avg_red_chi2:.4f} · {n_datasets} datasets · npar = {npar}"
        self._results_group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)
        self._result_text.setHtml(success_html("Batch fit converged", detail=stats))

    def _results_with_curves(
        self,
        model: CompositeModel,
        results_dict: dict[int, FitResult],
    ) -> dict[
        int, tuple[FitResult, tuple[np.ndarray, np.ndarray], tuple[tuple[str, np.ndarray], ...]]
    ]:
        results_with_curves = {}
        for dataset in self._datasets:
            result = results_dict[int(dataset.run_number)]
            param_dict = {parameter.name: parameter.value for parameter in result.parameters}
            n_samples = _fit_curve_sample_count(
                model,
                param_dict,
                float(dataset.time.min()),
                float(dataset.time.max()),
            )
            t_fit = np.linspace(dataset.time.min(), dataset.time.max(), n_samples)
            y_fit = model.function(t_fit, **param_dict)
            component_curves = tuple(
                model.evaluate_components(
                    t_fit,
                    additive_only=True,
                    **param_dict,
                )
            )
            results_with_curves[int(dataset.run_number)] = (
                result,
                (t_fit, y_fit),
                component_curves,
            )
        return results_with_curves

    def _emit_global_fit_success(
        self,
        *,
        model: CompositeModel,
        results_dict: dict[int, FitResult],
        fitted_global: ParameterSet,
        global_param_names: list[str],
    ) -> None:
        self._render_global_fit_success(
            results_dict=results_dict,
            fitted_global=fitted_global,
            global_param_names=global_param_names,
        )
        emitted_results = results_dict
        emitted_global = fitted_global
        if self._domain == "frequency":
            emitted_results = {}
            for run_number, result in results_dict.items():
                params, uncertainties = append_frequency_field_derived_parameters(
                    result.parameters,
                    result.uncertainties,
                )
                emitted_results[run_number] = FitResult(
                    success=result.success,
                    chi_squared=result.chi_squared,
                    reduced_chi_squared=result.reduced_chi_squared,
                    parameters=params,
                    uncertainties=uncertainties,
                    covariance=result.covariance,
                    covariance_parameters=list(result.covariance_parameters),
                    residuals=result.residuals,
                    message=result.message,
                    function_calls=result.function_calls,
                    gradient_calls=result.gradient_calls,
                    hessian_calls=result.hessian_calls,
                    edm=result.edm,
                    covariance_accurate=result.covariance_accurate,
                )
            emitted_global, _global_unc = append_frequency_field_derived_parameters(
                fitted_global,
                {},
            )
        self.global_fit_completed.emit(
            self._results_with_curves(model, emitted_results),
            emitted_global,
        )

    def _apply_fit_wizard_assessment(
        self,
        assessment: GlobalCandidateAssessment,
        recommendation: GlobalFitWizardRecommendation,
    ) -> None:
        """Apply a global-fit wizard assessment back into the global tab."""
        if not isinstance(assessment, GlobalCandidateAssessment):
            return
        if not assessment.is_successful:
            self._result_text.setText("<b>Global Fit Wizard failed</b>")
            return
        try:
            parsed = self._parse_parameter_configuration()
        except ValueError:
            parsed = None
        if parsed is not None:
            log_text = (
                self._fit_wizard_window.current_log_text()
                if self._fit_wizard_window is not None
                else self._cached_wizard_log_text
            )
            self._cache_wizard_analysis(
                recommendation,
                signature=self._wizard_context_signature(parsed),
                log_text=log_text,
            )

        self._set_composite_model(assessment.template.model)
        role_by_name = {name: "Global" for name in assessment.global_param_names}
        role_by_name.update({name: "Local" for name in assessment.local_param_names})
        role_by_name.update(
            {
                parameter.name: parameter.recommended_role
                for parameter in assessment.parameter_recommendations
            }
        )

        representative_run = self._datasets[0].run_number if self._datasets else None
        representative_result = (
            assessment.fit_results_by_run.get(int(representative_run))
            if representative_run is not None
            else None
        )
        fitted_by_name = {
            parameter.name: parameter
            for parameter in (
                representative_result.parameters if representative_result is not None else []
            )
        }
        display_values = _normalized_model_param_values(
            self._composite_model,
            {
                parameter.name: parameter.value
                for parameter in (
                    representative_result.parameters if representative_result is not None else []
                )
            },
        )
        self._updating_fraction_values = True

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                continue

            type_combo = self._param_table.cellWidget(row, 2)
            if isinstance(type_combo, QComboBox):
                if pname in assessment.fixed_param_names:
                    type_combo.setCurrentText("Fixed")
                else:
                    type_combo.setCurrentText(role_by_name.get(pname, "Global"))

            value_item = self._param_table.item(row, 1)
            fitted = fitted_by_name.get(pname)
            if value_item is not None and fitted is not None:
                value_item.setText(f"{display_values.get(pname, fitted.value):.6g}")

            bounds_item = self._param_table.item(row, 3)
            if bounds_item is not None and fitted is not None:
                min_text = "-inf" if not np.isfinite(fitted.min) else f"{float(fitted.min):g}"
                max_text = "inf" if not np.isfinite(fitted.max) else f"{float(fitted.max):g}"
                bounds_item.setText(f"{min_text}, {max_text}")
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        self._current_model = assessment.template.model
        self._current_global_params = list(assessment.global_param_names)
        self._status_text_from_global_wizard(assessment, recommendation)
        self.global_fit_completed.emit(
            {
                run_number: (
                    result,
                    assessment.fitted_curves_by_run[run_number],
                    assessment.component_curves_by_run[run_number],
                )
                for run_number, result in assessment.fit_results_by_run.items()
            },
            assessment.global_parameters,
        )

    def _status_text_from_global_wizard(
        self,
        assessment: GlobalCandidateAssessment,
        recommendation: GlobalFitWizardRecommendation,
    ) -> None:
        lines = [
            f"<b>Global Fit Wizard — {assessment.template.title}</b>",
            f"<b>{recommendation.metric.value} = {assessment.metric_value(recommendation.metric):.4f}</b>",
            f"<b>Global:</b> {', '.join(assessment.global_param_names) or 'None'}",
            f"<b>Local:</b> {', '.join(assessment.local_param_names) or 'None'}",
        ]
        if assessment.series_warnings:
            lines.append("<br><b>Warnings:</b>")
            lines.extend(f"  {warning}" for warning in assessment.series_warnings)
        self._result_text.setHtml("<br>".join(lines))

    def _on_fit_finished(self, results_dict: dict, fitted_global: list) -> None:
        """Handle successful fit completion."""
        self._update_mode_ui(preserve_result=True)

        model = self._current_model
        global_params = self._current_global_params

        # Display results
        if all(r.success for r in results_dict.values()):
            self._emit_global_fit_success(
                model=model,
                results_dict=results_dict,
                fitted_global=fitted_global,
                global_param_names=global_params,
            )
        else:
            failed = [run for run, r in results_dict.items() if not r.success]
            run_label_by_number = {ds.run_number: ds.run_label for ds in self._datasets}
            failed_labels = [run_label_by_number.get(run, str(run)) for run in failed]
            self._results_group.setStyleSheet("")
            self._result_text.setText(
                f"<b>Batch fit failed</b><br>Failed datasets: {failed_labels}"
            )

    def _on_fit_error(self, error_msg: str) -> None:
        """Handle fit error."""
        self._update_mode_ui(preserve_result=True)
        self._results_group.setStyleSheet("")
        mode_label = "grouped fit" if self.is_grouped_time_domain_mode() else "global fit"
        self._result_text.setText(f"<b>Error during {mode_label}:</b><br>{error_msg}")

    def _cache_grouped_simulate_seed(self, grouped_result) -> None:
        """Cache a multi-group simulate seed from a converged grouped fit.

        Stores, keyed by the active run number, the shared normalised model,
        its base parameter values (amplitudes forced to 1, backgrounds to 0 —
        the grouped contract) and the per-group amplitude/phase/N0 specs, so the
        Generate Synthetic Run dialog can re-create the ring.
        """
        if self._current_dataset is None or getattr(self._current_dataset, "run", None) is None:
            return
        try:
            run_number = int(self._current_dataset.run_number)
        except (TypeError, ValueError):
            return
        from asymmetry.core.fitting.grouped_time_domain import normalize_to_grouped_contract
        from asymmetry.core.simulate import group_specs_from_grouped_fit

        model = self._grouped_fit_model()
        shared_values = {
            parameter.name: float(parameter.value)
            for parameter in getattr(grouped_result, "shared_parameters", [])
        }
        # Start from the shared model's defaults updated with the fitted shared
        # values, then apply the grouped contract (amplitude→1, background→0).
        base = {name: float(model.param_defaults.get(name, 0.0)) for name in model.param_names}
        base.update({k: v for k, v in shared_values.items() if k in base})
        base = normalize_to_grouped_contract(model.param_names, base)
        specs = group_specs_from_grouped_fit(grouped_result)
        if not specs:
            return
        self._grouped_simulate_seed[run_number] = {
            "model": model.to_dict(),
            "base_parameters": base,
            "specs": [
                {
                    "group_id": spec.group_id,
                    "amplitude": spec.amplitude,
                    "relative_phase": spec.relative_phase,
                    "n0_weight": spec.n0_weight,
                    "label": spec.label,
                }
                for spec in specs
            ],
        }

    def grouped_simulate_seed_for_run(self, run_number: int) -> dict | None:
        """Return the cached multi-group simulate seed for a run, if any."""
        try:
            return self._grouped_simulate_seed.get(int(run_number))
        except (TypeError, ValueError):
            return None

    def _on_grouped_fit_finished(self, grouped_datasets: list[MuonDataset], grouped_result) -> None:
        """Handle successful grouped fit completion."""
        self._update_mode_ui(preserve_result=True)
        self._cache_grouped_simulate_seed(grouped_result)

        results_with_curves: dict[int, tuple[FitResult, tuple[np.ndarray, np.ndarray], tuple]] = {}
        grouped_model = build_grouped_count_model(self._current_model.function)
        datasets_by_group_id = {
            dataset.metadata.get("group_id"): dataset for dataset in grouped_datasets
        }

        shared_values = {
            parameter.name: parameter.value
            for parameter in getattr(grouped_result, "shared_parameters", [])
        }
        display_shared_values = _normalized_model_param_values(
            self._grouped_fit_model(), shared_values
        )
        shared_by_name = {
            parameter.name: parameter
            for parameter in getattr(grouped_result, "shared_parameters", [])
        }
        group_fit_by_id = {
            str(group_id): fit_result
            for group_id, fit_result in getattr(grouped_result, "group_results", {}).items()
        }
        model_row_by_name = _param_table_rows_by_name(self._group_model_table)
        previous_model_signal_state = self._group_model_table.blockSignals(True)
        try:
            for pname, row in model_row_by_name.items():
                value_item = self._group_model_table.item(row, 1)
                fitted = shared_by_name.get(pname)
                if value_item is not None and pname in display_shared_values:
                    value_item.setText(f"{float(display_shared_values[pname]):.6g}")
                bounds_item = self._group_model_table.item(row, 3)
                if bounds_item is not None and fitted is not None:
                    min_text = "-inf" if not np.isfinite(fitted.min) else f"{float(fitted.min):g}"
                    max_text = "inf" if not np.isfinite(fitted.max) else f"{float(fitted.max):g}"
                    bounds_item.setText(f"{min_text}, {max_text}")
        finally:
            self._group_model_table.blockSignals(previous_model_signal_state)
        self._synchronize_grouped_model_fraction_rows()

        previous_group_signal_state = self._group_param_table.blockSignals(True)
        try:
            for row in range(self._group_param_table.rowCount()):
                name_item = self._group_param_table.item(row, 0)
                pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
                if not isinstance(pname, str):
                    continue
                for offset, entry in enumerate(self._group_param_value_column_entries(), start=1):
                    fit_result = group_fit_by_id.get(str(entry))
                    if fit_result is None:
                        continue
                    fitted_by_name = {
                        parameter.name: parameter for parameter in fit_result.parameters
                    }
                    fitted = fitted_by_name.get(pname)
                    value_item = self._group_param_table.item(row, offset)
                    if value_item is not None and fitted is not None:
                        value_item.setText(f"{float(fitted.value):.6g}")
                first_entry = str(self._group_param_value_column_entries()[0])
                first_fit = group_fit_by_id.get(first_entry)
                if first_fit is not None:
                    fitted_by_name = {
                        parameter.name: parameter for parameter in first_fit.parameters
                    }
                    fitted = fitted_by_name.get(pname)
                    bounds_item = self._group_param_table.item(
                        row, self._group_param_bounds_column()
                    )
                    if bounds_item is not None and fitted is not None:
                        min_text = (
                            "-inf" if not np.isfinite(fitted.min) else f"{float(fitted.min):g}"
                        )
                        max_text = (
                            "inf" if not np.isfinite(fitted.max) else f"{float(fitted.max):g}"
                        )
                        bounds_item.setText(f"{min_text}, {max_text}")
        finally:
            self._group_param_table.blockSignals(previous_group_signal_state)

        for group_id, fit_result in grouped_result.group_results.items():
            dataset = datasets_by_group_id.get(group_id)
            if dataset is None:
                continue
            param_dict = {parameter.name: parameter.value for parameter in fit_result.parameters}
            for pname in self._grouped_fit_model().param_names:
                if is_amplitude_parameter(pname):
                    param_dict.setdefault(pname, 1.0)
            fit_source_time = np.asarray(
                self._current_dataset.time if self._current_dataset is not None else dataset.time,
                dtype=float,
            )
            finite_mask = np.isfinite(fit_source_time)
            if np.any(finite_mask):
                fit_t_min = float(np.min(fit_source_time[finite_mask]))
                fit_t_max = float(np.max(fit_source_time[finite_mask]))
            else:
                fit_t_min = float(dataset.time.min())
                fit_t_max = float(dataset.time.max())
            n_samples = _fit_curve_sample_count(
                self._current_model,
                param_dict,
                fit_t_min,
                fit_t_max,
            )
            t_fit = np.linspace(fit_t_min, fit_t_max, n_samples)
            y_fit = grouped_model(t_fit, **param_dict)
            results_with_curves[int(dataset.run_number)] = (fit_result, (t_fit, y_fit), tuple())

        n_groups = len(grouped_result.group_results)
        group_results = list(grouped_result.group_results.values())
        avg_red_chi2 = (
            sum(r.reduced_chi_squared for r in group_results) / n_groups if n_groups > 0 else 0.0
        )
        n_shared = len(grouped_result.shared_parameters)
        stats = f"{n_groups} groups · avg χ²/ν = {avg_red_chi2:.4f} · shared = {n_shared}"
        self._results_group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)
        self._result_text.setHtml(success_html("Grouped fit converged", detail=stats))
        self.grouped_fit_completed.emit(grouped_datasets, results_with_curves)

    def _cleanup_thread(self) -> None:
        """Clean up thread resources."""
        if self._fit_thread is not None:
            self._fit_thread.deleteLater()
            self._fit_thread = None
        if self._fit_worker is not None:
            self._fit_worker.deleteLater()
            self._fit_worker = None

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the global-fit tab state."""
        if self._fit_wizard_window is not None:
            recommendation = self._fit_wizard_window.current_recommendation()
            signature = self._cached_wizard_signature
            if recommendation is not None and signature is None:
                try:
                    parsed = self._parse_parameter_configuration()
                except ValueError:
                    parsed = None
                if parsed is not None:
                    signature = self._wizard_context_signature(parsed)
            if recommendation is not None and signature is not None:
                self._cache_wizard_analysis(
                    recommendation,
                    signature=signature,
                    log_text=self._fit_wizard_window.current_log_text(),
                )
        params = []
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else f"param_{i}"
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()

            value_item = self._param_table.item(i, 1)
            try:
                value = float(value_item.text()) if value_item else 0.0
            except ValueError:
                value = 0.0

            type_combo = self._param_table.cellWidget(i, 2)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Local"

            bounds_item = self._param_table.item(i, 3)
            bounds_text = bounds_item.text() if bounds_item else "-inf, inf"

            params.append(
                {
                    "name": param_name,
                    "value": value,
                    "type": type_text,
                    "bounds": bounds_text,
                }
            )

        normalized_values = _normalized_model_param_values(
            self._composite_model,
            {str(entry["name"]): float(entry.get("value", 0.0)) for entry in params},
        )

        state = {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            "parameters": [
                {**entry, "value": normalized_values.get(str(entry["name"]), entry["value"])}
                for entry in params
            ],
            "result_html": self._result_text.toHtml(),
            "group_parameters": [
                {
                    "name": name,
                    "value": entry.get("value", 0.0),
                    "group_values": dict(entry.get("group_values", {})),
                    "type": entry.get("type", ""),
                    "bounds": entry.get("bounds", "-inf, inf"),
                }
                for name, entry in self._current_group_param_table_state().items()
            ],
            "group_model_parameters": self._table_state_for(self._group_model_table),
        }
        wizard_state_by_run_set = self._serialize_wizard_cache_store()
        if wizard_state_by_run_set:
            state["wizard_state_by_run_set"] = wizard_state_by_run_set
        if (
            self._cached_wizard_recommendation is not None
            and self._cached_wizard_signature is not None
        ):
            state["wizard_state"] = {
                "signature": copy.deepcopy(self._cached_wizard_signature),
                "recommendation": serialize_global_fit_wizard_recommendation(
                    self._cached_wizard_recommendation
                ),
                "log_text": self._cached_wizard_log_text,
            }
        return state

    def restore_state(self, state: dict) -> None:
        """Restore global-fit tab state from a saved dict."""
        self._wizard_cache_by_run_set = {}
        self._set_active_wizard_cache(None, signature=None, log_text="")

        composite_data = state.get("composite_model")
        if isinstance(composite_data, dict):
            try:
                self._set_composite_model(CompositeModel.from_dict(composite_data))
            except ValueError:
                self._set_composite_model(
                    CompositeModel(["Exponential", "Constant"], operators=["+"])
                )

        params_data = {p["name"]: p for p in state.get("parameters", [])}
        normalized_state_values = _normalized_model_param_values(
            self._composite_model,
            {
                str(name): float(entry.get("value", 0.0))
                for name, entry in params_data.items()
                if entry.get("value") is not None
            },
        )
        self._updating_fraction_values = True
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str) and name_item:
                param_name = name_item.text()
            if param_name not in params_data:
                continue

            p_data = params_data[param_name]

            value_item = self._param_table.item(i, 1)
            if value_item:
                value_item.setText(
                    str(normalized_state_values.get(param_name, p_data.get("value", 0.0)))
                )

            type_combo = self._param_table.cellWidget(i, 2)
            if isinstance(type_combo, QComboBox):
                type_value = str(p_data.get("type") or "Local")
                idx = type_combo.findText(type_value)
                if idx >= 0:
                    type_combo.setCurrentIndex(idx)

            bounds_item = self._param_table.item(i, 3)
            if bounds_item:
                bounds_item.setText(p_data.get("bounds", "-inf, inf"))
        _configure_fraction_rows_in_table(
            self._param_table,
            self._composite_model,
            bounds_column=3,
            type_column=2,
        )
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        self._restore_group_param_table_state(state.get("group_parameters"))
        self._restore_table_state(self._group_model_table, state.get("group_model_parameters"))
        _configure_fraction_rows_in_table(
            self._group_model_table,
            self._grouped_fit_model(),
            bounds_column=3,
            type_column=2,
        )
        self._synchronize_grouped_model_fraction_rows()

        result_html = state.get("result_html")
        if isinstance(result_html, str) and result_html:
            self._result_text.setHtml(result_html)

        wizard_state_by_run_set = state.get("wizard_state_by_run_set")
        if isinstance(wizard_state_by_run_set, list):
            self._restore_wizard_cache_store(wizard_state_by_run_set)

        wizard_state = state.get("wizard_state")
        if isinstance(wizard_state, dict):
            recommendation = deserialize_global_fit_wizard_recommendation(
                wizard_state.get("recommendation")
            )
            signature = wizard_state.get("signature")
            if recommendation is not None and isinstance(signature, dict):
                self._cache_wizard_analysis(
                    recommendation,
                    signature=signature,
                    log_text=str(wizard_state.get("log_text", "")),
                )
        self._sync_active_wizard_cache_from_selection()
        self._update_mode_ui(preserve_result=True)

    def is_grouped_time_domain_mode(self) -> bool:
        """Return whether this is the group-membered (Individual-groups) surface.

        Member kind is fixed per instance (derived from the active representation),
        so this is a constant rather than a user-selected mode.
        """
        return self._member_kind == "groups"

    def _update_mode_ui(self, *, preserve_result: bool) -> None:
        grouped = self.is_grouped_time_domain_mode()
        self._param_group.setVisible(not grouped)
        self._grouped_context_label.setVisible(grouped)
        self._group_param_group.setVisible(grouped)
        self._group_model_group.setVisible(grouped)
        self._fit_btn.setText("Run Grouped Fit" if grouped else "Run Batch Fit")
        self._preview_btn.setVisible(grouped)
        _set_formula_label_text(
            self._formula_label,
            (
                _grouped_formula_string(self._grouped_fit_model())
                if grouped
                else self._composite_model.formula_string()
            ),
        )

        if grouped:
            grouped_groups, grouped_datasets, message = self._grouped_mode_context()
            desired_group_specs = self._grouped_parameter_specs(grouped_groups)
            if desired_group_specs != self._group_param_group_specs:
                preserved_state = self._current_group_param_table_state()
                if grouped_groups and not self._group_param_group_specs:
                    self._reset_initial_group_nuisance_placeholders(preserved_state)
                self._rebuild_group_nuisance_table(
                    preserved_state,
                    grouped_groups=grouped_groups if grouped_groups is not None else [],
                )
            ready = grouped_groups is not None and grouped_datasets is not None
            self._grouped_context_label.setText(message)
            self._fit_btn.setEnabled(ready and (not self._fit_blocked))
            self._fit_btn.setToolTip(self._fit_block_reason if self._fit_blocked else message)
            self._preview_btn.setEnabled(ready and (not self._fit_blocked))
            self._preview_btn.setToolTip(self._fit_block_reason if self._fit_blocked else message)
            self._fit_wizard_btn.setEnabled(False)
            self._fit_wizard_btn.setToolTip(
                "Global Fit Wizard is unavailable in grouped time-domain mode."
            )
            if not preserve_result:
                self._result_text.setText(message)
            return

        n = len(self._datasets)
        self._fit_btn.setEnabled((n > 1) and (not self._fit_blocked))
        self._fit_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")
        self._preview_btn.setEnabled(False)
        self._preview_btn.setToolTip("Preview is available only in grouped time-domain mode.")
        wizard_enabled = (n > 1) and (not self._fit_blocked) and self._domain == "time"
        self._fit_wizard_btn.setEnabled(wizard_enabled)
        if self._domain == "frequency":
            self._fit_wizard_btn.setToolTip(
                "Global Fit Wizard is currently available for time-domain fits."
            )
        else:
            self._fit_wizard_btn.setToolTip(self._fit_block_reason if self._fit_blocked else "")
        if preserve_result:
            return
        if n == 0:
            self._result_text.setText(
                "No datasets selected.\nSelect datasets in the browser to run a global fit."
            )
        elif n == 1:
            self._result_text.setText(
                "Batch fitting requires at least 2 datasets.\nCurrently have 1 selected dataset."
            )
        else:
            domain_label = "frequency spectra" if self._domain == "frequency" else "datasets"
            self._result_text.setText(
                f"{n} {domain_label} selected. Configure parameters and click Run Batch Fit."
            )

    def _grouped_mode_context(
        self,
    ) -> tuple[list[object] | None, list[MuonDataset] | None, str]:
        """Return the grouped time-domain context, memoised per active dataset.

        Building the grouped groups/datasets is relatively expensive on
        high-resolution histograms and is requested several times per
        selection/UI refresh.  The result depends only on the active dataset and
        the fit-blocked state, so it is cached and reused until the active
        dataset changes (see :meth:`set_current_dataset`).
        """
        cache = getattr(self, "_grouped_context_cache", None)
        member_ids = tuple(id(ds) for ds in self._grouped_member_datasets())
        key = (member_ids, bool(self._fit_blocked))
        if cache is not None and cache[0] == key:
            return cache[1]
        result = self._compute_grouped_mode_context()
        self._grouped_context_cache = (key, result)
        return result

    def _compute_grouped_mode_context(
        self,
    ) -> tuple[list[object] | None, list[MuonDataset] | None, str]:
        """Build grouped groups/datasets for every member run.

        Populates :attr:`_grouped_members` (``run -> groups``) for the series fit
        and returns ``(representative_groups, flat_datasets, message)`` for the UI
        — ``representative_groups`` is the first member's group structure (the
        per-group tables assume a consistent grouping across runs) and
        ``flat_datasets`` concatenates every member's grouped traces. A single
        member reduces to the previous single-run behaviour.
        """
        if self._fit_blocked:
            self._grouped_members = {}
            return None, None, self._fit_block_reason or "Grouped fitting is unavailable."

        member_datasets = self._grouped_member_datasets()
        if not member_datasets:
            self._grouped_members = {}
            return (
                None,
                None,
                "Grouped time-domain mode requires an active dataset in the "
                "FB Asymmetry or Individual Groups workspace.",
            )

        members: dict[int, list[object]] = {}
        grouped_datasets: list[MuonDataset] = []
        representative_groups: list[object] | None = None
        first_label = ""
        skipped: list[str] = []
        for dataset in member_datasets:
            if dataset is None or np.asarray(dataset.time).size == 0:
                continue
            time_values = np.asarray(dataset.time, dtype=float)
            finite_mask = np.isfinite(time_values)
            fit_t_min = float(np.min(time_values[finite_mask])) if np.any(finite_mask) else None
            fit_t_max = float(np.max(time_values[finite_mask])) if np.any(finite_mask) else None
            run_label = getattr(dataset, "run_label", str(dataset.run_number))
            try:
                groups = build_grouped_time_domain_groups(dataset, t_min=fit_t_min, t_max=fit_t_max)
                datasets = build_grouped_time_domain_datasets(dataset)
            except ValueError as exc:
                skipped.append(f"{run_label}: {exc}")
                continue
            members[int(dataset.run_number)] = groups
            grouped_datasets.extend(datasets)
            if representative_groups is None:
                representative_groups = groups
                first_label = run_label

        self._grouped_members = members
        if representative_groups is None or not grouped_datasets:
            reason = (
                "; ".join(skipped)
                if skipped
                else "Grouped time-domain mode requires a non-empty active dataset."
            )
            return None, None, reason

        n_runs = len(members)
        if n_runs == 1:
            message = (
                f"{len(grouped_datasets)} grouped traces from {first_label} are ready for fitting."
            )
        else:
            message = (
                f"{len(grouped_datasets)} grouped traces from {n_runs} runs are ready for fitting."
            )
        if skipped:
            message += f" (skipped {len(skipped)}: {'; '.join(skipped)})"
        return representative_groups, grouped_datasets, message

    def _setup_group_nuisance_table(self) -> None:
        self._rebuild_group_nuisance_table(preserved_state=None)

    def _grouped_parameter_specs(
        self,
        grouped_groups: list[object] | None = None,
    ) -> list[tuple[object, str]]:
        if grouped_groups is None:
            grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        if not grouped_groups:
            return []
        specs: list[tuple[object, str]] = []
        for index, group in enumerate(grouped_groups, start=1):
            group_id = getattr(group, "group_id", index)
            group_name = str(getattr(group, "group_name", f"Group {group_id}"))
            specs.append((group_id, group_name))
        return specs

    def _group_param_value_column_count(self) -> int:
        return max(1, len(self._group_param_group_specs))

    def _group_param_type_column(self) -> int:
        return 1 + self._group_param_value_column_count()

    def _group_param_bounds_column(self) -> int:
        return self._group_param_type_column() + 1

    def _group_param_value_column_entries(self) -> list[object]:
        if self._group_param_group_specs:
            return [group_id for group_id, _name in self._group_param_group_specs]
        return ["default"]

    def _current_group_param_table_state(self) -> dict[str, dict[str, object]]:
        state: dict[str, dict[str, object]] = {}
        value_entries = self._group_param_value_column_entries()
        type_column = self._group_param_type_column()
        bounds_column = self._group_param_bounds_column()
        for row in range(self._group_param_table.rowCount()):
            name_item = self._group_param_table.item(row, 0)
            if name_item is None:
                continue
            param_name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(param_name, str):
                continue
            group_values: dict[str, str] = {}
            fallback_value = ""
            for offset, entry in enumerate(value_entries, start=1):
                value_item = self._group_param_table.item(row, offset)
                value_text = value_item.text() if value_item is not None else ""
                group_values[str(entry)] = value_text
                if offset == 1:
                    fallback_value = value_text
            type_combo = self._group_param_table.cellWidget(row, type_column)
            bounds_item = self._group_param_table.item(row, bounds_column)
            state[param_name] = {
                "value": fallback_value,
                "group_values": group_values,
                "bounds": bounds_item.text() if bounds_item is not None else "-inf, inf",
                "type": type_combo.currentText() if isinstance(type_combo, QComboBox) else "",
            }
        return state

    def _rebuild_group_nuisance_table(
        self,
        preserved_state: dict[str, dict[str, object]] | None,
        *,
        grouped_groups: list[object] | None = None,
    ) -> None:
        defaults = {
            "N0": (100.0, "Local", "0, inf"),
            "background": (0.0, "Local", "0, inf"),
            "amplitude": (0.2, "Local", "-1, 1"),
            "relative_phase": (0.0, "Local", f"{-np.pi:.6g}, {np.pi:.6g}"),
        }
        grouped_groups = grouped_groups or []
        self._group_param_group_specs = self._grouped_parameter_specs(grouped_groups)
        value_headers = [name for _group_id, name in self._group_param_group_specs] or ["Value"]
        column_count = 1 + len(value_headers) + 2
        previous_signal_state = self._group_param_table.blockSignals(True)
        previous_rows = self._group_param_table.rowCount()
        previous_columns = self._group_param_table.columnCount()
        for row in range(previous_rows):
            for column in range(previous_columns):
                if self._group_param_table.cellWidget(row, column) is not None:
                    self._group_param_table.removeCellWidget(row, column)
        self._group_param_table.clearContents()
        self._group_param_table.setColumnCount(column_count)
        self._group_param_table.setHorizontalHeaderLabels(
            ["Parameter", *value_headers, "Type", "Bounds"]
        )
        self._group_param_table.setColumnWidth(0, 110)
        for offset in range(len(value_headers)):
            self._group_param_table.setColumnWidth(1 + offset, 90)
        self._group_param_table.setColumnWidth(self._group_param_type_column(), 100)
        self._group_param_table.setColumnWidth(self._group_param_bounds_column(), 150)
        self._group_param_table.setRowCount(len(GROUP_NUISANCE_PARAMS))

        n0_defaults_by_group: dict[str, float] = {}
        background_defaults_by_group: dict[str, float] = {}
        amplitude_defaults_by_group: dict[str, float] = {}
        relative_phase_defaults_by_group = _seed_group_relative_phases(grouped_groups)
        for group in grouped_groups:
            counts = np.asarray(getattr(group, "counts", []), dtype=float)
            if counts.size == 0:
                continue
            group_id = getattr(group, "group_id", None)
            background_default, n0_default, amplitude_default = _seed_group_background_and_n0(
                counts,
                time=getattr(group, "time", None),
            )
            n0_defaults_by_group[str(group_id)] = n0_default
            background_defaults_by_group[str(group_id)] = background_default
            amplitude_defaults_by_group[str(group_id)] = amplitude_default

        for row, name in enumerate(GROUP_NUISANCE_PARAMS):
            label_item = _make_param_name_item(_format_param_label(name), name)
            self._group_param_table.setItem(row, 0, label_item)

            previous = preserved_state.get(name, {}) if isinstance(preserved_state, dict) else {}
            default_value, type_text, bounds = defaults[name]
            previous_group_values = previous.get("group_values")
            if not isinstance(previous_group_values, dict):
                previous_group_values = {}
            previous_value = previous.get("value")
            fallback_value = str(previous_value) if previous_value not in (None, "") else ""
            for offset, entry in enumerate(self._group_param_value_column_entries(), start=1):
                entry_key = str(entry)
                entry_default = (
                    n0_defaults_by_group.get(entry_key, default_value)
                    if name == "N0"
                    else (
                        background_defaults_by_group.get(entry_key, default_value)
                        if name == "background"
                        else (
                            amplitude_defaults_by_group.get(entry_key, default_value)
                            if name == "amplitude"
                            else (
                                relative_phase_defaults_by_group.get(entry_key, default_value)
                                if name == "relative_phase"
                                else default_value
                            )
                        )
                    )
                )
                value_text = str(previous_group_values.get(entry_key, fallback_value))
                if not value_text:
                    value_text = f"{entry_default:.6g}"
                self._group_param_table.setItem(row, offset, QTableWidgetItem(value_text))

            type_combo = QComboBox()
            type_combo.addItems(["Global", "Local", "Fixed"])
            type_combo.setCurrentText(str(previous.get("type") or type_text))
            type_combo.currentTextChanged.connect(
                lambda _text, row=row: self._on_group_param_type_changed(row)
            )
            self._group_param_table.setCellWidget(row, self._group_param_type_column(), type_combo)
            self._group_param_table.setItem(
                row,
                self._group_param_bounds_column(),
                QTableWidgetItem(str(previous.get("bounds") or bounds)),
            )
            self._sync_group_param_row_values(row)
        self._group_param_table.blockSignals(previous_signal_state)

    def _sync_group_param_row_values(self, row: int, edited_column: int | None = None) -> None:
        if self._updating_group_param_values:
            return
        type_combo = self._group_param_table.cellWidget(row, self._group_param_type_column())
        if not isinstance(type_combo, QComboBox):
            return
        if type_combo.currentText() == "Local":
            return
        source_column = 1 if edited_column is None or edited_column < 1 else edited_column
        source_item = self._group_param_table.item(row, source_column)
        source_text = source_item.text() if source_item is not None else ""
        self._updating_group_param_values = True
        try:
            for column in range(1, 1 + self._group_param_value_column_count()):
                if column == source_column:
                    continue
                item = self._group_param_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem(source_text)
                    self._group_param_table.setItem(row, column, item)
                else:
                    item.setText(source_text)
        finally:
            self._updating_group_param_values = False

    def _on_group_param_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_group_param_values:
            return
        if item.column() < 1 or item.column() >= self._group_param_type_column():
            return
        self._sync_group_param_row_values(item.row(), edited_column=item.column())

    def _on_group_param_type_changed(self, row: int) -> None:
        self._sync_group_param_row_values(row)

    def _current_grouped_model_row_state(self) -> dict[str, dict[str, str]]:
        return self._table_state_map(self._group_model_table)

    def _rebuild_grouped_model_table(self, preserved_state: dict[str, dict[str, str]]) -> None:
        grouped_model = self._grouped_fit_model()
        grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        phase_defaults = _grouped_model_phase_defaults(grouped_model, grouped_groups or [])
        visible_param_names = [
            pname for pname in grouped_model.param_names if not is_amplitude_parameter(pname)
        ]
        self._updating_group_model_fraction_values = True
        self._group_model_table.setRowCount(len(visible_param_names))
        for row, pname in enumerate(visible_param_names):
            previous = preserved_state.get(pname, {})
            name_item = _make_param_name_item(_format_param_label(pname), pname)
            self._group_model_table.setItem(row, 0, name_item)

            default_val = grouped_model.param_defaults.get(pname, 0.0)
            default_type = "Global"
            if is_background_parameter(pname):
                default_val = 0.0
                default_type = "Fixed"
            elif pname in phase_defaults:
                default_val = phase_defaults[pname]
            self._group_model_table.setItem(
                row,
                1,
                QTableWidgetItem(str(previous.get("value") or default_val)),
            )
            type_combo = QComboBox()
            # Physics roles unified with the run-batch side: Global = shared across
            # runs (members), Local = per run (still shared across that run's groups),
            # Fixed = held. Legacy "Shared"/"Free" map to Global.
            type_combo.addItems(["Global", "Local", "Fixed"])
            previous_type = str(previous.get("type") or default_type)
            if previous_type in ("Shared", "Free"):
                previous_type = "Global"
            type_combo.setCurrentText(previous_type)
            self._group_model_table.setCellWidget(row, 2, type_combo)
            default_min = get_param_info(pname).default_min
            min_text = str(default_min) if default_min is not None else "-inf"
            bounds_text = str(previous.get("bounds") or f"{min_text}, inf")
            self._group_model_table.setItem(row, 3, QTableWidgetItem(bounds_text))
        _configure_fraction_rows_in_table(
            self._group_model_table,
            grouped_model,
            bounds_column=3,
            type_column=2,
        )
        self._updating_group_model_fraction_values = False
        self._synchronize_grouped_model_fraction_rows()

    def _parse_grouped_parameter_configuration(self) -> dict[str, object]:
        global_params: list[str] = []
        local_params: list[str] = []
        fixed_params: list[str] = []
        model_values: dict[str, float] = {}
        group_values: dict[str, dict[object, float]] = {}
        bounds: dict[str, tuple[float, float]] = {}
        # Cross-run roles of the fit-function (physics) parameters; drives the
        # derived relationship (global/batch) for a multi-run grouped series.
        physics_roles: dict[str, str] = {}

        group_value_entries = self._group_param_value_column_entries()
        group_type_column = self._group_param_type_column()
        group_bounds_column = self._group_param_bounds_column()
        actual_group_ids = [group_id for group_id, _name in self._group_param_group_specs]
        for row in range(self._group_param_table.rowCount()):
            name_item = self._group_param_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else f"group_param_{row}"

            row_values: dict[object, float] = {}
            first_value: float | None = None
            for offset, entry in enumerate(group_value_entries, start=1):
                try:
                    value = float(self._group_param_table.item(row, offset).text())
                except (TypeError, ValueError, AttributeError):
                    raise ValueError(
                        f"Error: Invalid value for {_format_param_label(pname)}"
                    ) from None
                if not np.isfinite(value):
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                    )
                if first_value is None:
                    first_value = value
                row_values[entry] = value

            bounds_text = self._group_param_table.item(row, group_bounds_column).text()
            try:
                lo_text, hi_text = [part.strip() for part in bounds_text.split(",", maxsplit=1)]
                min_val = float(lo_text) if lo_text != "-inf" else -float("inf")
                max_val = float(hi_text) if hi_text != "inf" else float("inf")
            except (TypeError, ValueError):
                min_val, max_val = -float("inf"), float("inf")

            for value in row_values.values():
                if np.isfinite(min_val) and value < min_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} is below minimum {min_val}"
                    )
                if np.isfinite(max_val) and value > max_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} is above maximum {max_val}"
                    )

            type_combo = self._group_param_table.cellWidget(row, group_type_column)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Local"
            if type_text == "Global":
                global_params.append(pname)
            elif type_text == "Local":
                local_params.append(pname)
            else:
                fixed_params.append(pname)

            if type_text == "Local" and actual_group_ids:
                group_values[pname] = {
                    group_id: row_values.get(
                        group_id, first_value if first_value is not None else 0.0
                    )
                    for group_id in actual_group_ids
                }
            else:
                shared_value = first_value if first_value is not None else 0.0
                target_ids = actual_group_ids or list(row_values.keys())
                group_values[pname] = {group_id: shared_value for group_id in target_ids}
            bounds[pname] = (min_val, max_val)

        for row in range(self._group_model_table.rowCount()):
            name_item = self._group_model_table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str):
                pname = name_item.text() if name_item else f"model_param_{row}"

            try:
                value = float(self._group_model_table.item(row, 1).text())
            except (TypeError, ValueError, AttributeError):
                raise ValueError(f"Error: Invalid value for {_format_param_label(pname)}") from None
            if not np.isfinite(value):
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                )

            bounds_text = self._group_model_table.item(row, 3).text()
            try:
                lo_text, hi_text = [part.strip() for part in bounds_text.split(",", maxsplit=1)]
                min_val = float(lo_text) if lo_text != "-inf" else -float("inf")
                max_val = float(hi_text) if hi_text != "inf" else float("inf")
            except (TypeError, ValueError):
                min_val, max_val = -float("inf"), float("inf")

            if np.isfinite(min_val) and value < min_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} value {value} is below minimum {min_val}"
                )
            if np.isfinite(max_val) and value > max_val:
                raise ValueError(
                    f"Error: Parameter {_format_param_label(pname)} value {value} is above maximum {max_val}"
                )

            type_combo = self._group_model_table.cellWidget(row, 2)
            type_text = type_combo.currentText() if isinstance(type_combo, QComboBox) else "Global"
            if type_text == "Fixed":
                fixed_params.append(pname)
                physics_roles[pname] = "fixed"
            elif type_text == "Local":
                # Free, shared across a run's groups, independent across runs.
                global_params.append(pname)
                physics_roles[pname] = "local"
            else:  # "Global" (and legacy "Shared"/"Free")
                global_params.append(pname)
                physics_roles[pname] = "global"

            model_values[pname] = value
            bounds[pname] = (min_val, max_val)

        grouped_model = self._grouped_fit_model()
        for pname in grouped_model.param_names:
            if pname in model_values:
                continue
            if is_amplitude_parameter(pname):
                model_values[pname] = 1.0
                bounds[pname] = (1.0, 1.0)
                if pname not in fixed_params:
                    fixed_params.append(pname)

        return {
            "global": global_params,
            "local": local_params,
            "fixed": fixed_params,
            "group_values": group_values,
            "model_values": model_values,
            "bounds": bounds,
            "physics_roles": physics_roles,
        }

    def _table_state_map(self, table: QTableWidget) -> dict[str, dict[str, str]]:
        state: dict[str, dict[str, str]] = {}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            if name_item is None:
                continue
            param_name = name_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(param_name, str):
                continue
            value_item = table.item(row, 1)
            bounds_item = table.item(row, 3)
            type_combo = table.cellWidget(row, 2)
            state[param_name] = {
                "value": value_item.text() if value_item is not None else "",
                "bounds": bounds_item.text() if bounds_item is not None else "-inf, inf",
                "type": type_combo.currentText() if isinstance(type_combo, QComboBox) else "",
            }
        return state

    def _table_state_for(self, table: QTableWidget) -> list[dict[str, object]]:
        state: list[dict[str, object]] = []
        for name, entry in self._table_state_map(table).items():
            try:
                value = float(entry.get("value", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            state.append(
                {
                    "name": name,
                    "value": value,
                    "type": entry.get("type", ""),
                    "bounds": entry.get("bounds", "-inf, inf"),
                }
            )
        return state

    def _restore_table_state(self, table: QTableWidget, payload: object) -> None:
        if not isinstance(payload, list):
            return
        by_name = {str(entry.get("name")): entry for entry in payload if isinstance(entry, dict)}
        for row in range(table.rowCount()):
            name_item = table.item(row, 0)
            pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(pname, str) or pname not in by_name:
                continue
            entry = by_name[pname]
            value_item = table.item(row, 1)
            if value_item is not None:
                value_item.setText(str(entry.get("value", 0.0)))
            type_combo = table.cellWidget(row, 2)
            if isinstance(type_combo, QComboBox):
                idx = type_combo.findText(str(entry.get("type", "")))
                if idx >= 0:
                    type_combo.setCurrentIndex(idx)
            bounds_item = table.item(row, 3)
            if bounds_item is not None:
                bounds_item.setText(str(entry.get("bounds", "-inf, inf")))

    def _restore_group_param_table_state(self, payload: object) -> None:
        if not isinstance(payload, list):
            return
        by_name = {str(entry.get("name")): entry for entry in payload if isinstance(entry, dict)}
        type_column = self._group_param_type_column()
        bounds_column = self._group_param_bounds_column()
        self._updating_group_param_values = True
        try:
            for row in range(self._group_param_table.rowCount()):
                name_item = self._group_param_table.item(row, 0)
                pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(pname, str) or pname not in by_name:
                    continue
                entry = by_name[pname]
                raw_group_values = entry.get("group_values")
                group_values = raw_group_values if isinstance(raw_group_values, dict) else {}
                fallback_value = str(entry.get("value", 0.0))
                for offset, group_entry in enumerate(
                    self._group_param_value_column_entries(), start=1
                ):
                    value_item = self._group_param_table.item(row, offset)
                    if value_item is None:
                        continue
                    value_item.setText(str(group_values.get(str(group_entry), fallback_value)))
                type_combo = self._group_param_table.cellWidget(row, type_column)
                if isinstance(type_combo, QComboBox):
                    idx = type_combo.findText(str(entry.get("type", "")))
                    if idx >= 0:
                        type_combo.setCurrentIndex(idx)
                bounds_item = self._group_param_table.item(row, bounds_column)
                if bounds_item is not None:
                    bounds_item.setText(str(entry.get("bounds", "-inf, inf")))
                self._sync_group_param_row_values(row)
        finally:
            self._updating_group_param_values = False

    def _update_group_parameter_defaults(self) -> None:
        had_group_columns = bool(self._group_param_group_specs)
        grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        preserved_state = self._current_group_param_table_state()
        if grouped_groups and not had_group_columns:
            self._reset_initial_group_nuisance_placeholders(preserved_state)
        self._rebuild_group_nuisance_table(
            preserved_state,
            grouped_groups=grouped_groups if grouped_groups is not None else [],
        )

    def _reset_initial_group_nuisance_placeholders(
        self,
        preserved_state: dict[str, dict[str, object]],
    ) -> None:
        self._clear_group_parameter_value_placeholders(preserved_state, ("N0", "background"))

    def _clear_group_parameter_value_placeholders(
        self,
        preserved_state: dict[str, dict[str, object]],
        param_names: tuple[str, ...] | list[str],
    ) -> None:
        for pname in param_names:
            state = preserved_state.get(pname)
            if not isinstance(state, dict):
                continue
            updated_state = dict(state)
            updated_state["value"] = ""
            updated_state["group_values"] = {}
            preserved_state[pname] = updated_state

    def _reset_group_parameter_estimates(self) -> None:
        grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        preserved_state = self._current_group_param_table_state()
        self._clear_group_parameter_value_placeholders(preserved_state, GROUP_NUISANCE_PARAMS)
        self._rebuild_group_nuisance_table(
            preserved_state,
            grouped_groups=grouped_groups if grouped_groups is not None else [],
        )


class FitPanel(QWidget):
    """Fit setup and results panel with tabbed interface.

    Contains tabs for single dataset fitting and global (multi-dataset) fitting.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    preview_requested = Signal(
        object, object, object
    )  # (preview_result, fitted_curve, component_curves)
    # Keep payload generic to preserve Python dict key/value types end-to-end.
    global_fit_started = Signal()  # forwarded from GlobalFitTab at worker launch
    global_fit_completed = Signal(object, object)  # (results_dict, global_params)
    grouped_fit_completed = Signal(object, object)  # (grouped_datasets, results_dict)
    share_function_with_group_requested = Signal(int)
    add_single_fit_to_series_requested = Signal()
    fit_range_edit_committed = Signal(float, float)  # forwarded from SingleFitTab

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._single_state_by_run: dict[int, dict] = {}
        self._active_single_run_number: int | None = None
        self._all_datasets: list[MuonDataset] = []  # Track all datasets for group sharing
        self._domain = "time"
        self._single_state_by_domain: dict[str, dict] = {}
        self._global_state_by_domain: dict[str, dict] = {}
        self._ui_state_by_domain: dict[str, dict] = {}

        # Create tab widget
        self._tabs = QTabWidget()

        # Single fit tab
        self._single_tab = SingleFitTab()
        self._single_tab.fit_completed.connect(self._on_single_fit_completed)
        self._single_tab.preview_requested.connect(self.preview_requested.emit)
        self._single_tab.share_function_with_group_requested.connect(
            self.share_function_with_group_requested.emit
        )
        self._single_tab.send_model_to_batch_requested.connect(self._on_send_model_to_batch)
        self._single_tab.add_to_series_requested.connect(
            self.add_single_fit_to_series_requested.emit
        )
        self._single_tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
        self._tabs.addTab(self._single_tab, "Single")

        # Batch fit tab (a global fit is the special case with shared parameters)
        self._global_tab = GlobalFitTab(member_kind="runs")
        self._global_tab.global_fit_started.connect(self.global_fit_started.emit)
        self._global_tab.global_fit_completed.connect(self.global_fit_completed.emit)
        self._global_tab.grouped_fit_completed.connect(self.grouped_fit_completed.emit)
        self._global_tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
        self._tabs.addTab(self._global_tab, "Batch")

        layout.addWidget(self._tabs)

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_domain(self, domain: str) -> None:
        """Switch the fit panel between time- and frequency-domain workflows."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            self._single_tab.set_domain(normalized)
            self._global_tab.set_domain(normalized)
            return

        old_domain = self._domain
        self._single_state_by_domain[old_domain] = self.get_single_state()
        self._global_state_by_domain[old_domain] = self.get_global_state()
        self._ui_state_by_domain[old_domain] = self.get_ui_state()

        self._domain = normalized
        self._single_state_by_run = {}
        self._active_single_run_number = None
        self._single_tab.set_domain(normalized)
        self._global_tab.set_domain(normalized)

        if normalized in self._single_state_by_domain:
            self.restore_single_state(self._single_state_by_domain[normalized])
        if normalized in self._global_state_by_domain:
            self.restore_global_state(self._global_state_by_domain[normalized])
        if normalized in self._ui_state_by_domain:
            self.restore_ui_state(self._ui_state_by_domain[normalized])

    def clear(self) -> None:
        """Reset all fit-panel domain state."""
        self._single_state_by_domain = {}
        self._global_state_by_domain = {}
        self._ui_state_by_domain = {}
        self._single_state_by_run = {}
        self._active_single_run_number = None
        self._all_datasets = []
        self._domain = "time"
        self._single_tab.set_domain("time")
        self._global_tab.set_domain("time")
        self._single_tab.set_dataset(None)
        self._global_tab.set_datasets([])
        self._global_tab.set_current_dataset(None)
        self._tabs.setCurrentIndex(0)

    def _on_single_fit_completed(self, fit_result, fitted_curve, component_curves) -> None:
        """Forward single-fit completion and cache seeds for global fitting."""
        dataset = self._single_tab._current_dataset
        if dataset is not None:
            run_number = int(dataset.run_number)
            self._global_tab.register_single_fit_seed(
                run_number,
                self._single_tab._composite_model,
                fit_result,
            )
            # Keep most recent tab state per run (parameters, function, and result text).
            self._single_state_by_run[run_number] = self._single_tab.get_state()
        self.fit_completed.emit(fit_result, fitted_curve, component_curves)

    def _run_number_from_dataset(self, dataset: MuonDataset | None) -> int | None:
        if dataset is None:
            return None
        try:
            return int(dataset.run_number)
        except (TypeError, ValueError):
            return None

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset for single fitting tab."""
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        self._single_tab.set_dataset(dataset)
        self._global_tab.set_current_dataset(dataset)

        run_number = self._run_number_from_dataset(dataset)
        self._active_single_run_number = run_number

        if run_number is None:
            return

        if run_number in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[run_number])
        else:
            # Unseen datasets should not inherit another run's fit UI/result state.
            default_model = (
                default_frequency_model()
                if self._domain == "frequency"
                else CompositeModel(["Exponential", "Constant"], operators=["+"])
            )
            self._single_tab._set_composite_model(default_model)
            self._single_tab._result_label.setText("No fit performed yet")

    def set_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the datasets for global fitting tab and track for group sharing."""
        self._all_datasets = datasets
        self._global_tab.set_datasets(datasets)

    def batch_datasets(self) -> list[MuonDataset]:
        """Return the datasets configured for the batch/integral-scan."""
        return self._global_tab.batch_datasets()

    def set_frequency_missing_spectra_status(
        self, missing_run_numbers: list[int], cached_count: int
    ) -> None:
        """Show frequency-domain global fit status for selected uncached runs."""
        self._global_tab.set_frequency_missing_spectra_status(missing_run_numbers, cached_count)

    def is_grouped_time_domain_mode(self) -> bool:
        """Return whether the global tab is in grouped time-domain mode."""
        return self._global_tab.is_grouped_time_domain_mode()

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit-action blocking to both single and global tabs."""
        self._single_tab.set_fit_blocked(blocked, reason)
        self._global_tab.set_fit_blocked(blocked, reason)

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Forward fit-range display update to both single and global tabs."""
        self._single_tab.set_fit_range_display(x_min, x_max)
        self._global_tab.set_fit_range_display(x_min, x_max)

    def single_fit_formula_string(self) -> str | None:
        """Return the active single-fit formula string, if available."""
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def global_fit_formula_string(self) -> str | None:
        """Return the active global-fit formula string, if available."""
        model = getattr(self._global_tab, "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def clear_fits_for_runs(self, run_numbers: list[int]) -> int:
        """Clear cached single/global fit state for specific dataset runs."""
        normalized_runs: set[int] = set()
        for run_number in run_numbers:
            try:
                normalized_runs.add(int(run_number))
            except (TypeError, ValueError):
                continue

        if not normalized_runs:
            return 0

        changed_runs: set[int] = set()
        for run_number in normalized_runs:
            if self._single_state_by_run.pop(run_number, None) is not None:
                changed_runs.add(run_number)

        changed_runs |= self._global_tab.remove_single_fit_seeds(normalized_runs)

        active_run = self._active_single_run_number
        if active_run is not None and active_run in normalized_runs:
            self._single_tab._result_label.setText("No fit performed yet")

        return len(changed_runs)

    def get_single_state_for_run(self, run_number: int) -> dict | None:
        """Return current single-fit state for one run, if available."""
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return None

        if self._active_single_run_number == run_key:
            state = self._single_tab.get_state()
            self._single_state_by_run[run_key] = state
            return copy.deepcopy(state)

        state = self._single_state_by_run.get(run_key)
        if isinstance(state, dict):
            return copy.deepcopy(state)
        return None

    def get_single_fit_wizard_cache_for_run(
        self,
        run_number: int,
    ) -> tuple[FitWizardRecommendation | None, dict[str, object] | None, str]:
        state = self.get_single_state_for_run(run_number)
        if not isinstance(state, dict):
            return None, None, ""
        wizard_state = state.get("wizard_state")
        if not isinstance(wizard_state, dict):
            return None, None, ""
        recommendation = deserialize_fit_wizard_recommendation(wizard_state.get("recommendation"))
        signature = wizard_state.get("signature")
        log_text = str(wizard_state.get("log_text", ""))
        return (
            recommendation,
            signature if isinstance(signature, dict) else None,
            log_text,
        )

    def persist_single_fit_wizard_cache_for_run(
        self,
        run_number: int,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
    ) -> None:
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return

        active_signature = (
            copy.deepcopy(signature)
            if isinstance(signature, dict)
            else {
                "run_number": run_key,
                "model": None,
            }
        )
        wizard_state = {
            "signature": active_signature,
            "recommendation": serialize_fit_wizard_recommendation(recommendation),
            "log_text": str(log_text),
        }

        if (
            self._active_single_run_number is not None
            and int(self._active_single_run_number) == run_key
        ):
            self._single_tab._cache_wizard_analysis(
                recommendation,
                signature=active_signature,
                log_text=log_text,
            )
            self._single_state_by_run[run_key] = self._single_tab.get_state()
            return

        state = self.get_single_state_for_run(run_key)
        if not isinstance(state, dict):
            recommended = recommendation.recommended_assessment
            if recommended is not None and recommended.fit_result.success:
                state = self._single_state_from_fit_result(
                    recommended.template.model,
                    recommended.fit_result,
                    source="Fit Wizard",
                )
            else:
                state = {
                    "model_name": "Composite",
                    "composite_model": (
                        recommendation.recommended_assessment.template.model.to_dict()
                        if recommendation.recommended_assessment is not None
                        else self._single_tab._composite_model.to_dict()
                    ),
                    "parameters": [],
                    "result_html": "No fit performed yet",
                }
        state["wizard_state"] = wizard_state
        self._single_state_by_run[run_key] = copy.deepcopy(state)

    def share_single_function_state(
        self,
        source_run_number: int,
        target_run_numbers: list[int],
        datasets_by_run: dict[int, MuonDataset] | None = None,
    ) -> int:
        """Copy source single-fit function/parameter state to target runs.

        The copied state intentionally clears fit-result text for targets because
        no fit has been run for those datasets yet.

        For field-specific parameters (like B_L), applies the target dataset's
        field value when *datasets_by_run* is provided, falling back to the
        pre-loaded ``_all_datasets`` list if it is not.
        """
        source_state = self.get_single_state_for_run(source_run_number)
        if not isinstance(source_state, dict):
            return 0

        # Build a run-number lookup from the supplied mapping, then fall back to
        # the stale _all_datasets list (populated by set_datasets).
        def _lookup_dataset(run_key: int) -> MuonDataset | None:
            if datasets_by_run is not None:
                return datasets_by_run.get(run_key)
            for ds in self._all_datasets:
                try:
                    if int(ds.run_number) == run_key:
                        return ds
                except (TypeError, ValueError):
                    pass
            return None

        updated = 0
        active_run = self._active_single_run_number
        for run_number in target_run_numbers:
            try:
                run_key = int(run_number)
            except (TypeError, ValueError):
                continue
            if run_key == int(source_run_number):
                continue

            shared_state = copy.deepcopy(source_state)
            shared_state["result_html"] = "No fit performed yet"

            target_dataset = _lookup_dataset(run_key)
            if target_dataset is not None and isinstance(shared_state.get("parameters"), list):
                for param_dict in shared_state["parameters"]:
                    pname = param_dict.get("name")
                    if isinstance(pname, str):
                        base_name, _index = split_parameter_name(pname)
                        file_value = _get_file_value_for_parameter(target_dataset, base_name)
                        if file_value is not None:
                            param_dict["value"] = file_value

            self._single_state_by_run[run_key] = shared_state
            if active_run is not None and run_key == active_run:
                self._single_tab.restore_state(self._single_state_by_run[run_key])
            updated += 1
        return updated

    def _result_html_from_fit(self, fit_result: object, source: str) -> str:
        """Build single-fit result HTML from a completed fit result object."""
        if getattr(fit_result, "success", False) is not True:
            message = str(getattr(fit_result, "message", "Fit failed"))
            return f"<b>{source} failed:</b> {message}"

        reduced = float(getattr(fit_result, "reduced_chi_squared", float("nan")))
        chi2 = float(getattr(fit_result, "chi_squared", float("nan")))
        lines = [
            f"<b>{source}</b>",
            f"<b>χ² = {chi2:.4f}</b>",
            f"<b>χ²ᵣ = {reduced:.4f}</b>",
            "<br><b>Parameters:</b>",
        ]

        uncertainties = getattr(fit_result, "uncertainties", {}) or {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if not isinstance(name, str):
                continue
            value = float(getattr(param, "value", 0.0))
            unc = float(uncertainties.get(name, 0.0))
            lines.append(f"  {_format_param_label(name)} = {value:.6f} ± {unc:.6f}")
        return "<br>".join(lines)

    def _single_state_from_fit_result(
        self,
        model: CompositeModel,
        fit_result: object,
        source: str,
        roles: dict[str, str] | None = None,
    ) -> dict:
        """Return single-tab state populated from a fitted model result.

        ``roles`` maps parameter names to their batch role (global/local/fixed);
        each is recorded on the param entry so the single tab can annotate how the
        parameter was classified in the batch fit.
        """
        roles = roles or {}
        values_by_name: dict[str, object] = {}
        for param in getattr(fit_result, "parameters", []):
            name = getattr(param, "name", None)
            if isinstance(name, str):
                values_by_name[name] = param

        params: list[dict[str, object]] = []
        for pname in model.param_names:
            param = values_by_name.get(pname)
            if param is None:
                value = float(model.param_defaults.get(pname, 0.0))
                fixed = False
                default_min = get_param_info(pname).default_min
                min_text = str(default_min) if default_min is not None else "-inf"
                max_text = "inf"
            else:
                try:
                    value = float(getattr(param, "value", model.param_defaults.get(pname, 0.0)))
                except (TypeError, ValueError):
                    value = float(model.param_defaults.get(pname, 0.0))
                fixed = bool(getattr(param, "fixed", False))

                min_val = getattr(param, "min", -float("inf"))
                max_val = getattr(param, "max", float("inf"))
                min_text = (
                    "-inf"
                    if min_val is None or not np.isfinite(float(min_val))
                    else str(float(min_val))
                )
                max_text = (
                    "inf"
                    if max_val is None or not np.isfinite(float(max_val))
                    else str(float(max_val))
                )

            params.append(
                {
                    "name": pname,
                    "value": value,
                    "fixed": fixed,
                    "min": min_text,
                    "max": max_text,
                    "role": roles.get(pname),
                }
            )

        return {
            "model_name": "Composite",
            "composite_model": model.to_dict(),
            "parameters": params,
            "result_html": self._result_html_from_fit(fit_result, source),
        }

    def register_global_fit_results(
        self, results_by_run: dict[int, tuple[object, object, object]]
    ) -> None:
        """Persist per-run single-tab state using the latest successful global fit."""
        model = self._global_tab._composite_model
        active_run = self._active_single_run_number
        roles = self._global_tab.param_role_map()

        for run_number, payload in results_by_run.items():
            if not isinstance(payload, tuple) or not payload:
                continue
            fit_result = payload[0]
            if getattr(fit_result, "success", False) is not True:
                continue
            self._global_tab.register_single_fit_seed(run_number, model, fit_result)
            run_state = self._single_state_from_fit_result(
                model, fit_result, source="Batch fit", roles=roles
            )
            self._single_state_by_run[int(run_number)] = run_state

            if active_run is not None and int(run_number) == int(active_run):
                self._single_tab.restore_state(run_state)

    # ── project state helpers ──────────────────────────────────────────

    def get_single_state(self) -> dict:
        """Return serialisable state of the single-fit tab."""
        if self._active_single_run_number is not None:
            self._single_state_by_run[self._active_single_run_number] = self._single_tab.get_state()

        active_state = self._single_tab.get_state()
        states_by_run = {
            str(run_number): dict(state)
            for run_number, state in self._single_state_by_run.items()
            if isinstance(state, dict)
        }
        combined_state = dict(active_state)
        combined_state["states_by_run"] = states_by_run
        combined_state["active_run_number"] = self._active_single_run_number
        return combined_state

    def get_domain_state(self, domain: str) -> dict:
        """Return serialisable fit state for one fitting domain."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            return {
                "domain": normalized,
                "single_fit_state": self.get_single_state(),
                "global_fit_state": self.get_global_state(),
                "fit_ui_state": self.get_ui_state(),
            }
        return {
            "domain": normalized,
            "single_fit_state": copy.deepcopy(self._single_state_by_domain.get(normalized, {})),
            "global_fit_state": copy.deepcopy(self._global_state_by_domain.get(normalized, {})),
            "fit_ui_state": copy.deepcopy(self._ui_state_by_domain.get(normalized, {})),
        }

    def restore_domain_state(self, domain: str, state: dict | None) -> None:
        """Restore serialisable fit state for one fitting domain."""
        normalized = coerce_domain(domain)
        if not isinstance(state, dict):
            state = {}
        self._single_state_by_domain[normalized] = copy.deepcopy(state.get("single_fit_state", {}))
        self._global_state_by_domain[normalized] = copy.deepcopy(state.get("global_fit_state", {}))
        self._ui_state_by_domain[normalized] = copy.deepcopy(state.get("fit_ui_state", {}))
        if normalized == self._domain:
            self.restore_single_state(self._single_state_by_domain[normalized])
            self.restore_global_state(self._global_state_by_domain[normalized])
            self.restore_ui_state(self._ui_state_by_domain[normalized])

    def restore_single_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        states_by_run: dict[int, dict] = {}
        raw_states = state.get("states_by_run") if isinstance(state, dict) else None
        if isinstance(raw_states, dict):
            for run_key, run_state in raw_states.items():
                if not isinstance(run_state, dict):
                    continue
                try:
                    run_number = int(run_key)
                except (TypeError, ValueError):
                    continue
                states_by_run[run_number] = dict(run_state)

        self._single_state_by_run = states_by_run

        active_run = self._active_single_run_number
        if active_run is not None and active_run in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[active_run])
            return

        # Backward-compatible legacy payloads (single shared state).
        if isinstance(state, dict):
            self._single_tab.restore_state(state)
            if active_run is not None:
                self._single_state_by_run[active_run] = self._single_tab.get_state()

    def get_global_state(self) -> dict:
        """Return serialisable state of the global-fit tab."""
        return self._global_tab.get_state()

    def get_grouped_state(self) -> dict:
        """Return the grouped-fit classification (physics roles + nuisance block)."""
        return self._global_tab.get_grouped_state()

    def send_single_model_to_batch(self) -> bool:
        """Copy the single-fit tab's model/fit function into the Batch tab.

        Returns ``True`` when a model was sent. The Single ⇄ Batch flow: build a
        model in Single, send it to seed a batch over the selected runs.
        """
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return False
        self._global_tab._set_composite_model(model)
        return True

    def _on_send_model_to_batch(self) -> None:
        """Handle the Single tab's 'Send Model to Batch' action."""
        if self.send_single_model_to_batch():
            self._tabs.setCurrentWidget(self._global_tab)

    def restore_global_state(self, state: dict) -> None:
        """Restore global-fit tab state from a saved dict."""
        self._global_tab.restore_state(state)

    def get_ui_state(self) -> dict:
        """Return serialisable UI state for the fit panel container."""
        return {"active_tab_index": int(self._tabs.currentIndex())}

    def restore_ui_state(self, state: dict) -> None:
        """Restore serialisable UI state for the fit panel container."""
        index = state.get("active_tab_index")
        if isinstance(index, int) and 0 <= index < self._tabs.count():
            self._tabs.setCurrentIndex(index)
