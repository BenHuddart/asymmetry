"""Fit panel — model selection, parameter table, and fit controls.

Mirrors WiMDA's Analyse → Fit dialog: choose a model, set initial
parameters, run the fit, and inspect results.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import html
import os
import re
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
from PySide6.QtCore import QEvent, QEventLoop, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QKeyEvent
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
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
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

from asymmetry.core.data.combine import (
    CombineError,
    coadd_member_windows,
    combine_runs,
    reduce_combined_run,
    runs_with_dataset_metadata,
)
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.count_domain import (
    COUNT_COSTS,
    RESERVED_COUNT_PARAMS,
    fb_overlay_curves,
    fit_fb_alpha,
    fit_single_histogram,
    single_histogram_overlay,
)
from asymmetry.core.fitting.domain_library import coerce_domain
from asymmetry.core.fitting.engine import FitCancelledError, FitEngine, FitResult
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
    build_count_group,
    build_grouped_count_model,
    build_grouped_time_domain_datasets,
    build_grouped_time_domain_groups,
    fit_grouped_series,
    fit_grouped_time_domain,
    validate_grouped_model_contract,
)
from asymmetry.core.fitting.parameters import (
    AffineTie,
    Parameter,
    ParameterSet,
    get_param_info,
    split_parameter_name,
)
from asymmetry.core.fitting.result_summary import fit_result_summary
from asymmetry.core.fitting.rrf_offset import (
    UnsupportedRRFComponentError,
    rrf_frequency_offsets,
)
from asymmetry.core.fitting.series import fit_asymmetry_series
from asymmetry.core.fitting.series_seeding import (
    SeriesPoint,
    diagnose_series,
    resolve_series_params,
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
from asymmetry.gui.fit_settings import fit_quality_confidence
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from asymmetry.gui.panels.initial_values_dialog import InitialValuesDialog
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.widgets import (
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    apply_param_table_style,
    build_primary_button_qss,
    configure_formula_label,
    error_html,
    fit_quality_chip_html,
    fit_quality_tooltip,
    info_html,
    make_formula_box,
    make_section,
    make_section_header,
    success_html,
    warning_html,
)
from asymmetry.gui.tasks import TaskRunner, TaskWorker
from asymmetry.gui.widgets.current_page_sizing import CurrentPageSizingMixin
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


def _seed_group_phase_degrees(grouped_groups: list[object]) -> dict[str, float]:
    """Return the FFT-estimated absolute phase (degrees) of each group's oscillation."""
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

    return phase_degrees_by_group


def _wrap_phase_rad(phase_deg: float) -> float:
    """Wrap a phase in degrees to radians on ``(-pi, pi]``."""
    return float(np.angle(np.exp(1j * np.deg2rad(phase_deg))))


def _seed_group_absolute_phases(grouped_groups: list[object]) -> dict[str, float]:
    """Return per-group *absolute* phase seeds in radians (wrapped to ``(-pi, pi]``).

    Grouped fits hold the shared model phase at zero, so each group's per-group
    phase nuisance carries the full FFT-estimated phase rather than an offset
    relative to the first group.
    """
    return {
        group_id: _wrap_phase_rad(phase_deg)
        for group_id, phase_deg in _seed_group_phase_degrees(grouped_groups).items()
    }


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


class _FloatLimitField(QLineEdit):
    """Compact text field for a fit-range limit (min/max).

    Replaces ``QDoubleSpinBox`` for the fit range: a plain typed field (the
    design's limit-field style) with no spin arrows and no reserved arrow
    padding, so it stays narrow on a 13" dock. A ``QDoubleValidator`` keeps
    entries numeric and in range. Exposes the small spinbox-compatible surface
    (``value``/``setValue``/``decimals``/``setDecimals``/``setRange``) that the
    shared fit-range plumbing already relies on, so both Fit tabs reuse it and
    ``editingFinished`` (a built-in ``QLineEdit`` signal) keeps its old wiring.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._decimals = 3
        self._value = 0.0
        self._validator = QDoubleValidator(-1000.0, 1000.0, self._decimals, self)
        self._validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(self._validator)
        self.setFont(mono_font(11.0))
        # A bare QLineEdit sizes to ~17 chars; cap it so the min/max pair stays
        # compact in the dock (fits "-1000.000" with room to spare).
        self.setMinimumWidth(56)
        self.setMaximumWidth(88)
        self.setText(self._format(self._value))
        # Normalise the display after a manual edit (e.g. "1" -> "1.000"). This
        # connects first, so the external editingFinished handler reads the
        # already-normalised text via value().
        self.editingFinished.connect(self._normalise_text)

    def _format(self, value: float) -> str:
        return f"{float(value):.{self._decimals}f}"

    def _clamp(self, value: float) -> float:
        """Clamp to the validator's range, matching QDoubleSpinBox.

        ``QDoubleValidator`` only rejects out-of-range *keystrokes*; it does not
        bound a programmatic ``setValue`` or an Intermediate entry committed on
        focus-out. The spinbox this replaces clamped both, so do the same here —
        otherwise an out-of-range fit limit could reach the engine.
        """
        return min(max(float(value), self._validator.bottom()), self._validator.top())

    def _normalise_text(self) -> None:
        self.setText(self._format(self.value()))

    def value(self) -> float:
        """Current value (clamped to range), or the last set value if blank."""
        try:
            return self._clamp(float(self.text()))
        except ValueError:
            return self._value

    def setValue(self, value: float) -> None:  # noqa: N802 — spinbox-API shim
        self._value = self._clamp(value)
        self.setText(self._format(self._value))

    def decimals(self) -> int:
        return self._decimals

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 — spinbox-API shim
        self._decimals = int(decimals)
        self._validator.setDecimals(self._decimals)
        self.setText(self._format(self.value()))

    def setRange(self, minimum: float, maximum: float) -> None:  # noqa: N802 — spinbox-API shim
        self._validator.setRange(minimum, maximum, self._decimals)


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
        self.setWindowTitle(f"Affine tie — {_format_param_label(param_name)}")
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
        spin = QDoubleSpinBox()
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
                    i, self.COL_NAME, _make_param_name_item(_format_param_label(pname), pname)
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
                raise ValueError(f"Invalid value for {_format_param_label(param_name)}") from exc

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
        # Optional provider of the rotating-frame ν₀ (MHz) supplied by the host
        # window; returns a frequency when an RRF fit should run (the plot's RRF
        # display is active), else None. Default no-op keeps the tab standalone.
        self._rrf_frequency_provider: Callable[[], float | None] = lambda: None
        self._last_fit_result: FitResult | None = None
        self._last_fit_parameters: ParameterSet | None = None
        self._pull_diagnostic_btn: QPushButton | None = None
        self._pull_diagnostic_window: QWidget | None = None
        #: Background fits run via the shared TaskRunner machinery; the
        #: worker handle exists only so the Stop button can cancel it.
        self._fit_call_runner = TaskRunner(self)
        self._fit_worker = None
        #: Bumped on every model (re)configuration. A fit snapshots it at
        #: launch; a mismatch at completion means the model was changed or
        #: Reset while the fit ran (Reset reuses the same object, so object
        #: identity alone would miss it), so the stale result is not applied.
        self._model_generation = 0

        # Model selection
        model_group, model_box = make_section("Model")
        model_layout = QFormLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_box.addLayout(model_layout)
        self._formula_box, self._formula_label = _make_formula_box()
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        self._share_group_btn = QPushButton("Share with Group")
        self._share_group_btn.setToolTip("Share this fit function with the selected data group.")
        self._share_group_btn.clicked.connect(self._on_share_function_with_group)
        self._share_group_btn.setEnabled(False)
        self._drop_background_btn = QPushButton("Drop background")
        self._drop_background_btn.setToolTip(
            "Remove the constant background term from the model.\n"
            "For amplitude calibration (e.g. a light-OFF A₀ run) a free background "
            "absorbs part of the initial asymmetry, splitting the fitted amplitude; "
            "drop it to fit the full A₀ with a single relaxation term."
        )
        self._drop_background_btn.clicked.connect(self._on_drop_background)
        self._drop_background_btn.setEnabled(False)
        self._send_to_batch_btn = QPushButton("Send to Batch")
        self._send_to_batch_btn.setToolTip(
            "Copy this fit function into the Batch tab to seed a batch fit over the selected runs."
        )
        self._send_to_batch_btn.clicked.connect(self.send_model_to_batch_requested.emit)
        self._add_to_series_btn = QPushButton("Add to Series...")
        self._add_to_series_btn.setToolTip(
            "Add this run's single fit to an existing batch series with a matching model."
        )
        self._add_to_series_btn.clicked.connect(self.add_to_series_requested.emit)

        # Single column of natural-width buttons. A side-by-side grid forced the
        # two button columns (~110px each) to set the whole Fit tab's minimum
        # width; stacking them lets the dock get genuinely narrow on a 13" screen,
        # and dropping the Expanding policy keeps each button only as wide as its
        # label needs (left-aligned) instead of stretching to fill the row.
        model_button_layout = QVBoxLayout()
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setSpacing(4)
        for _model_btn in (
            self._edit_model_btn,
            self._fit_wizard_btn,
            self._drop_background_btn,
            self._share_group_btn,
            self._send_to_batch_btn,
            self._add_to_series_btn,
        ):
            model_button_layout.addWidget(_model_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self._formula_row_label = QLabel("A(t):")
        model_layout.addRow(self._formula_row_label, self._formula_box)
        model_layout.addRow("", model_button_layout)
        layout.addWidget(model_group)

        # Fit range section
        fit_range_group, _fit_range_box = make_section("Fit range")
        fit_range_layout = QHBoxLayout()
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(4)
        _fit_range_box.addLayout(fit_range_layout)

        self._fit_range_min_spin = _FloatLimitField()

        self._fit_range_mid_label = QLabel("≤ <i>t</i> ≤")
        self._fit_range_mid_label.setTextFormat(Qt.TextFormat.RichText)

        self._fit_range_max_spin = _FloatLimitField()

        self._fit_range_unit_label = QLabel("μs")

        fit_range_layout.addWidget(self._fit_range_min_spin)
        fit_range_layout.addWidget(self._fit_range_mid_label)
        fit_range_layout.addWidget(self._fit_range_max_spin)
        fit_range_layout.addWidget(self._fit_range_unit_label)
        fit_range_layout.addStretch()
        layout.addWidget(fit_range_group)

        self._fit_range_min_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)
        self._fit_range_max_spin.editingFinished.connect(self._on_fit_range_spinbox_committed)

        # Parameter table — the shared Name·Value·Fix·Min·Max·Batch·Link·Tie
        # widget (columns/delegates/Fix-Link-Tie wiring/fraction sync live in
        # FitParameterTable). It self-connects itemChanged for fraction sync.
        param_group, param_layout = make_section("Parameters")
        self._param_table = FitParameterTable()
        param_layout.addWidget(self._param_table)
        layout.addWidget(param_group)

        # Buttons
        btn_layout = QGridLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setHorizontalSpacing(6)
        btn_layout.setVerticalSpacing(6)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setStyleSheet(build_primary_button_qss())
        self._fit_btn.clicked.connect(self._run_fit)
        # Stop replaces the Fit button while a worker-based fit runs.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Cancel the running fit; no result is recorded.")
        self._stop_btn.clicked.connect(self._on_stop_fit)
        self._stop_btn.hide()
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
        self._minos_checkbox = QCheckBox("Asymmetric errors")
        self._minos_checkbox.setToolTip(
            "After fitting, walk the χ² profile of each free parameter to get its "
            "asymmetric +/− 1σ interval (MINOS). Slower than the default symmetric "
            "errors; most useful at low statistics, near parameter bounds, or in "
            "strongly correlated fits where the parabolic error is unreliable."
        )
        btn_layout.addWidget(self._fit_btn, 0, 0)
        btn_layout.addWidget(self._stop_btn, 0, 0)
        btn_layout.addWidget(self._reset_btn, 0, 1)
        btn_layout.addWidget(self._preview_btn, 0, 2)
        btn_layout.addWidget(self._pull_diagnostic_btn, 1, 0, 1, 3)
        btn_layout.addWidget(self._minos_checkbox, 2, 0, 1, 3)
        btn_layout.setColumnStretch(3, 1)
        layout.addLayout(btn_layout)

        # Results
        layout.addWidget(make_section_header("Fit Results"))
        self._results_group = QFrame()
        self._results_group.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        results_layout = QVBoxLayout(self._results_group)
        self._result_label = QLabel("No fit performed yet")
        self._result_label.setWordWrap(True)
        results_layout.addWidget(self._result_label)
        layout.addWidget(self._results_group)

        # Spare vertical height pools here, below the results box, instead of
        # being claimed by an expanding parameter table.
        layout.addStretch(1)

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

    def set_rrf_frequency_provider(self, provider: Callable[[], float | None]) -> None:
        """Install the host's rotating-frame ν₀ provider (see __init__)."""
        self._rrf_frequency_provider = provider or (lambda: None)

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

    def current_fit_range_text(self) -> str | None:
        """Active fit range as a provenance string (µs/MHz), or ``None``."""
        return _fit_range_provenance_text(
            self._fit_range_min_spin, self._fit_range_max_spin, self._fit_range_unit_label
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

    def _update_drop_background_enabled(self) -> None:
        """Enable the Drop-background affordance only when there is one to drop."""
        reduced = _model_without_trailing_background(self._composite_model)
        self._drop_background_btn.setEnabled(self._domain == "time" and reduced is not None)

    def _on_drop_background(self) -> None:
        """Drop the constant background term for amplitude calibration."""
        reduced = _model_without_trailing_background(self._composite_model)
        if reduced is None:
            return
        self._set_composite_model(reduced)

    def _set_composite_model(self, model: CompositeModel) -> None:
        """Set the active composite model and rebuild the parameter table."""
        self._composite_model = model
        # Any model (re)build — including Reset, which reuses the same object —
        # invalidates an in-flight fit's table/diagnostic write-back. (The table
        # drops auxiliary non-model params from a prior restore in populate().)
        self._model_generation += 1
        _set_formula_label_text(self._formula_label, model.formula_string())
        _apply_domain_mismatch_warning(self._formula_label, model, self._domain)

        dataset_field = (
            self._current_dataset.run.field
            if self._current_dataset is not None and self._current_dataset.run is not None
            else 0.0
        )
        value_overrides = dict(_field_value_overrides(model, dataset_field))
        if self._domain == "frequency" and self._current_dataset is not None:
            # Frequency-domain peak seeds take precedence over the field seed.
            value_overrides.update(seed_peak_parameters_from_dataset(self._current_dataset, model))

        # shape_factor_a (instrument normalisation) is held by default alongside
        # the model's declared fixed-by-default parameters.
        fixed_names = set(model.fixed_by_default_params()) | {"shape_factor_a"}
        self._param_table.populate(model, value_overrides=value_overrides, fixed_names=fixed_names)
        self._update_drop_background_enabled()

    def _synchronize_fraction_value_rows(self, edited_param_name: str | None = None) -> None:
        self._param_table.synchronize_fractions(edited_param_name)

    @property
    def _updating_fraction_values(self) -> bool:
        """Proxy the parameter table's bulk-write guard.

        The table owns the guard now; existing call sites that wrap programmatic
        cell writes (fit-result write-back, RRF shift, restore) toggle this and
        the suppression still works because both read the same flag.
        """
        tbl = getattr(self, "_param_table", None)
        return bool(tbl.is_updating) if tbl is not None else False

    @_updating_fraction_values.setter
    def _updating_fraction_values(self, value: bool) -> None:
        tbl = getattr(self, "_param_table", None)
        if tbl is not None:
            tbl._updating = bool(value)

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
                value_item.setData(
                    _ValueUncertaintyDelegate._MINOS_ROLE,
                    (result.minos_errors or {}).get(param_name),
                )

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
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
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

    def model_and_seed(self) -> tuple[CompositeModel, ParameterSet]:
        """Return the active single-fit model and its current parameter seed.

        For headless fits (e.g. the Data Browser's "Re-fit as co-added") that
        reuse the configured single-fit model without touching the form. Raises
        :class:`ValueError` on a malformed parameter value, like the fit run.
        """
        return self._composite_model, self._parameter_set_from_table()

    def _parameter_set_from_table(self) -> ParameterSet:
        """Build a :class:`ParameterSet` from the parameter table.

        Raises :class:`ValueError` with a user-facing message on a malformed
        value (the only hard error; bad bounds fall back to ±inf). Shared by
        the fit run and the pull-distribution diagnostic.
        """
        return self._param_table.read_parameter_set()

    def current_seed_values(self) -> dict[str, str]:
        """Return the live parameter-table seed text keyed by parameter name.

        Used to seed the Batch tab from the current single-fit values rather than
        model defaults / stale state; non-finite cells are skipped.
        """
        return self._param_table.current_seed_values()

    def _run_fit(self) -> None:
        """Execute the fit."""
        if self._fit_blocked:
            message = self._fit_block_reason or "Fit is unavailable for the current selection."
            self._result_label.setText(f"ERROR: {message}")
            return

        if self._current_dataset is None:
            self._result_label.setText("ERROR: No dataset selected")
            return

        mismatch = _fit_domain_mismatch_message(self._domain, self._current_dataset)
        if mismatch is not None:
            self._result_label.setText(f"ERROR: {mismatch}")
            return

        if self._composite_model is None:
            self._result_label.setText("ERROR: No function defined")
            return

        missing = getattr(self._composite_model, "missing_component_names", ())
        if missing:
            self._result_label.setText(
                "ERROR: the model requires missing user function(s): "
                f"{', '.join(missing)}. Register them (Setup → User functions…) "
                "and reload the project."
            )
            return

        # Build parameter set from table
        try:
            parameters = self._parameter_set_from_table()
        except ValueError as exc:
            self._result_label.setText(f"ERROR: {exc}")
            return

        # Resolve the rotating-reference-frame offset, if the host's RRF display
        # is active. The fit then consumes RAW lab-frame data with the model's
        # rotation frequencies offset by ν₀, so it keeps exact per-bin
        # statistics while the engine's free parameter is the small, better-
        # conditioned δν; the parameter table stays lab-frame throughout.
        model = self._composite_model
        rrf_offsets: dict[str, float] | None = None
        rrf_nu0 = self._rrf_frequency_provider()
        if rrf_nu0:
            try:
                rrf_offsets = rrf_frequency_offsets(model, float(rrf_nu0))
            except UnsupportedRRFComponentError as exc:
                # A composite with an oscillating component that is not a pure
                # frame rotation (muonium, Bessel, …) cannot be safely offset;
                # refuse rather than silently leave a line in the lab frame.
                self._result_label.setText(
                    f"ERROR: cannot fit in the rotating frame — {exc} "
                    "Turn off the rotating frame (Options → Advanced) to fit this model."
                )
                return
            except ValueError:
                # No rotation component at all (e.g. a pure relaxation model):
                # the rotating frame does not apply; fit normally.
                rrf_offsets = None

        fit_seed = (
            _shift_rrf_parameters(parameters, rrf_offsets, sign=-1) if rrf_offsets else parameters
        )

        # Run the fit on a worker thread; the GUI (and Stop button) stay live.
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_label.setText("Fitting...")

        # Snapshot launch-time context: the user may switch run or model while
        # the worker runs, and the result must be interpreted against what was
        # actually fitted. The TaskRunner relay invokes these closures on the
        # GUI thread with each launch's own context, so a late result can
        # never be applied against a different launch's snapshot.
        dataset = self._current_dataset
        # Only thread the RRF offset when one is active, so the ordinary fit
        # path (and its test doubles) is unchanged.
        fit_kwargs: dict = {"minos": self._minos_checkbox.isChecked()}
        if rrf_offsets:
            fit_kwargs["frequency_offsets"] = rrf_offsets
        self._fit_worker = _start_fit_call(
            self,
            functools.partial(
                self._fit_engine.fit,
                dataset,
                model.function,
                fit_seed,
                **fit_kwargs,
            ),
            on_finished=(
                lambda result, p=parameters, d=dataset, m=model, g=self._model_generation, off=rrf_offsets, nu0=rrf_nu0: (
                    self._apply_single_fit_result(result, p, d, m, g, rrf_offsets=off, rrf_nu0=nu0)
                )
            ),
            on_error=self._on_single_fit_error,
            on_cancelled=self._on_single_fit_cancelled,
        )
        self._set_fit_busy(True)

    def _set_fit_busy(self, busy: bool) -> None:
        """Swap the Fit button for a Stop button (and back) around a worker fit."""
        self._stop_btn.setVisible(busy)
        self._stop_btn.setEnabled(busy)
        self._fit_btn.setVisible(not busy)
        if busy:
            self._fit_btn.setEnabled(False)
        else:
            # Re-derive from the gating contract rather than force-enable: the
            # run may have been removed or the panel blocked while the fit ran.
            self._fit_btn.setEnabled(self._current_dataset is not None and not self._fit_blocked)

    def _on_stop_fit(self) -> None:
        """Request cancellation of the active worker-based fit."""
        worker = self._fit_worker
        if worker is not None:
            self._stop_btn.setEnabled(False)
            self._result_label.setText("Cancelling fit…")
            worker.cancel()

    def _on_single_fit_cancelled(self) -> None:
        """Handle a cancelled single fit: restore the panel, record nothing."""
        self._set_fit_busy(False)
        self._fit_worker = None
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_label.setText("Fit cancelled — no result recorded.")

    def _on_single_fit_error(self, message: str) -> None:
        self._set_fit_busy(False)
        self._fit_worker = None
        self._result_label.setText(f"<b>Error during fit:</b><br>{message}")

    def shutdown_workers(self) -> None:
        """Cancel any running fit and wait for its thread (window close)."""
        self._fit_call_runner.shutdown()

    def wait_for_fit(self, timeout_ms: int = 30_000) -> bool:
        """Block (with a live event loop) until the launched fit completes."""
        return _wait_for_fit_thread(self, timeout_ms)

    def _apply_single_fit_result(
        self,
        result,
        parameters,
        dataset,
        model,
        model_generation,
        *,
        rrf_offsets=None,
        rrf_nu0=None,
    ) -> None:
        """Apply a completed single fit to the panel (GUI thread)."""
        self._set_fit_busy(False)
        self._fit_worker = None

        if not result.success:
            self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
            self._result_label.setText(f"<b>Fit failed:</b> {result.message}")
            return

        # The engine fitted the rotating-frame offsets δν; shift the result back
        # to the lab frame (ν = δν + ν₀, bounds with it) so every downstream
        # surface — the parameter table, the overlay curve drawn on raw data,
        # the recorded FitSlot, the pull diagnostic — reads in the lab frame. χ²,
        # uncertainties and covariance are frame-invariant (the offset is an
        # additive constant), so only the values/bounds move.
        rrf_note = ""
        if rrf_offsets:
            result.parameters = _shift_rrf_parameters(result.parameters, rrf_offsets, sign=+1)
            rrf_note = (
                "<br><i>frame: ν_RRF = "
                f"{float(rrf_nu0):.4f} MHz — fitted in the rotating frame; "
                "frequencies reported in the lab frame.</i>"
            )

        # A result is only "fresh" when the panel still shows the model AND run
        # it was fitted on. Otherwise the user navigated away mid-fit: applying
        # the values would corrupt a different model's seed table, arm the pull
        # diagnostic against the wrong run, overlay the curve on the wrong plot,
        # or record a FitSlot for the wrong run. The generation counter catches
        # Reset (which reuses the same model object, so identity alone misses).
        model_unchanged = (
            self._composite_model is model and self._model_generation == model_generation
        )
        dataset_unchanged = self._current_dataset is dataset

        warnings_note = _fit_warnings_html(result)
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_label.setText(_fit_success_html(result) + rrf_note + warnings_note)
        summary = _fit_summary(result)
        self._result_label.setToolTip(
            fit_quality_tooltip(summary.get("quality"), summary.get("params_at_bound"))
        )

        if not (model_unchanged and dataset_unchanged):
            if not model_unchanged:
                reason = "the model was changed or reset while it ran"
            else:
                run_id = dataset.metadata.get("run_number", "?")
                reason = f"run {run_id} is no longer selected"
            self._result_label.setText(
                _fit_success_html(result)
                + rrf_note
                + warnings_note
                + f"<br><i>This fit was not applied or recorded because {reason}. "
                "Restore the original model and run, then refit to keep it.</i>"
            )
            return

        # Fresh: remember the converged fit for the pull-distribution diagnostic.
        self._last_fit_result = result
        self._last_fit_parameters = parameters
        if self._pull_diagnostic_btn is not None:
            self._pull_diagnostic_btn.setEnabled(self._can_run_pull_diagnostic())

        display_values = _normalized_model_param_values(
            model,
            {parameter.name: parameter.value for parameter in result.parameters},
        )

        # Update table with fit results.
        minos_errors = result.minos_errors or {}
        self._updating_fraction_values = True
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else ""
            if param_name in result.parameters:
                fitted_value = display_values.get(param_name, result.parameters[param_name].value)
                val_item = self._param_table.item(i, 1)
                val_item.setText(f"{fitted_value:.6f}")
                unc = result.uncertainties.get(param_name, None)
                val_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)
                val_item.setData(
                    _ValueUncertaintyDelegate._MINOS_ROLE, minos_errors.get(param_name)
                )
                # A fresh single fit supersedes any piped-back batch role.
                _set_param_batch_role_cell(self._param_table, i, None)
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        param_dict = {p.name: p.value for p in result.parameters}
        n_samples = _fit_curve_sample_count(
            model,
            param_dict,
            float(dataset.time.min()),
            float(dataset.time.max()),
        )
        t_fit = np.linspace(dataset.time.min(), dataset.time.max(), n_samples)
        y_fit = model.function(t_fit, **param_dict)
        component_curves = model.evaluate_components(t_fit, additive_only=True, **param_dict)
        self.fit_completed.emit(result, (t_fit, y_fit), component_curves)

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

        state = {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            # The table serialises its rows (incl. auxiliary non-model params and
            # fraction-value normalisation).
            "parameters": self._param_table.parameters_state(),
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
                # Unregistered names (a user function whose plugin is not
                # installed) materialise as named zero-valued placeholders so
                # the saved model is never silently replaced; only structurally
                # malformed data falls back to the default model.
                restored = CompositeModel.from_dict(composite_data, allow_missing=True)
            except ValueError:
                self._set_composite_model(
                    CompositeModel(["Exponential", "Constant"], operators=["+"])
                )
            else:
                self._set_composite_model(restored)
                if restored.missing_component_names:
                    names = ", ".join(restored.missing_component_names)
                    self._result_label.setText(
                        f"<b>Missing user function(s):</b> {names}.<br>"
                        "The saved model is preserved (missing components plot as "
                        "zero) but cannot be fitted until they are registered — "
                        "see Setup → User functions…"
                    )

        # The table applies the saved values/fix/bounds/link/tie onto its rows
        # (the model was just rebuilt by _set_composite_model) and re-establishes
        # auxiliary non-model parameters that have no row.
        params_data = {p["name"]: p for p in state.get("parameters", []) if isinstance(p, dict)}
        self._param_table.restore_parameters(params_data)

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


#: (label, mode) for the batch-series seeding selector, shared by the Batch-tab
#: combobox and the ``Analysis ▸ Batch seeding`` menu so the two cannot drift.
#: "auto" picks per the order key; "as_provided" seeds every run independently;
#: "chain" carries each run's fit into the next (natural for an ordered T/B scan).
BATCH_SEEDING_MODES: tuple[tuple[str, str], ...] = (
    ("Auto", "auto"),
    ("Independent seeds", "as_provided"),
    ("Chain from previous run", "chain"),
)

#: mode token -> display label, for logging/lookups (reverse of the pairs above).
BATCH_SEEDING_LABELS: dict[str, str] = {mode: label for label, mode in BATCH_SEEDING_MODES}

#: Tooltip shared by the on-tab selector and the menu submenu.
BATCH_SEEDING_TOOLTIP = (
    "How each run in the batch is seeded:\n"
    "• Auto — choose per the series order key.\n"
    "• Independent seeds — every run starts from the shared seed values.\n"
    "• Chain from previous run — each run starts from the previous run's fit "
    "(best for an ordered temperature/field scan)."
)


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
    # (run_number, model, physics_values_by_name) — a converged single grouped
    # fit's shared physics, so the batch grouped surface can chain-seed per run.
    single_grouped_fit_recorded = Signal(int, object, object)
    grouped_preview_requested = Signal(object, object)  # (grouped_datasets, preview_curves)
    fit_range_edit_committed = Signal(float, float)  # (x_min, x_max) from spinbox commit
    # (dataset, {"result": FitResult|GroupedTimeDomainFitResult,
    #            "overlays": {group_id: (time, corrected_model)}}) — the fit result
    # plus the overlay curves for the Individual-Groups plot (displayed
    # lifetime-corrected scale).
    count_fit_completed = Signal(object, object)
    count_grouping_promoted = Signal(object)  # (dataset) — a count calibration hit the grouping
    share_function_with_group_requested = Signal(int)  # (source run) — single grouped surface
    send_grouped_model_to_batch_requested = Signal()  # single grouped surface → batch surface
    # Emitted with the new mode when the on-tab seeding selector changes, so the
    # Analysis ▸ Batch seeding menu can mirror it (two-way sync).
    batch_seeding_mode_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        member_kind: str = "runs",
        grouped_single: bool = False,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        # Member kind is fixed per instance and follows the active representation:
        # the groups surface (Individual-groups representation) is group-membered,
        # every other surface is run-membered. (Phase 3: scope is derived, not selected.)
        self._member_kind = member_kind if member_kind in ("runs", "groups") else "runs"
        # The single grouped fit (one dataset's detector groups) shares one
        # fit-function across its groups, so its physics params take the
        # single-fit-style Fix tickbox rather than the Global/Local/Fixed combo
        # (which only makes sense for the multi-run batch grouped fit).
        self._grouped_single = bool(grouped_single) and self._member_kind == "groups"

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
        # Lazily-computed per-(run, group) nuisance auto-seeds, cached under the
        # same key as the grouped context (invalidated together).
        self._grouped_seed_cache: tuple[object, dict] | None = None
        # Batch-series seeding mode (menu-bar "Batch seeding"); "auto" picks
        # chain-from-previous for ordered scans, else independent seeds.
        self._batch_seeding_mode = "auto"
        # Seeding metadata from the last block-separable F-B series fit (resolved
        # mode + reason + reseeded runs), surfaced in the results box.
        self._series_seeding_meta: dict[str, object] | None = None
        # Suggested per-run descending-frequency seeds from the last batch's
        # diagnostics, applied by the "Use suggested per-run seeds" signpost button.
        self._suggested_series_seeds: dict[int, dict[str, float]] = {}
        # In-batch co-add of successive grouped-series members before fitting
        # (WiMDA BatchFit Smooth/Bin). "off" disables; "bin"/"smooth" co-add
        # ``_coadd_window`` successive members per fit via combine_runs.
        self._coadd_mode = "off"
        self._coadd_window = 2
        # Count-domain fit target: "all" (fgAll, the existing grouped path),
        # "fb" (forward+backward with free alpha), or "single" (one histogram).
        self._count_fit_mode = "all"
        self._count_fit_cost = "poisson"
        self._count_single_side = "forward"
        # Phase 2 window/nuisance flexibility (all off by default).
        self._count_exclude: tuple[float, float] | None = None
        self._count_fit_t0 = False
        self._count_baseline = False
        # Phase 3 count-loss + double pulse.
        self._count_deadtime = False
        self._count_dpsep = 0.0  # μs; > 0 switches on the double-pulse model
        self._count_dpsep_fit = False  # locate dpsep by a coarse->fine scan
        self._last_count_dt0: float | None = None
        self._last_count_group: int | None = None
        # Fitted calibrations captured for the α/t0/background promote paths
        # (siblings of the deadtime DT0 promote). Each is suggest-only.
        self._last_count_alpha: tuple[float, float | None] | None = None
        self._last_count_t0_us: float | None = None
        self._last_count_bg: tuple[float | None, float | None] | None = None
        self._last_count_cal_group: int | None = None
        self._last_count_bin_width: float | None = None
        self._last_count_ref_run: int | None = None
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._composite_model = self._default_composite_model()
        self._applied_field_default_gauss = 0.0
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
        model_group, model_box = make_section("Model")
        model_layout = QFormLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_box.addLayout(model_layout)
        self._formula_box, self._formula_label = _make_formula_box()
        self._edit_model_btn = QPushButton("Edit Function...")
        self._edit_model_btn.clicked.connect(self._edit_function)
        self._fit_wizard_btn = QPushButton("Global Wizard...")
        self._fit_wizard_btn.setToolTip("Open the Global Fit Wizard.")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)
        # Single column of natural-width buttons (mirrors SingleFitTab): a
        # side-by-side grid forced both button columns to set the tab's minimum
        # width; stacking them left-aligned at natural width lets the Batch tab
        # get as narrow as the Single tab on a 13" screen.
        model_button_layout = QVBoxLayout()
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setSpacing(4)
        for _model_btn in (self._edit_model_btn, self._fit_wizard_btn):
            model_button_layout.addWidget(_model_btn, 0, Qt.AlignmentFlag.AlignLeft)
        # The single grouped surface can push its function to the run's
        # data-group peers (mirrors SingleFitTab's "Share with Group").
        self._share_group_btn: QPushButton | None = None
        if self._grouped_single:
            self._share_group_btn = QPushButton("Share with Group")
            self._share_group_btn.setToolTip(
                "Share this grouped fit function with the selected data group."
            )
            self._share_group_btn.setEnabled(False)
            self._share_group_btn.clicked.connect(self._on_share_function_with_group)
            model_button_layout.addWidget(self._share_group_btn, 0, Qt.AlignmentFlag.AlignLeft)
            self._send_to_batch_btn = QPushButton("Send to Batch")
            self._send_to_batch_btn.setToolTip(
                "Copy this grouped fit function and its seeds to the Batch surface."
            )
            self._send_to_batch_btn.clicked.connect(self.send_grouped_model_to_batch_requested.emit)
            model_button_layout.addWidget(self._send_to_batch_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self._formula_row_label = QLabel("A(t):")
        model_layout.addRow(self._formula_row_label, self._formula_box)
        model_layout.addRow("", model_button_layout)
        layout.addWidget(model_group)

        # Fit range section
        _fr_group, _fr_box = make_section("Fit range")
        _fr_layout = QHBoxLayout()
        _fr_layout.setContentsMargins(0, 0, 0, 0)
        _fr_layout.setSpacing(4)
        _fr_box.addLayout(_fr_layout)

        self._fit_range_min_spin = _FloatLimitField()

        self._fit_range_mid_label = QLabel("≤ <i>t</i> ≤")
        self._fit_range_mid_label.setTextFormat(Qt.TextFormat.RichText)

        self._fit_range_max_spin = _FloatLimitField()

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
        self._param_group, param_layout = make_section("Parameter Classification")

        param_header_layout = QHBoxLayout()
        param_header_layout.addStretch()
        self._param_help_btn = QPushButton("?")
        self._param_help_btn.setFixedWidth(28)
        self._param_help_btn.setToolTip("Explain Global, Local, Fixed, and File parameter roles")
        self._param_help_btn.clicked.connect(self._show_parameter_classification_help)
        param_header_layout.addWidget(self._param_help_btn)
        param_layout.addLayout(param_header_layout)

        self._param_table = QTableWidget(0, 4)
        # "Seed" (not "Value"): this column is the shared initial value applied to
        # every run in the batch, not a per-run fitted result. Per-run fitted
        # values live in the Parameters (trend) tab. Naming it "Value" misled
        # users into reading the static template as per-dataset output.
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Seed", "Type", "Bounds"])
        self._param_table.horizontalHeader().setStretchLastSection(False)
        _seed_header = self._param_table.horizontalHeaderItem(1)
        if _seed_header is not None:
            _seed_header.setToolTip(
                "Shared seed (initial value) applied to every run in the batch — "
                "not a per-run fitted result.\nSelecting different runs does not "
                "change it. Per-run fitted values appear in the Parameters tab "
                "after the batch fit completes."
            )
        self._param_table.setColumnWidth(0, PARAM_NAME_COL_WIDTH)  # Parameter name
        self._param_table.setColumnWidth(1, 76)  # Shared seed value
        self._param_table.setColumnWidth(2, 86)  # Type (dropdown)
        self._param_table.setColumnWidth(3, 104)  # Bounds
        _apply_param_table_style(self._param_table)
        # Tab commits the open editor on the editable columns (Value, Bounds);
        # without this Qt's focus traversal jumps to the Type combo and the
        # typed value is lost (see _CommitOnTabDelegate).
        self._param_table.setItemDelegate(_CommitOnTabDelegate(self._param_table))
        # Size to content (see _size_param_table_to_content): the dock scrolls
        # vertically as a whole, so the table shows all rows with no internal
        # scrollbar or empty rows.
        self._param_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._param_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._param_table.setWordWrap(False)
        self._param_table.itemChanged.connect(self._on_param_table_item_changed)
        param_layout.addWidget(self._param_table)
        layout.addWidget(self._param_group)

        self._grouped_context_label = QLabel()
        self._grouped_context_label.setWordWrap(True)
        self._grouped_context_label.hide()
        layout.addWidget(self._grouped_context_label)

        self._group_param_group, group_param_layout = make_section("Per-Group Parameters")
        self._group_param_table = QTableWidget(0, 4)
        self._group_param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Type", "Bounds"])
        self._group_param_table.horizontalHeader().setStretchLastSection(False)
        self._group_param_table.setColumnWidth(0, PARAM_NAME_COL_WIDTH)
        self._group_param_table.setColumnWidth(1, 78)
        self._group_param_table.setColumnWidth(2, 86)
        self._group_param_table.setColumnWidth(3, 104)
        _apply_param_table_style(self._group_param_table)
        self._group_param_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._group_param_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._group_param_table.setWordWrap(False)
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

        self._group_model_group, group_model_layout = make_section("Fit-Function Parameters")
        if self._grouped_single:
            # Single grouped fit: every detector group shares one fit-function,
            # so the physics params take the single-fit-style Fix tickbox instead
            # of the Global/Local/Fixed combo (per-group quantities are the
            # nuisance block). Link/Tie are hidden — the grouped engine does not
            # honour cross-parameter ties — and Batch role has no meaning here.
            self._group_model_table = FitParameterTable()
            for _hidden in (
                FitParameterTable.COL_BATCH,
                FitParameterTable.COL_LINK,
                FitParameterTable.COL_TIE,
            ):
                self._group_model_table.setColumnHidden(_hidden, True)
        else:
            self._group_model_table = QTableWidget(0, 4)
            self._group_model_table.setHorizontalHeaderLabels(
                ["Parameter", "Value", "Type", "Bounds"]
            )
            self._group_model_table.horizontalHeader().setStretchLastSection(False)
            self._group_model_table.setColumnWidth(0, PARAM_NAME_COL_WIDTH)
            self._group_model_table.setColumnWidth(1, 78)
            self._group_model_table.setColumnWidth(2, 86)
            self._group_model_table.setColumnWidth(3, 104)
            _apply_param_table_style(self._group_model_table)
            self._group_model_table.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._group_model_table.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self._group_model_table.setWordWrap(False)
            self._group_model_table.itemChanged.connect(self._on_group_model_table_item_changed)
        group_model_layout.addWidget(self._group_model_table)
        layout.addWidget(self._group_model_group)

        # In-batch co-add (WiMDA BatchFit Smooth/Bin): co-add successive members
        # through combine_runs before each series fit. Grouped-series mode only.
        self._coadd_group, _coadd_box = make_section("Co-add members")
        coadd_layout = QHBoxLayout()
        coadd_layout.setContentsMargins(0, 0, 0, 0)
        coadd_layout.setSpacing(6)
        _coadd_box.addLayout(coadd_layout)
        self._coadd_mode_combo = QComboBox()
        # Short labels (the tooltip below spells out Bin vs Smooth) so the row
        # does not set the Batch tab's minimum width past the other Fit tabs.
        self._coadd_mode_combo.addItem("Off", "off")
        self._coadd_mode_combo.addItem("Bin", "bin")
        self._coadd_mode_combo.addItem("Smooth", "smooth")
        self._coadd_mode_combo.setToolTip(
            "Co-add successive runs before fitting (WiMDA Smooth/Bin):\n"
            "• Bin — non-overlapping windows, one combined fit per N runs.\n"
            "• Smooth — sliding window stepped by one run.\n"
            "Counts are summed at the raw-histogram level, then fitted."
        )
        self._coadd_mode_combo.currentIndexChanged.connect(self._on_coadd_mode_changed)
        self._coadd_window_spin = QSpinBox()
        self._coadd_window_spin.setRange(2, 99)
        self._coadd_window_spin.setValue(self._coadd_window)
        self._coadd_window_spin.setToolTip("Number of successive runs co-added per fit.")
        self._coadd_window_spin.valueChanged.connect(self._on_coadd_window_changed)
        self._coadd_window_label = QLabel("runs per fit")
        coadd_layout.addWidget(self._coadd_mode_combo)
        coadd_layout.addWidget(self._coadd_window_spin)
        coadd_layout.addWidget(self._coadd_window_label)
        coadd_layout.addStretch()
        self._coadd_window_spin.setEnabled(False)
        self._coadd_window_label.setEnabled(False)
        self._coadd_group.hide()
        layout.addWidget(self._coadd_group)

        # Batch-series seeding selector, on the tab it governs (the same control
        # also lives in Analysis ▸ Batch seeding; the two stay in sync). Surfacing
        # it here makes "Chain from previous run" discoverable for ordered scans.
        # Omitted on the single grouped surface, which fits one dataset's groups —
        # there is no run series to seed across, so the control would be meaningless
        # (the same reason _grouped_single hides the Batch/Link/Tie columns).
        self._seeding_combo: QComboBox | None = None
        if not self._grouped_single:
            seeding_layout = QHBoxLayout()
            seeding_layout.setContentsMargins(0, 0, 0, 0)
            seeding_layout.setSpacing(6)
            self._seeding_label = QLabel("Seeding:")
            self._seeding_label.setToolTip(BATCH_SEEDING_TOOLTIP)
            self._seeding_combo = QComboBox()
            self._seeding_combo.setToolTip(BATCH_SEEDING_TOOLTIP)
            for label, mode in BATCH_SEEDING_MODES:
                self._seeding_combo.addItem(label, mode)
            self._seeding_combo.setCurrentIndex(
                self._seeding_combo.findData(self._batch_seeding_mode)
            )
            self._seeding_combo.currentIndexChanged.connect(self._on_seeding_combo_changed)
            seeding_layout.addWidget(self._seeding_label)
            seeding_layout.addWidget(self._seeding_combo)
            seeding_layout.addStretch()
            layout.addLayout(seeding_layout)

        # Fit buttons. A single-row HBox of these long-labelled actions
        # ("Run Batch Fit" + "Preview" + "Per-run seeds…" + the checkbox) set
        # the Batch tab's minimum width far past the Single tab; a compact grid
        # (mirroring SingleFitTab) keeps the widest row to two buttons.
        btn_layout = QGridLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setHorizontalSpacing(6)
        btn_layout.setVerticalSpacing(6)
        self._fit_btn = QPushButton("Run Batch Fit")
        self._fit_btn.setStyleSheet(build_primary_button_qss())
        self._fit_btn.clicked.connect(self._run_global_fit)
        self._fit_btn.setEnabled(False)
        # Stop replaces the disabled Fit button while a worker-based fit runs.
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setToolTip("Cancel the running fit; no partial result is recorded.")
        self._stop_btn.clicked.connect(self._on_stop_fit)
        self._stop_btn.hide()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview_requested)
        self._preview_btn.setEnabled(False)
        # The grouped batch surface hides the per-group table, so its per-(run,
        # group) nuisances are edited only through this dialog — name it for that.
        batch_grouped = self._member_kind == "groups" and not self._grouped_single
        self._initial_values_btn = QPushButton(
            "Edit per-group initial values…" if batch_grouped else "Per-run seeds…"
        )
        self._initial_values_btn.setToolTip(
            "Edit each (run, group)'s initial nuisance values (auto-seeded per dataset)."
            if batch_grouped
            else (
                "Edit each run's starting (seed) parameter values for the batch fit "
                "(the warm-start the outlier signpost points at)."
            )
        )
        self._initial_values_btn.clicked.connect(self._open_initial_values_dialog)
        self._minos_checkbox = QCheckBox("Asymmetric errors")
        self._minos_checkbox.setToolTip(
            "After fitting, report asymmetric +/− 1σ MINOS intervals (slower; most "
            "useful at low statistics or near parameter bounds)."
        )
        btn_layout.addWidget(self._fit_btn, 0, 0)
        btn_layout.addWidget(self._stop_btn, 0, 0)
        btn_layout.addWidget(self._preview_btn, 0, 1)
        btn_layout.addWidget(self._initial_values_btn, 1, 0, 1, 2)
        btn_layout.addWidget(self._minos_checkbox, 2, 0, 1, 2)
        btn_layout.setColumnStretch(2, 1)
        layout.addLayout(btn_layout)

        # Results display
        layout.addWidget(make_section_header("Batch Fit Results"))
        self._results_group = QFrame()
        self._results_group.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        results_layout = QVBoxLayout(self._results_group)
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumHeight(200)
        self._result_text.setText("No fit performed yet")
        results_layout.addWidget(self._result_text)
        layout.addWidget(self._results_group)

        # Seeding signpost — hidden until a batch's ν(T)/A(T) trend shows the
        # near-transition collapse/outlier signature. It points a struggling user
        # at the per-run "Per-run seeds…" warm-start and offers to apply the
        # descending-frequency seeds the diagnostics computed.
        self._seeding_signpost = QFrame()
        self._seeding_signpost.setObjectName("seedingSignpost")
        self._seeding_signpost.setStyleSheet(
            f"#seedingSignpost {{ border: 1px solid {tokens.WARN}; border-radius: 4px; }}"
        )
        signpost_layout = QVBoxLayout(self._seeding_signpost)
        signpost_layout.setContentsMargins(8, 6, 8, 6)
        signpost_layout.setSpacing(4)
        self._seeding_signpost_label = QLabel("")
        self._seeding_signpost_label.setWordWrap(True)
        signpost_layout.addWidget(self._seeding_signpost_label)
        signpost_btn_row = QHBoxLayout()
        signpost_btn_row.setSpacing(6)
        self._apply_suggested_seeds_btn = QPushButton("Use suggested per-run seeds")
        self._apply_suggested_seeds_btn.setToolTip(
            "Fill the per-run seed table with descending frequency seeds "
            "interpolated from the runs that fit cleanly, then re-run the batch."
        )
        self._apply_suggested_seeds_btn.clicked.connect(self._apply_suggested_series_seeds)
        self._open_initial_values_from_signpost_btn = QPushButton("Open per-run seeds…")
        self._open_initial_values_from_signpost_btn.setToolTip(
            "Open the per-run seed table to edit warm-start values by hand."
        )
        self._open_initial_values_from_signpost_btn.clicked.connect(
            self._open_initial_values_dialog
        )
        signpost_btn_row.addWidget(self._apply_suggested_seeds_btn)
        signpost_btn_row.addWidget(self._open_initial_values_from_signpost_btn)
        signpost_btn_row.addStretch(1)
        signpost_layout.addLayout(signpost_btn_row)
        self._seeding_signpost.hide()
        layout.addWidget(self._seeding_signpost)

        layout.addStretch()

        # Every background fit on this tab — global, grouped, grouped-series and
        # count-domain — runs through the shared TaskRunner (bounded shutdown,
        # GUI-thread callback relay). Only one fit runs at a time (shared Fit
        # button); _fit_worker / _count_fit_worker hold the live TaskWorker
        # handle for the Stop button.
        self._fit_worker: TaskWorker | None = None
        self._count_fit_worker: TaskWorker | None = None
        self._fit_call_runner = TaskRunner(self)

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
        self._grouped_seed_cache = None
        self._refresh_inherited_single_fit_defaults()
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
        previous = self._current_dataset
        self._current_dataset = dataset
        # A captured count-fit calibration belongs to the run it was fitted on.
        # When the active run changes, drop the captures so a promote button
        # cannot write one run's fitted α/t0/background/DT0 into another run's
        # grouping (with that run's number recorded as the reference).
        if previous is not dataset:
            self._clear_count_calibrations()
        # Invalidate the grouped-context memo whenever the active dataset
        # changes (its grouped groups depend only on this dataset).
        self._grouped_context_cache = None
        self._grouped_seed_cache = None
        self._refresh_field_parameter_defaults_for_current_dataset()
        # The shared model phase is held at zero in grouped fits; the per-group
        # phase lives in the per-group phase nuisance, reseeded by the call below.
        self._update_group_parameter_defaults()
        self._update_mode_ui(preserve_result=False)
        if self._share_group_btn is not None:
            self._share_group_btn.setEnabled(dataset is not None)

    def _on_share_function_with_group(self) -> None:
        """Emit the share-with-group request for the active run (single grouped)."""
        dataset = self._current_dataset
        if dataset is None:
            return
        try:
            run_number = int(dataset.run_number)
        except (TypeError, ValueError):
            return
        self.share_function_with_group_requested.emit(run_number)

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

    def current_fit_range_text(self) -> str | None:
        """Active fit range as a provenance string (µs/MHz), or ``None``."""
        return _fit_range_provenance_text(
            self._fit_range_min_spin, self._fit_range_max_spin, self._fit_range_unit_label
        )

    def _refresh_inherited_single_fit_defaults(self) -> None:
        """Apply single-fit seeds when every selected dataset shares one model.

        Works for both the FB batch surface (over ``_datasets``) and the grouped
        batch surface (over ``_member_datasets``). For grouped the shared model is
        adopted so the fit-time inheritance gate matches, but the FB parameter
        table is not filled (the grouped physics table is built separately).
        """
        self._inherited_seed_by_run = {}
        self._inherited_model_dict = None

        grouped = self._member_kind == "groups"
        datasets = self._member_datasets if grouped else self._datasets
        if len(datasets) < 2:
            return

        run_numbers: list[int] = []
        for ds in datasets:
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

        if not grouped:
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

    def _set_composite_model(
        self, model: CompositeModel, seed_values: dict[str, str] | None = None
    ) -> None:
        """Set the active composite model and rebuild classification rows.

        ``seed_values`` (parameter name → value text) supplies initial values
        that take priority over preserved state and model defaults. It is used
        when seeding a batch from the single-fit tab so the batch starts from
        the current single-fit seeds rather than defaults or stale preserved
        state (BUG B8c).
        """
        seed_values = seed_values or {}
        preserved_state = self._current_parameter_row_state()
        grouped_model_state = self._current_grouped_model_row_state()
        # A new model invalidates any per-run initial values keyed by old names.
        self._user_initial_values_by_run = {}
        self._user_grouped_initial_values = {}
        self._updating_fraction_values = True
        self._composite_model = model
        _set_formula_label_text(self._formula_label, model.formula_string())
        _apply_domain_mismatch_warning(self._formula_label, model, self._domain)

        # Seed any 'field' parameters from the applied field. The single grouped
        # (individual-groups) surface fits one dataset at a time, so use that
        # dataset's field — including for models with more than one oscillatory
        # component (every 'field' param, not just the first). The multi-run
        # batch surface uses the mean field across its loaded members.
        if self._grouped_single:
            single_field = _get_file_value_for_parameter(self._current_dataset, "field")
            seed_field_gauss = float(single_field) if single_field is not None else 0.0
        else:
            dataset_fields = [
                ds.run.field for ds in self._datasets if ds.run is not None and ds.run.field != 0.0
            ]
            seed_field_gauss = float(np.mean(dataset_fields)) if dataset_fields else 0.0
        field_overrides = _field_value_overrides(model, seed_field_gauss)
        frequency_seed_values: dict[str, list[float]] = {}
        if self._domain == "frequency":
            for dataset in self._datasets:
                for key, value in seed_peak_parameters_from_dataset(dataset, model).items():
                    frequency_seed_values.setdefault(key, []).append(float(value))
        frequency_overrides = {
            key: float(np.mean(values)) for key, values in frequency_seed_values.items() if values
        }
        self._applied_field_default_gauss = seed_field_gauss

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
            seed_text = seed_values.get(pname)
            if seed_text is not None:
                value_text = seed_text
            else:
                value_text = previous.get("value") or str(default_val)
            value_item = QTableWidgetItem(value_text)
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
        _size_param_table_to_content(self._param_table)
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
        # Default each (run, group) to its OWN dataset's auto-seed (per-dataset
        # FFT phase / counts), falling back to the shared table value; explicit
        # user overrides still win.
        auto_seeds = self._grouped_member_nuisance_seeds()

        params = [(name, _format_param_label(name), "local") for name in GROUP_NUISANCE_PARAMS]
        members: list[tuple[int, str]] = []
        values: dict[int, dict[str, float]] = {}
        for key, label, run, group_id in self._grouped_member_specs():
            members.append((key, label))
            user = self._user_grouped_initial_values.get(key, {})
            auto_for_run = auto_seeds.get(int(run), {})
            values[key] = {
                name: float(
                    user.get(
                        name,
                        auto_for_run.get(name, {}).get(
                            str(group_id), group_values.get(name, {}).get(group_id, 0.0)
                        ),
                    )
                )
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
            members, params, values, parent=self, title="Per-run seeds"
        )
        if dialog.exec():
            self._user_initial_values_by_run = dialog.edited_values()

    def batch_datasets(self) -> list[MuonDataset]:
        """Return the datasets currently configured for the batch/scan."""
        return list(self._datasets)

    def _run_global_fit(self) -> None:
        """Execute global fit on all datasets."""
        missing = getattr(self._composite_model, "missing_component_names", ())
        if missing:
            self._result_text.setText(
                "Error: the model requires missing user function(s): "
                f"{', '.join(missing)}. Register them (Setup → User functions…) "
                "and reload the project."
            )
            return

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

        mismatch = _fit_domain_mismatch_message(self._domain, self._datasets[0])
        if mismatch is not None:
            self._result_text.setText(f"Error: {mismatch}")
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

        # Run the global fit on the shared TaskRunner; the GUI (and Stop
        # button) stay live.
        self._result_text.setText("Fitting... This may take a moment for many datasets...")
        self._set_series_busy(True)

        # Store model for later use in callbacks (read by _on_fit_finished).
        self._current_model = self._composite_model
        self._current_global_params = global_params

        # The started signal lets listeners snapshot launch-time context (e.g.
        # which frequency representation the datasets came from) before any UI
        # refresh can change it; emit before the worker can produce a result.
        # The F-B asymmetry batch is *block-separable* when no Global parameter is
        # free (every free parameter is Local): each run is an independent
        # minimisation, so the batch can chain from the previous good run and
        # detect-and-reseed a run that lands on the spurious near-transition branch.
        # A real simultaneous fit (a free Global ties the runs together) cannot chain
        # and keeps the proven global_fit path.
        free_global_params = [
            name for name in global_params if name not in fixed_params and name not in file_params
        ]
        self._series_seeding_meta = None
        self.global_fit_started.emit()
        if not free_global_params:
            amplitude_param, frequency_param = resolve_series_params(model.param_names)
            self._fit_worker = _start_fit_call(
                self,
                functools.partial(
                    fit_asymmetry_series,
                    self._datasets,
                    self._composite_model.function,
                    global_params,
                    local_params,
                    initial_params,
                    fit_engine=self._fit_engine,
                    minos=self._minos_checkbox.isChecked(),
                    seeding=self._batch_seeding_mode,
                    order_key=self._asymmetry_series_order_key(),
                    amplitude_param=amplitude_param,
                    frequency_param=frequency_param,
                ),
                on_finished=self._on_asymmetry_series_finished,
                on_error=self._on_fit_error,
                on_cancelled=self._on_series_fit_cancelled,
            )
        else:
            # global_fit returns (results_dict, fitted_global) — unpack into the
            # two-argument finished handler on the GUI thread.
            self._fit_worker = _start_fit_call(
                self,
                functools.partial(
                    self._fit_engine.global_fit,
                    self._datasets,
                    self._composite_model.function,
                    global_params,
                    local_params,
                    initial_params,
                    minos=self._minos_checkbox.isChecked(),
                ),
                on_finished=lambda result: self._on_fit_finished(*result),
                on_error=self._on_fit_error,
                on_cancelled=self._on_series_fit_cancelled,
            )

    def _asymmetry_series_order_key(self) -> dict[int, float] | None:
        """Best-effort run → temperature/field order key for the F-B batch.

        Chaining follows the physical scan order; Auto only chains when a usable
        ordered key exists. Returns ``None`` when any selected run lacks scan
        metadata, so Auto safely falls back to independent seeds.
        """
        order: dict[int, float] = {}
        for dataset in self._datasets:
            meta = getattr(dataset, "metadata", None) or {}
            value: float | None = None
            for key in ("temperature", "temperature_k", "field", "field_g"):
                raw = meta.get(key)
                if raw is not None:
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        value = None
                    break
            if value is None:
                return None
            order[int(dataset.run_number)] = value
        return order or None

    def _on_asymmetry_series_finished(self, series: object) -> None:
        """Adapt a chained F-B series result into the shared finished handler.

        Stashes the resolved seeding mode/reason and any reseeded runs so the
        results box can report them, then routes the per-run results through the
        common :meth:`_on_fit_finished` path (which also runs the outlier signpost).
        """
        self._series_seeding_meta = {
            "seeding_used": getattr(series, "seeding_used", ""),
            "seeding_reason": getattr(series, "seeding_reason", ""),
            "reseeded_runs": tuple(getattr(series, "reseeded_runs", ())),
        }
        self._on_fit_finished(series.results, series.fitted_global)

    def _run_grouped_time_domain_fit(self) -> None:
        """Execute grouped time-domain fitting for the active dataset."""
        if self._fit_blocked:
            self._result_text.setText(
                self._fit_block_reason
                or "Grouped time-domain fit is unavailable for the current selection."
            )
            return

        # Count-domain targets (forward+backward with free alpha, or a single
        # histogram) fit raw counts via the dedicated count-domain driver instead
        # of the lifetime-corrected fgAll path.
        if self._count_fit_mode in ("fb", "single"):
            self._run_count_domain_fit()
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
        self._set_series_busy(True)
        self._current_model = grouped_model
        self._current_global_params = global_params

        # grouped_datasets is GUI-side launch context (not produced by the
        # engine); bind it into the finished closure, mirroring the engine
        # worker's old (grouped_datasets, result) two-argument emit.
        self._fit_worker = _start_fit_call(
            self,
            functools.partial(
                fit_grouped_time_domain,
                grouped_groups,
                grouped_model.function,
                global_params,
                local_params,
                initial_params,
                minos=self._minos_checkbox.isChecked(),
                cost=self._count_fit_cost,
            ),
            on_finished=lambda result, ds=grouped_datasets: self._on_grouped_fit_finished(
                ds, result
            ),
            on_error=self._on_fit_error,
            on_cancelled=self._on_series_fit_cancelled,
        )

    # --- count-domain fit modes (forward/backward free-alpha, single histogram) ---

    def set_count_fit_mode(self, mode: str) -> None:
        """Select the count-domain fit target: ``all``, ``fb`` or ``single``."""
        self._count_fit_mode = mode if mode in ("all", "fb", "single") else "all"

    def set_count_fit_cost(self, cost: str) -> None:
        """Select the count-fit cost: ``poisson`` (default) or ``gaussian``."""
        self._count_fit_cost = cost if cost in COUNT_COSTS else "poisson"

    def set_count_single_side(self, side: str) -> None:
        """Select which detector group the single-histogram fit targets."""
        self._count_single_side = "backward" if str(side).lower() == "backward" else "forward"

    def set_count_exclude(self, exclude: tuple[float, float] | None) -> None:
        """Set the optional interior exclude window (μs) for count fits."""
        if exclude is None or float(exclude[1]) <= float(exclude[0]):
            self._count_exclude = None
        else:
            self._count_exclude = (float(exclude[0]), float(exclude[1]))

    def set_count_fit_t0(self, enabled: bool) -> None:
        """Toggle a free time-zero offset parameter for count fits."""
        self._count_fit_t0 = bool(enabled)

    def set_count_baseline(self, enabled: bool) -> None:
        """Toggle a free stretched-exponential baseline-drift term for count fits."""
        self._count_baseline = bool(enabled)

    def set_count_deadtime(self, enabled: bool) -> None:
        """Toggle a free count-loss (deadtime DT0) term for count fits."""
        self._count_deadtime = bool(enabled)

    def set_count_dpsep(self, dpsep_us: float) -> None:
        """Set the ISIS double-pulse separation (μs); 0 disables the double-pulse model."""
        try:
            value = float(dpsep_us)
        except (TypeError, ValueError):
            value = 0.0
        self._count_dpsep = value if value > 0.0 else 0.0

    def set_count_dpsep_fit(self, enabled: bool) -> None:
        """Toggle locating dpsep by a coarse->fine scan rather than fixing it."""
        self._count_dpsep_fit = bool(enabled)

    def promote_count_deadtime(self, *, additive: bool = False) -> None:
        """Promote the last fitted deadtime DT0 into the grouping (WiMDA Send-to-Group)."""
        from asymmetry.core.transform.deadtime import promote_deadtime_to_grouping
        from asymmetry.core.transform.grouping import effective_group_indices

        dataset = self._current_dataset
        if self._last_count_dt0 is None or self._last_count_group is None:
            self._result_text.setText("Run a deadtime count fit first, then promote DT0.")
            return
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            self._result_text.setText("Promote needs the active run with detector histograms.")
            return
        grouping = dataset.run.grouping if isinstance(dataset.run.grouping, dict) else {}
        indices = effective_group_indices(grouping, int(self._last_count_group))
        change = promote_deadtime_to_grouping(
            grouping,
            float(self._last_count_dt0),
            n_histograms=len(dataset.run.histograms),
            detector_indices=indices or None,
            additive=additive,
        )
        before = next(iter(change["before"].values()), 0.0)
        after = next(iter(change["after"].values()), 0.0)
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_text.setHtml(
            success_html(
                "Deadtime promoted to grouping",
                detail=(
                    f"Group {self._last_count_group} detectors: "
                    f"{before:.5g} → {after:.5g} μs ({'added' if additive else 'replaced'}). "
                    "Re-reduce the run to apply."
                ),
            )
        )
        self.count_grouping_promoted.emit(dataset)

    def _count_fb_groups(self, dataset: MuonDataset) -> tuple[int, int]:
        grouping = (
            dataset.run.grouping if dataset.run and isinstance(dataset.run.grouping, dict) else {}
        )
        try:
            forward = int(grouping.get("forward_group", 1))
        except (TypeError, ValueError):
            forward = 1
        try:
            backward = int(grouping.get("backward_group", 2))
        except (TypeError, ValueError):
            backward = 2
        return forward, backward

    def _count_fit_range(self) -> tuple[float | None, float | None]:
        lo = float(self._fit_range_min_spin.value())
        hi = float(self._fit_range_max_spin.value())
        if hi > lo:
            return lo, hi
        return None, None

    def _count_n0_seed(self, dataset: MuonDataset, group_id: int) -> float:
        try:
            group = build_count_group(dataset, group_id, lifetime_corrected=False)
        except ValueError:
            return 1.0
        counts = np.asarray(group.counts, dtype=float)
        return float(counts[0]) if counts.size else 1.0

    def _count_fit_seed_params(self, dataset: MuonDataset, model, *, mode: str) -> ParameterSet:
        """Seed a count-fit parameter set with the model amplitude left FREE.

        Count modes recover the asymmetry amplitude itself (carried by the model),
        so unlike the normalised fgAll path the model's amplitude parameter is not
        pinned to 1; it is seeded at a typical calibration value and fitted.
        """
        # A model parameter named like a count-fit nuisance/structural slot would
        # be silently swallowed by the name-based dispatch; reject it up front.
        collisions = sorted(set(model.param_names) & RESERVED_COUNT_PARAMS)
        if collisions:
            raise ValueError(
                f"Model parameter(s) {collisions} collide with reserved count-fit names; "
                "rename them before running a count-domain fit."
            )

        config = self._parse_grouped_parameter_configuration()
        model_values = dict(config["model_values"])
        bounds = dict(config["bounds"])
        fixed = set(config["fixed"])

        params = ParameterSet()
        for name in model.param_names:
            lo, hi = bounds.get(name, (-float("inf"), float("inf")))
            # The asymmetry amplitude (unit "%") is what count modes recover, so
            # leave it free and seed it near a calibration value. Identify it by
            # its unit, not is_amplitude_parameter, which also matches rate
            # parameters like ``a_L`` and misses the standard ``A0``.
            if get_param_info(name).unit == "%":
                seed = float(model_values.get(name, 0.0))
                if abs(seed) <= 1e-6 or seed == 1.0:
                    seed = 20.0  # percent; typical transverse-field calibration amplitude
                params.add(Parameter(name=name, value=seed, min=lo, max=hi))
            else:
                # The individual-groups physics table fixes oscillation phase at 0
                # by default (the phase lives in the per-group relative_phase
                # nuisances of the asymmetry fit). The single-side / F+B count fits
                # have no such nuisance, so they recover the phase directly — keep
                # phase free here regardless of that default-fixed state.
                base_name, _index = split_parameter_name(name)
                params.add(
                    Parameter(
                        name=name,
                        value=float(model_values.get(name, 0.0)),
                        min=lo,
                        max=hi,
                        fixed=(name in fixed) and base_name != "phase",
                    )
                )

        forward, backward = self._count_fb_groups(dataset)
        if mode == "fb":
            params.add(Parameter(name="N0", value=self._count_n0_seed(dataset, forward), min=0.0))
            params.add(Parameter(name="background", value=0.0, min=0.0))
            params.add(Parameter(name="background_b", value=0.0, min=0.0))
            params.add(Parameter(name="alpha", value=1.0, min=0.05, max=20.0))
        else:
            target = backward if self._count_single_side == "backward" else forward
            params.add(Parameter(name="N0", value=self._count_n0_seed(dataset, target), min=0.0))
            params.add(Parameter(name="background", value=0.0, min=0.0))
        if self._count_fit_t0:
            params.add(Parameter(name="t0", value=0.0, min=-0.1, max=0.1))
        if self._count_baseline:
            params.add(Parameter(name="lambda_base", value=0.05, min=0.0, max=10.0))
            # beta held fixed at 1 (simple exponential); free beta is unstable.
            params.add(Parameter(name="beta_base", value=1.0, min=0.2, max=3.0, fixed=True))
        if self._count_deadtime:
            params.add(Parameter(name="DT0", value=0.005, min=0.0, max=0.5))
        if self._count_dpsep > 0.0:
            if self._count_dpsep_fit:
                # Free dpsep: a coarse->fine scan refines it around the instrument
                # value (the non-smooth pulse gate defeats gradient fitting). The
                # window brackets the seed so the scan refines, not blind-searches.
                lo = max(0.0, 0.5 * self._count_dpsep)
                hi = 1.5 * self._count_dpsep
                params.add(Parameter(name="dpsep", value=self._count_dpsep, min=lo, max=hi))
            else:
                # dpsep comes from the instrument; held fixed.
                params.add(Parameter(name="dpsep", value=self._count_dpsep, fixed=True))
        return params

    def _run_count_domain_fit(self) -> None:
        """Run a forward/backward free-alpha or single-histogram count fit."""
        dataset = self._current_dataset
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            self._result_text.setText(
                "Count-domain fits need an active run with detector histograms."
            )
            return
        if len(self._member_datasets) > 1:
            self._result_text.setText(
                "Count-domain α / single-histogram fits run on one run. "
                "Use the Single surface (the active run)."
            )
            return
        if self._composite_model is None:
            self._result_text.setText("Error: No function defined")
            return

        model = self._grouped_fit_model()
        try:
            params = self._count_fit_seed_params(dataset, model, mode=self._count_fit_mode)
        except ValueError as exc:
            self._result_text.setText(str(exc))
            return

        t_min, t_max = self._count_fit_range()
        cost = self._count_fit_cost
        forward, backward = self._count_fb_groups(dataset)

        self._result_text.setText("Fitting count-domain data…")
        minos = self._minos_checkbox.isChecked()
        # Launch-time context (dataset, groups, cost, side) is bound into the
        # result closures: the user may flip the cost/side controls or switch
        # run while the worker runs, and the rendered provenance must describe
        # the fit that actually ran. The TaskRunner relay invokes the closures
        # on the GUI thread.
        if self._count_fit_mode == "fb":
            call = functools.partial(
                fit_fb_alpha,
                dataset,
                forward,
                backward,
                model.function,
                params,
                cost=cost,
                t_min=t_min,
                t_max=t_max,
                exclude=self._count_exclude,
                minos=minos,
            )

            def on_finished(result, d=dataset, f=forward, b=backward, c=cost):
                self._set_series_busy(False)
                self._count_fit_worker = None
                self._render_count_fb_result(d, result, f, b, cost=c)

        else:
            target = backward if self._count_single_side == "backward" else forward
            side = self._count_single_side
            call = functools.partial(
                fit_single_histogram,
                dataset,
                target,
                model.function,
                params,
                side=side,
                cost=cost,
                t_min=t_min,
                t_max=t_max,
                exclude=self._count_exclude,
                minos=minos,
            )

            def on_finished(result, d=dataset, t=target, c=cost, s=side):
                self._set_series_busy(False)
                self._count_fit_worker = None
                self._render_count_single_result(d, result, t, cost=c, side=s)

        self._count_fit_worker = _start_fit_call(
            self,
            call,
            on_finished=on_finished,
            on_error=self._on_count_fit_error,
            on_cancelled=self._on_count_fit_cancelled,
        )
        self._set_series_busy(True)

    def _on_count_fit_error(self, message: str) -> None:
        self._set_series_busy(False)
        self._count_fit_worker = None
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_text.setHtml(error_html(f"Count-domain fit failed: {message}"))

    def _on_count_fit_cancelled(self) -> None:
        """Handle a cancelled count-domain fit: restore the panel, record nothing."""
        self._count_fit_worker = None
        self._on_series_fit_cancelled()

    def _render_count_fb_result(
        self, dataset, result, forward: int, backward: int, *, cost: str | None = None
    ) -> None:
        cost = cost if cost is not None else self._count_fit_cost
        if not result.success:
            self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
            self._result_text.setHtml(error_html(result.message or "Forward/backward fit failed"))
            return
        fwd = result.group_results[forward]
        self._store_count_deadtime(fwd, forward)
        self._store_count_fb_extras(dataset, result, forward, backward)
        alpha = fwd.parameters["alpha"].value
        alpha_err = fwd.uncertainties.get("alpha")
        rows = [self._count_param_row(fwd, name) for name in fwd.parameters.names]
        fwd_summary = _fit_summary(fwd)
        chip = fit_quality_chip_html(fwd_summary.get("quality"), fwd_summary.get("params_at_bound"))
        detail = (
            f"α = {self._fmt_value(alpha, alpha_err)} · χ²/ν = {fwd.reduced_chi_squared:.4f}{chip} "
            f"(cost: {cost})<br>" + "<br>".join(rows)
        )
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_text.setHtml(
            success_html(f"Forward/backward fit · groups {forward}/{backward}", detail=detail)
        )
        if self._current_dataset is dataset:
            self.count_fit_completed.emit(
                dataset,
                {
                    "result": result,
                    "overlays": self._count_overlays_for_fb(dataset, result, forward, backward),
                },
            )

    def _render_count_single_result(
        self,
        dataset,
        result,
        group_id: int,
        *,
        cost: str | None = None,
        side: str | None = None,
    ) -> None:
        cost = cost if cost is not None else self._count_fit_cost
        side = side if side is not None else self._count_single_side
        if not result.success:
            self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
            self._result_text.setHtml(error_html(result.message or "Single-histogram fit failed"))
            return
        self._store_count_deadtime(result, group_id)
        self._store_count_single_extras(dataset, result, group_id)
        rows = [self._count_param_row(result, name) for name in result.parameters.names]
        detail = f"χ²/ν = {result.reduced_chi_squared:.4f} (cost: {cost})<br>" + "<br>".join(rows)
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_text.setHtml(
            success_html(
                f"Single-histogram fit · group {group_id} ({side})",
                detail=detail,
            )
        )
        if self._current_dataset is dataset:
            self.count_fit_completed.emit(
                dataset,
                {
                    "result": result,
                    "overlays": self._count_overlays_for_single(dataset, result, group_id),
                },
            )

    def _count_overlays_for_single(self, dataset, result, group_id: int) -> dict:
        """Overlay curves for a single-histogram count fit (empty on failure)."""
        t_min, t_max = self._count_fit_range()
        try:
            return single_histogram_overlay(
                dataset,
                group_id,
                result,
                t_min=t_min,
                t_max=t_max,
                exclude=self._count_exclude,
            )
        except ValueError:
            return {}

    def _count_overlays_for_fb(self, dataset, result, forward: int, backward: int) -> dict:
        """Overlay curves for a forward/backward count fit (empty on failure)."""
        t_min, t_max = self._count_fit_range()
        try:
            return fb_overlay_curves(
                dataset,
                forward,
                backward,
                result,
                t_min=t_min,
                t_max=t_max,
                exclude=self._count_exclude,
            )
        except ValueError:
            return {}

    def _store_count_deadtime(self, fit_result, group_id: int) -> None:
        """Remember a converged DT0 so it can be promoted to the grouping."""
        if "DT0" in fit_result.parameters.names:
            self._last_count_dt0 = float(fit_result.parameters["DT0"].value)
            self._last_count_group = int(group_id)
        else:
            self._last_count_dt0 = None
            self._last_count_group = None

    def _clear_count_calibrations(self) -> None:
        """Drop every captured count-fit calibration (DT0/α/t0/background)."""
        self._last_count_dt0 = None
        self._last_count_group = None
        self._last_count_alpha = None
        self._last_count_t0_us = None
        self._last_count_bg = None
        self._last_count_cal_group = None
        self._last_count_bin_width = None
        self._last_count_ref_run = None

    def _reset_count_extras(self, dataset, group_id: int) -> None:
        """Clear the α/t0/background calibration cache and record the context."""
        self._last_count_alpha = None
        self._last_count_t0_us = None
        self._last_count_bg = None
        self._last_count_cal_group = int(group_id)
        self._last_count_bin_width = self._count_bin_width(dataset)
        self._last_count_ref_run = (
            int(dataset.run_number) if dataset is not None and dataset.run is not None else None
        )

    @staticmethod
    def _finite_or_none(value) -> float | None:
        """Coerce to a finite float, or ``None`` for missing/NaN/inf values.

        Captured calibrations feed a write into the grouping; a degenerate fit
        returning NaN/inf must not be promotable (it would corrupt the grouping
        and crash the integer t0_bin conversion).
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _count_bin_width(dataset) -> float | None:
        run = getattr(dataset, "run", None)
        histograms = getattr(run, "histograms", None) if run is not None else None
        if not histograms:
            return None
        try:
            return float(histograms[0].bin_width)
        except (TypeError, ValueError, AttributeError, IndexError):
            return None

    def _store_count_fb_extras(self, dataset, result, forward: int, backward: int) -> None:
        """Capture α / t0 / per-side background from a forward/backward count fit."""
        self._reset_count_extras(dataset, forward)
        fwd = result.group_results[forward]
        bwd = result.group_results[backward]
        if "alpha" in fwd.parameters.names:
            alpha = self._finite_or_none(fwd.parameters["alpha"].value)
            if alpha is not None:
                self._last_count_alpha = (
                    alpha,
                    self._finite_or_none(fwd.uncertainties.get("alpha")),
                )
        f_bg = (
            self._finite_or_none(fwd.parameters["background"].value)
            if "background" in fwd.parameters.names
            else None
        )
        b_bg = (
            self._finite_or_none(bwd.parameters["background_b"].value)
            if "background_b" in bwd.parameters.names
            else None
        )
        if f_bg is not None or b_bg is not None:
            self._last_count_bg = (f_bg, b_bg)
        shared = getattr(result, "shared_parameters", None)
        if shared is not None and "t0" in shared:
            self._last_count_t0_us = self._finite_or_none(shared["t0"].value)

    def _store_count_single_extras(self, dataset, result, group_id: int) -> None:
        """Capture t0 / background from a single-histogram count fit (no α here)."""
        self._reset_count_extras(dataset, group_id)
        params = result.parameters
        if "t0" in params.names:
            self._last_count_t0_us = self._finite_or_none(params["t0"].value)
        if "background" in params.names:
            value = self._finite_or_none(params["background"].value)
            if value is not None:
                # The single fit targets one side; promote only that side's value.
                self._last_count_bg = (
                    (value, None) if self._count_single_side == "forward" else (None, value)
                )

    def promote_count_alpha(self) -> None:
        """Promote the last forward/backward fitted α into the grouping (F7)."""
        from asymmetry.core.transform.promote import promote_alpha_to_grouping

        grouping = self._promote_target_grouping()
        if grouping is None:
            return
        if self._last_count_alpha is None:
            self._result_text.setText(
                "Run a Forward + Backward (free α) count fit first, then promote α."
            )
            return
        alpha, alpha_err = self._last_count_alpha
        change = promote_alpha_to_grouping(
            grouping, alpha, alpha_error=alpha_err, reference_run=self._last_count_ref_run
        )
        self._announce_promote(
            "Detector balance α promoted to grouping",
            f"α: {change['before']['alpha']:.5g} → {change['after']['alpha']:.5g} "
            "(method: count_fit). Re-reduce the run to apply.",
        )

    def promote_count_t0(self) -> None:
        """Promote the last fitted count-fit t₀ offset into the grouping t0_bin (F5)."""
        from asymmetry.core.transform.promote import promote_t0_to_grouping

        grouping = self._promote_target_grouping()
        if grouping is None:
            return
        if self._last_count_t0_us is None:
            self._result_text.setText(
                "Run a count fit with the t₀ offset nuisance enabled first, then promote t₀."
            )
            return
        bin_width = self._last_count_bin_width
        if not bin_width or bin_width <= 0.0:
            self._result_text.setText("Promote needs the run's bin width to convert t₀ to bins.")
            return
        change = promote_t0_to_grouping(
            grouping,
            self._last_count_t0_us,
            bin_width_us=bin_width,
            reference_run=self._last_count_ref_run,
            group_id=self._last_count_cal_group,
        )
        residual_ns = change["residual_us"] * 1000.0
        self._announce_promote(
            "Time-zero promoted to grouping",
            f"t0_bin: {change['before']['t0_bin']} → {change['after']['t0_bin']} "
            "(fitted t₀ applied run-wide; t0_bin is a single run-level index). "
            f"Sub-bin residual {residual_ns:+.2f} ns is not representable in the "
            "integer t0_bin. Re-reduce the run to apply.",
        )

    def promote_count_background(self) -> None:
        """Promote the last fitted flat count background into the grouping (N3)."""
        from asymmetry.core.transform.promote import promote_background_to_grouping

        grouping = self._promote_target_grouping()
        if grouping is None:
            return
        if self._last_count_bg is None:
            self._result_text.setText(
                "Run a count fit with a free background first, then promote the background."
            )
            return
        forward, backward = self._last_count_bg
        change = promote_background_to_grouping(
            grouping, forward=forward, backward=backward, reference_run=self._last_count_ref_run
        )
        before, after = change["before"], change["after"]
        self._announce_promote(
            "Flat background promoted to grouping (fixed mode)",
            f"Forward: {before['forward']:.5g} → {after['forward']:.5g} · "
            f"Backward: {before['backward']:.5g} → {after['backward']:.5g}. "
            "Re-reduce the run to apply.",
        )

    def _promote_target_grouping(self) -> dict | None:
        """Return the active run's grouping dict for a promote, or ``None``.

        Emits a guidance message into the result box when there is no usable
        run, mirroring the deadtime promote's preconditions.
        """
        dataset = self._current_dataset
        if dataset is None or dataset.run is None or not dataset.run.histograms:
            self._result_text.setText("Promote needs the active run with detector histograms.")
            return None
        if not isinstance(dataset.run.grouping, dict):
            dataset.run.grouping = {}
        return dataset.run.grouping

    def _announce_promote(self, title: str, detail: str) -> None:
        """Render a success banner for a calibration promote and notify the host."""
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_text.setHtml(success_html(title, detail=detail))
        if self._current_dataset is not None:
            self.count_grouping_promoted.emit(self._current_dataset)

    def _count_param_row(self, fit_result, name: str) -> str:
        value = fit_result.parameters[name].value
        err = fit_result.uncertainties.get(name)
        row = f"{_format_param_label(name)} = {self._fmt_value(value, err)}"
        minos = (getattr(fit_result, "minos_errors", None) or {}).get(name)
        if minos is not None and len(minos) == 2:
            lower, upper = float(minos[0]), float(minos[1])
            row += f" (+{upper:.2g} / {lower:.2g})"
        return row

    @staticmethod
    def _fmt_value(value: float, err: float | None) -> str:
        if err is None or not np.isfinite(err):
            return f"{value:.5g}"
        return f"{value:.5g} ± {err:.2g}"

    @staticmethod
    def _derive_grouped_relationship(
        physics_roles: dict[str, str],
        n_members: int,
    ) -> tuple[str | None, str | None]:
        """Derive the grouped-series relationship from the physics roles.

        Returns ``(relationship, error)``. ``relationship`` is ``individual`` (one
        member), ``global`` (≥1 physics param shared across runs — including the
        mixed case, where the per-run physics are routed through
        ``cross_run_local_params``), or ``batch`` (all physics independent per
        run). ``error`` is reserved for future invalid combinations; mixing Global
        and Local is now supported, so it is always ``None`` here.
        """
        physics_global = [name for name, role in physics_roles.items() if role == "global"]
        if n_members <= 1:
            return "individual", None
        return ("global" if physics_global else "batch"), None

    def set_batch_seeding_mode(self, mode: str) -> None:
        """Set the batch-series seeding mode ("auto"/"as_provided"/"chain").

        Drives the on-tab combobox in lock-step (signals blocked so this does not
        re-emit and bounce back to the menu). Callers: the menu action handler and
        state restore.
        """
        self._batch_seeding_mode = mode
        combo = getattr(self, "_seeding_combo", None)
        if combo is not None:
            index = combo.findData(mode)
            if index >= 0 and index != combo.currentIndex():
                blocked = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(blocked)

    def _on_seeding_combo_changed(self, _index: int) -> None:
        """On-tab seeding selector changed: apply it and notify the menu to mirror."""
        mode = str(self._seeding_combo.currentData() or "auto")
        self._batch_seeding_mode = mode
        self.batch_seeding_mode_changed.emit(mode)

    def _on_coadd_mode_changed(self, _index: int) -> None:
        """In-batch co-add mode changed: refresh the grouped-series context."""
        self._coadd_mode = str(self._coadd_mode_combo.currentData() or "off")
        self._coadd_window_spin.setEnabled(self._coadd_mode != "off")
        self._coadd_window_label.setEnabled(self._coadd_mode != "off")
        self._grouped_context_cache = None
        self._grouped_seed_cache = None
        self._update_mode_ui(preserve_result=False)

    def _on_coadd_window_changed(self, value: int) -> None:
        """In-batch co-add window size changed: refresh the grouped-series context."""
        self._coadd_window = max(2, int(value))
        if self._coadd_mode != "off":
            self._grouped_context_cache = None
            self._grouped_seed_cache = None
            self._update_mode_ui(preserve_result=False)

    def _set_series_busy(self, busy: bool) -> None:
        """Swap the Fit button for a Stop button (and back) around a worker fit."""
        self._stop_btn.setVisible(busy)
        self._stop_btn.setEnabled(busy)
        self._fit_btn.setVisible(not busy)
        if busy:
            self._fit_btn.setEnabled(False)
            # A new fit is starting: clear any stale seeding signpost until the
            # fresh results are diagnosed.
            signpost = getattr(self, "_seeding_signpost", None)
            if signpost is not None:
                signpost.hide()
        else:
            # Re-derive Fit/Preview enabled state from the real gating contract
            # (member count, grouped readiness, _fit_blocked) rather than
            # force-enable — the selection may have changed while the fit ran.
            self._update_mode_ui(preserve_result=True)

    def _on_stop_fit(self) -> None:
        """Request cancellation of the active worker-based fit."""
        # Only one fit runs at a time (shared Fit button); the live handle is
        # the count-domain worker or the global/grouped/series worker, both
        # TaskRunner workers with a cooperative cancel().
        worker = self._count_fit_worker or self._fit_worker
        if worker is not None and hasattr(worker, "cancel"):
            self._stop_btn.setEnabled(False)
            self._result_text.setText("Cancelling fit…")
            worker.cancel()

    def _on_series_fit_cancelled(self) -> None:
        """Handle a cancelled series fit: restore the panel, record nothing."""
        self._set_series_busy(False)
        self._fit_worker = None
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_text.setText("Fit cancelled — no result recorded.")

    @staticmethod
    def _grouped_series_order_key(members: dict) -> dict[int, float] | None:
        """Best-effort run → temperature/field order key from group metadata.

        Auto seeding chains only when this is a usable ordered scan; absent
        temperature/field metadata, this returns ``None`` and Auto safely falls back
        to independent seeds.
        """
        order: dict[int, float] = {}
        for run, groups in members.items():
            value = None
            for group in groups:
                meta = getattr(group, "metadata", None) or {}
                for key in ("temperature", "temperature_k", "field", "field_g"):
                    raw = meta.get(key)
                    if raw is not None:
                        try:
                            value = float(raw)
                        except (TypeError, ValueError):
                            value = None
                        break
                if value is not None:
                    break
            if value is None:
                return None
            order[int(run)] = value
        return order or None

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
        # Physics with the "local" role are fitted per run (shared across that run's
        # groups); the "global" ones stay shared across runs. The grouped engine
        # routes the per-run subset via cross_run_local_params.
        cross_run_local_params = [name for name, role in physics_roles.items() if role == "local"]
        initial_params = {
            run: self._build_grouped_initial_params(groups, grouped_config, run_number=run)
            for run, groups in members.items()
        }

        self._result_text.setText("Fitting grouped time-domain series...")
        self._set_series_busy(True)
        self._current_model = grouped_model
        self._current_global_params = global_params

        # grouped_datasets is GUI-side launch context; bind it into the
        # finished closure (the engine returns only the series result).
        self._fit_worker = _start_fit_call(
            self,
            functools.partial(
                fit_grouped_series,
                relationship,
                members,
                grouped_model.function,
                global_params,
                local_params,
                initial_params,
                minos=self._minos_checkbox.isChecked(),
                seeding=self._batch_seeding_mode,
                order_key=self._grouped_series_order_key(members),
                cost=self._count_fit_cost,
                cross_run_local_params=cross_run_local_params,
                # Independent (as_provided) batches are embarrassingly parallel; let the
                # engine fan the per-run fits across processes. For a large "global"
                # series the same pool drives the block-separable solver's inner per-run
                # fits (no-op for chain).
                max_workers=os.cpu_count(),
                # Large mixed global/local fits are near-separable (runs couple only
                # through the shared physics); let the engine alternate block-wise above
                # its free-parameter threshold instead of one monolithic minimisation.
                block_separable=True,
                # Report rigorous (marginal) shared-parameter errors by profiling them
                # over the locals, rather than the cheaper conditional errors.
                profile_shared_errors=True,
            ),
            on_finished=lambda result, ds=grouped_datasets: self._on_grouped_series_fit_finished(
                ds, result
            ),
            on_error=self._on_fit_error,
            on_cancelled=self._on_series_fit_cancelled,
        )

    def _on_grouped_series_fit_finished(self, grouped_datasets, series_result) -> None:
        """Handle a completed multi-run grouped-series fit (persist + plot).

        Builds per-(run,group) fit curves keyed by the synthetic member key and
        emits ``grouped_fit_completed`` → ``MainWindow._record_grouped_fit_series``
        persists the ``FitSeries(member_kind="groups")``. (Reflecting fitted values
        back into the per-group tables is deferred; the seeds remain shown.)
        """
        self._set_series_busy(False)
        self._fit_worker = None
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
        seeding_reason = getattr(series_result, "seeding_reason", "")
        if seeding_reason:
            stats += f"<br>Seeding: {seeding_reason}"
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
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

        # In batch grouped mode the per-group nuisance table is hidden, so each
        # dataset's groups are seeded from their own data (FFT phase, counts).
        # Precedence per nuisance: dialog override > per-(run, group) auto-seed >
        # the table's shared group_values fallback. The single grouped surface
        # keeps the table (user-editable) as the authoritative source.
        auto_for_run: dict[str, dict[str, float]] = {}
        if not self._grouped_single and run_number is not None:
            auto_for_run = self._grouped_member_nuisance_seeds().get(int(run_number), {})

        # Physics chain-seeding (batch only): when every member has a single
        # grouped fit under the current model, seed each run's Local physics from
        # its own single fit and Global/Fixed from the cross-run average — the
        # grouped analogue of FB's _effective_initial_values_by_run.
        physics_roles = dict(grouped_config.get("physics_roles", {}))
        run_physics_seed: dict[str, float] = {}
        physics_averages: dict[str, float] = {}
        if not self._grouped_single and run_number is not None and self._inherited_seed_by_run:
            if self._inherited_model_dict == self._composite_model.to_dict():
                member_runs = {int(r) for r in self._grouped_members}
                if member_runs and member_runs.issubset(self._inherited_seed_by_run):
                    run_physics_seed = self._inherited_seed_by_run.get(int(run_number), {})
                    physics_averages = self._inherited_param_averages(
                        {r: self._inherited_seed_by_run[r] for r in member_runs},
                        list(model_values),
                    )

        for index, group in enumerate(grouped_groups, start=1):
            user_values: dict[str, float] = {}
            if run_number is not None:
                member_key = _group_dataset_run_number(int(run_number), index)
                user_values = self._user_grouped_initial_values.get(member_key, {})
            params = ParameterSet()
            for name in GROUP_NUISANCE_PARAMS:
                per_group_values = nuisance_group_values.get(name, {})
                value = float(per_group_values.get(group.group_id, 0.0))
                auto_value = auto_for_run.get(name, {}).get(str(group.group_id))
                if auto_value is not None:
                    value = float(auto_value)
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
                seed_value = value
                if run_physics_seed or physics_averages:
                    if physics_roles.get(name) == "local":
                        if name in run_physics_seed:
                            seed_value = float(run_physics_seed[name])
                    elif name in physics_averages:  # global / fixed
                        seed_value = float(physics_averages[name])
                min_val, max_val = bounds[name]
                params.add(
                    Parameter(
                        name=name,
                        value=seed_value,
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
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
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
            # A partial batch passes only the converged members here; a dataset
            # whose fit failed has no entry, so skip it rather than KeyError.
            result = results_dict.get(int(dataset.run_number))
            if result is None:
                continue
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
        # Surface the engine's advisory warnings (scale / fixed-frequency traps)
        # beneath the success line. The same trap typically fires for every run,
        # so dedupe to one row per distinct message (first-seen order) rather than
        # repeating it per dataset.
        seen: set[str] = set()
        for run_result in results_dict.values():
            for message in getattr(run_result, "warnings", None) or []:
                if message not in seen:
                    seen.add(message)
                    self._result_text.append(warning_html("⚠ " + html.escape(str(message))))
        emitted_results = results_dict
        emitted_global = fitted_global
        if self._domain == "frequency":
            emitted_results = {}
            for run_number, result in results_dict.items():
                params, uncertainties = append_frequency_field_derived_parameters(
                    result.parameters,
                    result.uncertainties,
                )
                # Override only the two fields that change (the field-derived
                # parameters and their uncertainties); replace() carries every
                # other FitResult field through, so dof, minos_errors, warnings,
                # and any field added later survive instead of resetting to
                # defaults (a hand-copied constructor silently dropped dof and
                # minos_errors — see PR #108 follow-up).
                emitted_results[run_number] = dataclasses.replace(
                    result,
                    parameters=params,
                    uncertainties=uncertainties,
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
        self._set_series_busy(False)
        self._fit_worker = None
        self._update_mode_ui(preserve_result=True)

        model = self._current_model
        global_params = self._current_global_params

        # A partial batch failure must not discard the runs that converged: build
        # the series from the successful members and surface the failures as a
        # non-blocking warning. Only an all-failed batch takes the abort branch.
        successful = {run: r for run, r in results_dict.items() if r.success}
        failed = [run for run, r in results_dict.items() if not r.success]
        run_label_by_number = {ds.run_number: ds.run_label for ds in self._datasets}
        failed_labels = [run_label_by_number.get(run, str(run)) for run in failed]

        if successful:
            self._emit_global_fit_success(
                model=model,
                results_dict=successful,
                fitted_global=fitted_global,
                global_param_names=global_params,
            )
            if failed:
                # _emit_global_fit_success rendered the success box; append the
                # failure warning as a new paragraph beneath it rather than
                # overwriting it (non-blocking surfacing).
                self._result_text.append(
                    warning_html(f"{len(failed)} run(s) failed to converge: {failed_labels}")
                )
            self._append_series_seeding_note()
        else:
            self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
            self._result_text.setText(
                f"<b>Batch fit failed</b><br>Failed datasets: {failed_labels}"
            )

        # Inspect the per-run trend for the near-transition collapse/outlier
        # signature and signpost the per-run warm-start when it is present.
        self._update_seeding_signpost(model, results_dict)

    def _append_series_seeding_note(self) -> None:
        """Append the resolved seeding mode/reason and any reseeded runs."""
        meta = self._series_seeding_meta
        if not meta:
            return
        reason = str(meta.get("seeding_reason") or "")
        if reason:
            self._result_text.append(info_html(f"Seeding: {reason}"))
        reseeded = meta.get("reseeded_runs") or ()
        if reseeded:
            runs = ", ".join(str(r) for r in reseeded)
            self._result_text.append(
                info_html(f"Reseeded {len(reseeded)} run(s) off the spurious branch: {runs}")
            )

    def _update_seeding_signpost(self, model: object, results_dict: dict) -> None:
        """Diagnose the batch trend and show/hide the per-run-seed signpost.

        Builds per-run summaries (scan order + fitted amplitude/frequency) and runs
        the shared :func:`diagnose_series`. When a run collapsed to ~0 amplitude, sits
        off the frequency trend, or failed, the signpost is shown with the
        descending-frequency seeds the diagnostics computed; otherwise it is hidden.
        """
        self._suggested_series_seeds = {}
        signpost = getattr(self, "_seeding_signpost", None)
        if signpost is None:
            return
        param_names = list(getattr(model, "param_names", []) or [])
        if not param_names or len(results_dict) < 3:
            signpost.hide()
            return
        amplitude_param, frequency_param = resolve_series_params(param_names)
        if amplitude_param is None and frequency_param is None:
            signpost.hide()
            return
        order_key = self._asymmetry_series_order_key() or {}
        points: list[SeriesPoint] = []
        for run, result in results_dict.items():
            run = int(run)
            values = {p.name: p.value for p in getattr(result, "parameters", [])}
            points.append(
                SeriesPoint(
                    run=run,
                    order=float(order_key.get(run, run)),
                    amplitude=values.get(amplitude_param) if amplitude_param else None,
                    frequency=values.get(frequency_param) if frequency_param else None,
                    success=bool(getattr(result, "success", False)),
                )
            )
        diagnostics = diagnose_series(
            points, amplitude_param=amplitude_param, frequency_param=frequency_param
        )
        if not diagnostics.has_issues:
            signpost.hide()
            return
        self._suggested_series_seeds = dict(diagnostics.suggested_seeds)
        message = (
            f"<b>The {self._trend_axis_label()} trend has outliers.</b> "
            f"{diagnostics.reason[:1].upper() + diagnostics.reason[1:]}. "
            "Near-transition oscillatory fits are bistable — a per-run warm-start "
            "fixes it."
        )
        self._seeding_signpost_label.setText(info_html(message))
        self._apply_suggested_seeds_btn.setEnabled(bool(self._suggested_series_seeds))
        signpost.show()

    def _trend_axis_label(self) -> str:
        """Friendly name for the leading oscillatory parameter, for the signpost."""
        amplitude_param, frequency_param = resolve_series_params(
            list(getattr(self._current_model, "param_names", []) or [])
        )
        if frequency_param:
            return "frequency"
        if amplitude_param:
            return "amplitude"
        return "parameter"

    def _apply_suggested_series_seeds(self) -> None:
        """Apply the diagnostics' descending per-run seeds and re-run the batch.

        Merges the suggested frequency/amplitude seeds into the per-run Initial
        Values, switches to Independent seeds (so the warm-start is honoured as-is
        rather than overwritten by chaining — the proven manual recipe), then re-runs.
        """
        if not self._suggested_series_seeds:
            return
        for run, seed in self._suggested_series_seeds.items():
            merged = dict(self._user_initial_values_by_run.get(int(run), {}))
            merged.update({k: float(v) for k, v in seed.items()})
            self._user_initial_values_by_run[int(run)] = merged
        # Independent seeds honours per-run Initial Values verbatim; mirror it into
        # the menu via the existing sync signal.
        self.set_batch_seeding_mode("as_provided")
        self.batch_seeding_mode_changed.emit("as_provided")
        self._seeding_signpost.hide()
        self._result_text.append(
            info_html(
                "Applied descending per-run frequency seeds (Independent seeds). Re-running batch…"
            )
        )
        self._run_global_fit()

    def _on_fit_error(self, error_msg: str) -> None:
        """Handle fit error."""
        self._set_series_busy(False)
        self._fit_worker = None
        self._update_mode_ui(preserve_result=True)
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
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

    def update_grouped_phase_seed(self, run_number: int, phases_rad: dict[int, float]) -> bool:
        """Write per-group phases (radians) into the cached grouped fit seed.

        Used by the MaxEnt "Send phases to fit" exchange to push MaxEnt phases
        back onto the grouped time-domain fit's per-group ``relative_phase``.
        Returns ``True`` when an existing seed was updated, ``False`` when the
        run has no grouped fit to receive the phases.
        """
        try:
            seed = self._grouped_simulate_seed.get(int(run_number))
        except (TypeError, ValueError):
            return False
        if not isinstance(seed, dict) or not isinstance(seed.get("specs"), list):
            return False
        updated = False
        for spec in seed["specs"]:
            gid = spec.get("group_id")
            if gid in phases_rad:
                spec["relative_phase"] = float(phases_rad[gid])
                updated = True
        return updated

    def _on_grouped_fit_finished(self, grouped_datasets: list[MuonDataset], grouped_result) -> None:
        """Handle successful grouped fit completion."""
        self._set_series_busy(False)
        self._fit_worker = None
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
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_text.setHtml(success_html("Grouped fit converged", detail=stats))
        self.grouped_fit_completed.emit(grouped_datasets, results_with_curves)

        # Publish the run's shared physics so the batch grouped surface can
        # chain-seed each run from its own single grouped fit (FB parity).
        if self._grouped_single and self._current_dataset is not None:
            physics_values = {
                str(parameter.name): float(parameter.value)
                for parameter in getattr(grouped_result, "shared_parameters", [])
                if isinstance(getattr(parameter, "name", None), str)
                and np.isfinite(float(getattr(parameter, "value", float("nan"))))
            }
            try:
                run_number = int(self._current_dataset.run_number)
            except (TypeError, ValueError):
                run_number = None
            if run_number is not None and physics_values and self._composite_model is not None:
                self.single_grouped_fit_recorded.emit(
                    run_number, self._composite_model, physics_values
                )

    def register_grouped_single_fit_seed(
        self, run_number: int, model: CompositeModel, values_by_name: dict[str, float]
    ) -> None:
        """Store a single grouped fit's shared physics for batch chain-seeding.

        Mirrors :meth:`register_single_fit_seed` (which reads a ``FitResult``) but
        takes the already-extracted physics values from a grouped fit, then
        refreshes the inherited-seed cache (grouped-aware).
        """
        finite_values: dict[str, float] = {}
        for name, value in (values_by_name or {}).items():
            if not isinstance(name, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                finite_values[name] = numeric
        if not finite_values:
            return
        try:
            run_key = int(run_number)
        except (TypeError, ValueError):
            return
        self._single_fit_seed_by_run[run_key] = {
            "model": model.to_dict(),
            "values": finite_values,
        }
        self._refresh_inherited_single_fit_defaults()

    def shutdown_workers(self) -> None:
        """Cancel any running fit and wait for its thread (window close).

        Every fit runs on ``_fit_call_runner``, whose ``shutdown`` is bounded
        and Windows-safe: cancellation is cooperative (polled between cost
        evaluations), so a timed-out wait degrades to a reaped (leaked) thread
        rather than hanging closeEvent for the rest of a long migrad/MINOS step.
        """
        self._fit_call_runner.shutdown()

    def wait_for_fit(self, timeout_ms: int = 30_000) -> bool:
        """Block (with a live event loop) until the launched fit completes."""
        return _wait_for_fit_thread(self, timeout_ms)

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
                # As in the single-fit tab: unregistered (user) component names
                # degrade to named placeholders rather than silently replacing
                # the saved model.
                restored = CompositeModel.from_dict(composite_data, allow_missing=True)
            except ValueError:
                self._set_composite_model(
                    CompositeModel(["Exponential", "Constant"], operators=["+"])
                )
            else:
                self._set_composite_model(restored)
                if restored.missing_component_names:
                    names = ", ".join(restored.missing_component_names)
                    self._result_text.setText(
                        f"Missing user function(s): {names}. The saved model is "
                        "preserved (missing components plot as zero) but cannot be "
                        "fitted until they are registered — see Setup → User functions…"
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
        if isinstance(self._group_model_table, FitParameterTable):
            # The single grouped physics table has separate min/max columns (not a
            # single "min, max" bounds column), so the fraction 0–1 bounds must go
            # into COL_MIN/COL_MAX — matching the populate() path. Passing
            # bounds_column=3 here would write "0, 1" into the min field.
            _configure_fraction_rows_in_table(
                self._group_model_table,
                self._grouped_fit_model(),
                min_column=FitParameterTable.COL_MIN,
                max_column=FitParameterTable.COL_MAX,
            )
        else:
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
        # The batch grouped surface hides the per-group nuisance table — its
        # per-(run, group) values are auto-seeded and edited via the dialog
        # ("Edit per-group initial values…"). The single grouped surface keeps it.
        self._group_param_group.setVisible(grouped and self._grouped_single)
        self._group_model_group.setVisible(grouped)
        # In-batch co-add only applies to grouped-series fits (≥2 members).
        self._coadd_group.setVisible(grouped)
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
        key = (member_ids, bool(self._fit_blocked), self._coadd_mode, int(self._coadd_window))
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

        member_datasets, coadd_note = self._apply_inbatch_coadd(member_datasets)

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
            if coadd_note:
                reason = f"{coadd_note} {reason}"
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
        if coadd_note:
            message = f"{coadd_note} {message}"
        if skipped:
            message += f" (skipped {len(skipped)}: {'; '.join(skipped)})"
        return representative_groups, grouped_datasets, message

    def _apply_inbatch_coadd(
        self, member_datasets: list[MuonDataset]
    ) -> tuple[list[MuonDataset], str]:
        """Co-add successive members per the Smooth/Bin control before fitting.

        Returns ``(transformed_members, note)``. With co-add off, fewer than two
        members, or no source histograms available, the members pass through
        unchanged. Each co-add window sums the raw histograms of its members via
        :func:`combine_runs` and reduces the combined run to a member dataset, so
        the chain-seeding and grouped-contract paths see ordinary runs.
        """
        if self._coadd_mode == "off" or len(member_datasets) < 2:
            return member_datasets, ""

        windows = coadd_member_windows(
            len(member_datasets), mode=self._coadd_mode, window=self._coadd_window
        )
        if not windows:
            return (
                member_datasets,
                f"Co-add window of {self._coadd_window} exceeds the "
                f"{len(member_datasets)} selected runs; co-add skipped.",
            )

        verb = "binned" if self._coadd_mode == "bin" else "smoothed"
        combined: list[MuonDataset] = []
        failures = 0
        for indices in windows:
            window_datasets = [member_datasets[i] for i in indices]
            # Carry each dataset's displayed scalar overrides onto the run copies
            # so the combined member's event-weighted T/field match the browser.
            runs = runs_with_dataset_metadata(window_datasets)
            if len(runs) != len(window_datasets):
                # A member without source histograms can't be co-added; keep the
                # window's members un-combined rather than silently dropping data.
                combined.extend(window_datasets)
                failures += 1
                continue
            try:
                combined_run = combine_runs(runs, sign=1)
                combined.append(reduce_combined_run(combined_run))
            except (CombineError, ValueError):
                combined.extend(window_datasets)
                failures += 1
        note = f"Co-add ({verb}, {self._coadd_window} runs/fit): {len(windows)} combined members."
        if failures:
            note += f" ({failures} window(s) left un-combined — no source histograms.)"
        return combined, note

    def _setup_group_nuisance_table(self) -> None:
        self._rebuild_group_nuisance_table(preserved_state=None)

    def _grouped_member_nuisance_seeds(self) -> dict[int, dict[str, dict[str, float]]]:
        """Per-(run, group) nuisance auto-seeds: ``{run -> {param -> {group_id: value}}}``.

        Each member dataset's own groups are seeded independently (its own FFT
        phase and count statistics) via the shared seeding helpers, so a batch
        fit starts every dataset from its own estimates rather than a single
        representative run's. Computed lazily and cached under the same key as the
        grouped context, so no FFTs run on table rebuilds or selection changes —
        only when a fit launches or the per-group dialog opens.
        """
        self._grouped_mode_context()  # ensure _grouped_members is current
        member_ids = tuple(id(ds) for ds in self._grouped_member_datasets())
        key = (member_ids, bool(self._fit_blocked), self._coadd_mode, int(self._coadd_window))
        cache = self._grouped_seed_cache
        if cache is not None and cache[0] == key:
            return cache[1]

        seeds: dict[int, dict[str, dict[str, float]]] = {}
        for run, groups in self._grouped_members.items():
            phases = _seed_group_absolute_phases(groups)
            per_param: dict[str, dict[str, float]] = {name: {} for name in GROUP_NUISANCE_PARAMS}
            for group in groups:
                gid = str(getattr(group, "group_id", ""))
                counts = np.asarray(getattr(group, "counts", []), dtype=float)
                if counts.size:
                    background, n0, amplitude = _seed_group_background_and_n0(
                        counts, time=getattr(group, "time", None)
                    )
                    per_param["N0"][gid] = n0
                    per_param["background"][gid] = background
                    per_param["amplitude"][gid] = amplitude
                if gid in phases and "relative_phase" in per_param:
                    per_param["relative_phase"][gid] = phases[gid]
            seeds[int(run)] = per_param

        self._grouped_seed_cache = (key, seeds)
        return seeds

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
            # Phase is periodic, so the bounds give a full 2*pi of slack on either
            # side of the principal range. Absolute per-group seeds (wrapped to
            # (-pi, pi]) routinely land near +/-pi — e.g. the backward group of an
            # F-B pair sits ~pi from the forward group — and tight (-pi, pi] bounds
            # would trap such a seed on a limit with no wrap-around room.
            "relative_phase": (0.0, "Local", f"{-2.0 * np.pi:.6g}, {2.0 * np.pi:.6g}"),
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
        self._group_param_table.setColumnWidth(0, PARAM_NAME_COL_WIDTH)
        for offset in range(len(value_headers)):
            self._group_param_table.setColumnWidth(1 + offset, 78)
        self._group_param_table.setColumnWidth(self._group_param_type_column(), 86)
        self._group_param_table.setColumnWidth(self._group_param_bounds_column(), 104)
        self._group_param_table.setRowCount(len(GROUP_NUISANCE_PARAMS))

        n0_defaults_by_group: dict[str, float] = {}
        background_defaults_by_group: dict[str, float] = {}
        amplitude_defaults_by_group: dict[str, float] = {}
        relative_phase_defaults_by_group: dict[str, float] = {}
        # Only the (visible) single grouped table is FFT-seeded here. The batch
        # table is hidden and its per-(run, group) seeds come from the lazy
        # _grouped_member_nuisance_seeds helper at fit/dialog time, so skip the
        # per-group FFT/count estimates here to avoid a rebuild-time FFT storm.
        if self._grouped_single:
            # Grouped fits hold the shared model phase fixed at zero, so each
            # group's per-group phase nuisance carries the full *absolute* FFT
            # phase estimate rather than an offset relative to the first group.
            relative_phase_defaults_by_group = _seed_group_absolute_phases(grouped_groups)
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
            # The single grouped fit shares physics across the dataset's groups, so
            # its nuisance "shared across groups" option reads "Shared"; "Global"
            # (shared across runs) is reserved for the multi-run batch grouped fit.
            shared_label = "Shared" if self._grouped_single else "Global"
            type_combo.addItems([shared_label, "Local", "Fixed"])
            current_type = str(previous.get("type") or type_text)
            if current_type in ("Global", "Shared"):
                current_type = shared_label
            type_combo.setCurrentText(current_type)
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
        _size_param_table_to_content(self._group_param_table)

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

    def current_grouped_seed_values(self) -> dict[str, str]:
        """Return the live grouped physics-table seed text keyed by parameter name.

        The grouped analogue of :meth:`current_seed_values`: used to seed the
        Batch grouped surface from the Single grouped surface's physics values.
        """
        return {
            name: str(entry.get("value", ""))
            for name, entry in self._current_grouped_model_row_state().items()
            if str(entry.get("value", "")).strip()
        }

    def apply_grouped_physics_seeds(self, seed_values: dict[str, str]) -> None:
        """Write physics seed values into the grouped physics table by name.

        Both the single (FitParameterTable) and batch (combo) physics tables keep
        the name in column 0 and the value in column 1, so one path serves both.
        """
        if not seed_values:
            return
        table = self._group_model_table
        blocked = table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                name_item = table.item(row, 0)
                name = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
                if not isinstance(name, str):
                    name = name_item.text() if name_item is not None else None
                if name in seed_values:
                    value_item = table.item(row, 1)
                    if value_item is not None:
                        value_item.setText(str(seed_values[name]))
        finally:
            table.blockSignals(blocked)

    def _rebuild_grouped_model_table(self, preserved_state: dict[str, dict[str, str]]) -> None:
        grouped_model = self._grouped_fit_model()
        grouped_groups, _grouped_datasets, _message = self._grouped_mode_context()
        visible_param_names = [
            pname for pname in grouped_model.param_names if not is_amplitude_parameter(pname)
        ]
        if self._grouped_single:
            self._rebuild_grouped_single_model_table(
                grouped_model, visible_param_names, preserved_state
            )
            return
        self._updating_group_model_fraction_values = True
        self._group_model_table.setRowCount(len(visible_param_names))
        for row, pname in enumerate(visible_param_names):
            previous = preserved_state.get(pname, {})
            base_name, _index = split_parameter_name(pname)
            name_item = _make_param_name_item(_format_param_label(pname), pname)
            self._group_model_table.setItem(row, 0, name_item)

            default_val = grouped_model.param_defaults.get(pname, 0.0)
            default_type = "Global"
            if is_background_parameter(pname):
                default_val = 0.0
                default_type = "Fixed"
            elif base_name == "phase":
                # The per-group phase nuisance carries the absolute phase, so the
                # shared model phase is fixed at zero by default (user-adjustable).
                default_val = 0.0
                default_type = "Fixed"
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
        _size_param_table_to_content(self._group_model_table)

    def _rebuild_grouped_single_model_table(
        self,
        grouped_model: CompositeModel,
        visible_param_names: list[str],
        preserved_state: dict[str, dict[str, str]],
    ) -> None:
        """Populate the single grouped fit's physics table (shared Fix tickbox).

        Reuses :class:`FitParameterTable`. ``preserved_state`` (the shared
        {value, type, bounds} shape, from the edit-rebuild capture or a restored
        project) seeds value / Fix / bounds; otherwise:

        - ``field``/``B_L`` params seed from the run's applied field (every such
          param, so models with more than one oscillatory component all start at
          the applied field rather than the 100 G component default);
        - background params default fixed at 0;
        - ``phase`` params default fixed at 0 — the individual-groups fit holds
          the shared oscillation phase at zero and carries the full per-group
          phase in the ``relative_phase`` nuisances, removing the degeneracy
          between a shared phase and the per-group phase offsets.
        """
        table = self._group_model_table
        field_overrides = _field_value_overrides(
            grouped_model, float(self._applied_field_default_gauss)
        )
        value_overrides: dict[str, float] = {}
        fixed_names: set[str] = set()
        preserved_bounds: dict[str, tuple[str, str]] = {}
        for pname in visible_param_names:
            prev = preserved_state.get(pname, {})
            prev_value = str(prev.get("value", "")).strip()
            base_name, _index = split_parameter_name(pname)
            if prev_value:
                try:
                    value_overrides[pname] = float(prev_value)
                except ValueError:
                    pass
            elif is_background_parameter(pname):
                value_overrides[pname] = 0.0
            elif base_name == "phase":
                value_overrides[pname] = 0.0
            elif pname in field_overrides:
                value_overrides[pname] = field_overrides[pname]

            default_fixed = is_background_parameter(pname) or base_name == "phase"
            if str(prev.get("type", "")) == "Fixed" or (not prev and default_fixed):
                fixed_names.add(pname)

            bounds_text = str(prev.get("bounds", "")).strip()
            if bounds_text:
                try:
                    lo, hi = (part.strip() for part in bounds_text.split(",", maxsplit=1))
                    preserved_bounds[pname] = (lo, hi)
                except ValueError:
                    pass

        table.populate(
            grouped_model,
            param_names=visible_param_names,
            value_overrides=value_overrides,
            fixed_names=fixed_names,
        )
        # populate() resets bounds to defaults; restore any the user/project had.
        for row in range(table.rowCount()):
            name_item = table.item(row, FitParameterTable.COL_NAME)
            name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(name, str) or name not in preserved_bounds:
                continue
            lo, hi = preserved_bounds[name]
            min_item = table.item(row, FitParameterTable.COL_MIN)
            max_item = table.item(row, FitParameterTable.COL_MAX)
            if min_item is not None:
                min_item.setText(lo)
            if max_item is not None:
                max_item.setText(hi)

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
            # "Shared" is the single grouped fit's label for cross-group sharing;
            # it behaves exactly like "Global" (shared across the run's groups).
            if type_text == "Shared":
                type_text = "Global"
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

        if self._grouped_single:
            # Single grouped fit: physics params are read from the Fix-tickbox
            # table. Every group shares the function, so a free param is "global"
            # (shared across the dataset's groups) and a ticked one is "fixed";
            # there is no per-run "local" classification for one dataset.
            for param in self._group_model_table.read_parameter_set():
                pname = param.name
                value = float(param.value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} must be finite, got {value}"
                    )
                min_val, max_val = float(param.min), float(param.max)
                if np.isfinite(min_val) and value < min_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} "
                        f"is below minimum {min_val}"
                    )
                if np.isfinite(max_val) and value > max_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} "
                        f"is above maximum {max_val}"
                    )
                if param.fixed:
                    fixed_params.append(pname)
                    physics_roles[pname] = "fixed"
                else:
                    global_params.append(pname)
                    physics_roles[pname] = "global"
                model_values[pname] = value
                bounds[pname] = (min_val, max_val)
        else:
            for row in range(self._group_model_table.rowCount()):
                name_item = self._group_model_table.item(row, 0)
                pname = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
                if not isinstance(pname, str):
                    pname = name_item.text() if name_item else f"model_param_{row}"

                try:
                    value = float(self._group_model_table.item(row, 1).text())
                except (TypeError, ValueError, AttributeError):
                    raise ValueError(
                        f"Error: Invalid value for {_format_param_label(pname)}"
                    ) from None
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
                        f"Error: Parameter {_format_param_label(pname)} value {value} "
                        f"is below minimum {min_val}"
                    )
                if np.isfinite(max_val) and value > max_val:
                    raise ValueError(
                        f"Error: Parameter {_format_param_label(pname)} value {value} "
                        f"is above maximum {max_val}"
                    )

                type_combo = self._group_model_table.cellWidget(row, 2)
                type_text = (
                    type_combo.currentText() if isinstance(type_combo, QComboBox) else "Global"
                )
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
        if isinstance(table, FitParameterTable):
            # The single grouped physics table serialises in the same
            # {value, type, bounds} shape as the combo tables, with type =
            # Fixed / Shared (Shared = the run's groups share the value).
            for entry in table.parameters_state():
                name = str(entry["name"])
                state[name] = {
                    "value": str(entry.get("value", 0.0)),
                    "type": "Fixed" if entry.get("fixed") else "Shared",
                    "bounds": f"{entry.get('min', '-inf')}, {entry.get('max', 'inf')}",
                }
            return state
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
        if isinstance(table, FitParameterTable):
            # Apply the saved {value, type, bounds} entries onto the Fix-tickbox
            # table: type "Fixed" → checked, bounds → min/max.
            params_data: dict[str, dict] = {}
            for name, entry in by_name.items():
                bounds = str(entry.get("bounds", "-inf, inf"))
                try:
                    lo, hi = (part.strip() for part in bounds.split(",", maxsplit=1))
                except ValueError:
                    lo, hi = "-inf", "inf"
                params_data[name] = {
                    "name": name,
                    "value": entry.get("value", 0.0),
                    "fixed": str(entry.get("type", "")) == "Fixed",
                    "min": lo,
                    "max": hi,
                }
            table.restore_parameters(params_data)
            return
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


class _CurrentPageTabWidget(CurrentPageSizingMixin, QTabWidget):
    """A QTabWidget sized by its *current* tab, not the maximum over all tabs.

    A plain QTabWidget reports the largest size hint across every page, so the
    wide Batch tab would impose its width on the dock even while the compact
    Single tab is showing — forcing the inspector scroll area to scroll
    horizontally (a second scrollbar on top of the parameter table's own).
    Sizing to the visible tab (plus the tab bar) lets the dock follow it.
    """

    def _page_extra(self) -> QSize:
        return self.tabBar().sizeHint()


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
    # Forwarded from the Batch tab's on-tab seeding selector so the main window's
    # Analysis ▸ Batch seeding menu can mirror it (two-way sync).
    batch_seeding_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._single_state_by_run: dict[int, dict] = {}
        self._active_single_run_number: int | None = None
        # Optional mediator that supplies a per-(run, representation, projection)
        # single-fit restore payload, installed by the main window.  It keeps the
        # panel decoupled from the project model: ``set_dataset`` asks it for the
        # form payload to show, falling back to the run-keyed blob when unset or
        # when it returns ``None``.  See ``set_single_fit_restore_provider``.
        self._single_fit_restore_provider: Callable[[MuonDataset | None], dict | None] | None = None
        self._all_datasets: list[MuonDataset] = []  # Track all datasets for group sharing
        # Active single-fit projection (driven by the main window via
        # ``set_active_projection_label``); part of the binding identity that
        # guards the Single↔Batch tab-switch snapshot below.
        self._active_single_projection: str | None = None
        # Snapshot of the single-fit form taken when the user leaves the Single
        # tab, restored when they return to it for the *same* binding. Without
        # this, switching to Batch and back loses a hand-built (unfit) model:
        # once a run has been batched its per-projection slot exists but is
        # empty, so the restore provider blanks the form to the default model.
        self._single_form_snapshot: dict | None = None
        self._domain = "time"
        self._single_state_by_domain: dict[str, dict] = {}
        self._global_state_by_domain: dict[str, dict] = {}
        self._ui_state_by_domain: dict[str, dict] = {}

        # Create tab widget (sized to the visible tab so the wide Batch tab
        # doesn't force the dock — and a window-level horizontal scrollbar —
        # while the compact Single tab is showing).
        self._tabs = _CurrentPageTabWidget()

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
        self._global_tab.batch_seeding_mode_changed.connect(self.batch_seeding_mode_changed.emit)
        self._tabs.addTab(self._global_tab, "Batch")

        # Preserve the single-fit form across a Single↔Batch view switch (see #3
        # / _single_form_snapshot). Connected last so both tabs exist.
        self._tabs.currentChanged.connect(self._on_fit_tab_changed)

        # Echo of the projection a single fit is currently bound to (vector
        # multi-subplot view); hidden when fitting the default/non-projection
        # asymmetry. Driven by the main window via set_active_projection_label.
        self._projection_echo = QLabel("")
        self._projection_echo.setContentsMargins(6, 2, 6, 2)
        self._projection_echo.hide()
        layout.addWidget(self._projection_echo)

        layout.addWidget(self._tabs)

    def set_active_projection_label(self, projection: str | None, tint: str | None = None) -> None:
        """Show/hide the 'Fitting: <projection>' echo for the bound projection.

        ``tint`` colours the text to match the projection's subplot frame.
        """
        # Track the projection as part of the tab-switch snapshot's binding
        # identity: a snapshot only restores onto the same (run, projection).
        if projection != self._active_single_projection:
            self._single_form_snapshot = None
        self._active_single_projection = projection
        if not hasattr(self, "_projection_echo"):
            return
        if projection:
            self._projection_echo.setText(f"Fitting: {projection}")
            self._projection_echo.setStyleSheet(f"color: {tint}; font-weight: 500;" if tint else "")
            self._projection_echo.show()
        else:
            self._projection_echo.clear()
            self._projection_echo.setStyleSheet("")
            self._projection_echo.hide()

    def _on_fit_tab_changed(self, index: int) -> None:
        """Preserve the single-fit form across a Single↔Batch view switch.

        Leaving the Single tab snapshots its form; returning restores that
        snapshot when the binding (run + projection) is unchanged. This keeps a
        hand-built but unfit model alive across the round trip — without it, once
        a run has been batched its per-projection slot exists but is empty, so
        the restore provider blanks the form to the default model on re-bind.
        """
        single_index = self._tabs.indexOf(self._single_tab)
        if index == single_index:
            snapshot = self._single_form_snapshot
            if snapshot is not None and snapshot.get("run") == self._active_single_run_number:
                self._single_tab.restore_state(snapshot["state"])
        else:
            self._single_form_snapshot = {
                "run": self._active_single_run_number,
                "state": self.get_single_form_state(),
            }

    def set_batch_seeding_mode(self, mode: str) -> None:
        """Forward the batch-series seeding mode to the Batch tab."""
        self._global_tab.set_batch_seeding_mode(mode)

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_rrf_frequency_provider(self, provider: Callable[[], float | None]) -> None:
        """Forward the rotating-frame ν₀ provider to the single-fit tab."""
        self._single_tab.set_rrf_frequency_provider(provider)

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
        # Drop the tab-switch snapshot too, so a stale form can't be restored
        # onto the cleared panel when setCurrentIndex(0) below re-enters Single.
        self._single_form_snapshot = None
        self._active_single_projection = None
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

        # The main window's restore mediator is authoritative when it has an
        # opinion: a payload (possibly an empty dict, meaning "blank this unfit
        # projection") restores from the per-(run, representation, projection)
        # slot — the canonical store for single fits. ``None`` means "no
        # opinion", so fall back to the run-keyed blob (default slot / legacy
        # projects). Consulting it first avoids restoring the form twice.
        payload = (
            self._single_fit_restore_provider(dataset)
            if self._single_fit_restore_provider is not None
            else None
        )
        if payload is not None:
            self.restore_single_fit_ui(payload)
        elif run_number in self._single_state_by_run:
            self._single_tab.restore_state(self._single_state_by_run[run_number])
        else:
            # An unseen dataset the user has not customised inherits the model
            # and parameter setup currently shown (carry-forward) instead of
            # snapping back to the default on every row change. Its own model is
            # still saved and restored on return; only the previous run's fit
            # *result* is dropped (it belongs to the run it was computed on).
            self._carry_forward_single_fit_form()

    def _carry_forward_single_fit_form(self) -> None:
        """Inherit the previous selection's model + parameter setup, sans result.

        Reuses the seen-dataset restore path (so the composite model, seeds,
        bounds, fixed/free flags and link groups all transfer faithfully) but
        clears the fitted uncertainties and result label first — an unseen run
        has not been fit, so it must not display another run's result.
        """
        state = self._single_tab.get_state()
        for entry in state.get("parameters", []):
            entry["uncertainty"] = None
            entry["uncertainty_asymmetric"] = None
        self._single_tab.restore_state(state)
        if not self._single_tab._composite_model.missing_component_names:
            self._single_tab._result_label.setText("No fit performed yet")

    def _reset_single_fit_form(self) -> None:
        """Blank the single-fit form to its domain default ("No fit yet")."""
        default_model = (
            default_frequency_model()
            if self._domain == "frequency"
            else CompositeModel(["Exponential", "Constant"], operators=["+"])
        )
        self._single_tab._set_composite_model(default_model)
        self._single_tab._result_label.setText("No fit performed yet")

    def set_single_fit_restore_provider(
        self, provider: Callable[[MuonDataset | None], dict | None] | None
    ) -> None:
        """Install the per-projection single-fit restore mediator (or clear it).

        The main window passes a callable that maps the dataset being bound to
        the persisted single-fit form payload for the active ``(run,
        representation, projection)`` slot — or ``None`` to defer to the
        panel's own run-keyed state (the default / legacy-project path).
        """
        self._single_fit_restore_provider = provider

    def get_single_form_state(self) -> dict:
        """Return the single-fit *form* payload (no per-run/domain wrapping).

        This is exactly what :meth:`restore_single_fit_ui` consumes, so it is the
        payload the main window stores as a slot's ``ui_state``.
        """
        return copy.deepcopy(self._single_tab.get_state())

    def restore_single_fit_ui(self, payload: dict | None) -> None:
        """Restore (or blank) the single-fit form from a slot ``ui_state`` payload.

        A populated dict restores the form verbatim; an empty dict (or ``None``)
        blanks it — an unfit projection must never inherit another projection's
        fit. The run-keyed blob is deliberately *not* touched: it stays the
        per-run store that global seeding and group sharing read, while the
        per-projection slot is the source of truth for the single-fit form.
        """
        if isinstance(payload, dict) and payload:
            self._single_tab.restore_state(payload)
            # A real persisted fit is now shown; drop any stale tab-switch
            # snapshot so it can't override this fit on the next return to Single.
            self._single_form_snapshot = None
        else:
            self._reset_single_fit_form()

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

    def single_fit_model_and_seed(self) -> tuple[CompositeModel, ParameterSet]:
        """Return the active single-fit model and seed (for headless re-fits)."""
        return self._single_tab.model_and_seed()

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

    def single_fit_range_text(self) -> str | None:
        """Active single-fit range as a provenance string (see SingleFitTab)."""
        return self._single_tab.current_fit_range_text()

    def batch_fit_range_text(self) -> str | None:
        """Active batch/global/grouped fit range as a provenance string."""
        return self._global_tab.current_fit_range_text()

    def send_single_model_to_batch(self) -> bool:
        """Copy the single-fit tab's model and current seeds into the Batch tab.

        Returns ``True`` when a model was sent. The Single ⇄ Batch flow: build a
        model in Single, send it to seed a batch over the selected runs. The
        batch parameter seeds are taken from the single tab's current table
        values (which reflect the latest fit once one has run), so the batch
        starts from the values the user just set rather than model defaults or
        stale preserved state (BUG B8c).
        """
        model = getattr(self._single_tab, "_composite_model", None)
        if model is None:
            return False
        seed_values = self._single_tab.current_seed_values()
        self._global_tab._set_composite_model(model, seed_values=seed_values)
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

    def shutdown_workers(self) -> None:
        """Cancel running fits on both tabs and wait for their threads."""
        self._single_tab.shutdown_workers()
        self._global_tab.shutdown_workers()
