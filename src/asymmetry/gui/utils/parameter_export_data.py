"""Data-only parameter export snapshots and shared table writers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.fitting.axis_transforms import (
    AxisTransform,
)
from asymmetry.core.fitting.parameter_models import (
    ModelFitRange,
    effective_range_bounds,
)
from asymmetry.core.fitting.parameters import (
    get_param_info,
)
from asymmetry.core.utils.angles import wrap_angle_deg
from asymmetry.gui.tasks import TaskCancelledError
from asymmetry.gui.utils import gle_export

_PARAMETER_FIT_CURVE_SAMPLE_COUNT = 800


_IDENTITY_TRANSFORM = AxisTransform.identity()


_UNSET = object()


def _gle_series_color(index: int, override: str | None = None) -> str:
    """Named GLE colour for exported curve *index*, or *override* when given.

    *override* is the active series' phase colour when the curve being
    coloured is that whole series' identity colour (a lone y-parameter, or one
    subplot per parameter) — mirroring the on-screen substitution in
    :meth:`FitParametersPanel._single_series_color`.
    """
    if override is not None:
        return override
    primary = ["black", "blue", "red"]
    fallback = ["green", "orange", "purple", "brown", "magenta", "cyan", "olive"]
    if index < len(primary):
        return primary[index]
    return fallback[(index - len(primary)) % len(fallback)]


def _fit_overlay_color(index: int) -> str:
    colors = ["red", "green", "orange", "purple", "brown", "magenta", "cyan"]
    return colors[index % len(colors)]


def _fit_overlay_label(base: str, index: int, total: int) -> str:
    suffix = "" if index == 0 else f" #{index + 1}"
    if total <= 1:
        suffix = ""
    return f"fit {base}{suffix}"


def _safe_data_name(value: object) -> str:
    """Build a filesystem-safe stem for generated GLE data files."""
    text = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    return text or "series"


def _normalize_x_key(value: object) -> str:
    """Normalize persisted x-axis key to an internal identifier.

    Recognises the reserved run-level axes (``field``/``temperature``/``run``),
    the first-class ``angle`` axis, the ``param:<name>`` namespace used for
    parameter-vs-parameter trending (item 1), and the ``custom:<id>`` namespace
    for data-browser custom columns. Anything else collapses to ``run``.
    """
    text = str(value or "").strip()
    if text in ("field", "temperature", "run", "angle"):
        return text
    if text.startswith("param:") and len(text) > len("param:"):
        return text
    if text.startswith("custom:") and len(text) > len("custom:"):
        return text
    return "run"


def _x_param_name(x_key: str) -> str | None:
    """Return the fitted-parameter name for a ``param:<name>`` x-key, else None."""
    if x_key.startswith("param:"):
        return x_key[len("param:") :]
    return None


@dataclass
class _FitRow:
    run_number: int
    run_label: str
    field: float
    temperature: float
    values: dict[str, float]
    errors: dict[str, float]
    combined_from: list[int] | None = None
    covariance: dict[str, dict[str, float]] | None = None
    #: Per-run data-browser custom-column values, keyed by column id
    #: (``custom:<hex>``). Stored as raw text — they are free-form and may be
    #: empty or non-numeric for some runs; coercion to a float abscissa (and the
    #: dropping of invalid points) happens lazily in :meth:`_x_value`.
    custom_values: dict[str, str] = field(default_factory=dict)
    #: Originating time-domain fit provenance, surfaced so parameter-plot exports
    #: record which model produced the values and each run's goodness of fit.
    #: ``model_name`` is the model formula/label; the χ² fields are ``None`` when
    #: the source path did not supply them (e.g. legacy projects).
    model_name: str = ""
    chi_squared: float | None = None
    reduced_chi_squared: float | None = None
    #: ``batch_id``/``trend_member_key`` identify the
    #: FitSlot this row's include-toggle writes; ``include_in_trend`` is the
    #: current gate; ``quality_flags`` are the advisory member-quality flags
    #: (``failed``/``large_rel_err``/``bound_pinned``/``spurious_reseeded``).
    batch_id: str | None = None
    trend_member_key: int | None = None
    include_in_trend: bool = True
    quality_flags: list[str] = field(default_factory=list)
    computation_revision: str | None = None


def _x_custom_id(x_key: str) -> str | None:
    """Return the custom-column id for a ``custom:<id>`` x-key, else None."""
    return x_key if isinstance(x_key, str) and x_key.startswith("custom:") else None


def _coerce_abscissa(raw: object) -> float:
    """Coerce a free-text custom/Angle cell value to a numeric abscissa.

    Empty, non-numeric, and non-finite (``inf``/``nan``) values all map to NaN so
    the point is dropped (and counted in the skip note) rather than plotted at 0
    or corrupting the axis/fit.
    """
    text = str(raw).strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


@dataclass(frozen=True)
class PhaseDecoration:
    """Identity of the phase (Global Fit Wizard transitions) a series belongs to.

    Built by :meth:`MainWindow._refresh_trend_panel` from the owning
    ``DataGroup`` (``asymmetry.core.representation.group``) for every series
    whose ``FitSeries.group_id`` names a phase group (``DataGroup.is_phase``);
    ``None`` on a plain (non-phase) series — a series belonging to no phase is
    a real, first-class case, not sniffed from an optional field.

    ``color``/``color_dark`` are the light/dark swatch colours
    (:func:`asymmetry.gui.utils.phase_colors.phase_color`); the app currently
    forces a single light palette everywhere (see
    ``gui/styles/palette.py::apply_bench_style`` — Fusion + the BENCH palette,
    specifically so an OS dark mode never shows through), so every consumer
    here resolves ``color`` and leaves ``color_dark`` unused until a real
    dark-theme toggle exists.
    """

    color: str
    color_dark: str
    ordinal: int
    name: str
    #: The order axis this phase is defined along (``"temperature"``,
    #: ``"field"``, or ``"run"`` — ``DataGroup.order_key``, inherited from the
    #: parent series group). The range/boundary shading only ever applies when
    #: the plot's *effective* x-axis matches this key.
    axis_key: str
    range: tuple[float, float]
    lower: tuple[float, float] | None
    upper: tuple[float, float] | None


@dataclass(frozen=True)
class ParameterExportData:
    cancelled: Callable[[], bool] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise TaskCancelledError

    _rows: Any
    _model_fits: Any
    _x_transform: Any
    _y_transforms: Any
    _model_fit_transform_sig: Any
    _global_params: Any
    _global_param_uncertainties: Any
    _angle_x_field: Any
    _angle_wrap_period: Any
    display_params: Any
    selected_params: Any
    x_key: str
    abscissa: Any
    model_label: str
    overlaid: bool
    x_header: str
    y_headers: Any

    def _display_y_parameters(self):
        return self.display_params

    def _selected_y_parameters(self):
        return self.selected_params

    def _effective_x_key(self):
        return self.x_key

    def _export_abscissa_column(self):
        return self.abscissa

    def _export_model_label(self):
        return self.model_label

    def _series_to_plot(self):
        return (0, 1) if self.overlaid else (0,)

    def _transformed_x_export_header(self, x_key):
        return self.x_header

    def _transformed_y_export_header(self, name):
        return self.y_headers[name]

    def _write_fit_files(
        self, gle_path: Path, x_key: str, y_params: list[str]
    ) -> dict[tuple[str, int], Path]:
        """Write model-fit sidecar files with metadata and sampled curve points."""
        entries = list(self._iter_active_fit_ranges(x_key, y_params))
        if not entries:
            return {}

        unique_params = sorted({pname for pname, _, _ in entries})
        include_param_name = len(unique_params) > 1

        def _safe_token(name: str) -> str:
            return gle_export.safe_file_token(name, fallback="fit")

        base_names: list[str] = []
        for pname, _, _fit_range in entries:
            self.check_cancelled()
            if include_param_name:
                base_names.append(f"{gle_path.stem}_{_safe_token(pname)}")
            else:
                base_names.append(gle_path.stem)

        totals: dict[str, int] = {}
        for base in base_names:
            self.check_cancelled()
            totals[base] = totals.get(base, 0) + 1

        seen: dict[str, int] = {}
        written: dict[tuple[str, int], Path] = {}
        for (pname, r_idx, fit_range), base in zip(entries, base_names, strict=True):
            self.check_cancelled()
            seen[base] = seen.get(base, 0) + 1
            occurrence = seen[base]
            stem = base if occurrence == 1 else f"{base}_{occurrence}"
            fit_path = gle_path.with_name(f"{stem}.fit")

            if totals[base] == 1:
                fit_path = gle_path.with_name(f"{base}.fit")

            result = fit_range.result
            assert result is not None

            sampled = self._sample_fit_range_curve(
                fit_range,
                x_key=x_key,
                num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
            )
            if sampled is None:
                continue
            x_vals, y_vals = sampled

            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Parameter model fit curve\n")
                f.write("! Generated by Asymmetry (GLE-readable data file)\n")
                f.write(f"! parameter: {pname}\n")
                f.write(f"! x_variable: {x_key}\n")
                if fit_range.windows:
                    windows_text = " ".join(f"[{lo:g}, {hi:g}]" for lo, hi in fit_range.windows)
                    f.write(f"! fit_windows: {windows_text}\n")
                    f.write(f"! x_min: {x_vals[0]:.10g}\n")
                    f.write(f"! x_max: {x_vals[-1]:.10g}\n")
                else:
                    f.write(f"! x_min: {fit_range.x_min}\n")
                    f.write(f"! x_max: {fit_range.x_max}\n")
                f.write(f"! model_function: {fit_range.model.formula_string()}\n")
                f.write(f"! chi_squared: {result.chi_squared:.8g}\n")
                f.write(f"! reduced_chi_squared: {result.reduced_chi_squared:.8g}\n")
                f.write("! fitted_parameters:\n")
                for p in result.parameters:
                    self.check_cancelled()
                    err = result.uncertainties.get(p.name, np.nan)
                    if np.isfinite(err):
                        f.write(f"!   {p.name} = {p.value:.8g} +/- {err:.4g}\n")
                    else:
                        f.write(f"!   {p.name} = {p.value:.8g}\n")
                for xv, yv in zip(x_vals, y_vals, strict=True):
                    self.check_cancelled()
                    f.write(f"{xv:.10g} {yv:.10g}\n")

            written[(pname, r_idx)] = fit_path

        return written

    def _write_gle_data_file(self, data_path: Path) -> None:
        x_key = self._effective_x_key()
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Fit parameter data for GLE export\n")

            model_label = self._export_model_label()
            if model_label:
                f.write(f"! Model: {model_label}\n")

            x_active = not self._x_transform.is_identity
            transformed_y = self._transformed_y_params()
            if x_active:
                f.write(f"! X transform: {self._x_transform.expression_text}\n")
            for name in transformed_y:
                self.check_cancelled()
                f.write(f"! Y transform [{name}]: {self._y_transform_for(name).expression_text}\n")
            if len(self._series_to_plot()) > 1:
                f.write("! Note: overlaid series not exported; active series only.\n")

            if self._global_params is not None:
                f.write("! Global fitting parameters:\n")
                for param in self._global_params:
                    self.check_cancelled()
                    unit = get_param_info(param.name).unit
                    label = f"{param.name} ({unit})" if unit else param.name
                    err = self._global_param_uncertainties.get(param.name)
                    if err is not None:
                        f.write(f"!   {label} = {param.value:.6g} +/- {err:.6g}\n")
                    else:
                        f.write(f"!   {label} = {param.value:.6g}\n")

            f.write("!\n")

            headers = ["Run", "B_field(G)", "Temperature(K)"]
            for name in self._display_y_parameters():
                self.check_cancelled()
                unit = get_param_info(name).unit
                if unit:
                    headers.extend([f"{name} ({unit})", f"err_{name} ({unit})"])
                else:
                    headers.extend([name, f"err_{name}"])
            # A free-text x-axis (Angle/custom) is appended as a trailing column so
            # GLE can plot against it; param columns keep their fixed indices.
            abscissa = self._export_abscissa_column()
            abscissa_key = abscissa[0] if abscissa is not None else None
            if abscissa is not None:
                headers.append(abscissa[1])
            # Per-run goodness of fit as trailing columns. Appended *after* the
            # abscissa so the fixed parameter/abscissa column indices that
            # _gle_columns_for_param / _gle_x_column compute are unaffected.
            headers.extend(["reduced_chi2", "chi2"])
            # Transformed columns are appended *after* χ² for the same reason: the
            # raw-column indices stay put; the transformed indices are computed by
            # _gle_transformed_x_column / _gle_transformed_columns_for_param.
            if x_active:
                headers.append(self._transformed_x_export_header(x_key))
            for name in transformed_y:
                self.check_cancelled()
                header = self._transformed_y_export_header(name)
                headers.extend([header, f"err_{header}"])

            x_transformed = (
                self._apply_x_transform(
                    np.array([self._x_value(r, x_key) for r in rows], dtype=float), None
                )[0]
                if x_active
                else None
            )
            y_transformed = {name: self._series_y_arrays(rows, name) for name in transformed_y}

            f.write("! Column map:\n")
            for col_idx, name in enumerate(headers, start=1):
                self.check_cancelled()
                f.write(f"!   c{col_idx:>2} = {name}\n")
            combined_rows = [row for row in rows if row.combined_from]
            if combined_rows:
                f.write("!\n")
                f.write("! Combined run mapping:\n")
                for row in combined_rows:
                    self.check_cancelled()
                    combined_label = " + ".join(str(v) for v in row.combined_from or [])
                    f.write(f"!   {row.run_number} = {combined_label}\n")
            f.write("!\n")
            f.write("! " + " ".join(f"{h:>16}" for h in headers) + "\n")

            for i, row in enumerate(rows):
                self.check_cancelled()
                values: list[float] = [
                    float(row.run_number),
                    float(row.field),
                    float(row.temperature),
                ]
                for name in self._display_y_parameters():
                    self.check_cancelled()
                    values.append(row.values.get(name, np.nan))
                    values.append(row.errors.get(name, np.nan))
                if abscissa_key is not None:
                    values.append(self._x_value(row, abscissa_key))
                values.append(
                    row.reduced_chi_squared if row.reduced_chi_squared is not None else np.nan
                )
                values.append(row.chi_squared if row.chi_squared is not None else np.nan)
                if x_active:
                    values.append(float(x_transformed[i]))
                for name in transformed_y:
                    self.check_cancelled()
                    vals, errs = y_transformed[name]
                    values.extend([float(vals[i]), float(errs[i])])
                f.write(" ".join(f"{v:>16.8g}" for v in values) + "\n")

    def _gle_x_column(self, x_key: str) -> int:
        name = _x_param_name(x_key)
        if name is not None:
            # Parameter-vs-parameter: x is the fitted parameter's value column,
            # which the data file already emits (with its error column alongside).
            cols = self._gle_columns_for_param(name)
            return cols[0] if cols is not None else 1
        if x_key == "run":
            return 1
        if x_key == "field":
            return 2
        if x_key == "temperature":
            return 3
        # Angle / custom column: the trailing column appended by _write_gle_data_file
        # (after Run/B/T and the 2-per-parameter columns).
        return 4 + 2 * len(self._display_y_parameters())

    def _gle_columns_for_param(self, name: str) -> tuple[int, int] | None:
        display_params = self._display_y_parameters()
        if name not in display_params:
            return None
        idx = display_params.index(name)
        value_col = 4 + idx * 2
        error_col = value_col + 1
        return value_col, error_col

    def _gle_base_column_count(self) -> int:
        """1-indexed count of raw columns (Run/B/T, params, abscissa, χ²ᵣ, χ²)."""
        count = 3 + 2 * len(self._display_y_parameters())
        if self._export_abscissa_column() is not None:
            count += 1
        return count + 2  # reduced_chi2, chi2

    def _gle_transformed_x_column(self) -> int | None:
        """1-indexed transformed-x column (``None`` when x is untransformed)."""
        if self._x_transform.is_identity:
            return None
        return self._gle_base_column_count() + 1

    def _gle_transformed_columns_for_param(self, name: str) -> tuple[int, int] | None:
        """1-indexed (value, err) transformed columns for a plotted param.

        ``None`` for a parameter plotted through the identity lens: the data file
        writes a transformed column pair only for the transformed parameters.
        """
        transformed = self._transformed_y_params()
        if name not in transformed:
            return None
        base = self._gle_base_column_count()
        if not self._x_transform.is_identity:
            base += 1  # the transformed-x column is written first
        value_col = base + 1 + transformed.index(name) * 2
        return value_col, value_col + 1

    def _transformed_y_params(self) -> list[str]:
        """Selected y parameters carrying a non-identity lens, in display order."""
        return [
            n for n in self._selected_y_parameters() if not self._y_transform_for(n).is_identity
        ]

    def _gle_effective_x_column(self, x_key: str) -> int:
        """The x column the GLE plot should read — transformed when active."""
        transformed = self._gle_transformed_x_column()
        return transformed if transformed is not None else self._gle_x_column(x_key)

    def _gle_effective_columns_for_param(self, name: str) -> tuple[int, int] | None:
        return self._gle_transformed_columns_for_param(name) or self._gle_columns_for_param(name)

    def _x_value(self, row: _FitRow, x_key: str, *, fold: bool = True) -> float:
        name = _x_param_name(x_key)
        if name is not None:
            return float(row.values.get(name, float("nan")))
        # The Angle field and the generic custom columns both store free text per
        # run under their key in ``custom_values``; coerce to a numeric abscissa
        # (empty/non-numeric/non-finite → NaN so the point is dropped and counted
        # rather than plotted at 0 or corrupting the axis/fit).
        value_key = x_key if x_key == self._angle_x_key() else _x_custom_id(x_key)
        if value_key is not None:
            value = _coerce_abscissa(row.custom_values.get(value_key, ""))
            # Fold the Angle axis into its chosen period so equivalent orientations
            # overlay (no-op for non-angle custom columns or when folding is off).
            # ``fold=False`` recovers the raw scan coordinate — used by crossing
            # detection, which must follow the true rotation order, not the folded
            # display (folding would collapse distinct orientations onto one x).
            if fold and x_key == self._angle_x_key() and self._angle_wrap_period is not None:
                return wrap_angle_deg(value, self._angle_wrap_period)
            return value
        if x_key == "field":
            return row.field
        if x_key == "temperature":
            return row.temperature
        return float(row.run_number)

    def _angle_x_key(self) -> str | None:
        """Return the key identifying the first-class Angle x-axis, or None."""
        return self._angle_x_field[1] if self._angle_x_field is not None else None

    def _apply_x_transform(
        self, x_vals: np.ndarray, x_err: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform an abscissa array (and its optional per-point error)."""
        if self._x_transform.is_identity:
            return x_vals, x_err
        vals, errs = self._x_transform.apply(x_vals, x_err if x_err is not None else None)
        # Run-level axes carry no uncertainty; keep it absent after transform.
        return vals, (errs if x_err is not None else None)

    def _series_y_arrays(self, rows: list[_FitRow], y_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Build (and transform) one parameter's value/error arrays for *rows*."""
        y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
        y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
        transform = self._y_transform_for(y_name)
        if transform.is_identity:
            return y_vals, y_err
        return transform.apply(y_vals, y_err)

    def _y_transform_for(self, name: str) -> AxisTransform:
        """The lens one y parameter is plotted through (identity by default)."""
        return self._y_transforms.get(name, _IDENTITY_TRANSFORM)

    def _iter_active_fit_ranges(self, x_key: str, y_params: list[str] | None = None):
        """Yield (parameter_name, range_index, fit_range) for successful active fits."""
        allowed = set(y_params or self._display_y_parameters())
        for pname, fit in self._model_fits.items():
            if pname not in allowed:
                continue
            if not fit.active or fit.x_key != x_key:
                continue
            # A fit computed under a different transform lives in the old
            # coordinate; the screen suppresses its overlay, so the export must
            # not write a `.fit` sidecar the plot deliberately hides.
            if self._overlay_suppressed_for_transform(pname):
                continue
            for idx, fit_range in enumerate(fit.ranges):
                if fit_range.result is None or not fit_range.result.success:
                    continue
                yield pname, idx, fit_range

    def _sampled_fit_curves(
        self,
        param_name: str,
        x_key: str,
        num_points: int = _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        fit = self._model_fits.get(param_name)
        if fit is None or not fit.active or fit.x_key != x_key:
            return []

        curves: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, fit_range in enumerate(fit.ranges):
            sampled = self._sample_fit_range_curve(fit_range, x_key=x_key, num_points=num_points)
            if sampled is None:
                continue
            xs, ys = sampled
            curves.append((idx, xs, ys))
        return curves

    def _sample_fit_range_curve(
        self,
        fit_range: ModelFitRange,
        x_key: str,
        num_points: int = _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
        *,
        x_domain: object = _UNSET,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample one fit range's overlay curve.

        ``x_domain`` lets the off-thread trend recompute pass a GUI-thread
        snapshot of the data x-range. When it is omitted (``_UNSET``) the
        GUI-thread callers read it live from the rows; when passed explicitly
        (even as ``None``) the snapshot is used as-is, so the worker thread never
        touches ``self._rows``.
        """
        result = fit_range.result
        if result is None or not result.success:
            return None

        try:
            # Window-union envelope when windows exist, else the plain bounds
            # — keeps the overlay span consistent with the fitted mask.
            x_min, x_max = effective_range_bounds(fit_range)
        except ValueError:
            return None
        if x_min is None or x_max is None:
            domain = self._x_domain_for_sampling(x_key) if x_domain is _UNSET else x_domain
            if domain is None:
                return None
            if x_min is None:
                x_min = domain[0]
            if x_max is None:
                x_max = domain[1]

        if x_max <= x_min:
            return None

        sample_count = max(2, int(num_points))
        # Geometric sampling suits a bare (untransformed) field axis; once the
        # abscissa is transformed the domain is no longer a raw field, so sample
        # it linearly.
        if (
            x_key == "field"
            and self._x_transform.is_identity
            and float(x_min) > 0.0
            and float(x_max) > 0.0
        ):
            xs = np.geomspace(float(x_min), float(x_max), num=sample_count, dtype=float)
        else:
            xs = np.linspace(float(x_min), float(x_max), num=sample_count, dtype=float)
        kwargs = {p.name: p.value for p in result.parameters}
        try:
            ys = fit_range.model.function(xs, **kwargs)
        except KeyError:
            return None
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None

        xs = xs[mask]
        ys = ys[mask]
        order = np.argsort(xs)
        return xs[order], ys[order]

    def _x_domain_for_sampling(self, x_key: str) -> tuple[float, float] | None:
        if not self._rows:
            return None
        x_vals = np.array([self._x_value(r, x_key) for r in self._rows], dtype=float)
        # Sample in the transformed coordinate the fit lives in (min/max of the
        # transformed values, robust to non-monotonic transforms).
        x_vals, _ = self._apply_x_transform(x_vals, None)
        finite = x_vals[np.isfinite(x_vals)]
        if finite.size == 0:
            return None
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _overlay_suppressed_for_transform(self, param_name: str) -> bool:
        """Suppress a stored fit's overlay when this parameter's axes are
        transformed but the fit was computed under a *different* transform (its
        ranges live in the old coordinate). No-op when both the abscissa and this
        parameter's lens are identity, so the common path is unchanged — and
        another parameter's lens never touches it; the fit's pill/table stay
        usable, only the curve waits for a re-fit under the current transform."""
        if self._x_transform.is_identity and self._y_transform_for(param_name).is_identity:
            return False
        return self._model_fit_transform_sig.get(param_name) != self._transform_signature(
            param_name
        )

    def _transform_signature(self, name: str) -> tuple:
        """Identity of the transforms one parameter is plotted under, for
        cache/overlay invalidation."""
        y_transform = self._y_transform_for(name)
        return (
            self._x_transform.kind,
            self._x_transform.expression,
            y_transform.kind,
            y_transform.expression,
        )
