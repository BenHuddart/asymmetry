"""Captured global-fit data and shared export table writers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    ParameterGroupData,
    evaluate_parameter_model_fit,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.gui.tasks import TaskCancelledError
from asymmetry.gui.utils.gle_export import (
    safe_file_token,
)
from asymmetry.gui.utils.parameter_export_data import _PARAMETER_FIT_CURVE_SAMPLE_COUNT


@dataclass(frozen=True)
class GlobalExportData:
    cancelled: Callable[[], bool] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def check_cancelled(self):
        if self.cancelled is not None and self.cancelled():
            raise TaskCancelledError

    _groups: Any
    _result: Any
    _model: Any
    _curve_cache: Any
    _parameter_name: Any
    _x_key: str
    _x_label_override: str
    _group_variable_label_override: str
    _local_model_fits: Any
    _fit_x_min: float
    _fit_x_max: float

    def _write_local_parameter_fit_files(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> dict[str, Path]:
        fit_files: dict[str, Path] = {}
        for pname, _x_arr, _y_arr, _e_arr in traces:
            self.check_cancelled()
            fit = self._local_model_fits.get(pname)
            if fit is None or not fit.active or fit.x_key != self._local_group_x_key():
                continue

            curves = evaluate_parameter_model_fit(fit, num_points=200)
            if not curves:
                continue

            fit_path = gle_path.with_name(
                f"{gle_path.stem}_local_{self._safe_data_name(pname)}.fit"
            )
            info = get_param_info(pname)
            with open(fit_path, "w", encoding="utf-8") as f:
                f.write("! Local parameter model fit curve\n")
                f.write(f"! parameter: {info.plain_label()}\n")
                f.write(f"! x-axis: {self._local_group_axis_label_plain()}\n")
                f.write("! columns: x y\n")
                for cidx, curve in enumerate(curves):
                    self.check_cancelled()
                    if cidx > 0:
                        f.write("nan nan\n")
                    for xv, yv in zip(curve.x, curve.y, strict=True):
                        self.check_cancelled()
                        f.write(f"{float(xv):.10g} {float(yv):.10g}\n")

            fit_files[pname] = fit_path

        return fit_files

    def _write_local_parameter_data_file(
        self,
        gle_path: Path,
        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[Path, dict[str, tuple[int, int]]]:
        data_path = gle_path.with_name(f"{gle_path.stem}_local_parameters.dat")

        x_keys: dict[str, float] = {}
        for _pname, x_arr, _y_arr, _e_arr in traces:
            self.check_cancelled()
            for xv in x_arr:
                self.check_cancelled()
                x_keys[f"{float(xv):.12g}"] = float(xv)
        x_values = np.array(sorted(x_keys.values()), dtype=float)

        col_map: dict[str, tuple[int, int]] = {}
        columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            self.check_cancelled()
            y_out = np.full_like(x_values, np.nan, dtype=float)
            e_out = np.full_like(x_values, np.nan, dtype=float)
            local = {
                f"{float(x):.12g}": (float(y), float(e))
                for x, y, e in zip(x_arr, y_arr, e_arr, strict=True)
            }
            for i, xv in enumerate(x_values):
                self.check_cancelled()
                key = f"{float(xv):.12g}"
                if key in local:
                    yv, ev = local[key]
                    y_out[i] = yv
                    e_out[i] = ev if np.isfinite(ev) and ev >= 0 else np.nan
            columns[pname] = (y_out, e_out)
            y_col = 2 + idx * 2
            e_col = y_col + 1
            col_map[pname] = (y_col, e_col)

        global_cols: list[tuple[str, float, float]] = []
        if self._result is not None:
            for p in self._result.global_parameters:
                self.check_cancelled()
                err = self._result.global_uncertainties.get(p.name, np.nan)
                err_val = (
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
                global_cols.append((p.name, float(p.value), err_val))

        with open(data_path, "w", encoding="utf-8") as f:
            f.write("! Local parameter trends\n")

            if self._result is not None:
                for line in self._global_parameter_table_lines(self._result):
                    self.check_cancelled()
                    f.write(line + "\n")
                f.write("!\n")

            f.write(f"! columns: {self._local_group_axis_label_plain()}")
            for pname, _x, _y, _e in traces:
                self.check_cancelled()
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {pname}{unit} err_{pname}{unit}")
            for gname, _gval, _gerr in global_cols:
                self.check_cancelled()
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f" {gname}{unit} err_{gname}{unit}")
            f.write("\n")

            col_idx = 1
            f.write("! Column map:\n")
            f.write(f"!   c{col_idx:>2} = {self._local_group_axis_label_plain()}\n")
            col_idx += 1
            for pname, _x, _y, _e in traces:
                self.check_cancelled()
                info = get_param_info(pname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {pname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{pname}{unit}\n")
                col_idx += 1
            for gname, _gval, _gerr in global_cols:
                self.check_cancelled()
                info = get_param_info(gname)
                unit = f" ({info.unit})" if info.unit else ""
                f.write(f"!   c{col_idx:>2} = {gname}{unit}\n")
                col_idx += 1
                f.write(f"!   c{col_idx:>2} = err_{gname}{unit}\n")
                col_idx += 1

            for row_idx, xv in enumerate(x_values):
                self.check_cancelled()
                parts = [f"{xv:.10g}"]
                for pname, _x, _y, _e in traces:
                    self.check_cancelled()
                    yv, ev = columns[pname]
                    parts.append(f"{yv[row_idx]:.10g}")
                    parts.append(f"{ev[row_idx]:.10g}")
                for _gname, gval, gerr in global_cols:
                    self.check_cancelled()
                    parts.append(f"{gval:.10g}")
                    parts.append(f"{gerr:.10g}")
                f.write(" ".join(parts) + "\n")

        return data_path, col_map

    def _write_fit_subplot_files(self, gle_path: Path) -> dict[str, dict[str, object]]:
        """Write explicit per-group data and fit files for GLE export."""
        written: dict[str, dict[str, object]] = {}
        header_lines = self._fit_metadata_header_lines()
        x_axis_label = self._local_group_axis_label_plain()
        for group in self._groups:
            self.check_cancelled()
            token = self._safe_file_token(group.group_name or group.group_id)
            data_path = gle_path.with_name(f"{gle_path.stem}_{token}_data.dat")
            fit_path = gle_path.with_name(f"{gle_path.stem}_{token}_fit.fit")
            group_label = str(group.group_name or group.group_id)

            x = np.asarray(group.x, dtype=float)
            y = np.asarray(group.y, dtype=float)
            yerr = np.asarray(group.yerr, dtype=float)
            if yerr.shape != y.shape:
                yerr = np.full_like(y, np.nan, dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask] if yerr.size else np.full_like(y, np.nan, dtype=float)
            if x.size == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            yerr = yerr[order]

            with open(data_path, "w", encoding="utf-8") as f:
                f.write("! Cross-group fit data\n")
                f.write(f"! group: {group_label}\n")
                for line in header_lines:
                    self.check_cancelled()
                    f.write(line + "\n")
                f.write(f"! x-axis: {x_axis_label}\n")
                f.write("! columns: x y yerr\n")
                for xv, yv, ev in zip(x, y, yerr, strict=True):
                    self.check_cancelled()
                    ev_out = ev if np.isfinite(ev) and ev > 0 else np.nan
                    f.write(f"{xv:.10g} {yv:.10g} {ev_out:.10g}\n")

            sampled = self._sample_group_fit_curve(group)
            fit_written = False
            if sampled is not None:
                xs, ys = sampled
                with open(fit_path, "w", encoding="utf-8") as f:
                    f.write("! Cross-group fit model curve\n")
                    f.write(f"! group: {group_label}\n")
                    for line in header_lines:
                        self.check_cancelled()
                        f.write(line + "\n")
                    f.write(f"! x-axis: {x_axis_label}\n")
                    f.write("! columns: x y\n")
                    for xv, yv in zip(xs, ys, strict=True):
                        self.check_cancelled()
                        f.write(f"{xv:.10g} {yv:.10g}\n")
                fit_written = True

            written[group.group_id] = {
                "data_path": data_path,
                "fit_path": fit_path,
                "has_fit": fit_written,
                "has_err": bool(np.any(np.isfinite(yerr) & (yerr > 0))),
            }

        return written

    def _fit_metadata_header_lines(self) -> list[str]:
        """Return shared provenance lines for GLE fit-subplot exports.

        Mirrors the comprehensive local-parameter export header so the fit-data
        and model-curve files record *which* model was fitted, the values (and
        uncertainties) of the shared global parameters, and the fit quality —
        not just bare ``x y yerr`` columns.
        """
        lines: list[str] = []
        if self._model is not None:
            try:
                formula = self._model.formula_string()
            except Exception:
                formula = ""
            if formula:
                lines.append(f"! model: {formula}")

        result = self._result
        if result is not None:
            lines.extend(self._global_parameter_table_lines(result))
            if np.isfinite(result.chi_squared):
                lines.append(f"! chi_squared: {float(result.chi_squared):.8g}")
            if np.isfinite(result.reduced_chi_squared):
                lines.append(f"! reduced_chi_squared: {float(result.reduced_chi_squared):.8g}")
            lines.append(f"! error_mode: {result.error_mode}")
            if result.n_points:
                lines.append(f"! n_points: {int(result.n_points)}")
            if result.per_group_chi_squared:
                lines.append("! Per-group chi-squared:")
                for gid in sorted(result.per_group_chi_squared):
                    chi = result.per_group_chi_squared[gid]
                    npts = result.per_group_n_points.get(gid)
                    npts_txt = f" (n={int(npts)})" if npts is not None else ""
                    lines.append(f"!   {gid}: {float(chi):.8g}{npts_txt}")
            if result.global_correlations is not None:
                names, matrix = result.global_correlations
                lines.append("! Global parameter correlations:")
                lines.append("!   " + " ".join(str(n) for n in names))
                for i, name in enumerate(names):
                    row = matrix[i] if i < len(matrix) else []
                    row_txt = " ".join(f"{float(v):+.3f}" for v in row)
                    lines.append(f"!   {name}: {row_txt}")
        return lines

    @staticmethod
    def _global_parameter_table_lines(result) -> list[str]:
        """Return the ``! Global parameter table`` provenance lines for a result.

        Shared by the fit-subplot and local-parameter exports so the global /
        fixed parameter formatting (units, precision, ``nan`` sentinel, ``[fixed]``
        flag) stays identical across both. Lines have no trailing newline.
        """
        lines: list[str] = []
        if not (len(result.global_parameters) or len(result.fixed_parameters)):
            return lines
        lines.append("! Global parameter table:")
        lines.append("!   Parameter                     Value                Uncertainty")
        for p in result.global_parameters:
            info = get_param_info(p.name)
            unit = f" ({info.unit})" if info.unit else ""
            err = result.global_uncertainties.get(p.name)
            err_text = "nan"
            if err is not None and np.isfinite(err):
                err_text = f"{float(err):.10g}"
            lines.append(f"!   {p.name}{unit:<24} {float(p.value):>16.10g} {err_text:>24}")
        for p in result.fixed_parameters:
            info = get_param_info(p.name)
            unit = f" ({info.unit})" if info.unit else ""
            lines.append(f"!   {p.name}{unit} [fixed] = {float(p.value):.10g}")
        return lines

    def _sample_group_fit_curve(
        self, group: ParameterGroupData
    ) -> tuple[np.ndarray, np.ndarray] | None:
        # Reuse the warm curve cache when present (post-display it always is);
        # only fall back to an inline evaluation when the cache is cold.
        cached = self._cached_group_total_curve(group.group_id)
        if cached is not None:
            return cached
        if self._model is None:
            return None
        kwargs = self._model_kwargs_for_group(group.group_id)
        if not kwargs:
            return None

        xx = np.asarray(group.x, dtype=float)
        finite_x = xx[np.isfinite(xx)]
        if finite_x.size < 2:
            return None

        xx = np.sort(finite_x)
        if (
            np.isfinite(self._fit_x_min)
            and np.isfinite(self._fit_x_max)
            and self._fit_x_max > self._fit_x_min
        ):
            mask = (xx >= self._fit_x_min) & (xx <= self._fit_x_max)
            xx = xx[mask]
        if xx.size < 2:
            return None

        x_min = float(np.nanmin(xx))
        x_max = float(np.nanmax(xx))
        if self._local_group_x_key() == "field" and x_min > 0.0 and x_max > 0.0:
            xs = np.geomspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        else:
            xs = np.linspace(x_min, x_max, _PARAMETER_FIT_CURVE_SAMPLE_COUNT)
        try:
            ys = np.asarray(self._model.function(xs, **kwargs), dtype=float)
        except KeyError:
            return None
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            return None
        return xs[mask], ys[mask]

    def _cached_group_total_curve(self, group_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return a group's cached total ``(xx, yy)`` curve, or ``None`` if cold.

        The total fit curve is identical across both components flags, so either
        cache entry serves the GLE export. Returns ``None`` when neither flag has
        been computed yet (cold cache) or the entry carries an error.
        """
        for flag in (False, True):
            entry = self._curve_cache.get(flag)
            if not entry:
                continue
            curve = entry.get(group_id)
            if isinstance(curve, dict) and "error" not in curve and curve.get("xx") is not None:
                return curve["xx"], curve["yy"]
        return None

    def _model_kwargs_for_group(self, group_id: str) -> dict[str, float]:
        if self._result is None:
            return {}
        return self._model_kwargs(self._result, self._model, group_id)

    @staticmethod
    def _model_kwargs(result, model, group_id: str) -> dict[str, float]:
        """Build a group's model kwargs (global + fixed + local, defaults-filled).

        Pure — takes ``result``/``model`` explicitly so it is safe to call from
        the off-thread fit-curve worker as well as the GUI thread.
        """
        kwargs = {p.name: p.value for p in result.global_parameters}
        for p in result.fixed_parameters:
            kwargs[p.name] = p.value
        local = result.local_parameters.get(group_id)
        if local is not None:
            for p in local:
                kwargs[p.name] = p.value
        # Backward-compatible restore support: older saved cross-group states may
        # miss newer model parameters (e.g. D_perp). Fill from model defaults.
        missing = [name for name in getattr(model, "param_names", []) if name not in kwargs]
        defaults = getattr(model, "param_defaults", {})
        for name in missing:
            if isinstance(defaults, dict) and name in defaults:
                kwargs[name] = float(defaults[name])
        return kwargs

    def _safe_file_token(self, value: str) -> str:
        return safe_file_token(value, fallback="group")

    def _safe_data_name(self, value: str) -> str:
        # Unlike filename tokens, data names feed gleplot ``data_name=``
        # identifiers: keep the historical strictness (no hyphens, lowercase).
        return safe_file_token(str(value).replace("-", "_"), fallback="series").lower()

    def _local_group_x_key(self) -> str:
        if self._x_key == "field":
            return "temperature"
        if self._x_key == "temperature":
            return "field"
        return "run"

    def _local_group_axis_label_plain(self) -> str:
        known = {
            "field": "T (K)",
            "temperature": "B (G)",
            "run": "Group variable",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._group_variable_label_override:
            return self._group_variable_label_override
        return "Group variable"

    def _local_group_axis_label_gle(self) -> str:
        known = {
            "field": "{\\it T} (K)",
            "temperature": "{\\it B} (G)",
            "run": "Group variable",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._group_variable_label_override:
            return self._group_variable_label_override
        return "Group variable"

    def _x_label_gle(self) -> str:
        # Known keys keep the {\it B}/{\it T} classics; a custom study label is
        # rendered as plain text (GLE-safe) since it carries no math markup.
        known = {
            "field": "{\\it B} (G)",
            "temperature": "{\\it T} (K)",
            "run": "Run Number",
        }
        if self._x_key in known:
            return known[self._x_key]
        if self._x_label_override:
            return self._x_label_override
        return "x"

    def _parameter_label_gle(self, name: str | None) -> str:
        if not name:
            return "y"
        return get_param_info(name).gle_label()
