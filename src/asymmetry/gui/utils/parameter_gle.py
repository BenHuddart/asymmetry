"""Worker-safe parameter-trend GLE figures."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.gui.utils import gle_export
from asymmetry.gui.utils.gle_figure import gle_figure, gle_subplots
from asymmetry.gui.utils.parameter_export_data import (
    _PARAMETER_FIT_CURVE_SAMPLE_COUNT,
    ParameterExportData,
    PhaseDecoration,
    _fit_overlay_color,
    _fit_overlay_label,
    _gle_series_color,
    _normalize_x_key,
    _safe_data_name,
)


@dataclass(frozen=True)
class ParameterGleSnapshot(ParameterExportData):
    phase: Any
    x_label: str
    y_labels: Any
    legend_labels: Any
    log_x: bool
    log_y: frozenset[str]
    show_components: bool
    plot_mode: str
    _plot_annotations: Any

    @cached_property
    def sampled_ranges(self):
        curves = {}
        for name, fit in self._model_fits.items():
            if (
                not fit.active
                or fit.x_key != self.x_key
                or self._overlay_suppressed_for_transform(name)
            ):
                continue
            for fit_range in fit.ranges:
                self.check_cancelled()
                curves[id(fit_range)] = ParameterExportData._sample_fit_range_curve(
                    self, fit_range, fit.x_key
                )
        return curves

    def _sample_fit_range_curve(
        self, fit_range, x_key, num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT
    ):
        return self.sampled_ranges[id(fit_range)]

    def _gle_effective_y_label(self, name):
        return self.y_labels[name]

    def _is_log_y_for(self, name):
        return name in self.log_y

    def build(self, glp, gle_path, export_dir, *, cancelled=None):
        snapshot = replace(self, cancelled=cancelled)
        snapshot.check_cancelled()
        return snapshot.build_figure(glp, gle_path, export_dir)

    def build_figure(self, glp, gle_path, export_dir):
        if not hasattr(glp, "Axes") or not hasattr(glp.Axes, "errorbar_from_file"):
            raise RuntimeError("Update gleplot to a version supporting file-backed error bars.")
        data_path = gle_path.with_suffix(".dat")
        self._write_gle_data_file(data_path)
        x_key = self.x_key
        # GLE export always writes the active series only (see _export_gle) —
        # this is that series' phase, gated for the current axis/transform.
        phase = self.phase

        # Export fit sidecars for all active fits on this x-axis, regardless of
        # current y-selection, so restored projects always emit .fit files.
        all_active_fit_params = [
            pname
            for pname, fit in self._model_fits.items()
            if fit.active and _normalize_x_key(fit.x_key) == x_key
        ]
        fit_file_map = self._write_fit_files(gle_path, x_key, all_active_fit_params)

        display_params = self.display_params
        y_params = self.selected_params or ([display_params[0]] if display_params else [])
        if not y_params:
            return None

        x_label = self.x_label
        data_file_ref = data_path.name
        x_col = self._gle_effective_x_column(x_key)
        plot_mode = self.plot_mode
        rows = sorted(self._rows, key=lambda r: self._x_value(r, x_key))
        show_fit_legend = self._count_fit_curves(x_key, y_params) > 1

        if plot_mode == "Subplots" and len(y_params) > 1:
            fig, axes = gle_subplots(
                glp, nrows=len(y_params), ncols=1, figsize=(5.8, 3.0 * len(y_params)), sharex=True
            )
            subplot_axes = axes if isinstance(axes, list) else [axes]
            for idx, y_name in enumerate(y_params):
                cols = self._gle_effective_columns_for_param(y_name)
                if cols is None:
                    continue
                y_col, yerr_col = cols
                y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                ax = subplot_axes[idx]
                show_subplot_fit_legend = self._count_fit_curves_for_param(x_key, y_name) > 1
                self._add_gle_model_overlay(
                    ax,
                    y_name,
                    color=_fit_overlay_color(0),
                    yaxis="y",
                    include_labels=show_subplot_fit_legend,
                    fit_file_map=fit_file_map,
                )
                # One subplot per parameter — series identity, so a phase-owned
                # active series takes its phase colour (mirrors the on-screen
                # single-series Subplots substitution).
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=y_col,
                    yerr_col=yerr_col if has_err else None,
                    color=_gle_series_color(0, override=phase.color if phase else None),
                    marker="o",
                    markersize=5,
                    capsize=2,
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel(self.y_labels[y_name])
                if self.log_x:
                    ax.set_xscale("log")
                if not self.show_components and (y_name in self.log_y):
                    ax.set_yscale("log")
                if self.show_components:
                    ax.set_ylim(0.0, None)
                self._add_gle_annotations(ax, y_name)
                if phase is not None:
                    self._add_gle_phase_band(ax, phase, y_vals)
                if show_subplot_fit_legend:
                    ax.legend(loc="best")
        else:
            fig = gle_figure(glp, figsize=(5.8, 4.2))
            ax = fig.add_subplot(111)

            if len(y_params) == 2:
                left_name, right_name = y_params
                left_cols = self._gle_effective_columns_for_param(left_name)
                right_cols = self._gle_effective_columns_for_param(right_name)
                if left_cols is None or right_cols is None:
                    return None
                left_y_col, left_err_col = left_cols
                right_y_col, right_err_col = right_cols

                left_vals = np.array([r.values.get(left_name, np.nan) for r in rows], dtype=float)
                right_vals = np.array([r.values.get(right_name, np.nan) for r in rows], dtype=float)
                left_err = np.array([r.errors.get(left_name, np.nan) for r in rows], dtype=float)
                right_err = np.array([r.errors.get(right_name, np.nan) for r in rows], dtype=float)
                has_left_err = bool(np.any(np.isfinite(left_err) & (left_err > 0)))
                has_right_err = bool(np.any(np.isfinite(right_err) & (right_err > 0)))

                self._add_gle_model_overlay(
                    ax,
                    left_name,
                    color=_fit_overlay_color(0),
                    yaxis="y",
                    include_labels=show_fit_legend,
                    fit_file_map=fit_file_map,
                )
                self._add_gle_model_overlay(
                    ax,
                    right_name,
                    color=_fit_overlay_color(1),
                    yaxis="y2",
                    include_labels=show_fit_legend,
                    fit_file_map=fit_file_map,
                )
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=left_y_col,
                    yerr_col=left_err_col if has_left_err else None,
                    color=_gle_series_color(0),
                    marker="o",
                    markersize=5,
                    capsize=2,
                    yaxis="y",
                )
                ax.errorbar_from_file(
                    data_file_ref,
                    x_col=x_col,
                    y_col=right_y_col,
                    yerr_col=right_err_col if has_right_err else None,
                    color=_gle_series_color(1),
                    marker="o",
                    markersize=5,
                    capsize=2,
                    yaxis="y2",
                )
                ax.set_ylabel(self.y_labels[left_name], axis="y")
                ax.set_ylabel(self.y_labels[right_name], axis="y2")
                if self.show_components:
                    ax.set_ylim(0.0, None, axis="y")
                    ax.set_ylim(0.0, None, axis="y2")
                self._add_gle_annotations(ax, left_name)
                self._add_gle_annotations(ax, right_name)
                if phase is not None:
                    # Shared x-axis: one band spanning both parameters' extents.
                    self._add_gle_phase_band(ax, phase, left_vals, right_vals)
                if show_fit_legend:
                    ax.legend(loc="best")
            else:
                # A single parameter on this shared axis is the whole series'
                # curve (series identity, phase colour applies); several share
                # it to distinguish *parameters* (their own identity colours).
                single_param = len(y_params) == 1
                all_y_vals: list[np.ndarray] = []
                for idx, y_name in enumerate(y_params):
                    cols = self._gle_effective_columns_for_param(y_name)
                    if cols is None:
                        continue
                    y_col, yerr_col = cols
                    y_vals = np.array([r.values.get(y_name, np.nan) for r in rows], dtype=float)
                    all_y_vals.append(y_vals)
                    y_err = np.array([r.errors.get(y_name, np.nan) for r in rows], dtype=float)
                    has_err = bool(np.any(np.isfinite(y_err) & (y_err > 0)))
                    self._add_gle_model_overlay(
                        ax,
                        y_name,
                        color=_fit_overlay_color(idx),
                        yaxis="y",
                        include_labels=show_fit_legend,
                        fit_file_map=fit_file_map,
                    )
                    override = phase.color if (single_param and phase is not None) else None
                    ax.errorbar_from_file(
                        data_file_ref,
                        x_col=x_col,
                        y_col=y_col,
                        yerr_col=yerr_col if has_err else None,
                        color=_gle_series_color(idx, override=override),
                        marker="o",
                        markersize=5,
                        capsize=2,
                    )

                if len(y_params) == 1:
                    ax.set_ylabel(self._gle_effective_y_label(y_params[0]))
                else:
                    ax.set_ylabel("Parameter value")
                    if show_fit_legend:
                        ax.legend(loc="best")

                if self.show_components:
                    ax.set_ylim(0.0, None)
                elif len(y_params) == 1 and self._is_log_y_for(y_params[0]):
                    ax.set_yscale("log")
                elif len(y_params) > 1 and any(self._is_log_y_for(name) for name in y_params):
                    ax.set_yscale("log")

                self._add_gle_annotations(ax, "main")
                if phase is not None and all_y_vals:
                    self._add_gle_phase_band(ax, phase, *all_y_vals)

            ax.set_xlabel(x_label)
            if self.log_x:
                ax.set_xscale("log")

        try:
            fig.savefig(str(gle_path))
        except TypeError as exc:
            if "folder" in str(exc):
                gle_export.show_warning(
                    self, "gleplot update required", "Please update gleplot to a newer version."
                )
                return None
            raise

        fit_files = list(fit_file_map.values()) if fit_file_map else []
        return gle_export.GleExportBuild(files=[gle_path, data_path, *fit_files])

    def _add_gle_model_overlay(
        self,
        ax,
        param_name: str,
        color: str,
        yaxis: str = "y",
        include_labels: bool = False,
        fit_file_map: dict[tuple[str, int], Path] | None = None,
    ) -> None:
        fit = self._model_fits.get(param_name)
        if fit is None or not fit.active:
            return
        if fit.x_key != self.x_key:
            return
        if self._overlay_suppressed_for_transform(param_name):
            return

        show_components = self.show_components
        component_colors = ["lightblue", "lightgreen", "pink", "lightgray", "cyan", "yellow"]

        curves = self._sampled_fit_curves(
            param_name,
            x_key=fit.x_key,
            num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
        )
        for idx, (range_index, xs, ys) in enumerate(curves):
            line_color = _fit_overlay_color(idx) if len(curves) > 1 else color
            line_label = (
                _fit_overlay_label(self.legend_labels[param_name], idx, len(curves))
                if include_labels
                else None
            )
            base_name = f"{_safe_data_name(param_name)}_range_{int(range_index)}"

            if show_components and range_index < len(fit.ranges):
                fit_range = fit.ranges[range_index]
                result = fit_range.result
                if result is not None and result.success:
                    kwargs = {p.name: p.value for p in result.parameters}
                    components = fit_range.model.evaluate_components(
                        xs, additive_only=True, **kwargs
                    )
                    ordered_components = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xs, dtype=float)
                    for cidx, (_cname, comp_y) in enumerate(ordered_components):
                        cname = _safe_data_name(_cname)
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(
                            xs,
                            lower,
                            upper,
                            color=fill_color,
                            alpha=0.3,
                            data_name=f"component_{base_name}_{cname}_fill",
                        )
                        ax.plot(
                            xs,
                            upper,
                            linestyle="--",
                            color=fill_color,
                            linewidth=1,
                            yaxis=yaxis,
                            data_name=f"component_{base_name}_{cname}_edge",
                        )
                        cumulative = upper

            fit_path = None
            if fit_file_map is not None:
                fit_path = fit_file_map.get((param_name, int(range_index)))
            if fit_path is not None and hasattr(ax, "line_from_file"):
                ax.line_from_file(
                    fit_path.name,
                    x_col=1,
                    y_col=2,
                    color=line_color,
                    linestyle="-",
                    linewidth=1,
                    yaxis=yaxis,
                    label=line_label,
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    linestyle="-",
                    color=line_color,
                    linewidth=1,
                    yaxis=yaxis,
                    label=line_label,
                    data_name=f"model_{base_name}",
                )

    def _add_gle_annotations(self, ax, axis_tag: str) -> None:
        """Add user plot annotations to a specific exported axis."""
        for ann in self._plot_annotations:
            if str(ann.get("axis_tag", "main")) != axis_tag:
                continue
            text = str(ann.get("text", "")).strip()
            if not text:
                continue
            try:
                x = float(ann.get("x", 0.0))
                y = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            ax.text(x, y, text, color="black", ha="left")

    @staticmethod
    def _gle_phase_y_extent(*value_arrays: np.ndarray) -> tuple[float, float] | None:
        """A (padded) vertical extent spanning every finite value in *value_arrays*.

        gleplot has no ``axvspan``/``axvline`` (unlike matplotlib, whose blended
        transform spans the full axes height regardless of the y-limits) — the
        band and boundary lines are drawn as an ordinary ``fill_between``/``plot``
        rectangle instead, so it needs an explicit y-range. A 5% margin on the
        combined data extent mirrors matplotlib's own default autoscale margin.
        Returns ``None`` when no finite value is present (nothing to anchor to).
        """
        finite = np.concatenate(
            [np.asarray(values, dtype=float)[np.isfinite(values)] for values in value_arrays]
        )
        if finite.size == 0:
            return None
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if lo == hi:
            pad = abs(lo) * 0.05 or 0.5
        else:
            pad = (hi - lo) * 0.05
        return lo - pad, hi + pad

    def _add_gle_phase_band(self, ax, phase: PhaseDecoration, *value_arrays: np.ndarray) -> None:
        """Draw *phase*'s range shading and boundary lines on an exported GLE axis.

        Mirrors :meth:`_draw_phase_band`'s matplotlib picture with gleplot's
        ``fill_between``/``plot`` primitives (see :meth:`_add_gle_model_overlay`
        for the same primitives used for the component-stack fill): the range is
        a filled rectangle spanning ``_gle_phase_y_extent(*value_arrays)``, each
        boundary a dashed vertical line plus a fainter ``± half_gap`` rectangle.
        Always drawn on the primary (``"y"``) axis — the one caller that shares
        an axis with a second y-parameter (the two-parameter export) draws the
        band once there, since both parameters share the same x-range.
        """
        extent = self._gle_phase_y_extent(*value_arrays)
        if extent is None:
            return
        y_lo, y_hi = extent
        lo, hi = phase.range
        ax.fill_between([lo, hi], [y_lo, y_lo], [y_hi, y_hi], color=phase.color, alpha=0.12)
        for boundary in (phase.lower, phase.upper):
            if boundary is None:
                continue
            estimate, half_gap = boundary
            ax.plot(
                [estimate, estimate],
                [y_lo, y_hi],
                linestyle="--",
                color=phase.color,
                linewidth=1.2,
            )
            ax.fill_between(
                [estimate - half_gap, estimate + half_gap],
                [y_lo, y_lo],
                [y_hi, y_hi],
                color=phase.color,
                alpha=0.08,
            )

    def _ordered_components_for_stacking(
        self,
        components: list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        """Return a stable bottom-to-top stacking order for additive components.

        Heuristic:
        1. Put background-like components (bg/constant) at the bottom.
        2. For remaining components, place smoother/lower-variance traces lower.
        3. Break ties by mean magnitude (smaller first).
        """
        if not components:
            return []

        def _priority(name: str) -> int:
            lname = name.lower()
            if "bg" in lname or "background" in lname or "constant" in lname:
                return 0
            return 1

        scored: list[tuple[int, float, float, int, tuple[str, np.ndarray]]] = []
        for idx, item in enumerate(components):
            name, values = item
            arr = np.maximum(np.asarray(values, dtype=float), 0.0)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                mean_val = 0.0
                variability = 0.0
            else:
                mean_val = float(np.mean(finite))
                variability = float(np.std(finite) / max(mean_val, 1e-12))
            scored.append((_priority(name), variability, mean_val, idx, item))

        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [item for *_meta, item in scored]

    def _count_fit_curves(self, x_key: str, y_params: list[str]) -> int:
        count = 0
        for pname in y_params:
            count += len(
                self._sampled_fit_curves(
                    pname,
                    x_key,
                    num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
                )
            )
        return count

    def _count_fit_curves_for_param(self, x_key: str, param_name: str) -> int:
        return len(
            self._sampled_fit_curves(
                param_name,
                x_key,
                num_points=_PARAMETER_FIT_CURVE_SAMPLE_COUNT,
            )
        )
