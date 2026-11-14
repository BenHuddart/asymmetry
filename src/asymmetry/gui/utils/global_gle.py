"""Worker-safe GLE figures for global parameter fits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.fitting.parameter_models import (
    evaluate_parameter_model_fit,
)
from asymmetry.gui.utils.gle_export import (
    GleExportBuild,
    extract_gle_data_dependencies,
)
from asymmetry.gui.utils.gle_figure import gle_figure, gle_subplots
from asymmetry.gui.utils.global_export_data import GlobalExportData


@dataclass(frozen=True)
class GlobalGleSnapshot(GlobalExportData):
    share_x: bool
    aspect: float
    show_components: bool
    fit_log_x: bool
    fit_log_y: bool
    local_log_x: bool
    local_plot_mode: str
    selected_local: tuple[str, ...]
    _local_param_log_y: frozenset[str]
    _plot_annotations: Any
    _local_plot_annotations: Any
    _suppressed_group_label_tags: frozenset[str]

    @cached_property
    def sampled_groups(self):
        curves = {}
        for group in self._groups:
            self.check_cancelled()
            curves[group.group_id] = GlobalExportData._sample_group_fit_curve(self, group)
        return curves

    def _sample_group_fit_curve(self, group):
        return self.sampled_groups[group.group_id]

    def apply_scales(self, ax):
        if self.fit_log_x:
            ax.set_xscale("log")
        if self.fit_log_y and not self.show_components:
            ax.set_yscale("log")

    def group_label(self, group, ax):
        if group.group_id in self._suppressed_group_label_tags:
            return
        if any(
            ann.get("is_group_label") and ann.get("axis_tag") == group.group_id
            for ann in self._plot_annotations
        ):
            return
        x, y = np.asarray(group.x), np.asarray(group.y)
        if x.size and y.size:
            ax.text(
                float(np.nanmin(x) + 0.02 * np.ptp(x)),
                float(np.nanmax(y) - 0.05 * np.ptp(y)),
                group.group_name,
                ha="left",
                va="top",
            )

    def build_fits(self, glp, gle_path, export_dir, *, cancelled=None):
        snapshot = replace(self, cancelled=cancelled)
        snapshot.check_cancelled()
        return snapshot.save(
            snapshot._build_fit_subplot_gle_figure(glp, gle_path), gle_path, export_dir
        )

    def build_local(self, glp, gle_path, export_dir, *, cancelled=None):
        snapshot = replace(self, cancelled=cancelled)
        snapshot.check_cancelled()
        return snapshot.save(
            snapshot._build_local_parameter_gle_figure(glp, gle_path), gle_path, export_dir
        )

    def save(self, figure, gle_path, export_dir):
        figure.savefig(str(gle_path))
        return GleExportBuild(
            files=[
                gle_path,
                *(export_dir / name for name in extract_gle_data_dependencies(gle_path)),
            ]
        )

    def _build_fit_subplot_gle_figure(self, glp, gle_path: Path):
        share_x_axis = self.share_x
        subplot_height = 3.1
        subplot_aspect = float(self.aspect)
        plot_width = subplot_aspect * subplot_height
        fig, axes = gle_subplots(
            glp,
            nrows=max(1, len(self._groups)),
            ncols=1,
            figsize=(plot_width, max(5.2, subplot_height * max(1, len(self._groups)))),
            sharex=share_x_axis,
        )
        subplot_axes = axes if isinstance(axes, list) else [axes]
        file_map = self._write_fit_subplot_files(gle_path)

        x_label = self._x_label_gle()
        y_label = self._parameter_label_gle(self._parameter_name)
        for idx, group in enumerate(self._groups):
            ax = subplot_axes[idx]
            file_info = file_map.get(group.group_id)
            data_path = file_info.get("data_path") if isinstance(file_info, dict) else None
            if (
                isinstance(file_info, dict)
                and data_path is not None
                and hasattr(ax, "errorbar_from_file")
            ):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=2,
                    yerr_col=3 if bool(file_info.get("has_err", False)) else None,
                    color="black",
                    marker="o",
                    markersize=4,
                    capsize=2,
                )
            else:
                yerr = np.asarray(group.yerr, dtype=float)
                has_err = bool(np.any(np.isfinite(yerr) & (yerr > 0)))
                ax.errorbar(
                    group.x,
                    group.y,
                    yerr=group.yerr if has_err else None,
                    fmt="o",
                    linestyle="none",
                    marker="o",
                    color="black",
                    capsize=2,
                )

            fit_path = file_info.get("fit_path") if isinstance(file_info, dict) else None
            sampled = self._sample_group_fit_curve(group)
            if sampled is not None:
                xx, yy = sampled
            else:
                xx, yy = None, None

            if xx is not None and self.show_components and self._model is not None:
                try:
                    kwargs = self._model_kwargs_for_group(group.group_id)
                    components = self._model.evaluate_components(xx, additive_only=True, **kwargs)
                    ordered = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xx, dtype=float)
                    component_colors = [
                        "lightblue",
                        "lightgreen",
                        "pink",
                        "lightgray",
                        "cyan",
                        "yellow",
                    ]
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    for cidx, (_name, comp_y) in enumerate(ordered):
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_token = self._safe_data_name(_name)
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(
                            xx,
                            lower,
                            upper,
                            color=fill_color,
                            alpha=0.3,
                            data_name=f"component_{group_token}_{comp_token}_fill",
                        )
                        ax.plot(
                            xx,
                            upper,
                            linestyle="--",
                            linewidth=0.8,
                            color=fill_color,
                            data_name=f"component_{group_token}_{comp_token}_edge",
                        )
                        cumulative = upper
                except KeyError:
                    pass

            if (
                isinstance(file_info, dict)
                and bool(file_info.get("has_fit", False))
                and fit_path is not None
                and hasattr(ax, "line_from_file")
            ):
                ax.line_from_file(
                    fit_path.name,
                    x_col=1,
                    y_col=2,
                    color="red",
                    linestyle="-",
                    linewidth=1.5,
                )
            else:
                if xx is not None and yy is not None:
                    group_token = self._safe_data_name(group.group_name or group.group_id)
                    ax.plot(xx, yy, color="red", linewidth=1.5, data_name=f"model_{group_token}")
                else:
                    ax.text(0.02, 0.95, "Fit curve unavailable", color="red", ha="left")

            for ann in self._plot_annotations:
                if str(ann.get("axis_tag", "")) != group.group_id:
                    continue
                is_group_label = bool(ann.get("is_group_label"))
                if is_group_label and not share_x_axis:
                    continue
                try:
                    x_ann = float(ann.get("x", 0.0))
                    y_ann = float(ann.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                text = str(ann.get("text", "")).strip()
                if text:
                    ax.text(
                        x_ann,
                        y_ann,
                        text,
                        ha="left",
                        va="top" if is_group_label else "bottom",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5}
                        if is_group_label
                        else None,
                    )

            if share_x_axis:
                self.group_label(group, ax)
                ax.set_title("")
            else:
                ax.set_title(group.group_name)
            ax.set_ylabel(y_label)
            self.apply_scales(ax)
            if (not share_x_axis) or idx == len(self._groups) - 1:
                ax.set_xlabel(x_label)

        if hasattr(fig, "subplots_adjust"):
            fig.subplots_adjust(**self._fit_subplot_layout_params(subplot_aspect))

        return fig

    def _build_local_parameter_gle_figure(self, glp, gle_path: Path):
        local_param_names = sorted(
            {p.name for pset in self._result.local_parameters.values() for p in pset}
        )
        if not local_param_names:
            fig = gle_figure(glp, figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            return fig

        traces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for pname in local_param_names:
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []
            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None or pname not in pset:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(pset[pname].value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(
                    float(err) if err is not None and np.isfinite(err) and err >= 0 else np.nan
                )
            if not xs:
                continue
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            e_arr = np.asarray(es, dtype=float)
            order = np.argsort(x_arr)
            traces.append((pname, x_arr[order], y_arr[order], e_arr[order]))

        all_traces = list(traces)
        selected_names = self.selected_local
        if selected_names:
            traces = [t for t in traces if t[0] in selected_names]
        if not traces:
            fig = gle_figure(glp, figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title("No local parameters selected")
            return fig

        data_path, col_map = self._write_local_parameter_data_file(gle_path, all_traces)
        fit_file_map = self._write_local_parameter_fit_files(gle_path, all_traces)

        x_label = self._local_group_axis_label_gle()
        plot_mode = self.local_plot_mode
        if plot_mode == "Subplots" and len(traces) > 1:
            num_cols = 1
            num_rows = (len(traces) + num_cols - 1) // num_cols
            fig, axes = gle_subplots(
                glp,
                nrows=num_rows,
                ncols=num_cols,
                figsize=(7.2, max(4.8, 2.8 * num_rows)),
                sharex=True,
            )
            flat_axes = axes if isinstance(axes, list) else [axes]
            for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
                ax = flat_axes[idx]
                has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
                cols = col_map[pname]
                if hasattr(ax, "errorbar_from_file"):
                    ax.errorbar_from_file(
                        data_path.name,
                        x_col=1,
                        y_col=cols[0],
                        yerr_col=cols[1] if has_err else None,
                        marker="o",
                        color="black",
                        capsize=2,
                    )
                else:
                    ax.errorbar(
                        x_arr,
                        y_arr,
                        yerr=e_arr if has_err else None,
                        marker="o",
                        color="black",
                        linestyle="none",
                        capsize=2,
                    )
                ax.set_ylabel(self._parameter_label_gle(pname))
                row = idx // num_cols
                if row == num_rows - 1:
                    ax.set_xlabel(x_label)
                if self.local_log_x:
                    ax.set_xscale("log")
                param_log = pname in self._local_param_log_y
                if param_log:
                    ax.set_yscale("log")
                fit = self._local_model_fits.get(pname)
                if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                    fit_path = fit_file_map.get(pname)
                    if fit_path is not None and hasattr(ax, "line_from_file"):
                        ax.line_from_file(
                            fit_path.name,
                            x_col=1,
                            y_col=2,
                            color="red",
                            linestyle="-",
                            linewidth=1.5,
                        )
                    else:
                        for curve in evaluate_parameter_model_fit(fit, num_points=200):
                            ax.plot(
                                curve.x,
                                curve.y,
                                color="red",
                                linewidth=1.5,
                                data_name=f"model_local_{self._safe_data_name(pname)}",
                            )
                for ann in self._local_plot_annotations:
                    if str(ann.get("axis_tag", "")) != pname:
                        continue
                    text = str(ann.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        x_ann = float(ann.get("x", 0.0))
                        y_ann = float(ann.get("y", 0.0))
                    except (TypeError, ValueError):
                        continue
                    ax.text(x_ann, y_ann, text, ha="left")
            return fig

        fig = gle_figure(glp, figsize=(7.0, 4.5))
        ax = fig.add_subplot(111)
        for idx, (pname, x_arr, y_arr, e_arr) in enumerate(traces):
            color = ["black", "blue", "red"][idx % 3]
            has_err = bool(np.any(np.isfinite(e_arr) & (e_arr >= 0)))
            cols = col_map[pname]
            if hasattr(ax, "errorbar_from_file"):
                ax.errorbar_from_file(
                    data_path.name,
                    x_col=1,
                    y_col=cols[0],
                    yerr_col=cols[1] if has_err else None,
                    marker="o",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            else:
                ax.errorbar(
                    x_arr,
                    y_arr,
                    yerr=e_arr if has_err else None,
                    marker="o",
                    linestyle="none",
                    color=color,
                    capsize=2,
                    label=self._parameter_label_gle(pname),
                )
            fit = self._local_model_fits.get(pname)
            if fit is not None and fit.active and fit.x_key == self._local_group_x_key():
                fit_path = fit_file_map.get(pname)
                if fit_path is not None and hasattr(ax, "line_from_file"):
                    ax.line_from_file(
                        fit_path.name,
                        x_col=1,
                        y_col=2,
                        color=color,
                        linestyle="-",
                        linewidth=1.2,
                    )
                else:
                    for curve in evaluate_parameter_model_fit(fit, num_points=200):
                        ax.plot(
                            curve.x,
                            curve.y,
                            color=color,
                            linewidth=1.2,
                            data_name=f"model_local_{self._safe_data_name(pname)}",
                        )
        ax.set_xlabel(x_label)
        if self.local_log_x:
            ax.set_xscale("log")
        if len(traces) == 1:
            param_log = traces[0][0] in self._local_param_log_y
            if param_log:
                ax.set_yscale("log")
        if len(traces) == 1:
            ax.set_ylabel(self._parameter_label_gle(traces[0][0]))
        else:
            ax.set_ylabel("Parameter Value")
            ax.legend(loc="best")

        for ann in self._local_plot_annotations:
            axis_tag = str(ann.get("axis_tag", ""))
            if axis_tag not in {"main"} and not any(name == axis_tag for name, *_rest in traces):
                continue
            text = str(ann.get("text", "")).strip()
            if not text:
                continue
            try:
                x_ann = float(ann.get("x", 0.0))
                y_ann = float(ann.get("y", 0.0))
            except (TypeError, ValueError):
                continue
            ax.text(x_ann, y_ann, text, ha="left")

        return fig

    def _fit_subplot_layout_params(self, subplot_aspect: float) -> dict[str, float]:
        """Return aspect-aware subplot spacing to reduce clipping across shapes."""
        # Narrower plots need more side margins for y-labels/ticks; wider plots can use tighter margins.
        t = max(0.0, min(1.0, (2.8 - float(subplot_aspect)) / 1.8))
        left = 0.11 + 0.09 * t
        right = 0.992 - 0.045 * t
        return {
            "left": left,
            "right": right,
            "top": 0.968,
            "bottom": 0.065,
            "hspace": 0.50,
        }

    def _ordered_components_for_stacking(
        self, components: list[tuple[str, np.ndarray]]
    ) -> list[tuple[str, np.ndarray]]:
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
