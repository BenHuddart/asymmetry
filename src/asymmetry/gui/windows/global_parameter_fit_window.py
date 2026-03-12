"""Undocked window for cross-group global parameter fit results."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.parameter_models import CrossGroupFitResult, ParameterGroupData


class GlobalParameterFitWindow(QMainWindow):
    """Display cross-group fit data, fitted model curves, and global/local values."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Global Parameter Fit")
        self.resize(1200, 800)

        self._result: CrossGroupFitResult | None = None
        self._groups: list[ParameterGroupData] = []
        self._model = None
        self._parameter_name: str | None = None
        self._x_key: str = "run"
        self._fit_x_min: float = float("nan")
        self._fit_x_max: float = float("nan")

        self._axes_tag_map: dict[int, str] = {}
        self._plot_annotations: list[dict[str, object]] = []
        self._add_label_mode = False
        self._dragging_annotation: dict[str, object] | None = None

        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._left_canvas = None
        self._left_figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._left_figure = Figure(tight_layout=True)
            self._left_canvas = FigureCanvasQTAgg(self._left_figure)
            left_layout.addWidget(self._left_canvas)

            self._left_canvas.mpl_connect("button_press_event", self._on_canvas_button_press)
            self._left_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
            self._left_canvas.mpl_connect("button_release_event", self._on_canvas_button_release)
        except ImportError:
            pass

        right = QWidget()
        right_layout = QVBoxLayout(right)

        controls_row = QHBoxLayout()
        self._show_components_check = QCheckBox("Show components")
        self._show_components_check.toggled.connect(self._refresh_plot)
        controls_row.addWidget(self._show_components_check)

        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setCheckable(True)
        self._add_label_btn.toggled.connect(self._set_add_label_mode)
        controls_row.addWidget(self._add_label_btn)
        controls_row.addStretch()
        right_layout.addLayout(controls_row)

        self._params_table = QTableWidget(0, 3)
        self._params_table.setHorizontalHeaderLabels(["Parameter", "Value", "Uncertainty"])
        right_layout.addWidget(self._params_table)

        self._local_canvas = None
        self._local_figure = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._local_figure = Figure(tight_layout=True)
            self._local_canvas = FigureCanvasQTAgg(self._local_figure)
            right_layout.addWidget(self._local_canvas)
        except ImportError:
            pass

        self._export_btn = QPushButton("Export Plot to GLE")
        self._export_btn.clicked.connect(self._export_gle)
        right_layout.addWidget(self._export_btn)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])

    def has_result(self) -> bool:
        return self._result is not None

    def set_results(
        self,
        *,
        parameter_name: str,
        x_key: str,
        groups: list[ParameterGroupData],
        model,
        result: CrossGroupFitResult,
        fit_x_min: float = float("nan"),
        fit_x_max: float = float("nan"),
    ) -> None:
        self._parameter_name = parameter_name
        self._x_key = x_key
        self._groups = groups
        self._model = model
        self._result = result
        self._fit_x_min = float(fit_x_min)
        self._fit_x_max = float(fit_x_max)
        self._refresh_table()
        self._refresh_plot()
        self._refresh_local_parameter_plots()

    def _x_label(self) -> str:
        return {
            "field": "$B$ (G)",
            "temperature": "$T$ (K)",
            "run": "Run Number",
        }.get(self._x_key, "x")

    def _parameter_label(self, name: str | None) -> str:
        if not name:
            return "y"
        units = {
            "A": "%",
            "A0": "%",
            "A_bg": "%",
            "baseline": "%",
            "Lambda": "us^-1",
            "sigma": "us^-1",
            "Delta": "us^-1",
            "frequency": "MHz",
            "phase": "rad",
        }
        unit = units.get(name)
        return f"{name} ({unit})" if unit else name

    def _refresh_table(self) -> None:
        self._params_table.setRowCount(0)
        if self._result is None:
            return

        for p in self._result.global_parameters:
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(row, 0, QTableWidgetItem(p.name))
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            err = self._result.global_uncertainties.get(p.name)
            self._params_table.setItem(row, 2, QTableWidgetItem("" if err is None else f"{err:.3g}"))

        for p in self._result.fixed_parameters:
            row = self._params_table.rowCount()
            self._params_table.insertRow(row)
            self._params_table.setItem(row, 0, QTableWidgetItem(f"{p.name} (fixed)"))
            self._params_table.setItem(row, 1, QTableWidgetItem(f"{p.value:.6g}"))
            self._params_table.setItem(row, 2, QTableWidgetItem(""))

        self._params_table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        if self._left_canvas is None or self._left_figure is None:
            return
        self._left_figure.clear()
        self._axes_tag_map = {}
        if self._result is None or self._model is None:
            self._left_canvas.draw()
            return

        n = max(1, len(self._groups))
        x_label = self._x_label()
        y_label = self._parameter_label(self._parameter_name)
        for idx, group in enumerate(self._groups):
            ax = self._left_figure.add_subplot(n, 1, idx + 1)
            self._axes_tag_map[id(ax)] = group.group_id
            x = group.x
            y = group.y
            e = group.yerr
            ax.errorbar(x, y, yerr=e, fmt="o", color="black", capsize=2, label="Data")

            kwargs = {p.name: p.value for p in self._result.global_parameters}
            for p in self._result.fixed_parameters:
                kwargs[p.name] = p.value
            local = self._result.local_parameters.get(group.group_id)
            if local is not None:
                for p in local:
                    kwargs[p.name] = p.value
            if kwargs:
                xx = x
                if xx.size >= 2:
                    xx = xx.copy()
                    xx.sort()

                if np.isfinite(self._fit_x_min) and np.isfinite(self._fit_x_max) and self._fit_x_max > self._fit_x_min:
                    mask = (xx >= self._fit_x_min) & (xx <= self._fit_x_max)
                    xx = xx[mask]

                if xx.size >= 2:
                    xx = np.linspace(float(np.nanmin(xx)), float(np.nanmax(xx)), 200)

                if self._show_components_check.isChecked():
                    components = self._model.evaluate_components(xx, additive_only=True, **kwargs)
                    ordered = self._ordered_components_for_stacking(components)
                    cumulative = np.zeros_like(xx, dtype=float)
                    component_colors = ["#8ecae6", "#90be6d", "#f4a261", "#e5989b", "#bdb2ff", "#ffd166"]
                    for cidx, (_name, comp_y) in enumerate(ordered):
                        fill_color = component_colors[cidx % len(component_colors)]
                        comp_fill = np.maximum(np.asarray(comp_y, dtype=float), 0.0)
                        lower = cumulative
                        upper = cumulative + comp_fill
                        ax.fill_between(xx, lower, upper, color=fill_color, alpha=0.3, zorder=1)
                        ax.plot(xx, upper, linestyle="--", linewidth=0.8, color=fill_color, alpha=0.9, zorder=2)
                        cumulative = upper

                yy = self._model.function(xx, **kwargs)
                ax.plot(xx, yy, color="red", linewidth=1.5, label="Fit")
            ax.set_title(group.group_name)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(loc="best")
            if idx == n - 1:
                ax.set_xlabel(x_label)

        self._draw_plot_annotations()

        self._left_figure.tight_layout()
        self._left_canvas.draw()

    def _refresh_local_parameter_plots(self) -> None:
        if self._local_canvas is None or self._local_figure is None:
            return

        self._local_figure.clear()
        if self._result is None:
            self._local_canvas.draw()
            return

        local_param_names = sorted(
            {
                p.name
                for pset in self._result.local_parameters.values()
                for p in pset
            }
        )

        if not local_param_names:
            ax = self._local_figure.add_subplot(111)
            ax.set_title("No local parameters in this fit")
            ax.grid(True, alpha=0.3)
            self._local_canvas.draw()
            return

        n = len(local_param_names)
        x_label = {
            "field": "Group field (G)",
            "temperature": "Group temperature (K)",
            "run": "Group index",
        }.get(self._x_key, "Group variable")

        for idx, pname in enumerate(local_param_names):
            ax = self._local_figure.add_subplot(n, 1, idx + 1)
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []

            for group in self._groups:
                pset = self._result.local_parameters.get(group.group_id)
                if pset is None:
                    continue
                param = pset[pname] if pname in pset else None
                if param is None:
                    continue
                xs.append(float(group.group_variable_value))
                ys.append(float(param.value))
                err = self._result.local_uncertainties.get(group.group_id, {}).get(pname)
                es.append(float(err) if err is not None and np.isfinite(err) and err > 0 else np.nan)

            if xs:
                x_arr = np.asarray(xs, dtype=float)
                y_arr = np.asarray(ys, dtype=float)
                e_arr = np.asarray(es, dtype=float)
                order = np.argsort(x_arr)
                x_arr = x_arr[order]
                y_arr = y_arr[order]
                e_arr = e_arr[order]
                ax.scatter(x_arr, y_arr, s=16, zorder=6, color="C0")
                finite_err = np.isfinite(e_arr) & (e_arr > 0)
                if np.any(finite_err):
                    ax.errorbar(x_arr, y_arr, yerr=e_arr, fmt="none", ecolor="gray", capsize=2, elinewidth=1, zorder=5)

            ax.set_ylabel(pname)
            ax.set_title(f"Local {pname}")
            ax.grid(True, alpha=0.3)
            if idx == n - 1:
                ax.set_xlabel(x_label)

        self._local_figure.tight_layout()
        self._local_canvas.draw()

    def _ordered_components_for_stacking(self, components: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
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

    def _set_add_label_mode(self, enabled: bool) -> None:
        self._add_label_mode = bool(enabled)

    def _draw_plot_annotations(self) -> None:
        for ann in self._plot_annotations:
            ann["artist"] = None
            ax = self._axis_for_tag(str(ann.get("axis_tag", "")))
            if ax is None:
                continue
            text_artist = ax.text(
                float(ann.get("x", 0.0)),
                float(ann.get("y", 0.0)),
                str(ann.get("text", "")),
                fontsize=9,
                ha="left",
                va="bottom",
                zorder=9,
            )
            text_artist.set_picker(True)
            ann["artist"] = text_artist

    def _axis_for_tag(self, tag: str):
        if self._left_figure is None:
            return None
        for ax in self._left_figure.axes:
            if self._axes_tag_map.get(id(ax)) == tag:
                return ax
        return None

    def _annotation_at_event(self, event) -> dict[str, object] | None:
        for ann in self._plot_annotations:
            artist = ann.get("artist")
            if artist is None:
                continue
            contains, _ = artist.contains(event)
            if contains:
                return ann
        return None

    def _on_canvas_button_press(self, event) -> None:
        if event.inaxes is None:
            return

        ann = self._annotation_at_event(event)
        if event.button == 3 and ann is not None:
            self._plot_annotations.remove(ann)
            self._refresh_plot()
            return

        if event.button == 1 and event.dblclick and ann is not None:
            current = str(ann.get("text", ""))
            text, ok = QInputDialog.getText(self, "Edit Label", "Text:", text=current)
            if ok:
                ann["text"] = text
                self._refresh_plot()
            return

        if event.button == 1 and self._add_label_mode:
            text, ok = QInputDialog.getText(self, "Add Label", "Text:")
            if ok and text.strip():
                axis_tag = self._axes_tag_map.get(id(event.inaxes), "")
                self._plot_annotations.append(
                    {
                        "x": float(event.xdata),
                        "y": float(event.ydata),
                        "text": text.strip(),
                        "axis_tag": axis_tag,
                        "artist": None,
                    }
                )
                self._refresh_plot()
                self._add_label_btn.setChecked(False)
            return

        if event.button == 1 and ann is not None:
            self._dragging_annotation = ann

    def _on_canvas_motion(self, event) -> None:
        if self._dragging_annotation is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._dragging_annotation["x"] = float(event.xdata)
        self._dragging_annotation["y"] = float(event.ydata)
        artist = self._dragging_annotation.get("artist")
        if artist is not None:
            artist.set_position((float(event.xdata), float(event.ydata)))
            self._left_canvas.draw_idle()

    def _on_canvas_button_release(self, _event) -> None:
        self._dragging_annotation = None

    def _export_gle(self) -> None:
        if self._result is None or self._parameter_name is None:
            QMessageBox.information(self, "No result", "Run a cross-group fit first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Global Parameter Fit",
            "global_parameter_fit.gle",
            "GLE files (*.gle)",
        )
        if not path:
            return

        try:
            import importlib

            glp = importlib.import_module("gleplot")
            fig = glp.figure(figsize=(7.0, 4.5))
            ax = fig.add_subplot(111)
            for group in self._groups:
                ax.errorbar(group.x, group.y, yerr=group.yerr, marker="o", label=group.group_name)
            ax.legend(loc="best")
            fig.savefig(path)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", f"Could not export GLE: {exc}")
