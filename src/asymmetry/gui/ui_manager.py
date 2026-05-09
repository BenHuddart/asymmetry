"""Centralized UI policy for layout density, scaling, and panel visibility."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLayout,
    QMainWindow,
    QTableWidget,
    QToolBar,
)

COMPACT_MODE_SETTINGS_KEY = "ui/compact_mode"
UI_SCALE_SETTINGS_KEY = "ui/scale"
UI_SCALE_OPTIONS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)

_DEFAULT_UI_SCALE = 0.9
_DEFAULT_TOOLBAR_ICON_SIZE = QSize(16, 16)
_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
_SPIN_UP_ARROW_ICON = (_RESOURCE_DIR / "spin_up_arrow.svg").as_posix()
_SPIN_DOWN_ARROW_ICON = (_RESOURCE_DIR / "spin_down_arrow.svg").as_posix()


class UIManager(QObject):
    """Own UI density, scaling, and visibility behavior for the main window."""

    ui_scale_changed = Signal(float, float)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self._window = main_window
        self._settings = getattr(main_window, "_settings", QSettings())
        self._app = QApplication.instance()

        self.ui_scale = _DEFAULT_UI_SCALE

        self._dock_data_browser = getattr(main_window, "_dock_data_browser")
        self._dock_fit = getattr(main_window, "_dock_fit")
        self._dock_fourier = getattr(main_window, "_dock_fourier")
        self._dock_fit_parameters = getattr(main_window, "_dock_fit_parameters")
        self._dock_log = getattr(main_window, "_dock_log")

        self._data_browser = getattr(main_window, "_data_browser")
        self._fit_panel = getattr(main_window, "_fit_panel")
        self._fourier_panel = getattr(main_window, "_fourier_panel")
        self._fit_parameters_panel = getattr(main_window, "_fit_parameters_panel")
        self._log_panel = getattr(main_window, "_log_panel")
        self._plot_panel = getattr(main_window, "_plot_panel")

        self._main_toolbar = getattr(main_window, "_main_toolbar")
        self._ui_scale_actions: dict[float, QAction] = dict(
            getattr(main_window, "_ui_scale_actions", {})
        )

        self._base_font = QFont(self._app.font()) if self._app is not None else QFont()
        self._base_font_size = self._resolve_font_point_size(self._base_font)
        self._base_stylesheet = self._app.styleSheet() if self._app is not None else ""

        self._tracked_layouts = self._collect_layout_metrics()
        self._tracked_tables = self._collect_table_metrics()
        self._tracked_minimum_sizes = self._collect_minimum_size_metrics()
        self._tracked_toolbars = self._collect_toolbar_metrics()

    def restore_settings(self) -> None:
        """Restore persisted UI scale preferences and clear legacy compact-mode state."""
        self.ui_scale = _coerce_scale(
            self._settings.value(UI_SCALE_SETTINGS_KEY, _DEFAULT_UI_SCALE),
            default=_DEFAULT_UI_SCALE,
        )
        self._settings.remove(COMPACT_MODE_SETTINGS_KEY)
        self.apply_ui_scale()

    def bind_actions(self) -> None:
        """Connect menu and toolbar actions to UIManager-owned behavior."""
        for scale, action in self._ui_scale_actions.items():
            action.triggered.connect(lambda checked=False, s=scale: self.set_ui_scale(s))

    def set_ui_scale(self, scale: float) -> None:
        """Set the persisted base UI scale and reapply effective metrics."""
        next_scale = _coerce_scale(scale, default=self.ui_scale)
        if abs(next_scale - self.ui_scale) < 1e-9:
            self.apply_ui_scale()
            return
        self.ui_scale = next_scale
        self._settings.setValue(UI_SCALE_SETTINGS_KEY, next_scale)
        self._settings.sync()
        self.apply_ui_scale()

    def increase_scale(self) -> None:
        """Advance to the next configured UI scale option."""
        self.set_ui_scale(self._adjacent_scale(step=1))

    def decrease_scale(self) -> None:
        """Move to the previous configured UI scale option."""
        self.set_ui_scale(self._adjacent_scale(step=-1))

    def apply_ui_scale(self) -> None:
        """Apply effective font, stylesheet, layout, and widget metrics."""
        if self._app is not None:
            font = QFont(self._base_font)
            font.setPointSizeF(max(8.0, self._base_font_size * self.effective_scale))
            self._app.setFont(font)
            stylesheet = self.build_stylesheet(self.effective_scale)
            if self._base_stylesheet:
                stylesheet = f"{self._base_stylesheet}\n{stylesheet}".strip()
            self._app.setStyleSheet(stylesheet)

        self._apply_toolbar_metrics(self.effective_scale)
        self._apply_layout_metrics(self.effective_scale)
        self._apply_table_metrics(self.effective_scale)
        self._apply_minimum_size_metrics(self.effective_scale)

        self._sync_scale_controls()
        self.ui_scale_changed.emit(self.ui_scale, self.effective_scale)

    @property
    def effective_scale(self) -> float:
        """Return the active UI scale."""
        return self.ui_scale

    def show_panel(self, panel_key: str) -> None:
        """Show a panel in the standard dock layout."""
        dock = self._panel_dock(panel_key)
        dock.show()
        dock.raise_()

    def reset_layout(self) -> None:
        """Reset dock layout to the compact-friendly default shell."""
        self._window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_data_browser)
        self._window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit)
        self._window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fourier)
        self._window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._dock_fit_parameters,
        )
        self._window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)
        self._window.tabifyDockWidget(self._dock_fit, self._dock_fourier)
        self._window.tabifyDockWidget(self._dock_fit, self._dock_fit_parameters)

        defaults = self._default_full_visibility()
        self._dock_data_browser.setVisible(defaults["data"])
        self._dock_fit.setVisible(defaults["fit"])
        self._dock_fourier.setVisible(defaults["fourier"])
        self._dock_fit_parameters.setVisible(defaults["fit_parameters"])
        self._dock_log.setVisible(defaults["log"])
        self._window.resizeDocks(
            [self._dock_data_browser, self._dock_fit],
            [360, 360],
            Qt.Orientation.Horizontal,
        )
        self._window.resizeDocks([self._dock_log], [140], Qt.Orientation.Vertical)

    def build_stylesheet(self, scale: float) -> str:
        """Return the global stylesheet block for the requested scale."""
        button_padding_v = max(2, round(4 * scale))
        button_padding_h = max(4, round(8 * scale))
        input_padding_v = max(2, round(3 * scale))
        input_padding_h = max(4, round(6 * scale))
        border_radius = max(3, round(4 * scale))
        table_padding_v = max(1, round(2 * scale))
        table_padding_h = max(2, round(4 * scale))
        spin_button_width = max(16, round(18 * scale))
        spin_arrow_size = max(8, round(10 * scale))
        return f"""
QPushButton, QToolButton {{
    padding: {button_padding_v}px {button_padding_h}px;
    border: 1px solid #9aa4b2;
    border-radius: {border_radius}px;
    background-color: palette(button);
    color: palette(button-text);
}}
QPushButton:hover, QToolButton:hover {{
    border-color: #697586;
}}
QPushButton:disabled, QToolButton:disabled {{
    border-color: #c7ccd1;
    color: #7d8590;
}}
QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit, QPlainTextEdit {{
    padding: {input_padding_v}px {input_padding_h}px;
    border: 1px solid #9aa4b2;
    border-radius: {border_radius}px;
}}
QAbstractSpinBox {{
    padding-right: {spin_button_width + input_padding_h + 2}px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    width: {spin_button_width}px;
    background-color: palette(button);
    border-left: 1px solid #9aa4b2;
}}
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: {border_radius}px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-top: 1px solid #c7ccd1;
    border-bottom-right-radius: {border_radius}px;
}}
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{
    width: {spin_arrow_size}px;
    height: {spin_arrow_size}px;
}}
QAbstractSpinBox::up-arrow {{
    image: url({_SPIN_UP_ARROW_ICON});
}}
QAbstractSpinBox::down-arrow {{
    image: url({_SPIN_DOWN_ARROW_ICON});
}}
QTableWidget::item {{
    padding: {table_padding_v}px {table_padding_h}px;
}}
""".strip()

    def _panel_dock(self, panel_key: str) -> QDockWidget:
        return {
            "data": self._dock_data_browser,
            "fit": self._dock_fit,
            "fourier": self._dock_fourier,
            "fit_parameters": self._dock_fit_parameters,
            "log": self._dock_log,
        }[panel_key]

    def _default_full_visibility(self) -> dict[str, bool]:
        return {
            "data": True,
            "fit": False,
            "fourier": False,
            "fit_parameters": False,
            "log": True,
        }

    def _sync_scale_controls(self) -> None:
        if not self._ui_scale_actions:
            pass
        for scale, action in self._ui_scale_actions.items():
            action.blockSignals(True)
            action.setChecked(abs(scale - self.ui_scale) < 1e-9)
            action.blockSignals(False)

    def _adjacent_scale(self, *, step: int) -> float:
        options = list(UI_SCALE_OPTIONS)
        current_index = min(
            range(len(options)),
            key=lambda idx: abs(options[idx] - self.ui_scale),
        )
        next_index = max(0, min(len(options) - 1, current_index + step))
        return options[next_index]

    def _resolve_font_point_size(self, font: QFont) -> float:
        point_size = font.pointSizeF()
        if point_size > 0:
            return float(point_size)
        point_size_int = font.pointSize()
        if point_size_int > 0:
            return float(point_size_int)
        return 12.0

    def _collect_layout_metrics(self) -> list[tuple[QLayout, int, tuple[int, int, int, int]]]:
        tracked: list[tuple[QLayout, int, tuple[int, int, int, int]]] = []
        seen: set[int] = set()
        for widget in self._widgets_for_metric_scan():
            if widget is None:
                continue
            if widget.layout() is not None:
                self._append_layout_metric(widget.layout(), tracked, seen)
            for layout in widget.findChildren(QLayout):
                self._append_layout_metric(layout, tracked, seen)
        return tracked

    def _append_layout_metric(
        self,
        layout: QLayout,
        tracked: list[tuple[QLayout, int, tuple[int, int, int, int]]],
        seen: set[int],
    ) -> None:
        key = id(layout)
        if key in seen:
            return
        seen.add(key)
        margins = layout.contentsMargins()
        tracked.append(
            (
                layout,
                layout.spacing(),
                (margins.left(), margins.top(), margins.right(), margins.bottom()),
            )
        )

    def _collect_table_metrics(self) -> list[tuple[QTableWidget, int]]:
        tables: list[tuple[QTableWidget, int]] = []
        for table in self._tables_for_metric_scan():
            if table is None:
                continue
            base_height = table.verticalHeader().defaultSectionSize()
            tables.append((table, max(22, base_height)))
        return tables

    def _collect_minimum_size_metrics(self) -> list[tuple[Any, int, int]]:
        tracked: list[tuple[Any, int, int]] = []
        widgets = [
            self._dock_data_browser,
            self._dock_fit,
            self._dock_fourier,
            self._dock_fit_parameters,
            self._dock_log,
            self._data_browser,
            getattr(self._plot_panel, "_polarization_combo", None),
            getattr(self._fit_parameters_panel, "_y_selector_table", None),
        ]
        for widget in widgets:
            if widget is None:
                continue
            tracked.append((widget, widget.minimumWidth(), widget.minimumHeight()))
        return tracked

    def _collect_toolbar_metrics(self) -> list[tuple[QToolBar, QSize]]:
        tracked: list[tuple[QToolBar, QSize]] = []
        for toolbar in (self._main_toolbar,):
            icon_size = toolbar.iconSize()
            if not icon_size.isValid() or icon_size.isEmpty():
                icon_size = QSize(_DEFAULT_TOOLBAR_ICON_SIZE)
            tracked.append((toolbar, icon_size))
        return tracked

    def _apply_layout_metrics(self, scale: float) -> None:
        valid_layouts: list[tuple[QLayout, int, tuple[int, int, int, int]]] = []
        for layout, spacing, margins in self._tracked_layouts:
            try:
                if spacing >= 0:
                    layout.setSpacing(max(0, round(spacing * scale)))
                layout.setContentsMargins(
                    max(0, round(margins[0] * scale)),
                    max(0, round(margins[1] * scale)),
                    max(0, round(margins[2] * scale)),
                    max(0, round(margins[3] * scale)),
                )
            except RuntimeError:
                continue
            valid_layouts.append((layout, spacing, margins))
        self._tracked_layouts = valid_layouts

    def _apply_table_metrics(self, scale: float) -> None:
        row_height = max(22, round(26 * scale))
        valid_tables: list[tuple[QTableWidget, int]] = []
        for table, base_height in self._tracked_tables:
            try:
                table.verticalHeader().setDefaultSectionSize(
                    max(row_height, round(base_height * scale))
                )
            except RuntimeError:
                continue
            valid_tables.append((table, base_height))
        self._tracked_tables = valid_tables

    def _apply_minimum_size_metrics(self, scale: float) -> None:
        valid_widgets: list[tuple[Any, int, int]] = []
        for widget, min_width, min_height in self._tracked_minimum_sizes:
            try:
                if min_width > 0:
                    widget.setMinimumWidth(max(1, round(min_width * scale)))
                if min_height > 0:
                    widget.setMinimumHeight(max(1, round(min_height * scale)))
            except RuntimeError:
                continue
            valid_widgets.append((widget, min_width, min_height))
        self._tracked_minimum_sizes = valid_widgets

    def _apply_toolbar_metrics(self, scale: float) -> None:
        valid_toolbars: list[tuple[QToolBar, QSize]] = []
        for toolbar, base_icon_size in self._tracked_toolbars:
            try:
                toolbar.setIconSize(
                    QSize(
                        max(12, round(base_icon_size.width() * scale)),
                        max(12, round(base_icon_size.height() * scale)),
                    )
                )
            except RuntimeError:
                continue
            valid_toolbars.append((toolbar, base_icon_size))
        self._tracked_toolbars = valid_toolbars

    def _widgets_for_metric_scan(self) -> Sequence[Any]:
        return (
            self._window,
            self._plot_panel,
            self._data_browser,
            self._fit_panel,
            self._fit_parameters_panel,
            self._fourier_panel,
            self._log_panel,
        )

    def _tables_for_metric_scan(self) -> Sequence[QTableWidget | None]:
        return (
            getattr(self._data_browser, "_table", None),
            getattr(getattr(self._fit_panel, "_single_tab", None), "_param_table", None),
            getattr(getattr(self._fit_panel, "_global_tab", None), "_param_table", None),
            getattr(self._fit_parameters_panel, "_y_selector_table", None),
            getattr(self._fit_parameters_panel, "_table", None),
        )


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _coerce_scale(value: object, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(min(numeric, max(UI_SCALE_OPTIONS)), min(UI_SCALE_OPTIONS))
