"""Centralized UI policy for font-driven zoom and panel visibility.

The UI-scale mechanism is deliberately *font-driven* rather than per-widget
tracked. A scale change is exactly three publish steps:

1. Publish the scale to the font builders (:func:`set_ui_font_scale`) so any
   widget *built after* the change (dialogs, wizards, rebuilt sections) is born
   at the active scale.
2. Set the ``QApplication`` font to ``base × scale`` so every widget inheriting
   the application font re-scales live (the baseline font is cached on a
   QApplication dynamic property so repeated MainWindow construction never
   compounds the multiplier).
3. Regenerate one scale-derived QSS block appended to the cached base
   stylesheet. The block carries the scaled *chrome* font sizes
   (``#benchSectionHeader``, ``QHeaderView::section``, ``QGroupBox::title``,
   ``QDockWidget::title``) so existing chrome widgets — which set an explicit
   font that ignores the application font — re-scale live too. A size-only QSS
   ``font-size`` rule is merged onto the widget's existing font, so the
   builders' weight/letter-spacing/family survive and the two agree.

Widgets whose fonts QSS cannot reach — per-item ``QTableWidgetItem.setFont``
cells and explicit-mono panel fonts — re-derive their fonts from the builders on
the :attr:`UIManager.ui_scale_changed` signal, in the *owning* widget. There is
no central per-widget tracking layer.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
)

from asymmetry.gui.styles import tokens, typography
from asymmetry.gui.styles.fonts import set_ui_font_scale
from asymmetry.gui.styles.widgets import SECTION_HEADER_OBJECT_NAME

COMPACT_MODE_SETTINGS_KEY = "ui/compact_mode"
UI_SCALE_SETTINGS_KEY = "ui/scale"
UI_SCALE_OPTIONS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)

#: QApplication dynamic-property keys caching the pre-scale font/stylesheet
#: baseline. A UIManager constructed after another one has applied its scale
#: (every GUI test builds its own MainWindow against the shared QApplication)
#: must not capture the previous manager's *output* as its base: doing so
#: appended one rendered scale block to the app stylesheet per window and
#: multiplied the app font by the scale factor again each time.
_APP_BASE_STYLESHEET_PROP = "_asymmetry_base_stylesheet"
_APP_BASE_FONT_PROP = "_asymmetry_base_font"

_DEFAULT_UI_SCALE = 0.9
_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"
_SPIN_UP_ARROW_ICON = (_RESOURCE_DIR / "spin_up_arrow.svg").as_posix()
_SPIN_DOWN_ARROW_ICON = (_RESOURCE_DIR / "spin_down_arrow.svg").as_posix()
_CHECKMARK_ICON = (_RESOURCE_DIR / "checkmark.svg").as_posix()


class UIManager(QObject):
    """Own UI scaling and panel visibility behavior for the main window."""

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

        self._main_toolbar = getattr(main_window, "_main_toolbar")
        self._ui_scale_actions: dict[float, QAction] = dict(
            getattr(main_window, "_ui_scale_actions", {})
        )

        if self._app is not None:
            base_font = self._app.property(_APP_BASE_FONT_PROP)
            if base_font is None:
                base_font = QFont(self._app.font())
                self._app.setProperty(_APP_BASE_FONT_PROP, QFont(base_font))
            base_stylesheet = self._app.property(_APP_BASE_STYLESHEET_PROP)
            if base_stylesheet is None:
                base_stylesheet = self._app.styleSheet()
                self._app.setProperty(_APP_BASE_STYLESHEET_PROP, base_stylesheet)
            self._base_font = QFont(base_font)
            self._base_stylesheet = str(base_stylesheet)
        else:
            self._base_font = QFont()
            self._base_stylesheet = ""
        self._base_font_size = self._resolve_font_point_size(self._base_font)

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
        """Set the persisted base UI scale and reapply the effective font/stylesheet."""
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
        """Apply the effective UI scale via the three font-driven publish steps.

        1. Publish to the font builders so newly built widgets are born at scale.
        2. Set the application font to ``base × scale`` (existing inherited fonts
           re-scale live). The app font is set *before* the signal is emitted so
           owner hooks that read ``styles.metrics`` (which measures the live app
           font) see the new size.
        3. Regenerate the scale QSS block appended to the cached base stylesheet
           so explicit-font chrome (dock/section headers, table headers,
           group-box titles) re-scales live.
        """
        set_ui_font_scale(self.effective_scale)
        if self._app is not None:
            font = QFont(self._base_font)
            font.setPointSizeF(max(8.0, self._base_font_size * self.effective_scale))
            if self._app.font() != font:
                self._app.setFont(font)
            stylesheet = self.build_stylesheet(self.effective_scale)
            if self._base_stylesheet:
                stylesheet = f"{self._base_stylesheet}\n{stylesheet}".strip()
            # setStyleSheet reparses the sheet and repolishes every live widget
            # (~0.6s per call); skip it when the composed sheet is already active.
            if self._app.styleSheet() != stylesheet:
                self._app.setStyleSheet(stylesheet)

        self._apply_toolbar_icon_size()
        self._sync_scale_controls()
        self.ui_scale_changed.emit(self.ui_scale, self.effective_scale)

    def _apply_toolbar_icon_size(self) -> None:
        """Size the main toolbar's icons from the live application font.

        Derived once from the current font's line height (no per-widget capture
        list): icons that read as ~one text line tall track the UI scale for
        free, so a bigger zoom yields proportionally bigger tool icons.
        """
        toolbar = self._main_toolbar
        if toolbar is None:
            return
        line = QFontMetrics(self._app.font() if self._app is not None else QFont()).height()
        side = max(12, round(line * 0.9))
        toolbar.setIconSize(QSize(side, side))

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
        """Reset dock layout to the compact-friendly default shell.

        The inspector deck keeps its canonical tab order (Spectrum → Fit →
        Parameters); the main window re-applies per-representation visibility
        on top of this via _apply_inspector_for_domain.
        """
        self._window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_data_browser)
        self._window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fourier)
        self._window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_fit)
        self._window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self._dock_fit_parameters,
        )
        self._window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_log)
        for dock in (self._dock_fourier, self._dock_fit, self._dock_fit_parameters):
            dock.setFloating(False)
        self._window.tabifyDockWidget(self._dock_fourier, self._dock_fit)
        self._window.tabifyDockWidget(self._dock_fit, self._dock_fit_parameters)

        defaults = self._default_full_visibility()
        self._dock_data_browser.setVisible(defaults["data"])
        self._dock_fit.setVisible(defaults["fit"])
        self._dock_fourier.setVisible(defaults["fourier"])
        self._dock_fit_parameters.setVisible(defaults["fit_parameters"])
        self._dock_log.setVisible(defaults["log"])
        self._window.resizeDocks([self._dock_data_browser], [330], Qt.Orientation.Horizontal)
        self._window.resizeDocks([self._dock_log], [112], Qt.Orientation.Vertical)
        # Restore the inspector deck to its controls-fitting default width via
        # the same helper the launch path uses — the deck panes are tabified into
        # one region and must be resized as a group (the stale single-dock resize
        # this replaced left Reset Layout inconsistent with the launch default).
        # Mirror __init__ + showEvent: a synchronous pass seeds the layout, and a
        # deferred pass makes the width stick once the relayout from the re-add/
        # tabify above has settled.
        self._window._apply_default_dock_widths()
        QTimer.singleShot(0, self._window, self._window._apply_default_dock_widths)

    def build_stylesheet(self, scale: float) -> str:
        """Return the global stylesheet block for the requested scale.

        Carries the scaled chrome font sizes so explicit-font chrome widgets —
        which set a font that ignores the application font — re-scale live. Only
        ``font-size`` is set on those selectors so the builders' weight,
        letter-spacing, and (mono) family survive Qt's per-property merge; the
        size is single-sourced from the typography type scale, so the QSS and the
        Python builders can never drift.
        """
        button_padding_v = max(2, round(4 * scale))
        button_padding_h = max(4, round(8 * scale))
        input_padding_v = max(2, round(3 * scale))
        input_padding_h = max(4, round(6 * scale))
        border_radius = max(3, round(4 * scale))
        table_padding_v = max(1, round(2 * scale))
        table_padding_h = max(2, round(4 * scale))
        spin_button_width = max(16, round(18 * scale))
        spin_arrow_size = max(8, round(10 * scale))
        indicator_size = max(14, round(15 * scale))
        indicator_radius = max(2, round(3 * scale))
        # Chrome font-size single-sourced from the typography type scale (rather
        # than px/pt literals) and scaled so section/table/group/dock headers
        # track the UI scale on already-built widgets.
        header_pt = round(typography.SIZE_HEADER * scale, 2)
        t = tokens
        return f"""
QPushButton, QToolButton {{
    padding: {button_padding_v}px {button_padding_h}px;
    border: 1px solid {t.BORDER};
    border-radius: {border_radius}px;
    background-color: palette(button);
    color: palette(button-text);
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {t.BORDER_STRONG};
}}
QPushButton:disabled, QToolButton:disabled {{
    border-color: {t.BORDER};
    color: {t.TEXT_DIM};
}}
QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit, QPlainTextEdit {{
    padding: {input_padding_v}px {input_padding_h}px;
    border: 1px solid {t.BORDER};
    border-radius: {border_radius}px;
}}
QAbstractSpinBox {{
    padding-right: {spin_button_width + input_padding_h + 2}px;
}}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    width: {spin_button_width}px;
    background-color: palette(button);
    border-left: 1px solid {t.BORDER};
}}
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: {border_radius}px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-top: 1px solid {t.BORDER};
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
QComboBox::drop-down {{
    width: {spin_button_width}px;
    border-left: 1px solid {t.BORDER};
    background-color: palette(button);
    border-top-right-radius: {border_radius}px;
    border-bottom-right-radius: {border_radius}px;
}}
QComboBox::down-arrow {{
    width: {spin_arrow_size}px;
    height: {spin_arrow_size}px;
    image: url({_SPIN_DOWN_ARROW_ICON});
}}
QTableWidget::item {{
    padding: {table_padding_v}px {table_padding_h}px;
}}
QLabel#{SECTION_HEADER_OBJECT_NAME} {{
    font-size: {header_pt}pt;
}}
QDockWidget::title {{
    font-size: {header_pt}pt;
}}
QHeaderView::section {{
    font-size: {header_pt}pt;
}}
QGroupBox::title {{
    font-size: {header_pt}pt;
}}
QCheckBox::indicator,
QGroupBox::indicator,
QAbstractItemView::indicator {{
    width: {indicator_size}px;
    height: {indicator_size}px;
    border: 1px solid {t.BORDER_STRONG};
    border-radius: {indicator_radius}px;
    background-color: {t.SURFACE};
}}
QCheckBox::indicator:hover,
QGroupBox::indicator:hover,
QAbstractItemView::indicator:hover {{
    border-color: {t.ACCENT};
}}
QCheckBox::indicator:checked,
QGroupBox::indicator:checked,
QAbstractItemView::indicator:checked {{
    background-color: {t.ACCENT};
    border-color: {t.ACCENT};
    image: url({_CHECKMARK_ICON});
}}
QCheckBox::indicator:indeterminate,
QAbstractItemView::indicator:indeterminate {{
    background-color: {t.ACCENT_SOFT};
    border-color: {t.ACCENT};
}}
QCheckBox::indicator:disabled,
QGroupBox::indicator:disabled,
QAbstractItemView::indicator:disabled {{
    border-color: {t.BORDER};
    background-color: {t.SURFACE_ALT};
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
        # The inspector deck (fit/parameters) is visible by default; the
        # Spectrum dock joins it only for frequency views, applied by the main
        # window's _apply_inspector_for_domain on top of these defaults.
        return {
            "data": True,
            "fit": True,
            "fourier": False,
            "fit_parameters": True,
            "log": True,
        }

    def _sync_scale_controls(self) -> None:
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


def _coerce_scale(value: object, default: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(min(numeric, max(UI_SCALE_OPTIONS)), min(UI_SCALE_OPTIONS))
