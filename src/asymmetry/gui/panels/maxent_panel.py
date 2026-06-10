"""MaxEnt analysis panel for grouped-count spectral reconstruction."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.maxent import MaxEntConfig
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.widgets import apply_param_table_style


class MaxEntPanel(QWidget):
    """Controls for Maximum Entropy spectra."""

    #: Emitted when the "Show time-domain reconstruction" toggle changes.
    reconstruction_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table_group_ids: list[int] = []
        self._table_updating = False
        layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        content_layout = QVBoxLayout(content)

        groups_group = QGroupBox("Groups")
        groups_layout = QVBoxLayout(groups_group)
        self._group_table = QTableWidget(0, 3)
        self._group_table.setHorizontalHeaderLabels(["Include", "Group", "Phase (deg)"])
        apply_param_table_style(self._group_table)
        self._group_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._group_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._group_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._group_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._group_table.setMinimumHeight(100)
        groups_layout.addWidget(self._group_table)
        content_layout.addWidget(groups_group)

        spectrum_group = QGroupBox("Spectrum")
        spectrum_form = QFormLayout(spectrum_group)
        self._points_spin = QSpinBox()
        self._points_spin.setRange(8, 1 << 20)
        self._points_spin.setSingleStep(512)
        # 1024 resolves the line cleanly; the old 4096 default added a spurious
        # band-edge spike that dominated the spectrum even at low cycle counts.
        self._points_spin.setValue(1024)
        spectrum_form.addRow("Spectrum points:", self._points_spin)

        self._default_level_edit = QLineEdit("0.01")
        self._default_level_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._default_level_edit.setFont(mono_font(11.0))
        self._default_level_edit.setValidator(QDoubleValidator(1.0e-12, 1.0e6, 8, self))
        spectrum_form.addRow("Default level:", self._default_level_edit)
        content_layout.addWidget(spectrum_group)

        window_group = QGroupBox("Window")
        window_form = QFormLayout(window_group)
        self._auto_window_check = QCheckBox("Auto window from field")
        self._auto_window_check.setChecked(True)
        window_form.addRow(self._auto_window_check)

        self._half_width_edit = QLineEdit("300")
        self._half_width_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._half_width_edit.setFont(mono_font(11.0))
        self._half_width_edit.setValidator(QDoubleValidator(0.0, 1_000_000.0, 6, self))
        window_form.addRow("Half width (G):", self._half_width_edit)

        self._f_min_edit = QLineEdit("")
        self._f_min_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._f_min_edit.setFont(mono_font(11.0))
        self._f_min_edit.setValidator(QDoubleValidator(0.0, 1_000_000.0, 6, self))
        window_form.addRow("Min frequency (MHz):", self._f_min_edit)

        self._f_max_edit = QLineEdit("")
        self._f_max_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._f_max_edit.setFont(mono_font(11.0))
        self._f_max_edit.setValidator(QDoubleValidator(0.0, 1_000_000.0, 6, self))
        window_form.addRow("Max frequency (MHz):", self._f_max_edit)
        content_layout.addWidget(window_group)

        time_group = QGroupBox("Time")
        time_form = QFormLayout(time_group)
        self._t_min_edit = QLineEdit("")
        self._t_min_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._t_min_edit.setFont(mono_font(11.0))
        self._t_min_edit.setValidator(QDoubleValidator(-1_000_000.0, 1_000_000.0, 6, self))
        time_form.addRow("Start (μs):", self._t_min_edit)

        self._t_max_edit = QLineEdit("")
        self._t_max_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._t_max_edit.setFont(mono_font(11.0))
        self._t_max_edit.setValidator(QDoubleValidator(-1_000_000.0, 1_000_000.0, 6, self))
        time_form.addRow("End (μs):", self._t_max_edit)

        self._time_binning_spin = QSpinBox()
        self._time_binning_spin.setRange(1, 4096)
        self._time_binning_spin.setValue(1)
        time_form.addRow("Binning:", self._time_binning_spin)
        content_layout.addWidget(time_group)

        fit_group = QGroupBox("Cycle Refinement")
        fit_layout = QVBoxLayout(fit_group)
        fit_form = QFormLayout()
        self._inner_spin = QSpinBox()
        self._inner_spin.setRange(1, 200)
        self._inner_spin.setValue(12)
        fit_form.addRow("Inner iterations:", self._inner_spin)
        self._chi_target_edit = QLineEdit("1.0")
        self._chi_target_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._chi_target_edit.setFont(mono_font(11.0))
        self._chi_target_edit.setValidator(QDoubleValidator(1.0e-12, 1.0e6, 8, self))
        fit_form.addRow("χ² target / N:", self._chi_target_edit)
        fit_layout.addLayout(fit_form)

        self._fit_phases_check = QCheckBox("Fit phases")
        self._fit_phases_check.setChecked(True)
        self._fit_amplitudes_check = QCheckBox("Fit amplitudes")
        self._fit_amplitudes_check.setChecked(True)
        self._fit_backgrounds_check = QCheckBox("Fit backgrounds")
        self._fit_backgrounds_check.setChecked(True)
        self._fit_constant_background_check = QCheckBox("Fit constant background")
        self._fit_constant_background_check.setChecked(True)
        self._use_deadtime_check = QCheckBox("Use existing deadtime correction")
        self._use_deadtime_check.setChecked(True)
        for check in (
            self._fit_phases_check,
            self._fit_amplitudes_check,
            self._fit_backgrounds_check,
            self._fit_constant_background_check,
            self._use_deadtime_check,
        ):
            fit_layout.addWidget(check)
        content_layout.addWidget(fit_group)

        buttons = QGridLayout()
        self._cycle_one_btn = QPushButton("+1")
        self._cycle_five_btn = QPushButton("+5")
        self._cycle_twentyfive_btn = QPushButton("+25")
        self._converge_btn = QPushButton("Converge")
        self._restart_btn = QPushButton("Restart")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        buttons.addWidget(self._cycle_one_btn, 0, 0)
        buttons.addWidget(self._cycle_five_btn, 0, 1)
        buttons.addWidget(self._cycle_twentyfive_btn, 0, 2)
        buttons.addWidget(self._converge_btn, 1, 0, 1, 2)
        buttons.addWidget(self._restart_btn, 1, 2)
        buttons.addWidget(self._cancel_btn, 2, 0, 1, 3)
        content_layout.addLayout(buttons)

        self._show_reconstruction_check = QCheckBox("Show time-domain reconstruction")
        self._show_reconstruction_check.setChecked(False)
        self._show_reconstruction_check.setToolTip(
            "Overlay the per-group MaxEnt reconstruction on the measured data "
            "(time domain), with residuals. Available after a run."
        )
        self._show_reconstruction_check.toggled.connect(self.reconstruction_toggled.emit)
        content_layout.addWidget(self._show_reconstruction_check)

        self._apply_to_selection_btn = QPushButton("Apply settings to selected runs")
        content_layout.addWidget(self._apply_to_selection_btn)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self._progress_label = QLabel("")
        self._progress_label.setWordWrap(True)
        progress_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        progress_layout.addWidget(self._progress_bar)
        content_layout.addWidget(progress_group)

        diagnostics_group = QGroupBox("Diagnostics")
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self._diagnostics_label = QLabel("")
        self._diagnostics_label.setWordWrap(True)
        diagnostics_layout.addWidget(self._diagnostics_label)
        content_layout.addWidget(diagnostics_group)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        content_layout.addWidget(self._status_label)
        content_layout.addStretch()

        self._auto_window_check.toggled.connect(self._update_window_controls)
        self._update_window_controls()

    @staticmethod
    def _parse_float(text: str, default: float) -> float:
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _parse_optional_float(text: str) -> float | None:
        value = str(text).strip()
        if not value:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number

    @staticmethod
    def _format_float(value: object, default: str = "") -> str:
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int_keyed(table: dict | None, value_type) -> dict:
        """Return {int: value_type} from *table*, skipping malformed entries."""
        coerced: dict = {}
        for key, value in (table or {}).items():
            try:
                coerced[int(key)] = value_type(value)
            except (TypeError, ValueError):
                continue
        return coerced

    def _update_window_controls(self) -> None:
        auto = self._auto_window_check.isChecked()
        self._half_width_edit.setEnabled(auto)
        self._f_min_edit.setEnabled(not auto)
        self._f_max_edit.setEnabled(not auto)

    def set_group_definitions(
        self,
        group_names: dict[int, str],
        phase_table: dict[int, float] | None = None,
        enabled_table: dict[int, bool] | None = None,
    ) -> None:
        """Populate the editable group table.

        Tables can originate from project files, so malformed entries are
        skipped rather than raised.
        """
        phase_table = self._coerce_int_keyed(phase_table, float)
        enabled_table = self._coerce_int_keyed(enabled_table, bool)
        group_ids = sorted(int(group_id) for group_id in group_names)
        self._table_group_ids = group_ids
        self._group_table.setRowCount(len(group_ids))
        self._table_updating = True
        try:
            for row, group_id in enumerate(group_ids):
                include = QTableWidgetItem()
                include.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                include.setCheckState(
                    Qt.CheckState.Checked
                    if enabled_table.get(group_id, True)
                    else Qt.CheckState.Unchecked
                )
                self._group_table.setItem(row, 0, include)

                name = QTableWidgetItem(str(group_names[group_id]))
                name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._group_table.setItem(row, 1, name)

                phase = QTableWidgetItem(f"{phase_table.get(group_id, 0.0):.3f}")
                self._group_table.setItem(row, 2, phase)
        finally:
            self._table_updating = False

    def group_enabled_table(self) -> dict[int, bool]:
        """Return the per-group inclusion table."""
        enabled: dict[int, bool] = {}
        for row, group_id in enumerate(self._table_group_ids):
            item = self._group_table.item(row, 0)
            enabled[int(group_id)] = item is None or item.checkState() == Qt.CheckState.Checked
        return enabled

    def group_phase_table(self) -> dict[int, float]:
        """Return per-group phase seeds."""
        phases: dict[int, float] = {}
        for row, group_id in enumerate(self._table_group_ids):
            item = self._group_table.item(row, 2)
            phases[int(group_id)] = self._parse_float(item.text() if item else "", 0.0)
        return phases

    def selected_group_ids(self) -> list[int]:
        """Return included detector groups."""
        enabled = self.group_enabled_table()
        return [gid for gid in self._table_group_ids if enabled.get(int(gid), True)]

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the MaxEnt controls."""
        return {
            "n_spectrum_points": int(self._points_spin.value()),
            "default_level": self._parse_float(self._default_level_edit.text(), 0.01),
            "auto_window": bool(self._auto_window_check.isChecked()),
            "window_half_width_gauss": self._parse_float(self._half_width_edit.text(), 300.0),
            "f_min_mhz": self._parse_optional_float(self._f_min_edit.text()),
            "f_max_mhz": self._parse_optional_float(self._f_max_edit.text()),
            "t_min_us": self._parse_optional_float(self._t_min_edit.text()),
            "t_max_us": self._parse_optional_float(self._t_max_edit.text()),
            "time_binning_factor": int(self._time_binning_spin.value()),
            "inner_iterations": int(self._inner_spin.value()),
            "chi2_target_over_n": self._parse_float(self._chi_target_edit.text(), 1.0),
            "fit_phases": bool(self._fit_phases_check.isChecked()),
            "fit_amplitudes": bool(self._fit_amplitudes_check.isChecked()),
            "fit_backgrounds": bool(self._fit_backgrounds_check.isChecked()),
            "fit_constant_background": bool(self._fit_constant_background_check.isChecked()),
            "use_deadtime_correction": bool(self._use_deadtime_check.isChecked()),
            "show_reconstruction": bool(self._show_reconstruction_check.isChecked()),
            "selected_group_ids": self.selected_group_ids(),
            "group_enabled_table": self.group_enabled_table(),
            "group_phase_degrees": self.group_phase_table(),
        }

    def show_reconstruction_enabled(self) -> bool:
        """Return whether the reconstruction overlay toggle is checked."""
        return bool(self._show_reconstruction_check.isChecked())

    def set_show_reconstruction(self, checked: bool) -> None:
        """Set the reconstruction toggle without emitting ``reconstruction_toggled``.

        Used to keep the checkbox in step with the active workspace view when
        the view changes by some other route (e.g. the MaxEnt domain button).
        """
        blocker = self._show_reconstruction_check.blockSignals(True)
        self._show_reconstruction_check.setChecked(bool(checked))
        self._show_reconstruction_check.blockSignals(blocker)

    def maxent_config(self, *, cycles: int) -> MaxEntConfig:
        """Return a concrete core MaxEnt config for a cycle request."""
        state = self.get_state()
        return MaxEntConfig.from_dict({**state, "outer_cycles": int(cycles)})

    def restore_state(self, state: dict | None) -> None:
        """Restore panel controls from a saved dict."""
        if not isinstance(state, dict):
            return
        # ``None``/absent means "auto" (engine derives the grid from the data);
        # the spin box cannot represent that, so leave it unchanged rather
        # than silently pinning a hard-coded value into the next recipe.
        points = state.get("n_spectrum_points")
        if points is not None:
            try:
                self._points_spin.setValue(max(8, int(points)))
            except (TypeError, ValueError):
                pass
        self._default_level_edit.setText(
            self._format_float(state.get("default_level", 0.01), "0.01")
        )
        self._auto_window_check.setChecked(bool(state.get("auto_window", True)))
        self._half_width_edit.setText(
            self._format_float(state.get("window_half_width_gauss", 300.0), "300")
        )
        self._f_min_edit.setText(self._format_float(state.get("f_min_mhz"), ""))
        self._f_max_edit.setText(self._format_float(state.get("f_max_mhz"), ""))
        self._t_min_edit.setText(self._format_float(state.get("t_min_us"), ""))
        self._t_max_edit.setText(self._format_float(state.get("t_max_us"), ""))
        try:
            self._time_binning_spin.setValue(max(1, int(state.get("time_binning_factor", 1))))
        except (TypeError, ValueError):
            self._time_binning_spin.setValue(1)
        try:
            self._inner_spin.setValue(max(1, int(state.get("inner_iterations", 12))))
        except (TypeError, ValueError):
            self._inner_spin.setValue(12)
        self._chi_target_edit.setText(
            self._format_float(state.get("chi2_target_over_n", 1.0), "1.0")
        )
        self._fit_phases_check.setChecked(bool(state.get("fit_phases", True)))
        self._fit_amplitudes_check.setChecked(bool(state.get("fit_amplitudes", True)))
        self._fit_backgrounds_check.setChecked(bool(state.get("fit_backgrounds", True)))
        self._fit_constant_background_check.setChecked(
            bool(state.get("fit_constant_background", True))
        )
        self._use_deadtime_check.setChecked(bool(state.get("use_deadtime_correction", True)))
        blocker = self._show_reconstruction_check.blockSignals(True)
        self._show_reconstruction_check.setChecked(bool(state.get("show_reconstruction", True)))
        self._show_reconstruction_check.blockSignals(blocker)
        enabled = state.get("group_enabled_table")
        phases = state.get("group_phase_degrees")
        if self._table_group_ids and (isinstance(enabled, dict) or isinstance(phases, dict)):
            names = {
                int(group_id): self._group_table.item(row, 1).text()
                for row, group_id in enumerate(self._table_group_ids)
            }
            self.set_group_definitions(
                names,
                phases if isinstance(phases, dict) else None,
                enabled if isinstance(enabled, dict) else None,
            )
        self._update_window_controls()

    def set_status(self, message: str, *, success: bool = False, warning: bool = False) -> None:
        """Set status text below the action buttons.

        *warning* renders the message in the warning colour (and wins over
        *success*) so a divergence / early-stop notice is visible rather than
        buried in the small diagnostics line.
        """
        if warning:
            self._status_label.setText(
                f'<span style="color: {tokens.WARN}; font-weight: 600;">'
                f"{html.escape(str(message))}</span>"
            )
        elif success:
            self._status_label.setText(
                f'<span style="color: {tokens.OK};">{html.escape(str(message))}</span>'
            )
        else:
            self._status_label.setText(html.escape(str(message)))

    def set_busy(self, busy: bool) -> None:
        """Toggle MaxEnt calculation busy state."""
        buttons = (
            self._cycle_one_btn,
            self._cycle_five_btn,
            self._cycle_twentyfive_btn,
            self._converge_btn,
            self._restart_btn,
            self._apply_to_selection_btn,
        )
        for button in buttons:
            button.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._progress_bar.setVisible(busy)
        if busy:
            self._progress_bar.setRange(0, 0)
            self._progress_label.setText("Preparing MaxEnt calculation...")
        else:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
            self._progress_label.setText("")

    def set_progress(self, current: int, total: int, message: str) -> None:
        """Update progress indicator from the worker thread."""
        resolved_total = max(1, int(total))
        resolved_current = max(0, min(int(current), resolved_total))
        self._progress_bar.setRange(0, resolved_total)
        self._progress_bar.setValue(resolved_current)
        self._progress_label.setText(html.escape(str(message)))

    def set_diagnostics(self, diagnostics: dict | None) -> None:
        """Show a compact diagnostics summary."""
        if not isinstance(diagnostics, dict):
            self._diagnostics_label.setText("")
            return
        cycles = diagnostics.get("cycles") or []
        chi2 = diagnostics.get("chi2") or []
        test = diagnostics.get("test") or []
        entropy = diagnostics.get("entropy") or []
        if not cycles:
            self._diagnostics_label.setText("")
            return
        cycle = int(cycles[-1])
        chi2_value = float(chi2[-1]) if chi2 else 0.0
        test_value = float(test[-1]) if test else 0.0
        entropy_value = float(entropy[-1]) if entropy else 0.0
        self._diagnostics_label.setText(
            f"Cycle {cycle}; χ² {chi2_value:.4g}; TEST {test_value:.4g}; entropy {entropy_value:.4g}"
        )
