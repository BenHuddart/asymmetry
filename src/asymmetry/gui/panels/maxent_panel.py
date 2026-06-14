"""MaxEnt analysis panel for grouped-count spectral reconstruction."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from asymmetry.gui.panels.spectral_moments_widget import SpectralMomentsWidget
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import status_font
from asymmetry.gui.styles.widgets import apply_param_table_style, build_primary_button_qss


class MaxEntPanel(QWidget):
    """Controls for Maximum Entropy spectra."""

    #: Emitted when the "Show time-domain reconstruction" toggle changes.
    reconstruction_toggled = Signal(bool)
    #: Emitted when the reconstruction layout (per-group vs combined) changes.
    reconstruction_layout_changed = Signal(bool)
    #: Phase-exchange and calibration actions, handled by the main window.
    use_fitted_phases_requested = Signal()
    send_phases_to_fit_requested = Signal()
    fit_deadtime_requested = Signal()
    apply_deadtime_requested = Signal()
    export_spectrum_requested = Signal()
    export_log_requested = Signal()

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

        mode_group = QGroupBox("Mode")
        mode_form = QFormLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("General (multi-group)", userData="general")
        self._mode_combo.addItem("ZF / LF (two-group)", userData="zf_lf")
        self._mode_combo.setToolTip(
            "ZF/LF mode reconstructs a zero/longitudinal-field distribution from "
            "exactly two forward/backward groups, with phases pinned 0/180° and "
            "amplitudes tied through the run's α."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_form.addRow("Reconstruction:", self._mode_combo)
        content_layout.addWidget(mode_group)

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

        self._default_level_edit = self._make_numeric_edit(
            "0.01", minimum=1.0e-12, maximum=1.0e6, decimals=8
        )
        spectrum_form.addRow("Default level:", self._default_level_edit)
        content_layout.addWidget(spectrum_group)

        window_group = QGroupBox("Window")
        window_form = QFormLayout(window_group)
        self._auto_window_check = QCheckBox("Auto window from field")
        self._auto_window_check.setChecked(True)
        window_form.addRow(self._auto_window_check)

        self._half_width_edit = self._make_numeric_edit(
            "300", minimum=0.0, maximum=1_000_000.0, decimals=6
        )
        window_form.addRow("Half width (G):", self._half_width_edit)

        self._f_min_edit = self._make_numeric_edit("", minimum=0.0, maximum=1_000_000.0, decimals=6)
        window_form.addRow("Min frequency (MHz):", self._f_min_edit)

        self._f_max_edit = self._make_numeric_edit("", minimum=0.0, maximum=1_000_000.0, decimals=6)
        window_form.addRow("Max frequency (MHz):", self._f_max_edit)
        content_layout.addWidget(window_group)

        time_group = QGroupBox("Time")
        time_form = QFormLayout(time_group)
        self._t_min_edit = self._make_numeric_edit(
            "", minimum=-1_000_000.0, maximum=1_000_000.0, decimals=6
        )
        time_form.addRow("Start (μs):", self._t_min_edit)

        self._t_max_edit = self._make_numeric_edit(
            "", minimum=-1_000_000.0, maximum=1_000_000.0, decimals=6
        )
        time_form.addRow("End (μs):", self._t_max_edit)

        self._time_binning_spin = QSpinBox()
        self._time_binning_spin.setRange(1, 4096)
        self._time_binning_spin.setValue(1)
        time_form.addRow("Binning:", self._time_binning_spin)

        self._exclude_t_min_edit = self._make_numeric_edit(
            "",
            minimum=-1_000_000.0,
            maximum=1_000_000.0,
            decimals=6,
            tooltip=(
                "Start of an interior time window to exclude (e.g. a glitch). Points "
                "inside are de-weighted, not dropped — leave blank to disable."
            ),
        )
        time_form.addRow("De-weight from (μs):", self._exclude_t_min_edit)

        self._exclude_t_max_edit = self._make_numeric_edit(
            "", minimum=-1_000_000.0, maximum=1_000_000.0, decimals=6
        )
        time_form.addRow("De-weight to (μs):", self._exclude_t_max_edit)
        content_layout.addWidget(time_group)

        pulse_group = QGroupBox("Pulse shape (pulsed sources)")
        pulse_form = QFormLayout(pulse_group)
        self._pulse_mode_combo = QComboBox()
        self._pulse_mode_combo.addItem("Ignore", userData="ignore")
        self._pulse_mode_combo.addItem("Single pulse", userData="single")
        self._pulse_mode_combo.addItem("Double pulse", userData="double")
        self._pulse_mode_combo.setToolTip(
            "Correct the forward model for the finite muon pulse at a pulsed "
            "source (ISIS). Leave on 'Ignore' for continuous-source data."
        )
        pulse_form.addRow("Mode:", self._pulse_mode_combo)

        self._pulse_half_width_edit = self._make_numeric_edit(
            "0.05", minimum=0.0, maximum=1_000.0, decimals=6
        )
        pulse_form.addRow("Half-width (μs):", self._pulse_half_width_edit)

        self._pulse_separation_edit = self._make_numeric_edit(
            "0.324", minimum=0.0, maximum=1_000.0, decimals=6
        )
        pulse_form.addRow("Separation (μs):", self._pulse_separation_edit)
        content_layout.addWidget(pulse_group)

        fit_group = QGroupBox("Cycle Refinement")
        fit_layout = QVBoxLayout(fit_group)
        fit_form = QFormLayout()
        self._inner_spin = QSpinBox()
        self._inner_spin.setRange(1, 200)
        self._inner_spin.setValue(12)
        fit_form.addRow("Inner iterations:", self._inner_spin)
        self._chi_target_edit = self._make_numeric_edit(
            "1.0", minimum=1.0e-12, maximum=1.0e6, decimals=8
        )
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
        self._converge_btn.setStyleSheet(build_primary_button_qss())
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

        self._combine_reconstruction_check = QCheckBox("Combine groups on one axis")
        self._combine_reconstruction_check.setChecked(False)
        self._combine_reconstruction_check.setToolTip(
            "Overlay every selected group's reconstruction on a single colour-coded "
            "axis with a shared residuals strip, instead of stacking them per group."
        )
        self._combine_reconstruction_check.toggled.connect(self.reconstruction_layout_changed.emit)
        content_layout.addWidget(self._combine_reconstruction_check)

        self._apply_to_selection_btn = QPushButton("Apply settings to selected runs")
        content_layout.addWidget(self._apply_to_selection_btn)

        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        phase_buttons = QGridLayout()
        self._use_fitted_phases_btn = QPushButton("Use fitted phases")
        self._use_fitted_phases_btn.setToolTip(
            "Seed the group phases from the active run's grouped time-domain fit."
        )
        self._use_fitted_phases_btn.clicked.connect(self.use_fitted_phases_requested.emit)
        self._send_phases_to_fit_btn = QPushButton("Send phases to fit")
        self._send_phases_to_fit_btn.setToolTip(
            "Write the current MaxEnt group phases back to the grouped time-domain fit."
        )
        self._send_phases_to_fit_btn.clicked.connect(self.send_phases_to_fit_requested.emit)
        phase_buttons.addWidget(self._use_fitted_phases_btn, 0, 0)
        phase_buttons.addWidget(self._send_phases_to_fit_btn, 0, 1)
        calibration_layout.addLayout(phase_buttons)
        self._phase_provenance_label = QLabel("")
        self._phase_provenance_label.setWordWrap(True)
        self._phase_provenance_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        calibration_layout.addWidget(self._phase_provenance_label)

        deadtime_buttons = QGridLayout()
        self._fit_deadtime_btn = QPushButton("Fit deadtime")
        self._fit_deadtime_btn.setToolTip(
            "Estimate per-detector deadtime from the early-time count decay."
        )
        self._fit_deadtime_btn.clicked.connect(self.fit_deadtime_requested.emit)
        self._apply_deadtime_btn = QPushButton("Apply to grouping")
        self._apply_deadtime_btn.setToolTip(
            "Apply the fitted deadtime to the run's grouping deadtime correction."
        )
        self._apply_deadtime_btn.setEnabled(False)
        self._apply_deadtime_btn.clicked.connect(self.apply_deadtime_requested.emit)
        deadtime_buttons.addWidget(self._fit_deadtime_btn, 0, 0)
        deadtime_buttons.addWidget(self._apply_deadtime_btn, 0, 1)
        calibration_layout.addLayout(deadtime_buttons)
        self._deadtime_label = QLabel("")
        self._deadtime_label.setWordWrap(True)
        self._deadtime_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        calibration_layout.addWidget(self._deadtime_label)
        content_layout.addWidget(calibration_group)

        self._specbg_group = QGroupBox("Zero-frequency background (ZF/LF)")
        self._specbg_group.setCheckable(True)
        self._specbg_group.setChecked(False)
        self._specbg_group.setToolTip(
            "Subtract a zero-centred pseudo-Voigt model of the static central "
            "peak from the displayed field-distribution spectrum."
        )
        specbg_form = QFormLayout(self._specbg_group)
        self._specbg_gaussian_edit = self._make_numeric_edit(
            "0.1", minimum=0.0, maximum=1_000.0, decimals=6
        )
        specbg_form.addRow("Gaussian width (MHz):", self._specbg_gaussian_edit)
        self._specbg_lorentzian_edit = self._make_numeric_edit(
            "0.1", minimum=0.0, maximum=1_000.0, decimals=6
        )
        specbg_form.addRow("Lorentzian width (MHz):", self._specbg_lorentzian_edit)
        self._specbg_fraction_edit = self._make_numeric_edit(
            "0.5", minimum=0.0, maximum=1.0, decimals=6
        )
        specbg_form.addRow("Lorentzian fraction:", self._specbg_fraction_edit)
        content_layout.addWidget(self._specbg_group)

        export_buttons = QGridLayout()
        self._export_spectrum_btn = QPushButton("Export spectrum…")
        self._export_spectrum_btn.clicked.connect(self.export_spectrum_requested.emit)
        self._export_log_btn = QPushButton("Export log…")
        self._export_log_btn.clicked.connect(self.export_log_requested.emit)
        export_buttons.addWidget(self._export_spectrum_btn, 0, 0)
        export_buttons.addWidget(self._export_log_btn, 0, 1)
        content_layout.addLayout(export_buttons)

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

        # Spectral moments over the MaxEnt reconstruction (the canonical input);
        # the same widget class as the Fourier panel, wired by the host.
        self._moments_widget = SpectralMomentsWidget()
        content_layout.addWidget(self._moments_widget)

        self._status_label = QLabel("")
        self._status_label.setFont(status_font())
        self._status_label.setWordWrap(True)
        content_layout.addWidget(self._status_label)
        content_layout.addStretch()

        self._auto_window_check.toggled.connect(self._update_window_controls)
        self._update_window_controls()
        self._update_mode_dependent_controls()

    def _make_numeric_edit(
        self,
        text: str,
        *,
        minimum: float,
        maximum: float,
        decimals: int,
        tooltip: str | None = None,
    ) -> QLineEdit:
        """Return a right-aligned mono-font numeric ``QLineEdit`` with a validator.

        Collapses the repeated alignment/font/validator boilerplate shared by the
        panel's ~dozen numeric fields into one place.
        """
        edit = QLineEdit(text)
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        edit.setFont(mono_font(11.0))
        edit.setValidator(QDoubleValidator(minimum, maximum, decimals, self))
        if tooltip:
            edit.setToolTip(tooltip)
        return edit

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

    def apply_phase_table(self, phases_deg: dict[int, float]) -> int:
        """Set the Phase column for matching group ids; return the count updated.

        Used by the "Use fitted phases" exchange to seed the MaxEnt phases from a
        grouped time-domain fit without rebuilding the whole group table.
        """
        updated = 0
        self._table_updating = True
        try:
            for row, group_id in enumerate(self._table_group_ids):
                if int(group_id) in phases_deg:
                    item = self._group_table.item(row, 2)
                    if item is not None:
                        item.setText(self._format_float(phases_deg[int(group_id)], "0.0"))
                        updated += 1
        finally:
            self._table_updating = False
        return updated

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
            "exclude_t_min_us": self._parse_optional_float(self._exclude_t_min_edit.text()),
            "exclude_t_max_us": self._parse_optional_float(self._exclude_t_max_edit.text()),
            "pulse_mode": str(self._pulse_mode_combo.currentData() or "ignore"),
            "pulse_half_width_us": self._parse_float(self._pulse_half_width_edit.text(), 0.05),
            "pulse_separation_us": self._parse_float(self._pulse_separation_edit.text(), 0.324),
            "mode": str(self._mode_combo.currentData() or "general"),
            "specbg_enabled": bool(self._specbg_group.isChecked()),
            "specbg_gaussian_width_mhz": self._parse_float(self._specbg_gaussian_edit.text(), 0.1),
            "specbg_lorentzian_width_mhz": self._parse_float(
                self._specbg_lorentzian_edit.text(), 0.1
            ),
            "specbg_lorentzian_fraction": self._parse_float(self._specbg_fraction_edit.text(), 0.5),
            "inner_iterations": int(self._inner_spin.value()),
            "chi2_target_over_n": self._parse_float(self._chi_target_edit.text(), 1.0),
            "fit_phases": bool(self._fit_phases_check.isChecked()),
            "fit_amplitudes": bool(self._fit_amplitudes_check.isChecked()),
            "fit_backgrounds": bool(self._fit_backgrounds_check.isChecked()),
            "fit_constant_background": bool(self._fit_constant_background_check.isChecked()),
            "use_deadtime_correction": bool(self._use_deadtime_check.isChecked()),
            "show_reconstruction": bool(self._show_reconstruction_check.isChecked()),
            # Display-only layout preference (not a MaxEntConfig field — from_dict
            # ignores it); persisted with the panel state so the layout sticks.
            "reconstruction_combined": bool(self._combine_reconstruction_check.isChecked()),
            "selected_group_ids": self.selected_group_ids(),
            "group_enabled_table": self.group_enabled_table(),
            "group_phase_degrees": self.group_phase_table(),
            # Additive, namespaced moments sub-dict (no schema bump; W1).
            "moments": self._moments_widget.get_state(),
        }

    @property
    def moments_widget(self) -> SpectralMomentsWidget:
        """The spectral-moments control hosted under the reconstruction."""
        return self._moments_widget

    def show_reconstruction_enabled(self) -> bool:
        """Return whether the reconstruction overlay toggle is checked."""
        return bool(self._show_reconstruction_check.isChecked())

    def reconstruction_combined(self) -> bool:
        """Return whether the reconstruction overlay uses the combined layout."""
        return bool(self._combine_reconstruction_check.isChecked())

    def mode(self) -> str:
        """Return the selected reconstruction mode ("general" / "zf_lf")."""
        return str(self._mode_combo.currentData() or "general")

    def set_phase_provenance(self, text: str) -> None:
        """Show a provenance line for the last phase exchange (which fit, when)."""
        self._phase_provenance_label.setText(str(text))

    def set_deadtime_text(self, text: str, *, can_apply: bool = False) -> None:
        """Show the fitted-deadtime summary and enable/disable the apply button."""
        self._deadtime_label.setText(str(text))
        self._apply_deadtime_btn.setEnabled(bool(can_apply))

    def _on_mode_changed(self, _index: int) -> None:
        self._update_mode_dependent_controls()

    def _update_mode_dependent_controls(self) -> None:
        """Reflect the active mode: SpecBG is only meaningful in ZF/LF mode."""
        is_zf_lf = self.mode() == "zf_lf"
        self._specbg_group.setEnabled(is_zf_lf)

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
        self._moments_widget.restore_state(state.get("moments"))
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
        self._exclude_t_min_edit.setText(self._format_float(state.get("exclude_t_min_us"), ""))
        self._exclude_t_max_edit.setText(self._format_float(state.get("exclude_t_max_us"), ""))
        pulse_mode = str(state.get("pulse_mode", "ignore"))
        pulse_index = self._pulse_mode_combo.findData(pulse_mode)
        self._pulse_mode_combo.setCurrentIndex(pulse_index if pulse_index >= 0 else 0)
        self._pulse_half_width_edit.setText(
            self._format_float(state.get("pulse_half_width_us", 0.05), "0.05")
        )
        self._pulse_separation_edit.setText(
            self._format_float(state.get("pulse_separation_us", 0.324), "0.324")
        )
        mode = str(state.get("mode", "general"))
        mode_index = self._mode_combo.findData(mode)
        blocker = self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self._mode_combo.blockSignals(blocker)
        self._specbg_group.setChecked(bool(state.get("specbg_enabled", False)))
        self._specbg_gaussian_edit.setText(
            self._format_float(state.get("specbg_gaussian_width_mhz", 0.1), "0.1")
        )
        self._specbg_lorentzian_edit.setText(
            self._format_float(state.get("specbg_lorentzian_width_mhz", 0.1), "0.1")
        )
        self._specbg_fraction_edit.setText(
            self._format_float(state.get("specbg_lorentzian_fraction", 0.5), "0.5")
        )
        self._update_mode_dependent_controls()
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
        self._show_reconstruction_check.setChecked(bool(state.get("show_reconstruction", False)))
        self._show_reconstruction_check.blockSignals(blocker)
        blocker = self._combine_reconstruction_check.blockSignals(True)
        self._combine_reconstruction_check.setChecked(
            bool(state.get("reconstruction_combined", False))
        )
        self._combine_reconstruction_check.blockSignals(blocker)
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
                f'<span style="color: {tokens.OK};">● {html.escape(str(message))}</span>'
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
