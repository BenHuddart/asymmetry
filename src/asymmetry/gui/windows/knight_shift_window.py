"""The Knight shift analysis window.

A non-modal utility window that owns the frequency→Knight-shift derivation as a
first-class analysis, instead of squeezing it into the trend panel's derived
columns. The window is a thin shell over
:mod:`asymmetry.core.fitting.knight_analysis`: the trend panel supplies an
immutable :class:`~asymmetry.core.fitting.knight_analysis.KnightAnalysisInput`
snapshot, the sidebar edits a
:class:`~asymmetry.core.fitting.knight_shift.KnightShiftConfig`, and every
control change re-runs the pure :func:`~asymmetry.core.fitting.knight_analysis.
evaluate` and redraws.

The window never mutates the trend table. Publishing ``K[...]`` columns back to
the table is an explicit footer action (`Send K columns to trend table`) that
emits :attr:`KnightShiftWindow.apply_config_requested` for the MainWindow to
route — the reverse of the old flow where conversion silently rewrote the
table. Data refresh works the same way: :attr:`refresh_requested` asks the
owner for a fresh snapshot, so the window has no direct panel dependency and
stays constructible headless (tests, no-matplotlib installs).

Sidebar reads top-to-bottom as the pipeline: Source → Conversion → Branches →
Model fit. The joint K(θ) fit (classification-EM with per-angle Hungarian
assignment, run off-thread via :class:`~asymmetry.gui.tasks.TaskRunner`)
realigns the plotted branches so each follows one physical curve through
crossings; the fitted model curves overlay in branch colours and dashed
markers flag the angles where the assignment swaps. The run-keyed assignment
persists in :class:`~asymmetry.core.fitting.knight_analysis.KnightJointFitState`
and survives snapshot refreshes; a changed display unit only marks the fitted
curves stale (the assignment is unit-independent).
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.angular_assignment import ANGULAR_MODELS
from asymmetry.core.fitting.knight_analysis import (
    KnightAnalysisInput,
    KnightAnalysisResult,
    KnightAnalysisState,
    KnightCorrection,
    KnightJointFitState,
    apply_assignment,
    assignment_swap_positions,
    evaluate,
    run_joint_fit,
)
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
    label_for_unit,
)
from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    sample_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet, get_param_info
from asymmetry.core.utils.angles import wrap_angle_deg
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import field_width_for
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.action_footer import ActionFooter
from asymmetry.gui.widgets.no_scroll_spin import NoScrollSpinBox
from asymmetry.gui.widgets.panel_section import PanelSection

#: Display-unit choices in combo order.
_UNIT_CHOICES: tuple[tuple[str, KnightShiftUnit], ...] = (
    ("Auto (ppm / %)", KnightShiftUnit.AUTO),
    ("ppm", KnightShiftUnit.PPM),
    ("percent", KnightShiftUnit.PERCENT),
    ("fraction", KnightShiftUnit.FRACTION),
)

#: Sample-shape choices for the Lorentz/demag correction, in combo order.
_SHAPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Sphere (N = 1/3)", "sphere"),
    ("Thin plate, B ∥ plane (N = 0)", "plate_parallel"),
    ("Thin plate, B ⊥ plane (N = 1)", "plate_perpendicular"),
    ("Long cylinder, B ∥ axis (N = 0)", "cylinder_axial"),
    ("Long cylinder, B ⊥ axis (N = 1/2)", "cylinder_transverse"),
    ("Custom N", "custom"),
)

#: Fixed per-branch plot colours — the Okabe-Ito trace palette (colour-blind
#: safe), cycled when a model somehow carries more components than colours.
_BRANCH_COLORS = (
    tokens.TRACE_BLUE,
    tokens.TRACE_VERMILLION,
    tokens.TRACE_GREEN,
    tokens.TRACE_MAGENTA,
    tokens.TRACE_ORANGE,
    tokens.TRACE_SKY,
    tokens.TRACE_BLACK,
    tokens.TRACE_YELLOW,
)


class KnightShiftWindow(QMainWindow):
    """Knight-shift conversion and joint K(θ) model fitting for one series."""

    #: Emitted with the current :class:`KnightShiftConfig` when the user asks to
    #: publish ``K[...]`` columns back to the trend table.
    apply_config_requested = Signal(object)
    #: Emitted when the window wants a fresh snapshot from the owning trend
    #: series (Refresh button, or the owner may call :meth:`set_snapshot` any
    #: time the series is refit).
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Knight shift analysis")
        self.resize(980, 620)

        self._snapshot: KnightAnalysisInput | None = None
        #: The raw (label-ordered) derivation; the plot shows the realigned view
        #: via :meth:`_display_result` when a joint fit applies.
        self._result: KnightAnalysisResult | None = None
        self._state = KnightAnalysisState(config=KnightShiftConfig(enabled=True))
        #: Guard so programmatic control updates never re-enter _reevaluate.
        self._updating_controls = False
        self._tasks = TaskRunner(self)
        self._joint_running = False

        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        outer.addWidget(splitter, 1)

        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_plot_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 680])

        self._footer = ActionFooter(central)
        self._fit_btn = self._footer.add_primary("Run joint K(θ) fit")
        self._fit_btn.setToolTip(
            "Fit all branches jointly, assigning each angle's components one-to-one "
            "to the curve they fit best — resolves branch labels through crossings. "
            "Needs at least two branches and Angle as the scan axis."
        )
        self._fit_btn.clicked.connect(self._on_run_joint_fit)
        self._fit_btn.setEnabled(False)
        self._send_btn = self._footer.add_secondary("Send K columns to trend table")
        self._send_btn.setToolTip(
            "Publish the converted K[…] columns to the trend table so they can be "
            "plotted and exported alongside the fitted parameters."
        )
        self._send_btn.clicked.connect(self._on_send_to_trend)
        self._send_btn.setEnabled(False)
        outer.addWidget(self._footer)

        self._reevaluate()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget(self)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(10)

        # Source ---------------------------------------------------------------
        self._source_section = PanelSection(
            "Source", hint="Fitted series supplying the frequencies.", parent=sidebar
        )
        self._source_label = QLabel("No fitted series", sidebar)
        self._source_label.setWordWrap(True)
        self._source_detail = QLabel("", sidebar)
        self._source_detail.setWordWrap(True)
        self._source_detail.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._refresh_btn = QPushButton("Refresh from trend", sidebar)
        self._refresh_btn.setToolTip(
            "Rebuild the snapshot from the trend panel's current rows (after a refit "
            "or series change)."
        )
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        self._source_section.addWidget(self._source_label)
        self._source_section.addWidget(self._source_detail)
        self._source_section.addWidget(self._refresh_btn)
        layout.addWidget(self._source_section)

        # Conversion -----------------------------------------------------------
        self._conversion_section = PanelSection(
            "Conversion", hint="Reference and display unit for K.", parent=sidebar
        )
        self._ref_field_radio = QRadioButton("Applied field (γ_µ·B)", sidebar)
        self._ref_component_radio = QRadioButton("Designated component", sidebar)
        self._ref_group = QButtonGroup(self)
        self._ref_group.addButton(self._ref_field_radio)
        self._ref_group.addButton(self._ref_component_radio)
        self._ref_field_radio.setChecked(True)
        self._ref_component_combo = QComboBox(sidebar)
        self._ref_component_combo.setEnabled(False)
        unit_row = QWidget(sidebar)
        unit_layout = QHBoxLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setSpacing(6)
        unit_layout.addWidget(QLabel("Unit", unit_row))
        self._unit_combo = QComboBox(unit_row)
        for label, _unit in _UNIT_CHOICES:
            self._unit_combo.addItem(label)
        unit_layout.addWidget(self._unit_combo, 1)
        self._components_label = QLabel("Components", sidebar)
        self._components_box = QWidget(sidebar)
        self._components_layout = QVBoxLayout(self._components_box)
        self._components_layout.setContentsMargins(0, 0, 0, 0)
        self._components_layout.setSpacing(2)
        self._component_checks: dict[str, QCheckBox] = {}

        self._correction_check = QCheckBox("Lorentz/demag correction", sidebar)
        self._correction_check.setToolTip(
            "Correct the measured shift for the Lorentz and demagnetizing fields: "
            "K_µ = K_exp − (1/3 − N)·χ (Amato & Morenzoni Eq. 5.60). Vanishes for "
            "a sphere. Assumes the demagnetization factor N along the field stays "
            "fixed as the sample rotates — exact for a sphere, an approximation "
            "otherwise."
        )
        shape_row = QWidget(sidebar)
        shape_layout = QHBoxLayout(shape_row)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        shape_layout.setSpacing(6)
        shape_layout.addWidget(QLabel("Shape", shape_row))
        self._shape_combo = QComboBox(shape_row)
        for label, key in _SHAPE_CHOICES:
            self._shape_combo.addItem(label, key)
        shape_layout.addWidget(self._shape_combo, 1)
        n_chi_row = QWidget(sidebar)
        n_chi_layout = QHBoxLayout(n_chi_row)
        n_chi_layout.setContentsMargins(0, 0, 0, 0)
        n_chi_layout.setSpacing(6)
        n_chi_layout.addWidget(QLabel("N", n_chi_row))
        self._custom_n_edit = QLineEdit("0.3333", n_chi_row)
        self._custom_n_edit.setToolTip(
            "Demagnetization factor along the applied field (SI convention, 0–1)."
        )
        n_chi_layout.addWidget(self._custom_n_edit, 1)
        n_chi_layout.addWidget(QLabel("χ (SI)", n_chi_row))
        self._chi_edit = QLineEdit("0", n_chi_row)
        self._chi_edit.setToolTip(
            "Volume susceptibility, SI dimensionless (multiply a CGS emu/cm³ "
            "value by 4π). The K error bars do not include a χ uncertainty."
        )
        n_chi_layout.addWidget(self._chi_edit, 1)

        for widget in (
            self._ref_field_radio,
            self._ref_component_radio,
            self._ref_component_combo,
            unit_row,
            self._correction_check,
            shape_row,
            n_chi_row,
            self._components_label,
            self._components_box,
        ):
            self._conversion_section.addWidget(widget)
        layout.addWidget(self._conversion_section)

        self._ref_field_radio.toggled.connect(self._on_controls_changed)
        self._ref_component_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._correction_check.toggled.connect(self._on_controls_changed)
        self._shape_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._custom_n_edit.textChanged.connect(self._on_controls_changed)
        self._chi_edit.textChanged.connect(self._on_controls_changed)

        # Branches ---------------------------------------------------------------
        self._branches_section = PanelSection(
            "Branches", hint="One K trace per converted component.", parent=sidebar
        )
        self._branches_box = QWidget(sidebar)
        self._branches_layout = QVBoxLayout(self._branches_box)
        self._branches_layout.setContentsMargins(0, 0, 0, 0)
        self._branches_layout.setSpacing(2)
        self._crossings_label = QLabel("", sidebar)
        self._crossings_label.setWordWrap(True)
        self._branches_section.addWidget(self._branches_box)
        self._branches_section.addWidget(self._crossings_label)
        layout.addWidget(self._branches_section)

        # Model fit ------------------------------------------------------------
        self._fit_section = PanelSection(
            "Model fit", hint="Joint K(θ) fit with per-angle assignment.", parent=sidebar
        )
        model_row = QWidget(sidebar)
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(6)
        model_layout.addWidget(QLabel("Model", model_row))
        self._model_combo = QComboBox(model_row)
        for name in ANGULAR_MODELS:
            self._model_combo.addItem(name)
        model_layout.addWidget(self._model_combo, 1)
        self._model_info_btn = QPushButton("Info…", model_row)
        self._model_info_btn.setToolTip(
            "Formula, parameters, and applicability of the selected K(θ) model."
        )
        self._model_info_btn.clicked.connect(self._on_model_info)
        model_layout.addWidget(self._model_info_btn, 0)
        iter_row = QWidget(sidebar)
        iter_layout = QHBoxLayout(iter_row)
        iter_layout.setContentsMargins(0, 0, 0, 0)
        iter_layout.setSpacing(6)
        iter_layout.addWidget(QLabel("Max iterations", iter_row))
        self._max_iter_spin = NoScrollSpinBox(iter_row)
        self._max_iter_spin.setRange(1, 200)
        self._max_iter_spin.setValue(25)
        iter_layout.addWidget(self._max_iter_spin, 1)
        self._rescale_check = QCheckBox("Scale errors by √χ²ᵣ", sidebar)
        self._rescale_check.setToolTip(
            "Inflate the fitted parameter uncertainties by √χ²ᵣ when χ²ᵣ > 1 "
            "(the standard scale-factor treatment for a model that does not "
            "fully describe the data). Display only — the stored fit is unchanged."
        )
        self._rescale_check.toggled.connect(self._on_rescale_toggled)
        self._fit_results_label = QLabel("", sidebar)
        self._fit_results_label.setWordWrap(True)
        self._fit_results_label.setTextFormat(Qt.TextFormat.RichText)
        self._clear_fit_btn = QPushButton("Clear fit", sidebar)
        self._clear_fit_btn.setToolTip(
            "Discard the joint fit: branches return to their raw component labels."
        )
        self._clear_fit_btn.clicked.connect(self._on_clear_joint_fit)
        self._clear_fit_btn.setEnabled(False)
        self._fit_section.addWidget(model_row)
        self._fit_section.addWidget(iter_row)
        self._fit_section.addWidget(self._rescale_check)
        self._fit_section.addWidget(self._fit_results_label)
        self._fit_section.addWidget(self._clear_fit_btn)
        layout.addWidget(self._fit_section)

        layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(field_width_for(30, scroll))
        return scroll

    def _build_plot_area(self) -> QWidget:
        area = QWidget(self)
        layout = QVBoxLayout(area)
        layout.setContentsMargins(6, 0, 0, 0)
        layout.setSpacing(4)

        view_row = QWidget(area)
        view_layout = QHBoxLayout(view_row)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(12)
        self._fold_check = QCheckBox("Fold 180°", view_row)
        self._fold_check.setToolTip(
            "Wrap the angle axis into one 180° period so symmetry-equivalent orientations overlay."
        )
        self._markers_check = QCheckBox("Crossing markers", view_row)
        self._markers_check.setChecked(True)
        self._markers_check.setToolTip(
            "Mark scan intervals where components cross or run nearly degenerate."
        )
        view_layout.addWidget(self._fold_check)
        view_layout.addWidget(self._markers_check)
        view_layout.addStretch(1)
        layout.addWidget(view_row)

        self._figure = None
        self._canvas = None
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            self._figure, self._canvas = create_canvas(layout="tight")
        except ImportError:
            layout.addWidget(QLabel("matplotlib is not installed", area), 1)
        else:
            layout.addWidget(self._canvas, 1)

        self._fold_check.toggled.connect(self._on_view_changed)
        self._markers_check.toggled.connect(self._on_view_changed)
        return area

    # ── Snapshot / state ─────────────────────────────────────────────────────

    def set_snapshot(self, snapshot: KnightAnalysisInput | None) -> None:
        """Install a fresh measurement snapshot and re-derive everything.

        Rebuilds the component checkboxes and the reference combo to the
        snapshot's components, preserving the current config's selections where
        the names still exist.
        """
        self._snapshot = snapshot
        self._rebuild_component_controls()
        self._reevaluate()

    def get_state(self) -> dict:
        """Serializable window state (config + source binding + view flags)."""
        self._state.config = self._config_from_controls()
        self._state.correction = self._current_correction()
        self._state.fold_180 = self._fold_check.isChecked()
        self._state.show_markers = self._markers_check.isChecked()
        self._state.rescale_errors = self._rescale_check.isChecked()
        if self._snapshot is not None:
            self._state.source_batch_id = self._snapshot.batch_id
            self._state.source_group_id = self._snapshot.group_id
            self._state.x_key = self._snapshot.x_key
        return self._state.to_dict()

    def restore_state(self, state: object) -> None:
        """Restore a persisted state (before or after a snapshot arrives)."""
        self._state = KnightAnalysisState.from_dict(state)
        # The window *is* the conversion; a disabled legacy config just means
        # the dock columns were off, not that the analysis is meaningless.
        self._state.config.enabled = True
        self._apply_config_to_controls(self._state.config)
        self._updating_controls = True
        try:
            self._fold_check.setChecked(self._state.fold_180)
            self._markers_check.setChecked(self._state.show_markers)
            self._rescale_check.setChecked(self._state.rescale_errors)
            correction = self._state.correction
            self._correction_check.setChecked(correction.enabled)
            shape_index = self._shape_combo.findData(correction.shape)
            if shape_index >= 0:
                self._shape_combo.setCurrentIndex(shape_index)
            self._custom_n_edit.setText(f"{correction.custom_n:g}")
            self._chi_edit.setText(f"{correction.chi_volume_si:g}")
            joint = self._state.joint
            if joint is not None:
                index = self._model_combo.findText(joint.model_name)
                if index >= 0:
                    self._model_combo.setCurrentIndex(index)
                self._max_iter_spin.setValue(max(1, min(200, int(joint.max_iter))))
        finally:
            self._updating_controls = False
        self._reevaluate()

    def current_config(self) -> KnightShiftConfig:
        """The conversion config as currently edited (always enabled)."""
        return self._config_from_controls()

    # ── Control ↔ config mapping ─────────────────────────────────────────────

    def _config_from_controls(self) -> KnightShiftConfig:
        mode = (
            REFERENCE_COMPONENT
            if self._ref_component_radio.isChecked()
            else REFERENCE_APPLIED_FIELD
        )
        reference = self._ref_component_combo.currentData()
        if reference is None and self._ref_component_combo.count() == 0:
            # No snapshot yet: the empty combo carries no information, so a
            # restored-but-not-yet-displayed reference must survive round-trips
            # through this method (restore_state runs before set_snapshot).
            reference = self._state.config.reference_component
        checks = self._component_checks
        checked = tuple(name for name, box in checks.items() if box.isChecked())
        if not checks:
            # Likewise pre-snapshot: keep the pending subset instead of
            # collapsing it to "all components".
            components = self._state.config.components
        else:
            # All boxes ticked persists as "all components" so new components
            # joining the series after a refit are converted too.
            components = () if len(checked) == len(checks) else checked
        unit = _UNIT_CHOICES[max(0, self._unit_combo.currentIndex())][1]
        return KnightShiftConfig(
            enabled=True,
            reference_mode=mode,
            reference_component=str(reference)
            if mode == REFERENCE_COMPONENT and reference
            else None,
            unit=unit,
            components=components,
        )

    def _apply_config_to_controls(self, config: KnightShiftConfig) -> None:
        self._updating_controls = True
        try:
            if config.reference_mode == REFERENCE_COMPONENT:
                self._ref_component_radio.setChecked(True)
            else:
                self._ref_field_radio.setChecked(True)
            if config.reference_component is not None:
                index = self._ref_component_combo.findData(config.reference_component)
                if index >= 0:
                    self._ref_component_combo.setCurrentIndex(index)
            for unit_index, (_label, unit) in enumerate(_UNIT_CHOICES):
                if unit is config.unit:
                    self._unit_combo.setCurrentIndex(unit_index)
                    break
            for name, box in self._component_checks.items():
                box.setChecked(not config.components or name in config.components)
        finally:
            self._updating_controls = False

    def _rebuild_component_controls(self) -> None:
        """Match the component checkboxes and reference combo to the snapshot."""
        config = self._config_from_controls() if self._component_checks else self._state.config
        self._updating_controls = True
        try:
            while self._components_layout.count():
                item = self._components_layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self._component_checks = {}
            self._ref_component_combo.clear()
            components = self._snapshot.components if self._snapshot is not None else ()
            for name, kind in components:
                box = QCheckBox(f"{name} ({kind})", self._components_box)
                box.setChecked(not config.components or name in config.components)
                box.toggled.connect(self._on_controls_changed)
                self._components_layout.addWidget(box)
                self._component_checks[name] = box
                self._ref_component_combo.addItem(name, name)
            if config.reference_component is not None:
                index = self._ref_component_combo.findData(config.reference_component)
                if index >= 0:
                    self._ref_component_combo.setCurrentIndex(index)
            has_components = bool(self._component_checks)
            self._components_label.setVisible(has_components)
            self._ref_component_radio.setEnabled(has_components)
        finally:
            self._updating_controls = False

    # ── Derivation + display ─────────────────────────────────────────────────

    def _on_controls_changed(self, *_args: object) -> None:
        if self._updating_controls:
            return
        self._ref_component_combo.setEnabled(self._ref_component_radio.isChecked())
        self._reevaluate()

    def _on_view_changed(self, *_args: object) -> None:
        if not self._updating_controls:
            self._redraw()

    def _on_rescale_toggled(self, *_args: object) -> None:
        if not self._updating_controls:
            self._state.rescale_errors = self._rescale_check.isChecked()
            self._update_fit_controls()

    def _current_correction(self) -> KnightCorrection:
        """The Lorentz/demag correction as currently edited."""

        def _parse(edit: QLineEdit, fallback: float) -> float:
            try:
                value = float(edit.text().strip())
            except ValueError:
                return fallback
            return value if value == value else fallback

        return KnightCorrection(
            enabled=self._correction_check.isChecked(),
            shape=str(self._shape_combo.currentData() or "sphere"),
            custom_n=_parse(self._custom_n_edit, 1.0 / 3.0),
            chi_volume_si=_parse(self._chi_edit, 0.0),
        )

    def _reevaluate(self) -> None:
        config = self._config_from_controls()
        correction = self._current_correction()
        self._state.config = config
        self._state.correction = correction
        self._custom_n_edit.setEnabled(self._shape_combo.currentData() == "custom")
        if self._snapshot is None:
            self._result = None
        else:
            self._result = evaluate(self._snapshot, config, correction)
        self._update_source_labels()
        self._update_branch_rows()
        self._update_fit_controls()
        self._update_status()
        self._redraw()

    # ── Joint K(θ) fit ───────────────────────────────────────────────────────

    def _joint_applies(self) -> bool:
        """Whether the stored joint fit matches the current branch set.

        The run-keyed assignment stays applicable across snapshot refreshes as
        long as the branch count is unchanged; a different component selection
        (different branch names) invalidates it.
        """
        joint = self._state.joint
        result = self._result
        if joint is None or result is None or not joint.assignment:
            return False
        if len(result.branches) < 2:
            return False
        perm_len = len(next(iter(joint.assignment.values())))
        if perm_len != len(result.branches):
            return False
        if joint.curves and {c.branch_name for c in joint.curves} != {
            b.name for b in result.branches
        }:
            return False
        return True

    def _joint_curves_fresh(self) -> bool:
        """Whether the fitted curves match the current display unit and correction.

        Either change shifts/rescales every K value, so drawn curves and quoted
        parameters would no longer overlay the data; the assignment itself stays
        valid (a common offset or scale cannot reorder branches).
        """
        joint = self._state.joint
        result = self._result
        return (
            joint is not None
            and result is not None
            and joint.unit == result.unit.value
            and abs(joint.correction_offset - self._current_correction().offset()) < 1e-12
        )

    def _display_result(self) -> KnightAnalysisResult | None:
        """The result as plotted: realigned by the joint fit when it applies."""
        if self._result is not None and self._joint_applies():
            return apply_assignment(self._result, self._state.joint)
        return self._result

    def _can_run_joint_fit(self) -> bool:
        return (
            not self._joint_running
            and self._snapshot is not None
            and self._snapshot.x_key == "angle"
            and self._result is not None
            and len(self._result.branches) >= 2
        )

    def _on_run_joint_fit(self) -> None:
        if not self._can_run_joint_fit():
            return
        result = self._result
        model_name = self._model_combo.currentText()
        max_iter = int(self._max_iter_spin.value())
        correction_offset = self._current_correction().offset()
        self._joint_running = True
        self._fit_btn.setEnabled(False)
        self._footer.show_progress("Fitting K(θ)…")
        self._tasks.start(
            lambda _worker: run_joint_fit(
                result,
                model_name=model_name,
                max_iter=max_iter,
                correction_offset=correction_offset,
            ),
            on_finished=self._on_joint_fit_ready,
            on_error=self._on_joint_fit_error,
        )

    def _on_joint_fit_ready(self, joint: KnightJointFitState) -> None:
        self._joint_running = False
        self._footer.hide_progress()
        self._state.joint = joint
        self._update_fit_controls()
        self._update_status()
        self._redraw()

    def _on_joint_fit_error(self, message: str) -> None:
        self._joint_running = False
        self._footer.hide_progress()
        self._update_fit_controls()
        self._footer.set_status(f"Joint fit failed: {message}")

    def _on_clear_joint_fit(self) -> None:
        self._state.joint = None
        self._update_fit_controls()
        self._update_status()
        self._redraw()

    def _on_model_info(self) -> None:
        """Open the shared component-info dialog for the selected K(θ) model."""
        from asymmetry.core.fitting.parameter_models import PARAMETER_MODEL_COMPONENTS
        from asymmetry.gui.widgets.component_info_dialog import show_component_info_dialog

        definition = PARAMETER_MODEL_COMPONENTS.get(self._model_combo.currentText())
        if definition is not None:
            show_component_info_dialog(self, definition)

    def _update_fit_controls(self) -> None:
        self._fit_btn.setEnabled(self._can_run_joint_fit())
        joint = self._state.joint
        self._clear_fit_btn.setEnabled(joint is not None)
        if joint is None:
            self._fit_results_label.setText("")
            return
        applies = self._joint_applies()
        lines: list[str] = []
        if not applies:
            lines.append("<i>Stored fit does not match the current branches — re-run.</i>")
        elif not self._joint_curves_fresh():
            result = self._result
            cause = (
                "display unit"
                if result is not None and joint.unit != result.unit.value
                else "Lorentz/demag correction"
            )
            lines.append(f"<i>Fitted curves predate the current {cause} — re-run to refresh.</i>")
        table = self._fit_results_table(joint)
        if table:
            lines.append(table)
        if joint.message and not joint.converged:
            lines.append(f"<i>{joint.message}</i>")
        self._fit_results_label.setText("<br>".join(lines))

    def _fit_results_table(self, joint: KnightJointFitState) -> str:
        """The fitted parameters as a compact rich-text table.

        One row per branch (colour chip + K subscript), one column per model
        parameter plus χ²ᵣ. K-type parameters (no intrinsic unit) carry the fit
        unit in the header; θ0 keeps its own degree unit from the parameter
        registry. Values are shown to the precision the (optionally √χ²ᵣ-scaled)
        uncertainty supports.
        """
        from asymmetry.gui.utils.formatting import format_param_label, format_value_error

        if not joint.curves:
            return ""
        rescale = self._rescale_check.isChecked()
        try:
            fit_unit = KnightShiftUnit(joint.unit)
        except ValueError:
            fit_unit = KnightShiftUnit.AUTO
        # A joint fit migrated from a legacy project can carry AUTO — the
        # concrete unit it actually ran in was never recorded (the curves are
        # stale-by-construction; see _migrate_legacy_joint_fit). AUTO has no
        # label, so the K headers simply go unannotated.
        fit_unit_label = "" if fit_unit is KnightShiftUnit.AUTO else label_for_unit(fit_unit)

        def _html_subscript(label: str) -> str:
            # "K_iso (…)" → "K<sub>iso</sub> (…)": the unicode registry keeps
            # ASCII underscores for latin subscripts; rich text can do better.
            head, _, tail = label.partition(" ")
            if head.count("_") == 1 and not head.endswith("_"):
                base, sub = head.split("_")
                head = f"{base}<sub>{sub}</sub>"
            return f"{head} {tail}".strip()

        param_names = [name for name, _v, _e in joint.curves[0].parameters]
        headers = []
        for name in param_names:
            label = format_param_label(name)
            if get_param_info(name).unit is None and fit_unit_label:
                label = f"{label} ({fit_unit_label})"
            headers.append(_html_subscript(label))

        subscript_by_branch = {}
        if self._result is not None:
            subscript_by_branch = {b.name: b.subscript for b in self._result.branches}

        muted = tokens.TEXT_MUTED
        header_cells = "".join(
            f"<td align='center' style='color:{muted};'>{label}</td>" for label in [*headers, "χ²ᵣ"]
        )
        rows = [f"<tr><td></td>{header_cells}</tr>"]
        any_scaled = False
        for index, curve in enumerate(joint.curves):
            color = _BRANCH_COLORS[index % len(_BRANCH_COLORS)]
            chi2r = curve.reduced_chi_squared
            # The PDG-style scale factor: only ever inflates (χ²ᵣ < 1 is left
            # alone — an over-good fit does not license shrinking the errors).
            factor = math.sqrt(chi2r) if rescale and chi2r == chi2r and chi2r > 1.0 else 1.0
            any_scaled = any_scaled or factor > 1.0
            subscript = subscript_by_branch.get(curve.branch_name, str(index + 1))
            chip = f"<span style='color:{color};'>●</span>&nbsp;K<sub>{subscript}</sub>&nbsp;&nbsp;"
            cells = "".join(
                f"<td align='right'>{format_value_error(value, error * factor)}</td>"
                for _name, value, error in curve.parameters
            )
            chi_cell = f"<td align='right'>{chi2r:.3g}</td>" if chi2r == chi2r else "<td></td>"
            rows.append(f"<tr><td>{chip}</td>{cells}{chi_cell}</tr>")
        note = f"<div style='color:{muted};'>(errors ×√χ²ᵣ)</div>" if any_scaled else ""
        return f"<table cellspacing='0' cellpadding='2'>{''.join(rows)}</table>{note}"

    def _update_source_labels(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            self._source_label.setText("No fitted series")
            self._source_detail.setText(
                "Fit a series and open this window from the Fit Parameters panel."
            )
            return
        self._source_label.setText(snapshot.source_label or "Current trend series")
        self._source_detail.setText(
            f"{len(snapshot.points)} runs · {len(snapshot.components)} components · "
            f"x: {snapshot.x_label}"
        )

    def _update_branch_rows(self) -> None:
        while self._branches_layout.count():
            item = self._branches_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        result = self._result
        if result is None or not result.branches:
            self._crossings_label.setText("")
            self._branches_layout.addWidget(
                QLabel("No branches — no convertible components.", self._branches_box)
            )
            return
        for index, branch in enumerate(result.branches):
            color = _BRANCH_COLORS[index % len(_BRANCH_COLORS)]
            row = QLabel(
                f"<span style='color:{color};'>●</span> K<sub>{branch.subscript}</sub>"
                f" ← {branch.component} · {len(branch.k)} points",
                self._branches_box,
            )
            row.setTextFormat(Qt.TextFormat.RichText)
            self._branches_layout.addWidget(row)
        n_cross = len(result.crossings)
        if n_cross:
            self._crossings_label.setText(
                f"{n_cross} crossing{'s' if n_cross != 1 else ''} flagged along the scan — "
                "branch labels may swap there."
            )
        else:
            self._crossings_label.setText("No crossings flagged.")

    def _update_status(self) -> None:
        result = self._result
        if self._snapshot is None or result is None:
            self._footer.set_status("No data — open from a fitted trend series.")
            self._send_btn.setEnabled(False)
            return
        parts = [
            f"{len(result.branches)} branch{'es' if len(result.branches) != 1 else ''}",
            f"{len(self._snapshot.points)} runs",
        ]
        if result.unit_label:
            parts.append(f"unit {result.unit_label}")
        if self._current_correction().enabled:
            parts.append("Lorentz/demag corrected")
        if result.skipped_points:
            parts.append(f"{result.skipped_points} skipped")
        if self._joint_applies():
            joint = self._state.joint
            swaps = len(assignment_swap_positions(self._result, joint))
            parts.append(
                f"joint {joint.model_name} fit"
                + (f" · {swaps} swap{'s' if swaps != 1 else ''}" if swaps else "")
            )
        self._footer.set_status(" · ".join(parts))
        self._send_btn.setEnabled(bool(result.branches))

    def _on_send_to_trend(self) -> None:
        self.apply_config_requested.emit(self._config_from_controls())

    def _redraw(self) -> None:
        if self._figure is None or self._canvas is None:
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        snapshot, result = self._snapshot, self._display_result()
        fold = self._fold_check.isChecked() and snapshot is not None and snapshot.x_key == "angle"
        joint_active = self._joint_applies()
        if result is not None and result.branches:
            for index, branch in enumerate(result.branches):
                color = _BRANCH_COLORS[index % len(_BRANCH_COLORS)]
                xs = [wrap_angle_deg(x) if fold else x for x in branch.x]
                ys = [k * result.scale for k in branch.k]
                errs = [e * result.scale for e in branch.k_err]
                label = f"$K_{{{branch.subscript}}}$"
                # Included and excluded points draw separately so the excluded
                # ones read as greyed-out context rather than data.
                for included_flag, alpha, fill in ((True, 1.0, color), (False, 0.35, "none")):
                    sel = [i for i, inc in enumerate(branch.included) if inc == included_flag]
                    if not sel:
                        continue
                    ax.errorbar(
                        [xs[i] for i in sel],
                        [ys[i] for i in sel],
                        yerr=[errs[i] for i in sel],
                        fmt="o",
                        ms=4,
                        color=color,
                        markerfacecolor=fill,
                        alpha=alpha,
                        linestyle="none",
                        label=label if included_flag else None,
                    )
            # Fitted K(θ) curves (drawn unfolded: the raw scan coordinate is the
            # frame the fit ran in), only while the fit unit matches the display.
            if joint_active and not fold and self._joint_curves_fresh():
                self._draw_joint_curves(ax, result)
            if self._markers_check.isChecked() and not fold:
                if joint_active:
                    # Assignment swaps are the crossings the fit resolved — a
                    # firmer signal than the raw proximity flags.
                    for x_swap in assignment_swap_positions(self._result, self._state.joint):
                        ax.axvline(
                            x_swap,
                            color=tokens.BORDER_STRONG,
                            linestyle="--",
                            linewidth=1.0,
                            zorder=0,
                        )
                else:
                    for event in result.crossings:
                        mid = 0.5 * (float(event.x_left) + float(event.x_right))
                        ax.axvline(
                            mid, color=tokens.BORDER, linestyle="--", linewidth=1.0, zorder=0
                        )
            unit_suffix = f" ({result.unit_label})" if result.unit_label else ""
            symbol = r"$K_\mu$" if self._current_correction().enabled else "K"
            ax.set_ylabel(f"{symbol}{unit_suffix}")
            ax.set_xlabel(snapshot.x_label if snapshot is not None else "")
            ax.legend(loc="best", fontsize="small")
        else:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "No Knight-shift branches to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.grid(alpha=0.3)
        self._canvas.draw_idle()

    def _draw_joint_curves(self, ax: object, result: KnightAnalysisResult) -> None:
        """Overlay the fitted K(θ) model curves in branch colours."""
        joint = self._state.joint
        finite = [x for b in result.branches for x in b.x if x == x]
        if joint is None or not finite:
            return
        x_min, x_max = min(finite), max(finite)
        model = ParameterCompositeModel([joint.model_name])
        by_name = {c.branch_name: c for c in joint.curves}
        for index, branch in enumerate(result.branches):
            curve = by_name.get(branch.name)
            if curve is None or not curve.success:
                continue
            parameters = ParameterSet(
                [Parameter(name=name, value=value) for name, value, _e in curve.parameters]
            )
            xs, ys = sample_parameter_model(model, parameters, x_min, x_max)
            if xs.size == 0:
                continue
            ax.plot(
                xs,
                ys,
                color=_BRANCH_COLORS[index % len(_BRANCH_COLORS)],
                linewidth=1.4,
                alpha=0.85,
                zorder=1,
            )

    def closeEvent(self, event: object) -> None:  # noqa: N802 — Qt override
        self._tasks.shutdown()
        super().closeEvent(event)
