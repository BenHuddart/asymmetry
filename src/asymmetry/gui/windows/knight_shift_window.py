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

Sidebar reads top-to-bottom as the pipeline: Source → Conversion → Branches;
phase 2 adds branch assignment and the joint K(θ) model fit below them.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.knight_analysis import (
    KnightAnalysisInput,
    KnightAnalysisResult,
    KnightAnalysisState,
    evaluate,
)
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    REFERENCE_COMPONENT,
    KnightShiftConfig,
    KnightShiftUnit,
)
from asymmetry.core.utils.angles import wrap_angle_deg
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import field_width_for
from asymmetry.gui.widgets.action_footer import ActionFooter
from asymmetry.gui.widgets.panel_section import PanelSection

#: Display-unit choices in combo order.
_UNIT_CHOICES: tuple[tuple[str, KnightShiftUnit], ...] = (
    ("Auto (ppm / %)", KnightShiftUnit.AUTO),
    ("ppm", KnightShiftUnit.PPM),
    ("percent", KnightShiftUnit.PERCENT),
    ("fraction", KnightShiftUnit.FRACTION),
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
    """Knight-shift conversion and (phase 2) K(θ) model fitting for one series."""

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
        self._result: KnightAnalysisResult | None = None
        self._state = KnightAnalysisState(config=KnightShiftConfig(enabled=True))
        #: Guard so programmatic control updates never re-enter _reevaluate.
        self._updating_controls = False

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
        for widget in (
            self._ref_field_radio,
            self._ref_component_radio,
            self._ref_component_combo,
            unit_row,
            self._components_label,
            self._components_box,
        ):
            self._conversion_section.addWidget(widget)
        layout.addWidget(self._conversion_section)

        self._ref_field_radio.toggled.connect(self._on_controls_changed)
        self._ref_component_combo.currentIndexChanged.connect(self._on_controls_changed)
        self._unit_combo.currentIndexChanged.connect(self._on_controls_changed)

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
        self._state.fold_180 = self._fold_check.isChecked()
        self._state.show_markers = self._markers_check.isChecked()
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

    def _reevaluate(self) -> None:
        config = self._config_from_controls()
        self._state.config = config
        if self._snapshot is None:
            self._result = None
        else:
            self._result = evaluate(self._snapshot, config)
        self._update_source_labels()
        self._update_branch_rows()
        self._update_status()
        self._redraw()

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
        if result.skipped_points:
            parts.append(f"{result.skipped_points} skipped")
        self._footer.set_status(" · ".join(parts))
        self._send_btn.setEnabled(bool(result.branches))

    def _on_send_to_trend(self) -> None:
        self.apply_config_requested.emit(self._config_from_controls())

    def _redraw(self) -> None:
        if self._figure is None or self._canvas is None:
            return
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        snapshot, result = self._snapshot, self._result
        fold = self._fold_check.isChecked() and snapshot is not None and snapshot.x_key == "angle"
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
            if self._markers_check.isChecked() and not fold:
                for event in result.crossings:
                    mid = 0.5 * (float(event.x_left) + float(event.x_right))
                    ax.axvline(mid, color=tokens.BORDER, linestyle="--", linewidth=1.0, zorder=0)
            unit_suffix = f" ({result.unit_label})" if result.unit_label else ""
            ax.set_ylabel(f"K{unit_suffix}")
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
