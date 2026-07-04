"""WiMDA-style shared grouping dialog with alpha estimation and .grp I/O.

The dialog edits detector grouping once and applies it across multiple datasets
in the active project. Grouping definitions can be saved to and loaded from
``.grp`` files.

This module holds the :class:`GroupingDialog` shell. The line-based ``.grp``
serialization and the alpha-estimate display helpers live in
:mod:`asymmetry.gui.windows.grouping.grp_io`; the dialog's static
``serialize_grp``/``parse_grp`` methods and the module-level
``_format_value_with_uncertainty``/``_ALPHA_METHOD_ITEMS`` names delegate to (or
alias) that module so the historical class/module API is unchanged after the
package split.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.instrument import (
    CANONICAL_VECTOR_AXES,
    derive_projection_pairs,
    detect_instrument,
    get_instrument_layout,
    instrument_display_name,
    recommend_grouping_preset,
    variant_for_histograms,
)
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    DeadtimePolicy,
    GroupingProfile,
    ProfileFingerprint,
    profile_fingerprint_for_run,
)
from asymmetry.core.transform import (
    apply_grouping,
    available_background_modes,
    calibrate_deadtime_from_histograms,
    estimate_deadtime_from_histograms,
    excluded_detector_indices,
    filter_excluded_indices,
    find_t0_for_run,
    fit_tail_background,
    format_detector_list,
    parse_detector_list,
    resolve_background_mode,
    resolve_binning_mode,
)
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import apply_param_table_style, clear_layout
from asymmetry.gui.windows.grouping.background_dialog import (
    BackgroundDialog,
    BackgroundReferenceRunCandidate,
)
from asymmetry.gui.windows.grouping.deadtime_dialog import (
    DeadtimeDialog,
    DeadtimeSourceRun,
    deadtime_status_text,
)
from asymmetry.gui.windows.grouping.grp_io import (
    ALPHA_METHOD_ITEMS as _ALPHA_METHOD_ITEMS,
)
from asymmetry.gui.windows.grouping.grp_io import (
    format_value_with_uncertainty as _format_value_with_uncertainty,
)
from asymmetry.gui.windows.grouping.grp_io import (
    parse_grp as _parse_grp,
)
from asymmetry.gui.windows.grouping.grp_io import (
    serialize_grp as _serialize_grp,
)
from asymmetry.gui.windows.grouping.preview_pane import GroupingPreviewPane
from asymmetry.gui.windows.grouping.profile_bridge import (
    payload_matches_preset,
    preset_payload,
    profile_from_form_payload,
)
from asymmetry.gui.windows.grouping.scope_panel import ScopePanel


class GroupingDialog(QDialog):
    """Profile editor for detector grouping.

    The dialog edits an in-memory *draft* grouping profile (a
    :class:`~asymmetry.core.project.profiles.GroupingProfile`). Runs of a
    fingerprint ``(instrument, histogram_count)`` inherit the fingerprint's
    active profile; per-run divergence is an explicit "release from profile"
    exception managed in the scope panel. Nothing touches the project or runs
    until Apply.

    The form controls (binning, deadtime, background, period, alpha) edit the
    draft through a flat ``run.grouping`` *payload* (see
    :mod:`asymmetry.gui.windows.grouping.profile_bridge`); a *preview run*
    selector supplies the per-run facts (t0, good-bin window) shown read-only,
    and changing it never discards draft edits.

    Parameters
    ----------
    datasets
        Datasets available in the active project. Datasets without raw
        histograms are ignored.
    profiles
        The project's existing grouping profiles. When omitted (or empty for the
        current fingerprint), the draft is synthesized from the current/reference
        run's payload as ``"Default (<instrument>)"``.
    selected_run_number
        Optional run number used as the initial preview run.
    selected_run_numbers
        Optional run numbers; only the first choice steers the initial preview
        run / fingerprint. No longer a broadcast selection.
    parent
        Parent Qt widget.
    """

    def __init__(
        self,
        datasets: list[MuonDataset],
        *,
        profiles: list[GroupingProfile] | None = None,
        overridden_run_numbers: list[int] | None = None,
        selected_run_number: int | None = None,
        selected_run_numbers: list[int] | None = None,
        parent=None,
    ) -> None:
        """Create a grouping profile editor for project datasets."""
        super().__init__(parent)
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._project_profiles = list(profiles or [])
        self._overridden_run_numbers = {int(v) for v in (overridden_run_numbers or [])}
        self._selected_run_number = selected_run_number
        self._selected_run_numbers = (
            {int(v) for v in selected_run_numbers} if selected_run_numbers is not None else None
        )
        # Draft-editing state (set up once the reference/preview run is known).
        self._draft_dirty = False
        self._suppress_dirty = False
        self._draft_name = ""
        self._fingerprint: ProfileFingerprint | None = None

        self.setWindowTitle("Grouping")
        self.resize(940, 560)

        if not self._datasets:
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("No runs are available in the active project."))
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)
            return

        self._reference_dataset = self._choose_reference_dataset()
        self._run = self._reference_dataset.run
        assert self._run is not None
        self._fingerprint = profile_fingerprint_for_run(self._run)
        # The draft this editor mutates. Seeded below from the active profile for
        # the fingerprint, or synthesized from the reference run's payload.
        self._draft = self._initial_draft()
        self._draft_name = self._draft.name
        # The draft resolved against the preview run: a full payload with the
        # historical ``run.grouping`` shape that the form controls seed from. It
        # merges the draft's shareable settings with the preview run's per-run
        # facts, so shareable reads (groups, alpha, binning) come from the draft
        # and per-run reads (t0, good bins) come from the run.
        seed = self._seed_source()

        self._groups = self._load_groups(seed)
        self._group_names: dict[int, str] = self._load_group_names(seed)
        self._included_groups: dict[int, bool] = self._load_included_groups(seed)
        self._projection_specs: list[dict] | None = self._load_projection_specs(seed)
        self._vector_axis_pairs: dict[str, tuple[int, int]] = {}
        self._vector_alpha_spins: dict[str, QDoubleSpinBox] = {}
        self._vector_forward_labels: dict[str, QLabel] = {}
        self._vector_backward_labels: dict[str, QLabel] = {}
        self._vector_estimate_buttons: dict[str, QPushButton] = {}
        self._estimate_all_btn: QPushButton | None = None
        # Last successful estimate per slot ("single" or axis name):
        # (alpha, alpha_error, reference_run). Used to attach provenance to
        # the payload only while the spin still holds the estimated value.
        self._alpha_estimate_state: dict[str, tuple[float, float | None, int]] = {}

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # \u2500\u2500 Top bar: profile selector + preview run + preset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.setMinimumContentsLength(20)
        self._rebuild_profile_combo()
        self._profile_combo.activated.connect(self._on_profile_combo_activated)
        profile_row.addWidget(self._profile_combo)

        rename_btn = QPushButton("Rename\u2026")
        rename_btn.setAutoDefault(False)
        rename_btn.setDefault(False)
        rename_btn.clicked.connect(self._on_rename_profile)
        profile_row.addWidget(rename_btn)

        # The reference-run combo is now a non-destructive preview-run selector:
        # it changes only the per-run facts shown, never the draft.
        self._reference_combo = QComboBox()
        for ds in self._fingerprint_datasets():
            self._reference_combo.addItem(ds.run_label, int(ds.run_number))
        self._set_combo_to_run(self._reference_combo, int(self._reference_dataset.run_number))
        self._reference_combo.currentIndexChanged.connect(self._on_reference_dataset_changed)
        profile_row.addWidget(QLabel("Preview run"))
        profile_row.addWidget(self._reference_combo)
        profile_row.addStretch()
        root.addLayout(profile_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumContentsLength(18)
        self._preset_combo.activated.connect(self._on_preset_combo_activated)
        preset_row.addWidget(self._preset_combo)
        self._preset_chip = QLabel("")
        self._preset_chip.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        preset_row.addWidget(self._preset_chip)
        preset_row.addStretch()
        root.addLayout(preset_row)
        # The preset combo is populated at the end of __init__, once the
        # detector-layout resolution state (_detector_layout_instrument_name)
        # exists in the form section below.

        # Non-blocking transverse-field nudge: shown when the reference run is
        # transverse-field but the grouping is still on a longitudinal preset
        # (which washes out the precession). Points the user at the Detector
        # Layout editor, where the recommended preset can be applied.
        self._tf_hint_label = QLabel()
        self._tf_hint_label.setWordWrap(True)
        self._tf_hint_label.setStyleSheet(f"color: {tokens.WARN};")
        self._tf_hint_label.setVisible(False)
        root.addWidget(self._tf_hint_label)

        # \u2500\u2500 Main split: left pane (datasets + groups) | right pane (form) \u2500
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)

        # Left pane
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        self._scope_panel = ScopePanel()
        self._scope_panel.setMaximumHeight(180)
        self._scope_panel.changed.connect(self._on_scope_changed)
        left_layout.addWidget(self._scope_panel)
        self._refresh_scope_panel()

        self._group_table = QTableWidget(0, 4)
        self._group_table.setHorizontalHeaderLabels(
            ["Group", "Include", "Name", "Detector Indices (1-based)"]
        )
        apply_param_table_style(self._group_table)
        self._group_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._group_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._group_table.setMinimumHeight(0)
        left_layout.addWidget(self._group_table)
        self._populate_group_table()

        detector_layout_btn = QPushButton("Detector Layout\u2026")
        detector_layout_btn.setAutoDefault(False)
        detector_layout_btn.setDefault(False)
        detector_layout_btn.setToolTip(
            "Open the interactive detector schematic editor to assign detectors to groups visually."
        )
        detector_layout_btn.clicked.connect(self._on_detector_layout)
        left_layout.addWidget(detector_layout_btn)
        left_layout.addStretch()
        splitter.addWidget(left_pane)

        # Right pane
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(4, 4, 0, 0)
        right_layout.setSpacing(0)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(12)
        self._forward_combo = QComboBox()
        self._backward_combo = QComboBox()
        self._forward_combo.setMinimumWidth(220)
        self._backward_combo.setMinimumWidth(220)
        self._forward_combo.setMinimumContentsLength(18)
        self._backward_combo.setMinimumContentsLength(18)
        self._refresh_group_combo_items()

        grouping = self._seed_source().grouping
        self._grouping_preset_name: str | None = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._detector_layout_instrument_name: str | None = (
            str(grouping.get("instrument")).strip() if grouping.get("instrument") else None
        )
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(grouping.get("forward_group", 1)),
            int(grouping.get("backward_group", 2)),
        )
        self._set_combo_to_group(self._forward_combo, forward_gid)
        self._set_combo_to_group(self._backward_combo, backward_gid)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setDecimals(6)
        self._alpha_spin.setRange(0.01, 1000.0)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))

        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = self._bin_index_base(grouping)
        default_t0_internal = self._default_t0_bin(grouping, max_bin)
        default_t_good = self._default_t_good_offset(grouping, default_t0_internal, max_bin)

        self._t0_spin = QSpinBox()
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._t0_spin.setValue(default_t0_internal + index_base)

        self._t_good_offset_spin = QSpinBox()
        self._t_good_offset_spin.setRange(0, max_bin)
        self._t_good_offset_spin.setValue(default_t_good)
        self._t0_spin.valueChanged.connect(self._on_t0_changed)
        self._on_t0_changed()

        self._last_good_spin = QSpinBox()
        self._last_good_spin.setRange(index_base, max_bin + index_base)
        default_first_good = min(max_bin, default_t0_internal + default_t_good)
        default_last_good = int(grouping.get("last_good_bin", max_bin))
        if default_last_good < default_first_good:
            default_last_good = default_first_good
        self._last_good_spin.setValue(default_last_good + index_base)

        self._bunch_spin = QSpinBox()
        self._bunch_spin.setRange(1, 10000)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        self._bunch_spin.setValue(requested_bunching)
        self._bunch_spin.setMaximumWidth(100)
        self._bunch_spin.setToolTip("Set any bunching factor >= 1.")

        self._binning_mode_combo = QComboBox()
        for label, key, tooltip in (
            ("Fixed", "fixed", "Merge a fixed number of raw bins (bunching factor)."),
            (
                "Variable",
                "variable",
                "Bin width grows exponentially from the initial width at t = 0 "
                "to the late width at 10 µs.",
            ),
            (
                "Constant error",
                "constant_error",
                "Bin width grows as exp(t/τ_µ) so every output bin holds "
                "roughly equal counts and the error per bin stays flat.",
            ),
        ):
            self._binning_mode_combo.addItem(label, key)
            self._binning_mode_combo.setItemData(
                self._binning_mode_combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )
        self._bin0_spin = QDoubleSpinBox()
        self._bin0_spin.setDecimals(4)
        self._bin0_spin.setRange(0.0001, 100.0)
        self._bin0_spin.setSuffix(" µs")
        self._bin10_spin = QDoubleSpinBox()
        self._bin10_spin.setDecimals(4)
        self._bin10_spin.setRange(0.0001, 100.0)
        self._bin10_spin.setSuffix(" µs")
        # Create the labels beside their spins so _on_binning_mode_changed can
        # show/hide both without a getattr/ordering dance (it runs during this
        # setup, before the layout row that positions them is built).
        self._bin0_label = QLabel("Initial bin")
        self._bin10_label = QLabel("Bin at 10 µs")
        initial_mode, initial_bin0, initial_bin10 = resolve_binning_mode(grouping)
        self._bin0_spin.setValue(initial_bin0)
        self._bin10_spin.setValue(initial_bin10)
        self._set_binning_mode(initial_mode)
        self._binning_mode_combo.currentIndexChanged.connect(self._on_binning_mode_changed)
        self._on_binning_mode_changed()

        self._exclude_edit = QLineEdit()
        self._exclude_edit.setPlaceholderText("e.g. 1,5,10-15 (1-based detector ids)")
        self._exclude_edit.setToolTip(
            "Detectors to exclude from every group sum (dead or hot detectors). "
            "Raw histograms are kept; exclusion applies at grouping time."
        )
        self._set_excluded_detectors_text(grouping)

        self._find_t0_btn = QPushButton("Find t0")
        self._find_t0_btn.setAutoDefault(False)
        self._find_t0_btn.setDefault(False)
        self._find_t0_btn.setToolTip(
            "Estimate t0 from the reference run (prompt peak at continuous "
            "sources, pulse-edge midpoint at pulsed sources) and fill the "
            "override — nothing is applied until you press Apply."
        )
        self._find_t0_btn.clicked.connect(self._on_find_t0)

        # Deadtime and background are configured in dedicated dialogs
        # (deadtime_dialog.py / background_dialog.py); the form only shows a
        # compact status row + "Configure…" button per cluster. State lives
        # here (not in inline widgets) so ``_current_grouping_payload`` and the
        # ``.grp`` load/save paths are unchanged.
        self._deadtime_mode = (
            self._default_deadtime_mode(grouping)
            if bool(grouping.get("deadtime_correction", False))
            else "off"
        )
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(grouping)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(grouping)
        self._deadtime_estimated_us: float | None = (
            float(grouping["deadtime_estimated_us"])
            if grouping.get("deadtime_estimated_us") is not None
            else None
        )
        self._deadtime_source_run: int | None = (
            int(grouping["deadtime_reference_run"])
            if grouping.get("deadtime_reference_run") is not None
            else None
        )

        self._deadtime_status_row = QWidget()
        deadtime_status_layout = QHBoxLayout(self._deadtime_status_row)
        deadtime_status_layout.setContentsMargins(0, 0, 0, 0)
        self._deadtime_status_label = QLabel("")
        self._deadtime_status_label.setWordWrap(True)
        deadtime_status_layout.addWidget(self._deadtime_status_label, stretch=1)
        self._deadtime_configure_btn = QPushButton("Configure…")
        self._deadtime_configure_btn.setAutoDefault(False)
        self._deadtime_configure_btn.setDefault(False)
        self._deadtime_configure_btn.clicked.connect(self._on_configure_deadtime)
        deadtime_status_layout.addWidget(self._deadtime_configure_btn)
        self._update_deadtime_status()

        payload = grouping.get("background_run")
        self._background_run_payload: dict[str, Any] | None = (
            dict(payload) if isinstance(payload, dict) else None
        )
        self._background_mode = "none"
        if bool(grouping.get("background_correction", False)):
            self._background_mode = resolve_background_mode(grouping)

        self._background_status_row = QWidget()
        background_status_layout = QHBoxLayout(self._background_status_row)
        background_status_layout.setContentsMargins(0, 0, 0, 0)
        self._background_status_label = QLabel("")
        self._background_status_label.setWordWrap(True)
        background_status_layout.addWidget(self._background_status_label, stretch=1)
        self._background_configure_btn = QPushButton("Configure…")
        self._background_configure_btn.setAutoDefault(False)
        self._background_configure_btn.setDefault(False)
        self._background_configure_btn.clicked.connect(self._on_configure_background)
        background_status_layout.addWidget(self._background_configure_btn)

        self._period_mode_label = QLabel("RG Mode")
        self._period_mode_group = QButtonGroup(self)
        self._period_mode_buttons: dict[str, QRadioButton] = {}
        self._period_mode_widget = QWidget()
        period_layout = QHBoxLayout(self._period_mode_widget)
        period_layout.setContentsMargins(0, 0, 0, 0)
        period_layout.setSpacing(10)

        period_specs = [
            ("Red", str(PeriodMode.RED), "#c00000"),
            ("Green", str(PeriodMode.GREEN), "#008000"),
            ("G minus R", str(PeriodMode.GREEN_MINUS_RED), "#0000c0"),
            ("G plus R", str(PeriodMode.GREEN_PLUS_RED), "#800080"),
        ]
        for idx, (label, mode_key, color) in enumerate(period_specs):
            btn = QRadioButton(label)
            btn.setStyleSheet(f"color: {color};")
            self._period_mode_group.addButton(btn, idx)
            self._period_mode_buttons[mode_key] = btn
            period_layout.addWidget(btn)
        period_layout.addStretch()
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))

        calibrate_btn = QPushButton("Calibrate…")
        calibrate_btn.setAutoDefault(False)
        calibrate_btn.setDefault(False)
        calibrate_btn.setToolTip(
            "Open the Alpha calibration dialog: pick a transverse-field "
            "calibration run and see α balance the asymmetry about zero."
        )
        calibrate_btn.clicked.connect(self._estimate_alpha)

        # The estimation *method* is now chosen inside the calibration dialog, so
        # the inline method combo is retired from the visible row. It is kept as a
        # hidden control because the current-method key still seeds the payload's
        # ``alpha_method`` provenance (and a calibration writes the chosen method
        # back into it).
        self._alpha_method_combo = QComboBox()
        for label, key, explanation in _ALPHA_METHOD_ITEMS:
            self._alpha_method_combo.addItem(label, key)
            self._alpha_method_combo.setItemData(
                self._alpha_method_combo.count() - 1,
                explanation,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._set_alpha_method(str(grouping.get("alpha_method", "diamagnetic")))
        self._alpha_method_combo.setVisible(False)

        # Provenance status for the single alpha: "calibrated" reads the estimate
        # summary, "manual" once the spin is hand-edited.
        self._alpha_provenance_label = QLabel("")
        self._alpha_provenance_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._alpha_provenance_label.setWordWrap(True)

        self._alpha_result_label = QLabel("")
        self._alpha_result_label.setWordWrap(True)

        self._forward_row_label = QLabel("Forward Group")
        self._backward_row_label = QLabel("Backward Group")
        self._alpha_row_label = QLabel("Alpha")

        form.addRow(self._forward_row_label, self._forward_combo)
        form.addRow(self._backward_row_label, self._backward_combo)

        self._single_alpha_widget = QWidget()
        alpha_row = QHBoxLayout(self._single_alpha_widget)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.addWidget(self._alpha_spin)
        alpha_row.addWidget(calibrate_btn)
        alpha_row.addWidget(self._alpha_provenance_label, stretch=1)
        form.addRow(self._alpha_row_label, self._single_alpha_widget)
        form.addRow("", self._alpha_result_label)
        # A hand-edit of the alpha spin clears calibration provenance → "manual".
        self._alpha_spin.valueChanged.connect(self._on_alpha_spin_edited)
        self._reseed_alpha_provenance_from_grouping(grouping)
        self._refresh_alpha_provenance_label()

        # Vector alpha widget: one row per declared projection. The rows are
        # built dynamically (see :meth:`_rebuild_vector_alpha_table`) because the
        # projection set varies by preset — canonical EMU axes (P_z/P_y/P_x),
        # GPS WEP's FB/UD, the MuSR/HiFi transverse pairs, etc. The grid below
        # is the empty container; rows are (re)created from the current
        # projection pairs whenever they change.
        self._vector_alpha_widget = QWidget()
        self._vector_alpha_layout = QGridLayout(self._vector_alpha_widget)
        self._vector_alpha_layout.setContentsMargins(0, 0, 0, 0)
        self._vector_alpha_layout.setHorizontalSpacing(12)
        self._vector_alpha_layout.setVerticalSpacing(8)

        form.addRow(self._vector_alpha_widget)

        t0_row_widget = QWidget()
        t0_row = QHBoxLayout(t0_row_widget)
        t0_row.setContentsMargins(0, 0, 0, 0)
        t0_row.addWidget(self._t0_spin)
        t0_row.addWidget(self._find_t0_btn)
        form.addRow("t0 Bin", t0_row_widget)
        form.addRow("t_good Offset", self._t_good_offset_spin)
        form.addRow("Last Good Bin", self._last_good_spin)
        binning_row_widget = QWidget()
        binning_row = QHBoxLayout(binning_row_widget)
        binning_row.setContentsMargins(0, 0, 0, 0)
        binning_row.addWidget(self._binning_mode_combo)
        binning_row.addWidget(self._bunch_spin)
        binning_row.addWidget(self._bin0_label)
        binning_row.addWidget(self._bin0_spin)
        binning_row.addWidget(self._bin10_label)
        binning_row.addWidget(self._bin10_spin)
        form.addRow("Binning", binning_row_widget)
        self._on_binning_mode_changed()
        form.addRow("Exclude Detectors", self._exclude_edit)
        form.addRow("Deadtime", self._deadtime_status_row)
        form.addRow("Background", self._background_status_row)
        self._update_background_status()
        self._map_periods_btn = QPushButton("Map periods…")
        self._map_periods_btn.setAutoDefault(False)
        self._map_periods_btn.setDefault(False)
        self._map_periods_btn.setToolTip(
            "Sum arbitrary subsets of this run's periods into the red and "
            "green sets (multi-period runs)."
        )
        self._map_periods_btn.clicked.connect(self._on_map_periods)
        self.period_mapping_request: dict[str, Any] | None = None
        period_row_widget = QWidget()
        period_row_box = QHBoxLayout(period_row_widget)
        period_row_box.setContentsMargins(0, 0, 0, 0)
        period_row_box.addWidget(self._period_mode_widget)
        period_row_box.addWidget(self._map_periods_btn)
        form.addRow(self._period_mode_label, period_row_widget)
        self._update_map_periods_visibility()

        right_layout.addLayout(form)
        right_layout.addStretch()

        # Live asymmetry preview of the preview run under the current draft.
        # Fixed-height so it never fights the form for space; it reduces off the
        # GUI thread (debounced) and redraws as the form is edited.
        self._preview_pane = GroupingPreviewPane()
        right_layout.addWidget(self._preview_pane)

        splitter.addWidget(right_pane)

        splitter.setSizes([330, 520])
        root.addWidget(splitter, stretch=1)

        self._update_vector_mode_controls(grouping)

        # ── Bottom bar: file I/O + action buttons ────────────────────────
        load_btn = QPushButton("Load .grp")
        load_btn.setAutoDefault(False)
        load_btn.setDefault(False)
        load_btn.clicked.connect(self._load_grp_file)
        save_btn = QPushButton("Save .grp")
        save_btn.setAutoDefault(False)
        save_btn.setDefault(False)
        save_btn.clicked.connect(self._save_grp_file)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.clicked.connect(self.reject)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setAutoDefault(False)
        self._apply_btn.setDefault(False)
        self._apply_btn.clicked.connect(self._on_apply)

        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(load_btn)
        bottom_bar.addWidget(save_btn)
        bottom_bar.addStretch()
        bottom_bar.addWidget(cancel_btn)
        bottom_bar.addWidget(self._apply_btn)
        root.addLayout(bottom_bar)

        self._update_period_mode_visibility()
        self._update_grouping_recommendation()
        self._rebuild_preset_combo()
        self._connect_dirty_tracking()
        self._refresh_preset_chip(self._current_grouping_payload())
        self._update_apply_enabled()
        self._connect_preview_refresh()
        self._refresh_preview()

    def _choose_reference_dataset(self) -> MuonDataset:
        """Return preferred reference dataset for initial grouping values."""
        if self._selected_run_number is not None:
            for ds in self._datasets:
                if int(ds.run_number) == int(self._selected_run_number):
                    return ds
        return self._datasets[0]

    # ------------------------------------------------------------------
    # Draft profile plumbing (M2)
    # ------------------------------------------------------------------

    def _fingerprint_datasets(self) -> list[MuonDataset]:
        """Datasets whose run matches the editor's current fingerprint."""
        if self._fingerprint is None:
            return list(self._datasets)
        out: list[MuonDataset] = []
        for ds in self._datasets:
            if ds.run is None:
                continue
            if profile_fingerprint_for_run(ds.run).matches(self._fingerprint):
                out.append(ds)
        return out

    def _profiles_for_fingerprint(self) -> list[GroupingProfile]:
        """Project profiles matching the editor's current fingerprint."""
        if self._fingerprint is None:
            return []
        return [p for p in self._project_profiles if p.fingerprint.matches(self._fingerprint)]

    def _default_draft_name(self) -> str:
        """Synthesized draft name for a fingerprint with no saved profile.

        Requires positive instrument identification: a generic facility token
        (``"PSI"``) or an empty instrument never names a profile after an
        instrument — the neutral ``"Default (<N> detectors)"`` is used instead,
        so an unresolved file cannot masquerade as a specific spectrometer.
        """
        instrument = ""
        if self._fingerprint is not None and self._fingerprint.instrument:
            instrument = instrument_display_name(self._fingerprint.instrument)
        if not instrument:
            grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
            instrument = str(grouping.get("instrument", "") or "")
        if not instrument or instrument.strip().upper() == "PSI":
            n = len(self._run.histograms) if self._run is not None and self._run.histograms else 0
            if n:
                return f"Default ({n} detectors)"
            return "Default"
        return f"Default ({instrument})"

    def _initial_draft(self) -> GroupingProfile:
        """Return the draft profile the editor opens on.

        The active profile for the current fingerprint when the project has one;
        otherwise a fresh draft synthesized from the reference run's own payload
        (named ``"Default (<instrument>)"``). A copy is always returned so editing
        the draft never mutates a stored project profile.
        """
        assert self._fingerprint is not None
        for profile in self._profiles_for_fingerprint():
            if profile.active:
                return GroupingProfile.from_dict(profile.to_dict())
        payload = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        return profile_from_form_payload(
            payload,
            name=self._default_draft_name(),
            fingerprint=self._fingerprint,
            active=True,
        )

    def _seed_source(self):
        """A run-like shim whose ``grouping`` is the draft resolved for preview.

        The form's ``_load_*`` helpers read ``.grouping`` for the shareable
        settings; per-run facts (t0, good-bin window, file deadtime, period
        tables) are merged in from the preview run by
        :func:`resolve_effective_grouping`, so the shim carries the preview run's
        histograms/metadata too.
        """
        from asymmetry.gui.windows.grouping.profile_bridge import (
            payload_from_profile_for_preview,
        )

        payload = payload_from_profile_for_preview(self._draft, self._run)

        class _SeedSource:
            grouping = payload
            histograms = self._run.histograms
            metadata = self._run.metadata
            source_file = getattr(self._run, "source_file", None)

        return _SeedSource()

    def _mark_dirty(self) -> None:
        """Record that the draft diverges from its opened state."""
        if not self._suppress_dirty:
            self._draft_dirty = True

    def _sync_draft_from_form(self) -> None:
        """Lift the current form payload back into the draft profile.

        Called before the draft is read (Apply, preset-chip refresh, profile
        switch). Also refreshes the preset chip / clears a stale preset marker
        when the groups have drifted from the named preset.
        """
        payload = self._current_grouping_payload()
        self._refresh_preset_chip(payload)
        # Re-read the payload after a possible drift-clear so the draft does not
        # carry a stale ``grouping_preset``.
        payload = self._current_grouping_payload()
        assert self._fingerprint is not None
        self._draft = profile_from_form_payload(
            payload,
            name=self._draft_name,
            fingerprint=self._fingerprint,
            active=True,
        )

    # -- profile selector -------------------------------------------------

    def _rebuild_profile_combo(self) -> None:
        """(Re)populate the profile selector for the current fingerprint."""
        combo = self._profile_combo
        combo.blockSignals(True)
        combo.clear()
        names = [p.name for p in self._profiles_for_fingerprint()]
        if self._draft_name and self._draft_name not in names:
            names.append(self._draft_name)
        for name in names:
            combo.addItem(name, name)
        combo.addItem("New…", "__new__")
        combo.addItem("Duplicate…", "__duplicate__")
        idx = combo.findData(self._draft_name)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _on_profile_combo_activated(self, index: int) -> None:
        """Switch the active draft, or create a new / duplicated profile."""
        data = self._profile_combo.itemData(index)
        if data == "__new__":
            self._create_new_profile(duplicate=False)
            return
        if data == "__duplicate__":
            self._create_new_profile(duplicate=True)
            return
        name = str(data)
        if name == self._draft_name:
            return
        if not self._confirm_discard_before_switch():
            self._select_profile_in_combo(self._draft_name)
            return
        for profile in self._profiles_for_fingerprint():
            if profile.name == name:
                self._draft = GroupingProfile.from_dict(profile.to_dict())
                self._draft_name = name
                self._draft_dirty = False
                self._reseed_form_from_draft()
                return

    def _create_new_profile(self, *, duplicate: bool) -> None:
        """Prompt for a name and start a fresh (or duplicated) draft."""
        from PySide6.QtWidgets import QInputDialog

        base = f"{self._draft_name} copy" if duplicate else self._default_draft_name()
        name, accepted = QInputDialog.getText(
            self,
            "Duplicate Profile" if duplicate else "New Profile",
            "Profile name:",
            text=self._unique_profile_name(base),
        )
        name = str(name).strip()
        if not accepted or not name:
            self._select_profile_in_combo(self._draft_name)
            return
        if not self._confirm_discard_before_switch():
            self._select_profile_in_combo(self._draft_name)
            return
        assert self._fingerprint is not None
        if duplicate:
            self._sync_draft_from_form()
            self._draft = GroupingProfile.from_dict(self._draft.to_dict())
        else:
            payload = self._run.grouping if isinstance(self._run.grouping, dict) else {}
            self._draft = profile_from_form_payload(
                payload, name=name, fingerprint=self._fingerprint, active=True
            )
        self._draft.name = name
        self._draft_name = name
        self._draft_dirty = True
        self._reseed_form_from_draft()

    def _on_rename_profile(self) -> None:
        """Rename the current draft profile."""
        from PySide6.QtWidgets import QInputDialog

        name, accepted = QInputDialog.getText(
            self, "Rename Profile", "Profile name:", text=self._draft_name
        )
        name = str(name).strip()
        if not accepted or not name or name == self._draft_name:
            return
        self._draft_name = name
        self._draft.name = name
        self._draft_dirty = True
        self._rebuild_profile_combo()

    def _unique_profile_name(self, base: str) -> str:
        """Return *base* suffixed to avoid clashing with existing profile names."""
        existing = {p.name for p in self._profiles_for_fingerprint()}
        if base not in existing:
            return base
        index = 2
        while f"{base} ({index})" in existing:
            index += 1
        return f"{base} ({index})"

    def _select_profile_in_combo(self, name: str) -> None:
        idx = self._profile_combo.findData(name)
        if idx >= 0:
            self._profile_combo.blockSignals(True)
            self._profile_combo.setCurrentIndex(idx)
            self._profile_combo.blockSignals(False)

    def _confirm_discard_before_switch(self) -> bool:
        """Ask to discard unsaved draft edits before switching profiles."""
        if not self._draft_dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Discard changes",
            f"Discard unsaved changes to profile '{self._draft_name}'?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _reseed_form_from_draft(self) -> None:
        """Rebuild every form control from the current draft (+ preview run)."""
        self._suppress_dirty = True
        try:
            self._reload_controls_from_seed()
        finally:
            self._suppress_dirty = False
        self._rebuild_profile_combo()
        self._rebuild_preset_combo()
        self._refresh_scope_panel()

    # -- preset dropdown + chip ------------------------------------------

    def _current_instrument_layout(self):
        """Instrument layout matching the preview run, for the preset dropdown."""
        n_histo = len(self._run.histograms) if self._run and self._run.histograms else 0
        metadata = (
            dict(self._run.metadata) if self._run and isinstance(self._run.metadata, dict) else {}
        )
        return self._resolve_detector_layout(n_histo, metadata)

    def _rebuild_preset_combo(self) -> None:
        """Populate the preset dropdown from the preview run's instrument."""
        combo = self._preset_combo
        combo.blockSignals(True)
        combo.clear()
        try:
            layout = self._current_instrument_layout()
            for name in layout.presets:
                combo.addItem(name, name)
        except (KeyError, AttributeError):
            pass
        combo.blockSignals(False)
        # The chip needs the form controls; skip until they exist (they are built
        # after the top bar during __init__).
        if hasattr(self, "_forward_combo"):
            self._refresh_preset_chip(self._current_grouping_payload())

    def _on_preset_combo_activated(self, index: int) -> None:
        """Apply the selected instrument preset to the draft immediately."""
        preset_name = self._preset_combo.itemData(index)
        if not preset_name:
            return
        try:
            layout = self._current_instrument_layout()
        except (KeyError, AttributeError):
            return
        payload = preset_payload(layout, str(preset_name))
        if payload is None:
            return
        self._apply_preset_payload_to_form(payload)
        self._mark_dirty()

    def _apply_preset_payload_to_form(self, payload: dict[str, Any]) -> None:
        """Adopt a preset's groups/names/slots/projections into the form state."""
        self._groups = {
            int(gid): sorted(max(0, int(d) - 1) for d in dets)
            for gid, dets in payload.get("groups", {}).items()
        }
        self._group_names = {int(k): str(v) for k, v in payload.get("group_names", {}).items()}
        self._included_groups = {int(gid): True for gid in self._groups}
        self._grouping_preset_name = payload.get("grouping_preset")
        if payload.get("instrument"):
            self._detector_layout_instrument_name = str(payload["instrument"])
        projections = payload.get("projections")
        self._projection_specs = (
            [dict(p) for p in projections if isinstance(p, dict)] if projections else None
        )
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(payload.get("forward_group", 1)),
            int(payload.get("backward_group", 2)),
        )
        self._refresh_group_combo_items(forward_gid=forward_gid, backward_gid=backward_gid)
        self._populate_group_table()
        self._update_vector_mode_controls()
        self._update_grouping_recommendation()
        self._refresh_preset_chip(self._current_grouping_payload())

    def _refresh_preset_chip(self, payload: dict[str, Any]) -> None:
        """Show ``Preset: <name>`` / ``Custom (edited from <name>)``.

        Clears a stale ``grouping_preset`` marker when the groups have drifted
        from the named preset, so a drifted draft never stores it.
        """
        preset_name = self._grouping_preset_name
        if not preset_name:
            self._preset_chip.setText("Custom")
            return
        try:
            layout = self._current_instrument_layout()
        except (KeyError, AttributeError):
            self._preset_chip.setText(f"Preset: {preset_name}")
            return
        if payload_matches_preset(payload, layout, preset_name):
            self._preset_chip.setText(f"Preset: {preset_name}")
        else:
            self._preset_chip.setText(f"Custom (edited from {preset_name})")
            self._grouping_preset_name = None

    # -- scope panel ------------------------------------------------------

    def _refresh_scope_panel(self) -> None:
        """Repopulate the scope panel for the current fingerprint + draft."""
        runs: list[tuple[int, str, bool]] = []
        for ds in self._fingerprint_datasets():
            rn = int(ds.run_number)
            runs.append((rn, ds.run_label, rn in self._overridden_run_numbers))
        self._scope_panel.set_runs(runs, profile_name=self._draft_name)
        if hasattr(self, "_apply_btn"):
            self._update_apply_enabled()

    def _on_scope_changed(self) -> None:
        """React to a release/reattach in the scope panel."""
        self._mark_dirty()
        self._update_apply_enabled()

    def _load_group_names(self, run) -> dict[int, str]:
        """Load group names from run metadata, returning an empty dict if absent."""
        grouping = run.grouping or {}
        raw = grouping.get("group_names")
        if not isinstance(raw, dict):
            return {}
        result: dict[int, str] = {}
        for key, val in raw.items():
            try:
                gid = int(key)
                result[gid] = str(val)
            except (TypeError, ValueError):
                continue
        return result

    def _load_projection_specs(self, run) -> list[dict] | None:
        """Load declared projection specs from run grouping, if present."""
        grouping = getattr(run, "grouping", None) or {}
        specs = grouping.get("projections")
        if isinstance(specs, list) and specs:
            return [dict(s) for s in specs if isinstance(s, dict)]
        return None

    def _load_included_groups(self, run) -> dict[int, bool]:
        """Load per-group include flags from run metadata, defaulting missing rows to True."""
        grouping = run.grouping or {}
        raw = grouping.get("included_groups")
        result: dict[int, bool] = {}
        if isinstance(raw, dict):
            for key, val in raw.items():
                try:
                    gid = int(key)
                except (TypeError, ValueError):
                    continue
                result[gid] = bool(val)
        for gid in self._groups:
            result.setdefault(int(gid), True)
        return result

    def _reference_field_direction(self) -> str | None:
        """Applied-field geometry of the reference run, or None when unknown."""
        metadata = getattr(self._run, "metadata", None)
        if isinstance(metadata, dict):
            value = str(metadata.get("field_direction", "")).strip()
            return value or None
        return None

    def _update_grouping_recommendation(self) -> None:
        """Show or hide the transverse-field grouping nudge for the reference run.

        Fires when the run is transverse-field but the grouping is still on a
        non-recommended (typically longitudinal) preset, per
        :func:`asymmetry.core.instrument.recommend_grouping_preset`.  The hint is
        purely advisory — it points at the Detector Layout editor where the user
        applies the preset; nothing is changed automatically.
        """
        direction = self._reference_field_direction()
        n_histo = len(self._run.histograms) if self._run and self._run.histograms else 0
        metadata = (
            dict(self._run.metadata) if self._run and isinstance(self._run.metadata, dict) else {}
        )
        layout = self._resolve_detector_layout(n_histo, metadata)
        recommended = recommend_grouping_preset(layout, direction)
        if recommended is None or recommended == self._grouping_preset_name:
            self._tf_hint_label.setVisible(False)
            return
        self._tf_hint_label.setText(
            f"Transverse-field run: the current grouping washes out the precession. "
            f"Open Detector Layout… and apply ‘{recommended}’."
        )
        self._tf_hint_label.setVisible(True)

    def _reference_is_psi(self) -> bool:
        """Return True when the current reference run uses PSI detector conventions."""
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        if self._run is not None:
            metadata.update(getattr(self._run, "metadata", {}) or {})
            grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
            if grouping.get("psi_format"):
                return True
        facility = str(metadata.get("facility", "")).strip().lower()
        return facility == "psi" or bool(metadata.get("psi_format"))

    @staticmethod
    def _beam_direction_label(label: object) -> str | None:
        """Return the beam-direction sense of a group name, or ``None``.

        Recognises the spelled-out forward/backward names (``"Forward"``,
        ``"Fwd"``, ``"Backward"``, ``"Bwd"``) plus, as defence in depth for
        user-defined groups, the single-letter PSI names ``"F"``/``"B"`` and the
        compound spin-rotator names (``"F+U"``, ``"B+D"``, …) whose *first*
        beam-axis letter sets the sense.

        This only classifies groups that are named by **beam** direction, so it
        drives the PSI beam→analysis swap in
        :meth:`_analysis_pair_for_reference`.  Because the instrument presets now
        declare their analysis slots directly (the Backward-named group sits in
        the analysis-forward slot), the swap condition — forward slot holds a
        *forward*-named group **and** backward slot holds a *backward*-named group
        — is false for a fixed preset, so applying a preset does not get
        re-swapped (see the exactly-once regression test).
        """
        token = re.sub(r"[^a-z0-9]+", "", str(label).lower())
        if not token:
            return None
        # Compound spin-rotator names (e.g. "F+U", "B+D") compact to "fu"/"bd";
        # the leading beam-axis letter sets the sense. A bare "f"/"b" is the PSI
        # single-letter name.
        if token.startswith(("forw", "fwd")) or "forward" in token or token[0] == "f":
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token or token[0] == "b":
            return "backward"
        return None

    def _analysis_pair_for_reference(self, forward_gid: int, backward_gid: int) -> tuple[int, int]:
        """Map PSI beam-forward/backward selections to analysis spin-forward/backward."""
        if not self._reference_is_psi():
            return int(forward_gid), int(backward_gid)
        forward_name = self._group_names.get(int(forward_gid), "")
        backward_name = self._group_names.get(int(backward_gid), "")
        if (
            self._beam_direction_label(forward_name) == "forward"
            and self._beam_direction_label(backward_name) == "backward"
        ):
            return int(backward_gid), int(forward_gid)
        return int(forward_gid), int(backward_gid)

    def _load_groups(self, run) -> dict[int, list[int]]:
        """Load detector groups from run metadata or default half/half groups."""
        grouping = run.grouping or {}
        groups_raw = grouping.get("groups")

        groups: dict[int, list[int]] = {}
        if isinstance(groups_raw, dict):
            for key, values in groups_raw.items():
                try:
                    gid = int(key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(values, list):
                    continue
                idxs = []
                for val in values:
                    try:
                        # Persisted groups are 1-based in project/run metadata.
                        idxs.append(max(0, int(val) - 1))
                    except (TypeError, ValueError):
                        continue
                if idxs:
                    groups[gid] = sorted(set(idxs))

        if len(groups) >= 2:
            return groups

        n = len(run.histograms)
        if n <= 0:
            try:
                n = int(run.metadata.get("histograms_count", 0))
            except (TypeError, ValueError):
                n = 0
        if n <= 0:
            n = 2
        split = max(1, n // 2)
        return {1: list(range(0, split)), 2: list(range(split, n)) or list(range(0, n))}

    def _max_bin_index_for_reference_dataset(self) -> int:
        """Return max index used by good-bin controls for the reference dataset."""
        if self._run is not None and self._run.histograms:
            return max(0, self._run.histograms[0].n_bins - 1)
        return max(0, int(self._reference_dataset.n_points) - 1)

    def _bin_index_base(self, grouping: dict[str, Any] | None = None) -> int:
        """Return displayed index base (0- or 1-based) for bin controls."""
        if grouping is None:
            grouping = (
                self._run.grouping
                if self._run is not None and isinstance(self._run.grouping, dict)
                else {}
            )
        raw_base = grouping.get("bin_index_base", 0)
        try:
            return 1 if int(raw_base) == 1 else 0
        except (TypeError, ValueError):
            return 0

    def _default_t0_bin(self, grouping: dict[str, Any], max_bin: int) -> int:
        """Return initial internal t0 bin from grouping metadata or histograms."""
        raw_t0 = grouping.get("t0_bin")
        if raw_t0 is None and self._run is not None and self._run.histograms:
            raw_t0 = self._run.histograms[0].t0_bin
        try:
            return max(0, min(max_bin, int(raw_t0)))
        except (TypeError, ValueError):
            return 0

    def _default_t_good_offset(self, grouping: dict[str, Any], t0_bin: int, max_bin: int) -> int:
        """Return initial t_good offset from grouping metadata."""
        raw_offset = grouping.get("t_good_offset")
        if raw_offset is None:
            raw_first = grouping.get("first_good_bin", t0_bin)
            try:
                raw_offset = int(raw_first) - int(t0_bin)
            except (TypeError, ValueError):
                raw_offset = 0
        try:
            return max(0, min(max_bin, int(raw_offset)))
        except (TypeError, ValueError):
            return 0

    def _on_t0_changed(self) -> None:
        """Constrain t_good offset so t0 + offset remains in the histogram range."""
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_bin = max(0, min(max_bin, int(self._t0_spin.value()) - base))
        max_offset = max(0, max_bin - t0_bin)
        self._t_good_offset_spin.setMaximum(max_offset)
        if int(self._t_good_offset_spin.value()) > max_offset:
            self._t_good_offset_spin.setValue(max_offset)

    def _resolve_good_bin_limits_from_controls(self) -> tuple[int, int, int, int]:
        """Return validated ``(t0_bin, t_good_offset, first_good_bin, last_good_bin)``."""
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_bin = max(0, min(max_bin, int(self._t0_spin.value()) - base))
        t_good_offset = max(0, int(self._t_good_offset_spin.value()))
        first_good_bin = t0_bin + t_good_offset
        last_good_bin = max(0, min(max_bin, int(self._last_good_spin.value()) - base))
        return t0_bin, t_good_offset, first_good_bin, last_good_bin

    def _set_combo_to_run(self, combo: QComboBox, run_number: int) -> None:
        """Set reference-run combo to the provided run number if it exists."""
        idx = combo.findData(run_number)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_reference_dataset_changed(self) -> None:
        """Switch the *preview* run without discarding draft edits.

        Changing the preview run only refreshes the per-run facts (t0, good-bin
        window, file deadtime) shown; the draft's shareable settings are first
        lifted out of the form, then re-seeded against the new preview run so an
        in-progress edit survives the switch.
        """
        run_number = int(self._reference_combo.currentData())
        dataset = next(
            (ds for ds in self._fingerprint_datasets() if int(ds.run_number) == run_number),
            None,
        )
        if dataset is None or dataset.run is None:
            return

        # Preserve in-progress draft edits across the preview switch.
        self._sync_draft_from_form()
        self._reference_dataset = dataset
        self._run = dataset.run
        self._suppress_dirty = True
        try:
            self._reload_controls_from_seed()
        finally:
            self._suppress_dirty = False

    def _reload_controls_from_seed(self) -> None:
        """Re-seed every form control from the draft resolved for the preview run."""
        grouping = self._seed_source().grouping
        self._groups = self._load_groups(self._seed_source())
        self._populate_group_table()

        self._grouping_preset_name = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._group_names = self._load_group_names(self._seed_source())
        self._included_groups = self._load_included_groups(self._seed_source())
        self._projection_specs = self._load_projection_specs(self._seed_source())
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(grouping.get("forward_group", 1)),
            int(grouping.get("backward_group", 2)),
        )
        self._refresh_group_combo_items(forward_gid=forward_gid, backward_gid=backward_gid)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))
        self._set_alpha_method(str(grouping.get("alpha_method", "diamagnetic")))
        self._alpha_estimate_state.clear()
        self._reseed_alpha_provenance_from_grouping(grouping)
        self._alpha_result_label.setText("")
        self._refresh_alpha_provenance_label()
        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = self._bin_index_base(grouping)
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._t_good_offset_spin.setRange(0, max_bin)
        self._last_good_spin.setRange(index_base, max_bin + index_base)
        default_t0_internal = self._default_t0_bin(grouping, max_bin)
        default_t_good = self._default_t_good_offset(grouping, default_t0_internal, max_bin)
        self._t0_spin.setValue(default_t0_internal + index_base)
        self._t_good_offset_spin.setValue(default_t_good)
        self._on_t0_changed()
        default_first_good = min(max_bin, default_t0_internal + default_t_good)
        default_last_good = int(grouping.get("last_good_bin", max_bin))
        if default_last_good < default_first_good:
            default_last_good = default_first_good
        self._last_good_spin.setValue(default_last_good + index_base)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        self._bunch_spin.setValue(requested_bunching)
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(grouping)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(grouping)
        self._deadtime_mode = (
            self._default_deadtime_mode(grouping)
            if bool(grouping.get("deadtime_correction", False))
            else "off"
        )
        self._deadtime_estimated_us = (
            float(grouping["deadtime_estimated_us"])
            if grouping.get("deadtime_estimated_us") is not None
            else None
        )
        self._deadtime_source_run = (
            int(grouping["deadtime_reference_run"])
            if grouping.get("deadtime_reference_run") is not None
            else None
        )
        self._update_deadtime_status()
        payload = grouping.get("background_run")
        self._background_run_payload = dict(payload) if isinstance(payload, dict) else None
        self._background_mode = (
            resolve_background_mode(grouping)
            if bool(grouping.get("background_correction", False))
            else "none"
        )
        self._update_background_status()
        mode, bin0_us, bin10_us = resolve_binning_mode(grouping)
        self._bin0_spin.setValue(bin0_us)
        self._bin10_spin.setValue(bin10_us)
        self._set_binning_mode(mode)
        self._on_binning_mode_changed()
        self._set_excluded_detectors_text(grouping)
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))
        self._update_vector_mode_controls(grouping)
        self._update_period_mode_visibility()
        self._update_map_periods_visibility()
        self._update_grouping_recommendation()
        # The preset dropdown follows the preview run's instrument; the chip
        # follows the (possibly drifted) draft.
        if hasattr(self, "_preset_combo"):
            self._rebuild_preset_combo()
        # Preview-run change / profile switch: recompute against the new state.
        if hasattr(self, "_preview_pane"):
            self._refresh_preview()

    def _reference_has_file_deadtime(self, grouping: dict[str, Any]) -> bool:
        """Return whether the reference run provides file deadtime values."""
        values = grouping.get("dead_time_us") if isinstance(grouping, dict) else None
        if not isinstance(values, list):
            return False
        n_histograms = len(self._run.histograms) if self._run is not None else 0
        required = max(1, n_histograms)
        if len(values) < required:
            return False
        for value in values:
            try:
                if float(value) != 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _default_deadtime_mode(self, grouping: dict[str, Any]) -> str:
        """Return preferred deadtime mode for the current grouping state."""
        mode = (
            str(grouping.get("deadtime_mode", grouping.get("deadtime_method", ""))).strip().lower()
        )
        if mode in {"file", "manual", "estimate"}:
            return mode
        if mode == "load":
            return "manual"
        return "file"

    def _set_deadtime_mode(self, mode: str) -> None:
        """Set the active deadtime mode state (drives the status row)."""
        mode = str(mode).strip().lower()
        self._deadtime_mode = mode if mode in {"off", "file", "manual", "estimate"} else "file"

    def _current_deadtime_mode(self) -> str:
        """Return the active deadtime mode, or ``off`` when disabled."""
        return self._deadtime_mode

    def _initial_manual_deadtime_method(self, grouping: dict[str, Any]) -> str:
        """Return provenance for explicit manual deadtime values."""
        method = str(grouping.get("deadtime_method", "manual")).strip().lower()
        if method == "calibrate":
            return "calibrate"
        return "manual"

    def _histogram_deadtime_count(self) -> int:
        """Return the detector count for the current reference run."""
        if self._run is None:
            return 0
        return len(self._run.histograms)

    def _normalize_deadtime_values(self, values: object) -> list[float]:
        """Return finite deadtime values matching the reference histogram count."""
        if not isinstance(values, list):
            return []
        expected = self._histogram_deadtime_count()
        if expected <= 0 or len(values) != expected:
            return []
        normalized: list[float] = []
        for value in values:
            try:
                tau_us = float(value)
            except (TypeError, ValueError):
                return []
            if not np.isfinite(tau_us):
                return []
            normalized.append(max(0.0, tau_us))
        return normalized

    def _reference_file_deadtime_values(
        self, grouping: dict[str, Any] | None = None
    ) -> list[float]:
        """Return file deadtime values for the current reference run, if any."""
        if grouping is None:
            grouping = self._run.grouping if self._run is not None else {}
        if not self._reference_has_file_deadtime(grouping or {}):
            return []
        return self._normalize_deadtime_values((grouping or {}).get("dead_time_us"))

    def _initial_manual_deadtime_values(self, grouping: dict[str, Any]) -> list[float]:
        """Return initial explicit deadtime values for manual editing."""
        values = self._normalize_deadtime_values(grouping.get("dead_time_us"))
        mode = (
            str(grouping.get("deadtime_mode", grouping.get("deadtime_method", ""))).strip().lower()
        )
        method = str(grouping.get("deadtime_method", "")).strip().lower()
        if values and (mode in {"manual", "load"} or method in {"manual", "calibrate", "load"}):
            return values
        file_values = self._reference_file_deadtime_values(grouping)
        if file_values:
            return file_values
        n_histograms = self._histogram_deadtime_count()
        default_tau = float(grouping.get("deadtime_manual_us", 0.01) or 0.01)
        return [default_tau] * n_histograms

    def _estimate_deadtime_us_from_reference(self) -> float | None:
        """Estimate a uniform deadtime value from the current reference run."""
        if self._run is None or not self._run.histograms:
            return None
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        _t0_bin, t_good_offset, _first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        try:
            good_frames = float(grouping.get("good_frames", 1.0))
        except (TypeError, ValueError):
            good_frames = 1.0
        return estimate_deadtime_from_histograms(
            self._run.histograms,
            t_good_offset=t_good_offset,
            last_good_bin=last_good_bin,
            num_good_frames=good_frames,
        )

    def _resolve_deadtime_payload(self, *, show_warnings: bool = False) -> dict[str, Any] | None:
        """Return resolved deadtime payload for the current dialog state."""
        mode = self._current_deadtime_mode()
        if mode == "off":
            return {"deadtime_correction": False, "deadtime_mode": "off"}

        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        n_histograms = len(self._run.histograms) if self._run is not None else 0
        payload: dict[str, Any] = {"deadtime_correction": True, "deadtime_mode": mode}

        if mode == "file":
            if not self._reference_has_file_deadtime(grouping):
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Deadtime Unavailable",
                        "The reference run does not provide file deadtime values.",
                    )
                return None
            payload["deadtime_method"] = "file"
            return payload

        if mode == "manual":
            values_us = list(self._deadtime_manual_values_us)
            if not values_us or any(value <= 0.0 for value in values_us):
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Invalid Deadtime",
                        "Manual deadtime values must be greater than zero for every detector.",
                    )
                return None
            payload.update(
                {
                    "deadtime_method": self._deadtime_manual_method,
                    "deadtime_manual_us": float(values_us[0]),
                    "dead_time_us": values_us,
                }
            )
            if self._deadtime_manual_method == "calibrate":
                payload["deadtime_reference_run"] = int(self._reference_dataset.run_number)
            return payload

        if mode == "estimate":
            tau_us = self._estimate_deadtime_us_from_reference()
            if tau_us is None or tau_us <= 0.0:
                if show_warnings:
                    QMessageBox.warning(
                        self,
                        "Deadtime Estimate Failed",
                        "The reference run did not provide enough valid early-time counts to estimate deadtime.",
                    )
                return None
            payload.update(
                {
                    "deadtime_method": "estimate",
                    "deadtime_estimated_us": tau_us,
                    "deadtime_reference_run": int(self._reference_dataset.run_number),
                    "dead_time_us": [tau_us] * n_histograms,
                }
            )
            return payload

        return None

    def _calibrate_deadtime_from_reference(self) -> None:
        """Calibrate a per-detector deadtime table from the reference run."""
        if self._run is None or not self._run.histograms:
            return
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        _t0_bin, t_good_offset, _first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        try:
            good_frames = float(grouping.get("good_frames", 1.0))
        except (TypeError, ValueError):
            good_frames = 1.0
        values = calibrate_deadtime_from_histograms(
            self._run.histograms,
            t_good_offset=t_good_offset,
            last_good_bin=last_good_bin,
            num_good_frames=good_frames,
        )
        if not values:
            QMessageBox.warning(
                self,
                "Deadtime Calibration Failed",
                "The reference run did not provide enough valid early-time counts to calibrate per-detector deadtime values.",
            )
            return
        self._deadtime_manual_values_us = list(values)
        self._deadtime_manual_method = "calibrate"
        self._set_deadtime_mode("manual")
        self._update_deadtime_status()

    def _update_deadtime_status(self) -> None:
        """Refresh the compact deadtime status row from the current state."""
        self._deadtime_status_label.setText(
            deadtime_status_text(self._deadtime_policy_from_state())
        )

    def _deadtime_policy_from_state(self) -> DeadtimePolicy:
        """Build the :class:`DeadtimePolicy` the status row / dialog reflect."""
        mode = self._deadtime_mode
        if mode == "off":
            return DeadtimePolicy(mode="off")
        if mode == "file":
            return DeadtimePolicy(mode="from_file")
        if mode == "manual":
            return DeadtimePolicy(
                mode="manual",
                values=list(self._deadtime_manual_values_us),
                method=self._deadtime_manual_method,
                source_run=self._deadtime_source_run,
            )
        return DeadtimePolicy(
            mode="estimate",
            estimated_us=self._deadtime_estimated_us,
            source_run=self._deadtime_source_run,
        )

    def _on_configure_deadtime(self) -> None:
        """Open the deadtime dialog, seeded from the current state."""
        n_detectors = len(self._run.histograms) if self._run is not None else 0
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        file_values = self._reference_file_deadtime_values(grouping)

        source_runs: list[DeadtimeSourceRun] = []
        for ds in self._fingerprint_datasets():
            if ds.run is None or not ds.run.histograms:
                continue
            run_grouping = ds.run.grouping if isinstance(ds.run.grouping, dict) else {}
            try:
                good_frames = float(run_grouping.get("good_frames", 1.0))
            except (TypeError, ValueError):
                good_frames = 1.0
            source_runs.append(
                DeadtimeSourceRun(
                    run_number=int(ds.run_number),
                    label=f"{ds.run_label} (run {ds.run_number})",
                    histograms=ds.run.histograms,
                    good_frames=good_frames,
                )
            )

        peak_rates: list[float] = []
        bin_width_us = 0.0
        good_frames = 1.0
        if self._run is not None and self._run.histograms:
            bin_width_us = float(self._run.histograms[0].bin_width)
            try:
                good_frames = float(grouping.get("good_frames", 1.0))
            except (TypeError, ValueError):
                good_frames = 1.0
            for hist in self._run.histograms:
                counts = np.asarray(hist.counts, dtype=np.float64)
                peak = float(np.max(counts)) if counts.size else 0.0
                peak_rates.append(peak / bin_width_us if bin_width_us > 0.0 else 0.0)

        dlg = DeadtimeDialog(
            n_detectors=n_detectors,
            mode=self._deadtime_mode,
            file_values_us=file_values,
            manual_values_us=list(self._deadtime_manual_values_us),
            manual_method=self._deadtime_manual_method,
            estimated_us=self._deadtime_estimated_us,
            source_run=self._deadtime_source_run or int(self._reference_dataset.run_number),
            source_runs=source_runs,
            reference_run_number=int(self._reference_dataset.run_number),
            peak_rates_per_us=peak_rates,
            bin_width_us=bin_width_us,
            good_frames=good_frames,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        policy = dlg.get_policy()
        self._deadtime_mode = policy.mode if policy.mode != "from_file" else "file"
        if policy.mode == "manual":
            self._deadtime_manual_values_us = list(policy.values)
            self._deadtime_manual_method = policy.method or "manual"
            self._deadtime_source_run = policy.source_run
        elif policy.mode == "estimate":
            self._deadtime_estimated_us = policy.estimated_us
            self._deadtime_source_run = policy.source_run
        self._update_deadtime_status()
        self._mark_dirty()
        self._refresh_preview()

    def _on_configure_background(self) -> None:
        """Open the background dialog, seeded from the current state."""
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        has_fixed = any(
            isinstance(grouping.get(key), (list, tuple))
            for key in ("background_fixed_values", "background_fix", "bkg_fix")
        )
        candidates = [
            BackgroundReferenceRunCandidate(
                run_number=int(ds.run_number),
                label=f"{ds.run_label} (run {ds.run_number})",
                source_file=str(ds.run.source_file or "") if ds.run is not None else "",
                good_frames=(ds.run.grouping or {}).get("good_frames")
                if ds.run is not None and isinstance(ds.run.grouping, dict)
                else None,
            )
            for ds in self._datasets
            if ds.run is not None
            and ds.run.histograms
            and int(ds.run_number) != int(self._reference_dataset.run_number)
        ]

        preview = None
        if self._run is not None and self._run.histograms:
            forward_gid = int(self._forward_combo.currentData())
            backward_gid = int(self._backward_combo.currentData())
            forward_indices = self._filtered_group_indices(forward_gid)
            backward_indices = self._filtered_group_indices(backward_gid)
            n_hist = len(self._run.histograms)
            if (
                forward_indices
                and backward_indices
                and max(forward_indices) < n_hist
                and max(backward_indices) < n_hist
            ):
                t0_bin, _offset, _first, last_good = self._resolve_good_bin_limits_from_controls()
                forward_counts = apply_grouping(self._run.histograms, forward_indices)
                backward_counts = apply_grouping(self._run.histograms, backward_indices)
                bin_width = float(self._run.histograms[0].bin_width)
                preview = (forward_counts, backward_counts, bin_width, t0_bin, last_good)

        # Attach the sample side's good_frames so a payload built from the
        # picked candidate can compute the frame-ratio scale immediately.
        reference_grouping = (
            self._run.grouping
            if self._run is not None and isinstance(self._run.grouping, dict)
            else {}
        )
        background_run_payload = (
            dict(self._background_run_payload) if self._background_run_payload else None
        )
        if background_run_payload is not None:
            background_run_payload.setdefault(
                "good_frames_sample", reference_grouping.get("good_frames")
            )

        dlg = BackgroundDialog(
            available_modes=self._available_background_modes(),
            has_fixed_values=has_fixed,
            initial_mode=self._background_mode,
            background_run_payload=background_run_payload,
            reference_run_candidates=candidates,
            preview=preview,
            forward_label="F",
            backward_label="B",
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        policy = dlg.get_policy()
        self._background_mode = policy.mode
        if policy.mode == "reference_run":
            payload = dict(policy.details.get("background_run") or {})
            payload["good_frames_sample"] = reference_grouping.get("good_frames")
            self._background_run_payload = payload
        self._update_background_status()
        self._mark_dirty()
        self._refresh_preview()

    def _available_background_modes(self) -> tuple[str, ...]:
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        if self._run is not None:
            metadata.update(getattr(self._run, "metadata", {}) or {})
        source_file = str(getattr(self._run, "source_file", "") or metadata.get("source_file", ""))
        return available_background_modes(metadata=metadata, source_file=source_file)

    def _current_background_mode(self) -> str:
        return self._background_mode

    def _update_background_status(self) -> None:
        """Show the per-mode summary under the background status row."""
        mode = self._current_background_mode()
        if mode == "tail_fit":
            self._background_status_label.setText(self._tail_fit_preview_text())
            return
        if mode == "reference_run":
            payload = self._background_run_payload or {}
            run_number = payload.get("run_number")
            label = f"run {run_number}" if run_number else str(payload.get("source_file", ""))
            sample = payload.get("good_frames_sample")
            reference = payload.get("good_frames_reference")
            try:
                scale = float(sample) / float(reference)
                self._background_status_label.setText(
                    f"Subtract {label}, frame-ratio scale {scale:.4g}."
                )
            except (TypeError, ValueError, ZeroDivisionError):
                self._background_status_label.setText(
                    f"Subtract {label} (frame ratio resolved at reduction)."
                )
            return
        self._background_status_label.setText(f"Background: {mode}" if mode != "none" else "")

    def _tail_fit_preview_text(self) -> str:
        """Run the tail fit on the reference run's groups for display."""
        if self._run is None or not self._run.histograms:
            return ""
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        t0_bin, _offset, _first, last_good = self._resolve_good_bin_limits_from_controls()
        bin_width = float(self._run.histograms[0].bin_width)
        parts: list[str] = []
        for name, gid in (("F", forward_gid), ("B", backward_gid)):
            indices = self._filtered_group_indices(gid)
            if not indices or max(indices) >= len(self._run.histograms):
                continue
            counts = apply_grouping(self._run.histograms, indices)
            fit = fit_tail_background(
                counts,
                bin_width_us=bin_width,
                t0_bin=int(t0_bin),
                last_good_bin=int(last_good),
            )
            if not fit.ok:
                parts.append(f"{name}: {fit.message}")
                continue
            value = _format_value_with_uncertainty(fit.rate_per_us, fit.rate_error_per_us)
            note = " (consistent with zero)" if fit.consistent_with_zero else ""
            parts.append(f"{name}: {value} counts/µs{note}")
        return "Tail-fit background — " + "; ".join(parts) if parts else ""

    def _on_apply(self) -> None:
        """Validate form values before accepting the dialog."""
        t0_bin, t_good_offset, first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        max_bin = self._max_bin_index_for_reference_dataset()
        if first_good_bin > max_bin:
            QMessageBox.warning(
                self,
                "Invalid Good-Data Window",
                "t_good offset places first good bin beyond the histogram range.",
            )
            return
        if last_good_bin < first_good_bin:
            QMessageBox.warning(
                self,
                "Invalid Good-Data Window",
                "Last good bin must be greater than or equal to t0 + t_good offset.",
            )
            return
        if (
            self._deadtime_mode != "off"
            and self._resolve_deadtime_payload(show_warnings=True) is None
        ):
            return
        excluded = self._current_excluded_detectors()
        if excluded is None:
            return  # parse error already shown
        if excluded:
            forward_gid = int(self._forward_combo.currentData())
            backward_gid = int(self._backward_combo.currentData())
            for label, gid in (("Forward", forward_gid), ("Backward", backward_gid)):
                remaining = self._filtered_group_indices(gid)
                if not remaining:
                    QMessageBox.warning(
                        self,
                        "Detector Exclusion",
                        f"{label} group would have no detectors left after exclusion.",
                    )
                    return
        # Lift the validated form state into the draft profile so
        # get_grouping_result() returns it. The draft is considered saved once
        # applied, so the close guard does not re-prompt.
        self._sync_draft_from_form()
        self._draft_dirty = False
        self.accept()

    # ------------------------------------------------------------------
    # Dirty tracking, apply-gating, close guard (M2)
    # ------------------------------------------------------------------

    def _connect_dirty_tracking(self) -> None:
        """Mark the draft dirty whenever an editable form control changes."""
        self._alpha_spin.valueChanged.connect(self._mark_dirty)
        self._alpha_method_combo.currentIndexChanged.connect(self._mark_dirty)
        self._forward_combo.currentIndexChanged.connect(self._mark_dirty)
        self._backward_combo.currentIndexChanged.connect(self._mark_dirty)
        self._bunch_spin.valueChanged.connect(self._mark_dirty)
        self._binning_mode_combo.currentIndexChanged.connect(self._mark_dirty)
        self._bin0_spin.valueChanged.connect(self._mark_dirty)
        self._bin10_spin.valueChanged.connect(self._mark_dirty)
        self._exclude_edit.textEdited.connect(self._mark_dirty)
        # Deadtime/background dirty-tracking happens inline in
        # ``_on_configure_deadtime``/``_on_configure_background`` (the
        # Configure… dialogs mark the draft dirty on Accept).
        self._group_table.itemChanged.connect(self._mark_dirty)
        for button in self._period_mode_buttons.values():
            button.toggled.connect(self._mark_dirty)

    # ------------------------------------------------------------------
    # Live preview pane
    # ------------------------------------------------------------------

    def _connect_preview_refresh(self) -> None:
        """Refresh the live preview whenever an editable control changes.

        Reuses the same control set as dirty-tracking (groups, alpha, binning,
        exclusions, deadtime, background, period) so every edit routed to the
        draft also drives a debounced recompute. The pane itself coalesces rapid
        edits, so connecting broadly here is cheap.
        """
        self._alpha_spin.valueChanged.connect(self._refresh_preview)
        self._alpha_method_combo.currentIndexChanged.connect(self._refresh_preview)
        self._forward_combo.currentIndexChanged.connect(self._refresh_preview)
        self._backward_combo.currentIndexChanged.connect(self._refresh_preview)
        self._bunch_spin.valueChanged.connect(self._refresh_preview)
        self._binning_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self._bin0_spin.valueChanged.connect(self._refresh_preview)
        self._bin10_spin.valueChanged.connect(self._refresh_preview)
        self._t0_spin.valueChanged.connect(self._refresh_preview)
        self._t_good_offset_spin.valueChanged.connect(self._refresh_preview)
        self._last_good_spin.valueChanged.connect(self._refresh_preview)
        self._exclude_edit.textEdited.connect(self._refresh_preview)
        self._group_table.itemChanged.connect(self._refresh_preview)
        for button in self._period_mode_buttons.values():
            button.toggled.connect(self._refresh_preview)
        # Deadtime and background live in their own dialogs; the accept paths in
        # _on_configure_deadtime/_on_configure_background call _refresh_preview()
        # directly after folding the returned policy into the draft.

    def _refresh_preview(self, *args: object) -> None:
        """Recompute the live asymmetry preview for the draft + preview run.

        Resolves the current form payload against the preview run into the
        effective ``run.grouping`` shape and hands it to the pane, which debounces
        and reduces off the GUI thread. Any resolution error is swallowed and
        surfaced as a muted status message rather than a popup — the preview is
        advisory. Datasets without raw histograms (co-added curves) make the pane
        hide itself with a note.
        """
        pane = getattr(self, "_preview_pane", None)
        if pane is None:
            return
        run = self._run
        histograms = list(run.histograms) if run is not None and run.histograms else []
        try:
            grouping = self._preview_effective_grouping()
        except Exception:  # noqa: BLE001 — advisory preview, never crash the dialog
            grouping = None
        if grouping is None:
            pane.request_preview(
                histograms=None,
                grouping={},
                run_number=self._preview_run_number(),
            )
            return
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        if run is not None:
            metadata.update(getattr(run, "metadata", {}) or {})
        facility = str(metadata.get("facility", metadata.get("instrument", "")))
        pane.request_preview(
            histograms=histograms,
            grouping=grouping,
            facility=facility,
            run_number=self._preview_run_number(),
        )

    def _preview_run_number(self) -> int | None:
        if self._reference_dataset is None:
            return None
        try:
            return int(self._reference_dataset.run_number)
        except (TypeError, ValueError):
            return None

    def _preview_effective_grouping(self) -> dict[str, Any] | None:
        """Resolve the current form payload against the preview run.

        Builds a throwaway draft profile from the live form payload (without
        mutating ``self._draft`` or its dirty state) and resolves it against the
        preview run into the full ``run.grouping`` shape the reduction consumes.
        """
        if self._run is None or not self._run.histograms or self._fingerprint is None:
            return None
        from asymmetry.core.project.profiles import resolve_effective_grouping

        payload = self._current_grouping_payload()
        profile = profile_from_form_payload(
            payload,
            name=self._draft_name,
            fingerprint=self._fingerprint,
            active=True,
        )
        return resolve_effective_grouping(profile, self._run)

    def _update_apply_enabled(self) -> None:
        """Disable Apply (with a reason) when no run inherits this profile."""
        inheriting = self._scope_panel.inheriting_run_numbers()
        enabled = bool(inheriting)
        self._apply_btn.setEnabled(enabled)
        if enabled:
            self._apply_btn.setToolTip(
                f"Save profile '{self._draft_name}' and apply it to "
                f"{len(inheriting)} inheriting run(s)."
            )
        else:
            self._apply_btn.setToolTip(
                "No run of this fingerprint inherits the profile "
                "(all released). Reattach a run to apply."
            )

    def _clear_dirty(self) -> None:
        """Disarm the unsaved-draft guard (used by the test teardown fixture)."""
        self._draft_dirty = False

    def _guard_discard(self) -> bool:
        """Return whether it is safe to close (prompting on a dirty draft)."""
        if not self._draft_dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Discard changes",
            f"Discard changes to profile '{self._draft_name}'?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def reject(self) -> None:
        """Reject with an unsaved-draft guard (Cancel / Esc)."""
        if self._datasets and not self._guard_discard():
            return
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802  (Qt override)
        """Close with an unsaved-draft guard (window ✕)."""
        if self._datasets and not self._guard_discard():
            event.ignore()
            return
        self._teardown_preview()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """Tear the preview runner down on every dialog dismissal (accept/reject)."""
        self._teardown_preview()
        super().done(result)

    def _teardown_preview(self) -> None:
        """Cancel and join the preview runner (idempotent)."""
        pane = getattr(self, "_preview_pane", None)
        if pane is not None:
            pane.shutdown()

    def _populate_group_table(self) -> None:
        """Render the detector-group table used as grouping context."""
        self._group_table.setRowCount(len(self._groups))
        for row, gid in enumerate(sorted(self._groups)):
            self._group_table.setItem(row, 0, QTableWidgetItem(str(gid)))
            include_item = QTableWidgetItem()
            include_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            include_item.setCheckState(
                Qt.CheckState.Checked
                if self._included_groups.get(int(gid), True)
                else Qt.CheckState.Unchecked
            )
            self._group_table.setItem(row, 1, include_item)
            name = self._group_names.get(gid, "")
            self._group_table.setItem(row, 2, QTableWidgetItem(name))
            detectors = [str(idx + 1) for idx in self._groups[gid]]
            self._group_table.setItem(row, 3, QTableWidgetItem(", ".join(detectors)))
        self._group_table.resizeColumnsToContents()
        visible_rows = min(max(len(self._groups), 3), 5)
        row_height = max(24, self._group_table.verticalHeader().defaultSectionSize())
        header_height = self._group_table.horizontalHeader().height()
        frame = 2 * self._group_table.frameWidth()
        height = header_height + visible_rows * row_height + frame + 8
        self._group_table.setMinimumHeight(0)
        self._group_table.setMaximumHeight(height)

    def _current_included_groups(self) -> dict[int, bool]:
        """Return the include-checkbox state from the group table."""
        included: dict[int, bool] = {}
        for row in range(self._group_table.rowCount()):
            gid_item = self._group_table.item(row, 0)
            include_item = self._group_table.item(row, 1)
            if gid_item is None or include_item is None:
                continue
            try:
                gid = int(gid_item.text())
            except (TypeError, ValueError):
                continue
            included[gid] = include_item.checkState() == Qt.CheckState.Checked
        for gid in self._groups:
            included.setdefault(int(gid), True)
        self._included_groups = included
        return dict(included)

    def _detect_vector_axis_pairs(self) -> dict[str, tuple[int, int]]:
        """Return projection group pairs for the current grouping.

        Prefers an explicit projection declaration (set when a multi-projection
        preset is applied) and falls back to the legacy canonical vector group
        names; see :func:`asymmetry.core.instrument.derive_projection_pairs`.
        """
        if not self._groups or not isinstance(self._group_names, dict):
            return {}
        return derive_projection_pairs(
            self._groups, self._group_names, getattr(self, "_projection_specs", None)
        )

    def _vector_alpha_key(self, axis: str) -> str:
        """Return grouping payload key for a canonical vector axis."""
        return {
            "P_x": "alpha_x",
            "P_y": "alpha_y",
            "P_z": "alpha_z",
        }.get(axis, "alpha")

    def _legacy_vector_alpha_key(self, axis: str) -> str:
        """Return legacy payload key for backward compatibility."""
        return {
            "P_x": "alpha_px",
            "P_y": "alpha_py",
            "P_z": "alpha_pz",
        }.get(axis, "alpha")

    def _alpha_value_for_axis(self, grouping: dict[str, Any], axis: str, fallback: float) -> float:
        """Return best alpha value for *axis* from grouping payload/state."""
        key = self._vector_alpha_key(axis)
        legacy_key = self._legacy_vector_alpha_key(axis)
        try:
            return float(
                grouping.get(
                    key,
                    grouping.get(legacy_key, grouping.get("alpha", fallback)),
                )
            )
        except (TypeError, ValueError):
            return float(fallback)

    def _group_display_name(self, gid: int) -> str:
        """Return display text for a group ID with optional group name."""
        label = str(self._group_names.get(gid, "")).strip()
        if label and label != str(gid):
            return f"{gid}: {label}"
        return str(gid)

    def _refresh_group_combo_items(
        self,
        *,
        forward_gid: int | None = None,
        backward_gid: int | None = None,
    ) -> None:
        """Rebuild forward/backward group combo entries from current group state."""
        if forward_gid is None:
            try:
                forward_gid = int(self._forward_combo.currentData())
            except (TypeError, ValueError):
                forward_gid = 1
        if backward_gid is None:
            try:
                backward_gid = int(self._backward_combo.currentData())
            except (TypeError, ValueError):
                backward_gid = 2

        self._forward_combo.blockSignals(True)
        self._backward_combo.blockSignals(True)
        self._forward_combo.clear()
        self._backward_combo.clear()
        for gid in sorted(self._groups):
            display = self._group_display_name(gid)
            self._forward_combo.addItem(display, gid)
            self._backward_combo.addItem(display, gid)
        self._set_combo_to_group(self._forward_combo, int(forward_gid))
        self._set_combo_to_group(self._backward_combo, int(backward_gid))
        self._forward_combo.blockSignals(False)
        self._backward_combo.blockSignals(False)

    @staticmethod
    def _is_canonical_vector_pairs(pairs: dict[str, tuple[int, int]]) -> bool:
        """True when the per-axis alpha table (P_x/P_y/P_z rows) can render *pairs*.

        The vector-alpha table and the vector branches of the grouping payload
        are hardcoded to the canonical EMU axes. A transverse-field dual-grouping
        preset declares non-canonical projections (e.g. ``Top-Bottom`` /
        ``Fwd-Back``); those drive the plot chip bar but are handled here with the
        ordinary single-alpha control, so the canonical table is skipped to avoid
        empty rows and a stale P_z-spin alpha. The reduction still consumes each
        projection's own declared alpha (see
        :meth:`MainWindow._resolve_vector_alpha_values`); the single-alpha control
        only supplies the base alpha those projections fall back to when they do
        not declare one of their own.
        """
        return any(axis in pairs for axis in CANONICAL_VECTOR_AXES)

    def _update_vector_mode_controls(self, grouping_values: dict[str, Any] | None = None) -> None:
        """Toggle between single-alpha and per-projection alpha controls.

        The per-projection alpha table is shown whenever the grouping resolves to
        any projection pairs — the canonical EMU axes (P_z/P_y/P_x) *and* the
        non-canonical presets (GPS WEP's FB/UD, the MuSR/HiFi transverse pairs).
        Only a plain forward/backward grouping with no projections falls back to
        the single-alpha control.
        """
        if grouping_values is None:
            grouping_values = {}

        pairs = self._detect_vector_axis_pairs()
        self._vector_axis_pairs = pairs
        canonical = self._is_canonical_vector_pairs(pairs)
        vector_mode = bool(pairs)

        self._forward_row_label.setVisible(not vector_mode)
        self._forward_combo.setVisible(not vector_mode)
        self._backward_row_label.setVisible(not vector_mode)
        self._backward_combo.setVisible(not vector_mode)
        self._alpha_row_label.setVisible(not vector_mode)
        self._single_alpha_widget.setVisible(not vector_mode)
        self._vector_alpha_widget.setVisible(vector_mode)

        if not vector_mode:
            self._rebuild_vector_alpha_table([], grouping_values, canonical)
            if "alpha" in grouping_values:
                try:
                    self._alpha_spin.setValue(float(grouping_values.get("alpha", 1.0)))
                except (TypeError, ValueError):
                    pass
            return

        ordered = self._ordered_projection_labels(pairs, canonical)
        self._rebuild_vector_alpha_table(ordered, grouping_values, canonical)

        # The canonical EMU table stays anchored to P_z: the single-alpha
        # control (still the persisted base alpha) and the forward/backward
        # combos track the P_z pair, matching the original behaviour.
        if canonical and "P_z" in pairs:
            pz_fwd, pz_bwd = pairs["P_z"]
            self._set_combo_to_group(self._forward_combo, pz_fwd)
            self._set_combo_to_group(self._backward_combo, pz_bwd)
            self._alpha_spin.setValue(float(self._vector_alpha_spins["P_z"].value()))

    @staticmethod
    def _ordered_projection_labels(pairs: dict[str, tuple[int, int]], canonical: bool) -> list[str]:
        """Order the per-projection alpha rows for display.

        Canonical EMU keeps the historical P_z-first ordering; non-canonical
        presets follow their declared projection order.
        """
        if canonical:
            # P_z first, matching the historical per-axis table ordering.
            return [axis for axis in reversed(CANONICAL_VECTOR_AXES) if axis in pairs]
        return list(pairs)

    def _projection_alpha_for_label(self, label: str, fallback: float) -> float:
        """Return the declared alpha for a non-canonical projection *label*."""
        for spec in self._projection_specs or []:
            if isinstance(spec, dict) and str(spec.get("label")) == label:
                try:
                    return float(spec.get("alpha", fallback))
                except (TypeError, ValueError):
                    return float(fallback)
        return float(fallback)

    def _rebuild_vector_alpha_table(
        self,
        ordered_labels: list[str],
        grouping_values: dict[str, Any],
        canonical: bool,
    ) -> None:
        """Rebuild the per-projection alpha grid for the current projections.

        Columns: projection label | Forward group | Backward group | α spin |
        Estimate button, with a trailing "Estimate All α" button. Rows are keyed
        by projection label (the canonical EMU labels are P_x/P_y/P_z), so the
        spins, group labels and estimate buttons are all rebuilt whenever the
        projection set changes.
        """
        layout = self._vector_alpha_layout
        clear_layout(layout)
        self._vector_alpha_spins.clear()
        self._vector_forward_labels.clear()
        self._vector_backward_labels.clear()
        self._vector_estimate_buttons.clear()
        self._estimate_all_btn = None
        if not ordered_labels:
            return

        layout.addWidget(QLabel("Forward"), 0, 1)
        layout.addWidget(QLabel("Backward"), 0, 2)
        layout.addWidget(QLabel("α"), 0, 3)

        fallback = float(self._alpha_spin.value())
        for row_idx, label in enumerate(ordered_labels, start=1):
            fwd_gid, bwd_gid = self._vector_axis_pairs.get(label, (None, None))
            layout.addWidget(QLabel(label), row_idx, 0)
            fwd_label = QLabel(
                self._group_display_name(int(fwd_gid)) if fwd_gid is not None else "-"
            )
            bwd_label = QLabel(
                self._group_display_name(int(bwd_gid)) if bwd_gid is not None else "-"
            )
            layout.addWidget(fwd_label, row_idx, 1)
            layout.addWidget(bwd_label, row_idx, 2)
            self._vector_forward_labels[label] = fwd_label
            self._vector_backward_labels[label] = bwd_label

            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0.01, 1000.0)
            if canonical:
                spin.setValue(self._alpha_value_for_axis(grouping_values, label, fallback))
            else:
                spin.setValue(self._projection_alpha_for_label(label, fallback))
            layout.addWidget(spin, row_idx, 3)
            self._vector_alpha_spins[label] = spin

            btn = QPushButton("Estimate α")
            btn.setAutoDefault(False)
            btn.setDefault(False)
            btn.clicked.connect(
                lambda _checked=False, slot=label: self._estimate_alpha_for_axis(slot)
            )
            layout.addWidget(btn, row_idx, 4)
            self._vector_estimate_buttons[label] = btn

        self._estimate_all_btn = QPushButton("Estimate All α")
        self._estimate_all_btn.setAutoDefault(False)
        self._estimate_all_btn.setDefault(False)
        self._estimate_all_btn.clicked.connect(self._estimate_all_alpha)
        layout.addWidget(self._estimate_all_btn, len(ordered_labels) + 1, 0, 1, 5)

    def _resolve_detector_layout(self, n_histo: int, metadata: dict[str, Any]):
        """Resolve the InstrumentLayout to show in the detector layout editor.

        Resolution order:

        1. A previously chosen layout name (``_detector_layout_instrument_name``)
           when it maps to a known layout.
        2. Facility-aware auto-detection. PSI data is always re-detected when the
           stored name resolves to an ISIS-only layout, because the PSI loader
           records the raw instrument string (e.g. ``"HIFI"``) which otherwise
           canonicalises to the unrelated ISIS HiFi layout instead of HAL-9500.
        3. HiFi as a final fallback.
        """
        instrument = None
        instrument_name = self._detector_layout_instrument_name
        if instrument_name:
            try:
                instrument = get_instrument_layout(instrument_name)
            except KeyError:
                instrument = None

        psi = self._reference_is_psi()
        if instrument is None or (psi and instrument.name in {"HiFi", "MuSR", "EMU"}):
            detected = detect_instrument(
                n_histo,
                metadata=metadata,
                source_file=self._run.source_file if self._run else None,
            )
            if detected:
                try:
                    instrument = get_instrument_layout(detected)
                except KeyError:
                    pass

        if instrument is None:
            instrument = get_instrument_layout("HiFi")
        else:
            # Correct to the layout variant whose detector count matches this run
            # (e.g. GPS 6-detector BIN vs GPS-RD 11-detector ROOT), so a stored
            # name does not pin the wrong-sized layout when the data format changes.
            instrument = get_instrument_layout(variant_for_histograms(instrument.name, n_histo))
        return instrument

    def _on_detector_layout(self) -> None:
        """Open the interactive detector layout editor as a sub-dialog."""
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        # Determine number of histograms for instrument auto-detection
        n_histo = len(self._run.histograms) if self._run and self._run.histograms else 0
        grouping = self._run.grouping or {} if self._run else {}
        metadata = (
            dict(self._run.metadata) if self._run and isinstance(self._run.metadata, dict) else {}
        )
        if isinstance(grouping, dict):
            for key in ("histogram_labels", "group_names"):
                if key in grouping and key not in metadata:
                    metadata[key] = grouping[key]

        # Resolve only to seed the editor; the chosen instrument is committed to
        # _detector_layout_instrument_name from the dialog result on Accept, so a
        # cancelled editor session leaves the stored selection untouched.
        instrument = self._resolve_detector_layout(n_histo, metadata)

        forward_gid = int(grouping.get("forward_group", self._forward_combo.currentData() or 1))
        backward_gid = int(grouping.get("backward_group", self._backward_combo.currentData() or 2))

        # Convert internal 0-based indices to 1-based for the layout editor
        groups_1based = {gid: [idx + 1 for idx in idxs] for gid, idxs in self._groups.items()}

        current_exclusion = self._current_excluded_detectors() or []
        dlg = DetectorLayoutDialog(
            instrument=instrument,
            groups=groups_1based,
            group_names=dict(self._group_names),
            initial_preset_name=self._grouping_preset_name,
            forward_group=forward_gid,
            backward_group=backward_gid,
            excluded_detectors=current_exclusion,
            projections=self._projection_specs,
            field_direction=self._reference_field_direction(),
            parent=self,
        )
        if dlg.exec() != DetectorLayoutDialog.DialogCode.Accepted:
            return

        result = dlg.get_result()
        self._exclude_edit.setText(format_detector_list(result.get("excluded_detectors", [])))
        # Preserve any per-projection alpha the user edited in the table: the
        # layout editor returns each projection with its preset-declared alpha,
        # which would otherwise silently revert an unsaved edit for a label that
        # still exists. Live spin values are only flushed into the payload on
        # Accept, so carry them across here for surviving labels.
        edited_alpha = {
            label: float(spin.value()) for label, spin in self._vector_alpha_spins.items()
        }
        result_projections = result.get("projections")
        if result_projections:
            merged: list[dict[str, Any]] = []
            for proj in result_projections:
                proj = dict(proj)
                label = str(proj.get("label"))
                if label in edited_alpha:
                    proj["alpha"] = edited_alpha[label]
                merged.append(proj)
            self._projection_specs = merged
        else:
            self._projection_specs = None

        # Write back: convert 1-based IDs back to 0-based internal indices
        new_groups_0based: dict[int, list[int]] = {}
        for gid, det_ids in result["groups"].items():
            new_groups_0based[gid] = sorted(max(0, d - 1) for d in det_ids)

        # Keep any previously defined groups that are not in the result
        # (preserves groups beyond the 8 shown in the editor)
        self._groups = new_groups_0based
        previous_included = dict(self._included_groups)
        self._included_groups = {
            int(gid): bool(previous_included.get(int(gid), True)) for gid in self._groups
        }
        self._group_names = result.get("group_names", {})
        preset_name = result.get("grouping_preset")
        # When the editor reports a match, adopt its name outright. When it
        # reports None (custom/drifted state), keep the *previous* preset name
        # for now: ``_refresh_preset_chip`` below re-derives drift from the
        # payload itself and clears it to None, which is what lets the chip
        # read "Custom (edited from <old preset>)" instead of a bare "Custom".
        if preset_name:
            self._grouping_preset_name = str(preset_name)
        instrument_name = result.get("instrument")
        self._detector_layout_instrument_name = str(instrument_name) if instrument_name else None

        # Update forward/backward combos
        new_fwd = result.get("forward_group", forward_gid)
        new_bwd = result.get("backward_group", backward_gid)
        new_fwd, new_bwd = self._analysis_pair_for_reference(int(new_fwd), int(new_bwd))
        self._refresh_group_combo_items(forward_gid=int(new_fwd), backward_gid=int(new_bwd))

        self._populate_group_table()
        self._update_vector_mode_controls()
        self._update_grouping_recommendation()
        self._mark_dirty()
        self._refresh_preset_chip(self._current_grouping_payload())
        self._refresh_preview()

    def _set_combo_to_group(self, combo: QComboBox, group_id: int) -> None:
        """Set combo box to a group ID if present, preserving defaults otherwise."""
        idx = combo.findData(group_id)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_alpha_method(self, method_key: str) -> None:
        """Select the alpha estimation method combo entry, defaulting to diamagnetic."""
        idx = self._alpha_method_combo.findData(str(method_key))
        self._alpha_method_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _current_alpha_method(self) -> str:
        """Return the grouping-dict key of the selected estimation method."""
        return str(self._alpha_method_combo.currentData() or "diamagnetic")

    def _set_binning_mode(self, mode_key: str) -> None:
        """Select the binning-mode combo entry, defaulting to fixed."""
        idx = self._binning_mode_combo.findData(str(mode_key))
        self._binning_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _current_binning_mode(self) -> str:
        return str(self._binning_mode_combo.currentData() or "fixed")

    def _on_binning_mode_changed(self) -> None:
        """Enable the controls for the active binning mode (WiMDA pattern)."""
        mode = self._current_binning_mode()
        self._bunch_spin.setEnabled(mode == "fixed")
        # bin0 applies to both non-fixed modes; bin10 only to variable.
        for widget in (self._bin0_spin, self._bin0_label):
            widget.setEnabled(mode != "fixed")
            widget.setVisible(mode != "fixed")
        for widget in (self._bin10_spin, self._bin10_label):
            widget.setEnabled(mode == "variable")
            widget.setVisible(mode == "variable")

    def _set_excluded_detectors_text(self, grouping: dict[str, Any]) -> None:
        raw = grouping.get("excluded_detectors")
        ids = [int(v) for v in raw] if isinstance(raw, (list, tuple)) else []
        self._exclude_edit.setText(format_detector_list(ids))

    def _excluded_index_set(self) -> set[int]:
        """0-based indices of currently excluded detectors (empty on parse error)."""
        ids = self._current_excluded_detectors() or []
        return set(excluded_detector_indices({"excluded_detectors": ids}))

    def _filtered_group_indices(self, gid: int) -> list[int]:
        """0-based indices of group *gid* with excluded detectors removed.

        Routes the dialog's 0-based ``self._groups`` through the same core
        exclusion primitive the reduction paths use, so the dialog cannot apply
        exclusion differently from the engine.
        """
        return filter_excluded_indices(
            list(self._groups.get(gid, [])),
            {"excluded_detectors": self._current_excluded_detectors() or []},
        )

    def _current_excluded_detectors(self) -> list[int] | None:
        """Parse the exclusion field; warn and return None on bad input."""
        text = self._exclude_edit.text().strip()
        if not text:
            return []
        try:
            return parse_detector_list(text)
        except ValueError as exc:
            QMessageBox.warning(self, "Detector Exclusion", str(exc))
            return None

    def _on_find_t0(self) -> None:
        """Estimate t0 from the reference run and fill the override spinner."""
        if self._run is None or not self._run.histograms:
            QMessageBox.warning(self, "Find t0", "Reference run has no histograms.")
            return
        metadata = dict(getattr(self._reference_dataset, "metadata", {}) or {})
        metadata.update(self._run.metadata or {})
        excluded = self._excluded_index_set()
        histograms = [hist for i, hist in enumerate(self._run.histograms) if i not in excluded]
        if not histograms:
            QMessageBox.warning(self, "Find t0", "All detectors are excluded.")
            return
        search = find_t0_for_run(histograms, metadata)
        if not search.ok:
            QMessageBox.warning(self, "Find t0", search.message)
            return
        index_base = self._bin_index_base()
        self._t0_spin.setValue(int(search.consensus_t0_bin) + index_base)
        strategy = "pulse-edge midpoint" if search.strategy == "pulse_edge" else "prompt peak"
        self._alpha_result_label.setText(
            f"t0 = bin {search.consensus_t0_bin + index_base} ({strategy}, "
            f"detector spread {search.spread_bins} bins) — press Apply to use it."
        )

    def _update_map_periods_visibility(self) -> None:
        """Show Map periods… only when the reference run has 3+ periods."""
        visible = len(self._sibling_period_datasets()) > 2
        self._map_periods_btn.setVisible(visible)

    def _sibling_period_datasets(self) -> list[MuonDataset]:
        """Per-period sibling datasets of the reference run, in period order."""
        if self._reference_dataset is None:
            return []
        metadata = getattr(self._reference_dataset, "metadata", {}) or {}
        try:
            count = int(metadata.get("period_count", 1))
        except (TypeError, ValueError):
            count = 1
        if count <= 2:
            return []
        source = metadata.get("source_run_number", metadata.get("run_number"))
        siblings = [
            ds
            for ds in self._datasets
            if ds.run is not None
            and (ds.metadata or {}).get("source_run_number") == source
            and (ds.metadata or {}).get("period_number") is not None
        ]
        siblings.sort(key=lambda ds: int((ds.metadata or {}).get("period_number", 0)))
        return siblings if len(siblings) == count else []

    def _on_map_periods(self) -> None:
        """Collect a period → red/green/ignore mapping for the reference run."""
        from asymmetry.gui.windows.period_mapping_dialog import PeriodMappingDialog

        siblings = self._sibling_period_datasets()
        if len(siblings) < 3:
            QMessageBox.warning(
                self,
                "Map Periods",
                "Period mapping needs the per-period datasets of a 3+ period "
                "run loaded in the project.",
            )
            return
        dialog = PeriodMappingDialog(siblings, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.mapping is None:
            return
        metadata = getattr(self._reference_dataset, "metadata", {}) or {}
        self.period_mapping_request = {
            "mapping": dialog.mapping,
            "source_run_number": metadata.get("source_run_number", metadata.get("run_number")),
            "period_run_numbers": [int(ds.run_number) for ds in siblings],
        }
        self._alpha_result_label.setText(
            "Period mapping recorded — a combined red/green dataset is created "
            "when you press Apply."
        )

    def _estimate_alpha(self) -> None:
        """Launch the Alpha calibration dialog for the single-alpha grouping.

        Replaces the old inline "Estimate α" action: the dialog lets the user
        pick a calibration run, method and window and *see* alpha balance the
        asymmetry. On OK the calibrated policy is written back into the alpha
        spin and its provenance (method, error, source run), so the payload's
        ``alpha_method`` / ``alpha_error`` / ``alpha_reference_run`` — and hence
        the resolved ``AlphaPolicy`` — carry the calibration exactly as the old
        inline estimate did.
        """
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        self._launch_calibration_dialog("single", forward_gid, backward_gid, self._alpha_spin)

    def _launch_calibration_dialog(
        self,
        slot: str,
        forward_gid: int,
        backward_gid: int,
        target_spin: QDoubleSpinBox,
        *,
        slot_label: str | None = None,
    ) -> None:
        """Open the calibration dialog for one group pair and write the result.

        *slot* is the ``_alpha_estimate_state`` key (``"single"`` or an axis /
        projection label) the calibration provenance is recorded under; *target_spin*
        is the alpha spin the calibrated value is written into.
        """
        if forward_gid == backward_gid:
            QMessageBox.warning(
                self, "Invalid Grouping", "Forward and backward groups must differ."
            )
            return
        if self._run is None or not self._run.histograms:
            QMessageBox.warning(self, "Alpha Calibration", "Reference run has no histograms.")
            return

        from asymmetry.gui.windows.grouping.alpha_calibration_dialog import (
            AlphaCalibrationDialog,
        )

        initial_policy = AlphaPolicy(
            mode="calibrated",
            value=float(target_spin.value()),
            method=self._current_alpha_method(),
            source_run=self._alpha_estimate_state.get(slot, (None, None, None))[2],
        )
        dialog = AlphaCalibrationDialog(
            self._fingerprint_datasets(),
            groups=self._groups,
            group_names=self._group_names,
            forward_group=int(forward_gid),
            backward_group=int(backward_gid),
            excluded_detectors=self._current_excluded_detectors() or [],
            initial_policy=initial_policy,
            slot_label=slot_label if slot != "single" else None,
            selected_run_number=int(self._reference_dataset.run_number),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        policy = dialog.result_policy()
        if policy is None:
            return
        self._apply_calibrated_policy(slot, target_spin, policy)

    def _apply_calibrated_policy(
        self, slot: str, target_spin: QDoubleSpinBox, policy: AlphaPolicy
    ) -> None:
        """Write a calibrated :class:`AlphaPolicy` back into the form + provenance."""
        # Record provenance the same shape the old inline estimate produced, so
        # _current_grouping_payload emits alpha_method/alpha_error/alpha_reference_run.
        self._alpha_estimate_state[slot] = (
            float(policy.value),
            policy.error,
            int(policy.source_run) if policy.source_run is not None else -1,
        )
        if policy.method:
            self._set_alpha_method(policy.method)
        self._suppress_alpha_provenance_reset = True
        try:
            target_spin.setValue(float(policy.value))
        finally:
            self._suppress_alpha_provenance_reset = False
        if slot == "single":
            self._refresh_alpha_provenance_label()
        elif slot == "P_z":
            self._suppress_alpha_provenance_reset = True
            try:
                self._alpha_spin.setValue(float(policy.value))
            finally:
                self._suppress_alpha_provenance_reset = False
            self._refresh_alpha_provenance_label()
        self._record_calibration_result_label(slot, policy)
        self._mark_dirty()

    def _record_calibration_result_label(self, slot: str, policy: AlphaPolicy) -> None:
        """Show the calibrated α in the shared result label."""
        method_label = next(
            (label for label, key, _ in _ALPHA_METHOD_ITEMS if key == policy.method),
            policy.method,
        )
        formatted = _format_value_with_uncertainty(policy.value, policy.error)
        slot_prefix = "" if slot == "single" else f"{slot}: "
        run_text = f", run {policy.source_run}" if policy.source_run is not None else ""
        self._alpha_result_label.setText(f"{slot_prefix}α = {formatted} — {method_label}{run_text}")

    def _on_alpha_spin_edited(self) -> None:
        """A hand-edit of the single alpha spin drops calibration provenance."""
        if getattr(self, "_suppress_alpha_provenance_reset", False):
            return
        self._alpha_estimate_state.pop("single", None)
        self._alpha_estimate_state.pop("P_z", None)
        self._refresh_alpha_provenance_label()

    def _reseed_alpha_provenance_from_grouping(self, grouping: dict[str, Any]) -> None:
        """Re-seed ``_alpha_estimate_state['single']`` from a resolved payload.

        A profile whose alpha policy is ``calibrated`` resolves to a payload
        carrying ``alpha_reference_run`` (and optionally ``alpha_error``); lifting
        it back into the estimate state lets the provenance label show the
        calibration on reopen instead of falling back to "manual".
        """
        reference = grouping.get("alpha_reference_run")
        if reference is None:
            return
        try:
            run = int(reference)
        except (TypeError, ValueError):
            return
        error = grouping.get("alpha_error")
        try:
            error_val = None if error is None else float(error)
        except (TypeError, ValueError):
            error_val = None
        self._alpha_estimate_state["single"] = (
            float(self._alpha_spin.value()),
            error_val,
            run,
        )

    def _refresh_alpha_provenance_label(self) -> None:
        """Reflect the single alpha's provenance ("calibrated …" or "manual")."""
        if not hasattr(self, "_alpha_provenance_label"):
            return
        recorded = self._alpha_estimate_state.get("single") or self._alpha_estimate_state.get("P_z")
        value = float(self._alpha_spin.value())
        if recorded is not None and abs(recorded[0] - value) < 1e-9:
            method_label = next(
                (
                    label
                    for label, key, _ in _ALPHA_METHOD_ITEMS
                    if key == self._current_alpha_method()
                ),
                self._current_alpha_method(),
            )
            formatted = _format_value_with_uncertainty(recorded[0], recorded[1])
            run = recorded[2]
            run_text = f" · run {run}" if run is not None and run >= 0 else ""
            self._alpha_provenance_label.setText(f"α = {formatted} · {method_label}{run_text}")
        else:
            self._alpha_provenance_label.setText("manual")

    def _estimate_alpha_for_axis(self, axis: str) -> None:
        """Launch the calibration dialog for one projection pair.

        *axis* is the projection slot/label — a canonical EMU axis (P_x/P_y/P_z)
        or a non-canonical projection label (FB, Top-Bottom, …). The dialog
        preselects that projection's forward/backward groups and writes the
        calibrated value + provenance back into the projection's alpha spin (and,
        for the canonical P_z anchor, the single-alpha spin too).
        """
        pair = self._vector_axis_pairs.get(axis)
        if pair is None:
            QMessageBox.warning(
                self, "Estimate Failed", f"No grouping pair is available for {axis}."
            )
            return
        spin = self._vector_alpha_spins.get(axis)
        if spin is None:
            return
        self._launch_calibration_dialog(axis, int(pair[0]), int(pair[1]), spin, slot_label=axis)

    def _estimate_all_alpha(self) -> None:
        """Estimate alpha for every projection pair in the current table."""
        for axis in self._ordered_projection_labels(
            self._vector_axis_pairs, self._is_canonical_vector_pairs(self._vector_axis_pairs)
        ):
            self._estimate_alpha_for_axis(axis)

    def _reference_has_two_period_data(self) -> bool:
        """Return True when the reference run contains two-period histograms."""
        if self._run is None:
            return False
        grouping = self._run.grouping if isinstance(self._run.grouping, dict) else {}
        period_histograms = grouping.get("period_histograms")
        if isinstance(period_histograms, list) and len(period_histograms) == 2:
            return True
        try:
            return int(self._run.metadata.get("period_count", 1)) == 2
        except (TypeError, ValueError):
            return False

    def _update_period_mode_visibility(self) -> None:
        """Show RG controls only for two-period reference datasets."""
        has_two_period = self._reference_has_two_period_data()
        self._period_mode_label.setVisible(has_two_period)
        self._period_mode_widget.setVisible(has_two_period)
        self._period_mode_widget.setEnabled(has_two_period)

    def _set_period_mode(self, mode_key: str) -> None:
        """Select a period-mode radio button, defaulting to RED."""
        btn = self._period_mode_buttons.get(str(mode_key))
        if btn is None:
            btn = self._period_mode_buttons[str(PeriodMode.RED)]
        btn.setChecked(True)

    def _current_period_mode(self) -> str:
        """Return key for the selected period mode."""
        for key, btn in self._period_mode_buttons.items():
            if btn.isChecked():
                return key
        return str(PeriodMode.RED)

    def _current_grouping_payload(self) -> dict[str, Any]:
        """Build the current grouping payload from UI controls."""
        forward_gid = int(self._forward_combo.currentData())
        backward_gid = int(self._backward_combo.currentData())
        t0_bin, t_good_offset, first_good_bin, last_good_bin = (
            self._resolve_good_bin_limits_from_controls()
        )
        deadtime_payload = self._resolve_deadtime_payload() or {
            "deadtime_correction": False,
            "deadtime_mode": "off",
        }
        # The canonical EMU axes drive the base alpha (P_z) and dedicated
        # alpha_x/y/z keys. Non-canonical projections persist their alpha inside
        # the projections payload instead (see :meth:`_projection_payload`), so
        # the base alpha they fall back to stays the single-alpha control.
        canonical = self._is_canonical_vector_pairs(self._vector_axis_pairs)
        if canonical and "P_z" in self._vector_axis_pairs:
            forward_gid, backward_gid = self._vector_axis_pairs["P_z"]
            self._set_combo_to_group(self._forward_combo, int(forward_gid))
            self._set_combo_to_group(self._backward_combo, int(backward_gid))

        alpha_value = float(self._alpha_spin.value())
        if canonical and "P_z" in self._vector_alpha_spins:
            alpha_value = float(self._vector_alpha_spins["P_z"].value())

        # Attach estimate provenance only while the spin still holds the
        # value the estimator produced (a manual edit invalidates it).
        alpha_provenance: dict[str, Any] = {"alpha_method": self._current_alpha_method()}
        slot = "P_z" if canonical else "single"
        recorded = self._alpha_estimate_state.get(slot)
        if recorded is not None and abs(recorded[0] - alpha_value) < 1e-9:
            if recorded[1] is not None:
                alpha_provenance["alpha_error"] = float(recorded[1])
            alpha_provenance["alpha_reference_run"] = int(recorded[2])

        return (
            {
                "groups": {
                    gid: [idx + 1 for idx in values] for gid, values in self._groups.items()
                },
                "included_groups": self._current_included_groups(),
                "group_names": dict(self._group_names),
                "grouping_preset": self._grouping_preset_name,
                "instrument": self._detector_layout_instrument_name,
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "forward_indices": list(self._groups.get(forward_gid, [])),
                "backward_indices": list(self._groups.get(backward_gid, [])),
                "alpha": alpha_value,
                "t0_bin": int(t0_bin),
                "t_good_offset": int(t_good_offset),
                "first_good_bin": int(first_good_bin),
                "last_good_bin": int(last_good_bin),
                "bunching_factor": int(self._bunch_spin.value()),
                "background_correction": self._current_background_mode() != "none",
                "background_mode": self._current_background_mode(),
                "period_mode": self._current_period_mode(),
                "bin_index_base": self._bin_index_base(),
            }
            | self._binning_payload()
            | self._exclusion_payload()
            | (
                {"background_run": dict(self._background_run_payload)}
                if self._current_background_mode() == "reference_run"
                and self._background_run_payload
                else {}
            )
            | alpha_provenance
            | (self._vector_alpha_payload() if canonical else {})
            # Always emit projections (empty when none) so the apply path can
            # distinguish "no projections" from "key omitted / don't touch".
            # Non-canonical projections carry their edited per-projection alpha.
            | {"projections": self._projection_payload(canonical)}
            | deadtime_payload
        )

    def _projection_payload(self, canonical: bool) -> list[dict[str, Any]]:
        """Return the projections list, injecting per-projection alpha edits.

        For non-canonical projections (GPS WEP's FB/UD, the MuSR/HiFi transverse
        pairs) each row's α spin is written back into that projection's ``alpha``,
        which :meth:`MainWindow._resolve_vector_alpha_values` consumes. When the
        spin still holds an estimate (a manual edit invalidates it), the estimate
        provenance — ``alpha_error`` and ``alpha_reference_run`` — is attached to
        the projection too, mirroring the canonical per-axis provenance in
        :meth:`_vector_alpha_payload`; a manual edit drops any stale provenance
        carried on the spec. The canonical EMU axes keep their alpha (and
        provenance) in the dedicated ``alpha_x/y/z`` keys, so their projection
        dicts are passed through untouched.
        """
        specs = self._projection_specs or []
        result: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            new_spec = dict(spec)
            label = str(new_spec.get("label"))
            if not canonical and label in self._vector_alpha_spins:
                value = float(self._vector_alpha_spins[label].value())
                new_spec["alpha"] = value
                recorded = self._alpha_estimate_state.get(label)
                if recorded is not None and abs(recorded[0] - value) < 1e-9:
                    if recorded[1] is not None:
                        new_spec["alpha_error"] = float(recorded[1])
                    new_spec["alpha_reference_run"] = int(recorded[2])
                else:
                    # A manual edit (or no estimate) invalidates stale provenance.
                    new_spec.pop("alpha_error", None)
                    new_spec.pop("alpha_reference_run", None)
            result.append(new_spec)
        return result

    def _vector_alpha_payload(self) -> dict[str, Any]:
        """Per-axis alpha values plus estimate provenance for vector mode.

        Mirrors the single-axis provenance (error + reference run) for each
        canonical axis — ``alpha_x_error`` / ``alpha_x_reference_run`` … — which
        Phase 1 recorded only for the scalar ``P_z`` slot. Provenance is
        attached only while the spin still holds the estimator's value (a manual
        edit invalidates it), matching the scalar path.
        """
        payload: dict[str, Any] = {}
        for axis, key in (("P_x", "alpha_x"), ("P_y", "alpha_y"), ("P_z", "alpha_z")):
            if axis not in self._vector_alpha_spins:
                continue
            value = float(self._vector_alpha_spins[axis].value())
            payload[key] = value
            recorded = self._alpha_estimate_state.get(axis)
            if recorded is not None and abs(recorded[0] - value) < 1e-9:
                if recorded[1] is not None:
                    payload[f"{key}_error"] = float(recorded[1])
                payload[f"{key}_reference_run"] = int(recorded[2])
        return payload

    def _binning_payload(self) -> dict[str, Any]:
        """Binning-mode keys: only non-fixed modes carry the width knobs."""
        mode = self._current_binning_mode()
        if mode == "fixed":
            return {}
        payload: dict[str, Any] = {
            "binning_mode": mode,
            "bin0_us": float(self._bin0_spin.value()),
        }
        if mode == "variable":
            payload["bin10_us"] = float(self._bin10_spin.value())
        return payload

    def _exclusion_payload(self) -> dict[str, Any]:
        # Always present: an empty list explicitly clears exclusions, so the
        # apply path never has to guess between "cleared" and "unspecified".
        ids = self._current_excluded_detectors() or []
        return {"excluded_detectors": [int(v) for v in ids]}

    def _save_grp_file(self) -> None:
        """Save current grouping configuration to a ``.grp`` file."""
        payload = self._current_grouping_payload()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Grouping",
            "grouping.grp",
            "Grouping files (*.grp);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".grp"):
            path += ".grp"

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.serialize_grp(payload))

    def _load_grp_file(self) -> None:
        """Load grouping configuration from a ``.grp`` file into the dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Grouping",
            "",
            "Grouping files (*.grp);;All files (*)",
        )
        if not path:
            return

        with open(path, encoding="utf-8") as handle:
            payload = self.parse_grp(handle.read())

        loaded_index_base = (
            1 if int(payload.get("bin_index_base", self._bin_index_base())) == 1 else 0
        )
        if isinstance(self._run.grouping, dict):
            self._run.grouping["bin_index_base"] = loaded_index_base

        loaded_groups = payload.get("groups", {})
        if not isinstance(loaded_groups, dict) or len(loaded_groups) < 2:
            QMessageBox.warning(
                self, "Invalid Grouping", "Loaded .grp file does not define at least two groups."
            )
            return

        groups: dict[int, list[int]] = {}
        for key, dets in loaded_groups.items():
            gid = int(key)
            idxs = [max(0, int(v) - 1) for v in dets]
            groups[gid] = sorted(set(idxs))

        self._groups = groups
        loaded_included_groups = payload.get("included_groups", {})
        if isinstance(loaded_included_groups, dict):
            self._included_groups = {
                int(k): bool(v) for k, v in loaded_included_groups.items() if str(k).strip()
            }
        else:
            self._included_groups = {int(gid): True for gid in groups}
        self._populate_group_table()
        self._refresh_group_combo_items(
            forward_gid=int(payload.get("forward_group", 1)),
            backward_gid=int(payload.get("backward_group", 2)),
        )

        loaded_group_names = payload.get("group_names", {})
        if isinstance(loaded_group_names, dict):
            self._group_names = {int(k): str(v) for k, v in loaded_group_names.items()}
            self._refresh_group_combo_items(
                forward_gid=int(payload.get("forward_group", 1)),
                backward_gid=int(payload.get("backward_group", 2)),
            )
        # Adopt any projections the .grp declared so the per-projection alpha
        # table (rebuilt by _update_vector_mode_controls below) reflects the
        # loaded grouping; a .grp without projections clears them.
        loaded_projections = payload.get("projections")
        if isinstance(loaded_projections, list) and loaded_projections:
            self._projection_specs = [dict(p) for p in loaded_projections if isinstance(p, dict)]
        else:
            self._projection_specs = None

        self._alpha_spin.setValue(float(payload.get("alpha", 1.0)))
        # Seed the canonical axis spins when present; the table is rebuilt (and
        # re-seeded) by _update_vector_mode_controls below, but only the
        # canonical rows exist as P_x/P_y/P_z slots.
        if ("alpha_x" in payload or "alpha_px" in payload) and "P_x" in self._vector_alpha_spins:
            self._vector_alpha_spins["P_x"].setValue(
                float(payload.get("alpha_x", payload.get("alpha_px", payload.get("alpha", 1.0))))
            )
        if ("alpha_y" in payload or "alpha_py" in payload) and "P_y" in self._vector_alpha_spins:
            self._vector_alpha_spins["P_y"].setValue(
                float(payload.get("alpha_y", payload.get("alpha_py", payload.get("alpha", 1.0))))
            )
        if ("alpha_z" in payload or "alpha_pz" in payload) and "P_z" in self._vector_alpha_spins:
            self._vector_alpha_spins["P_z"].setValue(
                float(payload.get("alpha_z", payload.get("alpha_pz", payload.get("alpha", 1.0))))
            )

        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = loaded_index_base
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._last_good_spin.setRange(index_base, max_bin + index_base)

        t0_bin = int(payload.get("t0_bin", self._t0_spin.value() - index_base))
        t0_bin = max(0, min(max_bin, t0_bin))
        self._t0_spin.setValue(t0_bin + index_base)
        self._on_t0_changed()

        raw_offset = payload.get("t_good_offset")
        if raw_offset is None:
            try:
                raw_offset = int(payload.get("first_good_bin", t0_bin)) - t0_bin
            except (TypeError, ValueError):
                raw_offset = self._t_good_offset_spin.value()
        t_good_offset = max(0, int(raw_offset))
        t_good_offset = min(t_good_offset, int(self._t_good_offset_spin.maximum()))
        self._t_good_offset_spin.setValue(t_good_offset)

        last_good_bin = int(payload.get("last_good_bin", self._last_good_spin.value() - index_base))
        last_good_bin = max(0, min(max_bin, last_good_bin))
        self._last_good_spin.setValue(last_good_bin + index_base)
        self._bunch_spin.setValue(int(payload.get("bunching_factor", self._bunch_spin.value())))
        self._deadtime_manual_values_us = self._initial_manual_deadtime_values(payload)
        self._deadtime_manual_method = self._initial_manual_deadtime_method(payload)
        self._deadtime_mode = (
            self._default_deadtime_mode(payload)
            if bool(payload.get("deadtime_correction", False))
            else "off"
        )
        self._deadtime_estimated_us = (
            float(payload["deadtime_estimated_us"])
            if payload.get("deadtime_estimated_us") is not None
            else None
        )
        self._deadtime_source_run = (
            int(payload["deadtime_reference_run"])
            if payload.get("deadtime_reference_run") is not None
            else None
        )
        self._update_deadtime_status()
        background_run_payload = payload.get("background_run")
        self._background_run_payload = (
            dict(background_run_payload) if isinstance(background_run_payload, dict) else None
        )
        self._background_mode = (
            resolve_background_mode(payload)
            if bool(payload.get("background_correction", False))
            else "none"
        )
        self._update_background_status()
        self._set_period_mode(str(payload.get("period_mode", PeriodMode.RED)))
        self._populate_group_table()
        self._update_vector_mode_controls(payload)

    @staticmethod
    def serialize_grp(payload: dict[str, Any]) -> str:
        """Serialize grouping payload to text ``.grp`` format.

        Thin delegate to :func:`asymmetry.gui.windows.grouping.grp_io.serialize_grp`;
        kept as a static method so the historical ``GroupingDialog.serialize_grp``
        call sites (and tests) keep working after the package split.
        """
        return _serialize_grp(payload)

    @staticmethod
    def parse_grp(text: str) -> dict[str, Any]:
        """Parse line-based ``.grp`` text into a grouping payload dictionary.

        Thin delegate to :func:`asymmetry.gui.windows.grouping.grp_io.parse_grp`.
        """
        return _parse_grp(text)

    def get_grouping_result(self) -> dict[str, Any] | None:
        """Return the applied grouping payload for inheriting runs.

        The dialog is now a profile editor: on Apply the draft is saved as the
        active profile and resolved onto every *inheriting* run of the
        fingerprint. For backward compatibility the returned dict is still the
        flat grouping payload the main window's ``_apply_grouping_settings_to_dataset``
        consumes, with ``run_numbers`` set to the inheriting runs (released runs
        are excluded, so overridden runs are left untouched).

        Returns
        -------
        dict or None
            Grouping payload plus ``run_numbers`` (inheriting runs). ``None``
            when no run histograms are available.
        """
        if self._run is None or not self._run.histograms:
            return None
        payload = self._current_grouping_payload()
        payload["run_numbers"] = sorted(self._scope_panel.inheriting_run_numbers())
        return payload

    def get_profile_result(self) -> dict[str, Any] | None:
        """Return the profile-editor result for the main window to reconcile.

        Returns
        -------
        dict or None
            ``{"profile": GroupingProfile, "inheriting": set[int],
            "released": set[int], "newly_released": set[int],
            "newly_reattached": set[int], "preview_run_number": int}`` — the
            saved draft profile and the scope reconciliation. ``None`` when no
            run is available.
        """
        if self._run is None or not self._run.histograms:
            return None
        # The draft was synced by _on_apply; re-sync defensively in case a caller
        # reads this without going through Apply.
        self._sync_draft_from_form()
        return {
            "profile": GroupingProfile.from_dict(self._draft.to_dict()),
            "inheriting": self._scope_panel.inheriting_run_numbers(),
            "released": self._scope_panel.released_run_numbers(),
            "newly_released": self._scope_panel.newly_released(),
            "newly_reattached": self._scope_panel.newly_reattached(),
            "preview_run_number": int(self._reference_dataset.run_number),
        }

    @property
    def draft_profile(self) -> GroupingProfile:
        """The current draft profile (a copy synced from the form)."""
        self._sync_draft_from_form()
        return GroupingProfile.from_dict(self._draft.to_dict())
