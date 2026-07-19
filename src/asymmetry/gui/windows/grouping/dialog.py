"""Shared grouping profile editor with alpha estimation and live preview.

The dialog edits the grouping profile for an instrument (and per-run overrides
for released runs) and applies the changes across the active project's
datasets.

This module holds the :class:`GroupingDialog` shell. The alpha-estimate display
helpers live in :mod:`asymmetry.gui.windows.grouping.format`; the module-level
``_format_value_with_uncertainty``/``_ALPHA_METHOD_ITEMS`` names alias that
module so the historical module API is unchanged after the package split.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
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
    variant_for_histograms,
)
from asymmetry.core.io import resolve_background_reference
from asymmetry.core.project.profiles import (
    AlphaPolicy,
    BackgroundPolicy,
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    profile_fingerprint_for_run,
    resolve_effective_grouping,
)
from asymmetry.core.transform import (
    available_background_modes,
    calibrate_deadtime_from_histograms,
    common_t0_for_groups,
    estimate_deadtime_from_histograms,
    excluded_detector_indices,
    filter_excluded_indices,
    find_t0_for_run,
    format_detector_list,
    parse_detector_list,
    resolve_background_mode,
    resolve_binning_mode,
)
from asymmetry.core.utils.constants import PeriodMode
from asymmetry.gui.styles import metrics, tokens
from asymmetry.gui.styles.widgets import (
    apply_param_table_style,
    build_stage_chip_qss,
    clear_layout,
    make_section_header,
    make_warning_banner,
)
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.no_scroll_spin import (
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSpinBox,
)
from asymmetry.gui.widgets.section_overflow_indicator import SectionOverflowIndicator
from asymmetry.gui.windows.grouping.alpha_section import (
    AlphaEstimateResult,
    AlphaSectionWidget,
    build_alpha_request,
    populate_calibration_run_combo,
    run_alpha_estimate,
)
from asymmetry.gui.windows.grouping.background_section import (
    BackgroundReferenceRunCandidate,
    BackgroundSectionWidget,
    background_status_text,
)
from asymmetry.gui.windows.grouping.beta_section import (
    BetaSectionWidget,
    beta_status_text,
)
from asymmetry.gui.windows.grouping.correction_card import CorrectionCard
from asymmetry.gui.windows.grouping.deadtime_section import (
    DeadtimeSectionWidget,
    DeadtimeSourceRun,
    deadtime_status_text,
)
from asymmetry.gui.windows.grouping.format import (
    ALPHA_METHOD_ITEMS as _ALPHA_METHOD_ITEMS,
)
from asymmetry.gui.windows.grouping.format import (
    format_value_with_uncertainty as _format_value_with_uncertainty,
)
from asymmetry.gui.windows.grouping.preview_pane import GroupingPreviewPane
from asymmetry.gui.windows.grouping.profile_bridge import (
    instrument_display_for_fingerprint,
    payload_from_profile_for_preview,
    payload_matches_preset,
    preset_payload,
    profile_from_form_payload,
)
from asymmetry.gui.windows.grouping.scope_panel import ScopePanel

#: Compare-pager cycle order (`_step_compare`/`_sync_compare_pager`). ``None``
#: ("off") is always available; the rest are gated by `_compare_stage_available`.
_COMPARE_CYCLE: tuple[str | None, ...] = (None, "deadtime", "background", "alpha", "beta", "raw")

#: Stage identity colours (chip outline = card stripe; see ``tokens.STAGE_*``).
_STAGE_COLORS: dict[str, tuple[str, str]] = {
    "deadtime": (tokens.STAGE_DEADTIME, tokens.STAGE_DEADTIME_SOFT),
    "background": (tokens.STAGE_BACKGROUND, tokens.STAGE_BACKGROUND_SOFT),
    "alpha": (tokens.STAGE_ALPHA, tokens.STAGE_ALPHA_SOFT),
    "beta": (tokens.STAGE_BETA, tokens.STAGE_BETA_SOFT),
}

#: Pager label text per focused stage (see `_sync_compare_pager`).
_COMPARE_STAGE_LABELS: dict[str, str] = {
    "deadtime": "without deadtime",
    "background": "without background",
    "alpha": "α = 1",
    "beta": "β = 1",
    "raw": "vs raw",
}

#: Correction-card compare indicator per focused stage (`_sync_correction_cards`).
#: "raw" is deliberately absent: the compound compare is not one stage's card.
_CARD_COMPARING_TEXTS: dict[str, str] = {
    "deadtime": "comparing: without deadtime",
    "background": "comparing: without background",
    "alpha": "comparing: α = 1 ghost",
    "beta": "comparing: β = 1 ghost",
}

#: Card status = the pipeline-chip summary minus the prefix that would
#: duplicate the card title ("Deadtime" + "off", not "Deadtime" + "Deadtime: off").
_CARD_STATUS_PREFIXES: dict[str, str] = {
    "deadtime": "Deadtime: ",
    "background": "Background: ",
    "alpha": "α = ",
    "beta": "β = ",
}


class GroupingDialog(QDialog):
    """Profile editor for detector grouping.

    The dialog edits an in-memory *draft* grouping profile (a
    :class:`~asymmetry.core.project.profiles.GroupingProfile`). Each run of a
    fingerprint ``(instrument, histogram_count)`` follows its *assigned*
    profile — several profiles can be in concurrent use (schema v17, e.g. one
    per sample), with one flagged the fingerprint's default for newly loaded
    runs. The scope panel moves runs between profiles ("Assign to ▸");
    per-run divergence is an explicit "release from profile" exception, with
    the assignment kept as the base profile Reattach returns the run to.
    Applying the edited profile reaches only the runs following it. Nothing
    touches the project or runs until Apply.

    The scope panel is the **selector**: the run selected there is the one the
    form previews and edits, and the *editing target* follows that run's status.
    An inheriting run edits the profile draft (through a flat ``run.grouping``
    *payload* — see :mod:`asymmetry.gui.windows.grouping.profile_bridge`); an
    overridden run edits that run's own override draft, seeded once from its
    stored payload and kept in ``self._override_drafts``. Selecting never
    prompts — each target keeps its draft — and **Apply** commits everything
    dirty: the profile to every inheriting run plus each edited override to its
    own run. The only guard is closing the window with uncommitted changes.

    Parameters
    ----------
    datasets
        Datasets available in the active project. Datasets without raw
        histograms are ignored.
    profiles
        The project's existing grouping profiles. When omitted (or empty for the
        current fingerprint), the draft is synthesized from the current/reference
        run's payload as ``"Default (<instrument>)"``.
    assigned_profiles
        Optional ``{run_number: profile_name}`` map of each run's recorded
        profile assignment (schema v17). Runs without an entry — or whose
        entry no longer names a profile of their fingerprint — are treated as
        following the fingerprint's default profile.
    selected_run_number
        Optional run number used as the initially selected run.
    selected_run_numbers
        Optional run numbers; only the first choice steers the initial selected
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
        assigned_profiles: dict[int, str] | None = None,
        selected_run_number: int | None = None,
        selected_run_numbers: list[int] | None = None,
        parent=None,
    ) -> None:
        """Create a grouping profile editor for project datasets."""
        super().__init__(parent)
        self._datasets = [ds for ds in datasets if ds.run is not None]
        self._project_profiles = list(profiles or [])
        self._overridden_run_numbers = {int(v) for v in (overridden_run_numbers or [])}
        self._assigned_profiles = {
            int(rn): str(name) for rn, name in (assigned_profiles or {}).items()
        }
        # Run→profile assignment maps (schema v17). ``_initial_assignments`` is
        # each run's assignment resolved when it is first listed; ``_session_
        # assignments`` is the authoritative working map. It deliberately
        # survives profile switches and scope-panel repopulates — assigning
        # runs to a profile and then switching the combo to edit that profile
        # must not lose the assignment.
        self._initial_assignments: dict[int, str] = {}
        self._session_assignments: dict[int, str] = {}
        #: Stored profile names deleted this session (committed on Apply).
        self._deleted_profiles: list[str] = []
        #: Names of profiles created this session (New…/Duplicate…). They are
        #: registered in ``self._project_profiles`` immediately so they persist
        #: in the selector and Assign-to menu; Apply commits them, Cancel
        #: drops them with the dialog.
        self._session_created: list[str] = []
        #: The stored name of the draft before an in-session rename, if any.
        self._renamed_from: str | None = None
        #: Pending per-fingerprint default-profile choice, keyed by fingerprint
        #: (survives instrument switches within the session).
        self._default_names: dict[tuple[str, int], str] = {}
        self._selected_run_number = selected_run_number
        self._selected_run_numbers = (
            {int(v) for v in selected_run_numbers} if selected_run_numbers is not None else None
        )
        # Draft-editing state (set up once the reference/preview run is known).
        self._draft_dirty = False
        self._suppress_dirty = False
        self._draft_name = ""
        self._fingerprint: ProfileFingerprint | None = None
        # Unified editing model: the scope-panel selection is the selector, and
        # the editing target follows the selected run's status. An inheriting run
        # edits the profile draft (``self._draft``); an overridden run edits that
        # run's own override draft, kept in ``self._override_drafts`` keyed by run
        # number. Override drafts accumulate across selections — switching never
        # prompts; each target keeps its draft — and Apply commits every dirty
        # one alongside the profile. ``_current_run`` mirrors the selected run so
        # the target can be resolved without touching the scope panel mid-reseed.
        self._override_drafts: dict[int, dict[str, Any]] = {}
        #: overridden runs whose override draft has uncommitted edits this session.
        self._override_dirty_runs: set[int] = set()
        #: overridden runs whose edits were committed by the last Apply (read by
        #: get_profile_result after the dirty set is cleared).
        self._committed_override_runs: set[int] = set()
        self._current_run: int | None = None

        # Off-thread worker for the background-configure preview grouping (the
        # preview pane owns its own runner for the live-preview reduction; this
        # one is the dialog's). Created before the "no runs" early return below
        # so ``done``/``closeEvent`` can unconditionally shut it down.
        self._tasks = TaskRunner(self)

        self.setWindowTitle("Grouping")
        # Wide enough that the grouping and corrections columns sit side by side
        # (the pipeline strip and section headers fit without horizontal
        # scrolling) and tall enough that both columns' default (deadtime-off)
        # state needs no vertical scrolling either.
        self.resize(1220, 680)

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
        #: The most recent resolved seed payload (set by :meth:`_seed_source`).
        #: The read-only auto-detect t0 display reads the consensus t0 the
        #: resolve already computed from here instead of re-scanning every
        #: detector a second time (see :meth:`_seed_t0_spin_from_detection`).
        self._last_resolved_seed: dict[str, Any] | None = None
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
        # Inline per-projection α estimate (replaces the retired modal). Estimates
        # run off-thread on this runner; "Estimate All" serialises them one axis at
        # a time (the queue below), so each result routes to its own spin.
        self._vector_alpha_tasks = TaskRunner(self)
        self._vector_estimate_token = 0
        self._vector_estimate_queue: list[str] = []
        self._vector_estimate_source_run: int | None = None
        #: Focused compare-in-preview stage ("deadtime"/"background"/"alpha"/
        #: "raw"/None) — preview-only, drives the ghost overlay (never the payload).
        self._compare_stage: str | None = None
        # Last successful estimate per slot ("single" or axis name):
        # (alpha, alpha_error, reference_run). Used to attach provenance to
        # the payload only while the spin still holds the estimated value.
        self._alpha_estimate_state: dict[str, tuple[float, float | None, int]] = {}
        # Digest of the deadtime + background settings a calibrated alpha was
        # measured under. When the corrections later change, the calibrated alpha
        # no longer balances the reduced (corrected) asymmetry — the staleness
        # banner flags that. ``None`` until an alpha is calibrated in this session
        # or re-seeded from a saved payload's ``alpha_correction_digest``.
        self._alpha_correction_digest: str | None = None

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # \u2500\u2500 Top bar: profile selector + preview run + preset \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile"))
        self._profile_combo = NoScrollComboBox()
        self._profile_combo.setMinimumContentsLength(20)
        self._rebuild_profile_combo()
        self._profile_combo.activated.connect(self._on_profile_combo_activated)
        profile_row.addWidget(self._profile_combo)

        rename_btn = QPushButton("Rename\u2026")
        rename_btn.setAutoDefault(False)
        rename_btn.setDefault(False)
        rename_btn.clicked.connect(self._on_rename_profile)
        profile_row.addWidget(rename_btn)

        delete_btn = QPushButton("Delete\u2026")
        delete_btn.setAutoDefault(False)
        delete_btn.setDefault(False)
        delete_btn.setToolTip(
            "Delete this profile. Runs assigned to it are reassigned to another "
            "profile of the instrument; the last profile cannot be deleted."
        )
        delete_btn.clicked.connect(self._on_delete_profile)
        profile_row.addWidget(delete_btn)

        # Default-for-new-runs marker (schema v17): freshly loaded runs of this
        # fingerprint are assigned to the default profile. Checking makes the
        # edited profile the default on Apply; it cannot be unchecked directly \u2014
        # defaultness moves by checking it on another profile.
        self._default_check = QCheckBox("Default for new runs")
        self._default_check.setToolTip(
            "Newly loaded runs of this instrument are assigned to the default "
            "profile (\u2605 in the selector). Check to make this profile the "
            "default on Apply; to move it, check it on another profile instead."
        )
        self._default_check.toggled.connect(self._on_default_toggled)
        profile_row.addWidget(self._default_check)
        self._refresh_default_checkbox()

        # Instrument switcher: lists every fingerprint present in the loaded
        # datasets. Hidden when the project holds a single instrument (nothing to
        # switch between); shown as "<display> \u2014 N runs" otherwise.
        self._instrument_label = QLabel("Instrument")
        self._instrument_combo = NoScrollComboBox()
        self._instrument_combo.setMinimumContentsLength(18)
        self._rebuild_instrument_combo()
        self._instrument_combo.activated.connect(self._on_instrument_combo_activated)
        profile_row.addWidget(self._instrument_label)
        profile_row.addWidget(self._instrument_combo)

        # The preview + editing target is now driven by the scope-panel
        # selection (built in the left pane below), so there is no separate
        # preview-run combo. ``_current_run`` tracks the selected run.
        self._current_run = int(self._reference_dataset.run_number)
        profile_row.addStretch()

        # Persistent editing-target strip, right-aligned in the top row so it
        # costs the right pane no vertical space: accent-tinted while editing
        # the profile ("Editing profile 'X' — applies to N runs"), warning-
        # tinted while editing a single run's override ("Editing override for
        # run N — this run only"). One of three redundant cues (with the
        # scope-list row tint and the "override *" chip).
        self._editing_strip = QLabel()
        profile_row.addWidget(self._editing_strip)
        root.addLayout(profile_row)

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
        self._scope_panel.selected.connect(self._on_scope_run_selected)
        left_layout.addWidget(self._scope_panel)
        self._refresh_scope_panel()

        # Preset row lives in the left column, directly above the group table it
        # seeds — keeping the top of the dialog to a single full-width row so the
        # right pane's tabs get the vertical space on small windows.
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.addWidget(QLabel("Preset"))
        self._preset_combo = NoScrollComboBox()
        self._preset_combo.setMinimumContentsLength(18)
        self._preset_combo.activated.connect(self._on_preset_combo_activated)
        preset_row.addWidget(self._preset_combo)
        self._preset_chip = QLabel("")
        self._preset_chip.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        preset_row.addWidget(self._preset_chip)
        preset_row.addStretch()
        left_layout.addLayout(preset_row)
        # The preset combo is populated at the end of __init__, once the
        # detector-layout resolution state (_detector_layout_instrument_name)
        # exists in the form section below.

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
        self._forward_combo = NoScrollComboBox()
        self._backward_combo = NoScrollComboBox()
        # setMinimumContentsLength sizes the combo to N characters of the
        # current font, so it tracks the UI zoom without a frozen pixel width.
        self._forward_combo.setMinimumContentsLength(18)
        self._backward_combo.setMinimumContentsLength(18)
        self._refresh_group_combo_items()

        grouping = seed.grouping
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

        self._alpha_spin = NoScrollDoubleSpinBox()
        self._alpha_spin.setDecimals(6)
        self._alpha_spin.setRange(0.01, 1000.0)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))

        max_bin = self._max_bin_index_for_reference_dataset()
        index_base = self._bin_index_base(grouping)
        default_t0_internal = self._default_t0_bin(grouping, max_bin)
        default_t_good = self._default_t_good_offset(grouping, default_t0_internal, max_bin)

        # t0 mode selector (From file / Manual / Auto-detect) — mirrors WiMDA's
        # FileValues checkbox: "From file" uses the header t0 per run and locks
        # the spinbox; "Manual" is the historical editable override; "Auto-detect"
        # runs the prompt-peak / pulse-edge search per run at resolution time.
        self._t0_mode_combo = NoScrollComboBox()
        for label, key, tooltip in (
            (
                "From file",
                "from_file",
                "Use each run's own file-derived t0 (per-detector values preserved).",
            ),
            ("Manual", "manual", "Type a common t0 override applied to every run as an offset."),
            (
                "Auto-detect",
                "auto_detect",
                "Search each run for t0 (prompt peak at continuous sources, "
                "pulse-edge midpoint at pulsed sources) at reduction time.",
            ),
        ):
            self._t0_mode_combo.addItem(label, key)
            self._t0_mode_combo.setItemData(
                self._t0_mode_combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole
            )
        self._t0_mode_combo.setMaximumWidth(130)

        self._t0_spin = NoScrollSpinBox()
        self._t0_spin.setRange(index_base, max_bin + index_base)
        self._t0_spin.setValue(default_t0_internal + index_base)

        # Provenance / per-run note shown beneath the mode selector.
        self._t0_mode_label = QLabel("")
        self._t0_mode_label.setWordWrap(True)
        self._t0_mode_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")

        self._t_good_offset_spin = NoScrollSpinBox()
        self._t_good_offset_spin.setRange(0, max_bin)
        self._t_good_offset_spin.setValue(default_t_good)
        self._t0_spin.valueChanged.connect(self._on_t0_changed)
        self._on_t0_changed()

        self._last_good_spin = NoScrollSpinBox()
        self._last_good_spin.setRange(index_base, max_bin + index_base)
        default_first_good = min(max_bin, default_t0_internal + default_t_good)
        default_last_good = int(grouping.get("last_good_bin", max_bin))
        if default_last_good < default_first_good:
            default_last_good = default_first_good
        self._last_good_spin.setValue(default_last_good + index_base)

        self._bunch_spin = NoScrollSpinBox()
        self._bunch_spin.setRange(1, 10000)
        requested_bunching = int(grouping.get("bunching_factor", 1))
        self._bunch_spin.setValue(requested_bunching)
        self._bunch_spin.setMaximumWidth(metrics.spin_width_for(5, self._bunch_spin))
        self._bunch_spin.setToolTip("Set any bunching factor >= 1.")

        self._binning_mode_combo = NoScrollComboBox()
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
        self._bin0_spin = NoScrollDoubleSpinBox()
        self._bin0_spin.setDecimals(4)
        self._bin0_spin.setRange(0.0001, 100.0)
        self._bin0_spin.setSuffix(" µs")
        self._bin10_spin = NoScrollDoubleSpinBox()
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
            "One-shot fill for Manual mode: estimate t0 from the reference run "
            "(prompt peak at continuous sources, pulse-edge midpoint at pulsed "
            "sources) into the spinbox. For a per-run search on every run, choose "
            "the Auto-detect mode instead. Nothing is applied until you press Apply."
        )
        self._find_t0_btn.clicked.connect(self._on_find_t0)

        # Deadtime and background are configured in dedicated dialogs
        # (deadtime_dialog.py / background_dialog.py); the form only shows a
        # compact status row + "Configure…" button per cluster. State lives
        # here (not in inline widgets) so ``_current_grouping_payload`` stays
        # the single payload assembly point.
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

        # Inline deadtime controls (the retired DeadtimeDialog's body) live in the
        # Corrections section; the dialog keeps the _deadtime_* state as the source
        # of truth the payload reads (the section folds edits back via `changed`).
        self._deadtime_section = DeadtimeSectionWidget()
        self._deadtime_section.changed.connect(self._on_deadtime_changed)

        payload = grouping.get("background_run")
        self._background_run_payload: dict[str, Any] | None = (
            dict(payload) if isinstance(payload, dict) else None
        )
        self._background_mode = "none"
        if bool(grouping.get("background_correction", False)):
            self._background_mode = resolve_background_mode(grouping)

        # Inline background controls (the retired BackgroundDialog's body) live in
        # the Corrections section; the dialog keeps _background_mode /
        # _background_run_payload as the source of truth the payload reads.
        self._background_section = BackgroundSectionWidget(self._background_reference_candidates)
        self._background_section.changed.connect(self._on_background_changed)

        self._period_mode_label = QLabel("RG Mode")
        self._period_mode_group = QButtonGroup(self)
        self._period_mode_buttons: dict[str, QRadioButton] = {}
        self._period_mode_widget = QWidget()
        period_layout = QHBoxLayout(self._period_mode_widget)
        period_layout.setContentsMargins(0, 0, 0, 0)
        period_layout.setSpacing(10)

        period_specs = [
            ("Red", str(PeriodMode.RED), tokens.PERIOD_RED),
            ("Green", str(PeriodMode.GREEN), tokens.PERIOD_GREEN),
            ("G minus R", str(PeriodMode.GREEN_MINUS_RED), tokens.PERIOD_DIFF),
            ("G plus R", str(PeriodMode.GREEN_PLUS_RED), tokens.PERIOD_SUM),
        ]
        for idx, (label, mode_key, color) in enumerate(period_specs):
            btn = QRadioButton(label)
            btn.setStyleSheet(f"color: {color};")
            self._period_mode_group.addButton(btn, idx)
            self._period_mode_buttons[mode_key] = btn
            period_layout.addWidget(btn)
        period_layout.addStretch()
        self._set_period_mode(str(grouping.get("period_mode", PeriodMode.RED)))

        # Single-α calibration is inline in the Corrections panel (the standalone
        # modal is kept only for the per-projection vector case). Estimate results
        # flow back into the α spin + provenance via _apply_calibrated_policy.
        self._alpha_section = AlphaSectionWidget()
        self._alpha_section.alpha_estimated.connect(self._on_alpha_section_estimated)

        # The estimation *method* is now chosen inside the calibration dialog, so
        # the inline method combo is retired from the visible row. It is kept as a
        # hidden control because the current-method key still seeds the payload's
        # ``alpha_method`` provenance (and a calibration writes the chosen method
        # back into it).
        self._alpha_method_combo = NoScrollComboBox()
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
        self._alpha_row_label = QLabel("α")

        # Width discipline for the (now narrow) grouping column: fields size to
        # their content and never stretch to fill the column. Widths derive from
        # the UI-font metrics (harness rule: no literal-pixel geometry).
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        for combo in (self._forward_combo, self._backward_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.setMaximumWidth(metrics.field_width_for(32, combo))
        for spin in (self._t0_spin, self._t_good_offset_spin, self._last_good_spin):
            spin.setMaximumWidth(metrics.spin_width_for(6, spin))
        self._exclude_edit.setMinimumWidth(metrics.field_width_for(24, self._exclude_edit))

        form.addRow(self._forward_row_label, self._forward_combo)
        form.addRow(self._backward_row_label, self._backward_combo)

        # The α value row, result and staleness banner (single mode) plus the
        # per-projection table (vector mode) live in the Corrections column's α
        # area, not the grouping form — assembled below once the vector widget
        # exists. Only the widgets are built here.
        self._single_alpha_widget = QWidget()
        alpha_row = QHBoxLayout(self._single_alpha_widget)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.addWidget(self._alpha_spin)
        alpha_row.addWidget(self._alpha_provenance_label, stretch=1)
        # Staleness banner: shown when the deadtime/background corrections change
        # after α was calibrated (α no longer centres the corrected asymmetry).
        self._alpha_stale_banner = make_warning_banner("", severity="warn")
        self._alpha_stale_banner.setVisible(False)
        # A hand-edit of the alpha spin clears calibration provenance → "manual".
        self._alpha_spin.valueChanged.connect(self._on_alpha_spin_edited)
        self._reseed_alpha_provenance_from_grouping(grouping)
        self._refresh_alpha_provenance_label()
        # Note: the initial staleness check runs at the end of __init__, once the
        # whole form exists (_correction_digest reads the full payload).

        # Vector alpha widget: one row per declared projection. The rows are
        # built dynamically (see :meth:`_rebuild_vector_alpha_table`) because the
        # projection set varies by preset — canonical EMU axes (P_z/P_y/P_x),
        # GPS WEP's FB/UD, the MuSR/HiFi transverse pairs, etc. The grid below
        # is the empty container; rows are (re)created from the current
        # projection pairs whenever they change.
        self._vector_alpha_widget = QWidget()
        vector_alpha_vbox = QVBoxLayout(self._vector_alpha_widget)
        vector_alpha_vbox.setContentsMargins(0, 0, 0, 0)
        vector_alpha_vbox.setSpacing(6)

        # One shared calibration run + method for every projection: alpha is
        # measured from the same TF calibration run and method for each axis, so
        # the per-axis Estimate buttons and "Estimate All" all use this pair with
        # the axis's own forward/backward groups (the estimate is inline — the
        # standalone modal that used to own these controls is retired).
        vector_controls = QHBoxLayout()
        vector_controls.setContentsMargins(0, 0, 0, 0)
        vector_controls.addWidget(QLabel("Calibration run"))
        self._vector_run_combo = NoScrollComboBox()
        vector_controls.addWidget(self._vector_run_combo, stretch=1)
        vector_controls.addWidget(QLabel("Method"))
        self._vector_method_combo = NoScrollComboBox()
        for label, key, explanation in _ALPHA_METHOD_ITEMS:
            self._vector_method_combo.addItem(label, key)
            self._vector_method_combo.setItemData(
                self._vector_method_combo.count() - 1, explanation, Qt.ItemDataRole.ToolTipRole
            )
        self._vector_method_combo.currentIndexChanged.connect(self._on_vector_method_changed)
        vector_controls.addWidget(self._vector_method_combo)
        vector_alpha_vbox.addLayout(vector_controls)

        vector_grid_widget = QWidget()
        self._vector_alpha_layout = QGridLayout(vector_grid_widget)
        self._vector_alpha_layout.setContentsMargins(0, 0, 0, 0)
        # 8px, not the form default 12: the α card's border + body margins spend
        # ~16px of the corrections column's width, and the 5-column grid's four
        # gaps at 12px pushed vector mode into a horizontal scrollbar at the
        # default dialog width. Four gaps at 8px hand those 16px back.
        self._vector_alpha_layout.setHorizontalSpacing(8)
        self._vector_alpha_layout.setVerticalSpacing(8)
        vector_alpha_vbox.addWidget(vector_grid_widget)

        # The α area for the Corrections column: single mode shows the α value
        # row + result + staleness banner; vector mode swaps in the per-projection
        # table in the same slot (_update_vector_mode_controls toggles visibility).
        self._alpha_area = QWidget()
        alpha_area_layout = QVBoxLayout(self._alpha_area)
        alpha_area_layout.setContentsMargins(0, 0, 0, 0)
        alpha_area_layout.setSpacing(8)
        alpha_area_form = QFormLayout()
        alpha_area_form.setVerticalSpacing(4)
        alpha_area_form.setHorizontalSpacing(12)
        alpha_area_form.addRow(self._alpha_row_label, self._single_alpha_widget)
        alpha_area_form.addRow("", self._alpha_result_label)
        alpha_area_form.addRow("", self._alpha_stale_banner)
        alpha_area_form.addRow(self._vector_alpha_widget)
        alpha_area_layout.addLayout(alpha_area_form)

        # β (intrinsic-asymmetry balance): a fixed user-entered scalar applied
        # with α in the same asymmetry formula (A = (F − αB)/(βF + αB)).
        # Scalar-only — in vector mode the whole card is hidden and the payload
        # omits the key, so per-projection reductions stay at β = 1
        # (docs/porting/beta-correction/).
        self._beta_section = BetaSectionWidget()
        self._beta_section.set_value(grouping.get("beta", 1.0))
        self._beta_section.changed.connect(self._on_beta_changed)

        # Kept as an attribute: the Grouping-column overflow pill uses this row as
        # the "t0 and binning" landmark.
        self._t0_row_widget = QWidget()
        t0_row = QHBoxLayout(self._t0_row_widget)
        t0_row.setContentsMargins(0, 0, 0, 0)
        t0_row.addWidget(self._t0_mode_combo)
        t0_row.addWidget(self._t0_spin)
        t0_row.addWidget(self._find_t0_btn)
        form.addRow("t0 Bin", self._t0_row_widget)
        form.addRow("", self._t0_mode_label)
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
        self._map_periods_btn = QPushButton("Map periods…")
        self._map_periods_btn.setAutoDefault(False)
        self._map_periods_btn.setDefault(False)
        self._map_periods_btn.setToolTip(
            "Sum arbitrary subsets of this run's periods into the red and "
            "green sets (multi-period runs)."
        )
        self._map_periods_btn.clicked.connect(self._on_map_periods)
        self.period_mapping_request: dict[str, Any] | None = None
        # Kept as an attribute: the "Periods" landmark for the Grouping-column pill.
        self._period_row_widget = QWidget()
        period_row_box = QHBoxLayout(self._period_row_widget)
        period_row_box.setContentsMargins(0, 0, 0, 0)
        period_row_box.addWidget(self._period_mode_widget)
        period_row_box.addWidget(self._map_periods_btn)
        form.addRow(self._period_mode_label, self._period_row_widget)
        self._update_map_periods_visibility()

        # The right pane is tabless: a full-width pipeline strip over a two-column
        # row — "Grouping and timing" (define what/when to group) on the left and
        # "Corrections" (correct the counts) on the right — with the compare pager
        # and the shared live preview pinned below BOTH columns. The single preview
        # keeps working across columns because it reduces from the draft's widget
        # *state* (read via `_current_grouping_payload`), never from which column is
        # focused. See docs/porting/correction-order-alpha-estimation.
        #: Compare checkboxes. Only "raw" remains (the pager-row checkbox): the
        #: per-stage compares are driven by the pipeline chips + pager and
        #: *displayed* on the focused correction card (see :meth:`_set_compare_stage`).
        self._compare_toggles: dict[str, QCheckBox] = {}

        # Pipeline strip spans the full right-pane width, above both columns.
        right_layout.addWidget(self._build_pipeline_strip())

        # ── Grouping and timing column (left) ──────────────────────────────
        # Groups, t0, binning, exclusions and periods. Its scroll sizes to its
        # content width (AdjustToContents), so the narrow column never fights the
        # corrections column for horizontal space and needs no h-scroll at rest.
        grouping_content = QWidget()
        grouping_layout = QVBoxLayout(grouping_content)
        grouping_layout.setContentsMargins(0, 0, 0, 0)
        grouping_layout.setSpacing(8)
        grouping_layout.addLayout(form)
        grouping_layout.addStretch()
        self._grouping_scroll = QScrollArea()
        self._grouping_scroll.setWidgetResizable(True)
        self._grouping_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._grouping_scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )
        self._grouping_scroll.setWidget(grouping_content)
        grouping_column = QWidget()
        # Hold the narrow column near its natural width so it never grows to
        # swallow the right pane (the AdjustToContents scroll over-reserves ~60px
        # otherwise); the corrections column (stretch 1) takes the rest. Derived
        # from the UI-font metrics so it tracks the zoom with the capped fields it
        # bounds, keeping every row inside the width with no horizontal scroll.
        grouping_column.setMaximumWidth(metrics.field_width_for(56))
        grouping_col_layout = QVBoxLayout(grouping_column)
        grouping_col_layout.setContentsMargins(0, 0, 0, 0)
        grouping_col_layout.setSpacing(2)
        grouping_col_layout.addWidget(self._build_column_header("Grouping and timing"))
        grouping_col_layout.addWidget(self._grouping_scroll)

        # ── Corrections column (right) ─────────────────────────────────────
        # Deadtime, background and α, each wrapped in a collapsible CorrectionCard
        # whose header carries the live status summary (and the compare indicator
        # while that stage's before/after ghost is focused). The α card body holds
        # the α area (single value + result + banner, or the per-projection vector
        # table) above the single-α calibration section.
        corrections_content = QWidget()
        corrections_layout = QVBoxLayout(corrections_content)
        corrections_layout.setContentsMargins(0, 0, 0, 0)
        # Tighter than the grouping form's 8px: the corrections column carries the
        # taller payload (deadtime/background/α cards), so the inter-card gap is
        # trimmed to keep its default state comfortably inside the viewport at the
        # default height (with headroom for the taller default fonts on other
        # platforms) without an outer scroll.
        corrections_layout.setSpacing(4)
        # The cards double as the section-overflow pill's landmarks.
        self._deadtime_card = CorrectionCard(
            "Deadtime", color=tokens.STAGE_DEADTIME, soft=tokens.STAGE_DEADTIME_SOFT
        )
        self._deadtime_card.set_body(self._deadtime_section)
        corrections_layout.addWidget(self._deadtime_card)
        self._background_card = CorrectionCard(
            "Background", color=tokens.STAGE_BACKGROUND, soft=tokens.STAGE_BACKGROUND_SOFT
        )
        self._background_card.set_body(self._background_section)
        corrections_layout.addWidget(self._background_card)
        self._alpha_card = CorrectionCard(
            "α (detector balance)", color=tokens.STAGE_ALPHA, soft=tokens.STAGE_ALPHA_SOFT
        )
        self._alpha_card.set_body(self._alpha_area)
        self._alpha_card.set_body(self._alpha_section)
        corrections_layout.addWidget(self._alpha_card)
        self._beta_card = CorrectionCard(
            "β (asymmetry balance)", color=tokens.STAGE_BETA, soft=tokens.STAGE_BETA_SOFT
        )
        self._beta_card.set_body(self._beta_section)
        corrections_layout.addWidget(self._beta_card)
        self._correction_cards: dict[str, CorrectionCard] = {
            "deadtime": self._deadtime_card,
            "background": self._background_card,
            "alpha": self._alpha_card,
            "beta": self._beta_card,
        }
        corrections_layout.addStretch()
        self._corrections_scroll = QScrollArea()
        self._corrections_scroll.setWidgetResizable(True)
        self._corrections_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._corrections_scroll.setWidget(corrections_content)
        corrections_column = QWidget()
        corrections_col_layout = QVBoxLayout(corrections_column)
        corrections_col_layout.setContentsMargins(0, 0, 0, 0)
        corrections_col_layout.setSpacing(2)
        corrections_col_layout.addWidget(self._build_column_header("Corrections"))
        corrections_col_layout.addWidget(self._corrections_scroll)

        self._update_deadtime_status()
        self._update_background_status()
        self._update_alpha_section()

        columns_row = QHBoxLayout()
        columns_row.setContentsMargins(0, 0, 0, 0)
        columns_row.setSpacing(8)
        columns_row.addWidget(grouping_column, stretch=0)
        columns_row.addWidget(corrections_column, stretch=1)
        right_layout.addLayout(columns_row, stretch=1)

        # Overlay pills naming the sections hidden below the fold on a short
        # window. The section lists are callables because vector mode swaps the α
        # slot and the deadtime section changes height live; _update_vector_mode_
        # controls / the visibility updaters call refresh() after those flips.
        self._corrections_overflow = SectionOverflowIndicator(
            self._corrections_scroll, self._corrections_overflow_sections
        )
        self._grouping_overflow = SectionOverflowIndicator(
            self._grouping_scroll, self._grouping_overflow_sections
        )

        # Compare pager: ◀/▶ + a muted label that step `_compare_stage` through
        # the configured corrections, directly above the preview so it works from
        # either column (the preview is pinned below both). Pure wrapper over the
        # same `_set_compare_stage` the section toggles and pipeline chips drive.
        right_layout.addWidget(self._build_compare_pager())

        # Live asymmetry preview of the preview run under the current draft.
        # Pinned below both columns, fixed-height so it never fights the form for
        # space; it reduces off the GUI thread (debounced) and redraws as edited.
        self._preview_pane = GroupingPreviewPane()
        right_layout.addWidget(self._preview_pane)

        splitter.addWidget(right_pane)

        splitter.setSizes([330, 860])
        root.addWidget(splitter, stretch=1)

        self._update_vector_mode_controls(grouping)

        # Correction cards open expanded iff their stage is active — an idle
        # correction costs one header row until it is needed. From here the
        # expansion is plain widget state for the dialog's lifetime (no QSettings);
        # a reseed that *activates* a stage re-expands its card
        # (:meth:`_auto_expand_active_cards`), but going inactive never collapses.
        for stage, card in self._correction_cards.items():
            card.set_expanded(self._correction_stage_active(stage))

        # ── Bottom bar: action buttons ───────────────────────────────────
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)
        cancel_btn.clicked.connect(self.reject)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setAutoDefault(False)
        self._apply_btn.setDefault(False)
        self._apply_btn.clicked.connect(self._on_apply)

        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        bottom_bar.addWidget(cancel_btn)
        bottom_bar.addWidget(self._apply_btn)
        root.addLayout(bottom_bar)

        self._update_period_mode_visibility()
        self._rebuild_preset_combo()
        self._seed_t0_mode_from_draft()
        self._connect_dirty_tracking()
        self._refresh_preset_chip(self._current_grouping_payload())
        self._update_apply_enabled()
        self._refresh_alpha_staleness()
        # Open focused on the α compare when α is already calibrated (single mode),
        # preserving the pre-tab auto-overlay; _sync_compare_toggles reflects it.
        if self._alpha_is_calibrated() and not bool(self._vector_axis_pairs):
            self._compare_stage = "alpha"
        self._sync_compare_toggles()
        self._connect_preview_refresh()
        self._refresh_preview()

        # Select the reference run in the scope panel so its row is highlighted
        # and the editing-target strip reflects the initial target. When that run
        # is itself overridden the form was already seeded from its override draft
        # by _seed_source (which resolves the target), so no re-seed is needed.
        self._scope_panel.set_current_run(int(self._reference_dataset.run_number))
        self._refresh_editing_strip()

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
        """Project profiles matching the editor's current fingerprint.

        Profiles deleted this session are filtered out — the deletion is
        committed on Apply, but the editor must stop offering them at once.
        """
        if self._fingerprint is None:
            return []
        return [
            p
            for p in self._project_profiles
            if p.fingerprint.matches(self._fingerprint) and p.name not in self._deleted_profiles
        ]

    # -- run→profile assignment + default profile (schema v17) -----------

    def _stored_names_for_fingerprint(self) -> list[str]:
        """Names of the fingerprint's stored (non-deleted) profiles."""
        return [p.name for p in self._profiles_for_fingerprint()]

    def _fingerprint_profile_names(self) -> list[str]:
        """Every assignable profile name: the stored ones plus the draft's."""
        names = self._stored_names_for_fingerprint()
        if self._draft_name and self._draft_name not in names:
            names.append(self._draft_name)
        return names

    def _fingerprint_key(self) -> tuple[str, int]:
        """Hashable key for the current fingerprint (case-folded instrument)."""
        assert self._fingerprint is not None
        return (
            str(self._fingerprint.instrument).lower(),
            int(self._fingerprint.histogram_count),
        )

    def _current_default_name(self) -> str:
        """The fingerprint's default-profile name, honouring pending changes."""
        key = self._fingerprint_key()
        pending = self._default_names.get(key)
        if pending and pending in self._fingerprint_profile_names():
            return pending
        return self._stored_default_name()

    def _stored_default_name(self) -> str:
        """The fingerprint's default per the stored ``active`` flags."""
        for profile in self._profiles_for_fingerprint():
            if profile.active:
                return profile.name
        return self._draft_name

    def _resolved_assignment(self, run_number: int) -> str:
        """A run's assigned profile name, falling back to the default.

        Resolves the recorded assignment passed at construction; a missing or
        stale name (the profile was deleted or renamed outside this run's
        record) falls back to the fingerprint's default profile — mirroring
        :func:`~asymmetry.core.project.profiles.assigned_profile_for_run`.
        """
        raw = self._assigned_profiles.get(int(run_number))
        if raw and str(raw) in self._fingerprint_profile_names():
            return str(raw)
        return self._current_default_name()

    def _newly_assigned(self) -> dict[int, str]:
        """Runs whose assignment changed this session (run → new profile)."""
        return {
            rn: name
            for rn, name in self._session_assignments.items()
            if name != self._initial_assignments.get(rn)
        }

    # -- instrument switcher (M2) ----------------------------------------

    def _project_fingerprints(self) -> list[ProfileFingerprint]:
        """Distinct fingerprints present in the loaded datasets, first-seen order.

        Each fingerprint appears once; the order follows the datasets so the
        combo is stable across rebuilds. Datasets without a run are skipped.
        """
        seen: list[ProfileFingerprint] = []
        for ds in self._datasets:
            if ds.run is None:
                continue
            fingerprint = profile_fingerprint_for_run(ds.run)
            if not any(fp.matches(fingerprint) for fp in seen):
                seen.append(fingerprint)
        return seen

    def _datasets_for_fingerprint(self, fingerprint: ProfileFingerprint) -> list[MuonDataset]:
        """Datasets whose run matches *fingerprint* (any fingerprint, not just current)."""
        out: list[MuonDataset] = []
        for ds in self._datasets:
            if ds.run is None:
                continue
            if profile_fingerprint_for_run(ds.run).matches(fingerprint):
                out.append(ds)
        return out

    def _rebuild_instrument_combo(self) -> None:
        """(Re)populate the Instrument switcher for the loaded datasets.

        Lists every distinct fingerprint as ``"<display> — N runs"``. The switcher
        (and its label) are hidden when only one instrument is present, since
        there is nothing to switch between.
        """
        fingerprints = self._project_fingerprints()
        combo = self._instrument_combo
        combo.blockSignals(True)
        combo.clear()
        for index, fingerprint in enumerate(fingerprints):
            display = instrument_display_for_fingerprint(fingerprint, fingerprints)
            n_runs = len(self._datasets_for_fingerprint(fingerprint))
            noun = "run" if n_runs == 1 else "runs"
            combo.addItem(f"{display} — {n_runs} {noun}", index)
            if self._fingerprint is not None and fingerprint.matches(self._fingerprint):
                combo.setCurrentIndex(index)
        combo.blockSignals(False)
        single = len(fingerprints) <= 1
        self._instrument_combo.setVisible(not single)
        self._instrument_label.setVisible(not single)

    def _on_instrument_combo_activated(self, index: int) -> None:
        """Switch the editor to another instrument's fingerprint.

        Like the profile switcher, an instrument change edits against the profile
        draft, so if an overridden run is selected we first switch selection to an
        inheriting run of the *current* instrument (never prompting — its override
        draft is kept). Then prompts to discard unsaved profile-draft edits, and
        swaps the fingerprint, selected run, draft, and re-seeds every form
        control, scope panel, preset list, and preview through the shared path.
        """
        combo_index = self._instrument_combo.itemData(index)
        fingerprints = self._project_fingerprints()
        if combo_index is None or not (0 <= int(combo_index) < len(fingerprints)):
            return
        target = fingerprints[int(combo_index)]
        if self._fingerprint is not None and target.matches(self._fingerprint):
            return
        self._select_inheriting_run_before_profile_change()
        if not self._confirm_discard_before_switch():
            self._select_current_instrument_in_combo()
            return
        self._switch_to_fingerprint(target)

    def _select_current_instrument_in_combo(self) -> None:
        """Restore the combo selection to the current fingerprint (after a cancel)."""
        fingerprints = self._project_fingerprints()
        combo = self._instrument_combo
        combo.blockSignals(True)
        for i, fingerprint in enumerate(fingerprints):
            if self._fingerprint is not None and fingerprint.matches(self._fingerprint):
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)

    def _switch_to_fingerprint(self, fingerprint: ProfileFingerprint) -> None:
        """Adopt *fingerprint*: swap selected run, draft, and re-seed the form."""
        datasets = self._datasets_for_fingerprint(fingerprint)
        if not datasets or datasets[0].run is None:
            return
        self._fingerprint = fingerprint
        self._reference_dataset = datasets[0]
        self._run = datasets[0].run
        self._current_run = int(datasets[0].run_number)
        # Draft: the new instrument's default profile, or a fresh default.
        self._draft = self._initial_draft()
        self._draft_name = self._draft.name
        self._draft_dirty = False
        # A pending rename belongs to the outgoing fingerprint's draft; the
        # switch already confirmed discarding uncommitted edits.
        self._renamed_from = None
        # Re-seed every form control, scope panel, preset list, and preview; the
        # scope panel is repopulated for the new instrument's runs by
        # _reseed_form_from_draft → _refresh_scope_panel below.
        self._reseed_form_from_draft()
        self._rebuild_instrument_combo()
        # Select the new instrument's first run in the scope panel (drives the
        # editing strip + preview) and reflect the target.
        self._scope_panel.set_current_run(int(self._reference_dataset.run_number))
        self._refresh_editing_strip()

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
        target = self._editing_target()
        if target != "profile":
            # An overridden run is selected: seed from that run's override draft.
            # The draft is seeded once from the run's stored payload and then
            # accumulates edits (see :meth:`_override_draft_for`), so switching
            # back to it later restores the in-progress edits, not the file.
            payload = dict(self._override_draft_for(int(target)))
        else:
            # An ``auto_detect`` t0 or ``per_run_estimate`` alpha policy makes
            # this resolve scan/sum every detector on the GUI thread (hundreds
            # of ms at HiFi scale); show a wait cursor for the duration.
            expensive = (
                self._draft.t0_policy.mode == "auto_detect"
                or self._draft.alpha_policy.mode == "per_run_estimate"
            )
            cursor = self._busy_cursor() if expensive else contextlib.nullcontext()
            with cursor:
                payload = payload_from_profile_for_preview(self._draft, self._run)

        # Cache the resolved payload so the read-only auto-detect t0 display can
        # reuse the consensus t0 (and strategy/spread) the resolve just derived,
        # rather than running a second full-detector scan on the GUI thread.
        self._last_resolved_seed = payload

        class _SeedSource:
            grouping = payload
            histograms = self._run.histograms
            metadata = self._run.metadata
            source_file = getattr(self._run, "source_file", None)

        return _SeedSource()

    # -- editing-target resolution --------------------------------------

    def _editing_target(self) -> str | int:
        """Return the current editing target: ``"profile"`` or an overridden run number.

        The selection is the selector: an inheriting run (or no selection) edits
        the profile draft; an overridden run edits that run's own override draft.
        Resolved from ``self._current_run`` against the scope panel's live
        released-run set so a release/reattach flips the target immediately.
        """
        run = self._current_run
        if run is not None and self._run_is_overridden(int(run)):
            return int(run)
        return "profile"

    def _override_draft_for(self, run_number: int) -> dict[str, Any]:
        """Return the override draft for *run_number*, seeding it once if new.

        Seeded from the run's current effective settings (its stored override
        payload) the first time the run is selected as an override target; from
        then on the accumulated draft is returned so in-progress edits survive
        switching selection away and back.
        """
        run_number = int(run_number)
        draft = self._override_drafts.get(run_number)
        if draft is None:
            dataset = next(
                (ds for ds in self._fingerprint_datasets() if int(ds.run_number) == run_number),
                None,
            )
            run = dataset.run if dataset is not None else None
            grouping = run.grouping if run is not None and isinstance(run.grouping, dict) else {}
            draft = dict(grouping)
            self._override_drafts[run_number] = draft
        return draft

    def _mark_dirty(self) -> None:
        """Record that the current editing target diverges from its open state."""
        if self._suppress_dirty:
            return
        target = self._editing_target()
        if target == "profile":
            self._draft_dirty = True
        else:
            self._sync_override_draft_from_form(int(target))
            newly_dirty = int(target) not in self._override_dirty_runs
            self._override_dirty_runs.add(int(target))
            self._scope_panel.mark_override_dirty(int(target), True)
            if newly_dirty and hasattr(self, "_apply_btn"):
                # Reflect the new override in the Apply blast-radius label.
                self._update_apply_enabled()

    def _sync_override_draft_from_form(self, run_number: int) -> None:
        """Capture the live form payload into *run_number*'s override draft."""
        self._override_drafts[int(run_number)] = self._current_grouping_payload()

    def _sync_draft_from_form(self) -> None:
        """Lift the current form payload back into the current target's payload.

        Called before the target is read (Apply, preset-chip refresh, profile
        switch). Also refreshes the preset chip / clears a stale preset marker
        when the groups have drifted from the named preset.

        When an overridden run is the editing target the profile draft is left
        untouched: the form payload is captured into that run's override draft so
        Apply can write it back to the overridden run alone.
        """
        payload = self._current_grouping_payload()
        self._refresh_preset_chip(payload)
        # Re-read the payload after a possible drift-clear so the draft does not
        # carry a stale ``grouping_preset``.
        payload = self._current_grouping_payload()
        target = self._editing_target()
        if target != "profile":
            self._override_drafts[int(target)] = payload
            return
        assert self._fingerprint is not None
        self._draft = profile_from_form_payload(
            payload,
            name=self._draft_name,
            fingerprint=self._fingerprint,
            active=True,
        )
        # The t0 mode is an explicit selector, not something to infer from the
        # per-run t0 value the form carries: set it on the draft directly.
        self._draft.t0_policy = self._current_t0_policy()

    # -- profile selector -------------------------------------------------

    def _rebuild_profile_combo(self) -> None:
        """(Re)populate the profile selector for the current fingerprint.

        The fingerprint's default profile (the one newly loaded runs are
        assigned to) is marked with a ★ prefix; item data stays the bare name.
        """
        combo = self._profile_combo
        combo.blockSignals(True)
        combo.clear()
        names = self._fingerprint_profile_names()
        default_name = self._current_default_name()
        for name in names:
            display = f"★ {name}" if name == default_name else name
            combo.addItem(display, name)
        combo.addItem("New…", "__new__")
        combo.addItem("Duplicate…", "__duplicate__")
        idx = combo.findData(self._draft_name)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        self._refresh_default_checkbox()

    def _refresh_default_checkbox(self) -> None:
        """Sync the default-for-new-runs checkbox to the edited profile.

        Checked (and locked) when the edited profile is the fingerprint's
        default — unchecking without choosing a new default is meaningless, so
        defaultness only moves by checking the box on another profile.
        """
        if not hasattr(self, "_default_check"):
            return
        is_default = self._draft_name == self._current_default_name()
        blocked = self._default_check.blockSignals(True)
        try:
            self._default_check.setChecked(is_default)
        finally:
            self._default_check.blockSignals(blocked)
        self._default_check.setEnabled(not is_default)

    def _on_default_toggled(self, checked: bool) -> None:
        """Make the edited profile the fingerprint's default on Apply."""
        if not checked:
            # Unchecking is disabled while default; a spurious signal is a no-op.
            self._refresh_default_checkbox()
            return
        self._default_names[self._fingerprint_key()] = self._draft_name
        self._mark_dirty()
        self._rebuild_profile_combo()

    def _on_profile_combo_activated(self, index: int) -> None:
        """Switch the active draft, or create a new / duplicated profile.

        Changing the profile edits the profile draft, so if an overridden run is
        currently selected we first switch selection to an inheriting run (never
        prompting — the override draft is kept). If no run inherits, the profile
        cannot be edited: restore the combo and bail.
        """
        if not self._select_inheriting_run_before_profile_change():
            self._select_profile_in_combo(self._draft_name)
            return
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
        if self._load_stored_profile_into_draft(name):
            self._draft_dirty = False

    def _load_stored_profile_into_draft(self, name: str) -> bool:
        """Switch the editor to the stored profile *name* (draft reloaded).

        Discards any in-session rename of the outgoing draft (the caller has
        already confirmed the discard where one is needed). Returns whether the
        profile was found. Callers own the ``_draft_dirty`` flag.
        """
        for profile in self._profiles_for_fingerprint():
            if profile.name == name:
                self._draft = GroupingProfile.from_dict(profile.to_dict())
                self._draft_name = str(name)
                self._renamed_from = None
                self._reseed_form_from_draft()
                return True
        return False

    def _create_new_profile(self, *, duplicate: bool) -> None:
        """Prompt for a name and create a fresh (or duplicated) profile.

        The new profile is registered in the session's working list at once —
        it stays in the selector and the Assign-to menu while other profiles
        are edited, instead of living only in the draft and vanishing on the
        next switch. Apply commits every session-created profile; Cancel
        drops them.
        """
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
        if name in self._stored_names_for_fingerprint():
            QMessageBox.warning(
                self,
                "Duplicate Profile" if duplicate else "New Profile",
                f"A profile named '{name}' already exists for this instrument.",
            )
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
        self._project_profiles.append(GroupingProfile.from_dict(self._draft.to_dict()))
        self._session_created.append(name)
        # The creation itself is registered (and guarded via the structural
        # check), so the draft starts clean; edits dirty it as usual.
        self._draft_dirty = False
        self._renamed_from = None
        self._reseed_form_from_draft()

    def _on_rename_profile(self) -> None:
        """Rename the current draft profile (first-class, committed on Apply).

        When the draft corresponds to a stored profile, the original stored
        name is remembered in ``_renamed_from`` so Apply replaces the stored
        profile in place and the main window rewrites every run's assignment —
        rather than saving a second profile under the new name.
        """
        from PySide6.QtWidgets import QInputDialog

        name, accepted = QInputDialog.getText(
            self, "Rename Profile", "Profile name:", text=self._draft_name
        )
        name = str(name).strip()
        if not accepted or not name or name == self._draft_name:
            return
        taken = [n for n in self._stored_names_for_fingerprint() if n != self._draft_name]
        if self._renamed_from in taken:
            taken.remove(self._renamed_from)
        if name in taken:
            QMessageBox.warning(
                self,
                "Rename Profile",
                f"A profile named '{name}' already exists for this instrument.",
            )
            return
        old_name = self._draft_name
        if old_name in self._session_created:
            # A session-created profile renames in place — nothing is stored
            # in the project yet, so there is no rename to record.
            assert self._fingerprint is not None
            self._session_created[self._session_created.index(old_name)] = name
            for profile in self._project_profiles:
                if profile.fingerprint.matches(self._fingerprint) and profile.name == old_name:
                    profile.name = name
        elif self._renamed_from is None and old_name in self._stored_names_for_fingerprint():
            # Chained renames keep the original stored name.
            self._renamed_from = old_name
        self._draft_name = name
        self._draft.name = name
        key = self._fingerprint_key()
        if self._default_names.get(key) == old_name or self._current_default_name() == old_name:
            self._default_names[key] = name
        # Carry every reference to the old name across the working maps.
        for mapping in (self._initial_assignments, self._session_assignments):
            for rn, assigned in list(mapping.items()):
                if assigned == old_name:
                    mapping[rn] = name
        self._scope_panel.rename_profile(old_name, name)
        self._draft_dirty = True
        self._rebuild_profile_combo()
        self._refresh_editing_strip()

    def _on_delete_profile(self) -> None:
        """Delete the edited profile, reassigning its runs to another profile.

        Guarded: the last profile of a fingerprint cannot be deleted. Runs
        assigned to the profile (including released runs based on it) move to
        a reassignment target chosen by the user; the forced moves surface as
        assignment changes and are committed on Apply.
        """
        from PySide6.QtWidgets import QInputDialog

        # The stored profile this draft corresponds to (renames pending).
        stored_name = self._renamed_from or self._draft_name
        display_name = self._draft_name
        if stored_name not in self._stored_names_for_fingerprint():
            # Unsaved draft (New/Duplicate): nothing stored to delete — offer to
            # discard it and return to an existing profile.
            others = self._stored_names_for_fingerprint()
            if not others:
                QMessageBox.warning(
                    self,
                    "Delete Profile",
                    "This is the only profile for this instrument; every "
                    "instrument keeps at least one profile.",
                )
                return
            if not self._confirm_discard_before_switch():
                return
            self._load_stored_profile_into_draft(others[0])
            self._draft_dirty = False
            return
        others = [n for n in self._stored_names_for_fingerprint() if n != stored_name]
        if not others:
            QMessageBox.warning(
                self,
                "Delete Profile",
                "This is the only profile for this instrument; every "
                "instrument keeps at least one profile.",
            )
            return
        affected = sorted(
            rn for rn, assigned in self._session_assignments.items() if assigned == display_name
        )
        if affected:
            noun = "run" if len(affected) == 1 else "runs"
            target, ok = QInputDialog.getItem(
                self,
                "Delete Profile",
                f"Profile '{display_name}' has {len(affected)} assigned {noun}. Reassign them to:",
                others,
                0,
                False,
            )
            if not ok or not target:
                return
            target = str(target)
        else:
            answer = QMessageBox.question(
                self,
                "Delete Profile",
                f"Delete profile '{display_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            target = others[0]
        if stored_name in self._session_created:
            # Never stored in the project: dropping it from the working list
            # is the whole deletion.
            assert self._fingerprint is not None
            self._session_created.remove(stored_name)
            self._project_profiles = [
                p
                for p in self._project_profiles
                if not (p.fingerprint.matches(self._fingerprint) and p.name == stored_name)
            ]
        else:
            self._deleted_profiles.append(stored_name)
        self._renamed_from = None
        for rn, assigned in list(self._session_assignments.items()):
            if assigned == display_name:
                self._session_assignments[rn] = target
        key = self._fingerprint_key()
        if self._current_default_name() == display_name:
            self._default_names[key] = target
        self._scope_panel.remove_profile(display_name, target)
        self._load_stored_profile_into_draft(target)
        # The deletion (and its forced reassignments) commit on Apply; keep the
        # dirty flag armed so the close guard covers them.
        self._draft_dirty = True
        self._update_apply_enabled()

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
        # _apply_preset_payload_to_form repopulates the group table with its
        # itemChanged signal blocked (see _populate_group_table), so the preview
        # needs one explicit refresh here rather than relying on the old
        # per-cell itemChanged storm.
        self._refresh_preview()

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
        """Repopulate the scope panel for the current fingerprint + draft.

        Assignments are seeded from the session map (first resolving each
        newly listed run's recorded assignment), so in-session reassignments
        survive profile switches and repopulates.
        """
        runs: list[tuple[int, str, bool, str]] = []
        for ds in self._fingerprint_datasets():
            rn = int(ds.run_number)
            initial = self._initial_assignments.setdefault(rn, self._resolved_assignment(rn))
            assigned = self._session_assignments.setdefault(rn, initial)
            runs.append((rn, ds.run_label, rn in self._overridden_run_numbers, assigned))
        self._scope_panel.set_runs(
            runs,
            profile_name=self._draft_name,
            profile_names=self._fingerprint_profile_names(),
        )
        if hasattr(self, "_apply_btn"):
            self._update_apply_enabled()

    def _refresh_editing_strip(self) -> None:
        """Refresh the persistent editing-target strip above the form.

        Accent-tinted while editing the profile ("Editing profile 'X' — applies
        to N runs"); warning-tinted while editing a single run's override
        ("Editing override for run N — this run only"). The run count is the only
        "applies to many" signal (owner decision).
        """
        if not hasattr(self, "_editing_strip"):
            return
        target = self._editing_target()
        if target == "profile":
            n = len(self._scope_panel.runs_following(self._draft_name))
            noun = "run" if n == 1 else "runs"
            text = f"Editing profile '{self._draft_name}' — applies to {n} {noun}"
            # Cue when the previewed run follows another profile: edits still go
            # to this draft; the preview is a what-if on that run.
            current = self._current_run
            if current is not None and not self._run_is_overridden(int(current)):
                assigned = self._scope_panel.assigned_profile(int(current))
                if assigned and assigned != self._draft_name:
                    text += f" (preview run follows {assigned})"
            self._editing_strip.setText(text)
            self._editing_strip.setStyleSheet(
                f"color: {tokens.ACCENT}; font-weight: bold; "
                f"background: {tokens.ACCENT_SOFT}; border: 1px solid {tokens.ACCENT}; "
                f"border-radius: 3px; padding: 4px;"
            )
        else:
            self._editing_strip.setText(f"Editing override for run {int(target)} — this run only")
            self._editing_strip.setStyleSheet(
                f"color: {tokens.WARN}; font-weight: bold; "
                f"border: 1px solid {tokens.WARN}; border-radius: 3px; padding: 4px;"
            )

    def _on_scope_changed(self) -> None:
        """React to a release/reattach in the scope panel.

        A release/reattach is a profile-scope decision, so it marks the profile
        draft dirty. It can also flip the current run's editing target: releasing
        the selected run turns it into an override target (seeded on demand from
        its current effective settings), and reattaching flips it back to the
        profile. Reattaching a run whose override draft has uncommitted edits
        confirms first; on cancel the reattach is undone. The form is re-seeded
        from the (new) target so what is shown matches what edits go to.
        """
        # Reattach discards pending override edits — confirm for any run that
        # went override → inheriting with a dirty override draft. On cancel,
        # re-release it (undo the reattach) and stop.
        reattached_dirty = {
            rn for rn in self._override_dirty_runs if not self._run_is_overridden(int(rn))
        }
        if reattached_dirty and not self._confirm_discard_override(sorted(reattached_dirty)):
            for rn in reattached_dirty:
                self._scope_panel.set_released(int(rn), True)
            return
        # Sync the authoritative assignment map from the panel (an Assign-to
        # action lands here through the same ``changed`` signal).
        self._session_assignments.update(self._scope_panel.assignments())
        self._draft_dirty = True
        # Drop override drafts for any run that is now inheriting.
        for rn in list(self._override_drafts):
            if not self._run_is_overridden(int(rn)):
                self._override_drafts.pop(int(rn), None)
                self._override_dirty_runs.discard(int(rn))
                self._scope_panel.mark_override_dirty(int(rn), False)
        # Re-seed the form from the target the current run now resolves to.
        self._suppress_dirty = True
        try:
            self._reload_controls_from_seed()
        finally:
            self._suppress_dirty = False
        self._refresh_editing_strip()
        self._update_apply_enabled()

    def _on_scope_run_selected(self, run_number: int) -> None:
        """Switch the previewed + edited run to the scope-panel selection.

        Selection is preview + editing target in one: the form shows (and edits
        route to) the selected run's effective settings — the profile draft for
        an inheriting run, that run's override draft for an overridden one.
        Switching never prompts: the previous target keeps its in-progress draft
        (the profile draft, or the run's entry in ``self._override_drafts``).
        """
        run_number = int(run_number)
        dataset = next(
            (ds for ds in self._fingerprint_datasets() if int(ds.run_number) == run_number),
            None,
        )
        if dataset is None or dataset.run is None:
            return
        # Capture the outgoing target's edits before moving on, so nothing is
        # lost (accumulate, never prompt).
        self._sync_draft_from_form()
        self._current_run = run_number
        self._reference_dataset = dataset
        self._run = dataset.run
        self._suppress_dirty = True
        try:
            self._reload_controls_from_seed()
        finally:
            self._suppress_dirty = False
        self._refresh_editing_strip()
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
        # single-letter name. Only these exact compact forms are accepted —
        # a leading letter alone would misclassify unrelated names ("fit",
        # "baseline") and trigger a spurious swap.
        if token.startswith(("forw", "fwd")) or "forward" in token or token in ("f", "fu", "fd"):
            return "forward"
        if token.startswith(("back", "bwd")) or "backward" in token or token in ("b", "bu", "bd"):
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

    def _set_t0_mode_combo(self, mode: str) -> None:
        """Select *mode* in the t0 mode combo without emitting change signals."""
        idx = self._t0_mode_combo.findData(mode)
        if idx < 0:
            idx = self._t0_mode_combo.findData("from_file")
        blocked = self._t0_mode_combo.blockSignals(True)
        try:
            self._t0_mode_combo.setCurrentIndex(max(0, idx))
        finally:
            self._t0_mode_combo.blockSignals(blocked)

    def _seed_t0_mode_from_draft(self) -> None:
        """Set the t0 mode combo + manual value from the draft policy, then gate."""
        policy = self._draft.t0_policy
        self._set_t0_mode_combo(policy.mode)
        if policy.mode == "manual" and policy.value is not None:
            base = self._bin_index_base()
            max_bin = self._max_bin_index_for_reference_dataset()
            value = max(0, min(max_bin, int(policy.value)))
            blocked = self._t0_spin.blockSignals(True)
            try:
                self._t0_spin.setValue(value + base)
            finally:
                self._t0_spin.blockSignals(blocked)
        self._apply_t0_mode_to_controls()

    def _current_t0_mode(self) -> str:
        """The t0 policy mode currently selected (from_file / manual / auto_detect)."""
        data = self._t0_mode_combo.currentData()
        return str(data) if data else "from_file"

    def _current_t0_policy(self) -> T0Policy:
        """Build the draft :class:`T0Policy` from the t0 mode selector + spinbox.

        Manual mode carries the spinbox value (internal, base-adjusted). The
        other modes carry no value — resolution reads each run's file / detected
        t0. Auto-detect provenance is display-only and recomputed at resolve time.
        """
        mode = self._current_t0_mode()
        if mode == "manual":
            base = self._bin_index_base()
            max_bin = self._max_bin_index_for_reference_dataset()
            value = max(0, min(max_bin, int(self._t0_spin.value()) - base))
            return T0Policy(mode="manual", value=value)
        return T0Policy(mode=mode)

    def _apply_t0_mode_to_controls(self) -> None:
        """Gate the t0 spinbox / Find button and set the provenance note per mode.

        * ``from_file`` — spinbox read-only, shows the preview run's file t0;
          note records the per-run derivation.
        * ``manual`` — spinbox editable (the historical behaviour); Find t0 fills it.
        * ``auto_detect`` — spinbox read-only, shows the preview run's detected t0
          plus the strategy / spread provenance.
        """
        mode = self._current_t0_mode()
        self._t0_spin.setReadOnly(mode != "manual")
        self._find_t0_btn.setEnabled(mode == "manual")
        if mode == "from_file":
            self._t0_mode_label.setText("t0 from each run's file")
            self._seed_t0_spin_from_preview()
        elif mode == "auto_detect":
            self._seed_t0_spin_from_detection()
        else:  # manual
            self._t0_mode_label.setText("Common t0 override applied to every run")

    def _seed_t0_spin_from_preview(self) -> None:
        """Show the preview run's file-derived common t0 in the (read-only) spin.

        Derives the value from the run's own histograms and the current
        forward/backward groups — never from the stored payload ``t0_bin``,
        which can carry a manual/override shift (e.g. in override-editing
        mode). "From file" must always display the file value, and selecting
        it must genuinely clear any stored shift on Apply.
        """
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_internal = 0
        if self._run is not None and self._run.histograms:
            forward_idx = self._filtered_group_indices(int(self._forward_combo.currentData() or 1))
            backward_idx = self._filtered_group_indices(
                int(self._backward_combo.currentData() or 2)
            )
            n_hist = len(self._run.histograms)
            forward_idx = [i for i in forward_idx if 0 <= i < n_hist]
            backward_idx = [i for i in backward_idx if 0 <= i < n_hist]
            if forward_idx or backward_idx:
                t0_internal = common_t0_for_groups(self._run.histograms, forward_idx, backward_idx)
            else:
                t0_internal = max(h.t0_bin for h in self._run.histograms)
        t0_internal = max(0, min(max_bin, int(t0_internal)))
        blocked = self._t0_spin.blockSignals(True)
        try:
            self._t0_spin.setValue(t0_internal + base)
        finally:
            self._t0_spin.blockSignals(blocked)

    def _seed_t0_spin_from_detection(self) -> None:
        """Show the auto-detected common t0 in the read-only t0 spinbox.

        Prefers the consensus the resolve already computed (cached in
        ``_last_resolved_seed`` whenever the draft was resolved under
        ``auto_detect``). Re-running :func:`find_t0_for_run` here would scan
        every detector a second time on the GUI thread — hundreds of ms at HiFi
        scale — and, because it merged the reference-dataset metadata that
        core's :func:`resolve_effective_grouping` does not, could even display a
        t0 that disagreed with the one the reduction actually uses. Only an
        explicit toggle to auto-detect (no fresh resolve in scope) falls back to
        a scan, using the same ``run.metadata`` core does so the display cannot
        diverge, under a wait cursor.
        """
        if self._run is None or not self._run.histograms:
            self._t0_mode_label.setText("Auto-detect: preview run has no histograms")
            return
        resolved = self._last_resolved_seed
        # Require ``t0_bin`` too, not just the strategy: ``_apply_t0_policy``
        # writes the strategy before its ``delta == 0`` early return but leaves
        # ``t0_bin`` unwritten there, so a run with per-detector-only t0 could
        # carry provenance without a scalar consensus. Falling through to the
        # scan then yields the right value, whereas a ``t0_bin`` default of 0
        # would silently display the wrong t0.
        if resolved is not None and resolved.get("t0_search_strategy") and "t0_bin" in resolved:
            self._apply_detected_t0_to_spin(
                consensus_t0=int(resolved["t0_bin"]),
                strategy=str(resolved["t0_search_strategy"]),
                spread_bins=int(resolved.get("t0_search_spread_bins", 0)),
            )
            return
        with self._busy_cursor():
            search = find_t0_for_run(self._run.histograms, self._run.metadata or {})
        if not search.ok:
            self._t0_mode_label.setText(f"Auto-detect: {search.message}")
            return
        self._apply_detected_t0_to_spin(
            consensus_t0=int(search.consensus_t0_bin),
            strategy=str(search.strategy),
            spread_bins=int(search.spread_bins),
        )

    def _apply_detected_t0_to_spin(
        self, *, consensus_t0: int, strategy: str, spread_bins: int
    ) -> None:
        """Write a detected common t0 (+ provenance label) into the read-only spin."""
        base = self._bin_index_base()
        blocked = self._t0_spin.blockSignals(True)
        try:
            self._t0_spin.setValue(consensus_t0 + base)
        finally:
            self._t0_spin.blockSignals(blocked)
        label = "prompt peak" if strategy == "prompt_peak" else "pulse-edge midpoint"
        self._t0_mode_label.setText(
            f"Auto-detect: {label}, detector spread {spread_bins} bins (per run)"
        )

    @contextlib.contextmanager
    def _busy_cursor(self) -> Iterator[None]:
        """Show a wait cursor for the duration of a synchronous detector scan.

        The remaining GUI-thread ``find_t0_for_run`` scans (an explicit
        auto-detect toggle, the manual **Find t0** button) are one-shot and
        unavoidable without an async loading state; a wait cursor is the honest
        signal that the app is working rather than wedged.
        """
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()

    def _on_t0_mode_changed(self, *args: object) -> None:
        """React to a t0 mode change: gate controls, then refresh the preview."""
        self._apply_t0_mode_to_controls()
        self._refresh_preview()

    def _on_manual_t0_edited(self, *args: object) -> None:
        """A spinbox edit only dirties the draft while Manual mode is active."""
        if self._current_t0_mode() == "manual":
            self._mark_dirty()

    def _resolve_good_bin_limits_from_controls(self) -> tuple[int, int, int, int]:
        """Return validated ``(t0_bin, t_good_offset, first_good_bin, last_good_bin)``."""
        max_bin = self._max_bin_index_for_reference_dataset()
        base = self._bin_index_base()
        t0_bin = max(0, min(max_bin, int(self._t0_spin.value()) - base))
        t_good_offset = max(0, int(self._t_good_offset_spin.value()))
        first_good_bin = t0_bin + t_good_offset
        last_good_bin = max(0, min(max_bin, int(self._last_good_spin.value()) - base))
        return t0_bin, t_good_offset, first_good_bin, last_good_bin

    def _run_is_overridden(self, run_number: int) -> bool:
        """Return whether *run_number* currently carries a per-run override.

        The scope panel is the live source of truth (a run released this session
        counts as overridden immediately, a reattached one no longer does).
        """
        return int(run_number) in self._scope_panel.released_run_numbers()

    def _confirm_discard_override(self, run_numbers: list[int]) -> bool:
        """Ask to discard uncommitted override edits for *run_numbers*."""
        runs = ", ".join(str(rn) for rn in run_numbers)
        answer = QMessageBox.question(
            self,
            "Discard changes",
            f"Discard uncommitted override edits for run(s) {runs}?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def _select_inheriting_run_before_profile_change(self) -> bool:
        """Ensure the current run is inheriting before a profile/instrument change.

        Profile and instrument edits target the profile draft, so they only make
        sense while an inheriting run is selected. When the selection is on an
        overridden run, switch to the first inheriting run of the fingerprint
        first (keeping the override draft intact — switching never prompts).
        Returns ``False`` when no inheriting run exists (nothing to edit against).
        """
        if self._editing_target() == "profile":
            return True
        inheriting = sorted(self._scope_panel.inheriting_run_numbers())
        if not inheriting:
            return False
        self._scope_panel.set_current_run(int(inheriting[0]))
        return True

    def _reload_controls_from_seed(self) -> None:
        """Re-seed every form control from the draft resolved for the preview run."""
        seed = self._seed_source()
        grouping = seed.grouping
        self._groups = self._load_groups(seed)
        self._populate_group_table()

        self._grouping_preset_name = (
            str(grouping.get("grouping_preset")).strip()
            if grouping.get("grouping_preset")
            else None
        )
        self._group_names = self._load_group_names(seed)
        self._included_groups = self._load_included_groups(seed)
        self._projection_specs = self._load_projection_specs(seed)
        forward_gid, backward_gid = self._analysis_pair_for_reference(
            int(grouping.get("forward_group", 1)),
            int(grouping.get("backward_group", 2)),
        )
        self._refresh_group_combo_items(forward_gid=forward_gid, backward_gid=backward_gid)
        self._alpha_spin.setValue(float(grouping.get("alpha", 1.0)))
        self._beta_section.set_value(grouping.get("beta", 1.0))
        self._set_alpha_method(str(grouping.get("alpha_method", "diamagnetic")))
        self._alpha_estimate_state.clear()
        self._alpha_correction_digest = None
        self._reseed_alpha_provenance_from_grouping(grouping)
        self._alpha_result_label.setText("")
        self._refresh_alpha_provenance_label()
        self._refresh_alpha_staleness()
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
        # Re-assert the t0 mode gating so from_file/auto_detect show the correct
        # per-run value and the spinbox read-only state survives the reseed.
        self._seed_t0_mode_from_draft()
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
        self._update_alpha_section()
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
        # The preset dropdown follows the preview run's instrument; the chip
        # follows the (possibly drifted) draft.
        if hasattr(self, "_preset_combo"):
            self._rebuild_preset_combo()
        # A reseed that activates a stage must surface its controls: expand any
        # collapsed card whose stage became active (never collapses the rest).
        self._auto_expand_active_cards()
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
        """Re-seed the inline deadtime section from the current draft state."""
        grouping = (
            self._run.grouping
            if self._run is not None and isinstance(self._run.grouping, dict)
            else {}
        )
        peak_rates, bin_width_us, good_frames = self._deadtime_peak_rates(grouping)
        reference_run_number = (
            int(self._reference_dataset.run_number) if self._reference_dataset is not None else None
        )
        self._deadtime_section.configure(
            n_detectors=len(self._run.histograms) if self._run is not None else 0,
            mode=self._deadtime_mode,
            file_values_us=self._reference_file_deadtime_values(grouping),
            manual_values_us=list(self._deadtime_manual_values_us),
            manual_method=self._deadtime_manual_method,
            estimated_us=self._deadtime_estimated_us,
            source_run=self._deadtime_source_run or reference_run_number,
            source_runs=self._deadtime_source_runs(),
            reference_run_number=reference_run_number,
            peak_rates_per_us=peak_rates,
            bin_width_us=bin_width_us,
            good_frames=good_frames,
        )

    def _deadtime_source_runs(self) -> list[DeadtimeSourceRun]:
        """Candidate source runs (of the fingerprint) for Cal/Estimate."""
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
        return source_runs

    def _deadtime_peak_rates(self, grouping: dict[str, Any]) -> tuple[list[float], float, float]:
        """Per-detector peak early-time rate, bin width and good frames (summary line)."""
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
        return peak_rates, bin_width_us, good_frames

    def _on_deadtime_changed(self) -> None:
        """Fold an inline deadtime-section edit back into the draft + preview."""
        policy = self._deadtime_section.get_policy()
        self._deadtime_mode = policy.mode if policy.mode != "from_file" else "file"
        if policy.mode == "manual":
            self._deadtime_manual_values_us = list(policy.values)
            self._deadtime_manual_method = policy.method or "manual"
            self._deadtime_source_run = policy.source_run
        elif policy.mode == "estimate":
            self._deadtime_estimated_us = policy.estimated_us
            self._deadtime_source_run = policy.source_run
        self._mark_dirty()
        self._refresh_preview()
        self._refresh_alpha_staleness()

    def _on_beta_changed(self) -> None:
        """Fold an inline β edit back into the draft + preview.

        β sits after α in the compare cycle and never affects the α estimate
        (it is invisible to count ratios), so no staleness refresh is needed —
        only the chip/card summaries and the preview (``_refresh_preview``
        syncs the compare surfaces on its way in).
        """
        self._mark_dirty()
        self._refresh_preview()

    def _background_reference_candidates(self) -> list[BackgroundReferenceRunCandidate]:
        """Reference-run candidates for the background picker (excludes the preview run)."""
        return [
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

    def _background_has_fixed_values(self) -> bool:
        """Whether the run's grouping carries stored fixed background values."""
        grouping = (
            self._run.grouping
            if self._run is not None and isinstance(self._run.grouping, dict)
            else {}
        )
        return any(
            isinstance(grouping.get(key), (list, tuple))
            for key in ("background_fixed_values", "background_fix", "bkg_fix")
        )

    def _on_background_changed(self) -> None:
        """Fold an inline background-section edit back into the draft + preview."""
        self._background_mode = self._background_section.mode()
        if self._background_mode == "reference_run":
            reference_grouping = (
                self._run.grouping
                if self._run is not None and isinstance(self._run.grouping, dict)
                else {}
            )
            payload = dict(self._background_section.background_run_payload() or {})
            # Attach the sample side's good_frames so the frame-ratio scale can be
            # shown immediately (the reduction resolves it either way).
            payload["good_frames_sample"] = reference_grouping.get("good_frames")
            self._background_run_payload = payload
        self._mark_dirty()
        self._refresh_preview()
        self._refresh_alpha_staleness()

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
        """Re-seed the inline background section from the current draft state."""
        self._background_section.configure(
            available_modes=self._available_background_modes(),
            has_fixed_values=self._background_has_fixed_values(),
            mode=self._background_mode,
            background_run_payload=self._background_run_payload,
        )

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
        # Lift the validated form state back into the current editing target
        # (the profile draft, or the selected run's override draft) so the result
        # getters return it. Snapshot the dirty overrides as *committed* so
        # get_profile_result still reports them after the dirty set is cleared,
        # then disarm the close guard.
        self._sync_draft_from_form()
        self._committed_override_runs = set(self._pending_override_runs())
        self._draft_dirty = False
        self._override_dirty_runs.clear()
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
        # The t0 mode selector is a *shareable* profile choice, so it dirties the
        # draft (unlike the per-run t0/t_good spins). A manual t0 value likewise
        # dirties the draft, but only while Manual mode is selected.
        self._t0_mode_combo.currentIndexChanged.connect(self._mark_dirty)
        self._t0_spin.valueChanged.connect(self._on_manual_t0_edited)
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
        self._t0_mode_combo.currentIndexChanged.connect(self._on_t0_mode_changed)
        self._exclude_edit.textEdited.connect(self._refresh_preview)
        self._group_table.itemChanged.connect(self._refresh_preview)
        for button in self._period_mode_buttons.values():
            button.toggled.connect(self._refresh_preview)
        # Deadtime and background live in their own dialogs; the accept paths in
        # _on_configure_deadtime/_on_configure_background call _refresh_preview()
        # directly after folding the returned policy into the draft.

    # ------------------------------------------------------------------
    # Compare-in-preview toggles + pipeline strip (preview-only, never persisted)
    # ------------------------------------------------------------------

    def _build_pipeline_strip(self) -> QWidget:
        """The ``Deadtime → group → Background → α`` pipeline overview.

        Each correction is a chip showing its one-line summary; the ``group``
        divider (non-interactive — grouping lives in the Grouping column) makes the
        reduction *order* visible. Clicking a chip focuses that stage's compare (the
        same ``_compare_stage`` the section toggles drive) and scrolls it into view.
        The α chip is hidden in vector mode, where the per-projection table owns α.
        """
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 2)
        row.setSpacing(6)
        self._pipeline_chips: dict[str, QPushButton] = {}
        row.addWidget(self._make_pipeline_chip("deadtime"))
        row.addWidget(self._pipeline_arrow())
        group = QLabel("group")
        group.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        group.setToolTip("Detector grouping is set in the Grouping and timing column.")
        row.addWidget(group)
        row.addWidget(self._pipeline_arrow())
        row.addWidget(self._make_pipeline_chip("background"))
        # The α chip and its leading arrow live in a container hidden in vector mode.
        self._pipeline_alpha_widget = QWidget()
        alpha_row = QHBoxLayout(self._pipeline_alpha_widget)
        alpha_row.setContentsMargins(0, 0, 0, 0)
        alpha_row.setSpacing(6)
        alpha_row.addWidget(self._pipeline_arrow())
        alpha_row.addWidget(self._make_pipeline_chip("alpha"))
        row.addWidget(self._pipeline_alpha_widget)
        # β is scalar-only, so its chip hides with α's in vector mode.
        self._pipeline_beta_widget = QWidget()
        beta_row = QHBoxLayout(self._pipeline_beta_widget)
        beta_row.setContentsMargins(0, 0, 0, 0)
        beta_row.setSpacing(6)
        beta_row.addWidget(self._pipeline_arrow())
        beta_row.addWidget(self._make_pipeline_chip("beta"))
        row.addWidget(self._pipeline_beta_widget)
        row.addStretch()
        return widget

    @staticmethod
    def _pipeline_arrow() -> QLabel:
        arrow = QLabel("→")
        arrow.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        return arrow

    def _make_pipeline_chip(self, stage: str) -> QPushButton:
        chip = QPushButton()
        chip.setCheckable(True)
        chip.setAutoDefault(False)
        chip.setDefault(False)
        # The chip outline wears the stage's identity colour — the same colour
        # the stage's correction card wears as its stripe — so chip and card
        # read as one thing; the checked (compare-focused) chip fills the
        # stage's soft tint.
        color, soft = _STAGE_COLORS[stage]
        chip.setStyleSheet(build_stage_chip_qss(color, soft))
        chip.setToolTip("Compare this stage's before/after in the preview below.")
        chip.clicked.connect(lambda _checked=False, s=stage: self._on_pipeline_chip_clicked(s))
        self._pipeline_chips[stage] = chip
        return chip

    def _on_pipeline_chip_clicked(self, stage: str) -> None:
        """Focus (or unfocus) *stage*'s compare; expand and reveal its card."""
        self._set_compare_stage(None if self._compare_stage == stage else stage)
        card = getattr(self, "_correction_cards", {}).get(stage)
        scroll = getattr(self, "_corrections_scroll", None)
        if card is not None:
            card.set_expanded(True)
            if scroll is not None:
                scroll.ensureWidgetVisible(card)

    def _sync_pipeline_strip(self) -> None:
        """Refresh chip summaries, focus highlight, enabled + vector visibility."""
        if not hasattr(self, "_pipeline_chips"):
            return
        alpha_stale = self._alpha_is_stale()
        for stage, chip in self._pipeline_chips.items():
            chip.setText(self._pipeline_summary(stage))
            available = self._compare_stage_available(stage)
            chip.setEnabled(available)
            chip.setChecked(available and self._compare_stage == stage)
            if stage == "alpha":
                chip.setToolTip(
                    "α was calibrated under different corrections — re-estimate it "
                    "in the α section so it centres the corrected asymmetry."
                    if alpha_stale
                    else "Compare this stage's before/after in the preview below."
                )
        if hasattr(self, "_pipeline_alpha_widget"):
            self._pipeline_alpha_widget.setVisible(not bool(self._vector_axis_pairs))
        if hasattr(self, "_pipeline_beta_widget"):
            self._pipeline_beta_widget.setVisible(not bool(self._vector_axis_pairs))
        self._sync_correction_cards()

    def _sync_correction_cards(self) -> None:
        """Refresh each card's status summary, stale tint, and compare indicator.

        Runs from the end of :meth:`_sync_pipeline_strip`, so the card headers
        track the same seams that refresh the chips (mode edits, α calibration,
        staleness, compare-focus changes). "raw" shows no card indicator — the
        compound compare is not one stage's card; the pager label names it.
        """
        cards = getattr(self, "_correction_cards", None)
        if not cards:
            return
        for stage, card in cards.items():
            card.set_status(self._correction_card_status(stage))
            if stage == "alpha":
                card.set_stale(self._alpha_is_stale())
            comparing = self._compare_stage == stage and self._compare_stage_available(stage)
            card.set_comparing(_CARD_COMPARING_TEXTS[stage] if comparing else None)

    def _correction_card_status(self, stage: str) -> str:
        """The card's live status: the chip summary minus the title-duplicating prefix."""
        return self._pipeline_summary(stage).removeprefix(_CARD_STATUS_PREFIXES[stage])

    def _correction_stage_active(self, stage: str) -> bool:
        """Whether *stage* is configured to do anything to the reduction.

        Drives the card expansion policy (expanded iff active at open;
        auto-expand on a reseed that activates a collapsed card). α counts as
        active when calibrated, off-unity, or in vector mode (the per-projection
        table always carries real balances there).
        """
        if stage == "deadtime":
            return self._current_deadtime_mode() != "off"
        if stage == "background":
            return self._current_background_mode() != "none"
        if stage == "alpha":
            return (
                bool(self._vector_axis_pairs)
                or self._alpha_is_calibrated()
                or abs(float(self._alpha_spin.value()) - 1.0) > 1e-9
            )
        if stage == "beta":
            # Scalar-only: never active in vector mode (the card is hidden and
            # the payload omits the key there).
            return not bool(self._vector_axis_pairs) and self._beta_section.is_active()
        return False

    def _auto_expand_active_cards(self) -> None:
        """Expand any collapsed card whose stage is now active (never collapses).

        Called after the form is re-seeded (preset/profile/run switch through
        :meth:`_reload_controls_from_seed`): a stage the reseed just activated
        must not stay hidden behind a collapsed card. A stage going inactive
        keeps its card as the user left it.
        """
        cards = getattr(self, "_correction_cards", None)
        if not cards:
            return
        for stage, card in cards.items():
            if self._correction_stage_active(stage) and not card.is_expanded():
                card.set_expanded(True)

    def _pipeline_summary(self, stage: str) -> str:
        """The one-line chip summary for *stage* (reuses the section formatters)."""
        if stage == "deadtime":
            return deadtime_status_text(self._deadtime_section.get_policy())
        if stage == "beta":
            return beta_status_text(self._beta_section.value())
        if stage == "background":
            details = (
                {"background_run": self._background_run_payload}
                if self._background_run_payload
                else {}
            )
            return background_status_text(
                BackgroundPolicy(mode=self._current_background_mode(), details=details)
            )
        value = float(self._alpha_spin.value())
        stale_suffix = " · stale" if self._alpha_is_stale() else ""
        if self._alpha_is_calibrated():
            label = next(
                (lbl for lbl, key, _ in _ALPHA_METHOD_ITEMS if key == self._current_alpha_method()),
                self._current_alpha_method(),
            )
            return f"α = {value:.4f} · {label}{stale_suffix}"
        return f"α = {value:.4f}{stale_suffix}"

    def _corrections_overflow_sections(self) -> list[tuple[str, QWidget]]:
        """Corrections-column landmarks for the overflow pill, top-to-bottom.

        The correction cards are the landmarks. Short labels (the pill has no
        room for the "correction"/"subtraction" suffixes); α keeps its full
        display name and is always included — the α card stays in this column in
        vector mode too, where the per-projection table lives in its body.
        """
        sections: list[tuple[str, QWidget]] = [
            ("Deadtime", self._deadtime_card),
            ("Background", self._background_card),
            ("α (detector balance)", self._alpha_card),
        ]
        # The β card hides in vector mode (scalar-only), so it is a landmark
        # only while visible — isHidden(), not isVisibleTo(), for the same
        # focus-independence reason as the Periods landmark.
        if not self._beta_card.isHidden():
            sections.append(("β (asymmetry balance)", self._beta_card))
        return sections

    def _grouping_overflow_sections(self) -> list[tuple[str, QWidget]]:
        """Grouping-column landmarks for the overflow pill — coarse, top-to-bottom."""
        sections: list[tuple[str, QWidget]] = [("t0 and binning", self._t0_row_widget)]
        # The periods row container stays visible even when both of its
        # independently-gated children are hidden (RG radios: two-period data;
        # Map periods…: 3+ periods), so gate the landmark on the children's own
        # state — isHidden(), not isVisibleTo(), so the answer does not depend on
        # focus. Without this, single-period data lists a phantom "Periods".
        if not self._period_mode_widget.isHidden() or not self._map_periods_btn.isHidden():
            sections.append(("Periods", self._period_row_widget))
        return sections

    @staticmethod
    def _build_column_header(title: str) -> QLabel:
        """The uppercase BENCH section-header label used atop each column."""
        return make_section_header(title)

    def _on_compare_toggled(self, stage: str, checked: bool) -> None:
        """A compare checkbox was clicked ("raw") — focus that stage, or clear."""
        if getattr(self, "_syncing_compare", False):
            return  # programmatic sync, not a user click
        self._set_compare_stage(stage if checked else None)

    def _set_compare_stage(self, stage: str | None) -> None:
        """Focus (at most) one compare stage and refresh the preview."""
        self._compare_stage = stage
        self._sync_compare_toggles()
        self._refresh_preview()

    def _sync_compare_toggles(self) -> None:
        """Reflect ``_compare_stage`` across the compare surfaces.

        Syncs the pager-row "raw" checkbox (the only remaining checkbox — the
        per-stage compares are driven by chips + pager and displayed on the
        focused card), resets the focus if the focused stage is no longer
        available (e.g. its correction was switched off, or vector mode made α
        unavailable), then refreshes the pipeline strip (which refreshes the
        cards) and the pager, so no surface ever disagrees with the preview.
        """
        if not hasattr(self, "_compare_toggles"):
            return
        if self._compare_stage is not None and not self._compare_stage_available(
            self._compare_stage
        ):
            self._compare_stage = None
        self._syncing_compare = True
        try:
            for key, toggle in self._compare_toggles.items():
                available = self._compare_stage_available(key)
                toggle.setEnabled(available)
                toggle.setChecked(available and self._compare_stage == key)
        finally:
            self._syncing_compare = False
        self._sync_pipeline_strip()
        self._sync_compare_pager()

    def _build_compare_pager(self) -> QWidget:
        """The ◀/▶ pager row: steps ``_compare_stage`` through the cycle.

        Sits directly above the pinned preview so it works from either column.
        A pure wrapper over :meth:`_set_compare_stage` — same shared state the
        section toggles and pipeline chips drive; :meth:`_sync_compare_pager`
        (called from the single :meth:`_sync_compare_toggles` sync seam) keeps
        the label and arrow enabled-state in step.
        """
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        self._compare_prev_btn = QToolButton()
        self._compare_prev_btn.setArrowType(Qt.ArrowType.LeftArrow)
        self._compare_prev_btn.setToolTip("Previous comparison")
        self._compare_prev_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._compare_prev_btn.clicked.connect(lambda: self._step_compare(-1))
        row.addWidget(self._compare_prev_btn)
        self._compare_pager_label = QLabel("Comparing: off")
        self._compare_pager_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._compare_pager_label.setToolTip(
            "Comparing overlays one stage's before/after — the reduction always applies every stage."
        )
        row.addWidget(self._compare_pager_label)
        self._compare_next_btn = QToolButton()
        self._compare_next_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._compare_next_btn.setToolTip("Next comparison")
        self._compare_next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._compare_next_btn.clicked.connect(lambda: self._step_compare(1))
        row.addWidget(self._compare_next_btn)
        row.addStretch()
        # The compound "vs raw" toggle lives here (not in the Corrections column's
        # scroll content): it is one of the pager's stops, and pinning it beside
        # the pager keeps it reachable regardless of which column is focused — and
        # keeps the corrections content short enough to fit the viewport without
        # outer scrolling.
        raw = QCheckBox("Compare vs raw (uncorrected)")
        raw.setToolTip(
            "Preview only: overlay the fully-uncorrected asymmetry — no deadtime, no "
            "background, α = 1."
        )
        raw.toggled.connect(lambda checked: self._on_compare_toggled("raw", checked))
        self._compare_toggles["raw"] = raw
        row.addWidget(raw)
        return widget

    def _step_compare(self, direction: int) -> None:
        """Advance ``_compare_stage`` by *direction* (±1) to the next available
        stage in ``_COMPARE_CYCLE``, wrapping and skipping stages
        :meth:`_compare_stage_available` rejects. ``None`` ("off") is always a
        valid landing stage, so this always terminates within one lap.
        """
        n = len(_COMPARE_CYCLE)
        try:
            idx = _COMPARE_CYCLE.index(self._compare_stage)
        except ValueError:
            idx = 0
        for _ in range(n):
            idx = (idx + direction) % n
            stage = _COMPARE_CYCLE[idx]
            if stage is None or self._compare_stage_available(stage):
                self._set_compare_stage(stage)
                return

    def _sync_compare_pager(self) -> None:
        """Refresh the pager label + arrow enabled-state from ``_compare_stage``.

        Called from the end of :meth:`_sync_compare_toggles`, the single sync
        seam that already runs on every stage change, availability change, and
        vector-mode flip.
        """
        if not hasattr(self, "_compare_pager_label"):
            return  # sync can run before the pager is built
        available = [
            stage
            for stage in _COMPARE_CYCLE
            if stage is not None and self._compare_stage_available(stage)
        ]
        if self._compare_stage is None:
            self._compare_pager_label.setText("Comparing: off")
        else:
            name = _COMPARE_STAGE_LABELS[self._compare_stage]
            position = available.index(self._compare_stage) + 1
            self._compare_pager_label.setText(f"Comparing: {name} ({position}/{len(available)})")
        enabled = bool(available)
        self._compare_prev_btn.setEnabled(enabled)
        self._compare_next_btn.setEnabled(enabled)

    def _compare_stage_available(self, stage: str) -> bool:
        """Whether *stage* has a before/after to show (enable-when-active)."""
        alpha_off_unity = abs(float(self._alpha_spin.value()) - 1.0) > 1e-9
        beta_off_unity = not bool(self._vector_axis_pairs) and self._beta_section.is_active()
        if stage == "deadtime":
            return self._current_deadtime_mode() != "off"
        if stage == "background":
            return self._current_background_mode() != "none"
        if stage == "alpha":
            # The α compare/toggle is unavailable in vector mode (the per-projection
            # table in the Corrections column owns α there), so it never focuses.
            if bool(self._vector_axis_pairs):
                return False
            return self._alpha_is_calibrated() or alpha_off_unity
        if stage == "beta":
            # Scalar-only, like the α compare: never available in vector mode.
            return beta_off_unity
        if stage == "raw":
            return (
                self._current_deadtime_mode() != "off"
                or self._current_background_mode() != "none"
                or alpha_off_unity
                or beta_off_unity
            )
        return False

    def _refresh_preview(self, *args: object) -> None:
        """Recompute the live asymmetry preview for the draft + preview run.

        Builds a throwaway draft profile from the live form payload (cheap
        widget reads, without mutating ``self._draft`` or its dirty state) and
        hands it to the pane, which debounces, then resolves it against the
        preview run AND reduces on its worker thread. Resolution — a full
        per-detector t0 scan under an ``auto_detect`` policy, group sums under a
        per-run alpha estimate — must never run here: this slot fires per
        keystroke/click from nearly every form control. Any error is swallowed
        or surfaced as a muted status message rather than a popup — the preview
        is advisory. Datasets without raw histograms (co-added curves) make the
        pane hide itself with a note.
        """
        pane = getattr(self, "_preview_pane", None)
        if pane is None:
            return
        run = self._run
        try:
            profile = self._preview_draft_profile()
        except Exception:  # noqa: BLE001 — advisory preview, never crash the dialog
            profile = None
        if profile is None or run is None:
            pane.request_preview(
                histograms=None,
                grouping={},
                run_number=self._preview_run_number(),
            )
            return
        metadata: dict[str, Any] = {}
        if self._reference_dataset is not None:
            metadata.update(getattr(self._reference_dataset, "metadata", {}) or {})
        metadata.update(getattr(run, "metadata", {}) or {})
        facility = str(metadata.get("facility", metadata.get("instrument", "")))
        # Keep the toggles' enabled/checked state in step with the current draft,
        # and drop the focus if the focused stage is no longer available.
        self._sync_compare_toggles()
        # Compare view: the focused stage (if any) draws its before/after ghost over
        # the solid full-pipeline curve. The solid is never degraded, so the α
        # compare's residual-⟨A⟩ acceptance number is always read off the
        # fully-corrected curve. Preview-only — `compare_stage` never reaches the
        # persisted reduction.
        pane.request_preview_from_profile(
            profile=profile,
            run=run,
            facility=facility,
            run_number=self._preview_run_number(),
            compare_stage=self._compare_stage,
        )

    def _preview_run_number(self) -> int | None:
        if self._reference_dataset is None:
            return None
        try:
            return int(self._reference_dataset.run_number)
        except (TypeError, ValueError):
            return None

    def _preview_draft_profile(self) -> GroupingProfile | None:
        """Build a throwaway draft profile from the live form payload.

        Cheap (widget reads and dict lifting only) — resolution against the
        preview run happens on the preview pane's worker thread, not here.
        """
        if self._run is None or not self._run.histograms or self._fingerprint is None:
            return None
        payload = self._current_grouping_payload()
        return profile_from_form_payload(
            payload,
            name=self._draft_name,
            fingerprint=self._fingerprint,
            active=True,
        )

    def _pending_override_runs(self) -> list[int]:
        """Overridden runs with a dirty override draft, in ascending order."""
        return sorted(rn for rn in self._override_dirty_runs if self._run_is_overridden(int(rn)))

    def _has_structural_changes(self) -> bool:
        """Whether Apply must commit profile-level changes beyond run payloads.

        Covers reassignments, deletions, a pending rename, a brand-new (never
        stored) profile, and a moved default — each meaningful even when no run
        currently follows the edited profile.
        """
        if self._newly_assigned() or self._deleted_profiles or self._session_created:
            return True
        if self._renamed_from is not None:
            return True
        stored = self._stored_names_for_fingerprint()
        # A user-created (New/Duplicate) profile not yet stored is structural;
        # the auto-synthesized initial draft of a profile-less fingerprint is
        # not — it only becomes worth saving once a run follows it.
        if stored and self._draft_name not in stored:
            return True
        return self._current_default_name() != self._stored_default_name()

    def _update_apply_enabled(self) -> None:
        """Enable/label Apply, showing the blast radius of everything dirty.

        Apply commits the profile to every run following it plus each dirty
        override to its own run. The label shows the pending override count when
        any override is dirty ("Apply (profile + 2 overrides)"). Apply is
        disabled only when there is nothing to commit: no run follows the
        profile, no override has pending edits, and no structural change
        (reassignment, rename, deletion, new profile, moved default) waits.
        """
        followers = self._scope_panel.runs_following(self._draft_name)
        pending = self._pending_override_runs()
        enabled = bool(followers) or bool(pending) or self._has_structural_changes()
        self._apply_btn.setEnabled(enabled)
        if pending:
            noun = "override" if len(pending) == 1 else "overrides"
            self._apply_btn.setText(f"Apply (profile + {len(pending)} {noun})")
        else:
            self._apply_btn.setText("Apply")
        if not enabled:
            self._apply_btn.setToolTip(
                "No run of this instrument follows this profile (all released "
                "or assigned elsewhere) and no override has pending edits. "
                "Reattach or assign a run to apply."
            )
            return
        parts: list[str] = []
        if followers:
            parts.append(f"save profile '{self._draft_name}' to {len(followers)} run(s)")
        elif self._has_structural_changes():
            parts.append(f"save profile '{self._draft_name}'")
        if pending:
            parts.append(f"apply override edits to run(s) {', '.join(str(r) for r in pending)}")
        self._apply_btn.setToolTip("Apply will " + " and ".join(parts) + ".")

    def _clear_dirty(self) -> None:
        """Disarm the unsaved-draft guard (used by the test teardown fixture)."""
        self._draft_dirty = False
        self._override_dirty_runs.clear()
        self._deleted_profiles.clear()
        self._session_created.clear()
        self._renamed_from = None
        self._session_assignments = dict(self._initial_assignments)

    def _guard_discard(self) -> bool:
        """Return whether it is safe to close (prompting on any uncommitted edit).

        The single close-time guard for the unified model. It covers a dirty
        profile draft *and* every run with uncommitted override edits, and the
        prompt names both parts — e.g. "profile 'Default (GPS)' and overrides for
        runs 12, 15" — so nothing can be lost silently.
        """
        pending = self._pending_override_runs()
        structural = (
            bool(self._deleted_profiles)
            or bool(self._session_created)
            or self._renamed_from is not None
            or bool(self._newly_assigned())
        )
        if not self._draft_dirty and not pending and not structural:
            return True
        lost: list[str] = []
        if self._draft_dirty:
            lost.append(f"profile '{self._draft_name}'")
        if pending:
            noun = "override" if len(pending) == 1 else "overrides"
            run_word = "run" if len(pending) == 1 else "runs"
            lost.append(f"{noun} for {run_word} {', '.join(str(r) for r in pending)}")
        if structural and not self._draft_dirty:
            lost.append("pending profile changes (reassignment/rename/deletion)")
        answer = QMessageBox.question(
            self,
            "Discard changes",
            f"Discard uncommitted changes to {' and '.join(lost)}?",
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
        self._teardown_workers()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        """Tear the preview + background-configure runners down on dismissal."""
        self._teardown_workers()
        super().done(result)

    def _teardown_workers(self) -> None:
        """Cancel and join every worker runner (idempotent).

        The preview pane owns its runner for the live-preview reduction; the
        inline α section owns its single-α estimate runner;
        ``self._vector_alpha_tasks`` runs the inline per-projection α estimates;
        and ``self._tasks`` is this dialog's runner for the background-configure
        preview grouping. Every ``shutdown()`` is safe to call more than once.
        """
        pane = getattr(self, "_preview_pane", None)
        if pane is not None:
            pane.shutdown()
        alpha_section = getattr(self, "_alpha_section", None)
        if alpha_section is not None:
            alpha_section.shutdown()
        vector_tasks = getattr(self, "_vector_alpha_tasks", None)
        if vector_tasks is not None:
            vector_tasks.shutdown()
        tasks = getattr(self, "_tasks", None)
        if tasks is not None:
            tasks.shutdown()

    def _populate_group_table(self) -> None:
        """Render the detector-group table used as grouping context.

        ``itemChanged`` is connected to both ``_mark_dirty`` and
        ``_refresh_preview`` (which queues a debounced off-thread recompute on
        the preview pane's worker — resolution no longer runs synchronously
        here, see B3/#216), so populating without blocking signals fires up to
        4×N_groups redundant dirty marks + queued previews. Population is not a
        user edit, so the signal is blocked for the whole rebuild; callers that
        need the dirty/preview side effects for the triggering action invoke
        them explicitly once, after this returns.
        """
        blocked = self._group_table.blockSignals(True)
        try:
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
        finally:
            self._group_table.blockSignals(blocked)
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
        # The inline single-α section calibrates the single/P_z balance; the
        # per-projection vector table (in the α area, with its own inline Estimate)
        # owns the rest. The α card stays put in both modes — its body (single-α
        # widgets or the vector table) always lives in the Corrections column —
        # but the single-α calibration section is hidden in vector mode.
        # _sync_compare_toggles clears an α compare focus there (α is never
        # available in vector mode), so no ghost — and no card accent — is left.
        if hasattr(self, "_alpha_section"):
            self._alpha_section.setVisible(not vector_mode)
            self._sync_compare_toggles()
        # β is scalar-only: the whole card hides in vector mode and the payload
        # omits the key there (per-projection reductions stay at β = 1). The
        # widget keeps its value so leaving vector mode restores it.
        if hasattr(self, "_beta_card"):
            self._beta_card.setVisible(not vector_mode)

        if vector_mode:
            self._update_vector_alpha_controls()

        if not vector_mode:
            self._rebuild_vector_alpha_table([], grouping_values, canonical)
            if "alpha" in grouping_values:
                try:
                    self._alpha_spin.setValue(float(grouping_values.get("alpha", 1.0)))
                except (TypeError, ValueError):
                    pass
            self._refresh_overflow_indicators()
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

        # Section visibility just changed (single-α header vs per-projection
        # table), so re-derive which sections fall below the fold.
        self._refresh_overflow_indicators()

    def _refresh_overflow_indicators(self) -> None:
        """Recompute both overflow pills (no-op before they are constructed)."""
        for name in ("_corrections_overflow", "_grouping_overflow"):
            indicator = getattr(self, name, None)
            if indicator is not None:
                indicator.refresh()

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

            spin = NoScrollDoubleSpinBox()
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
        with self._busy_cursor():
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
        # The "Periods" overflow landmark is gated on this visibility.
        self._refresh_overflow_indicators()

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

    def _update_alpha_section(self) -> None:
        """(Re)seed the inline single-α section's run list + method."""
        section = getattr(self, "_alpha_section", None)
        if section is None:
            return
        section.configure(
            datasets=self._fingerprint_datasets(),
            method=self._current_alpha_method(),
            selected_run_number=int(self._reference_dataset.run_number),
            context_provider=self._alpha_estimate_context,
        )

    def _alpha_estimate_context(self) -> dict[str, Any]:
        """The current group pair + correction context for an inline α estimate.

        Read fresh at Estimate time so a group / forward-backward edit is honoured
        without re-seeding (which would reset the calibration-run selection).
        """
        try:
            forward_gid = int(self._forward_combo.currentData())
            backward_gid = int(self._backward_combo.currentData())
        except (TypeError, ValueError):
            forward_gid, backward_gid = 1, 2
        return {
            "groups": self._groups,
            "forward_group": forward_gid,
            "backward_group": backward_gid,
            "excluded_detectors": self._current_excluded_detectors() or [],
            "correction_provider": self._calibration_correction_provider(),
            "reference_resolver": self._calibration_reference_resolver,
            "facility": self._calibration_facility(),
        }

    def _on_alpha_section_estimated(self, policy: object) -> None:
        """Apply an inline single-α estimate to the α spin + provenance."""
        if isinstance(policy, AlphaPolicy):
            self._apply_calibrated_policy("single", self._alpha_spin, policy)

    def _calibration_correction_provider(self):
        """Build the per-dataset correction-payload provider for the alpha dialog.

        Resolves the current draft (deadtime + background config) against each
        candidate calibration run, so the alpha estimate runs on the same
        deadtime-corrected, background-subtracted counts the reduction will. The
        draft's alpha policy is forced to a fixed value so resolving does not
        trigger a nested per-run alpha estimate (the dialog measures alpha
        itself). Returns ``None`` when no fingerprint context is available.
        """
        if self._fingerprint is None:
            return None
        payload = self._current_grouping_payload()
        profile = profile_from_form_payload(
            payload, name=self._draft_name or "draft", fingerprint=self._fingerprint
        )
        profile = replace(profile, alpha_policy=AlphaPolicy(mode="fixed", value=1.0))

        def provide(dataset: MuonDataset) -> dict[str, Any]:
            if dataset.run is None:
                return {}
            return resolve_effective_grouping(profile, dataset.run)

        return provide

    def _calibration_reference_resolver(
        self, grouping: dict[str, Any]
    ) -> tuple[list, float] | None:
        """Resolve a ``reference_run`` background for the alpha calibration.

        Reuses an already-loaded reference run from the fingerprint's datasets (or
        loads the recorded source file, cached per path), returning the reference
        histograms and the good-frame scale. ``None`` on any failure, so the
        estimate degrades to no reference subtraction (and the dialog says so).
        """
        payload = grouping.get("background_run")
        if not isinstance(payload, dict):
            return None
        cache = getattr(self, "_background_run_cache", None)
        if cache is None:
            cache = {}
            self._background_run_cache = cache
        try:
            sample_frames = float(grouping.get("good_frames", 0.0)) or None
        except (TypeError, ValueError):
            sample_frames = None
        try:
            reference = resolve_background_reference(
                payload,
                sample_good_frames=sample_frames,
                datasets=self._fingerprint_datasets(),
                cache=cache,
            )
        except (ValueError, OSError):
            return None
        return reference.histograms, reference.scale

    def _calibration_facility(self) -> str:
        """Facility label for the alpha dialog's background tail-fit shortening."""
        metadata = (self._run.metadata if self._run is not None else None) or {}
        return str(metadata.get("facility", metadata.get("instrument", "")))

    # ------------------------------------------------------------------
    # Inline per-projection (vector) α estimate
    # ------------------------------------------------------------------

    def _update_vector_alpha_controls(self) -> None:
        """(Re)seed the shared vector calibration-run list + method combo.

        The run combo is repopulated preserving the current selection (falling
        back to the preview reference run); the method mirrors the single-α
        provenance method. Called whenever the vector table becomes visible.
        """
        combo = getattr(self, "_vector_run_combo", None)
        if combo is None:
            return
        current = combo.currentData()
        selected = int(current) if current is not None else int(self._reference_dataset.run_number)
        combo.blockSignals(True)
        populate_calibration_run_combo(combo, self._fingerprint_datasets(), selected)
        combo.blockSignals(False)
        method_combo = self._vector_method_combo
        idx = method_combo.findData(self._current_alpha_method())
        method_combo.blockSignals(True)
        method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        method_combo.blockSignals(False)

    def _on_vector_method_changed(self) -> None:
        """Keep the payload's α-method provenance in step with the vector combo."""
        self._set_alpha_method(self._current_vector_method())

    def _current_vector_method(self) -> str:
        return str(self._vector_method_combo.currentData() or "diamagnetic")

    def _vector_calibration_dataset(self) -> MuonDataset | None:
        """The dataset for the shared vector calibration-run selection."""
        run_number = self._vector_run_combo.currentData()
        if run_number is None:
            return None
        return next(
            (ds for ds in self._fingerprint_datasets() if int(ds.run_number) == int(run_number)),
            None,
        )

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
        # α is now measured under the current corrections; stamp their digest so
        # a later deadtime/background change re-raises the staleness banner.
        self._alpha_correction_digest = self._correction_digest()
        self._refresh_alpha_staleness()
        self._record_calibration_result_label(slot, policy)
        # Auto-focus the α compare on a fresh calibration (single mode only — the
        # α compare is unavailable in vector mode, where the per-projection table
        # owns α), preserving the old auto-overlay; the chips/pager can dismiss
        # it. The α card is expanded too so the result + provenance rows the
        # calibration just wrote are actually on screen.
        if not bool(self._vector_axis_pairs):
            self._set_compare_stage("alpha")
        if hasattr(self, "_alpha_card"):
            self._alpha_card.set_expanded(True)
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
        self._alpha_correction_digest = None
        self._refresh_alpha_provenance_label()
        self._refresh_alpha_staleness()

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
        digest = grouping.get("alpha_correction_digest")
        self._alpha_correction_digest = str(digest) if digest is not None else None

    #: Grouping keys whose change means a calibrated α no longer centres the
    #: corrected asymmetry (the deadtime + background correction surface).
    _CORRECTION_DIGEST_DEADTIME_KEYS = (
        "deadtime_mode",
        "deadtime_method",
        "dead_time_us",
        "deadtime_estimated_us",
    )
    _CORRECTION_DIGEST_BACKGROUND_KEYS = (
        "background_mode",
        "background_fixed_values",
        "background_ranges",
        "background_range",
        "background_run",
    )

    def _correction_digest(self) -> str:
        """Stable digest of the *effective* deadtime + background settings.

        Only the enabled corrections contribute, so an ``off`` deadtime with
        stale table values digests the same as a plain ``off`` — the banner fires
        on a change that would actually move the corrected counts, not on inert
        edits.
        """
        payload = self._current_grouping_payload()
        surface: dict[str, Any] = {}
        if payload.get("deadtime_correction"):
            surface["deadtime"] = {
                key: payload.get(key)
                for key in self._CORRECTION_DIGEST_DEADTIME_KEYS
                if payload.get(key) is not None
            }
        if payload.get("background_correction"):
            surface["background"] = {
                key: payload.get(key)
                for key in self._CORRECTION_DIGEST_BACKGROUND_KEYS
                if payload.get(key) is not None
            }
        blob = json.dumps(surface, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    @staticmethod
    def _alpha_provenance_holds(
        recorded_value: float, current_value: float, decimals: int = 6
    ) -> bool:
        """True when *current_value* (a spin's displayed α) still shows *recorded*.

        The estimate stores full precision but the α spins round to *decimals*, so a
        naive ``< 1e-9`` compare reads every real (non-round) calibration as a hand
        edit — silently dropping the "calibrated" provenance (from the saved profile
        too), the staleness banner and the α chip's stale flag. A displayed value is within
        half a display ULP of the recorded value; a genuine manual edit moves the
        spin to a different displayed value, well beyond that. (Tested against the
        raw difference rather than re-rounding both sides, so Python's banker's
        rounding can't disagree with Qt's at an exact half-ULP tie.)
        """
        return abs(float(recorded_value) - float(current_value)) < 0.5 * 10 ** (-decimals)

    def _alpha_is_calibrated(self) -> bool:
        """True when the single α spin still holds a calibrated estimate."""
        recorded = self._alpha_estimate_state.get("single") or self._alpha_estimate_state.get("P_z")
        if recorded is None:
            return False
        return self._alpha_provenance_holds(
            recorded[0], self._alpha_spin.value(), self._alpha_spin.decimals()
        )

    def _alpha_is_stale(self) -> bool:
        """True when a calibrated α no longer matches the current corrections.

        The digest of the deadtime/background settings α was measured under has
        diverged from the current one, so the calibrated α no longer centres the
        corrected asymmetry. Drives both the staleness banner and the " · stale"
        suffix (plus re-estimation tooltip) on the pipeline α chip.
        """
        return (
            self._alpha_is_calibrated()
            and self._alpha_correction_digest is not None
            and self._alpha_correction_digest != self._correction_digest()
        )

    def _refresh_alpha_staleness(self) -> None:
        """Show the banner (and flag the pipeline α chip) when α went stale."""
        if not hasattr(self, "_alpha_stale_banner"):
            return
        stale = self._alpha_is_stale()
        if stale:
            self._alpha_stale_banner.setText(
                "α was calibrated under different deadtime/background corrections — "
                "re-estimate so it centres the corrected asymmetry."
            )
        self._alpha_stale_banner.setVisible(stale)
        # There are no tabs to mark: staleness surfaces on the pipeline α chip
        # (" · stale" summary + a re-estimation tooltip). The chip is hidden in
        # vector mode, where the banner in the α area covers the same case.
        self._sync_pipeline_strip()

    def _refresh_alpha_provenance_label(self) -> None:
        """Reflect the single alpha's provenance ("calibrated …" or "manual")."""
        if not hasattr(self, "_alpha_provenance_label"):
            return
        recorded = self._alpha_estimate_state.get("single") or self._alpha_estimate_state.get("P_z")
        value = float(self._alpha_spin.value())
        if recorded is not None and self._alpha_provenance_holds(
            recorded[0], value, self._alpha_spin.decimals()
        ):
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
        """Inline-estimate α for one projection pair off the GUI thread.

        *axis* is the projection slot/label — a canonical EMU axis (P_x/P_y/P_z)
        or a non-canonical projection label (FB, Top-Bottom, …). The estimate uses
        the shared calibration run + method with the projection's own forward/
        backward groups; on success the calibrated value + provenance are written
        into the projection's α spin (and, for the canonical P_z anchor, the
        single-α spin too) via :meth:`_apply_calibrated_policy`.
        """
        pair = self._vector_axis_pairs.get(axis)
        if pair is None:
            QMessageBox.warning(
                self, "Estimate Failed", f"No grouping pair is available for {axis}."
            )
            return
        if axis not in self._vector_alpha_spins:
            return
        self._begin_vector_estimates([axis])

    def _estimate_all_alpha(self) -> None:
        """Estimate α for every projection pair, one axis at a time.

        Serialised (not fired in parallel) so each result routes to its own spin —
        the off-thread estimates share one runner and a single result token.
        """
        axes = [
            axis
            for axis in self._ordered_projection_labels(
                self._vector_axis_pairs, self._is_canonical_vector_pairs(self._vector_axis_pairs)
            )
            if axis in self._vector_alpha_spins and self._vector_axis_pairs.get(axis) is not None
        ]
        self._begin_vector_estimates(axes)

    def _begin_vector_estimates(self, axes: list[str]) -> None:
        """Queue *axes* for serialised inline estimation and start the first."""
        if self._run is None or not self._run.histograms:
            QMessageBox.warning(self, "Alpha Calibration", "Reference run has no histograms.")
            return
        if not axes:
            return
        self._vector_estimate_queue = list(axes)
        self._set_vector_estimate_enabled(False)
        self._start_next_vector_estimate()

    def _start_next_vector_estimate(self) -> None:
        """Kick off the next queued axis estimate, or re-enable when drained."""
        while self._vector_estimate_queue:
            axis = self._vector_estimate_queue[0]
            pair = self._vector_axis_pairs.get(axis)
            spin = self._vector_alpha_spins.get(axis)
            dataset = self._vector_calibration_dataset()
            if pair is None or spin is None or dataset is None or dataset.run is None:
                self._vector_estimate_queue.pop(0)
                continue
            forward_gid, backward_gid = int(pair[0]), int(pair[1])
            if forward_gid == backward_gid:
                QMessageBox.warning(
                    self, "Invalid Grouping", "Forward and backward groups must differ."
                )
                self._vector_estimate_queue.pop(0)
                continue
            self._vector_estimate_source_run = int(dataset.run_number)
            self._vector_estimate_token += 1
            token = self._vector_estimate_token
            request = build_alpha_request(
                token=token,
                dataset=dataset,
                groups=self._groups,
                forward_gid=forward_gid,
                backward_gid=backward_gid,
                excluded_detectors=self._current_excluded_detectors() or [],
                method=self._current_vector_method(),
                correction_provider=self._calibration_correction_provider(),
                reference_resolver=self._calibration_reference_resolver,
                facility=self._calibration_facility(),
            )
            self._vector_alpha_tasks.start(
                lambda worker, request=request: run_alpha_estimate(worker, request),
                on_finished=lambda result, axis=axis, token=token: (
                    self._on_vector_estimate_finished(axis, token, result)
                ),
                on_error=lambda message, axis=axis: self._on_vector_estimate_error(axis, message),
            )
            return
        self._set_vector_estimate_enabled(True)

    def _on_vector_estimate_finished(self, axis: str, token: int, result: object) -> None:
        """Apply one axis estimate (unless superseded) and start the next."""
        if token != self._vector_estimate_token:
            return  # a newer estimate supersedes this stale result
        if self._vector_estimate_queue and self._vector_estimate_queue[0] == axis:
            self._vector_estimate_queue.pop(0)
        spin = self._vector_alpha_spins.get(axis)
        if isinstance(result, AlphaEstimateResult) and result.estimate.ok and spin is not None:
            estimate = result.estimate
            self._apply_calibrated_policy(
                axis,
                spin,
                AlphaPolicy(
                    mode="calibrated",
                    value=float(estimate.alpha),
                    error=estimate.alpha_error,
                    method=estimate.method,
                    source_run=self._vector_estimate_source_run,
                ),
            )
        elif isinstance(result, AlphaEstimateResult) and not result.estimate.ok:
            QMessageBox.warning(
                self, "Alpha Calibration", f"Estimate failed: {result.estimate.message}"
            )
        self._start_next_vector_estimate()

    def _on_vector_estimate_error(self, axis: str, message: str) -> None:
        """Surface a worker error for one axis and continue the queue."""
        if self._vector_estimate_queue and self._vector_estimate_queue[0] == axis:
            self._vector_estimate_queue.pop(0)
        QMessageBox.warning(self, "Alpha Calibration", message)
        self._start_next_vector_estimate()

    def _set_vector_estimate_enabled(self, enabled: bool) -> None:
        """Enable/disable the per-axis + Estimate-All buttons while in flight."""
        for button in self._vector_estimate_buttons.values():
            button.setEnabled(enabled)
        if self._estimate_all_btn is not None:
            self._estimate_all_btn.setEnabled(enabled)

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
        # The "Periods" overflow landmark is gated on this visibility.
        self._refresh_overflow_indicators()

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
        if recorded is not None and self._alpha_provenance_holds(recorded[0], alpha_value):
            if recorded[1] is not None:
                alpha_provenance["alpha_error"] = float(recorded[1])
            alpha_provenance["alpha_reference_run"] = int(recorded[2])
            # Persist the corrections α was measured under so the staleness
            # banner can fire on reopen, not just within this session.
            if self._alpha_correction_digest is not None:
                alpha_provenance["alpha_correction_digest"] = self._alpha_correction_digest

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
            # β: scalar-only, emitted only when active outside vector mode so a
            # default payload stays byte-identical to a pre-β one.
            | (
                {"beta": float(self._beta_section.value())}
                if not self._vector_axis_pairs and self._beta_section.is_active()
                else {}
            )
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
                if recorded is not None and self._alpha_provenance_holds(recorded[0], value):
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
            if recorded is not None and self._alpha_provenance_holds(recorded[0], value):
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

    def _profile_broadcast_payload(self) -> dict[str, Any]:
        """Return the flat profile payload broadcast to inheriting runs.

        When the current editing target *is* the profile, the live form is the
        profile edit, so the live ``_current_grouping_payload`` is returned
        directly (preserving the historical form-driven keys — alpha provenance,
        deadtime table, binning knobs). When an overridden run is the selected
        target the form is showing that override instead, so the payload is built
        from the profile draft (``self._draft``) resolved against a representative
        inheriting run, never from the form.
        """
        if self._editing_target() == "profile":
            return self._current_grouping_payload()
        followers = sorted(self._scope_panel.runs_following(self._draft_name)) or sorted(
            self._scope_panel.inheriting_run_numbers()
        )
        reference_run = self._run
        for ds in self._fingerprint_datasets():
            if int(ds.run_number) in followers and ds.run is not None:
                reference_run = ds.run
                break
        return dict(payload_from_profile_for_preview(self._draft, reference_run))

    def get_grouping_result(self) -> dict[str, Any] | None:
        """Return the applied grouping payload for inheriting runs.

        The dialog is a profile editor: on Apply the draft is saved as the active
        profile and resolved onto every *inheriting* run of the instrument. The
        returned dict is the flat grouping payload the main window's
        ``_apply_grouping_settings_to_dataset`` consumes, with ``run_numbers``
        set to the inheriting runs. Per-run override edits are carried separately
        in :meth:`get_profile_result`.

        Returns
        -------
        dict or None
            Grouping payload plus ``run_numbers`` (inheriting runs). ``None``
            when no run histograms are available.
        """
        if self._run is None or not self._run.histograms:
            return None
        payload = self._profile_broadcast_payload()
        # Only the runs *following the edited profile* receive the profile
        # apply — runs assigned to another profile of the fingerprint are that
        # profile's business (schema v17).
        payload["run_numbers"] = sorted(self._scope_panel.runs_following(self._draft_name))
        return payload

    def get_profile_result(self) -> dict[str, Any] | None:
        """Return the profile-editor result for the main window to reconcile.

        Returns
        -------
        dict or None
            ``{"profile": GroupingProfile, "inheriting": set[int],
            "released": set[int], "newly_released": set[int],
            "newly_reattached": set[int], "preview_run_number": int,
            "override_edits": {run_number: payload}}`` — the saved draft profile,
            the scope reconciliation, and every dirty per-run override payload to
            apply to its run alone. ``None`` when no run is available.
        """
        if self._run is None or not self._run.histograms:
            return None
        # The current target was synced by _on_apply; re-sync defensively in case
        # a caller reads this without going through Apply. This captures the live
        # form into whichever draft it is currently editing.
        self._sync_draft_from_form()
        # A session-created profile that is the current draft carries its live
        # edits in the draft, not the registered working copy — sync it back so
        # ``created_profiles`` below reports the edited state.
        if self._draft_name in self._session_created and self._fingerprint is not None:
            for index, profile in enumerate(self._project_profiles):
                if profile.fingerprint.matches(self._fingerprint) and (
                    profile.name == self._draft_name
                ):
                    self._project_profiles[index] = GroupingProfile.from_dict(self._draft.to_dict())
                    break
        # Report every override edited this session: the still-dirty ones plus
        # any Apply just committed (Apply clears the dirty set before this runs).
        override_edits: dict[int, dict[str, Any]] = {}
        run_numbers = set(self._pending_override_runs()) | self._committed_override_runs
        for run_number in sorted(run_numbers):
            if not self._run_is_overridden(int(run_number)):
                continue
            draft = self._override_drafts.get(int(run_number))
            if draft is not None:
                override_edits[int(run_number)] = dict(draft)
        return {
            "profile": GroupingProfile.from_dict(self._draft.to_dict()),
            "inheriting": self._scope_panel.inheriting_run_numbers(),
            "released": self._scope_panel.released_run_numbers(),
            "newly_released": self._scope_panel.newly_released(),
            "newly_reattached": self._scope_panel.newly_reattached(),
            "preview_run_number": int(self._reference_dataset.run_number),
            "override_edits": override_edits,
            # Schema v17 assignment reconciliation: the authoritative run→profile
            # map, the changes made this session, the fingerprint's default
            # profile after this apply, and any pending rename/deletions.
            "assignments": {int(rn): str(name) for rn, name in self._session_assignments.items()},
            "newly_assigned": {int(rn): str(name) for rn, name in self._newly_assigned().items()},
            "default_profile": self._current_default_name(),
            "renamed_from": self._renamed_from,
            "deleted_profiles": list(self._deleted_profiles),
            "created_profiles": [
                GroupingProfile.from_dict(p.to_dict())
                for p in self._project_profiles
                if p.name in self._session_created
            ],
        }

    @property
    def draft_profile(self) -> GroupingProfile:
        """The current draft profile (a copy synced from the form)."""
        self._sync_draft_from_form()
        return GroupingProfile.from_dict(self._draft.to_dict())
