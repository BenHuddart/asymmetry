"""Dock-ready grouped time-domain fitting widget.

Mirrors the F-B asymmetry fit panel's structure: a **Single** tab that fits the
active run's detector groups jointly (one dataset) and a **Batch** tab that fits
a series across the selected runs. Both surfaces are grouped
:class:`~asymmetry.gui.panels.fit_panel.GlobalFitTab` instances; they differ only
in their member set (Single → the active run; Batch → the selection).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.parameters import split_parameter_name
from asymmetry.core.transform import resolve_background_mode
from asymmetry.gui.panels.fit_panel import (
    GlobalFitTab,
    _get_file_value_for_parameter,
)
from asymmetry.gui.styles.widgets import make_section
from asymmetry.gui.widgets.collapsible_section import CollapsibleSection

#: Fit-target choices (label, mode key) shown in the count-domain selector.
#: Labels stay short so the selector does not set the Fit dock's minimum width;
#: the mode key (item data) is what is persisted, not the label.
_FIT_TARGETS: tuple[tuple[str, str], ...] = (
    ("All groups", "all"),
    ("F + B (free α)", "fb"),
    ("Single group", "single"),
)
_FIT_COSTS: tuple[tuple[str, str], ...] = (("Poisson", "poisson"), ("Gaussian √N", "gaussian"))
_SINGLE_SIDES: tuple[tuple[str, str], ...] = (("Forward", "forward"), ("Backward", "backward"))


class MultiGroupFitWindow(QWidget):
    """Grouped time-domain fitting surface used inside the main fit dock."""

    grouped_fit_completed = Signal(object, object)
    grouped_preview_requested = Signal(object, object)
    fit_range_edit_committed = Signal(float, float)
    count_fit_completed = Signal(object, object)  # (dataset, {"result", "overlays"})
    count_grouping_promoted = Signal(object)  # (dataset) — a count calibration hit the grouping
    share_function_with_group_requested = Signal(int)  # (source run) — push form to peers

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._build_target_controls())

        self._tabs = QTabWidget()
        # Single = the active run's multi-group fit; Batch = the multi-run series.
        self._single_fit_tab = GlobalFitTab(self, member_kind="groups", grouped_single=True)
        self._batch_fit_tab = GlobalFitTab(self, member_kind="groups")
        for tab in (self._single_fit_tab, self._batch_fit_tab):
            tab.grouped_fit_completed.connect(self.grouped_fit_completed.emit)
            tab.grouped_preview_requested.connect(self.grouped_preview_requested.emit)
            tab.fit_range_edit_committed.connect(self.fit_range_edit_committed.emit)
            tab.count_fit_completed.connect(self.count_fit_completed.emit)
            tab.count_grouping_promoted.connect(self.count_grouping_promoted.emit)
        # The Single surface can push its function to the run's data-group peers.
        self._single_fit_tab.share_function_with_group_requested.connect(
            self.share_function_with_group_requested.emit
        )
        # A converged single grouped fit chain-seeds the batch surface per run
        # (mirrors how FB single fits seed the FB batch surface).
        self._single_fit_tab.single_grouped_fit_recorded.connect(
            self._batch_fit_tab.register_grouped_single_fit_seed
        )
        # "Send to Batch" copies the Single surface's model + seeds to Batch.
        self._single_fit_tab.send_grouped_model_to_batch_requested.connect(
            self._on_send_grouped_model_to_batch
        )
        self._tabs.addTab(self._single_fit_tab, "Single")
        self._tabs.addTab(self._batch_fit_tab, "Batch")
        layout.addWidget(self._tabs)
        self._run_label = ""
        # Per-run restore mediator for the Single (grouped) surface, installed by
        # the main window. Mirrors FitPanel.set_single_fit_restore_provider.
        self._single_grouped_restore_provider: (
            Callable[[MuonDataset | None], dict | None] | None
        ) = None
        # Per-run grouped Single-surface form store (mirrors FitPanel's
        # _single_state_by_run): keeps each run's in-progress function so moving
        # away and back restores it, and an unseen run carries the current
        # function forward instead of snapping to the default.
        self._single_grouped_state_by_run: dict[int, dict] = {}
        self._active_single_grouped_run: int | None = None
        # Snapshot of the Single form taken when leaving the Single tab, restored
        # on return (same run) so an in-progress unfit function survives a
        # Single↔Batch toggle (mirrors FitPanel._single_form_snapshot).
        self._single_grouped_form_snapshot: dict | None = None
        self._tabs.currentChanged.connect(self._on_grouped_tab_changed)
        self._sync_count_fit_target()

    def _build_target_controls(self) -> QWidget:
        """Build the fit-target selector + collapsed count-fit options / calibration.

        Only the **Target** is everyday (and **Side** when the single-group
        target is chosen); every other control configures a count fit (F+B /
        Single group) and is disabled for the lifetime-corrected "All groups"
        target, so the advanced count-fit options and the calibration promotes
        live in two sections collapsed by default — they no longer push the
        model table and Fit button down the dock.
        """
        box, outer = make_section("Fit target")

        # ── Always visible: Target (+ Side, only for the single-group target). ──
        self._target_form = QFormLayout()
        self._target_form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._target_form)

        # Store the mode key as item data so reordering the dropdowns can't remap
        # a selection to the wrong key.
        self._target_combo = QComboBox()
        for label, key in _FIT_TARGETS:
            self._target_combo.addItem(label, key)
        self._target_combo.currentIndexChanged.connect(self._sync_count_fit_target)
        self._target_form.addRow(QLabel("Target"), self._target_combo)

        self._side_combo = QComboBox()
        for label, key in _SINGLE_SIDES:
            self._side_combo.addItem(label, key)
        self._side_combo.currentIndexChanged.connect(self._sync_count_fit_target)
        self._side_label = QLabel("Single group")
        self._target_form.addRow(self._side_label, self._side_combo)

        # ── Collapsed: count-fit options (Cost / Skip / Nuisances / Double pulse). ──
        self._count_options_section = CollapsibleSection("Count-fit options", expanded=False)
        options_form = QFormLayout()
        options_form.setContentsMargins(0, 0, 0, 0)
        self._count_options_section.addLayout(options_form)
        outer.addWidget(self._count_options_section)

        self._cost_combo = QComboBox()
        for label, key in _FIT_COSTS:
            self._cost_combo.addItem(label, key)
        self._cost_combo.currentIndexChanged.connect(self._sync_count_fit_target)
        options_form.addRow(QLabel("Cost"), self._cost_combo)

        # Interior exclude window (μs): inactive while max ≤ min.
        self._exclude_min = QDoubleSpinBox()
        self._exclude_max = QDoubleSpinBox()
        for spin in (self._exclude_min, self._exclude_max):
            spin.setDecimals(3)
            spin.setRange(0.0, 1000.0)
            spin.setSingleStep(0.1)
            # Keep these count-config fields narrow so the grouped page does not
            # set the Fit dock's minimum width past the other tabs: drop the spin
            # arrows (edit by typing/keys/wheel) and cap the width.
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setMaximumWidth(76)
            spin.valueChanged.connect(self._sync_count_fit_target)
        exclude_row = QWidget()
        exclude_layout = QHBoxLayout(exclude_row)
        exclude_layout.setContentsMargins(0, 0, 0, 0)
        exclude_layout.addWidget(self._exclude_min)
        exclude_layout.addWidget(QLabel("–"))
        exclude_layout.addWidget(self._exclude_max)
        # Relabelled from the bare "Exclude (μs)": the count fit *drops* these
        # bins from the fit (a hard skip), unlike the MaxEnt de-weight window
        # that keeps the FFT grid. The semantics differ; the labels now say so.
        self._exclude_label = QLabel("Skip (μs)")
        self._exclude_label.setToolTip("Interior bins to skip (exclude window, μs)")
        options_form.addRow(self._exclude_label, exclude_row)

        self._t0_check = QCheckBox("Fit t₀ offset")
        self._t0_check.toggled.connect(self._sync_count_fit_target)
        self._baseline_check = QCheckBox("Fit baseline drift")
        self._baseline_check.toggled.connect(self._sync_count_fit_target)
        self._deadtime_check = QCheckBox("Fit deadtime DT₀")
        self._deadtime_check.toggled.connect(self._sync_count_fit_target)
        # Stack the nuisance toggles in a column: a single row of three
        # checkboxes set this dock page's minimum width well past the other Fit
        # tabs on a 13" screen.
        nuisance_row = QWidget()
        nuisance_layout = QVBoxLayout(nuisance_row)
        nuisance_layout.setContentsMargins(0, 0, 0, 0)
        nuisance_layout.setSpacing(2)
        nuisance_layout.addWidget(self._t0_check)
        nuisance_layout.addWidget(self._baseline_check)
        nuisance_layout.addWidget(self._deadtime_check)
        options_form.addRow(QLabel("Nuisances"), nuisance_row)

        # Double-pulse separation (μs); 0 = single pulse. Fixed from the
        # instrument, or located by a coarse->fine scan when "fit" is ticked.
        self._dpsep_spin = QDoubleSpinBox()
        self._dpsep_spin.setDecimals(3)
        self._dpsep_spin.setRange(0.0, 5.0)
        self._dpsep_spin.setSingleStep(0.01)
        self._dpsep_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._dpsep_spin.setMaximumWidth(76)
        self._dpsep_spin.valueChanged.connect(self._sync_count_fit_target)
        self._dpsep_fit_check = QCheckBox("fit")
        self._dpsep_fit_check.setToolTip(
            "Refine dpsep by a coarse→fine scan (the pulse-onset gate defeats gradient fitting)"
        )
        self._dpsep_fit_check.toggled.connect(self._sync_count_fit_target)
        dpsep_row = QWidget()
        dpsep_layout = QHBoxLayout(dpsep_row)
        dpsep_layout.setContentsMargins(0, 0, 0, 0)
        dpsep_layout.addWidget(self._dpsep_spin)
        dpsep_layout.addWidget(self._dpsep_fit_check)
        self._dpsep_label = QLabel("Double pulse (μs)")
        options_form.addRow(self._dpsep_label, dpsep_row)

        # ── Collapsed: calibration (promote fitted count terms → the grouping). ──
        self._calibration_section = CollapsibleSection("Calibration", expanded=False)
        calibration_form = QFormLayout()
        calibration_form.setContentsMargins(0, 0, 0, 0)
        self._calibration_section.addLayout(calibration_form)
        outer.addWidget(self._calibration_section)

        # Promote a fitted deadtime into the grouping correction (Send-to-Group).
        self._promote_btn = QPushButton("Promote DT₀")
        self._promote_btn.setToolTip("Promote the fitted deadtime DT₀ → grouping correction")
        self._promote_btn.clicked.connect(self._on_promote_deadtime)
        self._promote_additive = QCheckBox("accumulate")
        promote_row = QWidget()
        promote_layout = QVBoxLayout(promote_row)
        promote_layout.setContentsMargins(0, 0, 0, 0)
        promote_layout.setSpacing(2)
        promote_layout.addWidget(self._promote_btn)
        promote_layout.addWidget(self._promote_additive)
        self._promote_label = QLabel("Deadtime")
        calibration_form.addRow(self._promote_label, promote_row)

        # The α / t₀ / background promote siblings (suggest-only Send-to-Group):
        # each writes a fitted count-domain calibration into the grouping with
        # provenance, before/after, and a "re-reduce" message.
        self._promote_alpha_btn = QPushButton("α")
        self._promote_alpha_btn.setToolTip("Promote the fitted forward/backward α → grouping")
        self._promote_alpha_btn.clicked.connect(lambda: self._active_tab().promote_count_alpha())
        self._promote_t0_btn = QPushButton("t₀")
        self._promote_t0_btn.setToolTip("Promote the fitted t₀ offset → grouping t0_bin")
        self._promote_t0_btn.clicked.connect(lambda: self._active_tab().promote_count_t0())
        self._promote_bg_btn = QPushButton("background")
        self._promote_bg_btn.setToolTip(
            "Promote the fitted flat background → grouping (fixed mode)"
        )
        self._promote_bg_btn.clicked.connect(lambda: self._active_tab().promote_count_background())
        promote_row2 = QWidget()
        promote_layout2 = QGridLayout(promote_row2)
        promote_layout2.setContentsMargins(0, 0, 0, 0)
        promote_layout2.setHorizontalSpacing(4)
        promote_layout2.setVerticalSpacing(2)
        promote_layout2.addWidget(self._promote_alpha_btn, 0, 0)
        promote_layout2.addWidget(self._promote_t0_btn, 0, 1)
        promote_layout2.addWidget(self._promote_bg_btn, 1, 0, 1, 2)
        promote_layout2.setColumnStretch(2, 1)
        calibration_form.addRow(QLabel("Promote"), promote_row2)

        # N3 interpretive guard: the count fit consumes raw counts, so a grouping
        # background correction does NOT reach it. Surface that when active so a
        # user does not fix the fit's background to zero and bias N0/α. It lives
        # inside Count-fit options (the section that owns the background term) so
        # it does not occupy the always-visible header when collapsed.
        self._bg_active_note = QLabel(
            "Note: this run has a grouping background correction. The count fit "
            "reads raw counts, so its background term measures the full flat "
            "background — do not fix it to zero."
        )
        self._bg_active_note.setWordWrap(True)
        self._bg_active_note.setStyleSheet("color: palette(mid);")
        self._bg_active_note.setVisible(False)
        self._count_options_section.addWidget(self._bg_active_note)
        return box

    def _on_promote_deadtime(self) -> None:
        """Promote the active surface's last fitted deadtime to the grouping."""
        self._active_tab().promote_count_deadtime(additive=self._promote_additive.isChecked())

    def _sync_count_fit_target(self, *_args) -> None:
        """Push the selector state down to both grouped surfaces."""
        mode = self._target_combo.currentData()
        cost = self._cost_combo.currentData()
        side = self._side_combo.currentData()
        single = mode == "single"
        count_mode = mode != "all"
        # The Forward/Backward side only means anything for the single-group
        # target, so its row is shown only then (rather than greyed out).
        self._target_form.setRowVisible(self._side_combo, single)
        # The Poisson/Gaussian cost now applies to every grouped target,
        # including the lifetime-corrected fgAll ("all") fit — its grouped
        # driver routes through the same Cash/√N cost-factory seam.
        self._cost_combo.setEnabled(True)
        for widget in (
            self._exclude_min,
            self._exclude_max,
            self._exclude_label,
            self._t0_check,
            self._baseline_check,
            self._deadtime_check,
            self._dpsep_spin,
            self._dpsep_fit_check,
            self._dpsep_label,
            self._promote_btn,
            self._promote_additive,
            self._promote_label,
            self._promote_alpha_btn,
            self._promote_t0_btn,
            self._promote_bg_btn,
        ):
            widget.setEnabled(count_mode)

        ex_lo = float(self._exclude_min.value())
        ex_hi = float(self._exclude_max.value())
        exclude = (ex_lo, ex_hi) if ex_hi > ex_lo else None
        for tab in (self._single_fit_tab, self._batch_fit_tab):
            tab.set_count_fit_mode(mode)
            tab.set_count_fit_cost(cost)
            tab.set_count_single_side(side)
            tab.set_count_exclude(exclude)
            tab.set_count_fit_t0(self._t0_check.isChecked())
            tab.set_count_baseline(self._baseline_check.isChecked())
            tab.set_count_deadtime(self._deadtime_check.isChecked())
            tab.set_count_dpsep(float(self._dpsep_spin.value()))
            tab.set_count_dpsep_fit(self._dpsep_fit_check.isChecked())

    def _grouped_tabs(self) -> tuple[GlobalFitTab, GlobalFitTab]:
        return (self._single_fit_tab, self._batch_fit_tab)

    def _active_tab(self) -> GlobalFitTab:
        current = self._tabs.currentWidget()
        return current if isinstance(current, GlobalFitTab) else self._single_fit_tab

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Update the active grouped-fit dataset shown by both surfaces.

        Mirrors ``FitPanel.set_dataset``: the leaving run's Single form is saved,
        the surfaces re-bind to *dataset* (rebuilding the grouped model for its
        detector groups), then the Single form is resolved by precedence —
        recorded slot fit (restore provider) > this run's previously-edited form
        > carry-forward (keep the current function, drop the stale result) for an
        unseen, un-customised run.
        """
        # Save the form the Single surface is leaving, so returning to that run
        # restores its in-progress setup.
        if self._active_single_grouped_run is not None:
            self._single_grouped_state_by_run[self._active_single_grouped_run] = (
                self._single_fit_tab.get_state()
            )

        for tab in self._grouped_tabs():
            tab.set_current_dataset(dataset)
        self._update_background_note(dataset)

        run_number: int | None = None
        if dataset is None:
            self._run_label = ""
        else:
            self._run_label = str(getattr(dataset, "run_label", dataset.run_number))
            try:
                run_number = int(dataset.run_number)
            except (TypeError, ValueError):
                run_number = None
        self._active_single_grouped_run = run_number

        if dataset is None:
            return

        payload = (
            self._single_grouped_restore_provider(dataset)
            if self._single_grouped_restore_provider is not None
            else None
        )
        if isinstance(payload, dict) and payload:
            self.restore_single_grouped_ui(payload)
        elif run_number is not None and run_number in self._single_grouped_state_by_run:
            self._single_fit_tab.restore_state(self._single_grouped_state_by_run[run_number])
        else:
            self._carry_forward_single_grouped_form()

    def _carry_forward_single_grouped_form(self) -> None:
        """Inherit the current Single function for an unseen run, dropping its result.

        The grouped model already persists across ``set_current_dataset``, so
        re-applying the current form keeps the function/seeds; the previous run's
        result is then cleared (an unseen run has not been fit). Mirrors
        ``FitPanel._carry_forward_single_fit_form``.
        """
        # restore_state only *sets* a non-empty result_html, so the result widget
        # is cleared directly (mirrors FitPanel clearing the single result label).
        self._single_fit_tab.restore_state(self._single_fit_tab.get_state())
        self._single_fit_tab._result_text.clear()

    def _on_grouped_tab_changed(self, index: int) -> None:
        """Preserve the Single form across a Single↔Batch tab switch.

        Leaving Single snapshots its form; returning restores that snapshot when
        the run is unchanged — keeping a hand-built but unfit grouped function
        alive across the round trip (mirrors ``FitPanel._on_fit_tab_changed``).
        """
        if self._tabs.widget(index) is self._single_fit_tab:
            snapshot = self._single_grouped_form_snapshot
            if snapshot is not None and snapshot.get("run") == self._active_single_grouped_run:
                self._single_fit_tab.restore_state(snapshot["state"])
        else:
            self._single_grouped_form_snapshot = {
                "run": self._active_single_grouped_run,
                "state": self._single_fit_tab.get_state(),
            }

    def _on_send_grouped_model_to_batch(self) -> None:
        """Copy the Single grouped surface's model + physics seeds to the Batch surface."""
        model = getattr(self._single_fit_tab, "_composite_model", None)
        if model is None:
            return
        seeds = self._single_fit_tab.current_grouped_seed_values()
        self._batch_fit_tab._set_composite_model(model)
        self._batch_fit_tab.apply_grouped_physics_seeds(seeds)
        self._tabs.setCurrentWidget(self._batch_fit_tab)

    def share_single_grouped_function_state(
        self,
        source_run: int,
        target_runs: list[int],
        datasets_by_run: dict[int, MuonDataset] | None = None,
    ) -> int:
        """Copy the source run's grouped Single function into each target run's store.

        Each peer inherits the function (model + seeds + Fix/role setup) on its
        next selection via the per-run store, but **not** the source run's fit
        result (a peer has not been fit). Returns the number of peers written.

        Mirrors ``FitPanel.share_single_function_state``: for field-specific
        parameters (like ``B_L``), the peer's own field-derived seed is applied
        from its dataset when *datasets_by_run* is provided, so a peer at a
        different applied field gets its own seed rather than the source run's.
        Both the grouped-fit model parameters (``parameters``) and the per-group
        model parameters (``group_model_parameters``) are re-seeded.
        """
        source = int(source_run)
        if source == self._active_single_grouped_run:
            source_state = self._single_fit_tab.get_state()
        else:
            source_state = self._single_grouped_state_by_run.get(source)
        if not isinstance(source_state, dict) or not source_state:
            return 0
        written = 0
        for target in target_runs:
            try:
                target_run = int(target)
            except (TypeError, ValueError):
                continue
            if target_run == source:
                continue
            shared_state = copy.deepcopy(source_state)
            # The peer has not been fit; carry the function, not the result.
            shared_state["result_html"] = "No fit performed yet"

            # Re-seed file-specific parameters (e.g. B_L) from the peer's own
            # dataset so a peer at a different applied field is not stuck with
            # the source run's field value.
            target_dataset = (
                datasets_by_run.get(target_run) if datasets_by_run is not None else None
            )
            if target_dataset is not None:
                for key in ("parameters", "group_model_parameters"):
                    self._reseed_field_params(shared_state.get(key), target_dataset)

            self._single_grouped_state_by_run[target_run] = shared_state
            # Refresh now only if this peer is the run currently on screen.
            if target_run == self._active_single_grouped_run:
                self._single_fit_tab.restore_state(shared_state)
                self._single_fit_tab._result_text.clear()
            written += 1
        return written

    @staticmethod
    def _reseed_field_params(param_entries: object, dataset: MuonDataset) -> None:
        """Apply *dataset*'s file-specific seeds onto a parameter-entry list.

        Each entry is a ``{"name": ..., "value": ...}`` dict; field-like
        parameters (resolved via :func:`split_parameter_name`) get the dataset's
        own field-derived value. Non-field parameters are left untouched.
        """
        if not isinstance(param_entries, list):
            return
        for param_dict in param_entries:
            if not isinstance(param_dict, dict):
                continue
            pname = param_dict.get("name")
            if not isinstance(pname, str):
                continue
            base_name, _index = split_parameter_name(pname)
            file_value = _get_file_value_for_parameter(dataset, base_name)
            if file_value is not None:
                param_dict["value"] = file_value

    def clear_grouped_single_state(self) -> None:
        """Drop all per-run grouped Single forms (project close / new project).

        Mirrors ``FitPanel.clear`` resetting ``_single_state_by_run`` — without
        this the session-only store bleeds a closed project's forms into the
        next project's runs that reuse the same run numbers.
        """
        self._single_grouped_state_by_run = {}
        self._active_single_grouped_run = None
        self._single_grouped_form_snapshot = None

    def prune_grouped_single_state(self, run_numbers) -> None:
        """Forget stored grouped Single forms for *run_numbers* (removed / refit).

        Mirrors ``FitPanel.clear_fits_for_runs`` so a removed or re-loaded run
        does not resurrect a stale function.
        """
        for raw in run_numbers:
            try:
                run_number = int(raw)
            except (TypeError, ValueError):
                continue
            self._single_grouped_state_by_run.pop(run_number, None)
            if self._active_single_grouped_run == run_number:
                self._active_single_grouped_run = None

    def set_single_grouped_restore_provider(
        self, provider: Callable[[MuonDataset | None], dict | None] | None
    ) -> None:
        """Install the per-run Single-surface restore mediator (or clear it)."""
        self._single_grouped_restore_provider = provider

    def single_grouped_form_state(self) -> dict:
        """Full restorable form state of the Single (grouped) surface.

        This is exactly what :meth:`restore_single_grouped_ui` consumes, so it is
        the payload the main window stores as a grouped single fit's ``ui_state``.
        """
        return copy.deepcopy(self._single_fit_tab.get_state())

    def restore_single_grouped_ui(self, payload: dict | None) -> None:
        """Restore the Single surface from a slot ``ui_state`` payload.

        A populated dict restores the form verbatim; ``None``/empty leaves the
        surface untouched (a run with no grouped single fit must not blank a
        model the user is setting up — there are no projections to confuse here).
        """
        if isinstance(payload, dict) and payload:
            self._single_fit_tab.restore_state(payload)

    def _update_background_note(self, dataset: MuonDataset | None) -> None:
        """Show the N3 interpretive note when the run's grouping corrects background."""
        grouping = getattr(getattr(dataset, "run", None), "grouping", None)
        active = False
        if isinstance(grouping, dict) and bool(grouping.get("background_correction", False)):
            active = resolve_background_mode(grouping) != "none"
        self._bg_active_note.setVisible(active)

    def set_member_datasets(self, datasets: list[MuonDataset]) -> None:
        """Set the member runs for the Batch grouped surface (the series)."""
        self._batch_fit_tab.set_member_datasets(datasets)

    def get_grouped_state(self) -> dict:
        """Return the grouped-fit classification from the active surface."""
        return self._active_tab().get_grouped_state()

    def grouped_simulate_seed_for_run(self, run_number: int) -> dict | None:
        """Return a cached multi-group simulate seed for a run, if available.

        Seeds the Generate Synthetic Run dialog's per-group amplitude/phase
        table from this run's last converged grouped time-domain fit.
        """
        for tab in self._grouped_tabs():
            seed = tab.grouped_simulate_seed_for_run(run_number)
            if seed is not None:
                return seed
        return None

    def update_grouped_phase_seed(self, run_number: int, phases_rad: dict[int, float]) -> bool:
        """Write per-group phases (radians) into the cached grouped fit seed.

        Used by the MaxEnt "Send phases to fit" exchange; returns ``True`` when a
        grouped fit seed on either surface received the phases.
        """
        updated = False
        for tab in self._grouped_tabs():
            if hasattr(tab, "update_grouped_phase_seed") and tab.update_grouped_phase_seed(
                run_number, phases_rad
            ):
                updated = True
        return updated

    def set_fit_range_display(self, x_min: float | None, x_max: float | None) -> None:
        """Update fit-range spinboxes on both surfaces to match the plot range."""
        for tab in self._grouped_tabs():
            tab.set_fit_range_display(x_min, x_max)

    def current_fit_range_text(self) -> str | None:
        """Active grouped-fit range as a provenance string, from the current tab."""
        tab = self._tabs.currentWidget()
        getter = getattr(tab, "current_fit_range_text", None)
        return getter() if callable(getter) else None

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Apply fit blocking rules from the main window context to both surfaces."""
        for tab in self._grouped_tabs():
            tab.set_fit_blocked(blocked, reason)

    def dock_title(self) -> str:
        """Return the preferred fit-dock title for the current grouped dataset."""
        if self._run_label:
            return f"Multi-Group Fit — {self._run_label}"
        return "Multi-Group Fit"

    def grouped_fit_formula_string(self) -> str | None:
        """Return the active grouped-fit formula string, if available."""
        model = getattr(self._active_tab(), "_composite_model", None)
        if model is None:
            return None
        try:
            return str(model.formula_string())
        except Exception:
            return None

    def get_state(self) -> dict:
        """Return serialisable grouped-fit state (both surfaces) for persistence."""
        return {
            "single": self._single_fit_tab.get_state(),
            "batch": self._batch_fit_tab.get_state(),
            # NEW-R1: the count-fit skip window is a window-level control (shared
            # by both surfaces), so it persists here rather than per-tab. It is
            # the last exclusion that did not round-trip through the project.
            "count_skip_window": [
                float(self._exclude_min.value()),
                float(self._exclude_max.value()),
            ],
        }

    def restore_state(self, state: dict) -> None:
        """Restore grouped-fit state from project persistence.

        Accepts the new ``{single, batch}`` shape; a legacy single-surface state
        dict is applied to both surfaces for backward compatibility.
        """
        if not isinstance(state, dict):
            return
        if "single" in state or "batch" in state:
            if isinstance(state.get("single"), dict):
                self._single_fit_tab.restore_state(state["single"])
            if isinstance(state.get("batch"), dict):
                self._batch_fit_tab.restore_state(state["batch"])
        else:
            self._single_fit_tab.restore_state(state)
            self._batch_fit_tab.restore_state(state)
        self._restore_count_skip_window(state.get("count_skip_window"))

    def _restore_count_skip_window(self, window) -> None:
        """Restore the persisted count-fit skip window (NEW-R1); absent → unchanged."""
        if not isinstance(window, (list, tuple)) or len(window) < 2:
            return
        try:
            lo, hi = float(window[0]), float(window[1])
        except (TypeError, ValueError):
            return
        self._exclude_min.setValue(lo)
        self._exclude_max.setValue(hi)
        self._sync_count_fit_target()
