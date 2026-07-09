"""Single-dataset fit tab (``SingleFitTab``).

Split out of ``fit_panel.py`` (Phase 2 mechanical split).

What lives here: ``SingleFitTab(FitTabBase)``, the tab used to fit one
dataset (or one grouped selection in time domain) against a single composite
model. Entry points: ``set_dataset`` feeds the active dataset in;
``current_seed_values`` reads the parameter table's current seeds; the
``_run_fit``/``_on_stop_fit`` pair is the fit-lifecycle boundary (started via
``tab_base._start_fit_call``, results routed back through the panel's
``_on_single_fit_completed``); ``get_state``/``restore_state`` serialize the
tab for project persistence.
"""

import copy
import functools
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import (
    CompositeModel,
    migrate_legacy_fraction_state,
)
from asymmetry.core.fitting.domain_library import coerce_domain
from asymmetry.core.fitting.engine import FitEngine, FitResult
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    FitWizardRecommendation,
    deserialize_fit_wizard_recommendation,
    serialize_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import (
    Parameter,
    ParameterSet,
)
from asymmetry.core.fitting.rrf_offset import (
    UnsupportedRRFComponentError,
    rrf_frequency_offsets,
)
from asymmetry.core.fitting.spectral import (
    default_frequency_model,
    seed_peak_parameters_from_dataset,
)
from asymmetry.gui.panels.fit_function_builder import FitFunctionBuilderDialog
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.metrics import char_width
from asymmetry.gui.styles.widgets import (
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    build_primary_button_qss,
    fit_quality_tooltip,
    make_section_header,
    success_html,
)
from asymmetry.gui.tasks import TaskRunner
from asymmetry.gui.widgets.panel_section import PanelSection
from asymmetry.gui.windows.fit_wizard_window import FitWizardWindow

from .seeding import _field_value_overrides
from .tab_base import (
    FitParameterTable,
    FitTabBase,
    _apply_domain_mismatch_warning,
    _fit_curve_sample_count,
    _fit_domain_mismatch_message,
    _fit_success_html,
    _fit_summary,
    _fit_warnings_html,
    _model_without_trailing_background,
    _normalized_model_param_values,
    _set_formula_label_text,
    _set_param_batch_role_cell,
    _shift_rrf_parameters,
    _start_fit_call,
    _ValueUncertaintyDelegate,
    _wait_for_fit_thread,
    dataset_error_oversampling,
)


class SingleFitTab(FitTabBase):
    """Single dataset fitting interface.

    Provides model selection, parameter configuration, and fit execution for a
    single dataset. Emits signals when fit completes successfully.

    Attributes
    ----------
    fit_completed : Signal
        Emitted with (FitResult, tuple, list) when fit finishes successfully.
        The tuple contains (t_fit, y_fit) arrays for plotting the fit curve,
        and the list contains per-component additive curves as
        (component_name, y_component).

    Methods
    -------
    set_dataset(dataset)
        Set the current dataset to fit.
    """

    fit_completed = Signal(object, object, object)  # (FitResult, fitted_curve, component_curves)
    preview_requested = Signal(
        object, object, object
    )  # (FitResult, fitted_curve, component_curves)
    share_function_with_group_requested = Signal(int)
    send_model_to_batch_requested = Signal()
    add_to_series_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._current_dataset: MuonDataset | None = None
        self._fit_blocked = False
        self._fit_block_reason = ""
        self._fit_engine = FitEngine()
        self._domain = "time"
        self._composite_model = CompositeModel(["Exponential", "Constant"], operators=["+"])
        self._fit_wizard_window: FitWizardWindow | None = None
        self._cached_wizard_recommendation: FitWizardRecommendation | None = None
        self._cached_wizard_signature: dict[str, object] | None = None
        self._cached_wizard_log_text = ""
        # Optional provider of the rotating-frame ν₀ (MHz) supplied by the host
        # window; returns a frequency when an RRF fit should run (the plot's RRF
        # display is active), else None. Default no-op keeps the tab standalone.
        self._rrf_frequency_provider: Callable[[], float | None] = lambda: None
        self._last_fit_result: FitResult | None = None
        self._last_fit_parameters: ParameterSet | None = None
        # See set_has_recorded_fit: whether the active run has a *persisted*
        # single fit, independent of _last_fit_result (this session's transient
        # in-memory result).
        self._has_recorded_fit = False
        self._pull_diagnostic_btn: QPushButton | None = None
        self._pull_diagnostic_window: QWidget | None = None
        #: Background fits run via the shared TaskRunner machinery; the
        #: worker handle exists only so the Stop button can cancel it.
        self._fit_call_runner = TaskRunner(self)
        self._fit_worker = None
        #: Bumped on every model (re)configuration. A fit snapshots it at
        #: launch; a mismatch at completion means the model was changed or
        #: Reset while the fit ran (Reset reuses the same object, so object
        #: identity alone would miss it), so the stale result is not applied.
        self._model_generation = 0

        # Model selection
        model_group = PanelSection("Model")
        model_layout = QFormLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_group.addLayout(model_layout)
        self._build_formula_box()
        self._fit_wizard_btn = QPushButton("Fit Wizard...")
        self._fit_wizard_btn.clicked.connect(self._open_fit_wizard)
        self._fit_wizard_btn.setEnabled(False)

        # The four advanced model actions (Drop background / Share with Group /
        # Send to Batch / Add to Series) are collapsed into a single "⋯ More…"
        # overflow menu instead of four full-width button rows. On a 13-inch
        # screen those rows pushed the PARAMETERS table and the Fit button below
        # the fold; folding them lifts PARAMETERS into view (P1-2). They remain
        # QActions so enable-state and tooltips behave as before.
        self._more_menu = QMenu(self)
        self._more_menu.setToolTipsVisible(True)
        self._drop_background_action = self._more_menu.addAction("Drop background")
        self._drop_background_action.setToolTip(
            "Remove the constant background term from the model.\n"
            "For amplitude calibration (e.g. a light-OFF A₀ run) a free background "
            "absorbs part of the initial asymmetry, splitting the fitted amplitude; "
            "drop it to fit the full A₀ with a single relaxation term."
        )
        self._drop_background_action.triggered.connect(self._on_drop_background)
        self._drop_background_action.setEnabled(False)
        self._share_group_action = self._more_menu.addAction("Share with Group")
        self._share_group_action.setToolTip("Share this fit function with the selected data group.")
        self._share_group_action.triggered.connect(self._on_share_function_with_group)
        self._share_group_action.setEnabled(False)
        self._send_to_batch_action = self._more_menu.addAction("Send to Batch")
        self._send_to_batch_action.setToolTip(
            "Copy this fit function into the Batch tab to seed a batch fit over the selected runs."
        )
        self._send_to_batch_action.triggered.connect(self.send_model_to_batch_requested.emit)
        self._add_to_series_action = self._more_menu.addAction("Add to Series...")
        self._add_to_series_action.triggered.connect(self.add_to_series_requested.emit)
        self._update_add_to_series_enabled()

        self._more_btn = QToolButton()
        self._more_btn.setText("More…")
        self._more_btn.setToolTip("Advanced model actions")
        self._more_btn.setMenu(self._more_menu)
        self._more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        # Single column of natural-width buttons. A side-by-side grid forced the
        # two button columns (~110px each) to set the whole Fit tab's minimum
        # width; stacking them lets the dock get genuinely narrow on a 13" screen,
        # and dropping the Expanding policy keeps each button only as wide as its
        # label needs (left-aligned) instead of stretching to fill the row.
        model_button_layout = QVBoxLayout()
        model_button_layout.setContentsMargins(0, 0, 0, 0)
        model_button_layout.setSpacing(4)
        for _model_btn in (
            self._edit_model_btn,
            self._fit_wizard_btn,
            self._more_btn,
        ):
            model_button_layout.addWidget(_model_btn, 0, Qt.AlignmentFlag.AlignLeft)

        self._formula_row_label = QLabel("A(t):")
        model_layout.addRow(self._formula_row_label, self._formula_box)
        model_layout.addRow("", model_button_layout)
        layout.addWidget(model_group)

        # Fit range section
        fit_range_group = PanelSection("Fit range")
        fit_range_layout = QHBoxLayout()
        fit_range_layout.setContentsMargins(0, 0, 0, 0)
        fit_range_layout.setSpacing(4)
        fit_range_group.addLayout(fit_range_layout)

        self._build_fit_range_fields()

        self._fit_range_mid_label = QLabel("≤ <i>t</i> ≤")
        self._fit_range_mid_label.setTextFormat(Qt.TextFormat.RichText)

        self._fit_range_unit_label = QLabel("µs")

        fit_range_layout.addWidget(self._fit_range_min_spin)
        fit_range_layout.addWidget(self._fit_range_mid_label)
        fit_range_layout.addWidget(self._fit_range_max_spin)
        fit_range_layout.addWidget(self._fit_range_unit_label)
        fit_range_layout.addStretch()
        layout.addWidget(fit_range_group)

        # Parameter table — the shared Name·Value·Fix·Min·Max·Batch·Link·Tie
        # widget (columns/delegates/Fix-Link-Tie wiring/fraction sync live in
        # FitParameterTable). It self-connects itemChanged for fraction sync.
        param_group = PanelSection("Parameters")
        self._param_table = FitParameterTable()
        param_group.addWidget(self._param_table)
        layout.addWidget(param_group)

        # Buttons
        btn_layout = QGridLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setHorizontalSpacing(6)
        btn_layout.setVerticalSpacing(6)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setStyleSheet(build_primary_button_qss())
        self._fit_btn.clicked.connect(self._run_fit)
        # Stop replaces the Fit button while a worker-based fit runs.
        self._build_run_controls("Cancel the running fit; no result is recorded.")
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._reset_parameters)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        self._preview_btn.setEnabled(False)
        self._pull_diagnostic_btn = QPushButton("Pull diagnostic…")
        self._pull_diagnostic_btn.setToolTip(
            "Re-simulate this fit at matched statistics, refit each copy, and "
            "check that the parameter pulls are standard normal (honest errors)."
        )
        self._pull_diagnostic_btn.clicked.connect(self._on_pull_diagnostic)
        self._pull_diagnostic_btn.setEnabled(False)
        self._minos_checkbox = QCheckBox("Asymmetric errors")
        self._minos_checkbox.setToolTip(
            "After fitting, walk the χ² profile of each free parameter to get its "
            "asymmetric +/− 1σ interval (MINOS). Slower than the default symmetric "
            "errors; most useful at low statistics, near parameter bounds, or in "
            "strongly correlated fits where the parabolic error is unreliable."
        )
        btn_layout.addWidget(self._fit_btn, 0, 0)
        btn_layout.addWidget(self._stop_btn, 0, 0)
        btn_layout.addWidget(self._reset_btn, 0, 1)
        btn_layout.addWidget(self._preview_btn, 0, 2)
        btn_layout.addWidget(self._pull_diagnostic_btn, 1, 0, 1, 3)
        btn_layout.addWidget(self._minos_checkbox, 2, 0, 1, 3)
        btn_layout.setColumnStretch(3, 1)
        layout.addLayout(btn_layout)

        # Carry-forward provenance badge (D2/F6): dismissable notice that the
        # form currently shown was NOT fitted for the selected run — it was
        # either carried forward from another run or restored from an
        # in-session cache of an equally-unfit form. Cleared automatically the
        # moment a fit is recorded for this run (see FitPanel._on_single_fit_completed).
        self._carry_forward_badge = QFrame()
        self._carry_forward_badge.setObjectName("carryForwardBadge")
        self._carry_forward_badge.setStyleSheet(
            f"#carryForwardBadge {{ border: 1px solid {tokens.WARN}; border-radius: 4px; }}"
        )
        badge_layout = QHBoxLayout(self._carry_forward_badge)
        badge_layout.setContentsMargins(8, 4, 4, 4)
        badge_layout.setSpacing(4)
        self._carry_forward_badge_label = QLabel("")
        self._carry_forward_badge_label.setWordWrap(True)
        badge_layout.addWidget(self._carry_forward_badge_label, 1)
        self._carry_forward_badge_dismiss_btn = QPushButton("✕")
        self._carry_forward_badge_dismiss_btn.setToolTip("Dismiss")
        self._carry_forward_badge_dismiss_btn.setFixedWidth(char_width(3))
        # Match the flat, muted "✕" chrome used elsewhere (see dock_header.py's
        # close button) instead of the default Qt bezel.
        self._carry_forward_badge_dismiss_btn.setStyleSheet(
            "QPushButton { border: none; background: transparent; padding: 0 4px;"
            f" color: {tokens.TEXT_MUTED}; }}"
            f"QPushButton:hover {{ background-color: {tokens.SURFACE_HI}; border-radius: 3px; }}"
        )
        self._carry_forward_badge_dismiss_btn.clicked.connect(self._carry_forward_badge.hide)
        badge_layout.addWidget(self._carry_forward_badge_dismiss_btn)
        self._carry_forward_badge.hide()
        layout.addWidget(self._carry_forward_badge)

        # Results
        layout.addWidget(make_section_header("Fit Results"))
        self._results_group = QFrame()
        self._results_group.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        results_layout = QVBoxLayout(self._results_group)
        self._result_label = QLabel("No fit performed yet")
        self._result_label.setWordWrap(True)
        results_layout.addWidget(self._result_label)
        layout.addWidget(self._results_group)

        # Spare vertical height pools here, below the results box, instead of
        # being claimed by an expanding parameter table.
        layout.addStretch(1)

        self._set_composite_model(self._composite_model)

    def domain(self) -> str:
        """Return the current fitting domain."""
        return self._domain

    def set_domain(self, domain: str) -> None:
        """Switch labels and default model for time or frequency fitting."""
        normalized = coerce_domain(domain)
        if normalized == self._domain:
            return
        self._domain = normalized
        self._apply_fit_range_domain(self._domain)
        if self._domain == "frequency":
            self._fit_wizard_btn.setEnabled(False)
            self._fit_wizard_btn.setToolTip(
                "Fit Wizard is currently available for time-domain fits."
            )
            self._share_group_action.setEnabled(False)
            self._set_composite_model(default_frequency_model())
        else:
            self._fit_wizard_btn.setToolTip("")
            self._set_composite_model(CompositeModel(["Exponential", "Constant"], operators=["+"]))
        self.set_dataset(self._current_dataset)

    def show_carry_forward_badge(self, text: str) -> None:
        """Show the dismissable "not fitted for this run" provenance notice."""
        self._carry_forward_badge_label.setText(text)
        self._carry_forward_badge.show()

    def clear_carry_forward_badge(self) -> None:
        """Hide the carry-forward provenance notice (e.g. a real fit now exists)."""
        self._carry_forward_badge.hide()

    def set_dataset(self, dataset: MuonDataset | None) -> None:
        """Set the current dataset to fit."""
        self._current_dataset = dataset
        # A fit result belongs to the dataset it was fit on; drop it on change.
        self._last_fit_result = None
        self._last_fit_parameters = None
        # Reset to the safe default; FitPanel.set_dataset calls
        # set_has_recorded_fit(True) right after this when the run's own
        # persisted FitSlot (not just this session's in-memory result) is real.
        self._has_recorded_fit = False
        if self._pull_diagnostic_btn is not None:
            self._pull_diagnostic_btn.setEnabled(False)
        enabled = dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled and self._domain == "time")
        self._share_group_action.setEnabled(dataset is not None and self._domain == "time")
        self._update_add_to_series_enabled()

    def set_has_recorded_fit(self, has_fit: bool) -> None:
        """Track whether the active run has a persisted single fit (F18).

        Distinct from ``_last_fit_result`` (this *session's* in-memory
        ``FitResult``, cleared on every ``set_dataset``): reselecting a run
        that was fitted earlier — this session or a prior one, restored from
        the project model — must not wrongly disable "Add to Series...".
        Driven by ``FitPanel.set_dataset``, which already resolves this via
        its restore-mediator's ``own_slot`` provenance check.
        """
        self._has_recorded_fit = bool(has_fit)
        self._update_add_to_series_enabled()

    def _update_add_to_series_enabled(self) -> None:
        """Enable "Add to Series..." only once this run has a completed fit (F18)."""
        have_fit = (
            self._last_fit_result is not None and self._last_fit_result.success
        ) or self._has_recorded_fit
        self._add_to_series_action.setEnabled(have_fit)
        self._add_to_series_action.setToolTip(
            "Add this run's single fit to an existing batch series with a matching model."
            if have_fit
            else "Fit this run first — there is no completed single fit to add to a series."
        )

    def _can_run_pull_diagnostic(self) -> bool:
        """A successful time-domain fit on a run with histograms is required."""
        return (
            self._domain == "time"
            and self._last_fit_result is not None
            and self._last_fit_result.success
            and self._last_fit_parameters is not None
            and self._current_dataset is not None
            and self._current_dataset.run is not None
            and bool(self._current_dataset.run.histograms)
        )

    def _on_pull_diagnostic(self) -> None:
        """Open the pull-distribution diagnostic for the last converged fit."""
        if not self._can_run_pull_diagnostic():
            return
        from asymmetry.core.simulate import matched_statistics
        from asymmetry.gui.windows.pull_diagnostic_window import (
            PullDiagnosticWindow,
            make_engine_refit,
        )

        dataset = self._current_dataset
        run = dataset.run
        # The generating "truth" is the CONVERGED fit, not the pre-fit guesses:
        # FitEngine.fit does not mutate its input ParameterSet, so
        # _last_fit_parameters still holds the start values. Seed the refit
        # template from result.parameters while keeping the fit's
        # bounds/fixed/link metadata.
        fitted_values = {p.name: float(p.value) for p in self._last_fit_result.parameters}
        refit_template = copy.deepcopy(self._last_fit_parameters)
        for parameter in refit_template:
            if parameter.name in fitted_values:
                parameter.value = fitted_values[parameter.name]
        truth = {p.name: float(p.value) for p in refit_template}
        free = [p.name for p in refit_template if not getattr(p, "fixed", False)]
        time_range = (float(dataset.time.min()), float(dataset.time.max()))
        refit = make_engine_refit(
            self._composite_model, refit_template, t_min=time_range[0], t_max=time_range[1]
        )
        # Matched statistics: split the run's flat background off the gross
        # count so background is regenerated as background, not as extra signal.
        signal_events, background_per_bin = matched_statistics(run)
        window = PullDiagnosticWindow(
            template=run,
            model=self._composite_model,
            parameters=truth,
            refit=refit,
            track=free or list(truth),
            total_events=signal_events,
            background_per_bin=background_per_bin,
            time_range=time_range,
            parent=self,
        )
        self._pull_diagnostic_window = window
        window.show()

    def set_rrf_frequency_provider(self, provider: Callable[[], float | None]) -> None:
        """Install the host's rotating-frame ν₀ provider (see __init__)."""
        self._rrf_frequency_provider = provider or (lambda: None)

    def set_fit_blocked(self, blocked: bool, reason: str = "") -> None:
        """Enable/disable single-fit actions while preserving the active dataset."""
        self._fit_blocked = bool(blocked)
        self._fit_block_reason = str(reason)
        enabled = self._current_dataset is not None and (not self._fit_blocked)
        self._fit_btn.setEnabled(enabled)
        self._preview_btn.setEnabled(enabled)
        self._fit_wizard_btn.setEnabled(enabled and self._domain == "time")
        tooltip = self._fit_block_reason if self._fit_blocked else ""
        self._fit_btn.setToolTip(tooltip)
        self._preview_btn.setToolTip(tooltip)
        self._fit_wizard_btn.setToolTip(tooltip)

    def _wizard_context_signature(self) -> dict[str, object]:
        return {
            "run_number": (
                int(self._current_dataset.run_number)
                if self._current_dataset is not None
                and getattr(self._current_dataset, "run_number", None) is not None
                else None
            ),
            "model": self._composite_model.to_dict(),
        }

    def _wizard_base_signature_matches(
        self,
        cached_signature: dict[str, object] | None,
        current_signature: dict[str, object],
    ) -> bool:
        if not isinstance(cached_signature, dict):
            return False
        cached_model = cached_signature.get("model")
        return cached_signature.get("run_number") == current_signature.get("run_number") and (
            cached_model is None or cached_model == current_signature.get("model")
        )

    def _cache_wizard_analysis(
        self,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object],
        log_text: str = "",
    ) -> None:
        self._cached_wizard_recommendation = recommendation
        self._cached_wizard_signature = copy.deepcopy(signature)
        self._cached_wizard_log_text = str(log_text)

    def _on_fit_wizard_analysis_cached(
        self,
        recommendation: object,
        log_text: str,
        signature: object,
    ) -> None:
        if not isinstance(recommendation, FitWizardRecommendation) or not isinstance(
            signature, dict
        ):
            return
        self._cache_wizard_analysis(recommendation, signature=signature, log_text=log_text)

    def _on_share_function_with_group(self) -> None:
        """Request sharing the active single-fit function with the current data group."""
        if self._current_dataset is None:
            return
        try:
            run_number = int(self._current_dataset.run_number)
        except (TypeError, ValueError):
            return
        self.share_function_with_group_requested.emit(run_number)

    def _update_drop_background_enabled(self) -> None:
        """Enable the Drop-background affordance only when there is one to drop."""
        reduced = _model_without_trailing_background(self._composite_model)
        self._drop_background_action.setEnabled(self._domain == "time" and reduced is not None)

    def _on_drop_background(self) -> None:
        """Drop the constant background term for amplitude calibration."""
        reduced = _model_without_trailing_background(self._composite_model)
        if reduced is None:
            return
        self._set_composite_model(reduced)

    def _set_composite_model(self, model: CompositeModel, *, seed_frequency: bool = True) -> None:
        """Set the active composite model and rebuild the parameter table.

        Restore paths pass ``seed_frequency=False`` because restored parameter
        values are replayed by ``restore_parameters`` and must not be
        re-derived (and the restore-time dataset may still be the previous
        domain's).
        """
        self._composite_model = model
        # Any model (re)build — including Reset, which reuses the same object —
        # invalidates an in-flight fit's table/diagnostic write-back. (The table
        # drops auxiliary non-model params from a prior restore in populate().)
        self._model_generation += 1
        _set_formula_label_text(self._formula_label, model.formula_string())
        _apply_domain_mismatch_warning(self._formula_label, model, self._domain)

        dataset_field = (
            self._current_dataset.run.field
            if self._current_dataset is not None and self._current_dataset.run is not None
            else 0.0
        )
        value_overrides = dict(_field_value_overrides(model, dataset_field))
        if seed_frequency and self._domain == "frequency" and self._current_dataset is not None:
            # Frequency-domain peak seeds take precedence over the field seed.
            value_overrides.update(seed_peak_parameters_from_dataset(self._current_dataset, model))

        # shape_factor_a (instrument normalisation) is held by default alongside
        # the model's declared fixed-by-default parameters.
        fixed_names = set(model.fixed_by_default_params()) | {"shape_factor_a"}
        self._param_table.populate(model, value_overrides=value_overrides, fixed_names=fixed_names)
        self._update_drop_background_enabled()

    def reseed_frequency_peaks(self) -> None:
        """Re-derive frequency peak seeds from the current spectrum, in place.

        The peak position/height/width/background are field-dependent, so the
        default GaussianPeak seed computed while switching to the frequency
        domain (against whatever dataset was then current) is stale once the
        real spectrum is bound — and carry-forward/session-restore replay that
        stale seed verbatim, leaving ``nu0`` far off the displayed axis so a
        preview shows only the background. Recompute against ``_current_dataset``
        and write just the seed values, preserving the model, fixed flags and
        bounds. A no-op outside the frequency domain or with no dataset/model.
        """
        if self._domain != "frequency" or self._current_dataset is None:
            return
        if self._composite_model is None:
            return
        seeds = seed_peak_parameters_from_dataset(self._current_dataset, self._composite_model)
        self._param_table.apply_value_seeds(seeds)

    def _synchronize_fraction_value_rows(self, edited_param_name: str | None = None) -> None:
        self._param_table.synchronize_fractions(edited_param_name)

    @property
    def _updating_fraction_values(self) -> bool:
        """Proxy the parameter table's bulk-write guard.

        The table owns the guard now; existing call sites that wrap programmatic
        cell writes (fit-result write-back, RRF shift, restore) toggle this and
        the suppression still works because both read the same flag.
        """
        tbl = getattr(self, "_param_table", None)
        return bool(tbl.is_updating) if tbl is not None else False

    @_updating_fraction_values.setter
    def _updating_fraction_values(self, value: bool) -> None:
        tbl = getattr(self, "_param_table", None)
        if tbl is not None:
            tbl._updating = bool(value)

    def _edit_function(self) -> None:
        """Launch the fit-function builder dialog."""
        dialog = FitFunctionBuilderDialog(
            self, initial_model=self._composite_model, domain=self._domain
        )
        if dialog.exec():
            new_model = dialog.get_composite_model()
            if new_model is not None:
                self._set_composite_model(new_model)

    def _open_fit_wizard(self) -> None:
        """Launch or refresh the non-modal fit wizard window."""
        if self._current_dataset is None:
            QMessageBox.information(
                self, "Fit Wizard", "Select a dataset before opening the fit wizard."
            )
            return
        if self._domain == "frequency":
            QMessageBox.information(
                self,
                "Fit Wizard",
                "Fit Wizard is currently available for time-domain fits.",
            )
            return
        if self._fit_blocked:
            message = (
                self._fit_block_reason or "Fit actions are unavailable for the current selection."
            )
            QMessageBox.information(self, "Fit Wizard", message)
            return

        if self._fit_wizard_window is None:
            self._fit_wizard_window = FitWizardWindow(self)
            self._fit_wizard_window.apply_assessment_requested.connect(
                self._apply_fit_wizard_assessment
            )
            self._fit_wizard_window.analysis_cached.connect(self._on_fit_wizard_analysis_cached)

        signature = self._wizard_context_signature()

        self._fit_wizard_window.set_analysis_context(
            self._current_dataset,
            current_model=self._composite_model,
        )
        if self._cached_wizard_recommendation is not None and self._wizard_base_signature_matches(
            self._cached_wizard_signature, signature
        ):
            self._fit_wizard_window.set_cached_recommendation(
                self._cached_wizard_recommendation,
                signature=self._cached_wizard_signature,
                log_text=self._cached_wizard_log_text,
            )
        self._fit_wizard_window.show()
        self._fit_wizard_window.raise_()
        self._fit_wizard_window.activateWindow()

    def _reset_parameters(self) -> None:
        """Reset parameters to model defaults."""
        self._set_composite_model(self._composite_model)

    def _apply_fit_wizard_assessment(
        self,
        assessment: CandidateAssessment,
        recommendation: FitWizardRecommendation,
    ) -> None:
        """Apply a fit-wizard assessment back into the single-fit tab."""
        if self._current_dataset is None:
            return
        if not isinstance(assessment, CandidateAssessment):
            return

        result = assessment.fit_result
        if not result.success:
            self._result_label.setText(f"<b>Fit Wizard failed:</b> {result.message}")
            return

        self._set_composite_model(assessment.template.model)
        fitted_by_name = {parameter.name: parameter for parameter in result.parameters}
        display_values = _normalized_model_param_values(
            self._composite_model,
            {parameter.name: parameter.value for parameter in result.parameters},
        )
        self._updating_fraction_values = True

        for row in range(self._param_table.rowCount()):
            name_item = self._param_table.item(row, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                continue
            fitted = fitted_by_name.get(param_name)
            if fitted is None:
                continue

            value_item = self._param_table.item(row, 1)
            if value_item is not None:
                value_item.setText(f"{display_values.get(param_name, fitted.value):.6f}")
                unc = result.uncertainties.get(param_name, None)
                value_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)
                value_item.setData(
                    _ValueUncertaintyDelegate._MINOS_ROLE,
                    (result.minos_errors or {}).get(param_name),
                )

            min_item = self._param_table.item(row, 3)
            if min_item is not None:
                min_item.setText("-inf" if not np.isfinite(fitted.min) else f"{fitted.min:g}")

            max_item = self._param_table.item(row, 4)
            if max_item is not None:
                max_item.setText("inf" if not np.isfinite(fitted.max) else f"{fitted.max:g}")

            fix_widget = self._param_table.cellWidget(row, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox) if fix_widget else None
            if fix_checkbox is not None:
                fix_checkbox.setChecked(bool(fitted.fixed))
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        wizard_note = f"Fit Wizard — {assessment.template.title}"
        if assessment.residual_gate_reasons:
            wizard_note += " ⚠"
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        detail = _fit_success_html(result).split("<br>", 1)[1]
        self._result_label.setText(success_html(wizard_note, detail=detail))

        param_dict = {parameter.name: parameter.value for parameter in result.parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)
        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )
        self.fit_completed.emit(result, (t_fit, y_fit), component_curves)

    def _on_preview(self) -> None:
        """Generate and emit a preview fit curve with current parameters."""
        if self._fit_blocked:
            return

        if self._current_dataset is None:
            return

        if self._composite_model is None:
            return

        # Build parameter set from table
        parameters = ParameterSet()
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else f"param_{i}"

            # Parse value
            try:
                value = float(self._param_table.item(i, 1).text())
            except (ValueError, AttributeError):
                return

            # Check if fixed
            fix_widget = self._param_table.cellWidget(i, 2)
            fix_checkbox = fix_widget.findChild(QCheckBox)
            fixed = fix_checkbox.isChecked() if fix_checkbox else False

            # Parse bounds
            try:
                min_text = self._param_table.item(i, 3).text()
                min_val = float(min_text) if min_text and min_text != "-inf" else -float("inf")
            except (ValueError, AttributeError):
                min_val = -float("inf")

            try:
                max_text = self._param_table.item(i, 4).text()
                max_val = float(max_text) if max_text and max_text != "inf" else float("inf")
            except (ValueError, AttributeError):
                max_val = float("inf")

            param = Parameter(
                name=param_name,
                value=value,
                min=min_val,
                max=max_val,
                fixed=fixed,
            )
            parameters.add(param)

        param_dict = {p.name: p.value for p in parameters}
        n_samples = _fit_curve_sample_count(
            self._composite_model,
            param_dict,
            float(self._current_dataset.time.min()),
            float(self._current_dataset.time.max()),
        )
        # Generate fitted curve for plotting
        t_fit = np.linspace(
            self._current_dataset.time.min(),
            self._current_dataset.time.max(),
            n_samples,
        )
        y_fit = self._composite_model.function(t_fit, **param_dict)

        component_curves = self._composite_model.evaluate_components(
            t_fit,
            additive_only=True,
            **param_dict,
        )

        # Create a dummy result object for preview (not a real fit)
        preview_result = object()
        self.preview_requested.emit(preview_result, (t_fit, y_fit), component_curves)

    def model_and_seed(self) -> tuple[CompositeModel, ParameterSet]:
        """Return the active single-fit model and its current parameter seed.

        For headless fits (e.g. the Data Browser's "Re-fit as co-added") that
        reuse the configured single-fit model without touching the form. Raises
        :class:`ValueError` on a malformed parameter value, like the fit run.
        """
        return self._composite_model, self._parameter_set_from_table()

    def _parameter_set_from_table(self) -> ParameterSet:
        """Build a :class:`ParameterSet` from the parameter table.

        Raises :class:`ValueError` with a user-facing message on a malformed
        value (the only hard error; bad bounds fall back to ±inf). Shared by
        the fit run and the pull-distribution diagnostic.
        """
        return self._param_table.read_parameter_set()

    def current_seed_values(self) -> dict[str, str]:
        """Return the live parameter-table seed text keyed by parameter name.

        Used to seed the Batch tab from the current single-fit values rather than
        model defaults / stale state; non-finite cells are skipped.
        """
        return self._param_table.current_seed_values()

    def current_bounds(self) -> dict[str, str]:
        """Return the live ``"min, max"`` bounds text keyed by parameter name.

        Carries parameter bounds (not just seed values) when sending a model to
        the Batch tab, so a min/max set in the Single tab is not silently lost.
        """
        return self._param_table.current_bounds()

    def _run_fit(self) -> None:
        """Execute the fit."""
        if self._fit_blocked:
            message = self._fit_block_reason or "Fit is unavailable for the current selection."
            self._result_label.setText(f"ERROR: {message}")
            return

        if self._current_dataset is None:
            self._result_label.setText("ERROR: No dataset selected")
            return

        mismatch = _fit_domain_mismatch_message(self._domain, self._current_dataset)
        if mismatch is not None:
            self._result_label.setText(f"ERROR: {mismatch}")
            return

        if self._composite_model is None:
            self._result_label.setText("ERROR: No function defined")
            return

        missing = getattr(self._composite_model, "missing_component_names", ())
        if missing:
            self._result_label.setText(
                "ERROR: the model requires missing user function(s): "
                f"{', '.join(missing)}. Register them (Setup → User functions…) "
                "and reload the project."
            )
            return

        # Build parameter set from table
        try:
            parameters = self._parameter_set_from_table()
        except ValueError as exc:
            self._result_label.setText(f"ERROR: {exc}")
            return

        # Resolve the rotating-reference-frame offset, if the host's RRF display
        # is active. The fit then consumes RAW lab-frame data with the model's
        # rotation frequencies offset by ν₀, so it keeps exact per-bin
        # statistics while the engine's free parameter is the small, better-
        # conditioned δν; the parameter table stays lab-frame throughout.
        model = self._composite_model
        rrf_offsets: dict[str, float] | None = None
        rrf_nu0 = self._rrf_frequency_provider()
        if rrf_nu0:
            try:
                rrf_offsets = rrf_frequency_offsets(model, float(rrf_nu0))
            except UnsupportedRRFComponentError as exc:
                # A composite with an oscillating component that is not a pure
                # frame rotation (muonium, Bessel, …) cannot be safely offset;
                # refuse rather than silently leave a line in the lab frame.
                self._result_label.setText(
                    f"ERROR: cannot fit in the rotating frame — {exc} "
                    "Turn off the rotating frame (Options → Advanced) to fit this model."
                )
                return
            except ValueError:
                # No rotation component at all (e.g. a pure relaxation model):
                # the rotating frame does not apply; fit normally.
                rrf_offsets = None

        fit_seed = (
            _shift_rrf_parameters(parameters, rrf_offsets, sign=-1) if rrf_offsets else parameters
        )

        # Run the fit on a worker thread; the GUI (and Stop button) stay live.
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_label.setText("Fitting...")

        # Snapshot launch-time context: the user may switch run or model while
        # the worker runs, and the result must be interpreted against what was
        # actually fitted. The TaskRunner relay invokes these closures on the
        # GUI thread with each launch's own context, so a late result can
        # never be applied against a different launch's snapshot.
        dataset = self._current_dataset
        # Only thread the RRF offset when one is active, so the ordinary fit
        # path (and its test doubles) is unchanged.
        fit_kwargs: dict = {"minos": self._minos_checkbox.isChecked()}
        if rrf_offsets:
            fit_kwargs["frequency_offsets"] = rrf_offsets
        # Zero-padded FFT spectra carry correlated samples; thread the
        # effective-sample-size correction. Conditional for the same
        # test-double reason as above (time-domain fits never see the kwarg).
        oversampling = dataset_error_oversampling(dataset)
        if oversampling > 1.0:
            fit_kwargs["error_oversampling"] = oversampling
        self._fit_worker = _start_fit_call(
            self,
            functools.partial(
                self._fit_engine.fit,
                dataset,
                model.function,
                fit_seed,
                **fit_kwargs,
            ),
            on_finished=(
                lambda result, p=parameters, d=dataset, m=model, g=self._model_generation, off=rrf_offsets, nu0=rrf_nu0: (
                    self._apply_single_fit_result(result, p, d, m, g, rrf_offsets=off, rrf_nu0=nu0)
                )
            ),
            on_error=self._on_single_fit_error,
            on_cancelled=self._on_single_fit_cancelled,
        )
        self._set_fit_busy(True)

    def _set_fit_busy(self, busy: bool) -> None:
        """Swap the Fit button for a Stop button (and back) around a worker fit."""
        self._toggle_fit_stop_buttons(busy)
        if not busy:
            # Re-derive from the gating contract rather than force-enable: the
            # run may have been removed or the panel blocked while the fit ran.
            self._fit_btn.setEnabled(self._current_dataset is not None and not self._fit_blocked)

    def _on_stop_fit(self) -> None:
        """Request cancellation of the active worker-based fit."""
        worker = self._fit_worker
        if worker is not None:
            self._stop_btn.setEnabled(False)
            self._result_label.setText("Cancelling fit…")
            worker.cancel()

    def _on_single_fit_cancelled(self) -> None:
        """Handle a cancelled single fit: restore the panel, record nothing."""
        self._set_fit_busy(False)
        self._fit_worker = None
        self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        self._result_label.setText("Fit cancelled — no result recorded.")

    def _on_single_fit_error(self, message: str) -> None:
        self._set_fit_busy(False)
        self._fit_worker = None
        self._result_label.setText(f"<b>Error during fit:</b><br>{message}")

    def shutdown_workers(self) -> None:
        """Cancel any running fit and wait for its thread (window close)."""
        self._fit_call_runner.shutdown()

    def wait_for_fit(self, timeout_ms: int = 30_000) -> bool:
        """Block (with a live event loop) until the launched fit completes."""
        return _wait_for_fit_thread(self, timeout_ms)

    def _apply_single_fit_result(
        self,
        result,
        parameters,
        dataset,
        model,
        model_generation,
        *,
        rrf_offsets=None,
        rrf_nu0=None,
    ) -> None:
        """Apply a completed single fit to the panel (GUI thread)."""
        self._set_fit_busy(False)
        self._fit_worker = None

        if not result.success:
            self._results_group.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
            self._result_label.setText(f"<b>Fit failed:</b> {result.message}")
            return

        # The engine fitted the rotating-frame offsets δν; shift the result back
        # to the lab frame (ν = δν + ν₀, bounds with it) so every downstream
        # surface — the parameter table, the overlay curve drawn on raw data,
        # the recorded FitSlot, the pull diagnostic — reads in the lab frame. χ²,
        # uncertainties and covariance are frame-invariant (the offset is an
        # additive constant), so only the values/bounds move.
        rrf_note = ""
        if rrf_offsets:
            result.parameters = _shift_rrf_parameters(result.parameters, rrf_offsets, sign=+1)
            rrf_note = (
                "<br><i>frame: ν_RRF = "
                f"{float(rrf_nu0):.4f} MHz — fitted in the rotating frame; "
                "frequencies reported in the lab frame.</i>"
            )

        # A result is only "fresh" when the panel still shows the model AND run
        # it was fitted on. Otherwise the user navigated away mid-fit: applying
        # the values would corrupt a different model's seed table, arm the pull
        # diagnostic against the wrong run, overlay the curve on the wrong plot,
        # or record a FitSlot for the wrong run. The generation counter catches
        # Reset (which reuses the same model object, so identity alone misses).
        model_unchanged = (
            self._composite_model is model and self._model_generation == model_generation
        )
        dataset_unchanged = self._current_dataset is dataset

        warnings_note = _fit_warnings_html(result)
        self._results_group.setStyleSheet(RESULT_BOX_SUCCESS_STYLE)
        self._result_label.setText(_fit_success_html(result) + rrf_note + warnings_note)
        summary = _fit_summary(result)
        self._result_label.setToolTip(
            fit_quality_tooltip(summary.get("quality"), summary.get("params_at_bound"))
        )

        if not (model_unchanged and dataset_unchanged):
            if not model_unchanged:
                reason = "the model was changed or reset while it ran"
            else:
                run_id = dataset.metadata.get("run_number", "?")
                reason = f"run {run_id} is no longer selected"
            self._result_label.setText(
                _fit_success_html(result)
                + rrf_note
                + warnings_note
                + f"<br><i>This fit was not applied or recorded because {reason}. "
                "Restore the original model and run, then refit to keep it.</i>"
            )
            return

        # Fresh: remember the converged fit for the pull-distribution diagnostic.
        self._last_fit_result = result
        self._last_fit_parameters = parameters
        if self._pull_diagnostic_btn is not None:
            self._pull_diagnostic_btn.setEnabled(self._can_run_pull_diagnostic())
        self._update_add_to_series_enabled()

        display_values = _normalized_model_param_values(
            model,
            {parameter.name: parameter.value for parameter in result.parameters},
        )

        # Update table with fit results.
        minos_errors = result.minos_errors or {}
        self._updating_fraction_values = True
        for i in range(self._param_table.rowCount()):
            name_item = self._param_table.item(i, 0)
            param_name = name_item.data(Qt.ItemDataRole.UserRole) if name_item else None
            if not isinstance(param_name, str):
                param_name = name_item.text() if name_item else ""
            if param_name in result.parameters:
                fitted_value = display_values.get(param_name, result.parameters[param_name].value)
                val_item = self._param_table.item(i, 1)
                val_item.setText(f"{fitted_value:.6f}")
                unc = result.uncertainties.get(param_name, None)
                val_item.setData(_ValueUncertaintyDelegate._UNC_ROLE, unc)
                val_item.setData(
                    _ValueUncertaintyDelegate._MINOS_ROLE, minos_errors.get(param_name)
                )
                # A fresh single fit supersedes any piped-back batch role.
                _set_param_batch_role_cell(self._param_table, i, None)
        self._updating_fraction_values = False
        self._synchronize_fraction_value_rows()

        param_dict = {p.name: p.value for p in result.parameters}
        n_samples = _fit_curve_sample_count(
            model,
            param_dict,
            float(dataset.time.min()),
            float(dataset.time.max()),
        )
        t_fit = np.linspace(dataset.time.min(), dataset.time.max(), n_samples)
        y_fit = model.function(t_fit, **param_dict)
        component_curves = model.evaluate_components(t_fit, additive_only=True, **param_dict)
        self.fit_completed.emit(result, (t_fit, y_fit), component_curves)

    # ── project state helpers ──────────────────────────────────────────

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the single-fit tab state."""
        if self._fit_wizard_window is not None:
            recommendation = self._fit_wizard_window.current_recommendation()
            if recommendation is not None:
                signature = self._cached_wizard_signature
                if not isinstance(signature, dict):
                    signature = self._wizard_context_signature()
                self._cache_wizard_analysis(
                    recommendation,
                    signature=signature,
                    log_text=self._fit_wizard_window.current_log_text(),
                )

        state = {
            "model_name": "Composite",
            "composite_model": self._composite_model.to_dict(),
            # The table serialises its rows (incl. auxiliary non-model params and
            # fraction-value normalisation).
            "parameters": self._param_table.parameters_state(),
            "result_html": self._result_label.text(),
        }
        if (
            self._cached_wizard_recommendation is not None
            and self._cached_wizard_signature is not None
        ):
            state["wizard_state"] = {
                "signature": copy.deepcopy(self._cached_wizard_signature),
                "recommendation": serialize_fit_wizard_recommendation(
                    self._cached_wizard_recommendation
                ),
                "log_text": self._cached_wizard_log_text,
            }
        return state

    def restore_state(self, state: dict) -> None:
        """Restore single-fit tab state from a saved dict."""
        self._cached_wizard_recommendation = None
        self._cached_wizard_signature = None
        self._cached_wizard_log_text = ""

        # Migrate legacy ``fraction_<k>`` parameter entries (pre-rework projects)
        # to the n-1 free-fraction scheme before restoring the table rows.
        state = migrate_legacy_fraction_state(state)

        composite_data = state.get("composite_model")
        if isinstance(composite_data, dict):
            try:
                # Unregistered names (a user function whose plugin is not
                # installed) materialise as named zero-valued placeholders so
                # the saved model is never silently replaced; only structurally
                # malformed data falls back to the default model.
                restored = CompositeModel.from_dict(composite_data, allow_missing=True)
            except ValueError:
                fallback = (
                    default_frequency_model()
                    if self._domain == "frequency"
                    else CompositeModel(["Exponential", "Constant"], operators=["+"])
                )
                self._set_composite_model(fallback, seed_frequency=False)
            else:
                self._set_composite_model(restored, seed_frequency=False)
                if restored.missing_component_names:
                    names = ", ".join(restored.missing_component_names)
                    self._result_label.setText(
                        f"<b>Missing user function(s):</b> {names}.<br>"
                        "The saved model is preserved (missing components plot as "
                        "zero) but cannot be fitted until they are registered — "
                        "see Setup → User functions…"
                    )

        # The table applies the saved values/fix/bounds/link/tie onto its rows
        # (the model was just rebuilt by _set_composite_model) and re-establishes
        # auxiliary non-model parameters that have no row.
        params_data = {p["name"]: p for p in state.get("parameters", []) if isinstance(p, dict)}
        self._param_table.restore_parameters(params_data)

        result_html = state.get("result_html")
        if isinstance(result_html, str) and result_html:
            self._result_label.setText(result_html)

        wizard_state = state.get("wizard_state")
        if isinstance(wizard_state, dict):
            recommendation = deserialize_fit_wizard_recommendation(
                wizard_state.get("recommendation")
            )
            signature = wizard_state.get("signature")
            if recommendation is not None and isinstance(signature, dict):
                self._cached_wizard_recommendation = recommendation
                self._cached_wizard_signature = copy.deepcopy(signature)
                self._cached_wizard_log_text = str(wizard_state.get("log_text", ""))
