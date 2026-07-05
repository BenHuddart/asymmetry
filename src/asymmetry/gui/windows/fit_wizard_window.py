"""Non-modal guided fit wizard for single time-domain asymmetry spectra.

The window is a three-state, answer-first shell built on
:class:`~asymmetry.gui.windows.wizard_base.WizardWindowBase` via the
``_build_central()`` hook (so ``self._tabs`` stays ``None`` and no tab
scaffolding is created). The three states live in a ``QStackedWidget``:

* **Welcome** — a plain explanation, the run-context line, a prominent
  *Analyze* button, and a collapsed *Guide the analysis (optional)* section
  housing the existing scope selector + FFT/user-peak seeding UI.
* **Running** — the decision trail streams stage headlines as the core reports
  progress; Cancel stays visible (base chrome).
* **Result** — the answer card (verdict + confidence + overlay plot + apply +
  alternatives) above the six-step decision trail, whose steps expand inline to
  the re-parented deep panels (scope view, FFT+peaks, compare table). A
  *Copy analysis log* and a *Re-analyze* affordance sit alongside.

All verdict/confidence/no-structure prose comes from
``asymmetry.core.fitting.wizard_narrative`` — never re-worded here. The deep
panels are built once and re-parented between the Welcome guidance section and
the Result trail expansions (only one state is visible at a time). The external
surface (``set_analysis_context``, ``set_cached_recommendation``, the
``analysis_cached``/``apply_assessment_requested`` signals, and the four-key
``_analysis_signature``) is unchanged so ``single_tab.py`` needs no edits.
"""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import COMPONENTS, CompositeModel
from asymmetry.core.fitting.engine import FitCancelledError
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    FitWizardRecommendation,
    SelectionMetric,
    build_fit_wizard_recommendation,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.wizard_narrative import (
    build_wizard_trail,
    render_log_text,
)
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    estimate_screening_cost,
    resolve_scope_for_dataset,
)
from asymmetry.core.fourier.fft import fft_asymmetry
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import build_primary_button_qss, make_warning_banner
from asymmetry.gui.widgets.collapsible_section import CollapsibleSection
from asymmetry.gui.widgets.decision_trail import DecisionTrail, TrailSeparator
from asymmetry.gui.widgets.screen_sizing import resize_to_available
from asymmetry.gui.widgets.wizard_answer_card import WizardAnswerCard
from asymmetry.gui.widgets.wizard_scope_selector import WizardScopeSelector
from asymmetry.gui.windows.wizard_base import WizardWindowBase

#: Short human-readable labels for multiplet-match kinds shown in the peaks table.
_MULTIPLET_KIND_LABELS = {
    "larmor": "Larmor line",
    "muonium_low_tf": "muonium low-TF doublet",
    "muonium_high_tf": "muonium high-TF doublet",
    "muonium_zf": "muonium ZF triplet",
    "fmuf_linear": "F-mu-F triplet",
    "muf": "mu-F triplet",
}

#: Stacked-widget page indices.
_PAGE_WELCOME = 0
_PAGE_RUNNING = 1
_PAGE_RESULT = 2

#: Progress-message prefix → trail step key. Matched case-insensitively by
#: prefix so the core's coarse two-message fallback and any finer emits both
#: light the right step; an unmatched message only updates the status line.
_PROGRESS_STEP_PREFIXES: tuple[tuple[str, str], ...] = (
    ("stage 1", "families"),
    ("screening", "families"),
    ("spectral search", "spectrum"),
    ("expanding", "candidates"),
    ("stage 2", "candidates"),
    ("fitting", "candidates"),
)


class FitWizardWindow(WizardWindowBase):
    """Present a guided workflow for model recommendation and comparison."""

    apply_assessment_requested = Signal(
        object, object
    )  # CandidateAssessment, FitWizardRecommendation
    analysis_cached = Signal(object, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        # WizardWindowBase.__init__ builds the shared frame (heading/status/
        # controls row, TaskRunner, progress UI) and calls _build_central()
        # before this body resumes.
        super().__init__(parent)
        self.setWindowTitle("Fit Wizard")
        # Cap the default to the available screen so the title bar never opens
        # clipped above the menu bar on a 13-inch laptop (~800 px high).
        resize_to_available(self, 1180, 740)

        self._heading_label.setText("Fit Wizard")
        self._status_label.setText(
            "Open the fit wizard on a single spectrum to fingerprint the data and "
            "compare curated candidate models."
        )
        self._show_welcome()

    # ------------------------------------------------------------------
    # Content region (overrides the base tab scaffolding)
    # ------------------------------------------------------------------

    def _build_central(self) -> QWidget:
        """Build the three-state stacked content region.

        Runs during ``WizardWindowBase.__init__`` (before this subclass body
        resumes), so it initialises every result-state member the old
        ``_build_tabs`` used to, then constructs the Welcome/Running/Result
        pages and the deep panels re-parented between them.
        """
        # --- Result / analysis state ---
        self._dataset: MuonDataset | None = None
        self._current_model: CompositeModel | None = None
        self._recommendation: FitWizardRecommendation | None = None
        self._selected_key: str | None = None
        self._analysis_stale = False
        self._user_peaks: list[dict] = []
        self._fft_ax = None
        self._user_peak_artists: list = []
        self._peak_click_candidate: tuple[float, float, float] | None = None

        # --- Deep panels (built once, re-parented between states) ---
        self._scope_panel = self._build_scope_panel()
        self._fingerprint_panel = self._build_fingerprint_panel()
        self._compare_panel = self._build_compare_panel()

        # --- Analyze / Cancel wiring on the base controls row ---
        # The Start button is kept as ``_refresh_btn`` for parity with the base
        # driver and the tests; its label tracks Start/Re-run. It lives on the
        # Welcome page, not the controls row.
        self._refresh_btn = QPushButton("Analyze")
        self._refresh_btn.clicked.connect(self._start_analysis)
        self._progress_label.setStyleSheet(f"color: {tokens.WARN};")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_current_analysis)
        self._controls_row.addWidget(self._cancel_btn)
        self._controls_row.addStretch()

        # --- Stale banner (result-level warning above the stack) ---
        self._stale_banner = make_warning_banner(
            "Scope or peak seeds changed since the last analysis — the results below "
            "are stale. Re-run the analysis."
        )
        self._stale_banner.setVisible(False)
        self._central_layout.addWidget(self._stale_banner)

        # --- The stack ---
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_welcome_page())
        self._stack.addWidget(self._build_running_page())
        self._stack.addWidget(self._build_result_page())
        return self._stack

    # ------------------------------------------------------------------
    # Page construction
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        intro = QLabel(
            "The fit wizard analyses this spectrum, fits a set of physics-motivated "
            "candidate models to it, and recommends one with a confidence grade. It "
            "usually takes about a minute. The result is a starting point — you can "
            "apply it to the fit panel or ignore it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        analyze_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setStyleSheet(build_primary_button_qss())
        self._analyze_btn.clicked.connect(self._start_analysis)
        analyze_row.addWidget(self._analyze_btn)
        analyze_row.addStretch()
        layout.addLayout(analyze_row)

        self._welcome_hint_label = QLabel("")
        self._welcome_hint_label.setWordWrap(True)
        self._welcome_hint_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        layout.addWidget(self._welcome_hint_label)

        # Collapsed optional-guidance section: scope selector + FFT/peaks seeding.
        self._guidance_section = CollapsibleSection("Guide the analysis (optional)", expanded=False)
        self._guidance_scope_slot = QWidget()
        QVBoxLayout(self._guidance_scope_slot).setContentsMargins(0, 0, 0, 0)
        self._guidance_fingerprint_slot = QWidget()
        QVBoxLayout(self._guidance_fingerprint_slot).setContentsMargins(0, 0, 0, 0)
        guidance_hint = QLabel(
            "Use these only if you know something the run metadata does not. Narrow "
            "which physics families are screened, or seed a peak the automatic search "
            "may miss."
        )
        guidance_hint.setWordWrap(True)
        guidance_hint.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._guidance_section.addWidget(guidance_hint)
        self._guidance_section.addWidget(self._guidance_scope_slot)
        self._guidance_section.addWidget(self._guidance_fingerprint_slot)
        layout.addWidget(self._guidance_section)
        layout.addStretch()
        return page

    def _build_running_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QLabel("Analysing this spectrum…")
        header.setStyleSheet("font-weight: 600;")
        layout.addWidget(header)
        self._running_trail = DecisionTrail()
        layout.addWidget(self._running_trail)
        layout.addStretch()
        return page

    def _build_result_page(self) -> QWidget:
        # The result page is a QScrollArea: an expanded trail step (scope
        # selector, FFT panel, compare table) can push content past the window
        # with no other way to reach it. The scroll area itself is the stacked
        # page; the inner content widget keeps the layout below unchanged.
        content = QWidget()
        layout = QVBoxLayout(content)

        # Answer card at the top.
        self._answer_card = WizardAnswerCard()
        self._answer_card.apply_requested.connect(self._on_card_apply_requested)
        self._answer_card.selection_changed.connect(self._on_card_selection_changed)
        layout.addWidget(self._answer_card)

        layout.addWidget(TrailSeparator())

        # Result-level actions.
        actions_row = QHBoxLayout()
        self._copy_log_btn = QPushButton("Copy analysis log")
        self._copy_log_btn.clicked.connect(self._copy_analysis_log)
        actions_row.addWidget(self._copy_log_btn)
        self._reanalyze_btn = QPushButton("Re-analyze")
        self._reanalyze_btn.clicked.connect(self._reanalyze)
        actions_row.addWidget(self._reanalyze_btn)
        actions_row.addStretch()
        actions_row.addWidget(QLabel("Ranking metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        actions_row.addWidget(self._metric_combo)
        layout.addLayout(actions_row)

        # The decision trail below the card.
        self._result_trail = DecisionTrail()
        layout.addWidget(self._result_trail, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; }"
            " QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.setWidget(content)
        self._result_scroll = scroll
        return scroll

    # ------------------------------------------------------------------
    # Deep panels (built once; re-parented between states)
    # ------------------------------------------------------------------

    def _build_scope_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel(
            "Choose which candidate families the wizard screens. Start from a preset "
            "(or Auto, inferred from run metadata) and include/exclude individual "
            "components as needed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._scope_selector = WizardScopeSelector()
        self._scope_selector.scope_changed.connect(self._on_scope_changed)
        self._scope_selector.validity_changed.connect(self._on_scope_validity_changed)
        layout.addWidget(self._scope_selector, 1)
        return panel

    def _build_fingerprint_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._fingerprint_banner = QLabel("")
        self._fingerprint_banner.setWordWrap(True)
        layout.addWidget(self._fingerprint_banner)

        grid = QGridLayout()
        self._fingerprint_plot_widget = self._build_matplotlib_widget()
        grid.addWidget(self._fingerprint_plot_widget, 0, 0)

        self._fingerprint_table = QTableWidget(0, 2)
        self._fingerprint_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self._fingerprint_table.horizontalHeader().setStretchLastSection(True)
        grid.addWidget(self._fingerprint_table, 0, 1)

        self._peaks_table = QTableWidget(0, 5)
        self._peaks_table.setHorizontalHeaderLabels(
            ["Freq (MHz)", "SNR", "Width (MHz)", "Pattern", "Source"]
        )
        self._peaks_table.horizontalHeader().setStretchLastSection(True)
        self._peaks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._peaks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._peaks_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._peaks_table.itemSelectionChanged.connect(self._on_peaks_selection_changed)
        grid.addWidget(self._peaks_table, 1, 0, 1, 2)
        layout.addLayout(grid)

        peak_controls = QHBoxLayout()
        peak_hint = QLabel(
            "Click on the FFT to add a peak seed; click an existing red marker to remove it."
        )
        peak_hint.setWordWrap(True)
        peak_controls.addWidget(peak_hint, 1)
        self._remove_peak_btn = QPushButton("Remove Selected Peak")
        self._remove_peak_btn.setEnabled(False)
        self._remove_peak_btn.clicked.connect(self._remove_selected_peak)
        peak_controls.addWidget(self._remove_peak_btn)
        layout.addLayout(peak_controls)

        canvas = getattr(self._fingerprint_plot_widget, "_canvas", None)
        if canvas is not None:
            canvas.mpl_connect("button_press_event", self._on_fft_press)
            canvas.mpl_connect("motion_notify_event", self._on_fft_motion)
            canvas.mpl_connect("button_release_event", self._on_fft_release)
        return panel

    def _build_compare_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        controls_row = QHBoxLayout()
        metric_info_btn = QPushButton("Metric Info")
        metric_info_btn.clicked.connect(self._show_metric_info)
        controls_row.addWidget(metric_info_btn)
        residual_info_btn = QPushButton("Residual Checks")
        residual_info_btn.clicked.connect(self._show_residual_info)
        controls_row.addWidget(residual_info_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self._compare_table = QTableWidget(0, 8)
        self._compare_table.setHorizontalHeaderLabels(
            ["Candidate", "Score", "AIC", "AICc", "BIC", "Gate", "χ²ᵣ", "Params"]
        )
        self._compare_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._compare_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._compare_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._compare_table.setSortingEnabled(True)
        self._compare_table.itemSelectionChanged.connect(self._on_compare_selection_changed)
        layout.addWidget(self._compare_table)

        self._compare_warning_text = QTextEdit()
        self._compare_warning_text.setReadOnly(True)
        self._compare_warning_text.setMinimumHeight(90)
        layout.addWidget(self._compare_warning_text)
        return panel

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _reparent_into(self, slot: QWidget, panel: QWidget) -> None:
        """Move ``panel`` into ``slot``'s layout if it is not already there."""
        layout = slot.layout()
        if layout is None:
            layout = QVBoxLayout(slot)
            layout.setContentsMargins(0, 0, 0, 0)
        if panel.parent() is not slot:
            layout.addWidget(panel)
        panel.setVisible(True)

    def _show_welcome(self) -> None:
        """Enter the Welcome state: guidance panels housed in the expander."""
        self._reparent_into(self._guidance_scope_slot, self._scope_panel)
        self._reparent_into(self._guidance_fingerprint_slot, self._fingerprint_panel)
        self._update_start_button()
        self._stack.setCurrentIndex(_PAGE_WELCOME)

    def _show_running(self) -> None:
        """Enter the Running state: stream trail placeholders."""
        self._running_trail.stream_placeholders(_running_placeholder_steps())
        self._running_trail.set_status("Reading run conditions…")
        self._stack.setCurrentIndex(_PAGE_RUNNING)

    def _show_result(self) -> None:
        """Enter the Result state: card + trail with re-parented deep panels."""
        self._populate_result_state()
        self._stack.setCurrentIndex(_PAGE_RESULT)

    # ------------------------------------------------------------------
    # External surface (unchanged contract)
    # ------------------------------------------------------------------

    def set_analysis_context(
        self,
        dataset: MuonDataset,
        current_model: CompositeModel | None = None,
    ) -> None:
        """Prepare the wizard for a new dataset/model context (→ Welcome)."""
        self._dataset = dataset
        self._current_model = current_model
        self._cached_log_text = ""
        self._cached_signature = None
        self._analysis_request_id += 1
        self._user_peaks = []
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._heading_label.setText("Fit Wizard")
        self._status_label.setToolTip("")
        self.set_context_chips(self._context_chip_labels())
        self._recommendation = None
        self._selected_key = None
        # Install the scope resolver and reset the selector to Auto (signal-silent).
        self._scope_selector.set_resolver(self._resolve_scope)
        self._scope_selector.set_scope(None)
        self._scope_selector.refresh_from_context()
        # Render the time/FFT plot and the (user-only) peaks table straight away
        # so peak seeds can be added before the first analysis run.
        self._fingerprint_banner.setText("")
        self._populate_fingerprint_plot()
        self._populate_peaks_table()
        self._show_welcome()
        if self._analysis_in_progress:
            self._welcome_hint_label.setText(
                "Context updated while a previous analysis is still finishing. That "
                "result will be ignored; click Analyze once the wizard is ready."
            )
            self._status_label.setText(
                "Context updated while a previous analysis is still finishing."
            )
            return
        self._welcome_hint_label.setText(
            "Click Analyze to fingerprint this spectrum without blocking the main window."
        )
        self._status_label.setText(
            "Ready to fingerprint this spectrum. Click Analyze to run the wizard "
            "without blocking the main window."
        )
        self._set_busy(False)

    def _start_analysis(self) -> None:
        if self._dataset is None:
            self._status_label.setText("No dataset is available for the fit wizard.")
            return
        if self._analysis_in_progress:
            return
        if not self._scope_selector.is_valid():
            self._status_label.setText(
                "Select at least one candidate family in the guidance section to enable analysis."
            )
            return
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._status_label.setText(
            "Running fit wizard analysis in the background. You can keep using the main "
            "window while recommendations are prepared."
        )
        self._show_running()
        # The base bumps the request id, caches the signature, sets busy, calls
        # _reset_result_state(), then runs _create_worker_task() off-thread.
        self._run_analysis()

    def _create_worker_task(self, request_id: int):
        # Snapshot the widget-derived inputs NOW (submit time, GUI thread): the
        # scope payload → WizardScope and the user peaks → the core's plain float
        # list. The returned closure runs on the worker thread and must touch no
        # widgets, so it captures only these plain values plus worker.is_cancelled.
        dataset = self._dataset
        current_model = self._current_model
        scope = WizardScope.from_payload(self._scope_selector.current_scope())
        user_frequencies_mhz = [float(peak["freq_mhz"]) for peak in self._user_peaks] or None

        def task(worker):
            # The core's progress_callback takes a single message string; the
            # worker's progress signal is (current, total, message). Bridge them
            # so stage messages reach the GUI-thread _on_progress via the queued
            # relay (touches only worker.progress.emit — no widgets on-thread).
            return build_fit_wizard_recommendation(
                dataset,
                current_model=current_model,
                metric=SelectionMetric.AICC,
                scope=scope,
                user_frequencies_mhz=user_frequencies_mhz,
                progress_callback=lambda message: worker.progress.emit(0, 0, message),
                cancel_callback=worker.is_cancelled,
            )

        return task

    def _cancel_exceptions(self) -> tuple[type[BaseException], ...]:
        return (FitCancelledError,)

    def _reset_result_state(self) -> None:
        self._recommendation = None
        self._selected_key = None

    def _on_analysis_failed(self, message: str) -> None:
        # Keep the "Fit wizard analysis failed:" prefix (GlobalFitWizardWindow
        # keeps it too — the two wizards must match) and return to Welcome so the
        # metric combo cannot resurrect a stale success.
        self._recommendation = None
        # First line only in the header status — a multi-line exception message
        # would balloon the header band; the full text goes in the tooltip.
        failure_text = str(message).strip() or "unknown error"
        self._status_label.setText(f"Fit wizard analysis failed: {failure_text.splitlines()[0]}")
        self._status_label.setToolTip(failure_text)
        self._welcome_hint_label.setText(
            "The analysis failed. Adjust the guidance if needed and click Analyze again."
        )
        self._show_welcome()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        text = (message or "").strip()
        lowered = text.lower()
        matched = next(
            (key for prefix, key in _PROGRESS_STEP_PREFIXES if lowered.startswith(prefix)),
            None,
        )
        if matched is not None:
            self._running_trail.activate_step(matched)
        if text:
            self._running_trail.set_status(text)

    def _update_action_enablement(self, busy: bool) -> None:
        self._progress_label.setText("Analysis in progress..." if busy else "")
        self._cancel_btn.setVisible(busy)
        self._update_start_button()
        if self._recommendation is not None:
            self._metric_combo.setEnabled(not busy)
        else:
            self._metric_combo.setEnabled(False)

    def _update_start_button(self) -> None:
        """Refresh the Analyze/Re-run button state (both the welcome + base refs)."""
        busy = self._analysis_in_progress
        enabled = self._dataset is not None and not busy and self._scope_selector.is_valid()
        label = "Re-run Analysis" if self._analysis_stale and not busy else "Analyze"
        for button in (self._refresh_btn, getattr(self, "_analyze_btn", None)):
            if button is not None:
                button.setEnabled(enabled)
                button.setText(label)

    def _on_scope_changed(self, _scope: object) -> None:
        self._mark_analysis_stale("Scope changed")

    def _on_scope_validity_changed(self, is_valid: bool) -> None:
        if not is_valid and not self._analysis_in_progress:
            self._status_label.setText(
                "Select at least one candidate family in the guidance section to enable analysis."
            )
        self._update_start_button()

    def _mark_analysis_stale(self, reason: str) -> None:
        """Flag the displayed results as stale after a scope or peak-seed edit.

        An in-flight analysis is orphaned by bumping the base's request id (its
        terminal signal is discarded on arrival), never cancelled cooperatively.
        """
        if self._analysis_in_progress:
            self._analysis_request_id += 1
            self._set_busy(False)
            self._status_label.setText(
                f"{reason} while analysis was running; that result will be discarded. "
                "Re-run the analysis."
            )
            self._show_welcome()
        if self._recommendation is not None:
            self._analysis_stale = True
            self._stale_banner.setVisible(True)
        self._update_start_button()

    def _populate_results(self, result: object) -> None:
        recommendation = result
        self._recommendation = recommendation
        self._selected_key = recommendation.recommended_key
        if self._selected_key is None and recommendation.assessments:
            self._selected_key = recommendation.assessments[0].template.key
        self._status_label.setText(recommendation.summary)
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self.analysis_cached.emit(
            recommendation,
            self.current_log_text(),
            copy.deepcopy(self._cached_signature) if self._cached_signature is not None else None,
        )
        # The base cleared busy before calling us while _recommendation was None;
        # re-assert enablement now it is set.
        self._update_action_enablement(False)
        self._show_result()

    def set_cached_recommendation(
        self,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
    ) -> None:
        """Populate the window from an already-computed recommendation (→ Result)."""
        self._recommendation = recommendation
        self._cached_signature = copy.deepcopy(signature) if isinstance(signature, dict) else None
        self._selected_key = recommendation.recommended_key
        if self._selected_key is None and recommendation.assessments:
            self._selected_key = recommendation.assessments[0].template.key
        self._cached_log_text = str(log_text or "")
        # Restore scope + peak seeds from the signature. Legacy signatures without
        # these keys restore as Auto / no peaks. Cached state is never stale.
        signature_dict = signature if isinstance(signature, dict) else {}
        cached_scope = signature_dict.get("scope")
        self._scope_selector.set_scope(cached_scope if isinstance(cached_scope, dict) else None)
        cached_peaks = signature_dict.get("user_peaks")
        self._user_peaks = (
            [dict(peak) for peak in cached_peaks] if isinstance(cached_peaks, list) else []
        )
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._status_label.setText(recommendation.summary)
        self._set_busy(False)
        self._show_result()

    def _analysis_signature(self) -> dict[str, object]:
        return {
            "run_number": (
                int(self._dataset.run_number)
                if self._dataset is not None
                and getattr(self._dataset, "run_number", None) is not None
                else None
            ),
            "model": self._current_model.to_dict() if self._current_model is not None else None,
            "scope": self._scope_selector.current_scope(),
            "user_peaks": [dict(peak) for peak in self._user_peaks],
        }

    def current_recommendation(self) -> FitWizardRecommendation | None:
        return self._recommendation

    # ------------------------------------------------------------------
    # Welcome helpers
    # ------------------------------------------------------------------

    def _context_chip_labels(self) -> list[str]:
        """Header-band chip labels — run / field / temperature / sample, empty parts omitted."""
        if self._dataset is None:
            return []
        parts: list[str] = [f"Run {self._dataset.run_label}"]
        field = self._dataset.field
        if field is not None:
            parts.append(f"{field:g} G")
        temperature = self._dataset.temperature
        if temperature is not None:
            parts.append(f"{temperature:g} K")
        sample = self._dataset.metadata.get("sample")
        if sample:
            parts.append(str(sample))
        return parts

    def _reanalyze(self) -> None:
        """Return to the Welcome state so the user can steer, then Analyze again."""
        self._show_welcome()
        self._welcome_hint_label.setText(
            "Adjust the guidance if you like, then click Analyze to re-run."
        )

    def _copy_analysis_log(self) -> None:
        text = self._cached_log_text
        if not text and self._recommendation is not None:
            text = render_log_text(self._recommendation)
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.statusBar().showMessage("Analysis log copied to clipboard.")

    # ------------------------------------------------------------------
    # Scope resolver (unchanged)
    # ------------------------------------------------------------------

    def _resolve_scope(self, preset_id: str, overrides: dict) -> dict:
        """Adapt the core scope resolver to the WizardScopeSelector dict contract."""
        if self._dataset is None:
            return {
                "effective_preset": preset_id,
                "note": "Load a dataset first",
                "families": [],
                "estimate": [0, 0],
            }
        scope = WizardScope.from_payload(
            {
                "version": 1,
                "preset": preset_id,
                "include": overrides.get("include", []),
                "exclude": overrides.get("exclude", []),
            }
        )
        resolution = resolve_scope_for_dataset(self._dataset, scope)
        included = resolution.included_set
        reasons = {exc.name: exc.reason for exc in resolution.excluded_components}

        families: list[dict] = []
        by_category: dict[str, dict] = {}
        for name, definition in COMPONENTS.items():
            if definition.domain != "time":
                continue
            category = definition.category
            family = by_category.get(category)
            if family is None:
                family = {"key": category, "title": category, "components": []}
                by_category[category] = family
                families.append(family)
            family["components"].append(
                {
                    "name": name,
                    "included": name in included,
                    "reason": reasons.get(name, ""),
                    "cost": definition.cost.value,
                }
            )

        return {
            "effective_preset": resolution.effective_preset.value,
            "note": resolution.inference_note,
            "families": families,
            "estimate": list(estimate_screening_cost(resolution)),
        }

    # ------------------------------------------------------------------
    # Result-state population
    # ------------------------------------------------------------------

    def _populate_result_state(self) -> None:
        if self._recommendation is None or self._dataset is None:
            return
        # Answer card.
        self._answer_card.set_plot_data(
            np.asarray(self._dataset.time, dtype=float),
            np.asarray(self._dataset.asymmetry, dtype=float),
            np.asarray(self._dataset.error, dtype=float),
        )
        self._answer_card.set_recommendation(self._recommendation)
        self._answer_card.set_selected_key(self._selected_key)

        # Deep-panel content.
        self._fingerprint_banner.setText(self._fingerprint_banner_text())
        self._populate_fingerprint_table()
        self._populate_fingerprint_plot()
        self._populate_peaks_table()
        self._populate_compare_table()
        self._sync_selected_assessment()

        # Rebuild the trail from the recommendation (single source of truth), then
        # inject the re-parented deep panels for the steps that have one.
        trail = build_wizard_trail(self._recommendation)
        self._result_trail.set_steps(trail)
        self._reparent_into_trail_slot("conditions", self._scope_panel)
        self._reparent_into_trail_slot("spectrum", self._fingerprint_panel)
        self._reparent_into_trail_slot("candidates", self._compare_panel)

    def _reparent_into_trail_slot(self, key: str, panel: QWidget) -> None:
        """Move a deep panel into the result trail's step ``key`` expansion."""
        panel.setVisible(True)
        self._result_trail.set_step_detail_widget(key, panel)

    def _fingerprint_banner_text(self) -> str:
        """Assemble the fingerprint interpretation banner (deep-panel detail)."""
        if self._recommendation is None:
            return ""
        fingerprint = self._recommendation.fingerprint
        notes: list[str] = []
        if fingerprint.oscillatory_hint:
            notes.append(
                "Resolved structure supports an oscillatory interpretation: "
                f"FFT peak at {fingerprint.dominant_fft_frequency_mhz:.3f} MHz, "
                f"{fingerprint.dominant_fft_cycles_in_window:.2f} cycles across the window, "
                f"and {fingerprint.smoothed_turning_points} turning points in the smoothed trace."
            )
        elif fingerprint.dominant_fft_snr >= 3.0:
            notes.append(
                "A low-frequency FFT peak was found, but it is not being treated as a "
                f"strong oscillatory hint: {fingerprint.dominant_fft_frequency_mhz:.3f} MHz "
                f"spans only {fingerprint.dominant_fft_cycles_in_window:.2f} cycles and the "
                f"smoothed trace shows {fingerprint.smoothed_turning_points} turning points."
            )
        else:
            notes.append("No strong FFT peak was found in the default windowed transform.")
        if fingerprint.multi_rate_hint:
            notes.append(
                f"Semilog slope ratio {fingerprint.semilog_slope_ratio:.2f} suggests multiple "
                "relaxation rates or a distributed-rate envelope."
            )
        else:
            notes.append(
                f"Semilog slope ratio {fingerprint.semilog_slope_ratio:.2f} does not strongly "
                "demand a multi-rate model."
            )
        if fingerprint.kt_like_hint:
            notes.append(
                f"Late-time dip/recovery score {fingerprint.late_time_dip_recovery_score:.3f} "
                "suggests KT-like behaviour."
            )
        else:
            notes.append("Late-time recovery does not strongly favour a KT-like tail.")
        if self._recommendation.multiplet_matches:
            best_match = max(self._recommendation.multiplet_matches, key=lambda m: m.quality)
            notes.append(f"Pattern match: {best_match.note}")
        return " ".join(notes)

    def _build_matplotlib_widget(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            figure = Figure(tight_layout=True)
            canvas = FigureCanvasQTAgg(figure)
            container._figure = figure  # type: ignore[attr-defined]
            container._canvas = canvas  # type: ignore[attr-defined]
            layout.addWidget(canvas)
        except ImportError:
            fallback = QLabel("matplotlib not available — plot preview disabled")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            layout.addWidget(fallback)
        return container

    def _populate_fingerprint_table(self) -> None:
        if self._recommendation is None:
            return
        fingerprint = self._recommendation.fingerprint
        rows = [
            ("Tail estimate", f"{fingerprint.tail_estimate:.6f}"),
            ("Initial amplitude estimate", f"{fingerprint.initial_amplitude_estimate:.6f}"),
            ("Raw zero crossings", str(fingerprint.zero_crossings)),
            ("Smoothed zero crossings", str(fingerprint.smoothed_zero_crossings)),
            ("Smoothed turning points", str(fingerprint.smoothed_turning_points)),
            ("Dominant FFT frequency", f"{fingerprint.dominant_fft_frequency_mhz:.6f} MHz"),
            ("FFT peak SNR", f"{fingerprint.dominant_fft_snr:.3f}"),
            ("FFT cycles in window", f"{fingerprint.dominant_fft_cycles_in_window:.3f}"),
            ("Monotonic decay fraction", f"{fingerprint.monotonic_decay_fraction:.3f}"),
            ("Early-time curvature", f"{fingerprint.early_time_curvature:.6f}"),
            ("Semilog slope ratio", f"{fingerprint.semilog_slope_ratio:.6f}"),
            ("Late-time dip/recovery", f"{fingerprint.late_time_dip_recovery_score:.6f}"),
        ]
        self._fingerprint_table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            self._fingerprint_table.setItem(row, 0, QTableWidgetItem(label))
            self._fingerprint_table.setItem(row, 1, QTableWidgetItem(value))

    def _populate_fingerprint_plot(self) -> None:
        if self._dataset is None:
            self._fft_ax = None
            self._user_peak_artists = []
            self._draw_figure(self._fingerprint_plot_widget, None)
            return
        widget = self._fingerprint_plot_widget
        figure = getattr(widget, "_figure", None)
        canvas = getattr(widget, "_canvas", None)
        if figure is None or canvas is None:
            return

        figure.clear()
        ax_time = figure.add_subplot(2, 1, 1)
        ax_fft = figure.add_subplot(2, 1, 2)
        ax_time.errorbar(
            self._dataset.time,
            self._dataset.asymmetry,
            yerr=self._dataset.error,
            fmt=".",
            markersize=3,
            color=tokens.PLOT_DATA,
        )
        ax_time.set_xlabel("Time (µs)")
        ax_time.set_ylabel("Asymmetry")
        ax_time.set_title("Time Domain")

        centered_dataset = MuonDataset(
            time=np.asarray(self._dataset.time, dtype=float).copy(),
            asymmetry=np.asarray(self._dataset.asymmetry, dtype=float).copy()
            - (self._recommendation.fingerprint.tail_estimate if self._recommendation else 0.0),
            error=np.asarray(self._dataset.error, dtype=float).copy(),
            metadata=dict(self._dataset.metadata),
            run=self._dataset.run,
        )
        freq, _real, mag = fft_asymmetry(centered_dataset, window="hann", padding_factor=4)
        ax_fft.plot(freq, mag, color=tokens.ACCENT)
        ax_fft.set_xlabel("Frequency (MHz)")
        ax_fft.set_ylabel("|FFT|")
        ax_fft.set_title("Windowed FFT")

        self._fft_ax = ax_fft
        self._user_peak_artists = []
        analysis = self._recommendation.peak_analysis if self._recommendation else None
        if analysis is not None:
            for peak in analysis.peaks:
                if peak.source in ("fft", "residual_fft"):
                    ax_fft.axvline(
                        peak.frequency_mhz,
                        color=tokens.ACCENT,
                        alpha=0.35,
                        linewidth=1.2,
                    )
        self._refresh_peak_overlays()
        canvas.draw_idle()

    def _refresh_peak_overlays(self) -> None:
        """Redraw just the dashed user-peak markers on the FFT subplot."""
        if self._fft_ax is None:
            return
        for artist in self._user_peak_artists:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._user_peak_artists = []
        for peak in self._user_peaks:
            line = self._fft_ax.axvline(
                float(peak["freq_mhz"]),
                color=tokens.ACCENT_RED,
                alpha=0.9,
                linewidth=1.2,
                linestyle="--",
            )
            self._user_peak_artists.append(line)
        canvas = getattr(self._fingerprint_plot_widget, "_canvas", None)
        if canvas is not None:
            canvas.draw_idle()

    def _peaks_table_rows(self) -> list[dict]:
        """Assemble the merged detected + user peak rows for the peaks table."""
        analysis = self._recommendation.peak_analysis if self._recommendation else None
        resolution = float(analysis.resolution_mhz) if analysis is not None else 0.0
        matches = self._recommendation.multiplet_matches if self._recommendation else ()

        rows: list[dict] = []
        represented = [False] * len(self._user_peaks)
        if analysis is not None:
            for index, peak in enumerate(analysis.peaks):
                is_user = peak.source == "user"
                pattern = ""
                for match in matches:  # quality-descending
                    if index in match.peak_indices:
                        label = _MULTIPLET_KIND_LABELS.get(match.kind, match.kind)
                        pattern = f"{label} ({match.kind})"
                        break
                for u_idx, user_peak in enumerate(self._user_peaks):
                    if resolution > 0.0 and not represented[u_idx]:
                        if abs(float(user_peak["freq_mhz"]) - peak.frequency_mhz) <= resolution:
                            represented[u_idx] = True
                            is_user = True
                rows.append(
                    {
                        "freq_mhz": peak.frequency_mhz,
                        "snr": None if is_user else peak.snr,
                        "width_mhz": peak.width_mhz,
                        "pattern": pattern,
                        "source": "user" if is_user else peak.source,
                    }
                )
        for u_idx, user_peak in enumerate(self._user_peaks):
            if represented[u_idx]:
                continue
            rows.append(
                {
                    "freq_mhz": float(user_peak["freq_mhz"]),
                    "snr": None,
                    "width_mhz": None,
                    "pattern": "",
                    "source": "user",
                }
            )
        return rows

    def _populate_peaks_table(self) -> None:
        rows = self._peaks_table_rows()
        self._peaks_table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            freq_item = QTableWidgetItem()
            freq_item.setData(Qt.ItemDataRole.DisplayRole, float(entry["freq_mhz"]))
            freq_item.setData(Qt.ItemDataRole.UserRole, float(entry["freq_mhz"]))
            self._peaks_table.setItem(row, 0, freq_item)
            self._peaks_table.setItem(
                row,
                1,
                QTableWidgetItem("—") if entry["snr"] is None else _numeric_item(entry["snr"]),
            )
            self._peaks_table.setItem(
                row,
                2,
                QTableWidgetItem("—")
                if entry["width_mhz"] is None
                else _numeric_item(entry["width_mhz"]),
            )
            self._peaks_table.setItem(row, 3, QTableWidgetItem(entry["pattern"]))
            self._peaks_table.setItem(row, 4, QTableWidgetItem(entry["source"]))
        self._on_peaks_selection_changed()

    def _on_peaks_selection_changed(self) -> None:
        """Enable the remove button only when the selected row is a user peak."""
        row = self._peaks_table.currentRow()
        source_item = self._peaks_table.item(row, 4) if row >= 0 else None
        is_user = source_item is not None and source_item.text() == "user"
        self._remove_peak_btn.setEnabled(bool(is_user))

    def _remove_selected_peak(self) -> None:
        """Remove the user peak backing the selected row, then refresh state."""
        row = self._peaks_table.currentRow()
        if row < 0:
            return
        source_item = self._peaks_table.item(row, 4)
        if source_item is None or source_item.text() != "user":
            return
        freq_item = self._peaks_table.item(row, 0)
        if freq_item is None:
            return
        target = float(freq_item.data(Qt.ItemDataRole.UserRole))
        if not self._remove_user_peak_nearest(target):
            return
        self._refresh_peak_overlays()
        self._populate_peaks_table()
        self._mark_analysis_stale("Peak seeds changed")

    def _remove_user_peak_nearest(self, frequency_mhz: float) -> bool:
        """Drop the ``self._user_peaks`` entry nearest ``frequency_mhz``."""
        if not self._user_peaks:
            return False
        best_idx = min(
            range(len(self._user_peaks)),
            key=lambda i: abs(float(self._user_peaks[i]["freq_mhz"]) - frequency_mhz),
        )
        del self._user_peaks[best_idx]
        return True

    # --- interactive FFT peak editing (ALC press/motion/release convention) ---

    def _on_fft_press(self, event: object) -> None:
        if self._dataset is None or self._fft_ax is None:
            return
        if getattr(event, "inaxes", None) is not self._fft_ax:
            return
        if getattr(event, "button", None) != 1:
            return
        xdata = getattr(event, "xdata", None)
        if xdata is None:
            return
        self._peak_click_candidate = (
            float(xdata),
            float(getattr(event, "x", 0.0)),
            float(getattr(event, "y", 0.0)),
        )

    def _on_fft_motion(self, event: object) -> None:
        if self._peak_click_candidate is None:
            return
        _xdata, x0, y0 = self._peak_click_candidate
        if (
            abs(float(getattr(event, "x", x0)) - x0) > 3.0
            or abs(float(getattr(event, "y", y0)) - y0) > 3.0
        ):
            self._peak_click_candidate = None

    def _on_fft_release(self, event: object) -> None:
        candidate = self._peak_click_candidate
        self._peak_click_candidate = None
        if candidate is None or getattr(event, "button", None) != 1:
            return
        if self._dataset is None or self._fft_ax is None:
            return
        freq, x_press, _y_press = candidate
        removed = self._remove_user_peak_at_pixel(x_press)
        if not removed:
            self._user_peaks.append({"freq_mhz": float(freq), "source": "user"})
        self._refresh_peak_overlays()
        self._populate_peaks_table()
        self._mark_analysis_stale("Peak seeds changed")

    def _remove_user_peak_at_pixel(self, x_pixel: float) -> bool:
        """Remove the user peak whose marker is within ~12 device px of ``x_pixel``."""
        if self._fft_ax is None or not self._user_peaks:
            return False
        best_idx: int | None = None
        best_dist = 12.0
        for idx, peak in enumerate(self._user_peaks):
            px = float(self._fft_ax.transData.transform((float(peak["freq_mhz"]), 0.0))[0])
            dist = abs(px - x_pixel)
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            return False
        del self._user_peaks[best_idx]
        return True

    # ------------------------------------------------------------------
    # Compare table (deep panel)
    # ------------------------------------------------------------------

    def _populate_compare_table(self) -> None:
        if self._recommendation is None:
            return
        assessments = self._recommendation.sorted_assessments()
        self._compare_table.setSortingEnabled(False)
        self._compare_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            title_text = assessment.template.title
            if assessment.is_null_baseline:
                title_text = f"{title_text} (baseline)"
            elif assessment.is_disqualified:
                title_text = f"{title_text} (disqualified)"
            title_item = QTableWidgetItem(title_text)
            title_item.setData(Qt.ItemDataRole.UserRole, assessment.template.key)
            if assessment.template.key == self._recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            if not assessment.residual_gate_passed:
                title_item.setForeground(QBrush(QColor(tokens.ACCENT_RED)))
            if assessment.disqualification_reasons:
                title_item.setToolTip(
                    "Disqualified: " + "; ".join(assessment.disqualification_reasons)
                )
            self._compare_table.setItem(row, 0, title_item)
            self._compare_table.setItem(
                row, 1, _numeric_item(assessment.metric_value(self._recommendation.metric))
            )
            self._compare_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._compare_table.setItem(
                row,
                3,
                _numeric_item(assessment.aicc)
                if assessment.aicc is not None
                else QTableWidgetItem("AIC"),
            )
            self._compare_table.setItem(row, 4, _numeric_item(assessment.bic))
            gate_text = "Pass" if assessment.residual_gate_passed else "Warn"
            self._compare_table.setItem(row, 5, QTableWidgetItem(gate_text))
            self._compare_table.setItem(
                row, 6, _numeric_item(assessment.fit_result.reduced_chi_squared)
            )
            self._compare_table.setItem(row, 7, QTableWidgetItem(str(assessment.parameter_count)))
        self._compare_table.setSortingEnabled(True)
        self._compare_table.sortItems(1, Qt.SortOrder.AscendingOrder)

    def _sync_selected_assessment(self) -> None:
        if self._recommendation is None:
            return
        target_key = self._selected_key or self._recommendation.recommended_key
        if target_key is None and self._recommendation.assessments:
            target_key = self._recommendation.assessments[0].template.key
        self._selected_key = target_key
        for row in range(self._compare_table.rowCount()):
            item = self._compare_table.item(row, 0)
            if item is None:
                continue
            if item.data(Qt.ItemDataRole.UserRole) == target_key:
                self._compare_table.selectRow(row)
                break
        self._update_compare_warnings()

    def _on_metric_changed(self, text: str) -> None:
        if self._recommendation is None:
            return
        selected_key = self._selected_key
        self._recommendation = rerank_fit_wizard_recommendation(
            self._recommendation,
            SelectionMetric.from_value(text),
        )
        self._selected_key = selected_key or self._recommendation.recommended_key
        self._status_label.setText(self._recommendation.summary)
        self._answer_card.set_recommendation(self._recommendation)
        self._answer_card.set_selected_key(self._selected_key)
        self._fingerprint_banner.setText(self._fingerprint_banner_text())
        self._populate_compare_table()
        self._sync_selected_assessment()
        # The trail derives from the recommendation, so re-derive it after a re-rank.
        self._result_trail.set_steps(build_wizard_trail(self._recommendation))
        self._reparent_into_trail_slot("conditions", self._scope_panel)
        self._reparent_into_trail_slot("spectrum", self._fingerprint_panel)
        self._reparent_into_trail_slot("candidates", self._compare_panel)
        if isinstance(self._cached_signature, dict):
            self.analysis_cached.emit(
                self._recommendation,
                self.current_log_text(),
                copy.deepcopy(self._cached_signature),
            )

    def _on_compare_selection_changed(self) -> None:
        selected_items = self._compare_table.selectedItems()
        if not selected_items:
            return
        key = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(key, str):
            self._selected_key = key
            self._answer_card.set_selected_key(key)
        self._update_compare_warnings()

    def _on_card_selection_changed(self, key: str) -> None:
        """Keep the compare table + selected key in step with the card's choice."""
        self._selected_key = key
        for row in range(self._compare_table.rowCount()):
            item = self._compare_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                self._compare_table.blockSignals(True)
                self._compare_table.selectRow(row)
                self._compare_table.blockSignals(False)
                break
        self._update_compare_warnings()

    def _update_compare_warnings(self) -> None:
        assessment = self._selected_assessment()
        if assessment is None:
            self._compare_warning_text.setPlainText("")
            return
        messages: list[str] = []
        if assessment.residual_gate_passed:
            messages.append("Residual gate passed.")
        else:
            messages.append("Residual gate warning(s):")
            messages.extend(f"• {reason}" for reason in assessment.residual_gate_reasons)
        messages.append(f"Residual RMS: {assessment.residual_rms:.3f}")
        messages.append(f"Runs z score: {assessment.runs_z_score:.3f}")
        messages.append(f"Max |autocorrelation|: {assessment.max_abs_autocorrelation:.3f}")
        messages.append(f"Residual FFT peak SNR: {assessment.residual_fft_peak_snr:.3f}")
        self._compare_warning_text.setPlainText("\n".join(messages))

    def _selected_assessment(self) -> CandidateAssessment | None:
        if self._recommendation is None:
            return None
        return (
            self._recommendation.assessment_for_key(self._selected_key)
            or self._recommendation.recommended_assessment
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _on_card_apply_requested(self, assessment: object) -> None:
        if self._recommendation is None or not isinstance(assessment, CandidateAssessment):
            return
        self.apply_assessment_requested.emit(assessment, self._recommendation)
        self.statusBar().showMessage(f"Applied fit: {assessment.template.title}")

    # ------------------------------------------------------------------
    # Info dialogs
    # ------------------------------------------------------------------

    def _show_metric_info(self) -> None:
        QMessageBox.information(
            self,
            "Fit Wizard Metrics",
            (
                "AIC rewards fit quality while penalising parameter count.\n\n"
                "AICc is the wizard default because it adds a small-sample correction "
                "when the number of fitted points is not large compared with the number of "
                "free parameters.\n\n"
                "BIC penalises complexity more strongly and tends to favour simpler models."
            ),
        )

    def _show_residual_info(self) -> None:
        QMessageBox.information(
            self,
            "Residual Checks",
            (
                "The residual gate combines four lightweight diagnostics:\n"
                "• standardized residual RMS\n"
                "• runs-test z score\n"
                "• low-lag autocorrelation magnitude\n"
                "• residual FFT peak SNR\n\n"
                "Candidates that fail these checks are still shown, but the wizard will not "
                "recommend them automatically."
            ),
        )

    def _draw_figure(self, widget: QWidget, _content: object) -> None:
        figure = getattr(widget, "_figure", None)
        canvas = getattr(widget, "_canvas", None)
        if figure is None or canvas is None:
            return
        figure.clear()
        canvas.draw_idle()


def _running_placeholder_steps() -> tuple:
    """Pending trail headlines shown at run start (before results are known).

    These are the six trail keys with placeholder headlines; ``build_wizard_trail``
    replaces them wholesale on completion.
    """
    from asymmetry.core.fitting.wizard_narrative import TrailStep

    return (
        TrailStep("conditions", "Reading run conditions…", "conditions", ()),
        TrailStep("families", "Choosing physics families to consider…", "families", ()),
        TrailStep("spectrum", "Searching the spectrum for lines and patterns…", "spectrum", ()),
        TrailStep("candidates", "Fitting candidate models…", "candidates", ()),
        TrailStep("verdict", "Weighing the winner against a null baseline…", "verdict", ()),
        TrailStep("confidence", "Grading confidence…", "confidence", ()),
    )


def _numeric_item(value: float | None) -> QTableWidgetItem:
    item = QTableWidgetItem()
    if value is None or not np.isfinite(float(value)):
        item.setText("—")
        return item
    item.setData(Qt.ItemDataRole.DisplayRole, float(value))
    return item


def _bold_font(font: QFont) -> QFont:
    out = QFont(font)
    out.setBold(True)
    return out
