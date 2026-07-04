"""Non-modal guided fit wizard for single time-domain asymmetry spectra."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
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
    ConfidenceTier,
    FitWizardRecommendation,
    RecommendationVerdict,
    SelectionMetric,
    build_fit_wizard_recommendation,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fitting.wizard_scope import (
    WizardScope,
    estimate_screening_cost,
    resolve_scope_for_dataset,
)
from asymmetry.core.fourier.fft import fft_asymmetry
from asymmetry.gui.styles import tokens
from asymmetry.gui.widgets.screen_sizing import resize_to_available
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


class FitWizardWindow(WizardWindowBase):
    """Present a guided workflow for model recommendation and comparison."""

    apply_assessment_requested = Signal(
        object, object
    )  # CandidateAssessment, FitWizardRecommendation
    analysis_cached = Signal(object, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        # WizardWindowBase.__init__ builds the shared frame (heading/status/
        # controls row/tabs, TaskRunner, progress UI) and calls _build_tabs()
        # before this body resumes.
        super().__init__(parent)
        self.setWindowTitle("Fit Wizard")
        # Cap the default to the available screen so the title bar never opens
        # clipped above the menu bar on a 13-inch laptop (~800 px high). The tab
        # bodies already scroll, so the spacious preferred size is used only when
        # the display can hold it (P1-5).
        resize_to_available(self, 1180, 740)

        heading_font = QFont(self._heading_label.font())
        heading_font.setPointSize(max(heading_font.pointSize() + 4, 14))
        heading_font.setBold(True)
        self._heading_label.setFont(heading_font)
        self._heading_label.setText("Fit Wizard")
        self._status_label.setText(
            "Open the fit wizard on a single spectrum to fingerprint the data and compare curated candidate models."
        )
        self._update_navigation_buttons()
        self._refresh_btn.setEnabled(False)

    def _build_tabs(self) -> None:
        # The base calls this during __init__, before the subclass __init__ body
        # resumes, so the result-state members are initialised here.
        self._dataset: MuonDataset | None = None
        self._current_model: CompositeModel | None = None
        self._recommendation: FitWizardRecommendation | None = None
        self._selected_key: str | None = None
        # Scope + peak-seed state and the FFT interaction plumbing. User peaks are
        # GUI-side dicts ([{"freq_mhz": float, "source": "user"}]); the worker task
        # converts them to the core's plain float list at submit time.
        self._analysis_stale = False
        self._user_peaks: list[dict] = []
        self._fft_ax = None
        self._user_peak_artists: list = []
        self._peak_click_candidate: tuple[float, float, float] | None = None

        # The Start-analysis button sits before the base-owned progress widgets;
        # a trailing Cancel button (base-driven cooperative cancel) and a stretch
        # keep the row aligned as before.
        self._refresh_btn = QPushButton("Start Analysis")
        self._refresh_btn.clicked.connect(self._start_analysis)
        self._controls_row.insertWidget(0, self._refresh_btn)
        self._progress_label.setStyleSheet(f"color: {tokens.WARN};")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_current_analysis)
        self._controls_row.addWidget(self._cancel_btn)
        self._controls_row.addStretch()

        # Stale banner sits between the controls row and the tabs: appended to the
        # base's central layout (which currently holds heading/status/controls/tabs),
        # then re-ordered above the tabs so it reads as a result-level warning.
        self._stale_banner = QLabel(
            "Scope or peak seeds changed since the last analysis — the results below "
            "are stale. Re-run the analysis."
        )
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setStyleSheet(f"color: {tokens.ERROR}; font-weight: 600;")
        self._stale_banner.setVisible(False)
        # Insert directly above the tabs widget in the base's central layout.
        tabs_index = self._central_layout.indexOf(self._tabs)
        self._central_layout.insertWidget(tabs_index, self._stale_banner)

        self._scope_tab = QWidget()
        self._fingerprint_tab = QWidget()
        self._portfolio_tab = QWidget()
        self._compare_tab = QWidget()
        self._apply_tab = QWidget()
        self._tabs.addTab(self._scope_tab, "1. Scope")
        self._tabs.addTab(self._fingerprint_tab, "2. Fingerprint")
        self._tabs.addTab(self._portfolio_tab, "3. Candidate Portfolio")
        self._tabs.addTab(self._compare_tab, "4. Compare Fits")
        self._tabs.addTab(self._apply_tab, "5. Apply")

        self._build_scope_tab()
        self._build_fingerprint_tab()
        self._build_portfolio_tab()
        self._build_compare_tab()
        self._build_apply_tab()

        nav_row = QHBoxLayout()
        self._previous_btn = QPushButton("Previous")
        self._previous_btn.clicked.connect(self._go_previous_tab)
        nav_row.addWidget(self._previous_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._go_next_tab)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch()
        self._central_layout.addLayout(nav_row)

        self._tabs.currentChanged.connect(self._update_navigation_buttons)

    def set_analysis_context(
        self,
        dataset: MuonDataset,
        current_model: CompositeModel | None = None,
    ) -> None:
        """Prepare the wizard for a new dataset/model context."""
        self._dataset = dataset
        self._current_model = current_model
        self._cached_log_text = ""
        self._cached_signature = None
        self._analysis_request_id += 1
        self._user_peaks = []
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._heading_label.setText(f"Fit Wizard — Run {dataset.run_label}")
        self._recommendation = None
        # Install the scope resolver and reset the selector to Auto (signal-silent).
        self._scope_selector.set_resolver(self._resolve_scope)
        self._scope_selector.set_scope(None)
        self._scope_selector.refresh_from_context()
        self._tabs.setCurrentIndex(0)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._set_empty_state()
        # Render the time/FFT plot and the (user-only) peaks table straight away
        # so peak seeds can be added before the first analysis run.
        self._populate_fingerprint_plot()
        self._populate_peaks_table()
        if self._analysis_in_progress:
            self._status_label.setText(
                "Context updated while a previous analysis is still finishing. That result will be ignored; start a new analysis once the wizard is ready."
            )
            return
        self._status_label.setText(
            "Ready to fingerprint this spectrum. Click Start Analysis to run the wizard without blocking the main window."
        )
        self._set_busy(False)

    def _start_analysis(self) -> None:
        if self._dataset is None:
            self._status_label.setText("No dataset is available for the fit wizard.")
            self._set_empty_state()
            return
        if self._analysis_in_progress:
            return
        if not self._scope_selector.is_valid():
            self._status_label.setText(
                "Select at least one candidate family on the Scope tab to enable analysis."
            )
            return
        self._analysis_stale = False
        self._stale_banner.setVisible(False)
        self._status_label.setText(
            "Running fit wizard analysis in the background. You can keep using the main window while recommendations are prepared."
        )
        # The base bumps the request id, caches the signature, sets busy, calls
        # _reset_result_state(), then runs _create_worker_task() off-thread.
        self._run_analysis()

    def _create_worker_task(self, request_id: int):
        # Snapshot the widget-derived inputs NOW (submit time, GUI thread): the
        # scope payload → WizardScope and the user peaks → the core's plain float
        # list. The returned closure runs on the worker thread and must touch no
        # widgets, so it captures only these plain values plus worker.is_cancelled
        # as the engine's cooperative cancel_callback.
        dataset = self._dataset
        current_model = self._current_model
        scope = WizardScope.from_payload(self._scope_selector.current_scope())
        user_frequencies_mhz = [float(peak["freq_mhz"]) for peak in self._user_peaks] or None

        def task(worker):
            return build_fit_wizard_recommendation(
                dataset,
                current_model=current_model,
                metric=SelectionMetric.AICC,
                scope=scope,
                user_frequencies_mhz=user_frequencies_mhz,
                cancel_callback=worker.is_cancelled,
            )

        return task

    def _cancel_exceptions(self) -> tuple[type[BaseException], ...]:
        # The engine raises FitCancelledError when cancel_callback trips; the base
        # routes it to the clean "Analysis cancelled." path (not the error path).
        return (FitCancelledError,)

    def _reset_result_state(self) -> None:
        self._set_empty_state()

    def _on_analysis_failed(self, message: str) -> None:
        # Reproduce the pre-unification failure handling (old _on_analysis_error):
        # clear the recommendation (so a stale success can't be resurrected via
        # the metric combo), keep the "Fit wizard analysis failed:" prefix (which
        # GlobalFitWizardWindow also keeps — the two wizards must match), and
        # empty the result tabs. The base has already cleared busy.
        self._recommendation = None
        self._status_label.setText(f"Fit wizard analysis failed: {message}")
        self._set_empty_state()

    def _update_action_enablement(self, busy: bool) -> None:
        self._progress_label.setText("Analysis in progress..." if busy else "")
        self._cancel_btn.setVisible(busy)
        self._update_start_button()
        self._metric_combo.setEnabled(not busy and self._recommendation is not None)
        self._previous_btn.setEnabled(not busy and self._tabs.currentIndex() > 0)
        self._next_btn.setEnabled(not busy and self._tabs.currentIndex() < self._tabs.count() - 1)

    def _update_start_button(self) -> None:
        """Refresh the Start/Refresh/Re-run button text and enabled state.

        Enabled iff a dataset is present, no analysis is in flight, and the scope
        selector reports at least one included component.
        """
        busy = self._analysis_in_progress
        self._refresh_btn.setEnabled(
            self._dataset is not None and not busy and self._scope_selector.is_valid()
        )
        if self._analysis_stale and not busy:
            self._refresh_btn.setText("Re-run Analysis")
        elif self._recommendation is not None and not busy:
            self._refresh_btn.setText("Refresh Analysis")
        else:
            self._refresh_btn.setText("Start Analysis")

    def _on_scope_changed(self, _scope: object) -> None:
        self._mark_analysis_stale("Scope changed")

    def _on_scope_validity_changed(self, is_valid: bool) -> None:
        if not is_valid and not self._analysis_in_progress:
            self._status_label.setText(
                "Select at least one candidate family on the Scope tab to enable analysis."
            )
        self._update_start_button()

    def _mark_analysis_stale(self, reason: str) -> None:
        """Flag the displayed results as stale after a scope or peak-seed edit.

        Follows the ignore-stale convention: an in-flight analysis is orphaned by
        bumping the base's request id (its terminal signal is discarded on arrival
        by the base staleness guard), never cancelled cooperatively — the cancelled
        path would clash with the stale-status text.
        """
        if self._analysis_in_progress:
            self._analysis_request_id += 1
            self._set_busy(False)
            self._status_label.setText(
                f"{reason} while analysis was running; that result will be discarded. "
                "Re-run the analysis."
            )
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
        # The base cleared busy before calling us, while _recommendation was
        # still None; re-assert enablement now it is set so the metric combo and
        # button label match the pre-refactor final state.
        self._update_action_enablement(False)
        self._populate_from_recommendation()

    def set_cached_recommendation(
        self,
        recommendation: FitWizardRecommendation,
        *,
        signature: dict[str, object] | None = None,
        log_text: str = "",
    ) -> None:
        """Populate the window from an already-computed recommendation."""
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
        self._populate_from_recommendation()

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

    def _build_scope_tab(self) -> None:
        layout = QVBoxLayout(self._scope_tab)
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

    def _resolve_scope(self, preset_id: str, overrides: dict) -> dict:
        """Adapt the core scope resolver to the WizardScopeSelector dict contract.

        Groups in-registry-order TIME-domain components by their display
        ``category``; frequency-domain components are skipped entirely.
        """
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

    def _build_fingerprint_tab(self) -> None:
        layout = QVBoxLayout(self._fingerprint_tab)
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

    def _build_portfolio_tab(self) -> None:
        layout = QVBoxLayout(self._portfolio_tab)
        self._portfolio_banner = QLabel("")
        self._portfolio_banner.setWordWrap(True)
        layout.addWidget(self._portfolio_banner)

        self._portfolio_table = QTableWidget(0, 4)
        self._portfolio_table.setHorizontalHeaderLabels(
            ["Candidate", "Category", "Parameters", "Rationale"]
        )
        self._portfolio_table.horizontalHeader().setStretchLastSection(True)
        self._portfolio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._portfolio_table)

    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self._compare_tab)
        self._compare_banner = QLabel("")
        self._compare_banner.setWordWrap(True)
        layout.addWidget(self._compare_banner)

        # Confidence-tier / verdict / caveat banner. Styled per-tier in
        # _update_confidence_banner: high = muted note, medium = amber caveat,
        # no-significant-structure = unmissable amber statement. Hidden until the
        # recommendation carries a non-default tier or a caveat.
        self._confidence_banner = QLabel("")
        self._confidence_banner.setWordWrap(True)
        self._confidence_banner.setVisible(False)
        layout.addWidget(self._confidence_banner)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Ranking Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems([metric.value for metric in SelectionMetric])
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        controls_row.addWidget(self._metric_combo)

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

        self._compare_plot_widget = self._build_matplotlib_widget()
        layout.addWidget(self._compare_plot_widget)

        self._compare_warning_text = QTextEdit()
        self._compare_warning_text.setReadOnly(True)
        self._compare_warning_text.setMinimumHeight(90)
        layout.addWidget(self._compare_warning_text)

    def _build_apply_tab(self) -> None:
        layout = QVBoxLayout(self._apply_tab)
        self._apply_banner = QLabel("")
        self._apply_banner.setWordWrap(True)
        layout.addWidget(self._apply_banner)

        self._apply_selection_label = QLabel("")
        self._apply_selection_label.setWordWrap(True)
        layout.addWidget(self._apply_selection_label)

        self._apply_parameters_table = QTableWidget(0, 3)
        self._apply_parameters_table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Uncertainty"]
        )
        self._apply_parameters_table.horizontalHeader().setStretchLastSection(True)
        self._apply_parameters_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._apply_parameters_table)

        self._apply_warning_text = QTextEdit()
        self._apply_warning_text.setReadOnly(True)
        self._apply_warning_text.setMinimumHeight(100)
        layout.addWidget(self._apply_warning_text)

        button_row = QHBoxLayout()
        self._apply_recommended_btn = QPushButton("Apply Recommended Fit")
        self._apply_recommended_btn.clicked.connect(self._apply_recommended_fit)
        button_row.addWidget(self._apply_recommended_btn)

        self._apply_selected_btn = QPushButton("Apply Selected Fit")
        self._apply_selected_btn.clicked.connect(self._apply_selected_fit)
        button_row.addWidget(self._apply_selected_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

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

    def _set_empty_state(self) -> None:
        for table in (
            self._fingerprint_table,
            self._peaks_table,
            self._portfolio_table,
            self._compare_table,
            self._apply_parameters_table,
        ):
            table.setRowCount(0)
        self._remove_peak_btn.setEnabled(False)
        self._fingerprint_banner.setText("")
        self._portfolio_banner.setText("")
        self._compare_banner.setText("")
        self._confidence_banner.setText("")
        self._confidence_banner.setVisible(False)
        self._apply_banner.setText("")
        self._compare_warning_text.setPlainText("")
        self._apply_warning_text.setPlainText("")
        self._apply_selection_label.setText("")
        self._selected_key = None
        self._draw_figure(self._fingerprint_plot_widget, None)
        self._draw_figure(self._compare_plot_widget, None)
        self._apply_recommended_btn.setEnabled(False)
        self._apply_selected_btn.setEnabled(False)
        self._metric_combo.setEnabled(False)

    def _populate_from_recommendation(self) -> None:
        if self._recommendation is None or self._dataset is None:
            self._set_empty_state()
            return

        self._update_banners()
        self._populate_fingerprint_table()
        self._populate_fingerprint_plot()
        self._populate_peaks_table()
        self._populate_portfolio_table()
        self._populate_compare_table()
        self._sync_selected_assessment()
        self._update_apply_page()

    def _update_banners(self) -> None:
        if self._recommendation is None:
            return
        fingerprint = self._recommendation.fingerprint

        fingerprint_notes: list[str] = []
        if fingerprint.oscillatory_hint:
            fingerprint_notes.append(
                "Resolved structure supports an oscillatory interpretation: "
                f"FFT peak at {fingerprint.dominant_fft_frequency_mhz:.3f} MHz, "
                f"{fingerprint.dominant_fft_cycles_in_window:.2f} cycles across the window, "
                f"and {fingerprint.smoothed_turning_points} turning points in the smoothed trace."
            )
        elif fingerprint.dominant_fft_snr >= 3.0:
            fingerprint_notes.append(
                "A low-frequency FFT peak was found, but it is not being treated as a strong oscillatory hint: "
                f"{fingerprint.dominant_fft_frequency_mhz:.3f} MHz spans only "
                f"{fingerprint.dominant_fft_cycles_in_window:.2f} cycles and the smoothed trace shows "
                f"{fingerprint.smoothed_turning_points} turning points."
            )
        else:
            fingerprint_notes.append(
                "No strong FFT peak was found in the default windowed transform."
            )
        if fingerprint.multi_rate_hint:
            fingerprint_notes.append(
                f"Semilog slope ratio {fingerprint.semilog_slope_ratio:.2f} suggests multiple relaxation rates or a distributed-rate envelope."
            )
        else:
            fingerprint_notes.append(
                f"Semilog slope ratio {fingerprint.semilog_slope_ratio:.2f} does not strongly demand a multi-rate model."
            )
        if fingerprint.kt_like_hint:
            fingerprint_notes.append(
                f"Late-time dip/recovery score {fingerprint.late_time_dip_recovery_score:.3f} suggests KT-like behaviour."
            )
        else:
            fingerprint_notes.append("Late-time recovery does not strongly favour a KT-like tail.")

        if self._recommendation.multiplet_matches:
            best_match = max(self._recommendation.multiplet_matches, key=lambda m: m.quality)
            fingerprint_notes.append(f"Pattern match: {best_match.note}")

        self._fingerprint_banner.setText(" ".join(fingerprint_notes))
        self._portfolio_banner.setText(self._recommendation.summary)
        self._compare_banner.setText(self._recommendation.summary)
        self._update_confidence_banner()
        if self._recommendation.recommended_assessment is None:
            self._apply_banner.setText(
                "No candidate passed the automatic residual gate. You can still inspect and apply a manually selected fit."
            )
        else:
            recommended = self._recommendation.recommended_assessment.template.title
            self._apply_banner.setText(
                f"The wizard recommends {recommended}. Apply it directly or choose an alternative from the comparison step."
            )

    def _update_confidence_banner(self) -> None:
        """Render the confidence tier / verdict / caveat below the ranking summary.

        Three visual registers, all from design tokens (no raw hex):

        * ``NO_SIGNIFICANT_STRUCTURE`` — an unmissable amber statement that the
          data show no significant structure, carrying any caveat text.
        * ``MEDIUM`` confidence — a usable-but-caveated amber note (never styled
          as an error), showing the residual-structure caveat.
        * ``HIGH`` confidence — a muted "high confidence" note.
        """
        recommendation = self._recommendation
        if recommendation is None:
            self._confidence_banner.setText("")
            self._confidence_banner.setVisible(False)
            return

        verdict = recommendation.verdict
        confidence = recommendation.confidence
        caveat = (recommendation.caveat or "").strip()

        if verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
            text = "No significant structure: the data are consistent with the null baseline."
            if caveat:
                text = f"{text} {caveat}"
            style = (
                f"background-color: {tokens.WARN_BANNER_BG};"
                f" color: {tokens.WARN_BANNER_TEXT}; font-weight: 600; padding: 4px;"
            )
        elif confidence is ConfidenceTier.MEDIUM:
            text = "Medium confidence — usable, but read the caveat."
            if caveat:
                text = f"{text} {caveat}"
            style = (
                f"background-color: {tokens.WARN_BANNER_BG};"
                f" color: {tokens.WARN_BANNER_TEXT}; padding: 4px;"
            )
        elif confidence is ConfidenceTier.HIGH:
            text = "High confidence: the recommended fit passes every residual check."
            style = f"color: {tokens.TEXT_MUTED};"
        elif recommendation.recommended_assessment is None:
            # Genuine no-recommendation (NONE tier, no structured/no-structure
            # verdict, nothing recommended) — surface the caveat or a neutral note.
            text = caveat or "No recommendation could be made — inspect the comparison table."
            style = f"color: {tokens.TEXT_MUTED};"
        else:
            # A recommendation exists but the tier is the default NONE (e.g. an
            # explicit-template path or a pre-confidence payload). Don't contradict
            # the summary with a "no recommendation" claim: show the caveat if any,
            # else hide the banner entirely.
            caveat_text = caveat
            if not caveat_text:
                self._confidence_banner.setText("")
                self._confidence_banner.setVisible(False)
                return
            text = caveat_text
            style = f"color: {tokens.TEXT_MUTED};"

        self._confidence_banner.setText(text)
        self._confidence_banner.setStyleSheet(style)
        self._confidence_banner.setVisible(True)

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
            color="#1d3557",
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
        ax_fft.plot(freq, mag, color="#457b9d")
        ax_fft.set_xlabel("Frequency (MHz)")
        ax_fft.set_ylabel("|FFT|")
        ax_fft.set_title("Windowed FFT")

        self._fft_ax = ax_fft
        self._user_peak_artists = []
        # Auto-detected peaks (solid accent markers). User peaks — including any
        # user peak merged into the analysis as source "user" — are drawn by
        # _refresh_peak_overlays from self._user_peaks, so exclude them here.
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
        """Redraw just the dashed user-peak markers on the FFT subplot.

        Removes the artists tracked in ``self._user_peak_artists`` and redraws
        one dashed danger-coloured ``axvline`` per entry in ``self._user_peaks``.
        Never clears the figure, so the FFT trace and auto markers survive.
        """
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
        """Assemble the merged detected + user peak rows for the peaks table.

        Detected peaks come from ``recommendation.peak_analysis`` (SNR-desc); each
        user peak within one ``resolution_mhz`` of a listed peak is folded onto
        that row rather than repeated. Remaining user peaks are appended.
        """
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
                # Fold a matching user peak onto this row so it is not repeated.
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
            # Store the exact frequency for nearest-match removal from the table.
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
        # >3 device px of travel means this is a pan/drag, not a click.
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
        # Within ~12 device px of an existing user marker → remove it; else add.
        removed = self._remove_user_peak_at_pixel(x_press)
        if not removed:
            self._user_peaks.append({"freq_mhz": float(freq), "source": "user"})
        self._refresh_peak_overlays()
        self._populate_peaks_table()
        self._mark_analysis_stale("Peak seeds changed")

    def _remove_user_peak_at_pixel(self, x_pixel: float) -> bool:
        """Remove the user peak whose marker is within ~12 device px of ``x_pixel``.

        Resolution-independent (works pre-analysis) — hit-tests the x device
        coordinate of each user frequency through ``self._fft_ax.transData``.
        """
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

    def _populate_portfolio_table(self) -> None:
        if self._recommendation is None:
            return
        templates = list(self._recommendation.templates)
        self._portfolio_table.setRowCount(len(templates))
        for row, template in enumerate(templates):
            title_item = QTableWidgetItem(template.title)
            if template.key == self._recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            self._portfolio_table.setItem(row, 0, title_item)
            self._portfolio_table.setItem(row, 1, QTableWidgetItem(template.category))
            self._portfolio_table.setItem(row, 2, QTableWidgetItem(str(template.parameter_count)))
            self._portfolio_table.setItem(row, 3, QTableWidgetItem(template.rationale))

    def _populate_compare_table(self) -> None:
        if self._recommendation is None:
            return
        assessments = self._recommendation.sorted_assessments()
        self._compare_table.setSortingEnabled(False)
        self._compare_table.setRowCount(len(assessments))
        for row, assessment in enumerate(assessments):
            # Null baselines read as baselines, not candidates; disqualified
            # candidates carry their reasons as a tooltip and a "disqualified" tag.
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
                title_item.setForeground(QBrush(QColor("#9b2226")))
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
        self._update_compare_visuals()

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
        self._update_banners()
        self._populate_compare_table()
        self._sync_selected_assessment()
        self._update_apply_page()
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
        self._update_compare_visuals()
        self._update_apply_page()

    def _update_compare_visuals(self) -> None:
        assessment = self._selected_assessment()
        if assessment is None:
            self._compare_warning_text.setPlainText("")
            self._draw_figure(self._compare_plot_widget, None)
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

        if self._dataset is None:
            self._draw_figure(self._compare_plot_widget, None)
            return

        widget = self._compare_plot_widget
        figure = getattr(widget, "_figure", None)
        canvas = getattr(widget, "_canvas", None)
        if figure is None or canvas is None:
            return

        figure.clear()
        ax_fit = figure.add_subplot(3, 1, 1)
        ax_res = figure.add_subplot(3, 1, 2)
        ax_fft = figure.add_subplot(3, 1, 3)

        ax_fit.errorbar(
            self._dataset.time,
            self._dataset.asymmetry,
            yerr=self._dataset.error,
            fmt=".",
            markersize=3,
            color="#1d3557",
            label="Data",
        )
        ax_fit.plot(assessment.fitted_time, assessment.fitted_curve, color="#e63946", label="Fit")
        ax_fit.set_xlabel("Time (µs)")
        ax_fit.set_ylabel("Asymmetry")
        ax_fit.set_title(assessment.template.title)
        ax_fit.legend(loc="best")

        residuals = assessment.fit_result.residuals
        if residuals is not None and residuals.size:
            residual_time = np.asarray(self._dataset.time, dtype=float)[: residuals.size]
            ax_res.axhline(0.0, color="#999999", linewidth=1.0)
            ax_res.plot(residual_time, residuals, color="#2a9d8f")
            residual_dataset = MuonDataset(
                time=residual_time.copy(),
                asymmetry=np.asarray(residuals, dtype=float).copy(),
                error=np.asarray(self._dataset.error, dtype=float)[: residuals.size].copy(),
                metadata=dict(self._dataset.metadata),
                run=self._dataset.run,
            )
            freq, _real, mag = fft_asymmetry(residual_dataset, window="hann", padding_factor=4)
            ax_fft.plot(freq, mag, color="#264653")
        ax_res.set_xlabel("Time (µs)")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals")
        ax_fft.set_xlabel("Frequency (MHz)")
        ax_fft.set_ylabel("|FFT|")
        ax_fft.set_title("Residual FFT")
        canvas.draw_idle()

    def _selected_assessment(self) -> CandidateAssessment | None:
        if self._recommendation is None:
            return None
        return (
            self._recommendation.assessment_for_key(self._selected_key)
            or self._recommendation.recommended_assessment
        )

    def _update_apply_page(self) -> None:
        assessment = self._selected_assessment()
        if self._recommendation is None or assessment is None:
            self._apply_selection_label.setText("")
            self._apply_parameters_table.setRowCount(0)
            self._apply_warning_text.setPlainText("")
            self._apply_recommended_btn.setEnabled(False)
            self._apply_selected_btn.setEnabled(False)
            return

        recommended = self._recommendation.recommended_assessment
        if recommended is None:
            self._apply_selection_label.setText(
                f"Selected candidate: {assessment.template.title}. No automatic recommendation is available."
            )
        else:
            self._apply_selection_label.setText(
                f"Recommended: {recommended.template.title}. Selected: {assessment.template.title}."
            )

        params = list(assessment.fit_result.parameters)
        self._apply_parameters_table.setRowCount(len(params))
        for row, parameter in enumerate(params):
            unc = assessment.fit_result.uncertainties.get(parameter.name, 0.0)
            self._apply_parameters_table.setItem(
                row, 0, QTableWidgetItem(get_param_info(parameter.name).unicode_label())
            )
            self._apply_parameters_table.setItem(row, 1, _numeric_item(parameter.value))
            self._apply_parameters_table.setItem(row, 2, _numeric_item(unc))

        warnings: list[str] = []
        if assessment.residual_gate_reasons:
            warnings.append("Residual warnings:")
            warnings.extend(f"• {reason}" for reason in assessment.residual_gate_reasons)
        else:
            warnings.append("No residual warnings were raised for the selected candidate.")
        warnings.append(f"AIC = {assessment.aic:.3f}")
        warnings.append(
            f"AICc = {assessment.aicc:.3f}"
            if assessment.aicc is not None
            else "AICc fell back to AIC for this candidate."
        )
        warnings.append(f"BIC = {assessment.bic:.3f}")
        self._apply_warning_text.setPlainText("\n".join(warnings))

        self._apply_recommended_btn.setEnabled(recommended is not None)
        self._apply_selected_btn.setEnabled(assessment.is_successful)

    def _apply_recommended_fit(self) -> None:
        if self._recommendation is None or self._recommendation.recommended_assessment is None:
            return
        self.apply_assessment_requested.emit(
            self._recommendation.recommended_assessment,
            self._recommendation,
        )
        self.statusBar().showMessage(
            f"Applied recommended fit: {self._recommendation.recommended_assessment.template.title}"
        )

    def _apply_selected_fit(self) -> None:
        assessment = self._selected_assessment()
        if self._recommendation is None or assessment is None:
            return
        self.apply_assessment_requested.emit(assessment, self._recommendation)
        self.statusBar().showMessage(f"Applied selected fit: {assessment.template.title}")

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

    def _go_previous_tab(self) -> None:
        self._tabs.setCurrentIndex(max(self._tabs.currentIndex() - 1, 0))

    def _go_next_tab(self) -> None:
        self._tabs.setCurrentIndex(min(self._tabs.currentIndex() + 1, self._tabs.count() - 1))

    def _update_navigation_buttons(self) -> None:
        index = self._tabs.currentIndex()
        if self._analysis_in_progress:
            self._previous_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return
        self._previous_btn.setEnabled(index > 0)
        self._next_btn.setEnabled(index < self._tabs.count() - 1)

    def _draw_figure(self, widget: QWidget, _content: object) -> None:
        figure = getattr(widget, "_figure", None)
        canvas = getattr(widget, "_canvas", None)
        if figure is None or canvas is None:
            return
        figure.clear()
        canvas.draw_idle()


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
