"""Non-modal guided fit wizard for single time-domain asymmetry spectra."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.composite import CompositeModel
from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    FitWizardRecommendation,
    SelectionMetric,
    build_fit_wizard_recommendation,
    rerank_fit_wizard_recommendation,
)
from asymmetry.core.fitting.parameters import get_param_info
from asymmetry.core.fourier.fft import fft_asymmetry


class FitWizardWorker(QObject):
    """Run fit-wizard analysis off the UI thread."""

    finished = Signal(int, object)  # request_id, FitWizardRecommendation
    error = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        dataset: MuonDataset,
        current_model: CompositeModel | None,
        metric: SelectionMetric,
    ) -> None:
        super().__init__()
        self._request_id = request_id
        self._dataset = dataset
        self._current_model = current_model
        self._metric = metric

    def run(self) -> None:
        try:
            recommendation = build_fit_wizard_recommendation(
                self._dataset,
                current_model=self._current_model,
                metric=self._metric,
            )
        except Exception as exc:
            self.error.emit(self._request_id, str(exc))
            return
        self.finished.emit(self._request_id, recommendation)


class FitWizardWindow(QMainWindow):
    """Present a guided workflow for model recommendation and comparison."""

    apply_assessment_requested = Signal(object, object)  # CandidateAssessment, FitWizardRecommendation

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit Wizard")
        self.resize(1280, 920)

        self._dataset: MuonDataset | None = None
        self._current_model: CompositeModel | None = None
        self._recommendation: FitWizardRecommendation | None = None
        self._selected_key: str | None = None
        self._analysis_request_id = 0
        self._analysis_in_progress = False
        self._analysis_thread: QThread | None = None
        self._analysis_worker: FitWizardWorker | None = None

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        heading_layout = QVBoxLayout()
        self._heading_label = QLabel("Fit Wizard")
        heading_font = QFont(self._heading_label.font())
        heading_font.setPointSize(max(heading_font.pointSize() + 4, 14))
        heading_font.setBold(True)
        self._heading_label.setFont(heading_font)
        heading_layout.addWidget(self._heading_label)

        self._status_label = QLabel(
            "Open the fit wizard on a single spectrum to fingerprint the data and compare curated candidate models."
        )
        self._status_label.setWordWrap(True)
        heading_layout.addWidget(self._status_label)
        layout.addLayout(heading_layout)

        controls_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Start Analysis")
        self._refresh_btn.clicked.connect(self._start_analysis)
        controls_row.addWidget(self._refresh_btn)
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #9a6700;")
        self._progress_label.setVisible(False)
        controls_row.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(220)
        controls_row.addWidget(self._progress_bar)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._fingerprint_tab = QWidget()
        self._portfolio_tab = QWidget()
        self._compare_tab = QWidget()
        self._apply_tab = QWidget()
        self._tabs.addTab(self._fingerprint_tab, "1. Fingerprint")
        self._tabs.addTab(self._portfolio_tab, "2. Candidate Portfolio")
        self._tabs.addTab(self._compare_tab, "3. Compare Fits")
        self._tabs.addTab(self._apply_tab, "4. Apply")

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
        layout.addLayout(nav_row)

        self._tabs.currentChanged.connect(self._update_navigation_buttons)
        self._update_navigation_buttons()
        self._refresh_btn.setEnabled(False)

    def set_analysis_context(
        self,
        dataset: MuonDataset,
        current_model: CompositeModel | None = None,
    ) -> None:
        """Prepare the wizard for a new dataset/model context."""
        self._dataset = dataset
        self._current_model = current_model
        self._analysis_request_id += 1
        self._heading_label.setText(f"Fit Wizard — Run {dataset.run_label}")
        self._recommendation = None
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(SelectionMetric.AICC.value)
        self._metric_combo.blockSignals(False)
        self._set_empty_state()
        if self._analysis_in_progress:
            self._status_label.setText(
                "Context updated while a previous analysis is still finishing. That result will be ignored; start a new analysis once the wizard is ready."
            )
            return
        self._status_label.setText(
            "Ready to fingerprint this spectrum. Click Start Analysis to run the wizard without blocking the main window."
        )
        self._set_busy(False)

    def _refresh_analysis(self) -> None:
        self._start_analysis()

    def _start_analysis(self) -> None:
        if self._dataset is None:
            self._status_label.setText("No dataset is available for the fit wizard.")
            self._set_empty_state()
            return
        if self._analysis_in_progress:
            return

        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._cleanup_analysis_thread()

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        self._set_busy(True)
        self._status_label.setText(
            "Running fit wizard analysis in the background. You can keep using the main window while recommendations are prepared."
        )
        self._set_empty_state()

        self._analysis_thread = QThread(self)
        self._analysis_worker = FitWizardWorker(
            request_id=request_id,
            dataset=self._dataset,
            current_model=self._current_model,
            metric=SelectionMetric.AICC,
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._on_analysis_finished)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.error.connect(self._analysis_thread.quit)
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.error.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._cleanup_analysis_thread)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _set_busy(self, busy: bool) -> None:
        self._analysis_in_progress = busy
        self._progress_label.setVisible(busy)
        self._progress_bar.setVisible(busy)
        self._progress_label.setText("Analysis in progress..." if busy else "")
        self._refresh_btn.setEnabled(self._dataset is not None and not busy)
        self._refresh_btn.setText("Refresh Analysis" if (self._recommendation is not None and not busy) else "Start Analysis")
        self._metric_combo.setEnabled(not busy and self._recommendation is not None)
        self._previous_btn.setEnabled(not busy and self._tabs.currentIndex() > 0)
        self._next_btn.setEnabled(not busy and self._tabs.currentIndex() < self._tabs.count() - 1)

    def _on_analysis_finished(self, request_id: int, recommendation: object) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            self._status_label.setText(
                "Context changed while analysis was running. Click Start Analysis to generate recommendations for the current spectrum."
            )
            self._set_empty_state()
            return
        if not isinstance(recommendation, FitWizardRecommendation):
            self._set_busy(False)
            self._status_label.setText("Fit wizard analysis returned an unexpected result.")
            self._set_empty_state()
            return

        self._recommendation = recommendation
        self._selected_key = self._recommendation.recommended_key
        if self._selected_key is None and self._recommendation.assessments:
            self._selected_key = self._recommendation.assessments[0].template.key
        self._status_label.setText(self._recommendation.summary)
        self._metric_combo.blockSignals(True)
        self._metric_combo.setCurrentText(self._recommendation.metric.value)
        self._metric_combo.blockSignals(False)
        self._set_busy(False)
        self._populate_from_recommendation()
        self._refresh_btn.setText("Refresh Analysis")

    def _on_analysis_error(self, request_id: int, message: str) -> None:
        thread = self._analysis_thread
        if thread is not None:
            thread.quit()
            thread.wait()
        if request_id != self._analysis_request_id:
            self._set_busy(False)
            self._status_label.setText(
                "Context changed while analysis was running. Click Start Analysis to generate recommendations for the current spectrum."
            )
            self._set_empty_state()
            return
        self._set_busy(False)
        self._recommendation = None
        self._status_label.setText(f"Fit wizard analysis failed: {message}")
        self._set_empty_state()

    def _cleanup_analysis_thread(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._analysis_in_progress:
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

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
        layout.addLayout(grid)

    def _build_portfolio_tab(self) -> None:
        layout = QVBoxLayout(self._portfolio_tab)
        self._portfolio_banner = QLabel("")
        self._portfolio_banner.setWordWrap(True)
        layout.addWidget(self._portfolio_banner)

        self._portfolio_table = QTableWidget(0, 4)
        self._portfolio_table.setHorizontalHeaderLabels(["Candidate", "Category", "Parameters", "Rationale"])
        self._portfolio_table.horizontalHeader().setStretchLastSection(True)
        self._portfolio_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._portfolio_table)

    def _build_compare_tab(self) -> None:
        layout = QVBoxLayout(self._compare_tab)
        self._compare_banner = QLabel("")
        self._compare_banner.setWordWrap(True)
        layout.addWidget(self._compare_banner)

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
        self._apply_parameters_table.setHorizontalHeaderLabels(["Parameter", "Value", "Uncertainty"])
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
            self._portfolio_table,
            self._compare_table,
            self._apply_parameters_table,
        ):
            table.setRowCount(0)
        self._fingerprint_banner.setText("")
        self._portfolio_banner.setText("")
        self._compare_banner.setText("")
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
            fingerprint_notes.append("No strong FFT peak was found in the default windowed transform.")
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

        self._fingerprint_banner.setText(" ".join(fingerprint_notes))
        self._portfolio_banner.setText(self._recommendation.summary)
        self._compare_banner.setText(self._recommendation.summary)
        if self._recommendation.recommended_assessment is None:
            self._apply_banner.setText(
                "No candidate passed the automatic residual gate. You can still inspect and apply a manually selected fit."
            )
        else:
            recommended = self._recommendation.recommended_assessment.template.title
            self._apply_banner.setText(
                f"The wizard recommends {recommended}. Apply it directly or choose an alternative from the comparison step."
            )

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
        ax_time.set_xlabel("Time (μs)")
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
        canvas.draw_idle()

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
            title_item = QTableWidgetItem(assessment.template.title)
            title_item.setData(Qt.ItemDataRole.UserRole, assessment.template.key)
            if assessment.template.key == self._recommendation.recommended_key:
                title_item.setFont(_bold_font(title_item.font()))
            if not assessment.residual_gate_passed:
                title_item.setForeground(QBrush(QColor("#9b2226")))
            self._compare_table.setItem(row, 0, title_item)
            self._compare_table.setItem(row, 1, _numeric_item(assessment.metric_value(self._recommendation.metric)))
            self._compare_table.setItem(row, 2, _numeric_item(assessment.aic))
            self._compare_table.setItem(row, 3, _numeric_item(assessment.aicc) if assessment.aicc is not None else QTableWidgetItem("AIC"))
            self._compare_table.setItem(row, 4, _numeric_item(assessment.bic))
            gate_text = "Pass" if assessment.residual_gate_passed else "Warn"
            self._compare_table.setItem(row, 5, QTableWidgetItem(gate_text))
            self._compare_table.setItem(row, 6, _numeric_item(assessment.fit_result.reduced_chi_squared))
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
        ax_fit.set_xlabel("Time (μs)")
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
        ax_res.set_xlabel("Time (μs)")
        ax_res.set_ylabel("Residual")
        ax_res.set_title("Residuals")
        ax_fft.set_xlabel("Frequency (MHz)")
        ax_fft.set_ylabel("|FFT|")
        ax_fft.set_title("Residual FFT")
        canvas.draw_idle()

    def _selected_assessment(self) -> CandidateAssessment | None:
        if self._recommendation is None:
            return None
        return self._recommendation.assessment_for_key(self._selected_key) or self._recommendation.recommended_assessment

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
            self._apply_parameters_table.setItem(row, 0, QTableWidgetItem(get_param_info(parameter.name).unicode_label()))
            self._apply_parameters_table.setItem(row, 1, _numeric_item(parameter.value))
            self._apply_parameters_table.setItem(row, 2, _numeric_item(unc))

        warnings: list[str] = []
        if assessment.residual_gate_reasons:
            warnings.append("Residual warnings:")
            warnings.extend(f"• {reason}" for reason in assessment.residual_gate_reasons)
        else:
            warnings.append("No residual warnings were raised for the selected candidate.")
        warnings.append(f"AIC = {assessment.aic:.3f}")
        warnings.append(f"AICc = {assessment.aicc:.3f}" if assessment.aicc is not None else "AICc fell back to AIC for this candidate.")
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
