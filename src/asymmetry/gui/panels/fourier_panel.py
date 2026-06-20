"""Fourier analysis panel — filter selection, FFT controls, and WiMDA-style modes.

Mirrors WiMDA's Analyse → Fourier dialog.
"""

from __future__ import annotations

import html
from functools import lru_cache

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QDoubleValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fourier.correlation import DEFAULT_CORR_ORDER
from asymmetry.gui.panels.spectral_moments_widget import SpectralMomentsWidget
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import status_font
from asymmetry.gui.styles.widgets import apply_param_table_style, build_primary_button_qss
from asymmetry.gui.utils.latex_renderer import render_latex_to_html_image

_PHASE_MODE_LABELS = (
    "(Power)^1/2",
    "Phase Spectrum",
    "Cos",
    "Sin",
    "Phase",
    "phaseOptReal",
    "Real+Imag",
    "Resolution (Burg)",
    "Correlation (radical)",
)

_LEGACY_DISPLAY_TO_MODE = {
    "imaginary": "Sin",
    "magnitude": "(Power)^1/2",
    "phase": "Phase Spectrum",
    "phase_opt_real": "phaseOptReal",
    "phaseoptreal": "phaseOptReal",
    "power": "(Power)^1/2",
    "real": "Phase",
    "real+imag": "Real+Imag",
    "real_imag": "Real+Imag",
    "burg": "Resolution (Burg)",
    "correlation": "Correlation (radical)",
    "correlation (radical)": "Correlation (radical)",
}

_PHASE_ACTIVE_COLOR = tokens.ACCENT
_PHASE_INACTIVE_COLOR = tokens.TEXT_MUTED
_PHASE_AUTOFILLED_COLOR = tokens.OK

#: Number of editable frequency-exclusion rows (matches WiMDA's ten slots).
_MAX_EXCLUSION_ROWS = 10
#: PSI continuous-source RF fundamental for the harmonics preset.
_PSI_RF_FUNDAMENTAL_MHZ = 50.63


def _latex_html(latex: str, *, render_latex_images: bool) -> str:
    if render_latex_images:
        image_html = render_latex_to_html_image(latex, font_size=15, dpi=170)
        if image_html is not None:
            return image_html
    return f"<code>{html.escape(latex)}</code>"


@lru_cache(maxsize=2)
def _build_fourier_mode_info_html(render_latex_images: bool) -> str:
    sections = [
        (
            "Complex Spectrum",
            r"F(f)=C(f)+iS(f)",
            "Asymmetry follows WiMDA's grouped FFT workflow: the selected grouped signal is filtered, transformed, and then one display mode is derived from the complex spectrum.",
        ),
        (
            "(Power)^1/2",
            r"M(f)=\sqrt{C(f)^2+S(f)^2}=|F(f)|",
            "WiMDA's default button labeled '(Power)' with a nearby 1/2 marker plots the FFT magnitude, not the squared power.",
        ),
        (
            "Phase Spectrum",
            r"\phi_s(f)=\operatorname{atan2}(S(f),\ C(f))",
            "This plots the raw spectral angle for each frequency bin before any manual, table-driven, or automatic phase correction is applied.",
        ),
        (
            "Cos",
            r"C(f)=\Re\{F(f)\}",
            "This is the uncorrected cosine component. In WiMDA it corresponds to a fixed phase of 0 degrees.",
        ),
        (
            "Sin",
            r"S(f)=\Im\{F(f)\}",
            "This is the uncorrected sine component. In WiMDA it corresponds to a fixed phase of 90 degrees.",
        ),
        (
            "Phase",
            (
                r"P(f)=C(f)\cos\theta(f)-S(f)\sin\theta(f)",
                r"\theta(f)=2\pi\left(\frac{\phi_0}{360}+f\,t_0\right)",
            ),
            "This is the WiMDA-style phase-corrected spectrum. Only this mode uses the manual phase entry, automatic phase estimate, phase table, and t0 offset.",
        ),
        (
            "phaseOptReal",
            (
                r"\varphi(i)=c_0+c_1\cdot\frac{i-i_{\min}}{i_{\max}-i_{\min}}",
                r"\min_{\,c_0,\,c_1}\;\left[-\sum_i p_i\ln p_i\;+\;\gamma\sum_{F_\mathrm{re}<0}F_\mathrm{re}^2\right]",
            ),
            "Entropy-based automatic phase optimiser (musrfit phaseOptReal). "
            "Finds the two-parameter linear phase c\u2080\u202f+\u202fc\u2081\u00b7i/N that makes the "
            "real spectrum most compact and non-negative. Does not use the manual phase, t0 offset, or "
            "phase table. Requires iminuit.",
        ),
        (
            "Real+Imag",
            (r"C(f)=\Re\{F(f)\}", r"S(f)=\Im\{F(f)\}"),
            "Overlays the cosine (real) and sine (imaginary) quadratures on one axis. A "
            "correctly phased absorption line is purely real with a flat imaginary part, so "
            "residual structure in the imaginary channel flags an imperfect phase correction.",
        ),
        (
            "Resolution (Burg) \u2014 diagnostic",
            r"P(\nu)=\dfrac{P_m}{\left|1-\sum_{k=1}^{m}a_k z^k\right|^2}",
            "All-poles autoregressive super-resolution. Diagnostic only: it qualitatively "
            "resolves close lines from short windows and the FPE-optimal pole count hints at the "
            "line count, but it can split strong peaks and seed spurious baseline peaks, and "
            "carries no uncertainties. Use frequency-domain fitting or MaxEnt for quantitative results.",
        ),
    ]

    section_html: list[str] = [
        "<html><body>",
        "<h2>FFT Phase Modes</h2>",
        (
            "<p>WiMDA computes one complex FFT for each included group and then displays "
            "different projections of that same spectrum. Asymmetry follows the same model here.</p>"
        ),
    ]
    for title, latex, text in sections:
        block = (latex,) if isinstance(latex, str) else latex
        equations = "".join(
            _latex_html(item, render_latex_images=render_latex_images) for item in block
        )
        section_html.extend(
            (f"<h3>{html.escape(title)}</h3>", equations, f"<p>{html.escape(text)}</p>")
        )
    section_html.append(
        "<p><b>Practical note:</b> (Power)^1/2, Phase Spectrum, Cos, and Sin are all derived "
        "from the uncorrected complex FFT. Phase applies the selected phase correction first and then "
        "plots the phase-corrected real projection. phaseOptReal automatically determines the "
        "best phase using entropy minimisation; all manual phase controls are ignored.</p>"
    )
    section_html.append("</body></html>")
    return "".join(section_html)


def show_fourier_mode_info_dialog(parent: QWidget) -> QDialog:
    """Open a non-modal info dialog describing the WiMDA-style FFT modes."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("FFT Phase Modes")
    dialog.resize(760, 560)

    layout = QVBoxLayout(dialog)
    browser = QTextBrowser(dialog)
    browser.setOpenExternalLinks(False)
    browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    browser.setHtml(_build_fourier_mode_info_html(render_latex_images=False))
    layout.addWidget(browser)

    close_btn = QPushButton("Close", dialog)
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn)

    dialog.setModal(False)
    dialog.show()

    QTimer.singleShot(
        0,
        lambda: browser.setHtml(_build_fourier_mode_info_html(render_latex_images=True)),
    )

    return dialog


class FourierPanel(QWidget):
    """Controls for frequency-domain analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._table_group_ids: list[int] = []
        self._auto_filled_group_ids: set[int] = set()
        self._phase_table_updating = False
        self._mode_info_dialog: QDialog | None = None

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(scroll_area)

        content = QWidget()
        scroll_area.setWidget(content)
        content_layout = QVBoxLayout(content)

        phase_mode_group = QGroupBox("FFT Phase Mode")
        phase_mode_layout = QGridLayout(phase_mode_group)

        self._phase_mode_button_group = QButtonGroup(self)
        self._power_sqrt_radio = QRadioButton("(Power)^1/2")
        self._phase_spectrum_radio = QRadioButton("Phase Spectrum")
        self._cos_radio = QRadioButton("Cos")
        self._sin_radio = QRadioButton("Sin")
        self._phase_mode_radio = QRadioButton("Phase")
        self._real_imag_radio = QRadioButton("Real+Imag")
        self._phase_opt_real_radio = QRadioButton("phaseOptReal")
        self._phase_opt_real_radio.setStyleSheet(
            f"QRadioButton {{ color: {tokens.ACCENT}; font-weight: 600; padding-bottom: 2px; }}"
        )
        self._phase_opt_real_radio.setMinimumHeight(
            self._phase_opt_real_radio.sizeHint().height() + 4
        )
        self._burg_radio = QRadioButton("Resolution (Burg) — diagnostic")
        self._burg_radio.setStyleSheet(
            f"QRadioButton {{ color: {tokens.WARN}; font-weight: 600; padding-bottom: 2px; }}"
        )
        self._burg_radio.setToolTip(
            "Burg all-poles super-resolution. Diagnostic only: qualitatively "
            "resolves close lines and hints at the line count, but can split "
            "strong peaks and carries no uncertainties. Use fitting or MaxEnt "
            "for quantitative results."
        )
        self._correlation_radio = QRadioButton("Correlation (radical) — specialist")
        self._correlation_radio.setStyleSheet(
            f"QRadioButton {{ color: {tokens.WARN}; font-weight: 600; padding-bottom: 2px; }}"
        )
        self._correlation_radio.setToolTip(
            "Muoniated-radical correlation spectrum. Maps a transverse-field "
            "radical's Breit–Rabi line pair onto the muon hyperfine-coupling "
            "(Aμ) axis: a peak appears at Aμ. Specialist tool for identifying "
            "muoniated radicals; high transverse field only."
        )
        self._power_sqrt_radio.setChecked(True)

        mode_column = QWidget()
        mode_column_layout = QVBoxLayout(mode_column)
        mode_column_layout.setContentsMargins(0, 0, 0, 2)
        mode_column_layout.setSpacing(6)
        for button in (
            self._power_sqrt_radio,
            self._phase_spectrum_radio,
            self._cos_radio,
            self._sin_radio,
            self._phase_mode_radio,
            self._real_imag_radio,
            self._phase_opt_real_radio,
            self._burg_radio,
            self._correlation_radio,
        ):
            self._phase_mode_button_group.addButton(button)
            mode_column_layout.addWidget(button)
        mode_column_layout.addStretch()

        self._phase_mode_info_btn = QPushButton("Info")
        phase_mode_layout.addWidget(mode_column, 0, 0)
        phase_mode_layout.addWidget(
            self._phase_mode_info_btn,
            0,
            1,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        content_layout.addWidget(phase_mode_group)

        apodisation_group = QGroupBox("Apodisation")
        apodisation_form = QFormLayout(apodisation_group)

        self._filter_start_edit = QLineEdit("0.0")
        self._filter_start_edit.setFont(mono_font(11.0))
        apodisation_form.addRow("Filter start time (μs):", self._filter_start_edit)

        self._filter_time_constant_edit = QLineEdit("1.5")
        self._filter_time_constant_edit.setFont(mono_font(11.0))
        apodisation_form.addRow("Filter time constant (μs):", self._filter_time_constant_edit)

        self._filter_button_group = QButtonGroup(self)
        self._filter_lorentzian_radio = QRadioButton("Lorentzian")
        self._filter_gaussian_radio = QRadioButton("Gaussian")
        self._filter_none_radio = QRadioButton("None")
        self._filter_none_radio.setChecked(True)
        self._filter_button_group.addButton(self._filter_lorentzian_radio)
        self._filter_button_group.addButton(self._filter_gaussian_radio)
        self._filter_button_group.addButton(self._filter_none_radio)
        filter_mode_row = QWidget()
        filter_mode_layout = QHBoxLayout(filter_mode_row)
        filter_mode_layout.setContentsMargins(0, 0, 0, 0)
        filter_mode_layout.setSpacing(10)
        filter_mode_layout.addWidget(self._filter_lorentzian_radio)
        filter_mode_layout.addWidget(self._filter_gaussian_radio)
        filter_mode_layout.addWidget(self._filter_none_radio)
        filter_mode_layout.addStretch()
        filter_mode_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        apodisation_form.addRow(filter_mode_row)

        content_layout.addWidget(apodisation_group)

        groups_group = QGroupBox("Groups")
        groups_layout = QVBoxLayout(groups_group)
        self._phase_table = QTableWidget(0, 3)
        self._phase_table.setHorizontalHeaderLabels(["Include", "Group", "Phase (deg)"])
        apply_param_table_style(self._phase_table)
        self._phase_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._phase_table.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self._phase_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._phase_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._phase_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._phase_table.setMinimumHeight(100)
        groups_layout.addWidget(self._phase_table)
        groups_form = QFormLayout()
        self._use_phase_table_check = QCheckBox("Use per-group phase table")
        groups_form.addRow("Group phases:", self._use_phase_table_check)
        groups_layout.addLayout(groups_form)
        content_layout.addWidget(groups_group)

        phase_group = QGroupBox("Phase")
        phase_layout = QVBoxLayout(phase_group)
        phase_form = QFormLayout()

        self._phase_spin = QLineEdit("0")
        self._phase_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._phase_spin.setFont(mono_font(11.0))
        self._phase_spin.setValidator(QDoubleValidator(-3600.0, 3600.0, 6, self))
        phase_form.addRow("Phase (deg):", self._phase_spin)

        self._t0_offset_spin = QLineEdit("0")
        self._t0_offset_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._t0_offset_spin.setFont(mono_font(11.0))
        self._t0_offset_spin.setValidator(QDoubleValidator(-1000.0, 1000.0, 6, self))
        phase_form.addRow("t0 Offset (\u03bcs):", self._t0_offset_spin)

        self._auto_method_combo = QComboBox()
        self._auto_method_combo.addItems(["Peak", "Average"])
        phase_form.addRow("Auto method:", self._auto_method_combo)
        phase_layout.addLayout(phase_form)

        self._auto_phase_btn = QPushButton("Fill phases")
        self._auto_phase_btn.setToolTip("Fill per-group phase estimates from the data.")
        phase_layout.addWidget(self._auto_phase_btn)

        content_layout.addWidget(phase_group)

        fft_settings_group = QGroupBox("FFT settings")
        fft_settings_layout = QVBoxLayout(fft_settings_group)
        fft_settings_form = QFormLayout()

        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(1, 16)
        self._padding_spin.setValue(1)
        fft_settings_form.addRow("Zero-pad factor:", self._padding_spin)

        self._subtract_average_signal_check = QCheckBox("Subtract average signal")
        self._subtract_average_signal_check.setChecked(True)
        fft_settings_form.addRow(self._subtract_average_signal_check)

        self._estimate_average_error_check = QCheckBox("Estimate errors for averaged spectra")
        self._estimate_average_error_check.setText("Average errors")
        self._estimate_average_error_check.setToolTip("Estimate errors for averaged spectra.")
        fft_settings_form.addRow(self._estimate_average_error_check)

        self._average_summary_label = QLabel("")
        self._average_summary_label.setWordWrap(True)
        fft_settings_form.addRow("Average summary:", self._average_summary_label)
        fft_settings_layout.addLayout(fft_settings_form)
        content_layout.addWidget(fft_settings_group)

        content_layout.addWidget(self._build_conditioning_group())
        content_layout.addWidget(self._build_diamag_group())
        content_layout.addWidget(self._build_exclusions_group())
        # Spectral moments — a sibling of the advanced stack (range/cutoff control
        # over the lineshape-faithful spectrum); the host wires it up.
        self._moments_widget = SpectralMomentsWidget()
        content_layout.addWidget(self._moments_widget)

        content_layout.addStretch()

        # Pinned action footer — the primary action of this ~9-section panel is
        # "Compute FFT", which previously sat at the very bottom of the scroll
        # content and was unreachable at the default window size. Keep the action
        # cluster (background hint + Compute + Apply + status) outside the scroll
        # area so it stays visible at any scroll position.
        layout.addWidget(self._build_action_footer())

        self._use_phase_table_check.toggled.connect(self._update_phase_table_enabled)
        self._phase_table.itemChanged.connect(self._on_phase_table_item_changed)
        self._filter_lorentzian_radio.toggled.connect(self._update_filter_controls_enabled)
        self._filter_gaussian_radio.toggled.connect(self._update_filter_controls_enabled)
        self._filter_none_radio.toggled.connect(self._update_filter_controls_enabled)
        self._power_sqrt_radio.toggled.connect(self._update_phase_controls_enabled)
        self._phase_spectrum_radio.toggled.connect(self._update_phase_controls_enabled)
        self._cos_radio.toggled.connect(self._update_phase_controls_enabled)
        self._sin_radio.toggled.connect(self._update_phase_controls_enabled)
        self._phase_mode_radio.toggled.connect(self._update_phase_controls_enabled)
        self._real_imag_radio.toggled.connect(self._update_phase_controls_enabled)
        self._phase_opt_real_radio.toggled.connect(self._update_phase_controls_enabled)
        self._burg_radio.toggled.connect(self._update_phase_controls_enabled)
        self._burg_radio.toggled.connect(self._update_conditioning_enabled)
        self._correlation_radio.toggled.connect(self._update_phase_controls_enabled)
        self._correlation_radio.toggled.connect(self._update_conditioning_enabled)
        self._phase_mode_info_btn.clicked.connect(self._show_phase_mode_info)
        self._phase_spin.editingFinished.connect(self._normalize_phase_line_edits)
        self._t0_offset_spin.editingFinished.connect(self._normalize_phase_line_edits)
        self._update_phase_table_enabled(self._use_phase_table_check.isChecked())
        self._update_filter_controls_enabled()
        self._update_phase_controls_enabled()

    def _build_action_footer(self) -> QWidget:
        """Build the always-visible footer holding the Compute FFT action."""
        footer = QWidget()
        footer.setObjectName("fourierActionFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 6, 0, 0)
        footer_layout.setSpacing(4)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Plain)
        divider.setStyleSheet(f"color: {tokens.BORDER};")
        footer_layout.addWidget(divider)

        # Read-only hint: the grouping's pre-FFT background correction is
        # inherited by the Fourier input (F3) and is otherwise invisible here.
        self._background_hint_label = QLabel("")
        self._background_hint_label.setWordWrap(True)
        self._background_hint_label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self.set_background_hint(None)
        footer_layout.addWidget(self._background_hint_label)

        self._fft_btn = QPushButton("Compute FFT")
        self._fft_btn.setStyleSheet(build_primary_button_qss())
        footer_layout.addWidget(self._fft_btn)

        self._apply_to_selection_btn = QPushButton("Apply to selection")
        self._apply_to_selection_btn.setToolTip(
            "Copy this run's Fourier settings to the other selected runs and "
            "generate their spectra."
        )
        footer_layout.addWidget(self._apply_to_selection_btn)

        self._status_label = QLabel("")
        self._status_label.setFont(status_font())
        self._status_label.setWordWrap(True)
        footer_layout.addWidget(self._status_label)

        return footer

    # ── conditioning + exclusions sections ─────────────────────────────

    def _build_conditioning_group(self) -> QGroupBox:
        """Build the pulse-compensation + baseline conditioning section."""
        group = QGroupBox("Conditioning")
        form = QFormLayout(group)

        self._pulse_comp_check = QCheckBox("Pulse-response compensation")
        self._pulse_comp_check.setToolTip(
            "Divide the spectrum by the ISIS pulse amplitude R(f) to undo the "
            "high-frequency rolloff. Capped and cut off at the pulse node."
        )
        form.addRow(self._pulse_comp_check)

        self._pulse_width_edit = QLineEdit("")
        self._pulse_width_edit.setPlaceholderText("auto (from metadata)")
        self._pulse_width_edit.setFont(mono_font(11.0))
        self._pulse_width_edit.setValidator(QDoubleValidator(0.0, 10.0, 6, self))
        form.addRow("Pulse half-width (μs):", self._pulse_width_edit)

        self._pulse_max_gain_edit = QLineEdit("25")
        self._pulse_max_gain_edit.setFont(mono_font(11.0))
        self._pulse_max_gain_edit.setValidator(QDoubleValidator(1.0, 1000.0, 3, self))
        form.addRow("Max gain:", self._pulse_max_gain_edit)

        self._baseline_mode_combo = QComboBox()
        self._baseline_mode_combo.addItem("None", userData="none")
        self._baseline_mode_combo.addItem("Robust σ-clip", userData="sigma_clip")
        self._baseline_mode_combo.addItem("WiMDA single-pass", userData="wimda")
        form.addRow("Baseline offset:", self._baseline_mode_combo)

        self._baseline_kappa_edit = QLineEdit("2")
        self._baseline_kappa_edit.setFont(mono_font(11.0))
        self._baseline_kappa_edit.setValidator(QDoubleValidator(0.5, 10.0, 3, self))
        form.addRow("Clip κ (σ):", self._baseline_kappa_edit)

        self._burg_order_min_spin = QSpinBox()
        self._burg_order_min_spin.setRange(1, 200)
        self._burg_order_min_spin.setValue(2)
        self._burg_order_max_spin = QSpinBox()
        self._burg_order_max_spin.setRange(1, 200)
        self._burg_order_max_spin.setValue(40)
        burg_row = QWidget()
        burg_layout = QHBoxLayout(burg_row)
        burg_layout.setContentsMargins(0, 0, 0, 0)
        burg_layout.addWidget(self._burg_order_min_spin)
        burg_layout.addWidget(QLabel("to"))
        burg_layout.addWidget(self._burg_order_max_spin)
        burg_layout.addStretch()
        form.addRow("Burg pole scan:", burg_row)

        # Muoniated-radical correlation controls (revealed by the specialist
        # display-mode radio, as with the Burg pole-scan above).
        self._correlation_field_edit = QLineEdit("")
        self._correlation_field_edit.setPlaceholderText("auto (run field)")
        self._correlation_field_edit.setFont(mono_font(11.0))
        self._correlation_field_edit.setValidator(QDoubleValidator(0.0, 1.0e6, 3, self))
        self._correlation_field_edit.setToolTip(
            "Transverse field (Gauss) used for the Breit–Rabi pairing; defaults "
            "to the run's applied field."
        )
        form.addRow("Correlation field (G):", self._correlation_field_edit)

        self._correlation_order_spin = QSpinBox()
        self._correlation_order_spin.setRange(0, 10)
        self._correlation_order_spin.setValue(DEFAULT_CORR_ORDER)
        self._correlation_order_spin.setToolTip(
            "CorrFn ratio-penalty order: higher values suppress unequal-amplitude "
            "(spurious) line pairs more strongly. 0 = plain product."
        )
        form.addRow("Correlation order:", self._correlation_order_spin)

        self._pulse_comp_check.toggled.connect(self._update_conditioning_enabled)
        self._baseline_mode_combo.currentIndexChanged.connect(self._update_conditioning_enabled)
        self._update_conditioning_enabled()
        return group

    def _build_diamag_group(self) -> QGroupBox:
        """Build the single three-way diamagnetic-line control (F4).

        One mutually-exclusive choice replaces the two former checkboxes
        (time-domain fit-and-subtract in Conditioning, post-FFT band exclusion
        in Exclusions). Both ``.asymp`` keys (``remove_diamag`` /
        ``diamag_exclusion``) stay readable; :meth:`get_state` derives them from
        the selected mode and they remain mutually exclusive.
        """
        group = QGroupBox("Diamagnetic line")
        form = QFormLayout(group)

        self._diamag_mode_combo = QComboBox()
        self._diamag_mode_combo.addItem("Leave", userData="leave")
        self._diamag_mode_combo.addItem("Fit & subtract", userData="subtract")
        self._diamag_mode_combo.addItem("Exclude band", userData="band")
        self._diamag_mode_combo.setToolTip(
            "Leave: no diamagnetic handling.\n"
            "Fit & subtract: fit a damped cosine at the diamagnetic line and "
            "subtract it before the FFT, reporting the fitted field (preferred "
            "for correlation / Aμ work, falls back to nothing below 5 G).\n"
            "Exclude band: hard-zero a band centred on γ_μ·B after the FFT — the "
            "robust fallback for lines too strong or distorted to fit."
        )
        form.addRow("Diamagnetic line:", self._diamag_mode_combo)

        self._diamag_width_edit = QLineEdit("0.3")
        self._diamag_width_edit.setFont(mono_font(11.0))
        self._diamag_width_edit.setValidator(QDoubleValidator(0.0, 100.0, 4, self))
        self._diamag_width_edit.setToolTip(
            "Half-width of the excluded band, centred on γ_μ·B (used by 'Exclude band')."
        )
        form.addRow("Band half-width (MHz):", self._diamag_width_edit)

        self._diamag_mode_combo.currentIndexChanged.connect(self._update_diamag_controls_enabled)
        self._update_diamag_controls_enabled()
        return group

    def _build_exclusions_group(self) -> QGroupBox:
        """Build the frequency-range exclusions section."""
        group = QGroupBox("Exclusions")
        layout = QVBoxLayout(group)

        self._exclude_enabled_check = QCheckBox("Exclude frequency ranges")
        layout.addWidget(self._exclude_enabled_check)

        self._exclusion_table = QTableWidget(_MAX_EXCLUSION_ROWS, 2)
        self._exclusion_table.setHorizontalHeaderLabels(["Centre (MHz)", "Half-width (MHz)"])
        apply_param_table_style(self._exclusion_table)
        self._exclusion_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._exclusion_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self._exclusion_table.verticalHeader().setVisible(False)
        self._exclusion_table.setMinimumHeight(150)
        for row in range(_MAX_EXCLUSION_ROWS):
            for col in range(2):
                self._exclusion_table.setItem(row, col, QTableWidgetItem(""))
        layout.addWidget(self._exclusion_table)

        self._psi_preset_btn = QPushButton("PSI RF harmonics")
        self._psi_preset_btn.setToolTip("Fill DC + 50.63 MHz × 1–5.")
        self._psi_preset_btn.clicked.connect(self._apply_psi_harmonics_preset)
        layout.addWidget(self._psi_preset_btn)

        self._exclude_enabled_check.toggled.connect(self._update_exclusion_enabled)
        self._update_exclusion_enabled()
        return group

    def _update_conditioning_enabled(self) -> None:
        pulse_on = self._pulse_comp_check.isChecked()
        self._pulse_width_edit.setEnabled(pulse_on)
        self._pulse_max_gain_edit.setEnabled(pulse_on)
        baseline_on = str(self._baseline_mode_combo.currentData() or "none") != "none"
        self._baseline_kappa_edit.setEnabled(baseline_on)
        burg_on = self._burg_radio.isChecked()
        self._burg_order_min_spin.setEnabled(burg_on)
        self._burg_order_max_spin.setEnabled(burg_on)
        correlation_on = self._correlation_radio.isChecked()
        self._correlation_field_edit.setEnabled(correlation_on)
        self._correlation_order_spin.setEnabled(correlation_on)

    def _update_exclusion_enabled(self) -> None:
        enabled = self._exclude_enabled_check.isChecked()
        self._exclusion_table.setEnabled(enabled)
        self._psi_preset_btn.setEnabled(enabled)

    def _update_diamag_controls_enabled(self) -> None:
        """Enable the band half-width only for the 'Exclude band' mode."""
        self._diamag_width_edit.setEnabled(self._diamag_mode() == "band")

    def _diamag_mode(self) -> str:
        """Return the selected diamagnetic mode: ``leave`` / ``subtract`` / ``band``."""
        return str(self._diamag_mode_combo.currentData() or "leave")

    def _set_diamag_mode(self, mode: str) -> None:
        index = self._diamag_mode_combo.findData(mode)
        self._diamag_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _apply_psi_harmonics_preset(self) -> None:
        """Fill the exclusion table with DC + 50.63 MHz harmonics 1–5."""
        centres = [0.0, *(_PSI_RF_FUNDAMENTAL_MHZ * h for h in range(1, 6))]
        width = 0.5
        self._exclude_enabled_check.setChecked(True)
        for row in range(_MAX_EXCLUSION_ROWS):
            centre_text = f"{centres[row]:g}" if row < len(centres) else ""
            width_text = f"{width:g}" if row < len(centres) else ""
            self._exclusion_table.setItem(row, 0, QTableWidgetItem(centre_text))
            self._exclusion_table.setItem(row, 1, QTableWidgetItem(width_text))
        self._update_exclusion_enabled()

    def exclusion_ranges(self) -> list[tuple[float, float]]:
        """Return the editable ``(centre, half-width)`` exclusion ranges."""
        ranges: list[tuple[float, float]] = []
        for row in range(self._exclusion_table.rowCount()):
            centre_item = self._exclusion_table.item(row, 0)
            width_item = self._exclusion_table.item(row, 1)
            centre_text = centre_item.text().strip() if centre_item else ""
            width_text = width_item.text().strip() if width_item else ""
            if not centre_text or not width_text:
                continue
            try:
                centre = float(centre_text)
                width = float(width_text)
            except ValueError:
                continue
            if width > 0.0:
                ranges.append((centre, width))
        return ranges

    def _set_exclusion_ranges(self, ranges: list) -> None:
        for row in range(_MAX_EXCLUSION_ROWS):
            if (
                row < len(ranges)
                and isinstance(ranges[row], (list, tuple))
                and len(ranges[row]) == 2
            ):
                centre, width = ranges[row]
                self._exclusion_table.setItem(row, 0, QTableWidgetItem(f"{float(centre):g}"))
                self._exclusion_table.setItem(row, 1, QTableWidgetItem(f"{float(width):g}"))
            else:
                self._exclusion_table.setItem(row, 0, QTableWidgetItem(""))
                self._exclusion_table.setItem(row, 1, QTableWidgetItem(""))

    def set_background_hint(self, text: str | None) -> None:
        """Show the inherited grouping-background state above the FFT button (F3).

        ``text`` is the resolved mode description (e.g. ``"tail-fit"``) or
        ``None`` when no grouping background correction is active.
        """
        if text:
            self._background_hint_label.setText(f"Background: {text}, inherited from grouping")
        else:
            self._background_hint_label.setText("Background: off")

    def set_fft_status(self, message: str, *, success: bool = False) -> None:
        """Set the status label below the Compute FFT button."""
        if success:
            self._status_label.setText(
                f'<span style="color: {tokens.OK};">● {html.escape(str(message))}</span>'
            )
        else:
            self._status_label.setText(str(message))

    def _show_phase_mode_info(self) -> None:
        self._mode_info_dialog = show_fourier_mode_info_dialog(self)

    def _current_display_mode(self) -> str:
        if self._phase_spectrum_radio.isChecked():
            return "Phase Spectrum"
        if self._cos_radio.isChecked():
            return "Cos"
        if self._sin_radio.isChecked():
            return "Sin"
        if self._phase_mode_radio.isChecked():
            return "Phase"
        if self._real_imag_radio.isChecked():
            return "Real+Imag"
        if self._phase_opt_real_radio.isChecked():
            return "phaseOptReal"
        if self._burg_radio.isChecked():
            return "Resolution (Burg)"
        if self._correlation_radio.isChecked():
            return "Correlation (radical)"
        return "(Power)^1/2"

    def _current_filter_mode(self) -> str:
        if self._filter_lorentzian_radio.isChecked():
            return "lorentzian"
        if self._filter_gaussian_radio.isChecked():
            return "gaussian"
        return "none"

    @staticmethod
    def _format_float_text(value: float) -> str:
        return f"{float(value):g}"

    @staticmethod
    def _parse_float_text(text: str, default: float) -> float:
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _set_line_edit_text_color(widget: QLineEdit, color: str) -> None:
        widget.setStyleSheet(f"QLineEdit {{ color: {color}; }}")

    def _phase_value(self) -> float:
        return self._parse_float_text(self._phase_spin.text(), 0.0)

    def _t0_offset_value(self) -> float:
        return self._parse_float_text(self._t0_offset_spin.text(), 0.0)

    def _normalize_phase_line_edits(self) -> None:
        self._phase_spin.setText(self._format_float_text(self._phase_value()))
        self._t0_offset_spin.setText(self._format_float_text(self._t0_offset_value()))
        self._update_phase_colors()

    @staticmethod
    def _coerce_filter_mode(value: object) -> str:
        mode = str(value or "none").strip().lower()
        if mode in {"gaussian", "lorentzian", "none"}:
            return mode
        return "none"

    @staticmethod
    def _coerce_display_mode(value: object) -> str:
        text = str(value or "(Power)^1/2").strip()
        if text in _PHASE_MODE_LABELS:
            return text
        return _LEGACY_DISPLAY_TO_MODE.get(text.lower(), "(Power)^1/2")

    def _set_display_mode(self, value: object) -> None:
        mode = self._coerce_display_mode(value)
        self._power_sqrt_radio.setChecked(mode == "(Power)^1/2")
        self._phase_spectrum_radio.setChecked(mode == "Phase Spectrum")
        self._cos_radio.setChecked(mode == "Cos")
        self._sin_radio.setChecked(mode == "Sin")
        self._phase_mode_radio.setChecked(mode == "Phase")
        self._real_imag_radio.setChecked(mode == "Real+Imag")
        self._phase_opt_real_radio.setChecked(mode == "phaseOptReal")
        self._burg_radio.setChecked(mode == "Resolution (Burg)")
        self._correlation_radio.setChecked(mode == "Correlation (radical)")

    def _update_filter_controls_enabled(self) -> None:
        enabled = self._current_filter_mode() != "none"
        self._filter_start_edit.setEnabled(enabled)
        self._filter_time_constant_edit.setEnabled(enabled)

    def _update_phase_controls_enabled(self) -> None:
        mode = self._current_display_mode()
        is_phase_mode = mode == "Phase"
        is_entropy_mode = mode == "phaseOptReal"
        if self._cos_radio.isChecked():
            self._phase_spin.setText(self._format_float_text(0.0))
        elif self._sin_radio.isChecked():
            self._phase_spin.setText(self._format_float_text(90.0))
        # phaseOptReal: all manual phase controls disabled; optimizer handles everything
        self._phase_spin.setEnabled(is_phase_mode)
        self._t0_offset_spin.setEnabled(is_phase_mode)
        self._auto_method_combo.setEnabled(is_phase_mode)
        self._auto_phase_btn.setEnabled(is_phase_mode)
        self._use_phase_table_check.setEnabled(is_phase_mode and not is_entropy_mode)
        self._update_phase_table_enabled(
            self._use_phase_table_check.isChecked() and not is_entropy_mode
        )
        self._update_phase_colors()

    def _update_phase_table_enabled(self, enabled: bool) -> None:
        """Keep group selection active while gating phase-cell editing."""
        self._phase_table.setEnabled(True)
        phase_edit_enabled = bool(enabled) and self._current_display_mode() == "Phase"
        self._phase_table_updating = True
        try:
            for row in range(self._phase_table.rowCount()):
                phase_item = self._phase_table.item(row, 2)
                if phase_item is None:
                    continue
                phase_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | (Qt.ItemFlag.ItemIsEditable if phase_edit_enabled else Qt.ItemFlag(0))
                )
        finally:
            self._phase_table_updating = False
        self._update_phase_colors()

    def _update_phase_colors(self) -> None:
        is_phase_mode = self._current_display_mode() == "Phase"
        use_group_phases = is_phase_mode and self._use_phase_table_check.isChecked()
        self._set_line_edit_text_color(
            self._phase_spin,
            _PHASE_INACTIVE_COLOR if use_group_phases or not is_phase_mode else _PHASE_ACTIVE_COLOR,
        )
        self._phase_table_updating = True
        try:
            for row, group_id in enumerate(self._table_group_ids):
                phase_item = self._phase_table.item(row, 2)
                if phase_item is None:
                    continue
                if use_group_phases and int(group_id) in self._auto_filled_group_ids:
                    color = _PHASE_AUTOFILLED_COLOR
                elif use_group_phases:
                    color = _PHASE_ACTIVE_COLOR
                else:
                    color = _PHASE_INACTIVE_COLOR
                phase_item.setForeground(QColor(color))
        finally:
            self._phase_table_updating = False

    def _on_phase_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._phase_table_updating or item.column() != 2:
            return
        group_id = (
            self._table_group_ids[item.row()]
            if 0 <= item.row() < len(self._table_group_ids)
            else None
        )
        if group_id is not None and int(group_id) in self._auto_filled_group_ids:
            self._auto_filled_group_ids.discard(int(group_id))
        self._update_phase_colors()

    # ── project state helpers ──────────────────────────────────────────

    def group_phase_table(self) -> dict[int, float]:
        """Return the per-group phase table as ``{group_id: phase_deg}``."""
        phases: dict[int, float] = {}
        for row, group_id in enumerate(self._table_group_ids):
            item = self._phase_table.item(row, 2)
            text = item.text().strip() if item is not None else ""
            try:
                phases[int(group_id)] = float(text)
            except (TypeError, ValueError):
                phases[int(group_id)] = 0.0
        return phases

    def group_enabled_table(self) -> dict[int, bool]:
        """Return the per-group inclusion table as ``{group_id: enabled}``."""
        enabled: dict[int, bool] = {}
        for row, group_id in enumerate(self._table_group_ids):
            item = self._phase_table.item(row, 0)
            enabled[int(group_id)] = item is None or item.checkState() == Qt.CheckState.Checked
        return enabled

    def selected_group_ids(self) -> list[int]:
        """Return the ordered list of detector groups enabled for grouped FFTs."""
        enabled = self.group_enabled_table()
        return [group_id for group_id in self._table_group_ids if enabled.get(int(group_id), True)]

    def set_group_definitions(
        self,
        group_names: dict[int, str],
        phase_table: dict[int, float] | None = None,
        enabled_table: dict[int, bool] | None = None,
    ) -> None:
        """Populate the editable group-phase table for the active run."""
        current_phases = self.group_phase_table()
        current_enabled = self.group_enabled_table()
        if phase_table is not None:
            current_phases.update({int(k): float(v) for k, v in phase_table.items()})
        if enabled_table is not None:
            current_enabled.update({int(k): bool(v) for k, v in enabled_table.items()})

        group_ids = sorted(int(group_id) for group_id in group_names)
        self._table_group_ids = group_ids
        self._auto_filled_group_ids.intersection_update(group_ids)
        self._phase_table.setRowCount(len(group_ids))
        self._phase_table_updating = True
        try:
            for row, group_id in enumerate(group_ids):
                include_item = QTableWidgetItem()
                include_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                include_item.setCheckState(
                    Qt.CheckState.Checked
                    if current_enabled.get(int(group_id), True)
                    else Qt.CheckState.Unchecked
                )
                self._phase_table.setItem(row, 0, include_item)

                name_item = QTableWidgetItem(str(group_names[group_id]))
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._phase_table.setItem(row, 1, name_item)

                phase_value = current_phases.get(int(group_id), 0.0)
                phase_item = QTableWidgetItem(f"{float(phase_value):.3f}")
                self._phase_table.setItem(row, 2, phase_item)
        finally:
            self._phase_table_updating = False
        self._update_phase_table_enabled(self._use_phase_table_check.isChecked())

    def set_group_phases(self, phase_table: dict[int, float], *, auto_filled: bool = False) -> None:
        """Update existing group-phase rows without rebuilding the table."""
        current = self.group_phase_table()
        current.update({int(k): float(v) for k, v in phase_table.items()})
        if not self._table_group_ids:
            fallback_names = {int(k): f"Group {int(k)}" for k in current}
            self.set_group_definitions(fallback_names, current)
            if auto_filled:
                self._auto_filled_group_ids = {int(k) for k in phase_table}
                self._update_phase_colors()
            return
        self.set_group_definitions(
            {
                int(group_id): self._phase_table.item(row, 1).text()
                for row, group_id in enumerate(self._table_group_ids)
            },
            current,
            self.group_enabled_table(),
        )
        if auto_filled:
            self._auto_filled_group_ids = {int(k) for k in phase_table}
        self._update_phase_colors()

    def set_group_enabled(self, enabled_table: dict[int, bool]) -> None:
        """Update existing group-inclusion rows without rebuilding the table."""
        current = self.group_enabled_table()
        current.update({int(k): bool(v) for k, v in enabled_table.items()})
        if not self._table_group_ids:
            fallback_names = {int(k): f"Group {int(k)}" for k in current}
            self.set_group_definitions(fallback_names, enabled_table=current)
            return
        self.set_group_definitions(
            {
                int(group_id): self._phase_table.item(row, 1).text()
                for row, group_id in enumerate(self._table_group_ids)
            },
            self.group_phase_table(),
            current,
        )

    def group_auto_filled_ids(self) -> set[int]:
        """Return group IDs whose phase cells were last auto-estimated."""
        return {int(group_id) for group_id in self._auto_filled_group_ids}

    def group_phase_state(self) -> dict[str, object]:
        """Return the current group-phase UI state for one dataset/run."""
        return {
            "group_enabled_table": self.group_enabled_table(),
            "group_phase_table": self.group_phase_table(),
            "group_auto_filled_ids": sorted(self.group_auto_filled_ids()),
        }

    def restore_group_phase_state(
        self, state: dict[str, object] | None, group_names: dict[int, str]
    ) -> None:
        """Restore per-run group-phase state into the table."""
        enabled_table: dict[int, bool] = {}
        phase_table: dict[int, float] = {}
        auto_filled_ids: set[int] = set()
        if isinstance(state, dict):
            raw_enabled = state.get("group_enabled_table", {})
            if isinstance(raw_enabled, dict):
                for key, value in raw_enabled.items():
                    try:
                        enabled_table[int(key)] = bool(value)
                    except (TypeError, ValueError):
                        continue
            raw_phases = state.get("group_phase_table", {})
            if isinstance(raw_phases, dict):
                for key, value in raw_phases.items():
                    try:
                        phase_table[int(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
            raw_auto = state.get("group_auto_filled_ids", [])
            if isinstance(raw_auto, (list, tuple, set)):
                for value in raw_auto:
                    try:
                        auto_filled_ids.add(int(value))
                    except (TypeError, ValueError):
                        continue
        self._auto_filled_group_ids = set(auto_filled_ids)
        self.set_group_definitions(group_names, phase_table, enabled_table)

    def clear_average_summary(self) -> None:
        """Clear the averaged-spectrum summary text."""
        self._average_summary_label.setText("")

    def set_average_summary(
        self, *, mean_error: float, peak_signal_to_noise: float, group_count: int
    ) -> None:
        """Show a short summary for an averaged grouped FFT spectrum."""
        if group_count <= 0:
            self.clear_average_summary()
            return
        self._average_summary_label.setText(
            f"Average of {group_count} groups. Mean error {mean_error:.4g}; peak S/N {peak_signal_to_noise:.3g}."
        )

    def get_state(self) -> dict:
        """Return a serialisable snapshot of the Fourier panel settings."""
        return {
            "window": self._current_filter_mode(),
            "filter_start_us": self._parse_float_text(self._filter_start_edit.text(), 0.0),
            "filter_time_constant_us": self._parse_float_text(
                self._filter_time_constant_edit.text(),
                1.5,
            ),
            "padding": self._padding_spin.value(),
            "phase_degrees": self._phase_value(),
            "t0_offset_us": self._t0_offset_value(),
            "display": self._current_display_mode(),
            "subtract_average_signal": self._subtract_average_signal_check.isChecked(),
            "auto_phase_method": self._auto_method_combo.currentText(),
            "use_phase_table": self._use_phase_table_check.isChecked(),
            "estimate_average_error": self._estimate_average_error_check.isChecked(),
            "group_enabled_table": self.group_enabled_table(),
            "group_phase_table": self.group_phase_table(),
            "group_auto_filled_ids": sorted(self.group_auto_filled_ids()),
            "pulse_compensation": self._pulse_comp_check.isChecked(),
            "pulse_half_width_us": self._parse_float_text(self._pulse_width_edit.text(), 0.0),
            "pulse_max_gain": self._parse_float_text(self._pulse_max_gain_edit.text(), 25.0),
            "baseline_mode": str(self._baseline_mode_combo.currentData() or "none"),
            "baseline_kappa": self._parse_float_text(self._baseline_kappa_edit.text(), 2.0),
            "exclude_enabled": self._exclude_enabled_check.isChecked(),
            "diamag_exclusion": self._diamag_mode() == "band",
            "diamag_half_width_mhz": self._parse_float_text(self._diamag_width_edit.text(), 0.3),
            "exclusion_ranges": [[c, w] for c, w in self.exclusion_ranges()],
            "remove_diamag": self._diamag_mode() == "subtract",
            "burg_order_min": self._burg_order_min_spin.value(),
            "burg_order_max": self._burg_order_max_spin.value(),
            "correlation_reference_field_gauss": (
                self._parse_float_text(self._correlation_field_edit.text(), 0.0)
                if self._correlation_field_edit.text().strip()
                else None
            ),
            "correlation_order": self._correlation_order_spin.value(),
            # Additive, namespaced moments sub-dict (no schema bump; W1).
            "moments": self._moments_widget.get_state(),
        }

    @property
    def moments_widget(self) -> SpectralMomentsWidget:
        """The spectral-moments control hosted in the advanced stack."""
        return self._moments_widget

    def restore_state(self, state: dict) -> None:
        """Restore Fourier panel settings from a saved dict."""
        self._moments_widget.restore_state(state.get("moments"))
        window_mode = self._coerce_filter_mode(state.get("window", "none"))
        self._filter_lorentzian_radio.setChecked(window_mode == "lorentzian")
        self._filter_gaussian_radio.setChecked(window_mode == "gaussian")
        self._filter_none_radio.setChecked(window_mode == "none")
        self._filter_start_edit.setText(
            self._format_float_text(self._parse_float_text(state.get("filter_start_us", 0.0), 0.0))
        )
        self._filter_time_constant_edit.setText(
            self._format_float_text(
                self._parse_float_text(state.get("filter_time_constant_us", 1.5), 1.5)
            )
        )
        self._padding_spin.setValue(state.get("padding", 1))
        try:
            phase_degrees = float(state.get("phase_degrees", 0.0))
        except (TypeError, ValueError):
            phase_degrees = 0.0
        self._phase_spin.setText(self._format_float_text(phase_degrees))
        try:
            t0_offset_us = float(state.get("t0_offset_us", 0.0))
        except (TypeError, ValueError):
            t0_offset_us = 0.0
        self._t0_offset_spin.setText(self._format_float_text(t0_offset_us))
        self._set_display_mode(state.get("display", "(Power)^1/2"))
        self._subtract_average_signal_check.setChecked(
            bool(state.get("subtract_average_signal", True))
        )
        idx = self._auto_method_combo.findText(state.get("auto_phase_method", "Peak"))
        if idx >= 0:
            self._auto_method_combo.setCurrentIndex(idx)
        self._use_phase_table_check.setChecked(bool(state.get("use_phase_table", False)))
        self._estimate_average_error_check.setChecked(
            bool(state.get("estimate_average_error", False))
        )
        enabled_table_raw = state.get("group_enabled_table", {})
        parsed_enabled: dict[int, bool] = {}
        if isinstance(enabled_table_raw, dict):
            for key, value in enabled_table_raw.items():
                try:
                    parsed_enabled[int(key)] = bool(value)
                except (TypeError, ValueError):
                    continue
        phase_table_raw = state.get("group_phase_table", {})
        parsed_phases: dict[int, float] = {}
        parsed_auto_filled: set[int] = set()
        if isinstance(phase_table_raw, dict) and phase_table_raw:
            for key, value in phase_table_raw.items():
                try:
                    parsed_phases[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        auto_filled_raw = state.get("group_auto_filled_ids", [])
        if isinstance(auto_filled_raw, (list, tuple, set)):
            for value in auto_filled_raw:
                try:
                    parsed_auto_filled.add(int(value))
                except (TypeError, ValueError):
                    continue
        if parsed_enabled or parsed_phases:
            group_ids = sorted(set(parsed_enabled) | set(parsed_phases))
            group_names = {group_id: f"Group {group_id}" for group_id in group_ids}
            if group_names:
                self._auto_filled_group_ids = set(parsed_auto_filled)
                self.set_group_definitions(group_names, parsed_phases, parsed_enabled)
        self._pulse_comp_check.setChecked(bool(state.get("pulse_compensation", False)))
        pulse_width = state.get("pulse_half_width_us", 0.0)
        self._pulse_width_edit.setText(
            self._format_float_text(self._parse_float_text(pulse_width, 0.0))
            if self._parse_float_text(pulse_width, 0.0) > 0.0
            else ""
        )
        self._pulse_max_gain_edit.setText(
            self._format_float_text(self._parse_float_text(state.get("pulse_max_gain", 25.0), 25.0))
        )
        baseline_idx = self._baseline_mode_combo.findData(str(state.get("baseline_mode", "none")))
        if baseline_idx >= 0:
            self._baseline_mode_combo.setCurrentIndex(baseline_idx)
        self._baseline_kappa_edit.setText(
            self._format_float_text(self._parse_float_text(state.get("baseline_kappa", 2.0), 2.0))
        )
        self._exclude_enabled_check.setChecked(bool(state.get("exclude_enabled", False)))
        self._diamag_width_edit.setText(
            self._format_float_text(
                self._parse_float_text(state.get("diamag_half_width_mhz", 0.3), 0.3)
            )
        )
        exclusion_ranges = state.get("exclusion_ranges", [])
        if isinstance(exclusion_ranges, (list, tuple)):
            self._set_exclusion_ranges(list(exclusion_ranges))
        # Map the two legacy booleans onto the single three-way control. Both
        # readable; fit-and-subtract wins when a legacy project set both, with
        # the band half-width preserved (and noted) rather than discarded.
        remove_diamag = bool(state.get("remove_diamag", False))
        diamag_exclusion = bool(state.get("diamag_exclusion", False))
        if remove_diamag:
            self._set_diamag_mode("subtract")
            if diamag_exclusion:
                self.set_fft_status(
                    "Loaded with both diamagnetic paths set; using Fit & subtract "
                    "(the band half-width is kept)."
                )
        elif diamag_exclusion:
            self._set_diamag_mode("band")
        else:
            self._set_diamag_mode("leave")
        try:
            self._burg_order_min_spin.setValue(int(state.get("burg_order_min", 2)))
            self._burg_order_max_spin.setValue(int(state.get("burg_order_max", 40)))
        except (TypeError, ValueError):
            pass
        corr_field = state.get("correlation_reference_field_gauss")
        if corr_field is None:
            self._correlation_field_edit.setText("")
        else:
            self._correlation_field_edit.setText(
                self._format_float_text(self._parse_float_text(corr_field, 0.0))
            )
        try:
            self._correlation_order_spin.setValue(
                int(state.get("correlation_order", DEFAULT_CORR_ORDER))
            )
        except (TypeError, ValueError):
            pass
        self._update_conditioning_enabled()
        self._update_exclusion_enabled()
        self._update_diamag_controls_enabled()
        self._normalize_phase_line_edits()
        self._update_filter_controls_enabled()
        self._update_phase_controls_enabled()
