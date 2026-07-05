"""Window-agnostic answer card for the fit wizards.

The card is the answer-first surface of the redesigned wizard: a plain-language
verdict headline and confidence sentence, a data plot with the selected fitted
curve overlaid (with a residuals toggle), a primary "Apply this fit" button, and
an alternatives strip that swaps the overlaid/applied candidate.

It is deliberately window- and dataset-agnostic. All prose comes from
``asymmetry.core.fitting.wizard_narrative`` (never re-worded here); plot data
arrives as plain arrays via :meth:`set_plot_data` (no ``MuonDataset`` import),
so a future multi-dataset wizard can reuse it. It emits :attr:`apply_requested`
with the selected :class:`CandidateAssessment` and never reaches back into a
window.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.fitting.fit_wizard import (
    CandidateAssessment,
    ConfidenceTier,
    FitWizardRecommendation,
    RecommendationVerdict,
)
from asymmetry.core.fitting.wizard_narrative import (
    _template_family_map,
    confidence_statement,
    template_display_name,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import (
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    build_primary_button_qss,
    build_segmented_button_qss,
    make_confidence_chip,
)

#: Cap on how many alternative candidates the strip offers.
_MAX_ALTERNATIVES = 3


def _strip_trailing_gloss(title: str) -> str:
    """Drop a trailing top-level parenthesised gloss from ``title``.

    ``template_display_name`` appends " (<plain name>)" to a title, and
    ``<plain name>`` can itself contain nested parens (e.g. "static nuclear
    fields (Kubo-Toyabe)"). This walks back from the end to find the matching
    top-level "(" for the final ")" and cuts from there, so only the last
    balanced group is removed. If the string does not end with a balanced
    parenthesised group, it is returned unchanged.
    """
    text = title.rstrip()
    if not text.endswith(")"):
        return title
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        char = text[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return text[:index].rstrip()
    return title


def _plain_verdict_headline(recommendation: FitWizardRecommendation) -> str:
    """Return the card's verdict headline (plain physics, never re-worded).

    Uses the same narrative primitives the trail uses so the two never disagree:
    the null verdict reads as a result, and a structured winner reads as its
    plain-physics display name. Falls back to the recommendation summary only
    when there is genuinely no winner and no null verdict.
    """
    if recommendation.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
        return "Your data look like a simple decay — no oscillation worth fitting."
    winner = recommendation.recommended_assessment
    if winner is None:
        return recommendation.summary or "No confident recommendation could be formed."
    family_map = _template_family_map(recommendation.family_reports)
    family_key = family_map.get(winner.template.key)
    return template_display_name(family_key, winner.template.title)


def _plain_confidence_line(recommendation: FitWizardRecommendation) -> str:
    """Return the card's confidence line (from the narrative module, honestly).

    Mirrors the narrative :func:`confidence_statement` verbatim for the High /
    Medium / null-verdict cases. The one deliberate suppression: when a genuine
    winner exists but the tier is the default ``NONE`` (an explicit-template or
    pre-confidence payload) and the verdict is not the null result, the bare
    "no confident recommendation" fallback would contradict a shown best-model
    card, so the line is left empty rather than buried-but-misleading.
    """
    if (
        recommendation.confidence is ConfidenceTier.NONE
        and recommendation.verdict is not RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
        and recommendation.recommended_assessment is not None
    ):
        return ""
    return confidence_statement(recommendation)


class WizardAnswerCard(QWidget):
    """Answer-first card: verdict + confidence + overlay plot + apply + alternatives."""

    #: Emitted with the currently-selected assessment when Apply is pressed.
    apply_requested = Signal(object)  # CandidateAssessment
    #: Emitted with the selected assessment key whenever the selection changes.
    selection_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recommendation: FitWizardRecommendation | None = None
        self._selected_key: str | None = None
        self._time: np.ndarray | None = None
        self._asymmetry: np.ndarray | None = None
        self._error: np.ndarray | None = None
        self._alt_buttons: dict[str, QPushButton] = {}
        self._confidence_chip: QLabel | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._card_frame = QFrame(self)
        self._card_frame.setObjectName(RESULT_BOX_OBJECT_NAME)
        self._card_frame.setStyleSheet(RESULT_BOX_NEUTRAL_STYLE)
        outer.addWidget(self._card_frame)

        layout = QVBoxLayout(self._card_frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._verdict_label = QLabel("", self._card_frame)
        self._verdict_label.setWordWrap(True)
        verdict_font = self._verdict_label.font()
        verdict_font.setPointSize(max(verdict_font.pointSize() + 3, 14))
        verdict_font.setBold(True)
        self._verdict_label.setFont(verdict_font)
        # The verdict label's own stretch factor (1) absorbs all extra width,
        # so the chip (inserted right after it, see _rebuild_confidence_chip)
        # sits flush against the wrapped headline with no floating gap.
        header_row.addWidget(self._verdict_label, 1)
        self._header_row = header_row
        layout.addLayout(header_row)

        self._confidence_label = QLabel("", self._card_frame)
        self._confidence_label.setWordWrap(True)
        layout.addWidget(self._confidence_label)

        # Plot + residuals toggle.
        self._plot_widget = self._build_plot_widget()
        layout.addWidget(self._plot_widget, 1)

        toggle_row = QHBoxLayout()
        self._residuals_toggle = QCheckBox("Show residuals", self._card_frame)
        self._residuals_toggle.toggled.connect(self._redraw_plot)
        toggle_row.addWidget(self._residuals_toggle)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # Alternatives strip.
        self._alternatives_row = QHBoxLayout()
        self._alternatives_label = QLabel("Alternatives:", self._card_frame)
        self._alternatives_label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        self._alternatives_row.addWidget(self._alternatives_label)
        self._alternatives_row.addStretch()
        self._alternatives_container = QWidget(self._card_frame)
        self._alternatives_container.setLayout(self._alternatives_row)
        self._alternatives_container.setVisible(False)
        layout.addWidget(self._alternatives_container)

        # Apply.
        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply this fit", self._card_frame)
        self._apply_btn.setStyleSheet(build_primary_button_qss())
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        apply_row.addWidget(self._apply_btn)
        apply_row.addStretch()
        layout.addLayout(apply_row)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_plot_data(
        self,
        time: np.ndarray | None,
        asymmetry: np.ndarray | None,
        error: np.ndarray | None,
    ) -> None:
        """Provide the raw spectrum arrays the overlay is drawn against."""
        self._time = None if time is None else np.asarray(time, dtype=float)
        self._asymmetry = None if asymmetry is None else np.asarray(asymmetry, dtype=float)
        self._error = None if error is None else np.asarray(error, dtype=float)
        self._redraw_plot()

    def set_recommendation(self, recommendation: FitWizardRecommendation | None) -> None:
        """Populate the card from a recommendation; select the recommended key."""
        self._recommendation = recommendation
        self._sync_card_style()
        self._rebuild_confidence_chip()
        if recommendation is None:
            self._selected_key = None
            self._verdict_label.setText("")
            self._confidence_label.setText("")
            self._clear_alternatives()
            self._redraw_plot()
            return
        self._selected_key = recommendation.recommended_key
        if self._selected_key is None and recommendation.assessments:
            self._selected_key = recommendation.assessments[0].template.key
        self._verdict_label.setText(_plain_verdict_headline(recommendation))
        confidence_line = _plain_confidence_line(recommendation)
        self._confidence_label.setText(confidence_line)
        self._confidence_label.setVisible(bool(confidence_line))
        self._rebuild_alternatives()
        self._redraw_plot()

    # ── Card chrome (frame tint + confidence chip) ─────────────────────────

    def _is_high_confidence_winner(self) -> bool:
        """True when the recommendation is a real, high-confidence winner."""
        rec = self._recommendation
        if rec is None:
            return False
        return (
            rec.confidence is ConfidenceTier.HIGH
            and rec.verdict is not RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE
            and rec.recommended_assessment is not None
        )

    def _sync_card_style(self) -> None:
        """Tint the card frame green for a high-confidence winner, neutral otherwise."""
        if self._is_high_confidence_winner():
            style = RESULT_BOX_SUCCESS_STYLE
        else:
            style = RESULT_BOX_NEUTRAL_STYLE
        self._card_frame.setStyleSheet(style)

    def _confidence_chip_spec(self) -> tuple[str, str] | None:
        """Return ``(text, tier)`` for the header chip, or ``None`` to hide it."""
        rec = self._recommendation
        if rec is None:
            return None
        if rec.verdict is RecommendationVerdict.NO_SIGNIFICANT_STRUCTURE:
            return ("No structure to fit", "none")
        if rec.confidence is ConfidenceTier.HIGH:
            return ("High confidence", "high")
        if rec.confidence is ConfidenceTier.MEDIUM:
            return ("Medium confidence", "medium")
        return None

    def _rebuild_confidence_chip(self) -> None:
        """Rebuild the confidence chip (colours are baked in at construction)."""
        if self._confidence_chip is not None:
            self._header_row.removeWidget(self._confidence_chip)
            self._confidence_chip.setParent(None)
            self._confidence_chip.deleteLater()
            self._confidence_chip = None
        spec = self._confidence_chip_spec()
        if spec is None:
            return
        text, tier = spec
        chip = make_confidence_chip(text, tier)
        chip.setParent(self._card_frame)
        chip.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Appended after the verdict label, which owns the row's only stretch
        # factor — the chip sits flush against the headline, top-aligned.
        self._header_row.addWidget(chip, 0, Qt.AlignmentFlag.AlignTop)
        self._confidence_chip = chip

    def selected_assessment(self) -> CandidateAssessment | None:
        if self._recommendation is None:
            return None
        return (
            self._recommendation.assessment_for_key(self._selected_key)
            or self._recommendation.recommended_assessment
        )

    def selected_key(self) -> str | None:
        return self._selected_key

    def set_selected_key(self, key: str | None) -> None:
        if key == self._selected_key:
            return
        self._selected_key = key
        self._sync_alternative_styles()
        self._redraw_plot()
        if isinstance(key, str):
            self.selection_changed.emit(key)

    # ── Alternatives strip ─────────────────────────────────────────────────

    def _alternative_keys(self) -> list[str]:
        """Ordered alternative keys: comparable_keys first, then next-best.

        ``comparable_keys`` (similar-quality peers the core already surfaced)
        come first, then successful, non-disqualified, non-null candidates in
        ranked order — excluding the recommended key and duplicates. Capped at
        ``_MAX_ALTERNATIVES``.
        """
        rec = self._recommendation
        if rec is None:
            return []
        recommended = rec.recommended_key
        ordered: list[str] = []
        for key in rec.comparable_keys:
            if key and key != recommended and key not in ordered:
                ordered.append(key)
        for assessment in rec.sorted_assessments():
            key = assessment.template.key
            if key == recommended or key in ordered:
                continue
            if assessment.is_null_baseline or not assessment.is_successful:
                continue
            if assessment.is_disqualified:
                continue
            ordered.append(key)
        return ordered[:_MAX_ALTERNATIVES]

    def _alternative_title(self, assessment: CandidateAssessment) -> str:
        """The plain-physics glossed template title for an alternative."""
        rec = self._recommendation
        family_map = _template_family_map(rec.family_reports) if rec is not None else {}
        return template_display_name(
            family_map.get(assessment.template.key), assessment.template.title
        )

    def _metric_delta(self, assessment: CandidateAssessment) -> float | None:
        """Return ``assessment`` minus the recommended candidate's metric value.

        ``None`` when there is no recommended assessment, or either value is
        non-finite (so the badge is simply omitted rather than showing NaN/inf).
        """
        rec = self._recommendation
        recommended = rec.recommended_assessment if rec is not None else None
        if rec is None or recommended is None:
            return None
        value = assessment.metric_value(rec.metric)
        reference = recommended.metric_value(rec.metric)
        if not math.isfinite(value) or not math.isfinite(reference):
            return None
        return value - reference

    def _alternative_label(self, assessment: CandidateAssessment) -> str:
        """Button text: the plain title (gloss stripped) plus a metric-delta badge.

        The tooltip (:meth:`_alternative_tooltip`) keeps the full glossed name;
        only the button text is shortened, since the parenthesised family gloss
        makes the chip far too wide.
        """
        title = _strip_trailing_gloss(self._alternative_title(assessment))
        delta = self._metric_delta(assessment)
        if delta is None:
            return title
        return f"{title}  ·  {delta:+.1f}"

    def _alternative_tooltip(self, assessment: CandidateAssessment) -> str:
        """Full tooltip: display name, simpler-model note, and badge explanation."""
        rec = self._recommendation
        recommended = rec.recommended_assessment if rec is not None else None
        lines = [self._alternative_title(assessment)]
        if recommended is not None and assessment.parameter_count < recommended.parameter_count:
            lines.append("Similar quality with a simpler model (fewer parameters).")
        delta = self._metric_delta(assessment)
        if delta is not None and rec is not None:
            lines.append(
                f"{rec.metric.value} difference vs the recommendation: "
                f"{delta:+.2f} (lower is better)."
            )
        return "\n".join(lines)

    def _rebuild_alternatives(self) -> None:
        self._clear_alternatives()
        rec = self._recommendation
        if rec is None:
            return
        keys = self._alternative_keys()
        if not keys:
            return
        segmented_qss = build_segmented_button_qss(padding_h=8)
        for key in keys:
            assessment = rec.assessment_for_key(key)
            if assessment is None:
                continue
            button = QPushButton(self._alternative_label(assessment), self._alternatives_container)
            button.setCheckable(True)
            button.setStyleSheet(segmented_qss)
            button.setToolTip(self._alternative_tooltip(assessment))
            button.clicked.connect(lambda _checked=False, k=key: self.set_selected_key(k))
            # Insert before the trailing stretch.
            self._alternatives_row.insertWidget(self._alternatives_row.count() - 1, button)
            self._alt_buttons[key] = button
        self._alternatives_container.setVisible(bool(self._alt_buttons))
        self._sync_alternative_styles()

    def _clear_alternatives(self) -> None:
        for button in self._alt_buttons.values():
            self._alternatives_row.removeWidget(button)
            button.setParent(None)
            button.deleteLater()
        self._alt_buttons.clear()
        self._alternatives_container.setVisible(False)

    def _sync_alternative_styles(self) -> None:
        """Make the selected candidate visually explicit across the buttons."""
        for key, button in self._alt_buttons.items():
            button.setChecked(key == self._selected_key)

    # ── Apply ──────────────────────────────────────────────────────────────

    def _on_apply_clicked(self) -> None:
        assessment = self.selected_assessment()
        if assessment is not None:
            self.apply_requested.emit(assessment)

    # ── Plot ───────────────────────────────────────────────────────────────

    def _build_plot_widget(self) -> QWidget:
        container = QWidget(self)
        inner = QVBoxLayout(container)
        inner.setContentsMargins(0, 0, 0, 0)
        try:
            from asymmetry.gui.widgets.mpl_canvas import create_canvas

            figure, canvas = create_canvas(layout="tight")
            container._figure = figure  # type: ignore[attr-defined]
            container._canvas = canvas  # type: ignore[attr-defined]
            inner.addWidget(canvas)
        except ImportError:
            container._figure = None  # type: ignore[attr-defined]
            container._canvas = None  # type: ignore[attr-defined]
            fallback = QLabel("matplotlib not available — plot preview disabled", container)
            fallback.setWordWrap(True)
            inner.addWidget(fallback)
        return container

    def _redraw_plot(self) -> None:
        figure = getattr(self._plot_widget, "_figure", None)
        canvas = getattr(self._plot_widget, "_canvas", None)
        if figure is None or canvas is None:
            return
        figure.clear()
        if self._time is None or self._asymmetry is None:
            canvas.draw_idle()
            return

        assessment = self.selected_assessment()
        show_residuals = self._residuals_toggle.isChecked()

        if show_residuals and assessment is not None:
            ax_fit = figure.add_subplot(2, 1, 1)
            ax_res = figure.add_subplot(2, 1, 2)
        else:
            ax_fit = figure.add_subplot(1, 1, 1)
            ax_res = None

        yerr = self._error if self._error is not None else None
        ax_fit.errorbar(
            self._time,
            self._asymmetry,
            yerr=yerr,
            fmt=".",
            markersize=3,
            color=tokens.PLOT_DATA,
            label="Data",
        )
        if assessment is not None:
            ax_fit.plot(
                assessment.fitted_time,
                assessment.fitted_curve,
                color=tokens.PLOT_FIT,
                label="Fit",
            )
        ax_fit.set_xlabel("Time (µs)")
        ax_fit.set_ylabel("Asymmetry")
        # Only show the axes title when the user picked an alternative (it then
        # clarifies what is plotted); when the recommendation itself is
        # selected, the card headline right above already says as much, and
        # repeating it here is pure duplication.
        if (
            assessment is not None
            and self._recommendation is not None
            and assessment.template.key != self._recommendation.recommended_key
        ):
            ax_fit.set_title(assessment.template.title)
        ax_fit.legend(loc="best")

        if ax_res is not None and assessment is not None:
            residuals = assessment.fit_result.residuals
            if residuals is not None and getattr(residuals, "size", 0):
                res_time = np.asarray(self._time, dtype=float)[: residuals.size]
                ax_res.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=1.0)
                ax_res.plot(res_time, residuals, color=tokens.TRACE_GREEN)
            ax_res.set_xlabel("Time (µs)")
            ax_res.set_ylabel("Residual")
            ax_res.set_title("Residuals")

        canvas.draw_idle()
