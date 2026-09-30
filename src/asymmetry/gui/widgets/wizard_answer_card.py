"""Window-agnostic answer card for the fit wizards.

The card is the answer-first surface of the wizard: a plain-language verdict
headline and confidence sentence, a data plot with the selected fitted curve
overlaid (with a residuals toggle), and a primary "Apply this fit" button.

It is deliberately window- and dataset-agnostic. All prose comes from
``asymmetry.core.fitting.wizard_narrative`` (never re-worded here); plot data
arrives as plain arrays via :meth:`set_plot_data` (no ``MuonDataset`` import).
The card owns no selection: it reads the selected key through the callable its
owner hands it (the Compare panel's candidate A, see
``docs/plans/fit-wizard-compare.md`` D1), and is told to :meth:`redraw` when
that moves. It emits :attr:`apply_requested` with the selected key and never
reaches back into a window.
"""

from __future__ import annotations

from collections.abc import Callable

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
from asymmetry.gui.styles.metrics import row_height
from asymmetry.gui.styles.widgets import (
    RESULT_BOX_NEUTRAL_STYLE,
    RESULT_BOX_OBJECT_NAME,
    RESULT_BOX_SUCCESS_STYLE,
    build_primary_button_qss,
    make_confidence_chip,
)
from asymmetry.gui.utils.plot_decimation import decimate_for_preview

#: The overlay plot's height, in table rows.
_PLOT_ROWS = 11

#: Cap on points drawn in the answer card's data errorbar. Same disease as
#: the wizard fingerprint plot / grouping preview (see plot_decimation):
#: matplotlib ``errorbar`` over a full high-resolution run stalls the GUI
#: thread computing the error-bar collection's extents. Display-only — the
#: stored arrays stay full-resolution because the residuals panel needs the
#: real axis (rebinned by the recommendation's own factor) to pair with
#: fit-length residuals.
_MAX_ANSWER_PLOT_POINTS = 2000


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
    """Answer-first card: verdict + confidence + overlay plot of the selection + apply.

    ``selected_key`` returns the key of the candidate to draw and apply.
    """

    #: Emitted with the selected key when Apply is pressed.
    apply_requested = Signal(str)
    #: Emitted with an assessment key the card was asked to draw that carries no
    #: dense curves. The owner is expected to build them (off the GUI thread —
    #: see :func:`asymmetry.core.fitting.fit_wizard.assessment_with_curves`) and
    #: hand the card the updated recommendation via :meth:`refresh_curves`.
    curves_required = Signal(str)

    def __init__(
        self, selected_key: Callable[[], str | None], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._recommendation: FitWizardRecommendation | None = None
        self._selected_key = selected_key
        self._time: np.ndarray | None = None
        self._asymmetry: np.ndarray | None = None
        self._error: np.ndarray | None = None
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
        # A fixed height keeps the verdict compact on a scrolling result page,
        # so the Compare section below it starts within the first screen.
        self._plot_widget.setFixedHeight(row_height() * _PLOT_ROWS)
        layout.addWidget(self._plot_widget)

        toggle_row = QHBoxLayout()
        self._residuals_toggle = QCheckBox("Show residuals", self._card_frame)
        self._residuals_toggle.toggled.connect(self.redraw)
        toggle_row.addWidget(self._residuals_toggle)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # Apply.
        apply_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply this fit", self._card_frame)
        self._apply_btn.setStyleSheet(build_primary_button_qss())
        self._apply_btn.clicked.connect(lambda: self.apply_requested.emit(self._selected_key()))
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
        self.redraw()

    def set_recommendation(self, recommendation: FitWizardRecommendation | None) -> None:
        """Populate the card from a recommendation, drawing the owner's selection."""
        self._recommendation = recommendation
        self._sync_card_style()
        self._rebuild_confidence_chip()
        if recommendation is None:
            self._verdict_label.setText("")
            self._confidence_label.setText("")
            self.redraw()
            return
        self._verdict_label.setText(_plain_verdict_headline(recommendation))
        confidence_line = _plain_confidence_line(recommendation)
        self._confidence_label.setText(confidence_line)
        self._confidence_label.setVisible(bool(confidence_line))
        self.redraw()

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
        """The selected row of the recommendation; ``None`` while the card holds none."""
        if self._recommendation is None:
            return None
        return self._recommendation.assessment_for_key(self._selected_key())

    def refresh_curves(self, recommendation: FitWizardRecommendation) -> None:
        """Re-point the card at ``recommendation`` and redraw, keeping the selection.

        The answer to :attr:`curves_required`: ``recommendation`` is the same
        ranking the card already holds with one row's dense curves filled in, so
        the headline and the confidence line are unchanged by construction and
        only the plot is rebuilt. Passing a
        *differently ranked* recommendation here is a caller bug — use
        :meth:`set_recommendation` for that.
        """
        self._recommendation = recommendation
        self.redraw()

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

    def set_apply_enabled(self, enabled: bool) -> None:
        self._apply_btn.setEnabled(enabled)

    def redraw(self) -> None:
        """Redraw the plot for the owner's current selection."""
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

        # Decimate the drawn points only (never the stored arrays — the
        # residuals panel below needs full-resolution alignment).
        plot_time, plot_asymmetry, plot_error = decimate_for_preview(
            self._time,
            self._asymmetry,
            self._error if self._error is not None else np.zeros_like(self._time),
            _MAX_ANSWER_PLOT_POINTS,
        )
        yerr = plot_error if self._error is not None else None
        ax_fit.errorbar(
            plot_time,
            plot_asymmetry,
            yerr=yerr,
            fmt=".",
            markersize=3,
            color=tokens.PLOT_DATA,
            label="Data",
        )
        if assessment is not None:
            if assessment.fitted_time.size:
                ax_fit.plot(
                    assessment.fitted_time,
                    assessment.fitted_curve,
                    color=tokens.PLOT_FIT,
                    label="Fit",
                )
            else:
                # The dense-curve contract on ``CandidateAssessment``: a build
                # materialises curves only for the rows it exposes as its answer,
                # and this card can be pointed at any of the two-to-three dozen
                # candidates it assessed. Ask the owner for them — building one is
                # far too expensive for a draw — and draw the data alone until
                # they arrive via :meth:`refresh_curves`. This is the *only* place
                # the request is made, so every path that changes what is drawn
                # (a new A, a re-rank, the residuals toggle) is covered by it.
                self.curves_required.emit(assessment.template.key)
        ax_fit.set_xlabel("Time (µs)")
        ax_fit.set_ylabel("Asymmetry")
        # Title the axes only for a selection other than the recommendation; the
        # headline right above already names the recommendation.
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
                # The wizard fits a value-rebinned copy of a large record, so a
                # residual here is one MERGED bin, not one raw bin: pairing it
                # with the leading slice of the full time axis would draw the
                # whole residual series squeezed into the first 1/factor of the
                # record. Rebin the axis by the factor the recommendation
                # recorded, then slice as before (a fit range shorter than the
                # record still trims from the front).
                res_time = np.asarray(self._time, dtype=float)
                factor = self._recommendation.rebin_factor if self._recommendation else 1
                if factor > 1 and res_time.size >= factor:
                    # Mean of each merged bin — the same axis ``rebin`` would
                    # return, computed directly rather than through its
                    # value/error path.
                    n_bins = res_time.size // factor
                    res_time = res_time[: n_bins * factor].reshape(n_bins, factor).mean(axis=1)
                res_time = res_time[: residuals.size]
                ax_res.axhline(0.0, color=tokens.PLOT_ZERO_LINE, linewidth=1.0)
                ax_res.plot(res_time, residuals, color=tokens.TRACE_GREEN)
            else:
                # A recommendation restored from a project file carries the
                # residual *statistics* but not the series itself (it is the
                # bulkiest array in the payload and cannot be decimated without
                # misaligning it from its time axis — see
                # docs/reference/project_files.rst).
                ax_res.text(
                    0.5,
                    0.5,
                    "Residuals are not stored with a cached result —\n"
                    "re-run the wizard to see them.",
                    transform=ax_res.transAxes,
                    ha="center",
                    va="center",
                    color=tokens.TEXT_MUTED,
                )
            ax_res.set_xlabel("Time (µs)")
            ax_res.set_ylabel("Residual")
            ax_res.set_title("Residuals")

        canvas.draw_idle()
