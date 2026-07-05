"""Shared widget-styling helpers for the BENCH design language.

Import these in any panel or window that needs BENCH-consistent table headers,
formula code-boxes, result-group success states, button QSS, or HTML snippets.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import SIZE_NUMERIC, header_font, section_label_font

# ── Flat section header ───────────────────────────────────────────────────────

#: objectName used to scope flat section-header QSS without touching other labels.
SECTION_HEADER_OBJECT_NAME = "benchSectionHeader"


def make_section_header(text: str) -> QLabel:
    """Return a flat uppercase BENCH section-header label.

    The inspector tabs (Fit, Parameters, Multi-group fit) introduce each block
    with one of these instead of wrapping it in a QGroupBox — the design
    handoff's "BSection": 9.5pt bold, +0.4px tracking, muted grey, uppercased.

    Qt QSS ``text-transform`` is unreliable across platforms, so the text is
    uppercased here in Python.
    """
    label = QLabel(text.upper())
    label.setObjectName(SECTION_HEADER_OBJECT_NAME)
    label.setFont(section_label_font())
    label.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
    return label


def make_provenance_label() -> QLabel:
    """Return a hidden, word-wrapped, muted-text provenance line.

    Shared by every plot that summarises "which members contributed / were
    dropped" beneath its canvas (the ALC scan and the parameter trend plot) —
    hidden until the caller has something to say via ``setText`` +
    ``setVisible(True)`` (or the ALC panel's ``set_provenance`` convention).
    """
    label = QLabel("")
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
    label.hide()
    return label


def make_section(title: str) -> tuple[QWidget, QVBoxLayout]:
    """Return a ``(section widget, content layout)`` pair for a flat dock section.

    The widget holds a :func:`make_section_header` followed by an empty,
    margin-free content area. Add content to the returned layout (a widget, or a
    sub-layout via ``addLayout`` for form/row content) and add the widget to the
    parent layout. Because the header lives inside the widget, hiding the widget
    hides its header too. Replaces the repeated
    ``header + QWidget + zero-margin layout`` idiom across the inspector tabs.
    """
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(make_section_header(title))
    return container, layout


# ── Result box ────────────────────────────────────────────────────────────────

#: objectName on the result-box QFrame so the tint rules below never cascade
#: onto the child result label.
RESULT_BOX_OBJECT_NAME = "benchResultBox"

#: Neutral (no fit yet / cleared) result box — plain surface with a 1px border.
RESULT_BOX_NEUTRAL_STYLE = (
    f"QFrame#{RESULT_BOX_OBJECT_NAME} {{"
    f" background-color: {tokens.SURFACE};"
    f" border: 1px solid {tokens.BORDER};"
    " border-radius: 4px;"
    "}"
)

#: Converged result box — green success tint, matching the handoff's converged box.
RESULT_BOX_SUCCESS_STYLE = (
    f"QFrame#{RESULT_BOX_OBJECT_NAME} {{"
    f" background-color: {tokens.SUCCESS_BG};"
    f" border: 1px solid {tokens.SUCCESS_BORDER};"
    " border-radius: 4px;"
    "}"
)

# ── Layout helpers ──────────────────────────────────────────────────────────


def clear_layout(layout) -> None:
    """Remove every widget from *layout* and schedule it for deletion.

    Shared by panels that rebuild a dynamic section in place (parameter group
    tabs, fit ranges, the per-projection alpha table). Takes each item off the
    layout and calls ``deleteLater`` on its widget; sub-layout items (no widget)
    are simply detached, matching the prior per-call-site loops.
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


# ── Table style ───────────────────────────────────────────────────────────────


def apply_param_table_style(table: QTableWidget) -> None:
    """Apply BENCH styling to any parameter or data table.

    Sets a DemiBold section header, 11pt mono cell font, hides the row-number
    column, and tints alternating rows with surfaceAlt.
    """
    table.horizontalHeader().setFont(header_font())
    table.verticalHeader().setVisible(False)
    table.setFont(mono_font(SIZE_NUMERIC))
    table.setAlternatingRowColors(True)
    table.setStyleSheet(f"QTableWidget {{ alternate-background-color: {tokens.SURFACE_ALT}; }}")


# ── Formula label ─────────────────────────────────────────────────────────────


#: Zero-width space — an allowed line-break point (after a top-level operator).
_FORMULA_BREAK = "​"
#: Word joiner — forbids a line break at this position.
_FORMULA_JOIN = "⁠"


def insert_formula_break_points(formula: str) -> str:
    """Return ``formula`` with line-break opportunities only at top-level operators.

    A fit-function expression should wrap where ``*``/``+``/``-`` *combine
    different functions* (e.g. ``cos(...) * exp(...) + A_bg``) but never inside a
    function's arguments (``cos(2*pi*f*t + phi)``, ``exp(-(Delta^2/nu^2)*...)``) or
    between a coefficient and its function (``A_1*cos(...)``). Those joining
    operators are exactly the ones at parenthesis depth 0 *with* surrounding
    spaces.

    Qt's line breaker would otherwise also break at ``/``, ``-``, ``)`` etc. inside
    a tightly-packed argument, so a WORD JOINER is inserted between *every* adjacent
    pair to forbid breaks, and a ZERO-WIDTH SPACE replaces it only at the chosen
    top-level operators. The result wraps only between functions; a single function
    wider than the box scrolls horizontally instead of breaking.
    """

    def _is_token_char(ch: str) -> bool:
        # Letters, digits and underscore form a name (exp, Lambda, A_1, 2*pi);
        # Qt never breaks inside such a run, so no joiner is needed between two
        # of them — which keeps whole names searchable in the visible text.
        return ch.isalnum() or ch == "_"

    out: list[str] = []
    depth = 0
    length = len(formula)
    for index, char in enumerate(formula):
        # Spaces become non-breaking; the only place a line may break is the
        # explicit ZWSP inserted after a top-level operator below.
        out.append("\u00a0" if char == " " else char)
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if index == length - 1:
            continue
        nxt = formula[index + 1]
        is_top_level_join = (
            depth == 0
            and index > 0  # guard: index-0 would read formula[-1] (the last char)
            and char in "*+-"
            and formula[index - 1] == " "
            and nxt == " "
        )
        if is_top_level_join:
            out.append(_FORMULA_BREAK)
        elif not (_is_token_char(char) and _is_token_char(nxt)):
            # A boundary touching an operator/parenthesis is where Qt would
            # otherwise break (after '/', '-', ')', ...); forbid it.
            out.append(_FORMULA_JOIN)
    return "".join(out)


def configure_formula_label(label: QLabel) -> None:
    """Configure the inner QLabel of a :class:`FormulaBox` (or a bare code-box).

    The label is transparent — the code-box border/background lives on the
    surrounding scroll area, so the domain-mismatch warning (which sets the
    label's stylesheet to recolour the text) cannot clobber the box chrome. Word
    wrap is on, but :func:`insert_formula_break_points` restricts where it may
    actually break.
    """
    label.setFont(mono_font(SIZE_NUMERIC))
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setContentsMargins(6, 4, 6, 4)
    label.setStyleSheet("QLabel { background: transparent; }")


class FormulaBox(QScrollArea):
    """A code-box for a fit-function expression.

    Wraps the expression only at the top-level operators joining its terms (see
    :func:`insert_formula_break_points`), keeps each function and its arguments
    intact, scrolls horizontally when a single function is wider than the box,
    and sizes its height to the wrapped content (the inspector dock scrolls
    vertically for a very tall expression). Set the text via :meth:`set_formula`;
    ``self.label`` stays a plain QLabel so ``text()``/``toolTip()`` and the
    domain-mismatch warning keep working.
    """

    def __init__(self, parent: QLabel | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        configure_formula_label(self.label)
        self.label._formula_box = self  # back-ref so text setters can re-measure
        self.setObjectName("formulaBox")
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            f"QScrollArea#formulaBox {{ background-color: {tokens.SURFACE_ALT};"
            f" border: 1px solid {tokens.BORDER}; border-radius: 3px; }}"
            " QScrollArea#formulaBox > QWidget > QWidget { background: transparent; }"
        )
        self.refresh_height()

    def set_formula(self, formula: str) -> None:
        """Display ``formula`` (break-marked) and keep the raw text in the tooltip."""
        raw = str(formula)
        self.label.setText(insert_formula_break_points(raw))
        self.label.setToolTip(raw)
        self.refresh_height()

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt override
        super().resizeEvent(event)
        self.refresh_height()

    def refresh_height(self) -> None:
        """Resize the box to fit the wrapped content at the current width."""
        viewport_width = max(1, self.viewport().width())
        segment_width = self.label.minimumSizeHint().width()  # widest unbreakable run
        content_width = max(viewport_width, segment_width)
        if self.label.hasHeightForWidth():
            text_height = self.label.heightForWidth(content_width)
        else:
            text_height = self.label.sizeHint().height()
        scrollbar = (
            self.horizontalScrollBar().sizeHint().height() if segment_width > viewport_width else 0
        )
        line = self.label.fontMetrics().lineSpacing()
        self.setFixedHeight(max(text_height, line) + scrollbar + 2 * self.frameWidth() + 6)


def make_formula_box() -> tuple[FormulaBox, QLabel]:
    """Return a ``(box, label)`` pair for displaying a fit-function formula.

    Add ``box`` to the layout; keep ``label`` for the formula text-setter and the
    domain-mismatch warning (both go through the label, which re-measures the box
    via its back-reference).
    """
    box = FormulaBox()
    return box, box.label


# ── Table header font ─────────────────────────────────────────────────────────


def apply_param_table_header_font(table: QTableWidget) -> None:
    """Apply the BENCH column-header font to a table's horizontal header.

    Separated from apply_param_table_style so callers that handle row colours
    themselves can still pick up the canonical header font.
    """
    table.horizontalHeader().setFont(header_font())


def widest_button_width(button: QPushButton, *labels: str) -> int:
    """Return the widest ``sizeHint().width()`` ``button`` would need for ``labels``.

    Per-row action buttons in a table cell (e.g. "Model Fit" / "Model Fit*" /
    "Global fit (N groups)…") can be relabeled after the column width was last
    set — a plain ``ResizeToContents`` column only measures whichever label was
    current *at layout time*, and a later, wider relabel then clips. Callers
    that own a real, already-styled button should probe it with every label the
    row can ever show (including states set well after construction) and use
    the maximum as the column's minimum/fixed width, so the widest state always
    fits regardless of font scaling or style padding.

    This mutates and restores ``button``'s text; call it with the row's real
    button (not a throwaway probe) so the measured ``sizeHint`` reflects the
    same style/font the button will actually render with.
    """
    original_text = button.text()
    try:
        width = 0
        for label in labels:
            button.setText(label)
            width = max(width, button.sizeHint().width())
        return width
    finally:
        button.setText(original_text)


# ── Segmented toolbar / toggle-button QSS ────────────────────────────────────


def build_segmented_button_qss(
    *,
    min_width: int | None = None,
    padding_h: int = 10,
) -> str:
    """Return a per-widget QSS for a checkable QPushButton segmented-control cell.

    Must be applied via widget.setStyleSheet(), NOT the global application
    stylesheet.  Per-widget application is required because Qt's global
    stylesheet interacts with UIManager.build_stylesheet's generic QPushButton
    rules in ways that cause pseudo-state padding to revert unexpectedly.

    Args:
        min_width:  Optional minimum width in px (e.g. 28 for compact view-mode
                    buttons showing a single digit).
        padding_h:  Horizontal padding in px.  Use 10 (default) for domain/label
                    buttons, 6 for compact numbered buttons.
    """
    width_rule = f" min-width: {min_width}px;" if min_width is not None else ""
    return (
        # font-weight: 600 is in the base rule so the button is always sized for
        # bold text — the checked state only changes colours, never triggers reflow.
        f"QPushButton {{{width_rule} padding: 2px {padding_h}px; font-weight: 600;"
        f" background-color: {tokens.SURFACE}; color: {tokens.TEXT};"
        f" border: 1px solid {tokens.BORDER}; border-radius: 4px; }}"
        f" QPushButton:checked {{ padding: 2px {padding_h}px; font-weight: 600;"
        f" background-color: {tokens.ACCENT_SOFT}; color: {tokens.ACCENT};"
        f" border-color: {tokens.ACCENT}; }}"
        # A per-widget stylesheet replaces the global one entirely, so the
        # disabled state must be styled here too — without it, data-gated
        # segments (Individual groups / MaxEnt with no capable run) look
        # identical to clickable ones. Recede into the toolbar surface.
        f" QPushButton:disabled {{ padding: 2px {padding_h}px; font-weight: 600;"
        f" background-color: {tokens.SURFACE_ALT}; color: {tokens.TEXT_DIM};"
        f" border-color: {tokens.BORDER}; }}"
    )


def build_segmented_container_qss() -> str:
    """QSS for the QFrame that wraps a *joined* segmented control.

    The design-handoff segmented control draws ONE border around the whole
    group with 1px internal dividers — not separate fully-bordered buttons.
    The frame owns the outer border and radius; the cells inside are
    borderless (see :func:`build_segmented_cell_qss`).
    """
    return (
        f"QFrame {{ border: 1px solid {tokens.BORDER}; border-radius: 4px;"
        f" background-color: {tokens.SURFACE}; }}"
    )


def build_segmented_cell_qss(*, first: bool, last: bool, padding_h: int = 10) -> str:
    """QSS for one cell of a joined segmented control.

    Cells are borderless except for the 1px divider on their right edge;
    only the outer cells round their corners (3px, sitting just inside the
    container's 4px radius). The checked cell fills accent-soft with accent
    text — matching the design handoff, the divider stays neutral.
    """
    radius = (
        f" border-top-left-radius: {3 if first else 0}px;"
        f" border-bottom-left-radius: {3 if first else 0}px;"
        f" border-top-right-radius: {3 if last else 0}px;"
        f" border-bottom-right-radius: {3 if last else 0}px;"
    )
    divider = "" if last else f" border-right: 1px solid {tokens.BORDER};"
    base = f"padding: 2px {padding_h}px; font-weight: 600; border: none;{divider}{radius}"
    return (
        f"QPushButton {{ {base} background-color: transparent; color: {tokens.TEXT}; }}"
        f" QPushButton:checked {{ {base} background-color: {tokens.ACCENT_SOFT};"
        f" color: {tokens.ACCENT}; }}"
        f" QPushButton:disabled {{ {base} background-color: {tokens.SURFACE_ALT};"
        f" color: {tokens.TEXT_DIM}; }}"
    )


def build_primary_button_qss() -> str:
    """Accent-tinted treatment for a panel's primary action (Run Fit, …).

    The design handoff renders the primary action in the 'active' style —
    accent-soft fill, accent border and text — so the eye lands on the one
    button that advances the workflow.
    """
    return (
        f"QPushButton {{ padding: 3px 12px; font-weight: 600;"
        f" background-color: {tokens.ACCENT_SOFT}; color: {tokens.ACCENT};"
        f" border: 1px solid {tokens.ACCENT}; border-radius: 4px; }}"
        f" QPushButton:hover {{ background-color: {tokens.ACCENT_SOFT2}; }}"
        f" QPushButton:disabled {{ background-color: {tokens.SURFACE_ALT};"
        f" color: {tokens.TEXT_DIM}; border-color: {tokens.BORDER}; }}"
    )


def build_nav_button_qss() -> str:
    """Return a per-widget QSS for plot-panel pan/zoom/auto navigation buttons.

    Must be applied via widget.setStyleSheet() — see build_segmented_button_qss
    for the rationale.
    """
    return (
        f"QPushButton {{ min-width: 60px; padding: 2px 8px; font-weight: 600;"
        f" border: 1px solid {tokens.BORDER_STRONG}; border-radius: 4px; }}"
        f" QPushButton:checked {{ min-width: 60px; padding: 2px 8px; font-weight: 600;"
        f" background-color: {tokens.ACCENT_SOFT}; color: {tokens.ACCENT};"
        f" border: 2px solid {tokens.ACCENT}; }}"
    )


# ── Context chips, confidence chips & warning banners ────────────────────────

#: objectName used to scope context-chip QSS without touching other labels.
CONTEXT_CHIP_OBJECT_NAME = "benchContextChip"


def make_context_chip(text: str) -> QLabel:
    """Return a small pill-shaped context chip (run number, field, temperature).

    Used by the wizard header band (:class:`WizardWindowBase`) to surface the
    analysis context at a glance next to the window title.
    """
    label = QLabel(str(text))
    label.setObjectName(CONTEXT_CHIP_OBJECT_NAME)
    label.setStyleSheet(
        f"QLabel#{CONTEXT_CHIP_OBJECT_NAME} {{"
        f" background-color: {tokens.SURFACE_ALT};"
        f" color: {tokens.TEXT_MUTED};"
        f" border: 1px solid {tokens.BORDER};"
        " border-radius: 9px;"
        " padding: 1px 8px;"
        " }"
    )
    return label


#: objectName used to scope confidence-chip QSS per tier.
CONFIDENCE_CHIP_OBJECT_NAME = "benchConfidenceChip"

#: Per-tier (background, border, text) colours for :func:`make_confidence_chip`.
_CONFIDENCE_CHIP_COLOURS = {
    "high": (tokens.SUCCESS_BG, tokens.SUCCESS_BORDER, tokens.OK),
    "medium": (tokens.WARN_BANNER_BG, tokens.WARN, tokens.WARN_BANNER_TEXT),
    "none": (tokens.SURFACE_ALT, tokens.BORDER, tokens.TEXT_MUTED),
}


def make_confidence_chip(text: str, tier: Literal["high", "medium", "none"]) -> QLabel:
    """Return a coloured confidence chip for a wizard answer card.

    ``tier`` picks the colour treatment (green / amber / muted); ``text`` is the
    caller-supplied label (e.g. "High confidence"). The prose sentence stays a
    separate label — the chip is the at-a-glance grade only.
    """
    bg, border, fg = _CONFIDENCE_CHIP_COLOURS.get(tier, _CONFIDENCE_CHIP_COLOURS["none"])
    label = QLabel(str(text))
    label.setObjectName(CONFIDENCE_CHIP_OBJECT_NAME)
    label.setStyleSheet(
        f"QLabel#{CONFIDENCE_CHIP_OBJECT_NAME} {{"
        f" background-color: {bg};"
        f" color: {fg};"
        f" border: 1px solid {border};"
        " border-radius: 9px;"
        " padding: 1px 10px;"
        " font-weight: 600;"
        " }"
    )
    return label


#: objectName used to scope warning-banner QSS without touching other labels.
WARNING_BANNER_OBJECT_NAME = "benchWarningBanner"


def make_warning_banner(text: str = "") -> QLabel:
    """Return an amber, word-wrapped warning-banner strip.

    The non-blocking "out of date" convention (``tokens.WARN_BANNER_*``, first
    used by the Global Parameter Fit window): wizards show one when a scope or
    seed edit makes the displayed results stale. Callers control visibility.
    """
    label = QLabel(str(text))
    label.setObjectName(WARNING_BANNER_OBJECT_NAME)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"QLabel#{WARNING_BANNER_OBJECT_NAME} {{"
        f" background-color: {tokens.WARN_BANNER_BG};"
        f" color: {tokens.WARN_BANNER_TEXT};"
        " border-radius: 4px;"
        " padding: 6px 10px;"
        " font-weight: 600;"
        " }"
    )
    return label


# ── HTML snippets ─────────────────────────────────────────────────────────────


def success_html(label: str, *, detail: str | None = None) -> str:
    """Return rich-text HTML for a success status line (green header + muted detail).

    Args:
        label:  Primary line, e.g. "Fit converged".
        detail: Optional secondary line in muted colour, e.g. "χ²/ν = 1.04 · npar = 5".
    """
    html = f'<span style="color:{tokens.OK};font-weight:600;">{label}</span>'
    if detail:
        html += f'<br><span style="color:{tokens.TEXT_MUTED};">{detail}</span>'
    return html


#: Colour per χ² verdict. ``good`` = green; ``poor`` (χ² too high) = error red;
#: ``overdone`` (χ² suspiciously low — overestimated errors / over-flexible model)
#: = accent, reading as "suspicious", not "bad". Mirrors WiMDA's green/chocolate/
#: purple scheme onto the Asymmetry palette.
_FIT_VERDICT_COLOURS = {"good": tokens.OK, "poor": tokens.ERROR, "overdone": tokens.ACCENT}


def fit_quality_chip_html(quality: dict | None, params_at_bound: list[str] | None = None) -> str:
    """Return inline coloured verdict chip(s) for a fit summary.

    ``quality`` is the additive ``"quality"`` key from
    :func:`~asymmetry.core.fitting.result_summary.fit_result_summary` (verdict +
    target band + ``marginal`` hint); ``params_at_bound`` is its
    ``"params_at_bound"`` list. Renders, as available, a χ² verdict chip and an
    "at bound" chip. A χ²ᵣ that is numerically near 1 but reads poor/overdone only
    because the band tightens at high ν (``marginal``) is shown amber as
    "<verdict> (marginal)" rather than alarming red/accent. Returns "" when there
    is nothing to show.
    """
    chips = ""
    if quality and quality.get("verdict"):
        verdict = str(quality["verdict"])
        if quality.get("marginal"):
            # χ²ᵣ is numerically within ~0.2 of 1 (a good fit) and reads
            # poor/overdone only because the confidence band tightens at high ν.
            # Lead with a neutral phrase so the at-a-glance chip isn't alarming;
            # the verdict and the full explanation remain in the hover tooltip
            # (P3-1, presentation only — the band math is unchanged).
            colour = tokens.WARN
            label = "near-ideal (band-tight)"
        else:
            colour = _FIT_VERDICT_COLOURS.get(verdict, tokens.TEXT_MUTED)
            label = verdict
        chips += f' · <span style="color:{colour};font-weight:600;">{label}</span>'
    if params_at_bound:
        chips += f' · <span style="color:{tokens.WARN};font-weight:600;">at bound</span>'
    return chips


def fit_quality_tooltip(quality: dict | None, params_at_bound: list[str] | None = None) -> str:
    """Return a teaching tooltip explaining the χ² verdict and any at-bound params."""
    lines: list[str] = []
    if quality and quality.get("verdict"):
        verdict = str(quality["verdict"])
        low = quality.get("band_low")
        high = quality.get("band_high")
        confidence = quality.get("confidence")
        dof = quality.get("dof")
        lines.append(f"Fit quality: {verdict}.")
        if quality.get("marginal"):
            # The cuprate χ²ᵣ=1.10/ν=1927 case: numerically near-ideal, flagged only
            # by the tight high-ν band. Reassure rather than alarm.
            lines.append(
                f"χ²ᵣ is within ~0.2 of 1, so this is numerically a good fit — it reads "
                f"“{verdict}” only because the band is tight at this ν."
            )
        if low is not None and high is not None and confidence is not None:
            pct = int(round(float(confidence) * 100))
            band = f"{float(low):.3f}–{float(high):.3f}"
            nu = f" (ν = {int(dof)})" if dof is not None else ""
            lines.append(f"A good fit's χ²ᵣ falls in [{band}] at {pct}%{nu}.")
            # Defuse the "poor at χ²ᵣ≈1.08" alarm for high-statistics muon data: the
            # band is a confidence interval that tightens with ν, so a near-unity χ²ᵣ
            # can read "poor" yet be a good fit. Clarity only — the band math is unchanged.
            lines.append(
                "The band is a confidence interval (WiMDA Rgoodfit), not a fixed cut-off, "
                "and tightens as ν grows — so at high statistics a χ²ᵣ near 1 can read "
                "“poor” yet still be a good fit. Rebinning shrinks the errors, so a bunched "
                "fit can read a high χ²ᵣ from small systematics — inspect the residuals. "
                "Tune the band in Options ▸ Fit quality confidence."
            )
        lines.append(
            "“overdone” reproduces the data better than the errors allow — usually "
            "overestimated errors or an over-flexible model; “poor” is worse than the "
            "errors allow."
        )
    if params_at_bound:
        names = ", ".join(str(n) for n in params_at_bound)
        lines.append(
            f"Parameter(s) at a bound: {names}. A free parameter pinned on its min/max is "
            "usually poorly constrained — the data did not determine it, so the optimiser "
            "parked it on the boundary (the fit can still report “converged”). Widen the "
            "bound, fix it from independent knowledge, or simplify the model."
        )
    return " ".join(lines)


def info_html(label: str) -> str:
    """Return rich-text HTML for an informational status line (accent colour)."""
    return f'<span style="color:{tokens.ACCENT};">{label}</span>'


def warning_html(label: str) -> str:
    """Return rich-text HTML for a warning status line (warn colour)."""
    return f'<span style="color:{tokens.WARN};">{label}</span>'


def error_html(label: str) -> str:
    """Return rich-text HTML for an error status line (error colour)."""
    return f'<span style="color:{tokens.ERROR};">{label}</span>'


# ── Group-state buttons ───────────────────────────────────────────────────────


def style_group_state_button(
    button: QPushButton,
    state: Literal["active", "selected", "unselected"],
    *,
    base: str = "",
    palette: Literal["blue", "red"] = "blue",
) -> None:
    """Apply BENCH styling to a group-state toggle button in fit_parameters_panel.

    Args:
        base:    Optional QSS prefix (e.g. scale-dependent radius/padding rules)
                 prepended before the state rules.
        palette: ``"blue"`` (default) uses the standard blue accent;
                 ``"red"`` uses the red FitSeries accent for series buttons.

    States:
        active:     Currently focused/active group — ACCENT_SOFT fill, ACCENT border.
        selected:   Included in multi-group fit — ACCENT_SOFT2 fill, ACCENT border, dim text.
        unselected: Not participating — SURFACE fill, BORDER border, muted text.
    """
    if palette == "red":
        accent = tokens.ACCENT_RED
        soft = tokens.ACCENT_RED_SOFT
        soft2 = tokens.ACCENT_RED_SOFT2
    else:
        accent = tokens.ACCENT
        soft = tokens.ACCENT_SOFT
        soft2 = tokens.ACCENT_SOFT2
    if state == "active":
        button.setStyleSheet(
            base + f"QPushButton {{ border: 2px solid {accent};"
            f" background: {soft}; color: {accent};"
            f" font-weight: 700; }}"
        )
    elif state == "selected":
        button.setStyleSheet(
            base + f"QPushButton {{ border: 1px solid {accent};"
            f" background: {soft2}; color: {tokens.TEXT};"
            f" font-weight: 500; }}"
        )
    else:
        button.setStyleSheet(
            base + f"QPushButton {{ border: 1px solid {tokens.BORDER};"
            f" background: {tokens.SURFACE}; color: {tokens.TEXT_MUTED};"
            f" font-weight: 400; }}"
        )
