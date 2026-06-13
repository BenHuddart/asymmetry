"""Shared widget-styling helpers for the BENCH design language.

Import these in any panel or window that needs BENCH-consistent table headers,
formula code-boxes, result-group success states, button QSS, or HTML snippets.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGroupBox, QLabel, QPushButton, QSizePolicy, QTableWidget

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import header_font, section_label_font

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

#: Back-compat: a few call sites still toggle a results QGroupBox by this name.
#: New code should build a QFrame result box (RESULT_BOX_OBJECT_NAME) instead.
RESULTS_GROUP_SUCCESS_STYLE = (
    "QGroupBox {"
    f" background-color: {tokens.SUCCESS_BG};"
    f" border: 1px solid {tokens.SUCCESS_BORDER};"
    " border-radius: 4px;"
    " margin-top: 10px;"
    "}"
)


def apply_results_group_success(group: QGroupBox) -> None:
    """Apply green-tint success style to a results QGroupBox."""
    group.setStyleSheet(RESULTS_GROUP_SUCCESS_STYLE)


def clear_results_group_style(group: QGroupBox) -> None:
    """Remove any inline style from a results QGroupBox (revert to bench.qss default)."""
    group.setStyleSheet("")


# ── Table style ───────────────────────────────────────────────────────────────


def apply_param_table_style(table: QTableWidget) -> None:
    """Apply BENCH styling to any parameter or data table.

    Sets a DemiBold section header, 11pt mono cell font, hides the row-number
    column, and tints alternating rows with surfaceAlt.
    """
    hdr_font = QFont()
    hdr_font.setPointSizeF(9.5)
    hdr_font.setWeight(QFont.Weight.DemiBold)
    hdr_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.3)
    table.horizontalHeader().setFont(hdr_font)
    table.verticalHeader().setVisible(False)
    table.setFont(mono_font(11.0))
    table.setAlternatingRowColors(True)
    table.setStyleSheet(f"QTableWidget {{ alternate-background-color: {tokens.SURFACE_ALT}; }}")


# ── Formula label ─────────────────────────────────────────────────────────────


def configure_formula_label(label: QLabel) -> None:
    """Style a QLabel as a read-only mono code-box for formula display."""
    label.setFont(mono_font(11.0))
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    label.setTextFormat(Qt.TextFormat.PlainText)
    # Ignored (not Expanding) horizontally: a long, unbreakable token — e.g. a
    # custom user-function name — has no wrap point, so under Expanding the
    # label's minimum width would grow to fit it and drag the whole inspector
    # dock wide (re-introducing a window scrollbar). Ignored makes the box fill
    # the width it is given and clip the overflow instead; the full expression
    # stays available via the tooltip set in _set_formula_label_text.
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.MinimumExpanding)
    label.setMinimumWidth(0)
    label.setStyleSheet(
        f"QLabel {{ background-color: {tokens.SURFACE_ALT}; border: 1px solid {tokens.BORDER};"
        " border-radius: 3px; padding: 6px 8px; }"
    )
    line_height = label.fontMetrics().lineSpacing()
    label.setMinimumHeight(line_height * 3 + 16)


# ── Table header font ─────────────────────────────────────────────────────────


def apply_param_table_header_font(table: QTableWidget) -> None:
    """Apply the BENCH column-header font to a table's horizontal header.

    Separated from apply_param_table_style so callers that handle row colours
    themselves can still pick up the canonical header font.
    """
    table.horizontalHeader().setFont(header_font())


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


def fit_quality_chip_html(quality: dict | None) -> str:
    """Return an inline coloured verdict chip for a ``quality`` summary dict.

    ``quality`` is the additive ``"quality"`` key from
    :func:`~asymmetry.core.fitting.result_summary.fit_result_summary` (verdict +
    target band). Returns an empty string when no verdict is available.
    """
    if not quality or not quality.get("verdict"):
        return ""
    verdict = str(quality["verdict"])
    colour = _FIT_VERDICT_COLOURS.get(verdict, tokens.TEXT_MUTED)
    return f' · <span style="color:{colour};font-weight:600;">{verdict}</span>'


def fit_quality_tooltip(quality: dict | None) -> str:
    """Return a teaching tooltip explaining the χ² verdict and target band."""
    if not quality or not quality.get("verdict"):
        return ""
    verdict = str(quality["verdict"])
    low = quality.get("band_low")
    high = quality.get("band_high")
    confidence = quality.get("confidence")
    dof = quality.get("dof")
    lines = [f"Fit quality: {verdict}."]
    if low is not None and high is not None and confidence is not None:
        pct = int(round(float(confidence) * 100))
        band = f"{float(low):.3f}–{float(high):.3f}"
        nu = f" (ν = {int(dof)})" if dof is not None else ""
        lines.append(f"A good fit's χ²ᵣ falls in [{band}] at {pct}%{nu}.")
    lines.append(
        "“overdone” reproduces the data better than the errors allow — usually "
        "overestimated errors or an over-flexible model; “poor” is worse than the "
        "errors allow."
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


# ── Footer hint label ─────────────────────────────────────────────────────────


def apply_footer_hint(label: QLabel) -> None:
    """Apply the BENCH footer-hint style to a QLabel (surfaceAlt bg, muted text)."""
    label.setStyleSheet(
        f"QLabel {{ background-color: {tokens.SURFACE_ALT};"
        f" color: {tokens.TEXT_MUTED};"
        f" border-top: 1px solid {tokens.BORDER};"
        f" padding: 5px 8px; font-size: 10px; }}"
    )
