"""The fit tabs' results card: one surface carrying a fit's outcome and hand-offs.

Replaces the prose ``#resultBox`` both fit tabs used to render into with the
card grammar the Parameters panel established: a header strip (verdict tag ·
headline · meta), a rich-text detail line, an optional wrapping strip of
per-member verdict chips, and the hand-off buttons as segmented buttons in a
wrapping row.

The card is a pure view. It renders the plain :class:`FitCardSummary` /
:class:`MemberChip` payloads a tab hands it and emits a signal per gesture; it
owns no fit concept and imports nothing from ``asymmetry.core`` or from the
panels, so it can be built and tested on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.typography import SIZE_NUMERIC, footer_font, status_font
from asymmetry.gui.styles.widgets import (
    FIT_VERDICT_CHIP_COLOURS,
    NEUTRAL_CHIP_COLOURS,
    VERDICT_CHIP_OBJECT_NAME,
    build_segmented_button_qss,
    clear_layout,
    make_context_chip,
    verdict_chip_qss,
)
from asymmetry.gui.widgets.elided_label import ElidedLabel
from asymmetry.gui.widgets.flow_layout import FlowLayout

__all__ = ["TONE_COLOURS", "FitCardSummary", "FitResultsCard", "MemberChip"]

#: objectName so the card's chrome targets only the surface frame (a bare
#: ``QFrame { … }`` rule would cascade onto the child labels and buttons).
SURFACE_OBJECT_NAME = "fitResultsCardSurface"

#: The tone of a card's tag — how the outcome reads at a glance.
CardTone = Literal["ok", "warn", "error", "neutral"]

#: Tag (background, border, text) per tone. "ok"/"error" are the χ² verdict
#: chip's own green/red so a card and a verdict chip never disagree; "warn" is
#: the amber non-blocking banner convention (a flagged-but-usable fit). Public
#: so a verdict chip *beside* a card (the Batch tab's run row) reads the same
#: triple rather than restating it.
TONE_COLOURS: dict[str, tuple[str, str, str]] = {
    "ok": FIT_VERDICT_CHIP_COLOURS["good"],
    "warn": (tokens.WARN_BANNER_BG, tokens.WARN, tokens.WARN_BANNER_TEXT),
    "error": FIT_VERDICT_CHIP_COLOURS["poor"],
    "neutral": NEUTRAL_CHIP_COLOURS,
}


@dataclass(frozen=True)
class MemberChip:
    """One run's verdict chip in a batch card's member strip."""

    #: Run number the chip opens the results window for.
    run: int
    #: Chip label ("3001 ✓ 0.98").
    text: str
    #: The chip's (background, border, text) colours — this member's verdict.
    colours: tuple[str, str, str]
    #: Hover text: the member's advisory flags.
    tooltip: str


@dataclass(frozen=True)
class FitCardSummary:
    """Everything :class:`FitResultsCard` renders for a completed fit."""

    #: Header tag ("Fit ✓", "Fit ⚠", "Batch ✓", "Batch ⚠").
    tag: str
    tone: CardTone
    #: Muted phrase beside the tag ("converged", "3 of 4 converged").
    headline: str
    #: Right-hand mono read-out ("ndof 595 · npar 5").
    meta: str
    #: The body line, already rich text (χ²/ν, the verdict chip, warnings).
    detail_html: str
    #: One chip per member of a batch; empty for a single fit.
    members: tuple[MemberChip, ...] = ()
    #: Hover text for the body line — the χ² band behind a verdict chip it
    #: renders, which the line itself has no room to spell out.
    detail_tooltip: str = ""


def _wrapping_strip(parent: QWidget) -> tuple[QWidget, FlowLayout]:
    """A widget whose items wrap, and whose height follows the width it is given."""
    strip = QWidget(parent)
    layout = FlowLayout(strip)
    layout.setContentsMargins(0, 0, 0, 0)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    strip.setSizePolicy(policy)
    return strip, layout


class FitResultsCard(QWidget):
    """A fit's outcome: tag · headline · meta, a detail line, members, hand-offs."""

    #: A hand-off button was pressed; carries its label.
    action_triggered = Signal(str)
    #: A member chip was pressed; carries that member's run number.
    member_requested = Signal(int)

    def __init__(
        self,
        actions: Sequence[tuple[str, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName(SURFACE_OBJECT_NAME)
        self._surface.setStyleSheet(
            f"QFrame#{SURFACE_OBJECT_NAME} {{ border: 1px solid {tokens.BORDER};"
            f" background: {tokens.SURFACE}; border-radius: 4px; }}"
        )
        outer.addWidget(self._surface)

        surface_layout = QVBoxLayout(self._surface)
        surface_layout.setContentsMargins(8, 6, 8, 6)
        surface_layout.setSpacing(4)

        # ── Header: tag · headline · meta ───────────────────────────────────
        header = QWidget(self._surface)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        surface_layout.addWidget(header)

        self._tag = QLabel(header)
        self._tag.setObjectName(VERDICT_CHIP_OBJECT_NAME)
        header_row.addWidget(self._tag)

        self._headline = ElidedLabel("", header)
        # ElidedLabel paints its own text, so the muted colour is the pen's.
        self._headline.set_pen_color(tokens.TEXT_MUTED)
        header_row.addWidget(self._headline, 1)

        self._meta = QLabel(header)
        self._meta.setFont(footer_font())
        self._meta.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        header_row.addWidget(self._meta)

        self._meta_tag = make_context_chip("")
        header_row.addWidget(self._meta_tag)

        # ── Body: the detail line ───────────────────────────────────────────
        self._detail = QLabel(self._surface)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setWordWrap(True)
        self._detail.setFont(status_font())
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        surface_layout.addWidget(self._detail)

        # ── Members: one verdict chip per run of a batch ────────────────────
        self._members, self._members_layout = _wrapping_strip(self._surface)
        surface_layout.addWidget(self._members)

        # ── Hand-offs ───────────────────────────────────────────────────────
        action_strip, action_layout = _wrapping_strip(self._surface)
        surface_layout.addWidget(action_strip)

        #: What the card currently says, as one rich-text line — see
        #: :meth:`content_html`. Recorded as it is rendered rather than read back
        #: off the widgets, which cannot say whether a tag belongs to a summary.
        self._content_html = ""

        self._actions: dict[str, QPushButton] = {}
        for label, tooltip in actions:
            button = QPushButton(label, action_strip)
            button.setStyleSheet(build_segmented_button_qss())
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _checked=False, name=label: self.action_triggered.emit(name)
            )
            action_layout.addWidget(button)
            self._actions[label] = button

        self.set_message("")

    # ── Public API ──────────────────────────────────────────────────────────

    def set_message(
        self,
        text: str,
        *,
        tag: str = "No fit yet",
        tone: CardTone = "neutral",
        tooltip: str = "",
    ) -> None:
        """Show *text* as the card's whole content — a placeholder, progress or error.

        The completed-fit read-outs (members, the mono meta) go away, so a
        message can never be read as belonging to a fit that is no longer on the
        card. *tone* carries the tag's colour for a read-out replayed from saved
        state, where the outcome is known but there is no live result behind it;
        *tooltip* is the body line's hover text, and defaults to none so the next
        message never inherits the last one's.
        """
        self._apply_tag(tag, tone)
        self._headline.setText("")
        self._meta.setText("")
        self._detail.setText(text)
        self._detail.setToolTip(tooltip)
        self._set_members(())
        self._apply_meta()
        self._content_html = text

    def set_summary(self, summary: FitCardSummary) -> None:
        """Render a completed fit."""
        self._apply_tag(summary.tag, summary.tone)
        self._headline.setText(summary.headline)
        self._meta.setText(summary.meta)
        self._detail.setText(summary.detail_html)
        self._detail.setToolTip(summary.detail_tooltip)
        self._set_members(summary.members)
        self._apply_meta()
        self._content_html = f"<b>{summary.tag}</b> {summary.headline}<br>{summary.detail_html}"

    def tag_text(self) -> str:
        """The card's current tag, as the tabs persist it beside the read-out."""
        return self._tag.text()

    def content_html(self) -> str:
        """What the card currently says, as one rich-text line.

        A message is its own text; a summary folds its tag and headline in front
        of the detail. The fit tabs persist this line as a run's saved read-out
        and replay it through :meth:`set_message` when the run comes back, so the
        outcome survives rather than only the statistics.
        """
        return self._content_html

    def set_meta_tag(self, text: str | None, tooltip: str = "") -> None:
        """Show a tag in place of the mono meta read-out ("seeds from 3001").

        ``None`` drops it and hands the slot back to the summary's meta. The tag
        outlives a :meth:`set_message` — provenance of the values on the tab is
        not part of any one fit's outcome.
        """
        self._meta_tag.setText(text or "")
        self._meta_tag.setToolTip(tooltip)
        self._apply_meta()

    def set_action_enabled(self, label: str, enabled: bool, tooltip: str | None = None) -> None:
        """Enable or disable one hand-off, optionally replacing its hover text."""
        button = self._actions[label]
        button.setEnabled(enabled)
        if tooltip is not None:
            button.setToolTip(tooltip)

    # ── Internals ───────────────────────────────────────────────────────────

    def _apply_tag(self, text: str, tone: CardTone) -> None:
        self._tag.setText(text)
        self._tag.setStyleSheet(verdict_chip_qss(TONE_COLOURS[tone], widget="QLabel"))

    def _apply_meta(self) -> None:
        # The header's right-hand slot holds one of the two: a tag wins, since
        # it is set deliberately and the mono meta rides on whatever the last
        # summary said.
        tagged = bool(self._meta_tag.text())
        self._meta_tag.setVisible(tagged)
        self._meta.setVisible(bool(self._meta.text()) and not tagged)

    def _set_members(self, members: Sequence[MemberChip]) -> None:
        clear_layout(self._members_layout)
        for member in members:
            chip = QPushButton(member.text, self._members)
            chip.setObjectName(VERDICT_CHIP_OBJECT_NAME)
            chip.setFlat(True)
            chip.setFont(mono_font(SIZE_NUMERIC))
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(member.tooltip)
            chip.setStyleSheet(verdict_chip_qss(member.colours))
            chip.clicked.connect(
                lambda _checked=False, run=member.run: self.member_requested.emit(run)
            )
            self._members_layout.addWidget(chip)
        self._members.setVisible(bool(members))
