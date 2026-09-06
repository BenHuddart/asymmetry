"""The ⓘ popover behind a Data Browser phase header (transitions plan, D2).

The phase sub-headers in the Data Browser are deliberately compact — an
ordinal, a swatch and a range — because a series can carry five of them and a
header that spells out model, fit state and provenance turns the run table into
a wall of text. Everything the compact header leaves out lives here, one click
away: what was fitted, how the fit went, which parameters are shared, where the
breaks were estimated, and how the partition was found.

Two shapes, one frame:

* :meth:`PhaseInfoPopover.show_phase` — one phase: name, range, run count,
  **Model**, **Global fit**, **Shared**, **Boundaries**, **Found by**, and the
  three actions the phase's context menu also offers.
* :meth:`PhaseInfoPopover.show_series` — the parent series header: how many
  phases and transitions the partition has, and when it was found.

Provenance is read from
:attr:`~asymmetry.core.representation.group.DataGroup.phase_provenance`, the
plain dict D1 defined and the wizard (D3) writes. Every key is genuinely
optional — a phase exists as soon as the partition is applied, which is before
its own global fit has run — so a row is rendered only for the keys present
rather than showing "unknown" placeholders for a state that simply has not
happened yet.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from asymmetry.core.representation.group import DataGroup
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.metrics import char_width
from asymmetry.gui.styles.palette import is_dark_surface
from asymmetry.gui.styles.typography import SIZE_BODY, SIZE_NUMERIC, section_label_font
from asymmetry.gui.utils.phase_colors import (
    format_phase_boundary,
    format_phase_range,
    resolve_phase_color,
)
from asymmetry.gui.widgets.key_value_grid import KeyValueGrid

#: Provenance keys the popover renders, in display order, with their labels.
#: ``model_title`` / ``confidence`` are written by the wizard when it applies a
#: partition; ``fit_state``, ``reduced_chi_squared`` and ``shared_parameters``
#: are written by the per-phase global fit (D3). ``found_at`` is the wizard run
#: date as an ISO string.
_MODEL_KEY = "model_title"
_FIT_STATE_KEY = "fit_state"
_REDUCED_CHI2_KEY = "reduced_chi_squared"
_CONFIDENCE_KEY = "confidence"
_SHARED_KEY = "shared_parameters"
_FOUND_AT_KEY = "found_at"
_BREAKS_KEY = "selected_breaks"
_GAINS_KEY = "gains"

#: Ceiling on the value column, in characters of the monospaced font the grid
#: renders values in. The frame otherwise sizes itself to its content
#: (``adjustSize`` after each populate): a fixed width truncates the model title
#: and the boundary estimates on a perfectly ordinary phase (the longest row a
#: real phase carries, "Found by", is 43 characters). The cap only bites on a
#: pathological model title, which the fit panel spells out in full anyway.
_POPOVER_MAX_VALUE_CHARS = 60
#: Horizontal chrome outside the two text columns: the frame's two 12 px content
#: margins, the grid's 12 px column gap, and its 1 px border either side.
_POPOVER_CHROME_PX = 12 + 12 + 12 + 2


def _format_gains(gains: Any) -> str:
    """Spell the penalty-path marginal gains as ``"12.4, 3.1"``."""
    return ", ".join(f"{float(gain):.1f}" for gain in gains)


class PhaseInfoPopover(QFrame):
    """Frameless ``Qt.Popup`` detailing one phase (or a series' partition).

    A ``Qt.Popup`` window grabs the mouse and closes itself on the first click
    outside its own frame, which is exactly the dismissal a header indicator
    wants — no outside-click filter and no "is it still open" flag to keep in
    sync with the table.

    The three action buttons re-emit through this widget's signals rather than
    acting; the Data Browser routes them to the very same panel signals its
    phase context menu uses, so the popover and the menu can never drift into
    two different meanings of "Fit this phase…".
    """

    #: Emitted with the phase's group id when an action button is pressed. The
    #: popover closes first, so the host never acts against a stale frame.
    fit_requested = Signal(str)
    show_series_requested = Signal(str)
    rename_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("phaseInfoPopover")
        self.setStyleSheet(
            f"QFrame#phaseInfoPopover {{ background: {tokens.SURFACE}; "
            f"border: 1px solid {tokens.BORDER}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._title = QLabel()
        self._title.setFont(section_label_font())
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._subtitle = QLabel()
        self._subtitle.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        self._subtitle.setWordWrap(True)
        layout.addWidget(self._subtitle)

        self._grid = KeyValueGrid(self)
        layout.addWidget(self._grid)

        self._actions = QWidget(self)
        actions_layout = QHBoxLayout(self._actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        self._fit_button = QPushButton("Fit this phase…", self._actions)
        self._series_button = QPushButton("Show series", self._actions)
        self._rename_button = QPushButton("Rename", self._actions)
        for button in (self._fit_button, self._series_button, self._rename_button):
            button.setFont(self.font())
            actions_layout.addWidget(button)
        actions_layout.addStretch(1)
        layout.addWidget(self._actions)

        #: The phase the frame currently describes; the action buttons are only
        #: reachable while a phase is shown (``show_series`` hides them), so
        #: this is always a real phase id at the moment a button fires.
        self._group_id = ""
        self._fit_button.clicked.connect(self._on_fit)
        self._series_button.clicked.connect(self._on_show_series)
        self._rename_button.clicked.connect(self._on_rename)

    # ── content ─────────────────────────────────────────────────────────────

    def _set_rows(self, rows: list[tuple[str, str]]) -> None:
        """Fill the grid, eliding any value that would push past the width cap.

        ``KeyValueGrid``'s value labels do not wrap, so their text sets the
        layout's *minimum* width — which outranks ``setMaximumWidth`` and would
        stretch the frame across the window for a long model title. Eliding the
        text instead keeps the cap real, and every row short of it (all of them,
        for the wizard's own titles) is still shown in full.

        The frame's own maximum follows from the two columns it ends up with,
        measured in the grid's fonts rather than the proportional UI font — the
        value column is monospaced and around a third wider per character.
        """
        label_metrics = QFontMetrics(mono_font(SIZE_BODY))
        value_font = mono_font(SIZE_NUMERIC)
        value_metrics = QFontMetrics(value_font)
        label_column = max((label_metrics.horizontalAdvance(name) for name, _ in rows), default=0)
        budget = char_width(_POPOVER_MAX_VALUE_CHARS, value_font)
        self.setMaximumWidth(budget + label_column + _POPOVER_CHROME_PX)
        self._grid.set_rows(
            [
                (name, value_metrics.elidedText(value, Qt.TextElideMode.ElideRight, budget))
                for name, value in rows
            ]
        )

    def show_phase(self, phase: DataGroup, *, run_count: int) -> None:
        """Populate the frame for one *phase* and its member count."""
        self._group_id = phase.group_id
        dark = is_dark_surface(self.palette())
        color = resolve_phase_color(phase, dark=dark)
        self._title.setText(phase.name)
        self._title.setStyleSheet(f"QLabel {{ color: {color}; }}")

        span = format_phase_range(phase)
        runs = f"{run_count} run" if run_count == 1 else f"{run_count} runs"
        self._subtitle.setText(f"{span} · {runs}" if span else runs)

        provenance = phase.phase_provenance
        rows: list[tuple[str, str]] = []
        if _MODEL_KEY in provenance:
            rows.append(("Model", str(provenance[_MODEL_KEY])))
        fit_bits = [
            str(provenance[key])
            for key in (_FIT_STATE_KEY, _REDUCED_CHI2_KEY, _CONFIDENCE_KEY)
            if key in provenance
        ]
        if fit_bits:
            rows.append(("Global fit", " · ".join(fit_bits)))
        if _SHARED_KEY in provenance:
            shared = provenance[_SHARED_KEY]
            rows.append(("Shared", ", ".join(str(name) for name in shared) or "none"))
        rows.append(
            (
                "Boundaries",
                f"{format_phase_boundary(phase.phase_boundaries['lower'], phase.order_key)}"
                f"  →  "
                f"{format_phase_boundary(phase.phase_boundaries['upper'], phase.order_key)}",
            )
        )
        found_bits: list[str] = []
        if _FOUND_AT_KEY in provenance:
            found_bits.append(str(provenance[_FOUND_AT_KEY]))
        if _BREAKS_KEY in provenance:
            found_bits.append(f"{provenance[_BREAKS_KEY]} breaks")
        if _GAINS_KEY in provenance:
            found_bits.append(f"gains {_format_gains(provenance[_GAINS_KEY])}")
        if found_bits:
            rows.append(("Found by", " · ".join(found_bits)))
        self._set_rows(rows)

        self._actions.setVisible(True)
        self.adjustSize()

    def show_series(self, series: DataGroup, phases: list[DataGroup]) -> None:
        """Populate the frame for a partitioned *series* group.

        The series header's ⓘ answers a different question from a phase's —
        "how is this scan split?" rather than "what happened in this phase?" —
        so it shows the partition shape and its date, and offers no per-phase
        actions.
        """
        self._group_id = series.group_id
        self._title.setStyleSheet("")
        self._title.setText(series.name)
        count = len(phases)
        transitions = max(count - 1, 0)
        self._subtitle.setText(
            f"{count} phases · {transitions} transition{'' if transitions == 1 else 's'}"
        )
        rows = [
            (
                f"Phase {phase.phase_ordinal}",
                f"{phase.name} · {format_phase_range(phase)}",
            )
            for phase in phases
        ]
        dates = {
            str(phase.phase_provenance[_FOUND_AT_KEY])
            for phase in phases
            if _FOUND_AT_KEY in phase.phase_provenance
        }
        if dates:
            rows.append(("Found by", " · ".join(sorted(dates))))
        self._set_rows(rows)

        self._actions.setVisible(False)
        self.adjustSize()

    # ── actions ─────────────────────────────────────────────────────────────

    def _on_fit(self) -> None:
        group_id = self._group_id
        self.close()
        self.fit_requested.emit(group_id)

    def _on_show_series(self) -> None:
        group_id = self._group_id
        self.close()
        self.show_series_requested.emit(group_id)

    def _on_rename(self) -> None:
        group_id = self._group_id
        self.close()
        self.rename_requested.emit(group_id)


__all__ = ["PhaseInfoPopover"]
