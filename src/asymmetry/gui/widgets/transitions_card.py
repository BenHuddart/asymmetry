"""The Global Fit Wizard's *Transitions* card: the penalty path, made pickable.

A temperature or field series can cross one or more transitions, and the fit
function that describes the runs on one side does not describe the runs on the
other. The core's partition search answers "where are the breaks?" as a whole
*penalty path* — the best partition with exactly 0, 1, 2, … breaks — rather than
a single answer, and pre-selects the elbow. This card is that path made visible
and pickable: one row per path solution, the elbow row selected, a plain summary
of the selected row's transitions above it, and the two actions that follow from
a pick — run the coupled per-phase optimisation, and apply the resulting phases
to the project.

Like :mod:`asymmetry.gui.widgets.wizard_series_card` it is deliberately window-
and core-agnostic: it imports nothing from :mod:`asymmetry.core`. The host
adapts a ``PartitionPath`` into the plain :class:`TransitionRow` records below
and a set of optimised phase assessments into :class:`PhaseSummary` records, and
owns what "optimize" and "apply" mean — the card only emits the index of the row
the user picked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.fonts import mono_font
from asymmetry.gui.styles.metrics import row_height
from asymmetry.gui.styles.widgets import (
    build_primary_button_qss,
    build_segmented_button_qss,
    make_section_header,
)

#: Column order of the path table.
_COL_BREAKS = 0
_COL_BOUNDARIES = 1
_COL_GAIN = 2
_COL_STATUS = 3

#: Shown wherever a solution has nothing to report on that column — the
#: zero-break row has no boundary and no gain to compare against.
_EMPTY_CELL = "—"

#: Status words. "elbow" marks the path's own pre-selection; "verified" marks a
#: row whose phases have been fitted exactly (tier 3), so its total is measured
#: rather than closed-form.
_STATUS_ELBOW = "elbow"
_STATUS_VERIFIED = "verified"

#: The footnote under the table. The partition is scored with BIC whatever
#: ranking metric the user picked (a structural change between nested models is
#: nearly free under AIC), so the card says so rather than letting the metric
#: selector above imply otherwise.
_METRIC_FOOTNOTE = "Transitions are scored with BIC; the ranking metric applies within a phase."

#: Height allowance, in table rows, for the header row plus the frame.
_TABLE_CHROME_ROWS = 2


@dataclass(frozen=True)
class TransitionRow:
    """One solution of the penalty path, as the card states it.

    ``breaks`` is the solution's break count *after* structure merging;
    ``boundaries_text`` is the pre-formatted boundary list (already carrying the
    axis unit) or empty for a break-free row. ``gain_text`` is the pre-formatted
    ΔBIC against one fewer break, empty at the top of the path. ``summary`` is
    the plain-language sentence shown above the table while this row is
    selected.
    """

    breaks: int
    boundaries_text: str
    gain_text: str
    is_elbow: bool
    is_verified: bool
    excluded_note: str
    summary: str

    @property
    def status_text(self) -> str:
        """The Status cell: elbow / verified markers plus any exclusion note."""
        parts = [
            _STATUS_ELBOW if self.is_elbow else "",
            _STATUS_VERIFIED if self.is_verified else "",
            self.excluded_note,
        ]
        return " · ".join(part for part in parts if part)


@dataclass(frozen=True)
class PhaseSummary:
    """One optimised phase of the selected solution, for the per-phase strip.

    ``segment_index`` addresses the phase inside its solution, so a click can be
    routed back to that phase's assessment. ``color`` is the phase's identity
    swatch colour.
    """

    segment_index: int
    ordinal: int
    color: str
    range_text: str
    template_title: str
    roles_text: str
    confidence_text: str


class TransitionsCard(QWidget):
    """The penalty path, the transitions it implies, and the actions on it."""

    #: Emitted with the path index of the newly selected row.
    selection_changed = Signal(int)
    #: Emitted with the path index the user asked to optimise per phase.
    optimize_requested = Signal(int)
    #: Emitted with the path index the user asked to apply as phase groups.
    apply_requested = Signal(int)
    #: Emitted with the ``segment_index`` of a clicked phase in the strip.
    phase_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[TransitionRow] = []
        self._selected_index = 0
        self._phase_buttons: list[QPushButton] = []
        self._actions_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(make_section_header("Transitions"))

        self._summary_label = QLabel("", self)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Breaks", "Boundaries", "Gain", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(
            _COL_BOUNDARIES, QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeaderItem(_COL_GAIN).setToolTip(
            "ΔBIC against the solution with one fewer break."
        )
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        self._footnote = QLabel(_METRIC_FOOTNOTE, self)
        self._footnote.setWordWrap(True)
        self._footnote.setStyleSheet(f"QLabel {{ color: {tokens.TEXT_MUTED}; }}")
        layout.addWidget(self._footnote)

        # Per-phase strip: one clickable chip per optimised phase, hidden until
        # an optimisation has produced phase assessments.
        self._phase_strip = QFrame(self)
        self._phase_row = QHBoxLayout(self._phase_strip)
        self._phase_row.setContentsMargins(0, 0, 0, 0)
        self._phase_row.addStretch()
        self._phase_strip.setVisible(False)
        layout.addWidget(self._phase_strip)

        button_row = QHBoxLayout()
        self._optimize_btn = QPushButton("Optimize phases", self)
        self._optimize_btn.setStyleSheet(build_primary_button_qss())
        self._optimize_btn.clicked.connect(self._on_optimize_clicked)
        button_row.addWidget(self._optimize_btn)
        self._apply_btn = QPushButton("Apply phases", self)
        self._apply_btn.setStyleSheet(build_primary_button_qss())
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        self._apply_btn.setVisible(False)
        button_row.addWidget(self._apply_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_rows(self, rows: Sequence[TransitionRow], selected_index: int) -> None:
        """Render the whole path and select one row, without emitting."""
        self._rows = list(rows)
        self._selected_index = int(selected_index)
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._rows))
        for index, row in enumerate(self._rows):
            breaks_item = QTableWidgetItem(str(row.breaks))
            breaks_item.setFont(mono_font())
            self._table.setItem(index, _COL_BREAKS, breaks_item)
            self._table.setItem(
                index,
                _COL_BOUNDARIES,
                QTableWidgetItem(row.boundaries_text or _EMPTY_CELL),
            )
            gain_item = QTableWidgetItem(row.gain_text or _EMPTY_CELL)
            gain_item.setFont(mono_font())
            self._table.setItem(index, _COL_GAIN, gain_item)
            self._table.setItem(index, _COL_STATUS, QTableWidgetItem(row.status_text))
        self._table.setCurrentCell(self._selected_index, _COL_BREAKS)
        self._table.blockSignals(False)
        self._table.setMaximumHeight(row_height() * (len(self._rows) + _TABLE_CHROME_ROWS))
        self._sync_selected_row()

    def set_phases(self, phases: Sequence[PhaseSummary]) -> None:
        """Populate (or clear) the per-phase strip beneath the table."""
        self._clear_phases()
        for phase in phases:
            button = QPushButton(
                f"Phase {phase.ordinal} · {phase.range_text}\n"
                f"{phase.template_title}\n"
                f"{phase.roles_text}\n"
                f"{phase.confidence_text}",
                self._phase_strip,
            )
            # The phase's identity colour rides on a left stripe, mirroring the
            # Data Browser's 4 px phase stripe, so the label text keeps the
            # readable default foreground on every palette entry.
            button.setStyleSheet(
                build_segmented_button_qss(padding_h=8)
                + f" QPushButton {{ border-left: 4px solid {phase.color}; }}"
            )
            button.setToolTip(f"Show the fit for phase {phase.ordinal} in the detail tables.")
            button.clicked.connect(
                lambda _checked=False, index=phase.segment_index: self.phase_selected.emit(index)
            )
            self._phase_row.insertWidget(self._phase_row.count() - 1, button)
            self._phase_buttons.append(button)
        self._phase_strip.setVisible(bool(self._phase_buttons))

    def selected_index(self) -> int:
        """The path index of the selected row."""
        return self._selected_index

    def selected_row(self) -> TransitionRow | None:
        """The selected row, or ``None`` when the card holds no path."""
        if not self._rows:
            return None
        return self._rows[self._selected_index]

    def set_actions_enabled(self, enabled: bool) -> None:
        """Enable/disable both actions (the host disables them while busy)."""
        self._actions_enabled = enabled
        self._sync_buttons()

    def clear(self) -> None:
        """Reset to the empty state: no path, no phases, no summary."""
        self._rows = []
        self._selected_index = 0
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        self._table.blockSignals(False)
        self._summary_label.setText("")
        self._clear_phases()
        self._sync_buttons()

    # ── Internals ──────────────────────────────────────────────────────────

    def _clear_phases(self) -> None:
        for button in self._phase_buttons:
            self._phase_row.removeWidget(button)
            button.setParent(None)
            button.deleteLater()
        self._phase_buttons.clear()
        self._phase_strip.setVisible(False)

    def _on_selection_changed(self) -> None:
        row = self._table.currentRow()
        if row == self._selected_index:
            return
        self._selected_index = row
        self._sync_selected_row()
        self.selection_changed.emit(row)

    def _sync_selected_row(self) -> None:
        selected = self.selected_row()
        self._summary_label.setText(selected.summary if selected is not None else "")
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        selected = self.selected_row()
        # A break-free row is the ordinary single-segment optimise, which the
        # screening shortlist's own Optimize button already runs.
        has_breaks = selected is not None and selected.breaks >= 1
        self._optimize_btn.setEnabled(self._actions_enabled and has_breaks)
        verified = selected is not None and selected.is_verified
        self._apply_btn.setVisible(verified)
        self._apply_btn.setEnabled(self._actions_enabled and verified)

    def _on_optimize_clicked(self) -> None:
        self.optimize_requested.emit(self._selected_index)

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit(self._selected_index)
