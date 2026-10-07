"""Auto-detect's review: each estimate beside the current value, ticked to apply (D6).

- Rows still on defaults or earlier estimates start ticked; a row holding a
  typed value never does.
- An estimate with contrast below :data:`MIN_CONTRAST` cannot be applied.
- The setup row (sense with φ_RF, which is estimated for that sense) comes from
  the runs together and writes to every run. Each run then has a gain row and a
  baseline row per period (D9: the periods' baselines differ). ν_RF and the B₁
  axis are the user's: the header names them.
- Nothing changes until Apply, which reports the ticked values through
  :attr:`FrameReviewDialog.applied` for the host to write as estimates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from asymmetry.core.transform.rotating_frame import (
    MIN_CONTRAST,
    FrameEstimate,
    Provenance,
    RotatingFrame,
)
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import build_primary_button_qss, make_section_header

__all__ = ["FrameReviewDialog"]

FOOTER_NOTE = (
    "Ticked rows are written as estimates. Rows on defaults or earlier estimates start "
    "ticked; a row holding a typed value never does. An estimate with contrast below "
    f"{MIN_CONTRAST:g} cannot be applied."
)

#: A two-period run's periods by the app's red/green convention (red is period 1).
_PERIOD_NAMES = {2: ("red", "green")}


def _contrast(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def _sense(value: int) -> str:
    return "+1" if value > 0 else "−1"


def _shared(texts: set[str]) -> str:
    return texts.pop() if len(texts) == 1 else "mixed"


@dataclass(frozen=True)
class _Row:
    """One tickable proposal: its cells by grid column, and what Apply writes."""

    title: str
    #: ``{grid column: "current → estimate"}``.
    cells: dict[int, str]
    contrast: float
    typed: bool
    #: ``{run: {field: value}}``; per-period baselines go under ``"baselines"``.
    changes: dict[int, dict[str, object]]
    note: str


class FrameReviewDialog(QDialog):
    """Review an Auto-detect estimate against the runs' current frames."""

    #: ``({run: {field: value, "baselines": {period: (x, y)}}}, status text)`` on Apply.
    applied = Signal(object, str)

    def __init__(
        self,
        estimate: FrameEstimate,
        frames: Mapping[int, RotatingFrame],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-detect rotating frame")
        self._estimate = estimate
        values = list(frames.values())
        setup_typed = any(
            frame.provenance[name] is Provenance.TYPED
            for frame in values
            for name in ("sense", "rf_phase_deg")
        )
        setup = _Row(
            "Sense, φ_RF",
            {
                1: f"{_shared({_sense(f.sense) for f in values})} → {_sense(estimate.sense)}",
                2: f"{_shared({f'{f.rf_phase_deg:.1f}°' for f in values})}"
                f" → {estimate.rf_phase_deg:.1f}°",
            },
            estimate.contrast,
            setup_typed,
            {
                run: {"sense": estimate.sense, "rf_phase_deg": estimate.rf_phase_deg}
                for run in frames
            },
            f"contrast {_contrast(estimate.contrast)}",
        )
        per_run: list[_Row] = []
        for run in estimate.runs:
            frame = frames[run.run_key]
            note = f"contrast {_contrast(run.contrast)}"
            per_run.append(
                _Row(
                    f"Run {run.run_key} · gain",
                    {3: f"{frame.gain:.3f} → {run.gain:.3f}"},
                    run.contrast,
                    frame.provenance["gain"] is Provenance.TYPED,
                    {run.run_key: {"gain": run.gain}},
                    note,
                )
            )
            names = _PERIOD_NAMES.get(len(run.baselines), ())
            for period, (bx, by) in enumerate(run.baselines):
                current = frame.baselines[period]
                suffix = f" · {names[period]}" if names else ""
                per_run.append(
                    _Row(
                        f"Run {run.run_key}{suffix}",
                        {1: f"{current.x:.2f} → {bx:.2f}", 2: f"{current.y:.2f} → {by:.2f}"},
                        run.contrast,
                        current.provenance is Provenance.TYPED,
                        {run.run_key: {"baselines": {period: (bx, by)}}},
                        note,
                    )
                )
        self._ticks: list[tuple[QCheckBox, _Row]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.addWidget(make_section_header("Setup"), 0, 0)
        nu = _shared({f"{f.frequency_mhz:.6g} MHz" for f in values})
        axis = _shared({str(f.b1_axis) for f in values})
        header = self._muted(f"at ν_RF = {nu}, B₁ ∥ {axis} · from the runs together")
        grid.addWidget(header, 0, 1, 1, 4)
        for column, heading in enumerate(("Sense", "φ_RF"), start=1):
            grid.addWidget(self._muted(heading), 1, column)
        self._add_row(grid, 2, setup)
        grid.addWidget(make_section_header("Runs"), 3, 0)
        for column, heading in enumerate(("b_x (%)", "b_y (%)", "a_y/a_x"), start=1):
            grid.addWidget(self._muted(heading), 3, column)
        for line, row in enumerate(per_run, start=4):
            self._add_row(grid, line, row)
        layout.addLayout(grid)

        footer = self._muted(FOOTER_NOTE)
        footer.setWordWrap(True)
        layout.addWidget(footer)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._apply_btn = QPushButton()
        self._apply_btn.setStyleSheet(build_primary_button_qss())
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self._on_apply)
        buttons.addWidget(self._apply_btn)
        layout.addLayout(buttons)
        self._sync_apply()

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {tokens.TEXT_MUTED};")
        return label

    def _add_row(self, grid: QGridLayout, line: int, row: _Row) -> None:
        tick = QCheckBox(row.title)
        weak = row.contrast < MIN_CONTRAST
        tick.setChecked(not row.typed and not weak)
        tick.setEnabled(not weak)
        tick.toggled.connect(self._sync_apply)
        grid.addWidget(tick, line, 0)
        for column, text in row.cells.items():
            cell = QLabel(text)
            cell.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(cell, line, column)
        if weak:
            note = QLabel(f"contrast {_contrast(row.contrast)} < {MIN_CONTRAST:g} — cannot apply")
            note.setStyleSheet(f"color: {tokens.WARN};")
        else:
            note = self._muted(("typed · " if row.typed else "") + row.note)
        grid.addWidget(note, line, 4)
        self._ticks.append((tick, row))

    def ticked_count(self) -> int:
        return sum(tick.isChecked() for tick, _row in self._ticks)

    def changes(self) -> dict[int, dict[str, object]]:
        """``{run: {field: value, "baselines": {period: (x, y)}}}`` of the ticked rows."""
        changes: dict[int, dict[str, object]] = {}
        for tick, row in self._ticks:
            if not tick.isChecked():
                continue
            for run, values in row.changes.items():
                merged = changes.setdefault(run, {})
                for name, value in values.items():
                    if name == "baselines":
                        merged.setdefault("baselines", {}).update(value)
                    else:
                        merged[name] = value
        return changes

    def _sync_apply(self) -> None:
        count = self.ticked_count()
        self._apply_btn.setText(f"Apply {count} change{'' if count == 1 else 's'}")
        self._apply_btn.setEnabled(count > 0)

    def _on_apply(self) -> None:
        count = self.ticked_count()
        status = (
            f"✓ Applied {count} estimate{'' if count == 1 else 's'}"
            f" · contrast {_contrast(self._estimate.contrast)}"
        )
        self.applied.emit(self.changes(), status)
        self.accept()
