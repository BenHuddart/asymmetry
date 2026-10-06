"""Auto-detect's review: each estimate beside the current value, ticked to apply (D6).

- Rows still on defaults or earlier estimates start ticked; a row holding a
  typed value never does.
- An estimate with contrast below :data:`MIN_CONTRAST` cannot be applied.
- Setup rows (ν_RF, sense, φ_RF) come from the runs together and write to every
  run; run rows (b_x, b_y, a_y/a_x) come from each run alone.
- Nothing changes until Apply, which reports the ticked values through
  :attr:`FrameReviewDialog.applied` for the host to write as estimates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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

from asymmetry.core.transform.rotating_frame import FrameEstimate, Provenance, RotatingFrame
from asymmetry.gui.styles import tokens
from asymmetry.gui.styles.widgets import build_primary_button_qss, make_section_header

__all__ = ["MIN_CONTRAST", "FrameReviewDialog"]

#: Contrast below which an estimate is too weak to apply.
MIN_CONTRAST = 3.0

FOOTER_NOTE = (
    "Ticked rows are written as estimates. Rows on defaults or earlier estimates start "
    "ticked; a row holding a typed value never does. An estimate with contrast below 3 "
    "cannot be applied."
)

_FORMATS: dict[str, Callable[[object], str]] = {
    "frequency_mhz": lambda v: f"{v:.6g} MHz",
    "sense": lambda v: "+1" if v > 0 else "−1",
    "rf_phase_deg": lambda v: f"{v:.1f}°",
    "baseline_x": lambda v: f"{v:.2f}",
    "baseline_y": lambda v: f"{v:.2f}",
    "gain": lambda v: f"{v:.3f}",
}


def _contrast(value: float) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


@dataclass(frozen=True)
class _Row:
    """One tickable proposal: these fields' estimates, for these runs."""

    title: str
    values: dict[str, object]
    runs: tuple[int, ...]
    contrast: float
    note: str = ""


class FrameReviewDialog(QDialog):
    """Review an Auto-detect estimate against the runs' current frames."""

    #: ``({run: {field: value}}, status text)`` when the user applies.
    applied = Signal(object, str)

    def __init__(
        self,
        estimate: FrameEstimate,
        frames: Mapping[int, RotatingFrame],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto-detect rotating frame")
        self._frames = dict(frames)
        self._estimate = estimate
        runs = tuple(self._frames)
        setup = [
            _Row(
                "ν_RF",
                {"frequency_mhz": estimate.frequency_mhz},
                runs,
                estimate.contrast,
                self._frequency_note(estimate.frequency_mhz),
            ),
            _Row("Sense", {"sense": estimate.sense}, runs, estimate.contrast),
            _Row("φ_RF", {"rf_phase_deg": estimate.rf_phase_deg}, runs, estimate.contrast),
        ]
        per_run = [
            _Row(
                f"Run {run.run_key}",
                {"baseline_x": run.baseline_x, "baseline_y": run.baseline_y, "gain": run.gain},
                (run.run_key,),
                run.contrast,
                f"contrast {_contrast(run.contrast)}",
            )
            for run in estimate.runs
            if run.run_key in self._frames
        ]
        self._ticks: list[tuple[QCheckBox, _Row]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        grid.addWidget(make_section_header("Setup"), 0, 0)
        grid.addWidget(
            self._muted(f"from the runs together · contrast {_contrast(estimate.contrast)}"),
            0,
            1,
            1,
            4,
        )
        line = 1
        for row in setup:
            self._add_row(grid, line, row, span=3)
            line += 1
        grid.addWidget(make_section_header("Runs"), line, 0)
        for column, heading in enumerate(("b_x (%)", "b_y (%)", "a_y/a_x"), start=1):
            grid.addWidget(self._muted(heading), line, column)
        line += 1
        for row in per_run:
            self._add_row(grid, line, row, span=1)
            line += 1
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

    def _current(self, row: _Row, name: str) -> str:
        values = {getattr(self._frames[run], name) for run in row.runs}
        return _FORMATS[name](values.pop()) if len(values) == 1 else "mixed"

    def _frequency_note(self, estimate_mhz: float) -> str:
        """ν_RF's estimate is a check on the typed value, so its note is the agreement."""
        values = {frame.frequency_mhz for frame in self._frames.values()}
        if len(values) != 1:
            return ""
        current = values.pop()
        percent = abs(estimate_mhz - current) / current * 100.0
        return f"agrees within {percent:.2g} %" if percent < 1.0 else f"differs by {percent:.2g} %"

    def _add_row(self, grid: QGridLayout, line: int, row: _Row, *, span: int) -> None:
        tick = QCheckBox(row.title)
        typed = any(
            self._frames[run].provenance[name] is Provenance.TYPED
            for run in row.runs
            for name in row.values
        )
        weak = row.contrast < MIN_CONTRAST
        tick.setChecked(not typed and not weak)
        tick.setEnabled(not weak)
        tick.toggled.connect(self._sync_apply)
        grid.addWidget(tick, line, 0)
        for column, (name, value) in enumerate(row.values.items(), start=1):
            cell = QLabel(f"{self._current(row, name)} → {_FORMATS[name](value)}")
            cell.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(cell, line, column, 1, span)
        if weak:
            note = QLabel(f"contrast {_contrast(row.contrast)} < 3 — cannot apply")
            note.setStyleSheet(f"color: {tokens.WARN};")
        else:
            note = self._muted(("typed · " if typed else "") + row.note)
        grid.addWidget(note, line, 4)
        self._ticks.append((tick, row))

    def ticked_count(self) -> int:
        return sum(tick.isChecked() for tick, _row in self._ticks)

    def changes(self) -> dict[int, dict[str, object]]:
        """``{run: {field: value}}`` of the ticked rows."""
        changes: dict[int, dict[str, object]] = {}
        for tick, row in self._ticks:
            if tick.isChecked():
                for run in row.runs:
                    changes.setdefault(run, {}).update(row.values)
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
