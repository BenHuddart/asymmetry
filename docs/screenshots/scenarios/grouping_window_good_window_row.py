"""Grouping window: the good-window mode selector and its provenance line (D8).

Crops to the ``t_good Offset`` row (mode selector, offset spinbox), the
``Last Good Bin`` row beneath it, and the always-on read-only line beneath
*that* pair which carries the offset in time and, once Manual is selected,
the file's own values alongside the edited ones. The synthesised run is a
two-detector continuous source with an explicit file good window (offset 3
bins from t0, closing at bin 32); the scenario switches the selector to
Manual and types a different offset (7 bins) so the line shows both the
edited value and the file's, exactly the comparison Manual mode exists for.
Companion to :doc:`/reference/detector_grouping` and
:doc:`/reference/project_files`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run

from ._base import CaptureContext, Scenario, _optimize_png, register

_N_BINS = 60
_BIN_WIDTH_US = 0.016
_T0_BIN = 10
_PEAK_COUNTS = 6000.0
_BASELINE_COUNTS = 40.0
_N0 = 400.0
_LIFETIME_US = 2.19703
#: The file's own good window: offset 3 bins from t0, closing at bin 32 —
#: distinct from the Manual offset (below) so the provenance line's "File: …"
#: half reads differently from its "≈ … after t0" half.
_FILE_FIRST_GOOD_BIN = _T0_BIN + 3
_FILE_LAST_GOOD_BIN = 32
#: The offset typed into the Manual spinbox once the mode is switched —
#: chosen so ``offset * bin_width`` prints a clean, non-round time
#: (7 * 0.016 = 0.112 µs) and differs from the file's own offset of 3.
_MANUAL_OFFSET = 7


def _make_two_detector_run(seed: int = 2026) -> MuonDataset:
    """A two-detector continuous-source run with an explicit file good window.

    Built directly from :class:`Histogram`/:class:`Run` (not through a
    loader), as :mod:`grouping_window_t0_row` does, so the header values are
    exactly what this scenario asks for rather than whatever a loader would
    infer.
    """
    rng = np.random.default_rng(seed)
    bins = np.arange(_N_BINS)
    tail = np.where(
        bins > _T0_BIN,
        _N0 * np.exp(-(bins - _T0_BIN) * _BIN_WIDTH_US / _LIFETIME_US),
        0.0,
    )
    clean = np.full(_N_BINS, _BASELINE_COUNTS, dtype=float)
    clean[_T0_BIN] = _PEAK_COUNTS
    clean += tail
    histograms = [
        Histogram(
            counts=rng.poisson(clean).astype(float),
            bin_width=_BIN_WIDTH_US,
            t0_bin=_T0_BIN,
            good_bin_start=_FILE_FIRST_GOOD_BIN,
            good_bin_end=_FILE_LAST_GOOD_BIN,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": _T0_BIN,
        "t_good_offset": _FILE_FIRST_GOOD_BIN - _T0_BIN,
        "first_good_bin": _FILE_FIRST_GOOD_BIN,
        "last_good_bin": _FILE_LAST_GOOD_BIN,
        "bin_index_base": 0,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "included_groups": {1: True, 2: True},
    }
    run = Run(
        run_number=9002,
        histograms=histograms,
        metadata={
            "title": "PSI good-window calibration",
            "facility": "PSI",
            "temperature": 250.0,
            "field": 0.0,
        },
        grouping=grouping,
    )
    time = (bins[_T0_BIN:] - _T0_BIN) * _BIN_WIDTH_US
    return MuonDataset(
        time=time,
        asymmetry=np.zeros_like(time),
        error=np.ones_like(time),
        metadata={
            "run_number": run.run_number,
            "title": run.metadata["title"],
            "temperature": 250.0,
            "field": 0.0,
        },
        run=run,
    )


class GroupingWindowGoodWindowRowScenario(Scenario):
    name = "grouping_window_good_window_row"
    description = (
        "The grouping window's good-window mode selector, cropped to the "
        "t_good Offset and Last Good Bin rows and the provenance line beneath them."
    )
    size = (620, 260)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = _make_two_detector_run()
        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*dialog.preferred_window_size())
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The t0 row's detected value lands via a debounced background scan
        # (D11) that can still resize the dialog after it arrives — wait for
        # it to settle before touching the good-window controls below it, or
        # the rows this scenario crops could still move.
        _pump_until(lambda: _t0_line_settled(dialog))
        _pump_events(120)

        # Switch to Manual and type an offset that differs from the file's
        # own (3 bins) so the provenance line shows both halves distinctly.
        combo = dialog._good_window_mode_combo
        combo.setCurrentIndex(combo.findData("manual"))
        dialog._t_good_offset_spin.setValue(_MANUAL_OFFSET)
        _pump_events(80)

        row = dialog._good_window_row_widget
        last_good_spin = dialog._last_good_spin
        line = dialog._good_window_line_row_widget
        row_rect = QRect(row.mapTo(dialog, QPoint(0, 0)), row.size())
        last_good_rect = QRect(
            last_good_spin.mapTo(dialog, QPoint(0, 0)), last_good_spin.size()
        )
        line_rect = QRect(line.mapTo(dialog, QPoint(0, 0)), line.size())
        crop = row_rect.united(last_good_rect).united(line_rect).adjusted(-16, -12, 16, 4)
        # Stop at the grouping column's edge: the line's row spans the whole
        # column, so a crop a few pixels wider shows the corrections cards.
        column = dialog._grouping_scroll.viewport()
        crop.setRight(min(crop.right(), column.mapTo(dialog, QPoint(column.width(), 0)).x()))
        # And start at its left edge, so the row labels are never clipped.
        crop.setLeft(column.mapTo(dialog, QPoint(0, 0)).x())
        crop = crop.intersected(dialog.rect())

        pix = dialog.grab(crop)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)

        # The mode switch and the typed offset dirty the draft; clear the
        # guard explicitly rather than let the close-time "Discard changes?"
        # prompt fire and rely on auto-dismissal (docs/README.md § "Never
        # modal").
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _t0_line_settled(dialog) -> bool:
    text = dialog._t0_detected_label.text()
    return bool(text) and "…" not in text


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 10_000) -> None:
    """Pump a nested event loop until *predicate* holds (or the timeout lapses).

    The t0 detection lands via a queued cross-thread signal (``TaskRunner``),
    so the loop must be entered for the callback to run; the timeout is only
    a backstop.
    """
    if predicate():
        return
    loop = QEventLoop()
    check = QTimer()
    check.timeout.connect(lambda: loop.quit() if predicate() else None)
    check.start(10)
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(int(timeout_ms))
    loop.exec()
    check.stop()
    guard.stop()


register(GroupingWindowGoodWindowRowScenario())
