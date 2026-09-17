"""Grouping window: the file/detected time-zero line and its verdict (D11).

Crops to just the t0 row (mode selector, spinbox, Find t0 button) and the
always-on read-only line beneath it that carries the file t0, the detected
t0 with its strategy and spread, the signed difference, and the verdict — the
same line shown in every mode. The synthesised run is a two-detector
continuous source whose header t0 sits exactly on its own prompt peak, so the
file and detected values agree and the verdict is the quiet "ok" case (no
divergence message). Companion to
:doc:`/reference/data_reduction/t0_search` and
:doc:`/reference/detector_grouping`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run

from ._base import CaptureContext, Scenario, _optimize_png, register

_N_BINS = 300
_BIN_WIDTH_US = 0.01
_T0_BIN = 40
_PEAK_COUNTS = 6000.0
_BASELINE_COUNTS = 40.0
_N0 = 400.0
_LIFETIME_US = 2.19703


def _make_prompt_peak_run(seed: int = 2026) -> MuonDataset:
    """A two-detector continuous-source run with an unambiguous prompt peak.

    Built directly from :class:`Histogram`/:class:`Run` (not through a
    loader) so the header ``t0_bin`` is exactly the counts' true maximum bin:
    the file value and the auto-detect consensus agree by construction,
    giving the deterministic "ok" verdict this screenshot needs. The prompt
    spike (six counters' worth of positrons above the flat pre-t0 baseline)
    sits ~14 sigma above the Poisson noise on the decay tail that follows it,
    so the argmax is unambiguous with or without the noise draw.
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
            good_bin_start=_T0_BIN,
            good_bin_end=_N_BINS - 1,
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
        "t_good_offset": 0,
        "first_good_bin": _T0_BIN,
        "last_good_bin": _N_BINS - 1,
        "bin_index_base": 0,
        "bunching_factor": 1,
        "deadtime_correction": False,
        "included_groups": {1: True, 2: True},
    }
    run = Run(
        run_number=9001,
        histograms=histograms,
        metadata={
            "title": "PSI prompt-peak calibration",
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


class GroupingWindowT0RowScenario(Scenario):
    name = "grouping_window_t0_row"
    description = "The grouping window's file/detected time-zero line, cropped to the t0 row."
    size = (620, 220)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = _make_prompt_peak_run()
        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(1180, 760)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The detected value comes from a debounced background scan keyed on
        # the run's content digest (D11), not this call — wait for it rather
        # than a fixed sleep so the line never grabs mid "Detected: …".
        _pump_until(lambda: _line_settled(dialog))
        _pump_events(120)

        row = dialog._t0_row_widget
        label = dialog._t0_detected_label
        row_rect = QRect(row.mapTo(dialog, QPoint(0, 0)), row.size())
        label_rect = QRect(label.mapTo(dialog, QPoint(0, 0)), label.size())
        crop = row_rect.united(label_rect).adjusted(-16, -12, 16, 4)
        crop = crop.intersected(dialog.rect())

        pix = dialog.grab(crop)
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")
        _optimize_png(out_path)

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _line_settled(dialog) -> bool:
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


register(GroupingWindowT0RowScenario())
