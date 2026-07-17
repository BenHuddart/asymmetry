"""Inline alpha calibration on the grouping window's Corrections tab.

Opens the Grouping window directly on a synthesised YBCO transverse-field run and
selects the **Corrections** tab. Alpha (the detector-balance parameter) is
calibrated **inline** there — a calibration-run picker (weak-TF candidates
highlighted), an estimation method, and an **Estimate α** button — instead of a
separate modal dialog. Pressing Estimate α measures alpha on the *corrected*
forward/backward counts and drives the shared grouping preview (pinned below the
tabs), which overlays the α = 1 "before" ghost against the estimated-α "after"
curve and reports the residual baseline ⟨A⟩. Companion to
:doc:`/reference/detector_grouping` and :doc:`/reference/grouping_calibration`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register


class AlphaCalibrationDialogScenario(Scenario):
    name = "alpha_calibration_dialog"
    description = (
        "Inline alpha calibration in the grouping window's Corrections panel, with the "
        "shared before/after (α = 1 ↔ α̂) asymmetry preview."
    )
    size = (1180, 760)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = make_ybco_knight_grouped()

        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)

        # The inline α estimate runs on a background worker thread, so a single
        # click does not populate the result before the grab. Pump the event loop
        # until the worker's queued finished callback has landed (deterministic,
        # not a fixed sleep) so the captured α result and preview overlay always
        # show the estimate, never the transient "Computing estimate…" state.
        # Show the Corrections tab, where the inline α-calibration controls (run
        # picker, method, Estimate α, result) live alongside deadtime and
        # background — a first-class named tab rather than the foot of a long scroll.
        tabs = getattr(dialog, "_tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(getattr(dialog, "_corrections_tab_index", 1))
            _pump_events(80)

        section = getattr(dialog, "_alpha_section", None)
        if section is not None:
            section._on_estimate()
            _pump_until(lambda: section._tasks.active_count == 0)
            # Let the shared preview redraw the α = 1 ↔ α̂ overlay before grabbing.
            _pump_events(500)
            # Bring the α section into frame. With the adaptive deadtime layout
            # it already sits above the fold in the default (deadtime-off) state,
            # so this is a no-op there and still correct on smaller captures.
            corr_scroll = getattr(dialog, "_corrections_scroll", None)
            if corr_scroll is not None:
                corr_scroll.ensureWidgetVisible(section)
                _pump_events(80)

        pix = dialog.grab()
        out_path = ctx.output_dir / f"{self.name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not pix.save(str(out_path), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {out_path}")

        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return out_path


def _pump_events(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(milliseconds), loop.quit)
    loop.exec()
    QApplication.processEvents()


def _pump_until(predicate, timeout_ms: int = 10_000) -> None:
    """Pump a nested event loop until *predicate* holds (or the timeout lapses).

    The estimate lands via a queued cross-thread signal, so the loop must be
    entered for the callback to run; the timeout is only a backstop.
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


register(AlphaCalibrationDialogScenario())
