"""Grouping window: profile editor with its live asymmetry preview pane.

Opens the Grouping window directly on a synthesised YBCO TF run (four
detector histograms grouped one-per-group), showing the profile selector,
preset chip, scope panel, and the debounced live forward/backward asymmetry
preview together. Companion to :doc:`/reference/detector_grouping` and
:doc:`/reference/grouping_calibration`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ..data import make_ybco_knight_grouped
from ._base import CaptureContext, Scenario, register


class GroupingWindowProfileEditorScenario(Scenario):
    name = "grouping_window_profile_editor"
    description = "Grouping window profile editor, with the live asymmetry preview pane."
    size = (1180, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        dataset = make_ybco_knight_grouped()

        dialog = GroupingDialog([dataset], selected_run_number=int(dataset.run_number))
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        # The live preview reduces on a worker thread (TaskRunner) after a
        # debounce timer; give both time to fire before grabbing.
        _pump_events(500)

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


register(GroupingWindowProfileEditorScenario())
