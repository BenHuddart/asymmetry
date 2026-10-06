"""Detector Layout editor: the generic ring for an unknown instrument.

A 16-detector run from an instrument Asymmetry has no layout for opens the
editor on ``Generic (16 detectors)``: one ring of equal wedges, numbered
clockwise from the top, with the preset controls disabled. Companion to
:doc:`/reference/detector_grouping`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import generic_layout

from ._base import CaptureContext, Scenario, register


class GenericLayoutScenario(Scenario):
    name = "generic_layout"
    description = (
        "Detector Layout editor on the generic 16-detector ring offered for an "
        "instrument Asymmetry has no layout for."
    )
    size = (1100, 640)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        dialog = DetectorLayoutDialog(
            generic_layout(16),
            {1: list(range(1, 9)), 2: list(range(9, 17))},
            group_names={1: "Forward", 2: "Backward"},
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(200)

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


register(GenericLayoutScenario())
