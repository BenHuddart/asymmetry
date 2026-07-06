"""Detector Layout editor: EMU's Longitudinal (two-group F/B) preset.

Drives the visual Detector Layout editor with EMU's built-in
``Longitudinal`` preset — the standard two-group forward/backward split the
:doc:`/workflows/calibration_grouping_emu` tutorial builds its silver
calibration on. The schematic draws EMU's forward and backward detector rings
split into the two groups, the arrangement that starting point seeds before
:math:`\\alpha` is calibrated. Companion to
:doc:`/workflows/calibration_grouping_emu` and
:doc:`/reference/detector_grouping`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import get_instrument_layout

from ._base import CaptureContext, Scenario, register


class EmuLongitudinalLayoutScenario(Scenario):
    name = "emu_longitudinal_layout"
    description = (
        "Detector Layout editor on EMU's Longitudinal preset, the two-group "
        "forward/backward split used for the silver calibration."
    )
    size = (1100, 640)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        layout = get_instrument_layout("EMU")
        preset = layout.presets["Longitudinal"]
        groups = {gid: list(gd.detector_ids) for gid, gd in preset.groups.items()}
        group_names = {gid: gd.name for gid, gd in preset.groups.items()}
        projections = [proj.to_payload() for proj in preset.projections]

        dialog = DetectorLayoutDialog(
            layout,
            groups,
            group_names=group_names,
            initial_preset_name="Longitudinal",
            forward_group=preset.forward_group,
            backward_group=preset.backward_group,
            projections=projections,
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


register(EmuLongitudinalLayoutScenario())
