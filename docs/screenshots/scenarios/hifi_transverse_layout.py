"""Detector Layout editor: HiFi's Transverse (Vector) preset with overlaps.

Drives the visual Detector Layout editor directly with HiFi's built-in
``Transverse (Vector)`` preset, whose Left-Right and Top-Bottom detector
groups overlap at their boundary detectors. The schematic renders each
overlapping detector as several thin membership slices rather than a single
solid colour, which is the multi-membership rendering this screenshot is
meant to demonstrate. Companion to :doc:`/reference/detector_grouping`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.instrument import get_instrument_layout

from ._base import CaptureContext, Scenario, register


class HiFiTransverseLayoutScenario(Scenario):
    name = "hifi_transverse_layout"
    description = (
        "Detector Layout editor on HiFi's Transverse (Vector) preset, "
        "showing overlapping group membership as schematic slices."
    )
    size = (1100, 640)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.windows.detector_layout_dialog import DetectorLayoutDialog

        layout = get_instrument_layout("HiFi")
        preset = layout.presets["Transverse (Vector)"]
        groups = {gid: list(gd.detector_ids) for gid, gd in preset.groups.items()}
        group_names = {gid: gd.name for gid, gd in preset.groups.items()}
        projections = [proj.to_payload() for proj in preset.projections]

        dialog = DetectorLayoutDialog(
            layout,
            groups,
            group_names=group_names,
            initial_preset_name="Transverse (Vector)",
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


register(HiFiTransverseLayoutScenario())
