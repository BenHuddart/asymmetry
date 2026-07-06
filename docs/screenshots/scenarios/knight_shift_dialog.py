"""Knight Shift conversion dialog in a realistic angle-scan state.

Opens the Knight Shift conversion dialog (:class:`~asymmetry.gui.panels.
knight_shift_dialog.KnightShiftDialog`) directly, in the state it takes on
during an angle-resolved Knight-shift scan: two fitted precession-frequency
components to convert, the **Applied field** reference selected (ν_ref =
γ_µ·B, no reference line needed), and a component-crossing warning — the two
sites' K(θ) branches cross at the magic angles, so the raw component labels
can swap identity there. Companion to :doc:`/workflows/knight_shift_angle`
(step 3, "Convert to the Knight shift").
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ._base import CaptureContext, Scenario, register


class KnightShiftDialogScenario(Scenario):
    name = "knight_shift_dialog"
    description = (
        "Knight Shift conversion dialog: Applied-field reference, two frequency "
        "components, and a component-crossing warning from an angle scan."
    )
    size = (460, 520)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.fitting.knight_shift import (
            REFERENCE_APPLIED_FIELD,
            KnightShiftConfig,
            KnightShiftUnit,
        )
        from asymmetry.gui.panels.knight_shift_dialog import KnightShiftDialog

        # A two-site angle scan yields two precession-frequency components; the
        # branches cross at the two magic angles, so crossing_count = 2 fires
        # the label-swap warning the docs describe.
        config = KnightShiftConfig(
            enabled=True,
            reference_mode=REFERENCE_APPLIED_FIELD,
            unit=KnightShiftUnit.PERCENT,
        )
        dialog = KnightShiftDialog(
            available_components=["frequency_1", "frequency_2"],
            config=config,
            crossing_count=2,
        )
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(150)

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


register(KnightShiftDialogScenario())
