"""Generate-synthetic-run dialog configured for an EuO precession run.

Drives :class:`~asymmetry.gui.windows.simulate_dialog.SimulateDialog` with no
loaded run, so the *Template run* list offers only the two idealised
instruments. The **ideal pulsed F/B** built-in (an ISIS-style spectrometer,
the geometry EuO is measured on) is selected and its teaching-sensible event
budget pre-fills; the model is set to ``Oscillatory * Exponential`` — a damped
Larmor precession at ν ≈ 22 MHz, the below-Tc EuO signal from the archetype
gallery — with the parameter table populated. This is the "Built-in instrument
templates" / "Generate a synthetic run" walkthrough of
:doc:`/reference/simulation`: go straight from a model to a synthetic run.

No Generate is run (the dialog is never accepted and nothing is emitted), so
the capture shows the pre-Generate configuration deterministically without the
worker thread.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from ._base import CaptureContext, Scenario, register


class SimulateDialogScenario(Scenario):
    name = "simulate_dialog"
    description = (
        "Generate Synthetic Run dialog with the ideal pulsed F/B template and an "
        "EuO Oscillatory*Exponential model."
    )
    size = (600, 470)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.core.fitting.composite import CompositeModel
        from asymmetry.gui.windows.simulate_dialog import SimulateDialog

        # No loaded runs: the Template run list falls back to the two built-in
        # idealised instruments, and the dialog opens on the first (ideal
        # pulsed F/B), pre-filling its 40 MEv / zero-background defaults.
        dialog = SimulateDialog([])
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(120)

        # Below-Tc EuO ferromagnetic precession: a damped Larmor oscillation at
        # ν ≈ 22 MHz (matching the archetype-gallery EuO run), 22% amplitude,
        # and a slow relaxation. These are the values the fit panel / archetype
        # would seed; the dialog works in percent.
        dialog._model = CompositeModel(["Oscillatory", "Exponential"], operators=["*"])
        dialog._param_values = {
            "A_1": 22.0,
            "frequency": 22.0,
            "phase": 0.0,
            "Lambda": 0.2,
        }
        dialog._refresh_model_view()
        # The four-row model keeps the parameter table compact — cap its height
        # so the dialog does not stretch it into a large empty region (an empty
        # table reads as a broken program in a docs figure).
        dialog._param_table.setMaximumHeight(168)
        _pump_events(120)

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


register(SimulateDialogScenario())
