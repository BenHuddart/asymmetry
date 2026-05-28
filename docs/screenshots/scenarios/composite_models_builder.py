"""Fit Function Builder dialog populated with a composite model.

Drives the composite-model expression editor with
``Oscillatory + Exponential + Constant`` (or the fraction-group variant
``(Oscillatory + Exponential){frac} + Constant``) over the EuO critical-
region dataset, illustrating the GUI's free-form expression syntax. The
dialog is shown non-modally so ``widget.grab`` captures its contents.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel

from ._base import CaptureContext, Scenario, register


class CompositeModelsBuilderScenario(Scenario):
    name = "composite_models_builder"
    description = "Fit Function Builder dialog with an Oscillatory+Exponential+Constant model."
    size = (820, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.panels.fit_function_builder import (
            FitFunctionBuilderDialog,
        )

        initial_model = CompositeModel(
            ["Oscillatory", "Exponential", "Constant"], operators=["+", "+"]
        )
        dialog = FitFunctionBuilderDialog(initial_model=initial_model)
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


register(CompositeModelsBuilderScenario())
