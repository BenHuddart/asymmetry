"""Fit Function Builder showing a fraction-group expression.

Drives the composite-model expression editor with
``(Oscillatory + Exponential){frac} + Constant`` — the canonical
"two-component shared-amplitude budget" pattern useful for
muonium-pair populations, magnetic-volume-fraction analyses, and
critical-region composites. Companion to
:doc:`/user_guide/composite_models`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel

from ._base import CaptureContext, Scenario, register


class CompositeFractionsDialogScenario(Scenario):
    name = "composite_fractions_dialog"
    description = "Fit Function Builder with a fraction-group expression set."
    size = (840, 720)

    def capture(self, ctx: CaptureContext) -> Path:  # noqa: D401
        from asymmetry.gui.panels.fit_function_builder import (
            FitFunctionBuilderDialog,
        )

        # Two-component fraction group plus a separate Constant.
        # component_names: [Oscillatory, Exponential, Constant]
        # operators: ["+", "+"]
        # open/close parens with brace marker -> fraction_groups=[(0, 1)]
        initial_model = CompositeModel(
            ["Oscillatory", "Exponential", "Constant"],
            operators=["+", "+"],
            open_parentheses=[1, 0, 0],
            close_parentheses=[0, 1, 0],
            fraction_groups=[(0, 1)],
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


register(CompositeFractionsDialogScenario())
