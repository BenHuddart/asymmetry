"""Build Fit Function dialog with a library search in progress.

Drives the two-panel builder with an existing
``Oscillatory + Exponential + Constant`` model on the right (structured
rows), while the left-hand component library shows a live search for
``"kt"`` — an alias that ranks the Kubo-Toyabe family (``StaticGKT_ZF``,
``LongitudinalFieldKT``, ``DynamicGaussianKT``, ...) above unrelated
components, illustrating the searchable library over calculator-keypad
component entry. The dialog is shown non-modally so ``widget.grab``
captures its contents.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from asymmetry.core.fitting.composite import CompositeModel

from ._base import CaptureContext, Scenario, register


class CompositeModelsBuilderScenario(Scenario):
    name = "composite_models_builder"
    description = "Build Fit Function dialog with a library search ranking the KT family."
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

        # Type an alias into the component library search so the screenshot
        # shows ranked results (ranked hits + muted "alias" annotations)
        # rather than the empty-query category view.
        dialog._library.set_search_text("kt")
        _pump_events(100)

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
