"""A missing reference keeps its background request visible in the preview."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.gui.windows.grouping.dialog import GroupingDialog

from ._base import CaptureContext, Scenario, _optimize_png, register
from .grouping_window_good_window_row import _pump_events, _pump_until


class IncompletePreviewScenario(Scenario):
    name = "grouping_incomplete_preview"
    description = "Missing background reference and disabled Apply."
    size = (1180, 780)

    def capture(self, ctx: CaptureContext) -> Path:
        # Counts are deterministic Poisson means per acquisition bin; time is in µs.
        time = np.arange(200) * 0.02
        run = Run(
            run_number=8102,
            histograms=[Histogram(np.full(200, level), bin_width=0.02) for level in (120.0, 80.0)],
            grouping={
                "instrument": "generic_layout",
                "groups": {1: [1], 2: [2]},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "t0_bin": 0,
                "first_good_bin": 0,
                "last_good_bin": 199,
                "deadtime_correction": False,
                "background_correction": True,
                "background_mode": "reference_run",
                "background_run": {"run_number": 9999},
            },
        )
        dataset = MuonDataset(
            time,
            np.full(200, 20.0),
            np.ones(200),
            metadata={"run_number": 8102, "title": "Missing reference (synthetic)"},
            run=run,
        )
        dialog = GroupingDialog([dataset])
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        pane = dialog._preview_pane
        pane.flush()
        _pump_until(lambda: pane._last_result is not None and not pane.corrections_complete)
        if pane.corrections_complete or dialog._apply_btn.isEnabled():
            raise RuntimeError("Missing reference preview must disable Apply")
        _pump_events(80)
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(output), "PNG"):
            raise RuntimeError("Could not save incomplete preview screenshot")
        _optimize_png(output)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return output


register(IncompletePreviewScenario())
