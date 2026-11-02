"""Exclusions preserve authored channels and expose an unavailable empty group."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.utils.constants import MUON_LIFETIME_US
from asymmetry.gui.windows.grouping.dialog import GroupingDialog

from ._base import CaptureContext, Scenario, _optimize_png, register
from .grouping_window_good_window_row import _pump_events, _pump_until


class MembershipScenario(Scenario):
    size = (1180, 780)

    def capture(self, ctx: CaptureContext) -> Path:
        time = np.arange(200) * 0.02
        histograms = [
            Histogram(level * np.exp(-time / MUON_LIFETIME_US), bin_width=0.02)
            for level in (120.0, 50.0, 80.0)
        ]
        run = Run(
            run_number=8103,
            histograms=histograms,
            grouping={
                "instrument": "generic_layout",
                "groups": {1: [1, 2], 2: [2, 3]},
                "forward_group": 1,
                "backward_group": 2,
                "excluded_detectors": [2],
                "alpha": 1.0,
                "t0_bin": 0,
                "first_good_bin": 0,
                "last_good_bin": 199,
            },
        )
        dataset = MuonDataset(
            time,
            np.zeros(200),
            np.ones(200),
            metadata={"run_number": 8103, "title": "Synthetic channel selection"},
            run=run,
        )
        dialog = GroupingDialog([dataset])
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        pane = dialog._preview_pane
        pane.flush()
        _pump_until(lambda: pane._last_result is not None)
        if self.unavailable:
            dialog._exclude_edit.setText("1,2")
            dialog._refresh_preview()
            pane.flush()
            _pump_until(lambda: not pane.corrections_complete)
            if dialog._apply_btn.isEnabled():
                raise RuntimeError("An empty group must disable Apply")
        elif pane._last_result.facts.member_counts != (1, 1):
            raise RuntimeError("The preview must report effective channel membership")
        _pump_events(80)
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(output), "PNG"):
            raise RuntimeError("Could not save membership screenshot")
        _optimize_png(output)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return output


class ExclusionScenario(MembershipScenario):
    name = "grouping_membership_exclusion"
    description = "Synthetic excluded channel and effective member counts."
    unavailable = False


class UnavailableScenario(MembershipScenario):
    name = "grouping_membership_unavailable"
    description = "Synthetic empty group with disabled Apply."
    unavailable = True


register(ExclusionScenario())
register(UnavailableScenario())
