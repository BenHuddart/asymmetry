"""Synthetic unpolarised decay with a shared, fitted count-background level.

Poisson means are muon decay plus flat counts per acquisition bin; the time
axis is in μs. Fixed bunching exposes the shared level's contribution to the
asymmetry error bars without an oscillation in the tail-fit window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.utils.constants import MUON_LIFETIME_US

from ._base import CaptureContext, Scenario, _optimize_png, register
from .grouping_window_good_window_row import _pump_events, _pump_until

_SEED = 20261031


class BackgroundBinnedUncertaintyScenario(Scenario):
    name = "background_binned_uncertainty"
    description = "Tail-fit background and fixed bunching in the asymmetry preview."
    size = (1180, 780)
    requires_fit = True

    def capture(self, ctx: CaptureContext) -> Path:
        from asymmetry.gui.windows.grouping.dialog import GroupingDialog

        width, t0, n_bins = 0.05, 20, 340
        time = (np.arange(n_bins) - t0) * width
        signal = np.where(time >= 0, 250.0 * np.exp(-time / MUON_LIFETIME_US), 0.0)
        rng = np.random.default_rng(_SEED)
        raw = [rng.poisson(signal + level).astype(float) for level in (8.0, 12.0)]
        run = Run(
            run_number=8101,
            histograms=[Histogram(counts, bin_width=width, t0_bin=t0) for counts in raw],
            metadata={
                "title": "Unpolarised decay with count background (synthetic)",
                "simulation_seed": _SEED,
            },
            grouping={
                "instrument": "generic_layout",
                "groups": {1: [1], 2: [2]},
                "group_names": {1: "Forward", 2: "Backward"},
                "forward_group": 1,
                "backward_group": 2,
                "alpha": 1.0,
                "first_good_bin": t0,
                "last_good_bin": t0 + 159,
                "t0_bin": t0,
                "bunching_factor": 8,
                "background_correction": False,
            },
        )
        asymmetry, error = compute_asymmetry(*raw)
        dataset = MuonDataset(time, 100 * asymmetry, 100 * error, metadata=run.metadata, run=run)
        dialog = GroupingDialog([dataset], selected_run_number=run.run_number)
        dialog.resize(*self.size)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()
        _pump_events(100)
        combo = dialog._background_section._mode_combo
        combo.setCurrentIndex(combo.findData("tail_fit"))
        dialog._pipeline_chips["background"].click()
        dialog._preview_view_buttons["asymmetry"].click()
        pane = dialog._preview_pane
        pane.flush()
        _pump_until(
            lambda: (
                pane._tasks.active_count == 0
                and pane._last_result is not None
                and pane._last_result.counts.background_mode == "tail_fit"
                and pane._last_result.counts.background_level is not None
            )
        )
        if (
            pane._tasks.active_count
            or pane._last_result is None
            or pane._last_result.counts.background_mode != "tail_fit"
            or pane._last_result.counts.background_level is None
        ):
            raise RuntimeError("Background preview did not complete its tail fit")
        _pump_events(80)
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not dialog.grab().save(str(output), "PNG"):
            raise RuntimeError("Could not save the background uncertainty screenshot")
        _optimize_png(output)
        dialog._clear_dirty()
        dialog.close()
        dialog.deleteLater()
        _pump_events(40)
        return output


register(BackgroundBinnedUncertaintyScenario())
