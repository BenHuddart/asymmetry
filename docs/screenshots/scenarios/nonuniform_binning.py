"""Show whole-bin membership and retained support on a coarse synthetic grid."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.models import MODELS
from asymmetry.core.simulate import BUILTIN_TEMPLATES, simulate_run
from asymmetry.core.transform.rebin import BinningSpec
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

from ._base import CaptureContext, Scenario, register
from .bunching_comparison import _compose, _grab_plot


class NonuniformBinningScenario(Scenario):
    name = "nonuniform_binning"
    description = "Actual midpoint times and retained tails on a 0.1 μs native grid."
    size = (1600, 540)

    def capture(self, ctx: CaptureContext) -> Path:
        template = replace(
            BUILTIN_TEMPLATES["ideal_pulsed_fb"],
            n_detectors=2,
            n_bins=15,
            bin_width_us=0.1,
            t0_bin=0,
            forward_detectors=(1,),
            backward_detectors=(2,),
        ).build()
        run = simulate_run(
            template,
            MODELS["ExponentialRelaxation"].function,
            {"A0": 22.0, "Lambda": 0.5},
            total_events=200_000,
            seed=107,
            run_number=9507,
        )
        grouping = run.grouping | {"first_good_bin": 2, "last_good_bin": 13}
        panels = []
        for mode, label in (
            ("fixed", "Fixed ×5"),
            ("variable", "Variable"),
            ("constant_error", "Constant error"),
        ):
            result = reduce_grouped_asymmetry(
                histograms=run.histograms,
                grouping=grouping,
                forward_idx=[0],
                backward_idx=[1],
                alpha=1,
                use_deadtime=False,
                deadtime_mode="off",
                use_background=False,
                binning=BinningSpec(mode=mode, factor=5, bin0_us=0.12, bin10_us=0.25),
            )
            request = replace(
                run,
                grouping=grouping
                | {
                    "binning_mode": mode,
                    "bunching_factor": 5,
                    "bin0_us": 0.12,
                    "bin10_us": 0.25,
                },
            )
            merged = reduce_run(request, ReductionSettings(rebin=2))
            direct_merge = reduce_grouped_asymmetry(
                histograms=run.histograms,
                grouping=grouping,
                forward_idx=[0],
                backward_idx=[1],
                alpha=1,
                use_deadtime=False,
                deadtime_mode="off",
                use_background=False,
                binning=BinningSpec(mode=mode, factor=5, bin0_us=0.12, bin10_us=0.25, merge=2),
            )
            for key in ("time", "asymmetry", "error"):
                np.testing.assert_array_equal(getattr(merged, key), getattr(direct_merge, key))
            support = (
                "complete bins"
                if mode == "fixed"
                else f"bins, {result.time[0]:.2f}–{result.time[-1]:.2f} μs"
            )
            title = f"{label}: {result.time.size} {support}"
            dataset = MuonDataset(
                result.time, result.asymmetry, result.error, metadata={"title": title}
            )
            panels.append(_grab_plot(dataset, caption=title, size=(600, 640)))
        output = ctx.output_dir / f"{self.name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not _compose(panels).save(str(output), "PNG"):
            raise RuntimeError(f"Failed to save screenshot to {output}")
        return output


register(NonuniformBinningScenario())
