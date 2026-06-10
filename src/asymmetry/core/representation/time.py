"""Time-domain representations: F-B asymmetry and individual groups.

Both are driven by the detector grouping carried on the :class:`Run`.  The F-B
asymmetry representation reduces the forward/backward groups to a single
asymmetry curve; the groups representation yields one lifetime-corrected count
curve per included detector group (the input to the joint multi-group fit).

The reduction here delegates to the existing core transform helpers.  Full
parity with the GUI reduction (deadtime, background, period mode) is layered in
when representations are wired into the main window; the recipe is the seam.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
from asymmetry.core.representation.base import Representation, RepresentationType
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.grouping import effective_grouping, group_forward_backward
from asymmetry.core.transform.rebin import binned_fb_asymmetry, rebin, resolve_binning_mode


class TimeFBAsymmetry(Representation):
    """Forward-backward asymmetry curve for one run.

    Recipe keys (all optional; fall back to ``run.grouping``)::

        {"grouping_ref": {...grouping overrides...}}
    """

    rep_type = RepresentationType.TIME_FB_ASYMMETRY

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        histograms = list(run.histograms)
        if not histograms:
            raise ValueError("F-B asymmetry requires detector histograms.")
        grouping = effective_grouping(run, self.recipe.get("grouping_ref"))
        fb = group_forward_backward(histograms, grouping)
        forward_gid = fb.forward_gid
        backward_gid = fb.backward_gid
        alpha = fb.alpha
        common_t0 = fb.common_t0

        n = min(fb.forward.size, fb.backward.size)
        forward = fb.forward[:n]
        backward = fb.backward[:n]

        try:
            first_good = max(0, int(grouping.get("first_good_bin", 0)))
        except (TypeError, ValueError):
            first_good = 0
        try:
            last_good = int(grouping.get("last_good_bin", n - 1))
        except (TypeError, ValueError):
            last_good = n - 1
        last_good = min(last_good, n - 1)
        if first_good > last_good:
            first_good, last_good = 0, n - 1

        bin_width = float(histograms[0].bin_width)
        binning_mode, _, _ = resolve_binning_mode(grouping)
        if binning_mode != "fixed":
            time, asymmetry, error = binned_fb_asymmetry(
                forward,
                backward,
                grouping=grouping,
                common_t0=common_t0,
                bin_width_us=bin_width,
                alpha=alpha,
                first_good_bin=first_good,
                last_good_bin=last_good,
            )
        else:
            asymmetry, error = compute_asymmetry(forward, backward, alpha)
            asymmetry = asymmetry[first_good : last_good + 1]
            error = error[first_good : last_good + 1]
            axis_start = first_good - int(common_t0)
            time = (np.arange(asymmetry.size, dtype=float) + float(axis_start)) * bin_width

            try:
                bunch_factor = max(1, int(grouping.get("bunching_factor", 1)))
            except (TypeError, ValueError):
                bunch_factor = 1
            if bunch_factor > 1 and asymmetry.size > 0:
                time, asymmetry, error = rebin(time, asymmetry, error, bunch_factor)

        metadata = dict(run.metadata)
        metadata.update(
            {
                "run_number": run.run_number,
                "run_label": str(run.run_number),
                "plot_domain": "time",
                "x_label": "Time (μs)",
                "y_label": "Asymmetry",
                "forward_group": forward_gid,
                "backward_group": backward_gid,
                "alpha": alpha,
            }
        )
        return [
            MuonDataset(
                time=np.asarray(time, dtype=float),
                asymmetry=np.asarray(asymmetry, dtype=float),
                error=np.asarray(error, dtype=float),
                metadata=metadata,
                run=run,
            )
        ]


class TimeGroups(Representation):
    """Per-group lifetime-corrected count curves for one run.

    Fitting this representation is always the joint multi-group fit.

    Recipe keys (optional)::

        {"t_min": float | None, "t_max": float | None}
    """

    rep_type = RepresentationType.TIME_GROUPS

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        if not run.histograms:
            raise ValueError("Individual-groups representation requires detector histograms.")
        # build_grouped_time_domain_datasets reads grouping/histograms from .run.
        source = MuonDataset(
            time=np.asarray([], dtype=float),
            asymmetry=np.asarray([], dtype=float),
            error=np.asarray([], dtype=float),
            metadata={},
            run=run,
        )
        t_min = self.recipe.get("t_min")
        t_max = self.recipe.get("t_max")
        datasets = build_grouped_time_domain_datasets(
            source,
            t_min=None if t_min is None else float(t_min),
            t_max=None if t_max is None else float(t_max),
        )
        for dataset in datasets:
            dataset.metadata.setdefault("plot_domain", "time")
            dataset.run = run
        return datasets
