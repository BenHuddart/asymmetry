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
from asymmetry.core.transform.grouping import apply_grouping_aligned, common_t0_for_groups
from asymmetry.core.transform.rebin import rebin


def _resolve_group_indices(groups: dict, group_id: int) -> list[int]:
    """Return zero-based detector indices for *group_id*.

    Grouping entries are 1-based detector numbers (matching the convention in
    ``fourier.grouped`` / ``grouped_time_domain``); they are converted to
    zero-based histogram indices here.  Group keys may be ``int`` or ``str``.
    """
    entries = groups.get(group_id)
    if entries is None:
        entries = groups.get(str(group_id))
    if not isinstance(entries, list):
        return []
    indices: list[int] = []
    for value in entries:
        detector = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            indices.append(max(0, int(detector) - 1))
        except (TypeError, ValueError):
            continue
    return indices


def _effective_grouping(run: Run, recipe: dict) -> dict:
    """Merge the run grouping with any recipe overrides (recipe wins)."""
    base = dict(run.grouping) if isinstance(run.grouping, dict) else {}
    override = recipe.get("grouping_ref")
    if isinstance(override, dict):
        base.update(override)
    return base


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
        grouping = _effective_grouping(run, self.recipe)
        groups = grouping.get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ValueError("F-B asymmetry requires a detector grouping definition.")

        forward_gid = int(grouping.get("forward_group", 1))
        backward_gid = int(grouping.get("backward_group", 2))
        forward_indices = _resolve_group_indices(groups, forward_gid)
        backward_indices = _resolve_group_indices(groups, backward_gid)
        if not forward_indices or not backward_indices:
            raise ValueError("Forward/backward groups do not reference any detectors.")

        try:
            alpha = float(grouping.get("alpha", 1.0))
        except (TypeError, ValueError):
            alpha = 1.0

        common_t0 = common_t0_for_groups(histograms, forward_indices, backward_indices)
        forward = apply_grouping_aligned(histograms, forward_indices, common_t0_bin=common_t0)
        backward = apply_grouping_aligned(histograms, backward_indices, common_t0_bin=common_t0)
        n = min(forward.size, backward.size)
        forward = forward[:n]
        backward = backward[:n]

        asymmetry, error = compute_asymmetry(forward, backward, alpha)

        try:
            first_good = max(0, int(grouping.get("first_good_bin", 0)))
        except (TypeError, ValueError):
            first_good = 0
        try:
            last_good = int(grouping.get("last_good_bin", asymmetry.size - 1))
        except (TypeError, ValueError):
            last_good = asymmetry.size - 1
        last_good = min(last_good, asymmetry.size - 1)
        if first_good > last_good:
            first_good, last_good = 0, asymmetry.size - 1

        asymmetry = asymmetry[first_good : last_good + 1]
        error = error[first_good : last_good + 1]

        bin_width = float(histograms[0].bin_width)
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
