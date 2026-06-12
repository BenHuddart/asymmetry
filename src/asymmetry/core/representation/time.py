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
from asymmetry.core.maxent import (
    MaxEntConfig,
    ReconstructedGroup,
    build_maxent_input,
    reconstruct_group_signals,
    run_cycles,
)
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
    """Per-group count curves for one run (lifetime-corrected by default).

    Fitting this representation is always the joint multi-group fit.

    Recipe keys (optional)::

        {"t_min": float | None, "t_max": float | None,
         "lifetime_corrected": bool}

    With ``lifetime_corrected`` false the raw detector counts are returned
    (the GUI's Raw counts view; Poisson statistics are exact on them).
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
            lifetime_corrected=bool(self.recipe.get("lifetime_corrected", True)),
        )
        for dataset in datasets:
            dataset.metadata.setdefault("plot_domain", "time")
            dataset.run = run
        return datasets


def build_maxent_reconstruction_datasets(
    reconstructions: dict[int, ReconstructedGroup],
    run: Run,
) -> list[MuonDataset]:
    """Package per-group MaxEnt reconstructions as plottable overlay datasets.

    Each group becomes one :class:`MuonDataset` whose ``asymmetry`` is the
    observed (normalised) signal and whose ``error`` is σ; the forward-model
    reconstruction and the weighted residual ride along in ``metadata`` under
    ``maxent_model`` / ``maxent_residual`` for the overlay renderer.  The arrays
    are transient display state (representations persist recipes, not arrays).
    """
    datasets: list[MuonDataset] = []
    for group_id in sorted(reconstructions):
        recon = reconstructions[group_id]
        metadata = {
            "run_number": run.run_number,
            "run_label": f"{run.run_number} · {recon.group_name}",
            "plot_domain": "time",
            "x_label": "Time (μs)",
            "y_label": "Reconstruction (a.u.)",
            "group_id": int(recon.group_id),
            "group_name": str(recon.group_name),
            "maxent_reconstruction": True,
            "maxent_model": np.asarray(recon.model, dtype=float),
            "maxent_residual": np.asarray(recon.residual, dtype=float),
            "maxent_group_chi2": float(recon.chi2),
            "maxent_group_n_obs": int(recon.n_obs),
        }
        datasets.append(
            MuonDataset(
                time=np.asarray(recon.time_us, dtype=float),
                asymmetry=np.asarray(recon.data, dtype=float),
                error=np.asarray(recon.sigma, dtype=float),
                metadata=metadata,
                run=run,
            )
        )
    return datasets


class TimeMaxEntReconstruction(Representation):
    """Per-group MaxEnt time-domain reconstruction overlay for a run.

    Shares the MaxEnt recipe with :class:`FrequencyMaxEnt` (the same
    ``maxent_config`` block).  Like that representation it is expensive and is
    **not** recomputed on project load; the GUI caches the reconstruction
    directly from a completed run, and this ``compute`` is the on-demand
    fallback that re-runs MaxEnt from the recorded recipe.

    Recipe::

        {"maxent_config": {...same block as FrequencyMaxEnt...}}
    """

    rep_type = RepresentationType.TIME_MAXENT_RECON
    recompute_on_load = False

    def maxent_config(self) -> dict[str, Any]:
        """Return the raw ``maxent_config`` recipe block (possibly empty)."""
        config = self.recipe.get("maxent_config")
        return dict(config) if isinstance(config, dict) else {}

    def compute(self, run: Run, *, context: Any = None) -> list[MuonDataset]:
        # Imported here to avoid a module-level import cycle (frequency imports
        # base/spectrum/maxent; this keeps time independent of frequency at
        # import time while reusing its group-id resolver).
        from asymmetry.core.representation.frequency import _resolve_selected_group_ids

        if not run.histograms:
            raise ValueError("MaxEnt reconstruction requires detector histograms.")
        config_dict = self.maxent_config()
        config = MaxEntConfig.from_dict(config_dict)
        if config.selected_group_ids is None:
            config.selected_group_ids = _resolve_selected_group_ids(run, config_dict)
        maxent_input = build_maxent_input(run, config)
        result = run_cycles(maxent_input, config)
        reconstructions = reconstruct_group_signals(maxent_input, result.state)
        self.result_metadata = {
            "cycles": int(result.state.cycle),
            "maxent_chi2": result.metadata.get("maxent_chi2"),
        }
        return build_maxent_reconstruction_datasets(reconstructions, run)
