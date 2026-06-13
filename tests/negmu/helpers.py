"""Shared test helpers for negmu tests."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run


def make_fb_template(n_bins: int = 1024, bin_width: float = 0.016) -> Run:
    """Two-detector F/B template: group 1 = det 1 (forward), group 2 = det 2 (backward)."""
    histograms = [
        Histogram(
            counts=np.zeros(n_bins, dtype=float),
            bin_width=bin_width,
            t0_bin=0,
            good_bin_start=0,
            good_bin_end=n_bins - 1,
        )
        for _ in range(2)
    ]
    grouping = {
        "groups": {1: [1], 2: [2]},
        "group_names": {1: "Forward", 2: "Backward"},
        "forward_group": 1,
        "backward_group": 2,
        "alpha": 1.0,
        "t0_bin": 0,
        "t_good_offset": 0,
        "first_good_bin": 0,
        "last_good_bin": n_bins - 1,
        "bin_index_base": 1,
        "bunching_factor": 1,
        "good_frames": 1.0,
        "deadtime_correction": False,
        "dead_time_us": [0.0, 0.0],
        "included_groups": {1: True, 2: True},
    }
    return Run(
        run_number=0,
        histograms=histograms,
        metadata={"title": "F/B test template"},
        grouping=grouping,
        source_file="",
    )


def combine_groups(run_f: Run, run_b: Run, template: Run) -> Run:
    """Merge per-group simulation runs into one run.

    run_f carries the forward signal in histogram 0 (group 1).
    run_b carries the backward signal in histogram 1 (group 2).
    """
    h_fwd = run_f.histograms[0]
    h_bwd = run_b.histograms[1]
    combined = [
        Histogram(
            counts=h_fwd.counts.copy(),
            bin_width=h_fwd.bin_width,
            t0_bin=h_fwd.t0_bin,
            good_bin_start=h_fwd.good_bin_start,
            good_bin_end=h_fwd.good_bin_end,
        ),
        Histogram(
            counts=h_bwd.counts.copy(),
            bin_width=h_bwd.bin_width,
            t0_bin=h_bwd.t0_bin,
            good_bin_start=h_bwd.good_bin_start,
            good_bin_end=h_bwd.good_bin_end,
        ),
    ]
    return Run(
        run_number=0,
        histograms=combined,
        metadata={"title": "F/B combined"},
        grouping=template.grouping,
        source_file="",
    )


def make_dataset(run: Run) -> MuonDataset:
    return MuonDataset(
        time=np.array([]),
        asymmetry=np.array([]),
        error=np.array([]),
        metadata={},
        run=run,
    )
