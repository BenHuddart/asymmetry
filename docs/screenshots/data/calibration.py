"""Owned count-domain controls for calibration screenshots and demonstrations."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data.dataset import Histogram, Run
from asymmetry.core.simulate import reduce_run_to_dataset
from asymmetry.core.utils.constants import (
    GAUSS_TO_TESLA,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    MUON_LIFETIME_US,
)


def calibration_control(*, beta=1.0, vector=False, run_number=7101, relaxing=False):
    """Rounded count means with explicit acquisition windows and known balances."""
    time = np.arange(600) * 0.016
    decay = np.exp(-time / MUON_LIFETIME_US)
    histograms, groups, projections = [], {}, []
    for index, alpha in enumerate((1.25, 1.1, 0.9) if vector else (1.25,)):
        wave = 0.2 * np.exp(-0.15 * time) * np.cos(2 * np.pi * 1.3 * time + 0.17 + index)
        if relaxing:
            wave = 0.2 * np.exp(-0.4 * time)
        backgrounds = (0.0, 0.0) if relaxing else (7.0, 11.0)
        for values in (
            20000 * decay * (1 + wave) + backgrounds[0],
            20000 / alpha * decay * (1 - beta * wave) + backgrounds[1],
        ):
            histograms.append(Histogram(np.rint(values), 0.016))
            groups[len(histograms)] = [len(histograms)]
        projections.append(
            {
                "label": ("P_x", "P_y", "P_z")[index],
                "forward_group": index * 2 + 1,
                "backward_group": index * 2 + 2,
            }
        )
    grouping = {
        "groups": groups,
        "forward_group": 1,
        "backward_group": 2,
        "first_good_bin": 4,
        "last_good_bin": 498,
        "deadtime_correction": False,
        "background_correction": False,
    }
    if vector:
        grouping["projections"] = projections
    run = Run(
        run_number,
        histograms,
        metadata={
            "run_number": run_number,
            "title": "Synthetic decay-balance control"
            if relaxing
            else "Synthetic TF calibration control",
            "field": 0.0 if relaxing else 1.3 / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T / GAUSS_TO_TESLA,
        },
        grouping=grouping,
    )
    return reduce_run_to_dataset(run)
