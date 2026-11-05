"""Synthetic unused inverse pole for count and spectral screenshot journeys."""

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset


def with_unused_saturated_detector(dataset: MuonDataset) -> MuonDataset:
    """Keep selected spectra usable with a pole in an ungrouped acquired channel."""
    run = dataset.run
    histogram = run.histograms[0]
    frames = float(run.grouping.get("good_frames", 100.0))
    coefficients = [0.0] * len(run.histograms) + [1.0]
    # With d=1 µs, N=2 Δt F is above the inverse pole only in this unused channel.
    run.histograms.append(
        Histogram(
            np.full(histogram.n_bins, 2 * histogram.bin_width * frames),
            bin_width=histogram.bin_width,
            t0_bin=histogram.t0_bin,
        )
    )
    run.grouping.update(
        deadtime_correction=True,
        deadtime_mode="file",
        deadtime_file_us=coefficients,
        good_frames=frames,
    )
    return dataset
