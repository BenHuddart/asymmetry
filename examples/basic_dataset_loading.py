"""Construct a synthetic MuonDataset and inspect metadata."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data import Histogram, MuonDataset, Run


def main() -> None:
    counts = np.linspace(1200, 200, 64, dtype=float)
    histogram = Histogram(counts=counts, bin_width=0.02, t0_bin=0)

    run = Run(
        run_number=1234,
        histograms=[histogram],
        metadata={
            "title": "Synthetic reference run",
            "temperature": 15.0,
            "field": 50.0,
            "comment": "generated example",
        },
    )

    time = histogram.time_axis[:40]
    asymmetry = 22.0 * np.exp(-0.35 * time)
    error = np.full_like(time, 0.4)
    dataset = MuonDataset(time=time, asymmetry=asymmetry, error=error, run=run)

    print(dataset.summary())


if __name__ == "__main__":
    main()
