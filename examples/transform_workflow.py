"""Compute grouped asymmetry and alpha from synthetic histograms."""

from __future__ import annotations

import numpy as np

from asymmetry.core.data import Histogram
from asymmetry.core.transform import apply_grouping, compute_asymmetry, estimate_alpha


def main() -> None:
    rng = np.random.default_rng(7)
    n_bins = 200
    time = np.arange(n_bins, dtype=float) * 0.01

    base_forward = 3500.0 * np.exp(-time / 2.3)
    base_backward = 3000.0 * np.exp(-time / 2.3)

    histograms: list[Histogram] = []
    for scale in (1.00, 0.98, 1.02, 0.97):
        counts = rng.poisson(np.clip(scale * base_forward, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=0.01))
    for scale in (1.01, 0.99, 1.00, 1.03):
        counts = rng.poisson(np.clip(scale * base_backward, 1.0, None)).astype(float)
        histograms.append(Histogram(counts=counts, bin_width=0.01))

    forward = apply_grouping(histograms, [0, 1, 2, 3])
    backward = apply_grouping(histograms, [4, 5, 6, 7])

    alpha = estimate_alpha(forward, backward, first_good_bin=5, last_good_bin=160)
    asymmetry, error = compute_asymmetry(forward, backward, alpha=alpha)

    print(f"alpha = {alpha:.4f}")
    print(f"A(0) = {asymmetry[0]:.4f} +/- {error[0]:.4f}")
    print(f"A(1 us) = {asymmetry[100]:.4f} +/- {error[100]:.4f}")


if __name__ == "__main__":
    main()
