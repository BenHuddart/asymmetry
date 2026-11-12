"""One Gaussian fit-input policy for integral scans and their actual selections."""

import warnings

import numpy as np

from asymmetry.core.data.ratio_validity import ObservationReport, RatioReason


def usable_scan_observations(scan, selection=None):
    """Select usable observations before seeds, weights or error stabilisation.

    Marginal errors cannot supply a likelihood for known dependent scan points.
    Bare/legacy inputs retain the explicit independent-observation assumption.
    """
    selected = np.ones(scan.n_points, dtype=bool) if selection is None else selection.copy()
    usable = (
        np.isfinite(scan.x) & np.isfinite(scan.value) & np.isfinite(scan.error) & (scan.error > 0)
    )
    reasons = np.where(usable, RatioReason.USABLE, RatioReason.ZERO_VARIANCE)
    if scan.observations:
        for i, observation in enumerate(scan.observations):
            if observation is not None:
                reasons[i] = observation.reason
    usable &= reasons == RatioReason.USABLE
    counts = np.bincount(reasons[selected], minlength=3)
    omitted = ObservationReport(int(np.sum(selected & usable)), int(counts[1]), int(counts[2]))
    return selected & usable, omitted


def scan_fit_selection(scan, selection=None, *, report=True):
    """Check the actual independent-error fit selection after mathematical omissions."""
    selected, omitted = usable_scan_observations(scan, selection)
    if report and omitted.omitted:
        warnings.warn(omitted.summary(), UserWarning, stacklevel=2)
    if any(selected[a] and selected[b] for a, b in scan.correlated_pairs):
        raise ValueError(
            "Independent-error scan fit unavailable: selected points reuse counting observations "
            "with nonzero covariance. Inspect/export the scan or select an independent subset."
        )
    if not np.any(selected):
        raise ValueError(
            "Scan fit has no usable scan points (Gaussian observations). " + omitted.summary()
        )
    return selected


def scan_fit_input(scan, *, x_min=None, x_max=None):
    """Resolve an inclusive fit window and carry its observations and restrictions."""
    selected = (scan.x >= (-np.inf if x_min is None else x_min)) & (
        scan.x <= (np.inf if x_max is None else x_max)
    )
    return scan.select(scan_fit_selection(scan, selected))
