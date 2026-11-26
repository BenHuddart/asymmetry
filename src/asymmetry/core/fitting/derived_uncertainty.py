"""First-order uncertainty for derived fit quantities."""

import math
from collections.abc import Collection, Mapping
from itertools import combinations


def first_order_uncertainty(
    gradient: Mapping[str, float],
    uncertainties: Mapping[str, float],
    covariance: Mapping[str, Mapping[str, float]] | None = None,
    *,
    fixed_parameters: Collection[str] = (),
) -> float:
    """Return the standard uncertainty from the NIST law of propagation.

    Reference: https://physics.nist.gov/cuu/Uncertainty/combination.html.
    A required variance may come from the covariance diagonal or a recorded
    marginal. Fixed inputs and zero sensitivities contribute no variance.
    Unrecorded correlations are treated as zero.
    """
    active = {
        name: derivative
        for name, derivative in gradient.items()
        if derivative != 0.0 and name not in fixed_parameters
    }
    if any(not math.isfinite(derivative) for derivative in active.values()):
        return math.nan
    covariance = covariance or {}
    terms: list[float] = []

    for name, derivative in active.items():
        variance = covariance.get(name, {}).get(name, math.nan)
        if not math.isfinite(variance) or variance < 0.0:
            sigma = uncertainties.get(name, math.nan)
            if not math.isfinite(sigma) or sigma < 0.0:
                return math.nan
            variance = sigma * sigma
        terms.append(derivative * derivative * variance)

    for (left, dleft), (right, dright) in combinations(active.items(), 2):
        entries = [
            entry
            for entry in (
                covariance.get(left, {}).get(right, math.nan),
                covariance.get(right, {}).get(left, math.nan),
            )
            if math.isfinite(entry)
        ]
        if entries:
            terms.append(2.0 * dleft * dright * (sum(entries) / len(entries)))

    if any(not math.isfinite(term) for term in terms):
        return math.nan
    variance = sum(terms)
    if variance < -1e-9 * sum(map(abs, terms)):
        return math.nan
    return math.sqrt(max(0.0, variance))
