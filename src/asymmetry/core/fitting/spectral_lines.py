"""Named fitted frequencies and first-order uncertainties of their spacings."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

#: Relative spacing cutoff for selecting a geometrically close pair.
PAIR_SPLIT = 0.02


@dataclass(frozen=True)
class FittedFrequency:
    """Retain the signed fit coordinate and its parameter identity."""

    parameter: str
    value_mhz: float

    @property
    def magnitude_mhz(self) -> float:
        return abs(self.value_mhz)


@dataclass(frozen=True)
class FrequencySpacing:
    """Spacing of a pair ordered by absolute fitted frequency."""

    lower: FittedFrequency
    upper: FittedFrequency

    @property
    def separation_mhz(self) -> float:
        return self.upper.magnitude_mhz - self.lower.magnitude_mhz

    def standard_uncertainty_mhz(
        self,
        uncertainties: Mapping[str, float | None],
        covariance: Mapping[str, Mapping[str, float]] | None = None,
        *,
        fixed_parameters: Collection[str] = (),
    ) -> float | None:
        """Propagate fitted covariance by the first-order NIST uncertainty law.

        https://physics.nist.gov/cuu/Uncertainty/combination.html
        """
        active = tuple(
            line for line in (self.lower, self.upper) if line.parameter not in fixed_parameters
        )
        if not active:
            return 0.0

        # The sorted separation has a cusp at coincident magnitudes and at a
        # zero coordinate that can vary in the fit.
        if self.separation_mhz == 0.0 or any(line.value_mhz == 0.0 for line in active):
            return None

        covariance = covariance or {}
        variances = []
        for line in active:
            name = line.parameter
            variance = covariance.get(name, {}).get(name)
            if variance is None or not math.isfinite(variance) or variance < 0.0:
                sigma = uncertainties.get(name)
                if sigma is None or not math.isfinite(sigma) or sigma < 0.0:
                    return None
                variance = sigma * sigma
                if not math.isfinite(variance):
                    return None
            variances.append(variance)

        if len(active) == 1:
            return math.sqrt(variances[0])

        lower, upper = active
        cross = covariance.get(lower.parameter, {}).get(upper.parameter)
        if cross is None or not math.isfinite(cross):
            cross = covariance.get(upper.parameter, {}).get(lower.parameter)
        if cross is None or not math.isfinite(cross):
            return None
        # The derivatives of |upper| − |lower| are sign(upper) and −sign(lower).
        signs = math.copysign(1.0, lower.value_mhz) * math.copysign(1.0, upper.value_mhz)
        variance = sum(variances) - 2.0 * signs * cross
        return math.sqrt(variance) if math.isfinite(variance) and variance >= 0.0 else None


def fitted_frequency_spacings(parameters: Mapping[str, float | None]) -> list[FrequencySpacing]:
    """Return adjacent fitted lines in display order with their identities intact."""
    lines = sorted(
        (
            FittedFrequency(name, value)
            for name, value in parameters.items()
            if name.split("_")[0] == "frequency" and value is not None
        ),
        key=lambda line: line.magnitude_mhz,
    )
    return [FrequencySpacing(lower, upper) for lower, upper in zip(lines, lines[1:])]


def close_frequency_spacing(spacings: Sequence[FrequencySpacing]) -> FrequencySpacing | None:
    """Select the first adjacent pair inside the relative spacing cutoff."""
    return next(
        (pair for pair in spacings if pair.separation_mhz <= PAIR_SPLIT * pair.upper.magnitude_mhz),
        None,
    )
