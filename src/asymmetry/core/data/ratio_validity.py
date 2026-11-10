"""Mathematical usability of known histogram ratios, independent of display errors."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

RATIO_OBSERVATION_POLICY = "omit_undefined_zero_variance_v1"


class RatioReason(IntEnum):
    USABLE = 0
    UNDEFINED = 1
    ZERO_VARIANCE = 2


@dataclass(frozen=True)
class ObservationReport:
    """Counts within the requested Gaussian fit support; reasons are exclusive."""

    used: int
    undefined_ratio: int = 0
    zero_variance: int = 0

    @property
    def omitted(self) -> int:
        return self.undefined_ratio + self.zero_variance

    def to_dict(self) -> dict[str, int]:
        return dict(
            used=self.used,
            omitted=self.omitted,
            undefined_ratio=self.undefined_ratio,
            zero_variance=self.zero_variance,
        )

    def summary(self) -> str:
        return (
            f"Gaussian observations: used {self.used}, omitted {self.omitted} "
            f"(undefined ratio {self.undefined_ratio}, zero variance {self.zero_variance})."
        )


@dataclass(frozen=True)
class RatioValidity:
    """One byte per output bin, classified before replacing mathematical zero errors.

    Undefined ratios take precedence over zero variance. Full acquisition grids
    and contributing counts remain independent of Gaussian selection.
    """

    reasons: NDArray[np.uint8]

    def __post_init__(self):
        reasons = np.asarray(self.reasons)
        if reasons.ndim != 1 or not np.all(np.isin(reasons, list(RatioReason))):
            raise ValueError("Ratio validity requires one recognised reason per bin.")
        reasons = np.array(reasons, dtype=np.uint8, copy=True)
        reasons.setflags(write=False)
        object.__setattr__(self, "reasons", reasons)

    def __reduce__(self):
        return type(self), (self.reasons,)

    def select(self, selection) -> "RatioValidity":
        """Select the same bins as a curve crop."""
        return RatioValidity(self.reasons[selection])

    def report(self) -> ObservationReport:
        """Count exclusive reasons without inspecting plotted values or errors."""
        counts = np.bincount(self.reasons, minlength=3)
        return ObservationReport(*(int(n) for n in counts))

    def mathematical_error(self, displayed_error):
        """Recover defined zero variance for subsequent independent composition."""
        return np.where(self.reasons == RatioReason.ZERO_VARIANCE, 0.0, displayed_error)

    @classmethod
    def from_ratio(cls, defined, error) -> "RatioValidity":
        """Classify the actual propagated error before a display fallback."""
        return cls(
            np.where(
                ~defined,
                RatioReason.UNDEFINED,
                np.where(error == 0.0, RatioReason.ZERO_VARIANCE, RatioReason.USABLE),
            )
        )

    def combine(self, other: "RatioValidity", error) -> "RatioValidity":
        """A sum/difference needs defined constituents and positive final variance."""
        defined = (self.reasons != RatioReason.UNDEFINED) & (other.reasons != RatioReason.UNDEFINED)
        return self.from_ratio(defined, error)
