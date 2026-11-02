"""Raw loading retains editable requests whose independent quick-look is unavailable."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.data.dataset import Histogram
from asymmetry.core.transform.asymmetry import compute_asymmetry
from asymmetry.core.transform.fb_counts import AlignmentCrop
from asymmetry.core.transform.grouping import InvalidGroupingError, group_forward_backward
from asymmetry.core.transform.units import PERCENT_PER_FRACTION


@dataclass(frozen=True)
class QuickAsymmetry:
    """A loader-derived pair or an explicit unavailable reason alongside raw data."""

    asymmetry: NDArray[np.float64]
    error: NDArray[np.float64]
    common_t0: int
    alignment: AlignmentCrop | None
    unavailable_reason: str | None = None

    @property
    def metadata(self) -> dict:
        """Publish alignment only for a valid pair, otherwise its repair diagnostic."""
        if self.unavailable_reason is not None:
            return {"correction_error": self.unavailable_reason}
        return {"grouped_alignment": asdict(self.alignment)}


def quick_asymmetry(histograms: list[Histogram], grouping: dict) -> QuickAsymmetry:
    """Validate the default pair before exposing independent-count uncertainties."""
    try:
        grouped = group_forward_backward(histograms, grouping)
    except InvalidGroupingError as exc:
        return QuickAsymmetry(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            max(hist.t0_bin for hist in histograms),
            None,
            str(exc),
        )
    asymmetry, error = compute_asymmetry(grouped.forward, grouped.backward, alpha=grouped.alpha)
    return QuickAsymmetry(
        asymmetry * PERCENT_PER_FRACTION,
        error * PERCENT_PER_FRACTION,
        grouped.common_t0,
        grouped.counts.alignment,
    )
