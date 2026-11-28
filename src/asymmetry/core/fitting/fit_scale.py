"""Advisory for mismatched data and seed model asymmetry units."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


class AsymmetryScaleWarning(UserWarning):
    """Emitted when fit seeds and data appear to be on different asymmetry scales.

    The classic trap: fraction-scale seeds are fitted to a ``MuonDataset`` whose
    ``asymmetry`` is on the **percent** scale (``×100``). The fit either
    converges to a degenerate amplitude or to the wrong minimum. See
    :class:`asymmetry.core.data.dataset.MuonDataset` and its
    ``asymmetry_percent`` / ``asymmetry_fraction`` accessors.
    """


#: Magnitude heuristic for scale advice, not an estimator bound: signed counts
#: and general beta can give fraction-scale values beyond [-1, 1].
_FRACTION_SCALE_CEILING = 1.5

#: Minimum peak ratio (on top of a fraction/percent boundary crossing) before a
#: mismatch is flagged — keeps near-boundary noise quiet without masking the
#: ~100× percent-vs-fraction trap.
_SCALE_MISMATCH_RATIO = 10.0

#: Span (max − min) of a seeded curve over the window, relative to its peak,
#: below which the curve is flat and carries no amplitude to compare.
_FLAT_SEED_RELATIVE_SPAN = 1e-6


def warn_on_scale_mismatch(
    time: NDArray[np.float64],
    asymmetry: NDArray[np.float64],
    model_wrapper: Callable[..., NDArray[np.float64]],
    initial_values: Sequence[float],
    *,
    stacklevel: int,
) -> None:
    """Warn at a fraction/percent boundary crossing, never for poor seeds alone.

    Compare peak magnitudes: a decaying model's median measures its tail.
    A peak at most 1.5 is fraction-like; a larger peak can only be percent.
    Constant seed curves have no amplitude scale and are not judged.
    Advisory evaluation failures never change fitting outcomes."""
    try:
        asym = np.asarray(asymmetry, dtype=np.float64)
        data_finite = asym[np.isfinite(asym) & (asym != 0.0)]
        if data_finite.size == 0:
            return
        model_vals = np.asarray(model_wrapper(time, *initial_values), dtype=np.float64)
        model_finite = model_vals[np.isfinite(model_vals) & (model_vals != 0.0)]
        if model_finite.size == 0:
            return
        data_mag = float(np.max(np.abs(data_finite)))
        model_mag = float(np.max(np.abs(model_finite)))
        if data_mag <= 0.0 or model_mag <= 0.0:
            return
        if float(np.ptp(model_finite)) <= _FLAT_SEED_RELATIVE_SPAN * model_mag:
            return
        low, high = sorted((data_mag, model_mag))
        straddles_scale = low <= _FRACTION_SCALE_CEILING < high
        ratio = high / low
        if not straddles_scale or ratio < _SCALE_MISMATCH_RATIO:
            return
    except Exception:  # noqa: BLE001 - a guard must never break the fit it guards
        return

    data_is_larger = data_mag > model_mag
    likely = (
        "data looks percent-scale while the seeds look fraction-scale"
        if data_is_larger
        else "data looks fraction-scale while the seeds look percent-scale"
    )
    warnings.warn(
        f"Asymmetry scale mismatch: the data magnitude (~{data_mag:.3g}) and the "
        f"seeded model magnitude (~{model_mag:.3g}) differ by ~{ratio:.0f}×, so {likely}. "
        "Loaded MuonDataset.asymmetry is on the percent scale (×100); seed amplitudes "
        "to match, or pass ds.asymmetry_fraction / ds.asymmetry_percent explicitly.",
        AsymmetryScaleWarning,
        stacklevel=stacklevel,
    )
