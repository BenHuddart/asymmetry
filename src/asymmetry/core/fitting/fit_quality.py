"""Statistical χ² fit-quality assessment shared across fitting surfaces.

For a correct model with correctly estimated (Gaussian) errors, the fit χ²
follows the chi-squared distribution with ν = N − N_free degrees of freedom.
:func:`assess_fit_quality` turns a fitted χ² into a two-sided verdict at a
given confidence level R: a χ² in the lower tail (CDF < (1−R)/2) is flagged
``"overdone"`` — the fit reproduces the data *better* than the errors allow,
which usually means overestimated errors or an over-flexible model — while
the upper tail (CDF > (1+R)/2) is ``"poor"``. The accompanying band
[``band_low``, ``band_high``] is the χ²ᵣ range a good fit should fall in at
this ν; it tightens toward 1 as ν grows.

This reproduces the WiMDA Model-layer verdict semantics (``Chi2UpdateClick``
with ``Rgoodfit``, default 0.95, clamped to [0.5, 0.999]) with exact
inverse-CDF numerics. The helper is Qt-free and is shared with the
time-domain fit diagnostics (``fit-workflow-diagnostics``).

The verdict assumes the χ² was computed against *real* error estimates: with
unit weights or scatter-estimated errors (which force χ²ᵣ toward 1 by
construction) it carries no goodness information and callers should suppress
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Module-level import: scipy.stats is not pulled in by any other eagerly
# imported module, and a lazy in-function import would pay its ~1 s cold load
# on the GUI thread the first time a verdict is rendered.
from scipy.stats import chi2 as _chi2_dist

_CONFIDENCE_MIN = 0.5
_CONFIDENCE_MAX = 0.999

FitVerdict = Literal["good", "poor", "overdone"]


@dataclass(frozen=True)
class FitQuality:
    """Two-sided χ² assessment of a fit at confidence ``confidence``."""

    #: ``None`` when no verdict is possible (ν < 1 or non-finite χ²).
    verdict: FitVerdict | None
    chi2_reduced: float
    #: χ²ᵣ target band at this ν; NaN when ν < 1.
    band_low: float
    band_high: float
    confidence: float
    dof: int


def assess_fit_quality(
    chi_squared: float,
    dof: int,
    confidence: float = 0.95,
) -> FitQuality:
    """Assess a fitted χ² against the chi-squared distribution.

    Parameters
    ----------
    chi_squared
        The (weighted) sum of squared normalised residuals at the minimum.
    dof
        Degrees of freedom ν = N − N_free. With ν < 1 no verdict is possible.
    confidence
        Two-sided confidence level R; clamped to [0.5, 0.999].
    """
    confidence = min(max(float(confidence), _CONFIDENCE_MIN), _CONFIDENCE_MAX)
    alpha = 1.0 - confidence

    chi2_value = float(chi_squared)
    nu = int(dof)
    if nu < 1 or not (chi2_value >= 0.0):
        return FitQuality(
            verdict=None,
            chi2_reduced=float("nan"),
            band_low=float("nan"),
            band_high=float("nan"),
            confidence=confidence,
            dof=nu,
        )

    cdf = float(_chi2_dist.cdf(chi2_value, nu))
    if cdf < alpha / 2.0:
        verdict: FitVerdict = "overdone"
    elif cdf > 1.0 - alpha / 2.0:
        verdict = "poor"
    else:
        verdict = "good"

    return FitQuality(
        verdict=verdict,
        chi2_reduced=chi2_value / nu,
        band_low=float(_chi2_dist.ppf(alpha / 2.0, nu)) / nu,
        band_high=float(_chi2_dist.ppf(1.0 - alpha / 2.0, nu)) / nu,
        confidence=confidence,
        dof=nu,
    )
