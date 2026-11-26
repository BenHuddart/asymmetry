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

import math
from dataclasses import dataclass
from typing import Literal

_CONFIDENCE_MIN = 0.5
_CONFIDENCE_MAX = 0.999

FitVerdict = Literal["good", "poor", "overdone"]


def reduced_statistic(statistic: float, dof: int | None) -> float:
    """Return the objective per residual degree of freedom, or NaN when undefined."""
    return float(statistic) / dof if dof is not None and dof > 0 else math.nan


def recorded_statistic(statistic: float) -> float | None:
    """Encode an unavailable numeric statistic as JSON null."""
    return float(statistic) if math.isfinite(statistic) else None


def infer_historical_dof(statistic: float, reduced: float) -> int | None:
    """Recover ν only when the stored statistic establishes it uniquely."""
    if not (math.isfinite(statistic) and math.isfinite(reduced) and statistic > 0 and reduced > 0):
        return None
    ratio = statistic / reduced
    if not math.isfinite(ratio):
        return None
    rounded = round(ratio)
    # Dividing by max(ν, 1) discarded the distinction between ν = 1 and ν ≤ 0.
    return rounded if rounded > 1 and math.isclose(ratio, rounded) else None


def quality_inapplicability_reason(
    dof: int | None, observation_reason: str | None = None
) -> str | None:
    """Explain when the χ² reference law does not apply to a fit."""
    if dof is not None and dof <= 0:
        return "A χ² quality verdict requires positive residual degrees of freedom."
    if observation_reason is not None:
        return observation_reason
    if dof is None:
        return "Residual degrees of freedom are unknown."
    return None


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
    if nu < 1 or not math.isfinite(chi2_value) or chi2_value < 0.0:
        return FitQuality(
            verdict=None,
            chi2_reduced=float("nan"),
            band_low=float("nan"),
            band_high=float("nan"),
            confidence=confidence,
            dof=nu,
        )

    from scipy.special import chdtr, gammaincinv

    cdf = float(chdtr(nu, chi2_value))
    if cdf < alpha / 2.0:
        verdict: FitVerdict = "overdone"
    elif cdf > 1.0 - alpha / 2.0:
        verdict = "poor"
    else:
        verdict = "good"

    # The chi-square quantile is twice the unit-scale gamma quantile at ν/2.
    return FitQuality(
        verdict=verdict,
        chi2_reduced=chi2_value / nu,
        band_low=2.0 * float(gammaincinv(nu / 2.0, alpha / 2.0)) / nu,
        band_high=2.0 * float(gammaincinv(nu / 2.0, 1.0 - alpha / 2.0)) / nu,
        confidence=confidence,
        dof=nu,
    )
