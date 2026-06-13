"""Shared, JSON-serialisable summary of a fit result.

Both the run-batch and grouped-series recording paths convert a
:class:`~asymmetry.core.fitting.engine.FitResult` into the same compact shape so
a :attr:`~asymmetry.core.representation.series.FitSeries.results_by_run` entry
has one canonical structure for parameter trending.
"""

from __future__ import annotations

import math
from typing import Any

from asymmetry.core.fitting.fit_quality import assess_fit_quality

#: Two-sided confidence level R for the χ² good/poor/overdone verdict. Fixed at
#: WiMDA's ``Rgoodfit`` default; the helper accepts a ``confidence`` argument so
#: making this user-configurable later is a one-line change.
FIT_QUALITY_CONFIDENCE = 0.95


def _infer_dof(fit_result: Any) -> int:
    """Best-effort degrees of freedom ν for the χ² verdict.

    Prefers the explicit :attr:`FitResult.dof` (set at the core minimiser sites);
    falls back to inferring ν ≈ round(χ² / χ²ᵣ) for any legacy caller that does not
    populate it. Returns 0 ("unknown") when neither source is usable.
    """
    dof = int(getattr(fit_result, "dof", 0) or 0)
    if dof > 0:
        return dof
    chi2 = float(getattr(fit_result, "chi_squared", 0.0))
    reduced = float(getattr(fit_result, "reduced_chi_squared", 0.0))
    if chi2 > 0.0 and reduced > 0.0:
        return int(round(chi2 / reduced))
    return 0


def _quality_summary(fit_result: Any, confidence: float = FIT_QUALITY_CONFIDENCE) -> dict | None:
    """JSON-serialisable χ² verdict for *fit_result*, or ``None`` when none applies.

    Carries the good/poor/overdone verdict plus the target χ²ᵣ band so every fit
    surface can render the same chip and teaching tooltip (W7). ``None`` when no
    verdict is possible (ν < 1 or non-finite χ²) — surfaces then show χ²ᵣ bare.

    ``confidence`` is the two-sided level R for the band (default WiMDA's
    ``Rgoodfit`` = 0.95); the GUI passes the user-configured value.
    """
    chi2 = float(getattr(fit_result, "chi_squared", 0.0))
    dof = _infer_dof(fit_result)
    quality = assess_fit_quality(chi2, dof, confidence)
    if quality.verdict is None or not all(
        math.isfinite(v) for v in (quality.chi2_reduced, quality.band_low, quality.band_high)
    ):
        return None
    return {
        "verdict": quality.verdict,
        "chi2_reduced": float(quality.chi2_reduced),
        "band_low": float(quality.band_low),
        "band_high": float(quality.band_high),
        "confidence": float(quality.confidence),
        "dof": int(quality.dof),
    }


def fit_result_summary(fit_result: Any, *, confidence: float = FIT_QUALITY_CONFIDENCE) -> dict:
    """Return a JSON-serialisable summary of *fit_result*.

    Includes the fitted parameter values and uncertainties so a series'
    ``results_by_run`` can drive parameter trending. Two *additive* diagnostic keys
    ride alongside without changing the meaning of the existing fields:

    - ``"quality"`` — the χ² good/poor/overdone verdict + target band (or ``None``).
    - ``"uncertainties_asymmetric"`` — opt-in MINOS intervals ``{name: [lo, hi]}``
      (``lo < 0 < hi``), a display-only overlay. ``"uncertainties"`` stays the
      symmetric HESSE σ that every downstream surface consumes.

    ``confidence`` sets the two-sided level R of the χ² quality band (default
    WiMDA's ``Rgoodfit`` = 0.95); the GUI threads the user-configured value here.
    """
    parameters: dict[str, float] = {}
    parameter_set = getattr(fit_result, "parameters", None)
    if parameter_set is not None:
        for name in getattr(parameter_set, "names", []):
            try:
                parameters[str(name)] = float(parameter_set[name].value)
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    uncertainties = {
        str(k): float(v) for k, v in (getattr(fit_result, "uncertainties", {}) or {}).items()
    }
    asymmetric = {
        str(k): [float(lo), float(hi)]
        for k, (lo, hi) in (getattr(fit_result, "minos_errors", None) or {}).items()
    }
    return {
        "success": bool(getattr(fit_result, "success", False)),
        "chi_squared": float(getattr(fit_result, "chi_squared", 0.0)),
        "reduced_chi_squared": float(getattr(fit_result, "reduced_chi_squared", 0.0)),
        "parameters": parameters,
        "uncertainties": uncertainties,
        "uncertainties_asymmetric": asymmetric,
        "quality": _quality_summary(fit_result, confidence),
    }
