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

#: A "poor"/"overdone" verdict whose χ²ᵣ is within this absolute margin of 1.0 is
#: numerically near-ideal and only flips out of the band because the confidence
#: interval tightens as ν grows (the cuprate χ²ᵣ=1.10/ν=1927 case). The verdict
#: itself is unchanged — this flag only lets the GUI soften an alarming chip into
#: a "marginal" reading. (Errors after ``rebin`` are propagated correctly, so a
#: genuinely high χ²ᵣ — e.g. 6–22 after bunching — is NOT within this margin and
#: stays a true "poor": we surface that honestly rather than rescale errors.)
_CHI2_MARGINAL_ABS_TOL = 0.2

#: Relative tolerance for "a free parameter sits on a finite bound". Scale-aware
#: (see :func:`parameters_at_bound`) so it fires on both an exact rail and a
#: near-rail without flagging interior fits.
_BOUND_REL_TOL = 1.0e-3


def parameters_at_bound(parameters: Any, *, rel_tol: float = _BOUND_REL_TOL) -> list[str]:
    """Names of *free* parameters whose fitted value sits on a finite bound.

    A free (non-fixed, non-tied, non-linked-follower) parameter pinned at its
    ``min`` or ``max`` is a sign the fit is poorly constrained: the data did not
    determine it and the optimiser parked it on the boundary (maleic A_Mu→0,
    FµF 17319 r→2.5 Å, ZF-KT Δ→1.0, weak-signal rails). Such a fit can still
    report "converged" with finite errors, so this drives an advisory badge
    alongside the χ² verdict.

    Parameters the user *fixed* or tied are excluded by ranging over
    :attr:`ParameterSet.free_parameters` only — they are *meant* to hold a value.
    Only *finite* bounds are considered. The tolerance is scale-relative (a value
    within ``rel_tol`` of the bound, floored at ``rel_tol`` in absolute terms via
    the ``max(…, 1.0)`` scale) so an exact rail and a near-rail both fire while
    interior fits do not. Qt-free and headless-testable.
    """
    free = getattr(parameters, "free_parameters", None)
    if free is None:
        return []
    names: list[str] = []
    for p in free:
        try:
            value = float(p.value)
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(value):
            continue
        for bound in (getattr(p, "min", -math.inf), getattr(p, "max", math.inf)):
            if not math.isfinite(bound):
                continue
            if abs(value - bound) <= rel_tol * max(abs(bound), abs(value), 1.0):
                names.append(str(p.name))
                break
    return names


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
    chi2_reduced = float(quality.chi2_reduced)
    marginal = quality.verdict in ("poor", "overdone") and (
        abs(chi2_reduced - 1.0) <= _CHI2_MARGINAL_ABS_TOL
    )
    return {
        "verdict": quality.verdict,
        "chi2_reduced": chi2_reduced,
        "band_low": float(quality.band_low),
        "band_high": float(quality.band_high),
        "confidence": float(quality.confidence),
        "dof": int(quality.dof),
        # Additive presentation hint: χ²ᵣ is numerically near 1 and only reads
        # poor/overdone because the band is tight at this ν. Verdict unchanged.
        "marginal": bool(marginal),
    }


def fit_result_summary(fit_result: Any, *, confidence: float = FIT_QUALITY_CONFIDENCE) -> dict:
    """Return a JSON-serialisable summary of *fit_result*.

    Includes the fitted parameter values and uncertainties so a series'
    ``results_by_run`` can drive parameter trending. Two *additive* diagnostic keys
    ride alongside without changing the meaning of the existing fields:

    - ``"quality"`` — the χ² good/poor/overdone verdict + target band (or ``None``).
    - ``"params_at_bound"`` — names of free parameters pinned on a finite bound
      (a poorly-constrained / rail-to-bound signal), for an advisory badge.
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
        "params_at_bound": parameters_at_bound(parameter_set),
    }
