"""Shared, JSON-serialisable summary of a fit result.

Both the run-batch and grouped-series recording paths convert a
:class:`~asymmetry.core.fitting.engine.FitResult` into the same compact shape so
a :attr:`~asymmetry.core.representation.series.FitSeries.results_by_run` entry
has one canonical structure for parameter trending.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from asymmetry.core.fitting.engine import FitResult
from asymmetry.core.fitting.fit_quality import assess_fit_quality
from asymmetry.core.fitting.likelihood import Objective
from asymmetry.core.fitting.member_quality import member_quality_flags, parameters_at_bound

__all__ = [
    "fit_result_summary",
    "parameters_at_bound",
    "reduced_statistic_label",
    "reduced_statistic_summary",
]

#: Default two-sided confidence level R for the χ² good/poor/overdone verdict.
#: Muon-tuned to 0.999 (WiMDA's own clamp ceiling): high-statistics muon fits
#: routinely sit at ν of several hundred to a few thousand, where the band is
#: narrow, and in muon practice χ²ᵣ ≈ 1.05–1.2 at large ν is excellent. At the
#: WiMDA algorithm default R = 0.95 the band at ν ≈ 500 is only ~[0.88, 1.13], so
#: a routine χ²ᵣ ≈ 1.2 was alarmed as "poor" in red (corpus finding #6). The band
#: *math* (``assess_fit_quality``) is unchanged and follows WiMDA — this is the
#: product default for the verdict shown, and it stays user-tunable in
#: Options ▸ "Fit quality confidence". The ``confidence`` argument below threads
#: the configured value through; the core helper keeps the 0.95 algorithm default.
FIT_QUALITY_CONFIDENCE = 0.999

#: A "poor"/"overdone" verdict whose χ²ᵣ is within this absolute margin of 1.0 is
#: numerically near-ideal and only flips out of the band because the confidence
#: interval tightens as ν grows (the cuprate χ²ᵣ=1.10/ν=1927 case). The verdict
#: itself is unchanged — this flag only lets the GUI soften an alarming chip into
#: a "marginal" reading. (Errors after ``rebin`` are propagated correctly, so a
#: genuinely high χ²ᵣ — e.g. 6–22 after bunching — is NOT within this margin and
#: stays a true "poor": we surface that honestly rather than rescale errors.)
_CHI2_MARGINAL_ABS_TOL = 0.2


def _infer_dof(fit_result: FitResult) -> int:
    """Use recorded ν, or recover it from the reported statistic and reduced value."""
    if fit_result.dof > 0:
        return fit_result.dof
    statistic, reduced = fit_result.chi_squared, fit_result.reduced_chi_squared
    return (
        int(round(statistic / reduced))
        if math.isfinite(statistic) and math.isfinite(reduced) and statistic > 0 and reduced > 0
        else 0
    )


# A display threshold for an asymptotic approximation, not a sparse-count calibration.
_POISSON_MIN_EXPECTED_COUNT = 5.0


def reduced_statistic_label(objective: Objective | str | None, *, ascii: bool = False) -> str:
    """Name the recorded objective without inferring it from a stored numeric field."""
    if objective is None:
        return "Fit statistic/nu" if ascii else "Fit statistic/ν"
    if Objective(objective) is Objective.POISSON:
        return "Poisson deviance/nu" if ascii else "Poisson deviance/ν"
    return "chi^2/nu" if ascii else "χ²/ν"


def reduced_statistic_summary(summaries: Sequence[Mapping], *, mean: bool = False) -> str:
    """Describe homogeneous reduced statistics without pooling unknown likelihoods."""
    if not summaries:
        return ""
    objectives = {summary.get("objective") for summary in summaries}
    if len(objectives) != 1:
        return "Mixed fit objectives."
    objective = next(iter(objectives))
    if objective is None:
        return "Fit objective was not recorded."
    values = [
        float(summary["reduced_chi_squared"])
        for summary in summaries
        if isinstance(summary.get("reduced_chi_squared"), (int, float))
        and math.isfinite(summary["reduced_chi_squared"])
    ]
    if not values:
        return ""
    label = reduced_statistic_label(objective)
    if mean:
        return f"average {label} = {sum(values) / len(values):.4f}"
    extent = f"{min(values):.3g}–{max(values):.3g}" if len(values) > 1 else f"{values[0]:.3g}"
    return f"{label} {extent}"


def _quality_reason(fit_result: FitResult) -> str | None:
    """Explain when the selected likelihood has no displayed χ² approximation."""
    if fit_result.objective is Objective.POISSON:
        minimum = fit_result.minimum_expected_count
        if minimum is None:
            return "Poisson expected counts were not recorded; no χ² quality verdict is available."
        if minimum < _POISSON_MIN_EXPECTED_COUNT:
            return "Poisson deviance has no χ² quality verdict when any fitted expected count is below 5."
    return None


def _quality_summary(
    fit_result: FitResult, confidence: float = FIT_QUALITY_CONFIDENCE
) -> dict | None:
    """Return an applicable approximate χ² band, or no verdict.

    Poisson applicability uses the whole minimized objective's expected-count
    minimum. The confidence level controls the two-sided reference band.
    """
    if _quality_reason(fit_result) is not None:
        return None
    chi2 = fit_result.chi_squared
    dof = _infer_dof(fit_result)
    quality = assess_fit_quality(chi2, dof, confidence)
    if quality.verdict is None or not all(
        math.isfinite(v) for v in (quality.chi2_reduced, quality.band_low, quality.band_high)
    ):
        return None
    chi2_reduced = float(quality.chi2_reduced)
    # Only soften "poor": a near-unity χ²ᵣ that reads poor purely because the band
    # tightens at high ν is the alarming case to defuse. "overdone" already renders
    # in a non-alarming accent ("suspicious, not bad"), so it is left as-is.
    marginal = quality.verdict == "poor" and abs(chi2_reduced - 1.0) <= _CHI2_MARGINAL_ABS_TOL
    return {
        "verdict": quality.verdict,
        "assumption": (
            "χ² laws are approximate for nonlinear or bounded fits."
            if fit_result.objective is Objective.GAUSSIAN
            else "The Poisson deviance uses an approximate χ² law; fitted expected counts are at least 5."
        ),
        "chi2_reduced": chi2_reduced,
        "band_low": float(quality.band_low),
        "band_high": float(quality.band_high),
        "confidence": float(quality.confidence),
        "dof": int(quality.dof),
        # Additive presentation hint: χ²ᵣ is numerically near 1 and only reads
        # "poor" because the band is tight at this ν. Verdict itself unchanged.
        "marginal": bool(marginal),
    }


def fit_result_summary(
    fit_result: FitResult,
    *,
    confidence: float = FIT_QUALITY_CONFIDENCE,
    extra_flags: Sequence[str] = (),
) -> dict:
    """Record the likelihood, statistics, parameters and quality applicability.

    The minimum fitted expectation belongs to the whole objective, including
    coupled members. MINOS intervals are signed parameter offsets; symmetric
    HESSE uncertainties retain their own values. Advisory member flags never
    change trend inclusion. Confidence uses the configured two-sided band level.
    """
    parameters = {parameter.name: float(parameter.value) for parameter in fit_result.parameters}
    uncertainties = {str(name): float(value) for name, value in fit_result.uncertainties.items()}
    asymmetric = {
        str(name): [float(lo), float(hi)]
        for name, (lo, hi) in (fit_result.minos_errors or {}).items()
    }
    return {
        "success": bool(fit_result.success),
        "objective": fit_result.objective.value,
        "minimum_expected_count": (
            None
            if fit_result.minimum_expected_count is None
            else float(fit_result.minimum_expected_count)
        ),
        "chi_squared": float(fit_result.chi_squared),
        "reduced_chi_squared": float(fit_result.reduced_chi_squared),
        "parameters": parameters,
        "uncertainties": uncertainties,
        "uncertainties_asymmetric": asymmetric,
        "quality": _quality_summary(fit_result, confidence),
        "quality_reason": _quality_reason(fit_result),
        "params_at_bound": parameters_at_bound(fit_result.parameters),
        "quality_flags": sorted(member_quality_flags(fit_result, extra_flags=extra_flags)),
    }
