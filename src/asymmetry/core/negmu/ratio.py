"""EXPERIMENTAL — WORK IN PROGRESS. Negative-muon (μ⁻) capture-lifetime analysis.

This API is UNVALIDATED against real μ⁻ elemental-analysis data. No μ⁻ corpus
exists in this project; every result here has been exercised only against
synthetic histograms. The element lifetime values are literature-anchored
(Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987), via Blundell et al.,
Muon Spectroscopy: An Introduction, OUP 2022, Table C.1), but the fitting,
capture-ratio, and background machinery have NOT been checked against an
established tool (WiMDA, Mantid) on measured data. The API, parameter names, and
return shapes MAY CHANGE without notice. Do not rely on results for publication
without independent verification. This feature is deliberately NOT exposed in the
GUI fit builders. Promotion trigger for a GUI: real ISIS μ⁻ data AND a user.

Capture-ratio report: relative capture probabilities from fitted amplitude
ratios amp_i/amp_ref, with covariance-aware uncertainties. Adapts WiMDA's
RatioButtonClick (NegMuAnalyse.pas:455–620) as a derived-quantities function —
no new results framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from asymmetry.core.negmu.fit import CaptureModelSpec
from asymmetry.core.negmu.lifetimes import DECAY_BACKGROUND_LABEL

if TYPE_CHECKING:
    from asymmetry.core.fitting.engine import FitResult
    from asymmetry.core.fitting.grouped_time_domain import GroupedTimeDomainFitResult


@dataclass(frozen=True)
class CaptureRatio:
    """One amplitude ratio and its uncertainty."""

    numerator: str
    denominator: str
    ratio: float
    sigma: float


@dataclass(frozen=True)
class CaptureRatioReport:
    """Derived capture-probability ratios from a single fit result."""

    side: str  # "forward" | "backward" | "combined"
    reference: str
    ratios: tuple[CaptureRatio, ...]
    amplitudes: dict[str, float]
    amplitude_uncertainties: dict[str, float]


def capture_ratio_report(
    fit: FitResult,
    spec: CaptureModelSpec,
    *,
    reference: str,
    side: str = "forward",
) -> CaptureRatioReport:
    """Compute per-element capture-probability ratios amp_label/amp_reference.

    For each element label ≠ reference (and ≠ decayBG, excluded by default)::

        R = amp_label / amp_reference
        σ_R = R · sqrt((σ_i/amp_i)² + (σ_ref/amp_ref)² − 2·cov(i,ref)/(amp_i·amp_ref))

    where the covariance term is taken from ``fit.covariance`` /
    ``fit.covariance_parameters`` when both parameters are present in the
    covariance matrix, and falls back to 0 (pure quadrature) otherwise.
    """
    params = fit.parameters
    uncertainties = fit.uncertainties
    covariance = fit.covariance
    cov_param_list: list[str] = list(fit.covariance_parameters or [])

    amp_ref_name = f"amp_{reference}"
    amp_ref = params[amp_ref_name].value
    sigma_ref = uncertainties.get(amp_ref_name, 0.0)

    # All element labels except decayBG (use spec.labels() to preserve component order).
    all_labels = [lbl for lbl in spec.labels() if lbl != DECAY_BACKGROUND_LABEL]

    amplitudes: dict[str, float] = {}
    amp_uncertainties: dict[str, float] = {}
    ratios: list[CaptureRatio] = []

    for lbl in all_labels:
        amp_name = f"amp_{lbl}"
        if amp_name not in params:
            continue
        amp = params[amp_name].value
        sigma = uncertainties.get(amp_name, 0.0)
        amplitudes[lbl] = amp
        amp_uncertainties[lbl] = sigma

        if lbl == reference:
            continue

        ratio = amp / amp_ref if amp_ref != 0.0 else float("inf")

        # Zero-amplitude guard: σ_R formula divides by amp and amp_ref — undefined
        # when either is zero (component absent or at its lower bound).
        if amp == 0.0 or amp_ref == 0.0:
            sigma_ratio = float("inf")
        else:
            # Covariance-aware uncertainty; fall back to quadrature (cov_term=0) when absent.
            cov_term = 0.0
            if (
                covariance is not None
                and amp_name in cov_param_list
                and amp_ref_name in cov_param_list
            ):
                i = cov_param_list.index(amp_name)
                j = cov_param_list.index(amp_ref_name)
                cov_ij = float(covariance[i, j])
                cov_term = 2.0 * cov_ij / (amp * amp_ref)

            var = (sigma / amp) ** 2 + (sigma_ref / amp_ref) ** 2 - cov_term
            sigma_ratio = abs(ratio) * math.sqrt(max(0.0, var))

        ratios.append(
            CaptureRatio(
                numerator=lbl,
                denominator=reference,
                ratio=ratio,
                sigma=sigma_ratio,
            )
        )

    return CaptureRatioReport(
        side=side,
        reference=reference,
        ratios=tuple(ratios),
        amplitudes=amplitudes,
        amplitude_uncertainties=amp_uncertainties,
    )


def fb_capture_ratio_report(
    grouped: GroupedTimeDomainFitResult,
    spec: CaptureModelSpec,
    forward_group: int,
    backward_group: int,
    *,
    reference: str,
) -> dict[str, CaptureRatioReport]:
    """Per-side capture-ratio reports from a :func:`~fit.fit_capture_fb_alpha` result.

    Returns ``{'forward': CaptureRatioReport, 'backward': CaptureRatioReport}``.
    Since amplitudes are shared in the F+B fit, the two reports will have identical
    ratios by construction (they differ only in their ``side`` label).
    """
    result_f = grouped.group_results[int(forward_group)]
    result_b = grouped.group_results[int(backward_group)]
    return {
        "forward": capture_ratio_report(result_f, spec, reference=reference, side="forward"),
        "backward": capture_ratio_report(result_b, spec, reference=reference, side="backward"),
    }
