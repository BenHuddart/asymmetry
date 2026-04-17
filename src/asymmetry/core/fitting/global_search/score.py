"""Scoring utilities for staged global-search candidates."""

from __future__ import annotations

import math

from asymmetry.core.fitting.global_search.types import ModelScore


def score_exact_candidate(
    chi_squared: float,
    parameter_count: int,
    sample_count: int,
    *,
    primary_metric: str = "BIC",
) -> ModelScore:
    """Compute explicit model-selection scores from a least-squares objective."""
    k = max(int(parameter_count), 0)
    n = max(int(sample_count), 1)
    chi2 = float(chi_squared)
    aic = chi2 + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / max(n - k - 1, 1) if n > k + 1 else None
    bic = chi2 + k * math.log(n)
    return ModelScore(
        chi_squared=chi2,
        parameter_count=k,
        sample_count=n,
        bic=float(bic),
        aic=float(aic),
        aicc=None if aicc is None else float(aicc),
        primary_metric=primary_metric,
    )

