"""Likelihood information criteria and explicit ranking domains.

AIC: Akaike (1974); AICc: Sugiura (1978), Hurvich & Tsai (1989);
BIC: Schwarz (1978). The likelihood statistic may omit a data-only constant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Protocol, TypeVar

import numpy as np


class SelectionMetric(str, Enum):
    AIC = "AIC"
    AICC = "AICc"
    BIC = "BIC"

    @classmethod
    def from_value(cls, value: object) -> SelectionMetric:
        if isinstance(value, cls):
            return value
        text = str(value).strip().casefold()
        for metric in cls:
            if metric.value.casefold() == text:
                return metric
        raise ValueError(f"unknown selection metric: {value!r}")


@dataclass(frozen=True)
class InformationCriteria:
    """One objective's sufficient statistics; derived criteria are never stored."""

    minus_two_log_likelihood: float
    parameter_count: int
    sample_count: int | None

    def __post_init__(self):
        if (
            isinstance(self.minus_two_log_likelihood, bool)
            or not isinstance(self.minus_two_log_likelihood, Real)
            or not math.isfinite(self.minus_two_log_likelihood)
        ):
            raise ValueError("−2 ln L must be finite")
        if (
            isinstance(self.parameter_count, bool)
            or not isinstance(self.parameter_count, Integral)
            or self.parameter_count < 0
        ):
            raise ValueError("parameter count must be an integer ≥ 0")
        if self.sample_count is not None and (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, Integral)
            or self.sample_count < 1
        ):
            raise ValueError("sample count must be an integer ≥ 1 or unknown")

    def unavailable_reason(self, metric: SelectionMetric | str) -> str | None:
        metric = SelectionMetric.from_value(metric)
        if metric == SelectionMetric.AIC:
            return None
        if self.sample_count is None:
            return "sample count not recorded"
        denominator = self.sample_count - self.parameter_count - 1
        if metric == SelectionMetric.AICC and denominator <= 0:
            return f"n − k − 1 = {denominator}"
        return None

    def value(self, metric: SelectionMetric | str) -> float | None:
        metric = SelectionMetric.from_value(metric)
        if self.unavailable_reason(metric) is not None:
            return None
        k, n = self.parameter_count, self.sample_count
        if metric == SelectionMetric.BIC:
            return self.minus_two_log_likelihood + k * math.log(n)
        aic = self.minus_two_log_likelihood + 2 * k
        return aic if metric == SelectionMetric.AIC else aic + 2 * k * (k + 1) / (n - k - 1)


class _CriteriaItem(Protocol):
    @property
    def criteria(self) -> InformationCriteria | None: ...

    def unavailable_reason(self, metric: SelectionMetric) -> str | None: ...


_T = TypeVar("_T", bound=_CriteriaItem)


def ranking_value(criteria: InformationCriteria | None, metric: SelectionMetric | str) -> float:
    """Numerical ordering key; unavailable records remain outside finite rankings."""
    metric = SelectionMetric.from_value(metric)
    value = None if criteria is None else criteria.value(metric)
    return math.inf if value is None else value


def ranking_is_available(items: Iterable[_CriteriaItem], metric: SelectionMetric | str) -> bool:
    """Whether every published row can still be ranked under the requested criterion."""
    metric = SelectionMetric.from_value(metric)
    return all(
        item.criteria is not None and item.criteria.value(metric) is not None for item in items
    )


def rank(
    items: Iterable[_T], metric: SelectionMetric | str
) -> tuple[list[_T], list[tuple[_T, str]]]:
    """Stable ranking plus excluded items and reasons, using exactly one metric."""
    metric = SelectionMetric.from_value(metric)
    ranked, excluded = [], []
    for item in items:
        criteria = item.criteria
        reason = (
            item.unavailable_reason(metric)
            if criteria is None
            else criteria.unavailable_reason(metric)
        )
        if criteria is None and reason is None:
            raise ValueError("unavailable criteria require an exclusion reason")
        if reason is None:
            ranked.append(item)
        else:
            excluded.append((item, reason))
    ranked.sort(key=lambda item: item.criteria.value(metric))
    return ranked, excluded


def score_deltas(scores: Sequence[float]) -> list[float]:
    """Difference from the best finite score; unavailable scores have infinite delta."""
    values = np.asarray(scores, dtype=float)
    finite = np.isfinite(values)
    deltas = np.full(values.shape, math.inf)
    if finite.any():
        deltas[finite] = values[finite] - values[finite].min()
    return deltas.tolist()


def information_weights(scores: Sequence[float]) -> list[float]:
    """Normalised exp(−Δ/2) weights; unavailable scores receive zero weight."""
    raw = np.exp(-np.asarray(score_deltas(scores)) / 2)
    total = float(raw.sum())
    return [0.0] * len(raw) if total == 0 else (raw / total).tolist()


def partition_bic(
    minus_two_log_likelihood: float,
    shared_count: int,
    local_count: int,
    *,
    sample_count: int,
    local_log_sample_count: float,
) -> float:
    """Partition BIC from total samples and Σᵣ ln(nᵣ), retaining prefix-sum cost."""
    shared = InformationCriteria(minus_two_log_likelihood, shared_count, sample_count)
    InformationCriteria(0.0, local_count, sample_count)
    if not math.isfinite(local_log_sample_count) or local_log_sample_count < 0:
        raise ValueError("partition BIC requires a finite non-negative sum of log sample counts")
    return shared.value(SelectionMetric.BIC) + local_count * local_log_sample_count
