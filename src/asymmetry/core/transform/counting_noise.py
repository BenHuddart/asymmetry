"""Linear counting influences, sparse in bins with low-rank estimator terms.

Only marginal and same-bin covariance is evaluated. Estimator influences are
retained so a later curve mean includes reused estimation observations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, diags

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import Histogram, Run
    from asymmetry.core.transform.corrections import BackgroundReference

COUNTING_NOISE_POLICY = "retained_projection_counting_v1"


@dataclass(frozen=True, eq=False)
class CountingObservation:
    """One retained detector acquisition's independent variance, in raw bins."""

    variance: np.ndarray
    cohort: object

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class Influence:
    """Sparse direct response plus outer products for estimated corrections."""

    observation: CountingObservation
    direct: csr_matrix
    estimators: tuple[tuple[np.ndarray, np.ndarray], ...] = ()


@dataclass(frozen=True)
class CountingNoise:
    """Count or percent response to identified independent acquisition bins."""

    terms: tuple[Influence, ...]

    def transform(self, matrix: csr_matrix) -> CountingNoise:
        return CountingNoise(
            tuple(
                Influence(
                    t.observation,
                    matrix @ t.direct,
                    tuple(
                        (matrix @ sensitivity, influence) for sensitivity, influence in t.estimators
                    ),
                )
                for t in self.terms
            )
        )

    def scale(self, coefficients) -> CountingNoise:
        return self.transform(diags(coefficients, format="csr"))

    def subtract_estimate(self, influence: np.ndarray) -> CountingNoise:
        terms = []
        for term in self.terms:
            response = np.asarray(influence @ term.direct).ravel()
            for sensitivity, weights in term.estimators:
                response += (influence @ sensitivity) * weights
            terms.append(
                Influence(
                    term.observation,
                    term.direct,
                    (*term.estimators, (-np.ones(influence.size), response)),
                )
            )
        return CountingNoise(tuple(terms))

    def covariance(self, other: CountingNoise) -> np.ndarray:
        """Same-output-bin covariance from retained source responses."""
        return self._contract(other)[0]

    def _contract(self, other: CountingNoise) -> tuple[np.ndarray, np.ndarray]:
        size = self.terms[0].direct.shape[0]
        result = np.zeros(size)
        magnitude = np.zeros(size)
        for left in self.terms:
            for right in other.terms:
                if left.observation is not right.observation:
                    continue
                variance = left.observation.variance
                contribution = np.asarray(left.direct.multiply(right.direct) @ variance).ravel()
                result += contribution
                magnitude += np.abs(contribution)
                for sensitivity, weights in left.estimators:
                    contribution = sensitivity * (right.direct @ (variance * weights))
                    result += contribution
                    magnitude += np.abs(contribution)
                for sensitivity, weights in right.estimators:
                    contribution = sensitivity * (left.direct @ (variance * weights))
                    result += contribution
                    magnitude += np.abs(contribution)
                for ls, lw in left.estimators:
                    for rs, rw in right.estimators:
                        contribution = ls * rs * np.dot(lw * variance, rw)
                        result += contribution
                        magnitude += np.abs(contribution)
        return result, magnitude

    @property
    def error(self) -> np.ndarray:
        variance, magnitude = self._contract(self)
        tolerance = 128 * np.finfo(float).eps * magnitude
        if not np.all(np.isfinite(variance)) or np.any(variance < -tolerance):
            raise ValueError(
                "Retained counting influences must give finite nonnegative marginal variance"
            )
        # Opposing terms can cancel exactly, as when a packed mean subtracts
        # its own range estimate. Roundoff scales with the uncontracted noise.
        return np.sqrt(np.where(np.abs(variance) <= tolerance, 0.0, variance))


def combine_noise(*items: CountingNoise) -> CountingNoise:
    """Combine coefficients before contracting repeated acquisition observations."""
    terms = {}
    cohort = items[0].terms[0].observation.cohort
    if any(item.terms[0].observation.cohort is not cohort for item in items):
        raise ValueError("Reduce the transverse projections together as one counting cohort")
    for item in items:
        for term in item.terms:
            source = term.observation
            if source in terms:
                previous = terms[source]
                term = Influence(
                    source, previous.direct + term.direct, (*previous.estimators, *term.estimators)
                )
            terms[source] = term
    return CountingNoise(tuple(terms.values()))


def bin_matrix(edges, length, *, mean=False):
    """Linear sum/mean of complete selected intervals; no time-by-time matrix."""
    lengths = np.diff(edges)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    columns = np.arange(edges[0], edges[-1])
    values = np.repeat(1 / lengths, lengths) if mean else np.ones(columns.size)
    return csr_matrix((values, (rows, columns)), shape=(len(lengths), length))


def capture_counting_observations(
    run: Run, reference: BackgroundReference | None, *, cancel_check=None
) -> tuple[Run, BackgroundReference | None]:
    """Name borrowed detector acquisitions once before projection snapshots diverge.

    Declared source/period identity joins copied constituents. Unnamed synthetic
    acquisitions share only when the boundary explicitly supplies the same record.
    """
    registry: dict[tuple, CountingObservation] = {}
    cohort = object()

    def capture(histograms: list[Histogram], identity: tuple) -> list[Histogram]:
        result = []
        for channel, hist in enumerate(histograms):
            if cancel_check is not None:
                cancel_check()
            key = (*identity, channel)
            if key in registry:
                observation = registry[key]
                variance = hist.counts if hist.count_variance is None else hist.count_variance
                if not np.array_equal(
                    observation.variance, np.where(np.isfinite(variance), variance, 0)
                ):
                    raise ValueError(
                        "Repeated acquisition identity has inconsistent counting variance"
                    )
            else:
                variance = hist.counts if hist.count_variance is None else hist.count_variance
                owned = np.array(variance, dtype=float, copy=True)
                owned[~np.isfinite(owned)] = 0
                owned.setflags(write=False)
                observation = CountingObservation(owned, cohort)
                registry[key] = observation
            result.append(replace(hist, counting_observation=observation))
        return result

    def identity(source: Run, period: int) -> tuple:
        metadata = source.metadata
        return (
            source.source_file or id(source),
            metadata.get("source_run_number", source.run_number),
            metadata.get("data_period_number") or metadata.get("period_number") or period,
        )

    grouping = dict(run.grouping)
    if "period_inputs" in grouping:
        grouping["period_inputs"] = [
            replace(member, histograms=capture(member.histograms, identity(member, 1)))
            for member in grouping["period_inputs"]
        ]
        source = replace(run, grouping=grouping)
    elif "period_histograms" in grouping:
        grouping["period_histograms"] = [
            capture(histograms, (*identity(run, 1)[:2], index + 1))
            for index, histograms in enumerate(grouping["period_histograms"])
        ]
        source = replace(run, grouping=grouping)
    else:
        source = replace(
            run, grouping=grouping, histograms=capture(run.histograms, identity(run, 1))
        )
    if reference is not None:
        reference = replace(
            reference,
            histograms=capture(
                reference.histograms,
                (
                    reference.source_file or id(reference),
                    reference.run_number,
                    reference.period_number,
                ),
            ),
        )
    return source, reference
