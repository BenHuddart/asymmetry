"""Conditional scalar ratios and means of retained counting observations."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from asymmetry.core.data.ratio_validity import ObservationReport, RatioReason, RatioValidity
from asymmetry.core.transform.asymmetry import ratio_with_count_errors
from asymmetry.core.transform.counting_noise import CountingNoise, combine_noise
from asymmetry.core.transform.fb_counts import GroupCounts, linear_combination_variance


@dataclass(frozen=True)
class ScalarObservation:
    """Mathematical classification and exclusive selected-observation counts."""

    reason: RatioReason
    report: ObservationReport

    def to_dict(self):
        return {"reason": int(self.reason), "report": self.report.to_dict()}

    @classmethod
    def from_dict(cls, value):
        report = value["report"]
        return cls(
            RatioReason(value["reason"]),
            ObservationReport(report["used"], report["undefined_ratio"], report["zero_variance"]),
        )


@dataclass(frozen=True)
class ScalarReduction:
    """Value and mathematical error; undefined outputs are NaN, never error one."""

    value: float
    error: float
    observation: ScalarObservation
    noise: CountingNoise | None = None


@dataclass(frozen=True)
class RatioObservations:
    """Defined ratio values with mathematical errors and retained responses."""

    value: np.ndarray
    error: np.ndarray
    validity: RatioValidity
    noise: CountingNoise | None
    bank_responses: tuple[tuple[GroupCounts, np.ndarray], ...] = ()

    def difference(self, red):
        noise = (
            combine_noise(self.noise, red.noise.scale(-np.ones(self.value.size)))
            if self.noise is not None and red.noise is not None
            else None
        )
        if noise is None:
            error = np.hypot(self.error, red.error)
        else:
            adjustment = counting_variance_adjustment(
                self.error**2, retained_ratio_errors(self.noise) ** 2
            )
            adjustment += counting_variance_adjustment(
                red.error**2, retained_ratio_errors(red.noise) ** 2
            )
            variance = retained_ratio_errors(noise) ** 2
            error = counting_response_error(variance + adjustment, variance + np.abs(adjustment))
        validity = self.validity.combine(red.validity, error)
        return RatioObservations(
            self.value - red.value,
            error,
            validity,
            noise,
            self.bank_responses + tuple((g, -gradient) for g, gradient in red.bank_responses),
        )

    def scalar(self):
        """Keep a defined boundary value distinct from usable Gaussian variance."""
        reason = RatioReason(int(self.validity.reasons[0]))
        return ScalarReduction(
            float(self.value[0]) if reason != RatioReason.UNDEFINED else float("nan"),
            float(self.error[0]) if reason != RatioReason.UNDEFINED else float("nan"),
            ScalarObservation(reason, self.validity.report()),
            self.noise,
        )

    def mean(self, selected):
        if not np.any(selected):
            raise ValueError("Integration window selects no bins.")
        validity = self.validity.select(selected)
        report = validity.report()
        usable = selected & (self.validity.reasons == RatioReason.USABLE)
        if not report.used:
            raise ValueError("Time-bin mean has no usable ratio observations. " + report.summary())
        weights = usable.astype(float) / report.used
        noise = (
            self.noise.transform(csr_matrix(weights[None, :])) if self.noise is not None else None
        )
        if self.bank_responses:
            adjustment, magnitude, retained = 0.0, 0.0, []
            for group, gradient in self.bank_responses:
                coefficients = weights * gradient
                prepared = linear_combination_variance(
                    group, np.array([0, group.counts.size]), coefficients
                )[0]
                if group.noise is None:
                    remainder = prepared
                else:
                    response = group.noise.transform(csr_matrix(coefficients[None, :]))
                    retained.append(response)
                    remainder = float(
                        counting_variance_adjustment(prepared, response.scalar_error**2)
                    )
                adjustment += remainder
                magnitude += abs(remainder)
            variance = float(combine_noise(*retained).scalar_error ** 2) if retained else 0.0
            error = float(
                counting_response_error(
                    np.array([variance + adjustment]), np.array([variance + magnitude])
                )[0]
            )
        else:
            error = (
                float(noise.scalar_error)
                if noise is not None
                else float(np.sqrt(np.sum((self.error[usable] / report.used) ** 2)))
            )
        reason = RatioReason.ZERO_VARIANCE if error == 0 else RatioReason.USABLE
        return ScalarReduction(
            float(np.mean(self.value[usable])), error, ScalarObservation(reason, report), noise
        )


def count_ratio_observations(forward, backward, alpha, beta, selected=None):
    """Sum all selected counts before the gradient, or retain native ratios.

    Within-bank variance uses the prepared independent/shared model. Retained
    responses supply only cross-bank covariance, avoiding duplicate variance.
    """
    n = forward.counts.size
    if selected is None:
        edges = np.arange(n + 1)
        matrix = None
        f, b = forward.counts, backward.counts
        vf, vb = [linear_combination_variance(g, edges) for g in (forward, backward)]
    else:
        edges = np.array([0, n])
        matrix = csr_matrix(selected.astype(float)[None, :])
        f, b = [np.array([np.sum(g.counts[selected])]) for g in (forward, backward)]
        vf, vb = [linear_combination_variance(g, edges, selected) for g in (forward, backward)]
    value, displayed, validity = ratio_with_count_errors(
        f, b, np.sqrt(vf), np.sqrt(vb), alpha, beta
    )
    error = validity.mathematical_error(displayed)
    denominator = beta * f + alpha * b
    defined = denominator != 0
    gradients = [
        np.divide(alpha * (1 + beta) * counts, denominator**2, out=np.zeros_like(f), where=defined)
        for counts in (b, -f)
    ]
    noise = None
    if forward.noise is not None and backward.noise is not None:
        banks = tuple(
            (g.noise if matrix is None else g.noise.transform(matrix)).scale(gradient)
            for g, gradient in zip((forward, backward), gradients, strict=True)
        )
        noise = combine_noise(*banks)
        retained = sum(retained_ratio_errors(bank) ** 2 for bank in banks)
        adjustment = counting_variance_adjustment(error**2, retained)
        variance = retained_ratio_errors(noise) ** 2
        error = counting_response_error(variance + adjustment, variance + np.abs(adjustment))
        validity = RatioValidity.from_ratio(defined, error)
    responses = tuple(zip((forward, backward), gradients, strict=True)) if selected is None else ()
    return RatioObservations(value, error, validity, noise, responses)


def retained_ratio_errors(noise):
    """Resolve scalar responses without turning near cancellation into zero variance."""
    size = noise.terms[0].direct.shape[0]
    if size == 1:
        return np.array([noise.scalar_error])
    error = noise.error
    if any(term.estimators for term in noise.terms):
        for index in np.flatnonzero(error == 0):
            row = csr_matrix(([1.0], ([0], [index])), shape=(1, size))
            error[index] = noise.transform(row).scalar_error
    return error


def counting_variance_adjustment(prepared, retained):
    """Keep fitted within-bank variance without adding retained noise twice.

    A tail estimate can replace measured Poisson variance by its expected
    counts. Equal variance authorities differ only by contraction roundoff;
    clear that difference before shared coefficients cancel large terms.
    """
    difference = prepared - retained
    tolerance = 128 * np.finfo(float).eps * (np.abs(prepared) + np.abs(retained))
    return np.where(np.abs(difference) <= tolerance, 0.0, difference)


def counting_response_error(variance, magnitude):
    """Cancel supported signed responses within floating-point contraction error."""
    tolerance = 128 * np.finfo(float).eps * magnitude
    if not np.all(np.isfinite(variance)) or np.any(variance < -tolerance):
        raise ValueError("Counting responses require finite nonnegative propagated variance.")
    return np.sqrt(np.where(np.abs(variance) <= tolerance, 0.0, variance))


def curve_observations(value, error, validity=None, noise=None):
    """Honour supplied classification/noise, otherwise the explicit error contract."""
    if noise is not None:
        defined = (
            validity.reasons != RatioReason.UNDEFINED
            if validity is not None
            else np.isfinite(value)
        )
        error = retained_ratio_errors(noise)
        validity = RatioValidity.from_ratio(defined, error)
    elif validity is None:
        validity = RatioValidity.from_ratio(np.isfinite(value) & np.isfinite(error), error)
    return RatioObservations(value, validity.mathematical_error(error), validity, noise)
