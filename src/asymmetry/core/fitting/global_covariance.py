"""Observed curvature and marginal covariance of separable member objectives.

The information is half the Hessian of χ² or Poisson deviance. Eliminating
member-only columns gives the Schur complement; its inverse propagates shared
uncertainty back into the local blocks (JCGM 100:2008, section 5.2).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.likelihood import GaussianObservations

_CURVATURE_RELATIVE_TOLERANCE = 2e-3
_INFORMATION_EIGENVALUE_FLOOR = 1e-10


class CurvatureUnavailableError(ValueError):
    """The objective does not support a reliable local quadratic covariance."""


class _CurvatureStepError(CurvatureUnavailableError):
    """Changing derivative resolution may resolve unstable objective curvature."""


def _information_inverse(information, reference_diagonal=None):
    """Invert positive, well-conditioned curvature in dimensionless coordinates."""
    if not information.size:
        return information.copy()
    diagonal = np.diag(information)
    if not np.all(np.isfinite(information)) or np.any(diagonal <= 0):
        raise CurvatureUnavailableError("nonfinite or nonpositive objective curvature")
    scales = np.sqrt(diagonal if reference_diagonal is None else reference_diagonal)
    correlation = information / np.outer(scales, scales)
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues[0] <= _INFORMATION_EIGENVALUE_FLOOR * max(1.0, eigenvalues[-1]):
        raise CurvatureUnavailableError("singular or ill-conditioned objective curvature")
    return np.linalg.inv(correlation) / np.outer(scales, scales)


@dataclass(frozen=True)
class MarginalCovariance:
    """Selected covariance blocks without an all-members dense inverse.

    Each column's shared sensitivity is a row of ``response``. Local conditional
    covariance contributes only within that column's member; the shared term
    creates correlations between different members too.
    """

    shared: NDArray[np.float64]
    response: NDArray[np.float64]
    local_blocks: tuple[NDArray[np.float64], ...]
    local_positions: dict[int, tuple[int, int]]

    def block(self, indices: Sequence[int]) -> NDArray[np.float64]:
        response = self.response[list(indices)]
        covariance = response @ self.shared @ response.T
        for i, column in enumerate(indices):
            if column not in self.local_positions:
                continue
            owner, row = self.local_positions[column]
            for j, other in enumerate(indices):
                if other in self.local_positions:
                    other_owner, other_row = self.local_positions[other]
                    if owner == other_owner:
                        covariance[i, j] += self.local_blocks[owner][row, other_row]
        return covariance

    def uncertainty(self, index: int) -> float:
        return float(np.sqrt(self.block([index])[0, 0]))


def _member_information(evaluate, values, bounds, record, step_scale=1.0):
    """Differentiate the model, retaining residual-weighted second derivatives.

    For Gaussian data I = JᵀWJ − Σ (y−μ)W μʺ; for Poisson data
    I = Σ n/μ² J Jᵀ + (1−n/μ) μʺ. Thus nonlinear residual curvature is
    retained rather than replaced by the Gauss–Newton approximation.
    """
    base = evaluate(values)
    if not np.all(np.isfinite(base)):
        raise CurvatureUnavailableError("nonfinite fitted model")
    if isinstance(record, GaussianObservations):
        first = (base - record.values) / record.sigma**2
        second = 1 / record.sigma**2
    else:
        if np.any(base <= 0):
            raise CurvatureUnavailableError("Poisson covariance requires positive expected counts")
        first = 1 - record.counts / base
        second = record.counts / base**2
    size = len(values)
    if not size:
        return np.empty((2, 0, 0))
    steps = step_scale * np.finfo(float).eps ** 0.25 * np.maximum(np.abs(values), 1.0)
    distances = np.array([min(value - lo, hi - value) for value, (lo, hi) in zip(values, bounds)])
    steps = np.minimum(steps, distances / 3)
    if np.any(
        steps <= np.finfo(float).eps ** 0.5 * np.maximum(np.abs(values), np.finfo(float).tiny)
    ):
        raise CurvatureUnavailableError(
            "parameter at or too close to a bound for symmetric curvature"
        )
    estimates = []
    for step in (steps, steps / 2):
        jacobian = []
        information = np.empty((size, size))
        for i in range(size):
            offset = np.zeros(size)
            offset[i] = step[i]
            plus, minus = evaluate(values + offset), evaluate(values - offset)
            jacobian.append((plus - minus) / (2 * step[i]))
            information[i, i] = np.dot(first, (plus - 2 * base + minus) / step[i] ** 2)
        for i in range(size):
            for j in range(i + 1, size):
                a, b = np.zeros(size), np.zeros(size)
                a[i], b[j] = step[i], step[j]
                derivative = (
                    evaluate(values + a + b)
                    - evaluate(values + a - b)
                    - evaluate(values - a + b)
                    + evaluate(values - a - b)
                ) / (4 * step[i] * step[j])
                information[i, j] = information[j, i] = np.dot(first, derivative)
        jacobian = np.asarray(jacobian)
        information += (jacobian * second) @ jacobian.T
        estimates.append(information)
    coarse, fine = estimates
    scale = np.sqrt(
        np.outer(np.maximum(abs(np.diag(fine)), 1e-30), np.maximum(abs(np.diag(fine)), 1e-30))
    )
    if not np.all(np.isfinite(fine)) or np.any(
        abs(fine - coarse) > _CURVATURE_RELATIVE_TOLERANCE * scale
    ):
        raise _CurvatureStepError("objective curvature is numerically unstable")
    # Richardson removes the leading step² error; the fine estimate checks its propagation.
    return np.asarray([(4 * fine - coarse) / 3, fine])


def _schur_covariance(members, n_parameters, estimate, cancel_guard):
    """Eliminate member-only parameters from one estimate of observed information."""
    counts = Counter(column for columns, _ in members for column in columns)
    if set(counts) != set(range(n_parameters)):
        raise CurvatureUnavailableError("a free parameter has no observations")
    shared_columns = [column for column, count in sorted(counts.items()) if count > 1]
    shared_positions = {column: i for i, column in enumerate(shared_columns)}
    size = len(shared_columns)
    schur = np.zeros((size, size))
    shared_diagonal = np.zeros(size)
    response = np.zeros((n_parameters, size))
    response[shared_columns] = np.eye(size)
    local_blocks = []
    local_positions = {}
    for columns, estimates in members:
        cancel_guard()
        information = estimates[estimate]
        global_rows = [i for i, column in enumerate(columns) if column in shared_positions]
        local_rows = [i for i, column in enumerate(columns) if column not in shared_positions]
        shared = [shared_positions[columns[i]] for i in global_rows]
        local = tuple(columns[i] for i in local_rows)
        a = information[np.ix_(global_rows, global_rows)]
        b = information[np.ix_(global_rows, local_rows)]
        conditional = _information_inverse(information[np.ix_(local_rows, local_rows)])
        sensitivity = b @ conditional
        schur[np.ix_(shared, shared)] += a - sensitivity @ b.T
        shared_diagonal[shared] += np.diag(a)
        response[np.ix_(local, shared)] = -sensitivity.T
        local_positions.update(
            {column: (len(local_blocks), row) for row, column in enumerate(local)}
        )
        local_blocks.append(conditional)
    return MarginalCovariance(
        _information_inverse(schur, shared_diagonal), response, tuple(local_blocks), local_positions
    )


def _observed_covariance_at_step(problem, values, cancel_guard, step_scale):
    """Recover all free-column marginal blocks from the authoritative layout.

    Storage scales as the shared block plus one small block per member. A local
    parameter tied across members is a shared column for this calculation.
    """
    members = []
    for block in problem.blocks:
        global_columns = problem.block_global_columns[block.key]
        for dataset in block.datasets:
            run = dataset.run_number
            record = problem.observations[run]
            params = block.initial_params[run]
            mapping = {
                name: column for name, column in global_columns.items() if not params[name].fixed
            }
            mapping.update(problem.dataset_param_indices[run])
            columns = sorted(set(mapping.values()))
            fixed = problem.fixed_params[run]

            def evaluate(member_values):
                cancel_guard()
                varying = dict(zip(columns, member_values))
                arguments = dict(fixed)
                arguments.update({name: varying[column] for name, column in mapping.items()})
                return np.asarray(block.model_fn(record.time, **arguments), dtype=float)

            information = _member_information(
                evaluate,
                values[columns],
                [problem.param_bounds[c] for c in columns],
                record,
                step_scale,
            )
            members.append((columns, information))
    covariance, check = [
        _schur_covariance(members, len(values), estimate, cancel_guard) for estimate in (0, 1)
    ]
    blocks = [(covariance.shared, check.shared)]
    blocks.extend((covariance.block(columns), check.block(columns)) for columns, _ in members)
    for block, comparison in blocks:
        cancel_guard()
        scale = np.sqrt(np.outer(np.diag(block), np.diag(block)))
        if np.any(abs(block - comparison) > _CURVATURE_RELATIVE_TOLERANCE * scale):
            raise _CurvatureStepError("marginal covariance is numerically unstable")
    return covariance


def observed_covariance(problem, values, cancel_guard: Callable[[], None]):
    """Refine observed curvature at most four times, retaining the stability tolerance.

    A numerical model's parameter-dependent grid can cross a discretisation
    boundary at one perturbation size. Convergence of both member information
    and marginal blocks is required before publishing errors.
    """
    for step_scale in (1.0, 0.25, 0.0625, 0.015625):
        try:
            return _observed_covariance_at_step(problem, values, cancel_guard, step_scale)
        except _CurvatureStepError:
            continue
    return _observed_covariance_at_step(problem, values, cancel_guard, 0.00390625)
