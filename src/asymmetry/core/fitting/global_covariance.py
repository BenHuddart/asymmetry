"""Observed curvature and marginal covariance of separable member objectives.

The information is half the Hessian of χ² or Poisson deviance. Eliminating
member-only columns gives the Schur complement; its inverse propagates shared
uncertainty back into the local blocks (JCGM 100:2008, section 5.2).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from asymmetry.core.fitting.likelihood import GaussianObservations

_CURVATURE_RELATIVE_TOLERANCE = 2e-3
_INFORMATION_EIGENVALUE_FLOOR = 1e-10
_BOUND_RELATIVE_TOLERANCE = np.sqrt(np.finfo(float).eps)
_ACTIVE_BOUND_REASON = "parameter at a bound for symmetric covariance"
# Nonzero nodes and Taylor weights for first and second derivatives of f - f(0).
# Both stencils have O(h²) error, so the coarse and fine estimates can be extrapolated.
_CENTRED_STENCIL = ((-1, 1), (-0.5, 0.5), (1, 1))
_INWARD_STENCIL = ((1, 2, 3), (2, -0.5), (-5, 4, -1))


def member_jacobian_sparsity(
    point_counts: Sequence[int], parameter_columns: Sequence[tuple[int, ...]], n_columns: int
):
    """Structural residual-to-parameter dependence for a fit partitioned into members."""
    from scipy.sparse import csr_matrix

    n_rows = int(sum(point_counts))
    n_cols = n_columns
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    offset = 0
    for count, columns in zip(point_counts, parameter_columns, strict=True):
        if columns:
            block = np.asarray(columns, dtype=np.int64)
            rows.append(np.repeat(np.arange(offset, offset + count, dtype=np.int64), block.size))
            cols.append(np.tile(block, count))
        offset += count

    if not rows:
        return csr_matrix((n_rows, n_cols), dtype=np.int8)
    row_index = np.concatenate(rows)
    col_index = np.concatenate(cols)
    data = np.ones(row_index.size, dtype=np.int8)
    return csr_matrix((data, (row_index, col_index)), shape=(n_rows, n_cols))


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


def _information_at_steps(evaluate, values, base, first, second, steps, stencils):
    """Accumulate observed information using a tensor stencil for mixed derivatives."""
    offsets, samples, jacobian = [], [], []
    information = np.empty((len(values), len(values)))
    for i, (axis, (nodes, first_weights, second_weights)) in enumerate(
        zip(np.diag(steps), stencils)
    ):
        displacements = tuple(node * axis for node in nodes)
        deltas = np.asarray([evaluate(values + offset) - base for offset in displacements])
        offsets.append(displacements)
        samples.append(deltas)
        jacobian.append(np.asarray(first_weights) @ deltas[: len(first_weights)] / steps[i])
        information[i, i] = np.dot(first, np.asarray(second_weights) @ deltas) / steps[i] ** 2
        for j in range(i):
            mixed = np.zeros_like(base)
            for k, weight_i in enumerate(first_weights):
                for m, weight_j in enumerate(stencils[j][1]):
                    corner = evaluate(values + displacements[k] + offsets[j][m])
                    # Remove both axis terms before weighting to isolate the mixed difference.
                    mixed += weight_i * weight_j * (corner - base - deltas[k] - samples[j][m])
            information[i, j] = information[j, i] = np.dot(first, mixed) / (steps[i] * steps[j])
    jacobian = np.asarray(jacobian)
    return information + (jacobian * second) @ jacobian.T, jacobian


class _MemberInformation(NamedTuple):
    extrapolated: NDArray[np.float64]
    fine: NDArray[np.float64]
    gradient: NDArray[np.float64]
    gradient_resolution: NDArray[np.float64]


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
        return _MemberInformation(np.empty((0, 0)), np.empty((0, 0)), np.empty(0), np.empty(0))
    lower, upper = np.asarray(bounds).T
    left, right = values - lower, upper - values
    nearest_bound = np.minimum(left, right)
    if np.any(nearest_bound <= 0):
        raise CurvatureUnavailableError(_ACTIVE_BOUND_REASON)

    # Leave room for rounding at either bound. Cap before applying the retry scale
    # so even a narrow interval gets smaller probes.
    centred_room = nearest_bound / 2
    inward_room = np.maximum(left, right) / 4
    precision = np.finfo(float)
    nominal = precision.eps**0.25 * np.maximum(np.abs(values), 1.0)
    steps = step_scale * np.minimum(nominal, np.maximum(centred_room, inward_room))
    if np.any(steps <= precision.eps**0.5 * np.maximum(np.abs(values), precision.tiny)):
        raise CurvatureUnavailableError("parameter bounds leave insufficient derivative resolution")
    centred = steps <= centred_room
    stencils = [_CENTRED_STENCIL if use_centre else _INWARD_STENCIL for use_centre in centred]
    steps = np.where(~centred & (left > right), -steps, steps)
    (coarse, coarse_jacobian), (fine, fine_jacobian) = (
        _information_at_steps(evaluate, values, base, first, second, step, stencils)
        for step in (steps, steps / 2)
    )
    if not np.all(np.isfinite([coarse, fine])):
        raise _CurvatureStepError("objective curvature is numerically unstable")
    scale = np.sqrt(
        np.outer(np.maximum(abs(np.diag(fine)), 1e-30), np.maximum(abs(np.diag(fine)), 1e-30))
    )
    if np.any(abs(fine - coarse) > _CURVATURE_RELATIVE_TOLERANCE * scale):
        raise _CurvatureStepError("objective curvature is numerically unstable")
    # Richardson removes the leading step² error; the fine estimate checks its propagation.
    coarse_gradient, fine_gradient = coarse_jacobian @ first, fine_jacobian @ first
    gradient = (4 * fine_gradient - coarse_gradient) / 3
    # A shared gradient can be small through cancellation of large member scores.
    gradient_resolution = abs(fine_gradient - coarse_gradient) + _BOUND_RELATIVE_TOLERANCE * (
        np.abs(fine_jacobian) @ np.abs(first)
    )
    return _MemberInformation((4 * fine - coarse) / 3, fine, gradient, gradient_resolution)


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
    gradient = np.zeros_like(values)
    gradient_resolution = np.zeros_like(values)
    diagonal = np.zeros_like(values)
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
            gradient[columns] += information.gradient
            gradient_resolution[columns] += information.gradient_resolution
            diagonal[columns] += np.diag(information.extrapolated)
            members.append((columns, information))
    lower, upper = np.asarray(problem.param_bounds).reshape(-1, 2).T
    _check_active_bounds(values, lower, upper, gradient, gradient_resolution, diagonal)
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


def _check_active_bounds(values, lower, upper, gradient, gradient_resolution, diagonal):
    """Withhold symmetric covariance at a numerically active constraint."""
    # A bounded optimiser can stop just inside an active bound. At a lower
    # bound, a positive gradient points downhill. It must stay positive up
    # to the bound in the local quadratic to distinguish an interior minimum.
    for bound, sign in ((lower, 1), (upper, -1)):
        distance = sign * (values - bound)
        close = np.isfinite(bound) & (
            distance <= _BOUND_RELATIVE_TOLERANCE * np.maximum(abs(bound), 1)
        )
        outward_at_bound = sign * gradient[close] - diagonal[close] * distance[close]
        if np.any(outward_at_bound > gradient_resolution[close]):
            raise CurvatureUnavailableError(_ACTIVE_BOUND_REASON)


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
