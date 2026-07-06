"""Joint K(θ) fit with per-angle component assignment.

Fitting each raw-labelled Knight-shift component trace independently against angle
breaks at crossings, where the underlying grouped fit swaps which physical site a
label refers to. This module fits ``N`` angle-dependent curves *jointly* and, at
each angle, assigns that angle's ``N`` measured component values to the curves
they best match (a one-to-one optimal assignment). Each physical site is then
followed continuously through crossings, and the assignment realigns the
component identity.

The procedure is a classification-EM / alternating least squares: hold the
assignment, fit each curve to its assigned points; then hold the curves, reassign
each angle's points to curves by minimum weighted residual (Hungarian). Because
the objective is non-convex (label switching), a few seeds are tried and the
lowest-χ² solution kept.

Pure/deterministic — no Qt. Reuses the K(θ) basis models and ``fit_parameter_model``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from asymmetry.core.fitting.component_tracking import (
    Component,
    ScanPoint,
    detect_crossings,
)
from asymmetry.core.fitting.parameter_models import (
    ParameterCompositeModel,
    ParameterModelFitResult,
    fit_parameter_model,
)
from asymmetry.core.fitting.parameters import Parameter, ParameterSet

#: K(θ) basis models eligible for the joint fit (angle-scoped, Phase 5).
ANGULAR_MODELS: tuple[str, ...] = ("KnightAnisotropy", "AngularCos2")


@dataclass
class AngularAssignmentResult:
    """Outcome of a joint K(θ) fit with per-angle component assignment.

    ``curves[m]`` is the fit of curve ``m``; ``curve_values[m]``/``curve_errors[m]``
    are the realigned per-point values assigned to curve ``m`` (so curve ``m`` is a
    continuous trace through crossings). ``assignment[p][c]`` gives the curve index
    that component ``c`` at scan point ``p`` was assigned to.
    """

    success: bool
    converged: bool
    model_name: str
    angles: list[float]
    curves: list[ParameterModelFitResult]
    assignment: list[tuple[int, ...]]
    curve_values: list[list[float]]
    curve_errors: list[list[float]]
    total_chi_squared: float = 0.0
    dof: int = 0
    message: str = ""


def _seed_parameters(model_name: str, values: np.ndarray) -> ParameterSet:
    """Data-driven starting parameters for one K(θ) curve over ``values``."""
    finite = values[np.isfinite(values)]
    centre = float(np.mean(finite)) if finite.size else 0.0
    spread = float(np.max(finite) - np.min(finite)) if finite.size else 1.0
    if model_name == "AngularCos2":
        return ParameterSet(
            [
                Parameter("K_avg", centre),
                Parameter("K_amp", spread / 2.0 or 1.0),
                Parameter("theta0", 0.0),
            ]
        )
    return ParameterSet(
        [
            Parameter("K_iso", centre),
            Parameter("K_ax", spread or 1.0),
            # Mount/zero misalignment. Bounded to half the axial form's 180°
            # period: beyond ±90° the same curve re-parameterises with the
            # opposite-sign K_ax, so an open bound would let the optimiser pick
            # either label for the same physics.
            Parameter("theta0", 0.0, min=-90.0, max=90.0),
        ]
    )


def _identity_seed(n_points: int, n_components: int) -> list[np.ndarray]:
    """Assignment that keeps each component on its own curve (raw labels)."""
    base = np.arange(n_components)
    return [base.copy() for _ in range(n_points)]


def _continuity_seed(angles: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """Assignment that follows value-continuity from the previous ordered point.

    A good second seed: at each point, match components to the previous point's
    per-curve values by nearest value (Hungarian), so a curve tracks its own
    component across a smooth crossing.
    """
    order = np.argsort(angles, kind="stable")
    n_components = values.shape[1]
    assignment = [np.arange(n_components) for _ in range(len(angles))]
    prev_curve_value = values[order[0]].astype(float).copy()  # curve m -> value
    for idx in order:
        row = values[idx].astype(float)
        cost = np.abs(row[:, None] - prev_curve_value[None, :])
        cost = np.where(np.isfinite(cost), cost, 1e18)
        comp_idx, curve_idx = linear_sum_assignment(cost)
        perm = np.empty(n_components, dtype=int)
        perm[comp_idx] = curve_idx
        assignment[idx] = perm
        for comp, curve in zip(comp_idx, curve_idx):
            if np.isfinite(row[comp]):
                prev_curve_value[curve] = row[comp]
    return assignment


def _crossing_swap_seeds(
    angles: np.ndarray, values: np.ndarray, *, max_seeds: int = 6
) -> list[list[np.ndarray]]:
    """Seeds that swap component↔curve past each detected crossing.

    Two curves that cross and then have their labels swapped past the crossing are
    exactly the value-sorted *envelopes* — a non-physical solution the identity and
    continuity seeds cannot escape. Toggling the involved pair past a crossing
    converts the envelope labelling into the continued-through-crossing labelling;
    the lowest-χ² seed (selected by the caller) recovers the physical curves.
    Reuses :func:`component_tracking.detect_crossings` (Phase 3b).
    """
    n_points, n_components = values.shape
    order = np.argsort(angles, kind="stable")
    points = [
        ScanPoint(
            x=float(angles[order[k]]),
            components=tuple(Component(float(values[order[k], c])) for c in range(n_components)),
        )
        for k in range(n_points)
    ]
    events = detect_crossings(points)
    if not events:
        return []
    # One swap per (transition, component-pair): order_swap + near_degenerate can
    # both fire for the same crossing, and applying the swap twice would cancel.
    unique = {}
    for event in events:
        unique.setdefault((event.index_left, tuple(sorted(event.component_pair))), event)
    unique_events = list(unique.values())

    def build(selected) -> list[np.ndarray]:
        sorted_perm = [np.arange(n_components) for _ in range(n_points)]
        for event in sorted(selected, key=lambda e: e.index_left):
            i, j = event.component_pair
            for k in range(event.index_left + 1, n_points):
                perm = sorted_perm[k]
                perm[i], perm[j] = perm[j], perm[i]
        seed: list[np.ndarray] = [np.arange(n_components) for _ in range(n_points)]
        for k in range(n_points):
            seed[order[k]] = sorted_perm[k]
        return seed

    seeds = [build(unique_events)]  # all crossings swapped
    for event in unique_events[: max_seeds - 1]:  # plus each single crossing
        seeds.append(build([event]))
    return seeds


def _component_for_curve(perm: np.ndarray, curve: int) -> int:
    """Component index assigned to ``curve`` under permutation ``perm``."""
    return int(np.where(perm == curve)[0][0])


def _evaluate(
    model: ParameterCompositeModel, result: ParameterModelFitResult, angle: float
) -> float:
    kwargs = {p.name: p.value for p in result.parameters}
    return float(model.function(np.array([angle], dtype=float), **kwargs)[0])


def _fit_curves(
    angles: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None,
    assignment: list[np.ndarray],
    model: ParameterCompositeModel,
    model_name: str,
) -> list[ParameterModelFitResult]:
    """Fit each curve to the points currently assigned to it."""
    n_components = values.shape[1]
    fits: list[ParameterModelFitResult] = []
    for curve in range(n_components):
        comps = np.array([_component_for_curve(perm, curve) for perm in assignment])
        ys = values[np.arange(len(angles)), comps]
        es = errors[np.arange(len(angles)), comps] if errors is not None else None
        mask = np.isfinite(angles) & np.isfinite(ys)
        if es is not None:
            yerr = es[mask]
            if not np.all(np.isfinite(yerr) & (yerr > 0)):
                yerr = None
        else:
            yerr = None
        params = _seed_parameters(model_name, ys[mask])
        try:
            fits.append(fit_parameter_model(angles[mask], ys[mask], yerr, model, params))
        except Exception as exc:  # noqa: BLE001 - a failed curve must not abort the EM
            fits.append(ParameterModelFitResult(success=False, parameters=params, message=str(exc)))
    return fits


def _reassign(
    angles: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None,
    fits: list[ParameterModelFitResult],
    model: ParameterCompositeModel,
) -> tuple[list[np.ndarray], float]:
    """One-to-one reassign each point's components to curves; return cost."""
    n_components = values.shape[1]
    assignment: list[np.ndarray] = []
    total = 0.0
    for p, angle in enumerate(angles):
        predictions = [_evaluate(model, fit, angle) for fit in fits]
        cost = np.zeros((n_components, n_components), dtype=float)
        for comp in range(n_components):
            value = values[p, comp]
            sigma2 = 1.0
            if errors is not None and np.isfinite(errors[p, comp]) and errors[p, comp] > 0:
                sigma2 = float(errors[p, comp]) ** 2
            for curve in range(n_components):
                residual = value - predictions[curve]
                cost[comp, curve] = (
                    (residual * residual) / sigma2 if np.isfinite(residual) else 1e18
                )
        comp_idx, curve_idx = linear_sum_assignment(cost)
        perm = np.empty(n_components, dtype=int)
        perm[comp_idx] = curve_idx
        assignment.append(perm)
        total += float(cost[comp_idx, curve_idx].sum())
    return assignment, total


def _run_em(
    angles: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray | None,
    model: ParameterCompositeModel,
    model_name: str,
    seed: list[np.ndarray],
    max_iter: int,
) -> tuple[list[np.ndarray], list[ParameterModelFitResult], float, bool]:
    """Alternate fit/reassign from ``seed``; return best assignment seen."""
    assignment = [perm.copy() for perm in seed]
    best: tuple[list[np.ndarray], list[ParameterModelFitResult], float] | None = None
    converged = False
    for _ in range(max_iter):
        fits = _fit_curves(angles, values, errors, assignment, model, model_name)
        new_assignment, total = _reassign(angles, values, errors, fits, model)
        if best is None or total < best[2]:
            best = ([perm.copy() for perm in new_assignment], fits, total)
        if all(np.array_equal(a, b) for a, b in zip(assignment, new_assignment)):
            converged = True
            break
        assignment = new_assignment
    assert best is not None
    return best[0], best[1], best[2], converged


def fit_assigned_angular_curves(
    angles: Sequence[float],
    component_values: Sequence[Sequence[float]],
    component_errors: Sequence[Sequence[float]] | None = None,
    *,
    model_name: str = "KnightAnisotropy",
    max_iter: int = 25,
) -> AngularAssignmentResult:
    """Jointly fit ``N`` K(θ) curves, assigning each angle's components one-to-one.

    ``component_values`` is ``[scan_point][component]``; ``N`` curves are fitted
    (one per component). Returns the per-curve fits, the per-point assignment, and
    the realigned per-curve traces. The fit is seeded from both the identity
    (raw-label) and value-continuity assignments and the lowest-χ² result kept.
    """
    if model_name not in ANGULAR_MODELS:
        raise ValueError(f"Unknown angular model {model_name!r}; expected one of {ANGULAR_MODELS}")

    x = np.asarray(angles, dtype=float)
    values = np.asarray(component_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != x.shape[0]:
        raise ValueError("component_values must be [n_points][n_components] matching angles")
    n_points, n_components = values.shape
    errors = np.asarray(component_errors, dtype=float) if component_errors is not None else None
    if errors is not None and errors.shape != values.shape:
        raise ValueError("component_errors must match component_values shape")

    model = ParameterCompositeModel([model_name])
    n_params = len(model.param_names)
    empty = AngularAssignmentResult(
        success=False,
        converged=False,
        model_name=model_name,
        angles=x.tolist(),
        curves=[],
        assignment=[],
        curve_values=[],
        curve_errors=[],
        message="Too few points or components to fit",
    )
    if n_components < 1 or n_points < n_params:
        return empty

    seeds = [_identity_seed(n_points, n_components)]
    if n_components >= 2 and n_points >= 2:
        seeds.append(_continuity_seed(x, values))
        seeds.extend(_crossing_swap_seeds(x, values))

    best: tuple[list[np.ndarray], list[ParameterModelFitResult], float, bool] | None = None
    for seed in seeds:
        candidate = _run_em(x, values, errors, model, model_name, seed, max_iter)
        if best is None or candidate[2] < best[2]:
            best = candidate
    assert best is not None
    assignment, fits, total, converged = best

    # Realigned per-curve traces: curve m takes its assigned component each point.
    curve_values: list[list[float]] = []
    curve_errors: list[list[float]] = []
    for curve in range(n_components):
        comps = [_component_for_curve(perm, curve) for perm in assignment]
        curve_values.append([float(values[p, c]) for p, c in enumerate(comps)])
        curve_errors.append(
            [
                float(errors[p, c]) if errors is not None else float("nan")
                for p, c in enumerate(comps)
            ]
        )

    for fit in fits:
        _canonicalize_theta0(model_name, fit)

    return AngularAssignmentResult(
        success=all(fit.success for fit in fits),
        converged=converged,
        model_name=model_name,
        angles=x.tolist(),
        curves=fits,
        assignment=[tuple(int(v) for v in perm) for perm in assignment],
        curve_values=curve_values,
        curve_errors=curve_errors,
        total_chi_squared=float(total),
        dof=max(n_points * n_components - n_components * n_params, 0),
    )


def _canonicalize_theta0(model_name: str, fit: ParameterModelFitResult) -> None:
    """Fold a fitted θ0 into (−45°, 45°] via the model's exact reparameterisation.

    Both angular models are invariant under a 90° shift of θ0 with a sign flip
    of the anisotropic amplitude (the axial form additionally shifts
    ``K_iso → K_iso + K_ax/2``, since ``(3cos²t−1)/2 + (3sin²t−1)/2 = 1/2``), so
    the optimiser may return either representation of the same curve. Fold to
    the small-|θ0| one so amplitude signs read physically (a mount is expected
    to be *nearly* aligned) and equivalent curves report comparable θ0. Under
    the fold K_iso's uncertainty picks up K_ax's in quadrature (their covariance
    is not propagated here — a conservative approximation). In-place.
    """
    params = {p.name: p for p in fit.parameters}
    theta0 = params.get("theta0")
    if theta0 is None or not math.isfinite(theta0.value):
        return
    folded = math.remainder(float(theta0.value), 180.0)
    flips = 0
    while folded > 45.0:
        folded -= 90.0
        flips += 1
    while folded <= -45.0:
        folded += 90.0
        flips += 1
    if flips % 2 == 1:
        if model_name == "KnightAnisotropy" and "K_ax" in params:
            k_iso, k_ax = params.get("K_iso"), params["K_ax"]
            if k_iso is not None:
                k_iso.value = float(k_iso.value) + float(k_ax.value) / 2.0
                iso_err = fit.uncertainties.get("K_iso")
                ax_err = fit.uncertainties.get("K_ax")
                if iso_err is not None and ax_err is not None:
                    fit.uncertainties["K_iso"] = math.hypot(float(iso_err), float(ax_err) / 2.0)
            k_ax.value = -float(k_ax.value)
        elif model_name == "AngularCos2" and "K_amp" in params:
            params["K_amp"].value = -float(params["K_amp"].value)
        else:
            return
    theta0.value = folded
