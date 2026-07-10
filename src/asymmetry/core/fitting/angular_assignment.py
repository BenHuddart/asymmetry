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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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

#: K(θ) basis models eligible for the joint fit (angle-scoped, Phase 5/6).
ANGULAR_MODELS: tuple[str, ...] = ("KnightAnisotropy", "AngularCos2", "AngularFourier2")


@dataclass
class AngularAssignmentAlternative:
    """A near-degenerate runner-up assignment kept for discrimination (§3.3).

    Carries the competing per-point ``assignment`` (distinct from the winner's),
    its converged per-curve ``curves`` (canonicalised like the winner), the
    seed's ``total_chi_squared`` and whether that seed ``converged``. These are
    the labellings within a Δχ² window of the winner — the ones a new angle
    could resolve.
    """

    assignment: list[tuple[int, ...]]
    curves: list[ParameterModelFitResult]
    total_chi_squared: float
    converged: bool


@dataclass
class AngularAssignmentResult:
    """Outcome of a joint K(θ) fit with per-angle component assignment.

    ``curves[m]`` is the fit of curve ``m``; ``curve_values[m]``/``curve_errors[m]``
    are the realigned per-point values assigned to curve ``m`` (so curve ``m`` is a
    continuous trace through crossings). ``assignment[p][c]`` gives the curve index
    that component ``c`` at scan point ``p`` was assigned to. ``alternatives`` holds
    the distinct near-degenerate runner-up labellings (empty unless
    ``keep_alternatives > 0`` was requested).
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
    alternatives: list[AngularAssignmentAlternative] = field(default_factory=list)


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
    if model_name == "AngularFourier2":
        return ParameterSet(
            [
                Parameter("K_avg", centre),
                # Aligned start (the nested null): a misaligned axis is the
                # exception, not the default, so K_1 = 0 lets the optimiser
                # discover any first-harmonic leakage from the data itself.
                Parameter("K_1", 0.0),
                # Bounded to half the first harmonic's 360° period: beyond
                # ±90° the same curve re-parameterises with opposite-sign
                # K_1 (theta1 + 180 <-> -K_1), so an open bound would let the
                # optimiser pick either label for the same physics.
                Parameter("theta1", 0.0, min=-90.0, max=90.0),
                Parameter("K_amp", spread / 2.0 or 1.0),
                # Same bound rationale as AngularCos2/KnightAnisotropy's
                # theta0: beyond ±90° the second harmonic re-parameterises
                # with opposite-sign K_amp.
                Parameter("theta2", 0.0, min=-90.0, max=90.0),
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
    keep_alternatives: int = 0,
    alternative_delta_chi2: float = 9.0,
) -> AngularAssignmentResult:
    """Jointly fit ``N`` K(θ) curves, assigning each angle's components one-to-one.

    ``component_values`` is ``[scan_point][component]``; ``N`` curves are fitted
    (one per component). Returns the per-curve fits, the per-point assignment, and
    the realigned per-curve traces. The fit is seeded from both the identity
    (raw-label) and value-continuity assignments and the lowest-χ² result kept.

    When ``keep_alternatives > 0``, each seed's converged outcome is collected and
    the runners-up whose assignment differs from the winner (and each other) and
    whose ``total_chi_squared`` lies within ``alternative_delta_chi2`` of the
    winner are exposed, χ²-ordered and capped at ``keep_alternatives``, on
    :attr:`AngularAssignmentResult.alternatives` (canonicalised like the winner).
    The default (``keep_alternatives = 0``) leaves behaviour unchanged.
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

    outcomes = [_run_em(x, values, errors, model, model_name, seed, max_iter) for seed in seeds]
    best = min(outcomes, key=lambda candidate: candidate[2])
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

    alternatives = (
        _collect_alternatives(outcomes, best, model_name, keep_alternatives, alternative_delta_chi2)
        if keep_alternatives > 0
        else []
    )

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
        alternatives=alternatives,
    )


def _assignment_key(assignment: Sequence[np.ndarray]) -> tuple[tuple[int, ...], ...]:
    """Hashable, order-preserving key for an assignment (per-point permutations)."""
    return tuple(tuple(int(v) for v in perm) for perm in assignment)


def _collect_alternatives(
    outcomes: Sequence[tuple[list[np.ndarray], list[ParameterModelFitResult], float, bool]],
    winner: tuple[list[np.ndarray], list[ParameterModelFitResult], float, bool],
    model_name: str,
    keep_alternatives: int,
    delta_chi2: float,
) -> list[AngularAssignmentAlternative]:
    """Distinct near-degenerate runners-up within Δχ² of the winner, χ²-ordered.

    Dedupes by exact assignment equality against the winner and each other,
    keeps only those within ``delta_chi2`` of the winner's total χ², caps at
    ``keep_alternatives``, and canonicalises each kept alternative's curves.
    """
    winner_total = winner[2]
    seen: set[tuple[tuple[int, ...], ...]] = {_assignment_key(winner[0])}
    alternatives: list[AngularAssignmentAlternative] = []
    for cand_assignment, cand_fits, cand_total, cand_converged in sorted(
        outcomes, key=lambda candidate: candidate[2]
    ):
        if cand_total - winner_total > delta_chi2:
            break  # χ²-sorted: everything further out is beyond the window too
        key = _assignment_key(cand_assignment)
        if key in seen:
            continue
        seen.add(key)
        for fit in cand_fits:
            _canonicalize_theta0(model_name, fit)
        alternatives.append(
            AngularAssignmentAlternative(
                assignment=[tuple(int(v) for v in perm) for perm in cand_assignment],
                curves=cand_fits,
                total_chi_squared=float(cand_total),
                converged=bool(cand_converged),
            )
        )
        if len(alternatives) >= keep_alternatives:
            break
    return alternatives


#: Odd-flip covariance Jacobian rows, keyed by a fold key -> {output parameter
#: name: {input name: coefficient}}. The fold key is the model name for a
#: model with a single fold (``KnightAnisotropy``, ``AngularCos2``), or
#: ``"<model>:<angle_param>"`` for a model with more than one independent
#: fold (``AngularFourier2``). Any name absent from an entry's coefficients
#: (e.g. a fixed/absent parameter) keeps its identity row — unchanged by the
#: fold.
_FOLD_JACOBIAN_ROWS: dict[str, dict[str, dict[str, float]]] = {
    "KnightAnisotropy": {"K_iso": {"K_iso": 1.0, "K_ax": 0.5}, "K_ax": {"K_ax": -1.0}},
    "AngularCos2": {"K_amp": {"K_amp": -1.0}},
    "AngularFourier2:theta2": {"K_amp": {"K_amp": -1.0}},
    "AngularFourier2:theta1": {"K_1": {"K_1": -1.0}},
}


def _fold_covariance_jacobian(fold_key: str, names: Sequence[str]) -> np.ndarray | None:
    """Build one fold's odd-flip covariance Jacobian, restricted to ``names``.

    ``names`` is the covariance's own free-parameter order (fixed/absent
    parameters simply don't appear). Rows/columns the fold doesn't touch are
    the identity row. Returns ``None`` when ``fold_key`` has no registered
    fold (see :data:`_FOLD_JACOBIAN_ROWS` for the key convention).
    """
    rows = _FOLD_JACOBIAN_ROWS.get(fold_key)
    if rows is None:
        return None
    index = {name: i for i, name in enumerate(names)}
    jac = np.eye(len(names), dtype=float)
    for out_name, coeffs in rows.items():
        i = index.get(out_name)
        if i is None:
            continue
        jac[i, :] = 0.0
        for in_name, coeff in coeffs.items():
            j = index.get(in_name)
            if j is not None:
                jac[i, j] = coeff
    return jac


def _fold_covariance(fit: ParameterModelFitResult, fired_fold_keys: Sequence[str]) -> None:
    """Propagate Σ' = J Σ Jᵀ through the fired folds and refresh the marginals.

    No-op when ``fit.covariance`` is ``None`` (HESSE failed/didn't run, or a
    legacy fit) — the caller's quadrature fallback is the only signal then, or
    when none of ``fired_fold_keys`` has a registered Jacobian. Multiple fired
    folds compose into a single Jacobian (``AngularFourier2``'s θ1/θ2 folds
    are independent — each touches a disjoint parameter — so the composition
    order does not matter); the combined transform is applied once. Otherwise
    replaces ``fit.covariance`` with the transformed matrix (same
    ``(names, matrix)`` shape) and every affected marginal in
    ``fit.uncertainties`` with ``sqrt(diag(Σ'))``, exactly.
    """
    if fit.covariance is None:
        return
    names, matrix = fit.covariance
    combined = np.eye(len(names), dtype=float)
    applied = False
    for fold_key in fired_fold_keys:
        jac = _fold_covariance_jacobian(fold_key, names)
        if jac is None:
            continue
        combined = jac @ combined
        applied = True
    if not applied:
        return
    sigma = np.asarray(matrix, dtype=float)
    transformed = combined @ sigma @ combined.T
    fit.covariance = (list(names), transformed.tolist())
    for i, name in enumerate(names):
        variance = transformed[i, i]
        if math.isfinite(variance) and variance >= 0.0:
            fit.uncertainties[name] = math.sqrt(variance)


def _flip_knight_anisotropy(params: dict[str, Parameter], fit: ParameterModelFitResult) -> bool:
    """K_iso -> K_iso + K_ax/2, K_ax -> -K_ax (exact under the θ0 fold)."""
    k_iso, k_ax = params.get("K_iso"), params.get("K_ax")
    if k_ax is None:
        return False
    if k_iso is not None:
        k_iso.value = float(k_iso.value) + float(k_ax.value) / 2.0
        if fit.covariance is None:
            iso_err = fit.uncertainties.get("K_iso")
            ax_err = fit.uncertainties.get("K_ax")
            if iso_err is not None and ax_err is not None:
                fit.uncertainties["K_iso"] = math.hypot(float(iso_err), float(ax_err) / 2.0)
    k_ax.value = -float(k_ax.value)
    return True


def _make_sign_flip(name: str) -> Callable[[dict[str, Parameter], ParameterModelFitResult], bool]:
    """A flip that only negates ``name`` (no cross-term, no quadrature fallback)."""

    def _flip(params: dict[str, Parameter], fit: ParameterModelFitResult) -> bool:
        param = params.get(name)
        if param is None:
            return False
        param.value = -float(param.value)
        return True

    return _flip


@dataclass(frozen=True)
class _AngleFold:
    """One periodic angle parameter and its exact reparameterisation on flip.

    ``angle_param`` folds into ``(-step/2, step/2]`` in ``step``-degree
    increments; whenever an odd number of increments is needed, ``flip`` is
    invoked to apply the model's exact sign/shift reparameterisation to the
    other affected parameters (in place) and ``jacobian_key`` names the
    covariance transform in :data:`_FOLD_JACOBIAN_ROWS` for that flip.
    """

    angle_param: str
    step: float
    jacobian_key: str
    flip: Callable[[dict[str, Parameter], ParameterModelFitResult], bool]


#: Per-model fold specs. ``KnightAnisotropy``/``AngularCos2`` each have a
#: single θ0 fold (period 180°, canonical range (−45°, 45°]) as before.
#: ``AngularFourier2`` has two *independent* folds: θ2 folds exactly like
#: AngularCos2's θ0 (the second-harmonic phase, K_amp sign flip), and θ1
#: folds over the first harmonic's full 360° period (canonical range
#: (−90°, 90°], K_1 sign flip) since ``(K_1, θ1) ≡ (−K_1, θ1 + 180°)``.
_ANGLE_FOLDS: dict[str, tuple[_AngleFold, ...]] = {
    "KnightAnisotropy": (_AngleFold("theta0", 90.0, "KnightAnisotropy", _flip_knight_anisotropy),),
    "AngularCos2": (_AngleFold("theta0", 90.0, "AngularCos2", _make_sign_flip("K_amp")),),
    "AngularFourier2": (
        _AngleFold("theta2", 90.0, "AngularFourier2:theta2", _make_sign_flip("K_amp")),
        _AngleFold("theta1", 180.0, "AngularFourier2:theta1", _make_sign_flip("K_1")),
    ),
}


def _fold_periodic_angle(value: float, step: float) -> tuple[float, int]:
    """Fold ``value`` into ``(-step/2, step/2]``; return ``(folded, n_flips)``.

    ``n_flips`` is the number of ``step``-sized shifts applied — its parity is
    what the caller's odd-flip reparameterisation keys on. First reduces mod
    ``2*step`` (bringing it into ``(-step, step]``), then shifts by at most
    one ``step`` in either direction.
    """
    half = step / 2.0
    folded = math.remainder(value, 2.0 * step)
    flips = 0
    while folded > half:
        folded -= step
        flips += 1
    while folded <= -half:
        folded += step
        flips += 1
    return folded, flips


def _canonicalize_theta0(model_name: str, fit: ParameterModelFitResult) -> None:
    """Fold a fitted angle parameter into its canonical range, in place.

    Each of ``model_name``'s :data:`_ANGLE_FOLDS` entries is an exact
    reparameterisation the model is invariant under (see :class:`_AngleFold`
    and the module-level table for the per-model periods and amplitude sign
    flips — e.g. the axial form additionally shifts
    ``K_iso -> K_iso + K_ax/2``, since ``(3cos²t-1)/2 + (3sin²t-1)/2 = 1/2``),
    so the optimiser may return any equivalent representation of the same
    curve. Folding to the small-|angle| one makes amplitude signs read
    physically (a mount/rotation axis is expected to be *nearly* aligned) and
    equivalent curves report comparable angles. ``AngularFourier2``'s two
    folds (θ1, θ2) are independent and each applied when it fires.

    Every fold is a linear reparameterisation, so when ``fit.covariance`` is
    available it is propagated exactly (Σ' = J Σ Jᵀ, :func:`_fold_covariance`)
    and the affected marginal uncertainties are recomputed from
    ``sqrt(diag(Σ'))``. Only when no covariance is available (HESSE
    failed/didn't run, or a legacy fit) does ``KnightAnisotropy``'s K_iso
    uncertainty fall back to the quadrature approximation (K_ax's uncertainty
    added in quadrature, ignoring their correlation); the other folds are pure
    sign flips, whose marginal magnitude is unchanged either way.
    """
    folds = _ANGLE_FOLDS.get(model_name)
    if not folds:
        return
    params = {p.name: p for p in fit.parameters}
    fired_fold_keys: list[str] = []
    for angle_fold in folds:
        angle_param = params.get(angle_fold.angle_param)
        if angle_param is None or not math.isfinite(angle_param.value):
            continue
        folded, flips = _fold_periodic_angle(float(angle_param.value), angle_fold.step)
        if flips % 2 == 1:
            # A flip that can't find its target amplitude (a model variant
            # missing it) leaves this angle unfolded too, rather than
            # reporting a folded angle with no corresponding sign flip.
            if not angle_fold.flip(params, fit):
                continue
            fired_fold_keys.append(angle_fold.jacobian_key)
        angle_param.value = folded
    if fired_fold_keys:
        _fold_covariance(fit, fired_fold_keys)
