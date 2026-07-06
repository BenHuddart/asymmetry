"""Knight-shift analysis session: a GUI-free pipeline from fitted frequencies to K.

This module is the core model behind the Knight shift analysis window. It owns
the derivation pipeline that previously lived inline in the trend panel:

    fitted components (ν or B_µ per run)  →  K per component per run
                                          →  crossing detection along the scan

The three pieces are deliberately separate:

* :class:`KnightAnalysisInput` — an immutable *snapshot* of the measured
  quantities: one :class:`KnightPoint` per source run (abscissa, applied field,
  fitted component values/errors/covariance). The GUI builds it from the live
  trend rows; scripts can build it directly.
* :class:`~asymmetry.core.fitting.knight_shift.KnightShiftConfig` — the user's
  conversion choices (reference, unit, component subset), reused unchanged.
* :func:`evaluate` — the pure derivation producing a
  :class:`KnightAnalysisResult` of per-component :class:`KnightBranch` traces
  plus detected crossings. Nothing upstream is mutated.

Only the configuration and the *source binding* (which series, which axis) are
persisted — see :class:`KnightAnalysisState`. The point snapshot is rebuilt from
the source series on load, so saved projects cannot carry a stale copy of the
fitted values.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from asymmetry.core.fitting.component_tracking import (
    Component,
    CrossingEvent,
    ScanPoint,
    detect_crossings,
)
from asymmetry.core.fitting.knight_shift import (
    REFERENCE_APPLIED_FIELD,
    KnightShiftConfig,
    KnightShiftUnit,
    concrete_unit,
    knight_shift,
    label_for_unit,
    larmor_frequency_mhz,
    scale_for_unit,
)
from asymmetry.core.fitting.parameters import split_parameter_name

#: Oscillation components convertible to a Knight shift, by base parameter name.
#: ``frequency`` components are in MHz (referenced to γ_µ·B); ``field``
#: components are the fitted local field in Gauss (referenced to B itself).
KNIGHT_COMPONENT_KINDS = ("frequency", "field")


def _finite_or_nan(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


@dataclass(frozen=True)
class KnightPoint:
    """One source run's contribution to the analysis snapshot.

    ``x`` is the scan abscissa (already resolved by the caller — for a rotation
    scan the unfolded angle in degrees; NaN drops the point). ``values`` /
    ``errors`` carry the fitted component parameters; ``covariance`` is the
    (optional) nested parameter-covariance map used by the designated-component
    reference. ``include`` mirrors the trend include-gate so excluded runs stay
    visible-but-skipped rather than silently vanishing.
    """

    run_number: int
    run_label: str
    x: float
    field_gauss: float
    values: Mapping[str, float]
    errors: Mapping[str, float]
    covariance: Mapping[str, Mapping[str, float]] | None = None
    include: bool = True


@dataclass(frozen=True)
class KnightAnalysisInput:
    """Immutable snapshot of everything :func:`evaluate` reads.

    ``components`` lists the convertible components as ``(name, kind)`` in model
    order (kind per :data:`KNIGHT_COMPONENT_KINDS`); it is the source series'
    model-derived observable map when available, so muonium components whose
    ``field`` is the *applied* field are already excluded. ``x_key``/``x_label``
    identify the scan axis for display and persistence; ``source_label`` names
    the originating series/batch for the window header.
    """

    x_key: str
    x_label: str
    components: tuple[tuple[str, str], ...]
    points: tuple[KnightPoint, ...]
    source_label: str = ""
    batch_id: str | None = None
    group_id: str | None = None

    def component_names(self) -> tuple[str, ...]:
        return tuple(name for name, _kind in self.components)


@dataclass(frozen=True)
class KnightBranch:
    """One component's Knight-shift trace along the scan.

    Values are dimensionless fractions; the result-level unit applies a display
    scale. Entries align with ``run_numbers`` (points with a NaN abscissa or a
    missing component are dropped and counted at the result level; excluded
    points are kept, flagged by ``included``, so the GUI can grey them out).
    """

    name: str
    component: str
    kind: str
    subscript: str
    x: tuple[float, ...]
    k: tuple[float, ...]
    k_err: tuple[float, ...]
    run_numbers: tuple[int, ...]
    included: tuple[bool, ...]


@dataclass(frozen=True)
class KnightAnalysisResult:
    """Output of :func:`evaluate`: branches, crossings, and the resolved unit."""

    unit: KnightShiftUnit
    unit_label: str
    scale: float
    branches: tuple[KnightBranch, ...]
    crossings: tuple[CrossingEvent, ...]
    #: Points dropped from every branch: non-finite abscissa or missing component.
    skipped_points: int = 0

    def branch(self, name: str) -> KnightBranch | None:
        for candidate in self.branches:
            if candidate.name == name:
                return candidate
        return None


def branch_name(component: str) -> str:
    """Trace name for a component's Knight shift (``K[<component>]``)."""
    return f"K[{component}]"


def _component_subscript(component: str) -> str:
    _base, index = split_parameter_name(component)
    return index if index is not None else "1"


def _point_shift(
    point: KnightPoint,
    component: str,
    kind: str,
    config: KnightShiftConfig,
) -> tuple[float, float]:
    """Dimensionless K (and σ) for one component on one point.

    Mirrors the reference conventions of
    :func:`asymmetry.core.fitting.knight_shift.knight_shift`: the applied-field
    reference is exact (γ_µ·B for a frequency component, B itself for a local
    field component); the designated-component reference carries the fitted
    reference's uncertainty and its covariance with the component.
    """
    nu = point.values.get(component)
    if nu is None:
        return float("nan"), float("nan")
    sigma_nu = _finite_or_nan(point.errors.get(component, 0.0))
    if not math.isfinite(sigma_nu):
        sigma_nu = 0.0
    if config.reference_mode == REFERENCE_APPLIED_FIELD:
        nu_ref = point.field_gauss if kind == "field" else larmor_frequency_mhz(point.field_gauss)
        return knight_shift(float(nu), nu_ref, sigma_nu=sigma_nu)
    reference = config.reference_component
    nu_ref = point.values.get(reference) if reference else None
    if nu_ref is None:
        return float("nan"), float("nan")
    sigma_ref = _finite_or_nan(point.errors.get(reference, 0.0))
    if not math.isfinite(sigma_ref):
        sigma_ref = 0.0
    cov = 0.0
    if point.covariance is not None:
        cov = _finite_or_nan(point.covariance.get(component, {}).get(reference, 0.0))
        if not math.isfinite(cov):
            cov = 0.0
    return knight_shift(float(nu), float(nu_ref), sigma_nu=sigma_nu, sigma_ref=sigma_ref, cov=cov)


def selected_components(
    analysis_input: KnightAnalysisInput, config: KnightShiftConfig
) -> tuple[tuple[str, str], ...]:
    """The ``(name, kind)`` components the config actually converts.

    Applies the config's component subset (empty = all) and, for the
    designated-component reference, restricts to components of the *same kind*
    as the reference — dividing a frequency (MHz) by a field (Gauss) is
    meaningless — excluding the reference itself. Returns an empty tuple when
    the designated reference is not among the snapshot's components (emit
    nothing rather than guess).
    """
    components = [
        (name, kind)
        for name, kind in analysis_input.components
        if kind in KNIGHT_COMPONENT_KINDS and (not config.components or name in config.components)
    ]
    if config.reference_mode == REFERENCE_APPLIED_FIELD:
        return tuple(components)
    kind_by_name = dict(analysis_input.components)
    ref_kind = kind_by_name.get(config.reference_component) if config.reference_component else None
    if ref_kind is None:
        return ()
    return tuple(
        (name, kind)
        for name, kind in components
        if name != config.reference_component and kind == ref_kind
    )


def _detect_scan_crossings(
    analysis_input: KnightAnalysisInput,
) -> tuple[CrossingEvent, ...]:
    """Crossings/degeneracies of the raw components along the scan.

    Components are grouped by kind so the continuity matcher never compares a
    frequency (MHz) against a field (Gauss). Only points carrying *all* of a
    kind's components (with a finite abscissa) participate — a partial point
    would misalign the assignment problem.
    """
    events: list[CrossingEvent] = []
    for kind in KNIGHT_COMPONENT_KINDS:
        names = [name for name, k in analysis_input.components if k == kind]
        if len(names) < 2:
            continue
        points: list[ScanPoint] = []
        for point in sorted(analysis_input.points, key=lambda p: p.x):
            if not math.isfinite(point.x) or not point.include:
                continue
            values = [point.values.get(name) for name in names]
            if any(v is None for v in values):
                continue
            components = tuple(Component(frequency=float(v)) for v in values)  # type: ignore[arg-type]
            points.append(ScanPoint(x=float(point.x), components=components))
        events.extend(detect_crossings(points))
    return tuple(events)


def evaluate(
    analysis_input: KnightAnalysisInput, config: KnightShiftConfig
) -> KnightAnalysisResult:
    """Derive the Knight-shift branches and crossings for a snapshot + config.

    Pure: the snapshot is not mutated, and calling twice with the same inputs
    gives the same result. A disabled config yields an empty result (no
    branches, no crossings) with the unit still resolved so axis labels stay
    stable while the user toggles the conversion.
    """
    components = selected_components(analysis_input, config) if config.enabled else ()

    # First pass computes the dimensionless fractions so AUTO can pick a unit
    # from the full set before any branch is materialised.
    per_component: dict[str, list[tuple[KnightPoint, float, float]]] = {}
    for name, kind in components:
        rows: list[tuple[KnightPoint, float, float]] = []
        for point in sorted(analysis_input.points, key=lambda p: p.x):
            if not math.isfinite(point.x) or name not in point.values:
                continue
            k, sigma_k = _point_shift(point, name, kind, config)
            rows.append((point, k, sigma_k))
        per_component[name] = rows

    # A point is "skipped" only when no branch retains it (NaN abscissa, or all
    # selected components missing); a point missing just one component still
    # contributes to the others and is not counted.
    retained = {point.run_number for rows in per_component.values() for point, _k, _s in rows}
    skipped = (
        {point.run_number for point in analysis_input.points} - retained if components else set()
    )

    all_fractions = [k for rows in per_component.values() for _point, k, _s in rows]
    unit = concrete_unit(config.unit, all_fractions)

    branches: list[KnightBranch] = []
    for name, kind in components:
        rows = per_component[name]
        branches.append(
            KnightBranch(
                name=branch_name(name),
                component=name,
                kind=kind,
                subscript=_component_subscript(name),
                x=tuple(point.x for point, _k, _s in rows),
                k=tuple(k for _point, k, _s in rows),
                k_err=tuple(s for _point, _k, s in rows),
                run_numbers=tuple(point.run_number for point, _k, _s in rows),
                included=tuple(point.include for point, _k, _s in rows),
            )
        )

    # Crossings are a property of the raw scan, not of the conversion choices:
    # detect them whenever the analysis is enabled, even if a misconfigured
    # reference leaves nothing to convert (the flags still warn the user).
    crossings = _detect_scan_crossings(analysis_input) if config.enabled else ()
    return KnightAnalysisResult(
        unit=unit,
        unit_label=label_for_unit(unit),
        scale=scale_for_unit(unit),
        branches=tuple(branches),
        crossings=crossings,
        skipped_points=len(skipped),
    )


# ── Joint K(θ) fit with per-angle assignment ─────────────────────────────────


@dataclass(frozen=True)
class KnightJointCurve:
    """One fitted physical curve of a joint K(θ) fit.

    ``branch_name`` is the ``K[...]`` trace this curve occupies after
    realignment. ``parameters`` are ``(name, value, error)`` triples in the fit
    unit (see :attr:`KnightJointFitState.unit`).
    """

    branch_name: str
    parameters: tuple[tuple[str, float, float], ...]
    chi_squared: float
    reduced_chi_squared: float
    n_points: int
    success: bool

    def to_dict(self) -> dict:
        return {
            "branch_name": self.branch_name,
            "parameters": [[n, v, e] for n, v, e in self.parameters],
            "chi_squared": float(self.chi_squared),
            "reduced_chi_squared": float(self.reduced_chi_squared),
            "n_points": int(self.n_points),
            "success": bool(self.success),
        }

    @classmethod
    def from_dict(cls, data: object) -> KnightJointCurve | None:
        if not isinstance(data, dict):
            return None
        raw_params = data.get("parameters")
        parameters: list[tuple[str, float, float]] = []
        if isinstance(raw_params, list):
            for entry in raw_params:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    parameters.append(
                        (str(entry[0]), _finite_or_nan(entry[1]), _finite_or_nan(entry[2]))
                    )
        return cls(
            branch_name=str(data.get("branch_name") or ""),
            parameters=tuple(parameters),
            chi_squared=_finite_or_nan(data.get("chi_squared", float("nan"))),
            reduced_chi_squared=_finite_or_nan(data.get("reduced_chi_squared", float("nan"))),
            n_points=int(data.get("n_points", 0) or 0),
            success=bool(data.get("success", False)),
        )


@dataclass
class KnightJointFitState:
    """Persisted joint K(θ) fit: model, per-run assignment, per-curve parameters.

    ``assignment[run_number][component]`` is the curve index that component was
    assigned to at that run — the durable representation (run numbers survive
    refits and reordering; scan-point indices would not). ``unit`` records the
    concrete display unit the fit ran in: the assignment is unit-independent,
    but the curve *parameters* are not, so a unit change marks the curves stale
    until the fit is re-run.
    """

    model_name: str = "KnightAnisotropy"
    max_iter: int = 25
    unit: str = KnightShiftUnit.PERCENT.value
    converged: bool = False
    total_chi_squared: float = 0.0
    dof: int = 0
    message: str = ""
    assignment: dict[int, tuple[int, ...]] = field(default_factory=dict)
    curves: tuple[KnightJointCurve, ...] = ()

    def to_dict(self) -> dict:
        return {
            "model_name": str(self.model_name),
            "max_iter": int(self.max_iter),
            "unit": str(self.unit),
            "converged": bool(self.converged),
            "total_chi_squared": float(self.total_chi_squared),
            "dof": int(self.dof),
            "message": str(self.message),
            "assignment": {str(run): list(perm) for run, perm in self.assignment.items()},
            "curves": [curve.to_dict() for curve in self.curves],
        }

    @classmethod
    def from_dict(cls, data: object) -> KnightJointFitState | None:
        if not isinstance(data, dict):
            return None
        assignment: dict[int, tuple[int, ...]] = {}
        raw_assignment = data.get("assignment")
        if isinstance(raw_assignment, dict):
            for run, perm in raw_assignment.items():
                try:
                    assignment[int(run)] = tuple(int(c) for c in perm)
                except (TypeError, ValueError):
                    continue
        curves = []
        raw_curves = data.get("curves")
        if isinstance(raw_curves, list):
            curves = [c for c in (KnightJointCurve.from_dict(entry) for entry in raw_curves) if c]
        return cls(
            model_name=str(data.get("model_name") or "KnightAnisotropy"),
            max_iter=int(data.get("max_iter", 25) or 25),
            unit=str(data.get("unit") or KnightShiftUnit.PERCENT.value),
            converged=bool(data.get("converged", False)),
            total_chi_squared=_finite_or_nan(data.get("total_chi_squared", 0.0)),
            dof=int(data.get("dof", 0) or 0),
            message=str(data.get("message") or ""),
            assignment=assignment,
            curves=tuple(curves),
        )


def _joint_fit_matrices(
    result: KnightAnalysisResult,
) -> tuple[list[int], list[float], list[list[float]], list[list[float]]]:
    """Aligned per-point matrices for the joint fit.

    Only runs present (and included) in *every* branch participate — the
    one-to-one assignment problem needs the full component set at each scan
    point. Returns ``(run_numbers, angles, values, errors)`` sorted by angle,
    with values/errors in fractions (``[point][component]``).
    """
    branches = result.branches
    if not branches:
        return [], [], [], []
    per_branch: list[dict[int, tuple[float, float, float, bool]]] = [
        {
            run: (x, k, e, inc)
            for run, x, k, e, inc in zip(b.run_numbers, b.x, b.k, b.k_err, b.included)
        }
        for b in branches
    ]
    shared = set(per_branch[0])
    for mapping in per_branch[1:]:
        shared &= set(mapping)
    rows: list[tuple[float, int, list[float], list[float]]] = []
    for run in shared:
        if not all(mapping[run][3] for mapping in per_branch):
            continue  # excluded on some branch: keep it out of the fit entirely
        x = per_branch[0][run][0]
        rows.append(
            (
                x,
                run,
                [mapping[run][1] for mapping in per_branch],
                [mapping[run][2] for mapping in per_branch],
            )
        )
    rows.sort(key=lambda item: item[0])
    return (
        [run for _x, run, _v, _e in rows],
        [x for x, _run, _v, _e in rows],
        [v for _x, _run, v, _e in rows],
        [e for _x, _run, _v, e in rows],
    )


def run_joint_fit(
    result: KnightAnalysisResult,
    *,
    model_name: str = "KnightAnisotropy",
    max_iter: int = 25,
) -> KnightJointFitState:
    """Jointly fit all branches' K(θ) with per-angle component assignment.

    A thin bridge to :func:`asymmetry.core.fitting.angular_assignment.
    fit_assigned_angular_curves`: builds the aligned matrices in the result's
    display unit (so curve parameters read in the plotted unit), runs the
    classification-EM fit, and re-keys the assignment by run number.
    Raises ``ValueError`` with fewer than two branches or two shared points.
    """
    from asymmetry.core.fitting.angular_assignment import fit_assigned_angular_curves

    if len(result.branches) < 2:
        raise ValueError("The joint K(θ) fit needs at least two Knight-shift branches")
    runs, angles, values, errors = _joint_fit_matrices(result)
    if len(angles) < 2:
        raise ValueError("The joint K(θ) fit needs at least two scan points shared by all branches")

    scale = result.scale
    scaled_values = [[k * scale for k in row] for row in values]
    scaled_errors = [[e * scale for e in row] for row in errors]
    outcome = fit_assigned_angular_curves(
        angles, scaled_values, scaled_errors, model_name=model_name, max_iter=max_iter
    )

    curves = []
    for branch, fit in zip(result.branches, outcome.curves):
        parameters = tuple(
            (
                p.name,
                float(p.value),
                _finite_or_nan(fit.uncertainties.get(p.name, float("nan"))),
            )
            for p in fit.parameters
        )
        curves.append(
            KnightJointCurve(
                branch_name=branch.name,
                parameters=parameters,
                chi_squared=float(fit.chi_squared),
                reduced_chi_squared=float(fit.reduced_chi_squared),
                n_points=int(fit.n_points) or len(angles),
                success=bool(fit.success),
            )
        )
    return KnightJointFitState(
        model_name=model_name,
        max_iter=int(max_iter),
        unit=result.unit.value,
        converged=bool(outcome.converged),
        total_chi_squared=float(outcome.total_chi_squared),
        dof=int(outcome.dof),
        message=str(outcome.message or ""),
        assignment={run: tuple(perm) for run, perm in zip(runs, outcome.assignment)},
        curves=tuple(curves),
    )


def apply_assignment(
    result: KnightAnalysisResult, joint: KnightJointFitState
) -> KnightAnalysisResult:
    """Realign the branches so each follows its physical curve through crossings.

    For every run in the joint fit's assignment, the component values are
    permuted onto the branch of the curve they were assigned to
    (``perm[component] = curve``). Runs outside the assignment (new points, or
    points not shared by all branches at fit time) keep their raw labels. The
    input result is not mutated.
    """
    branches = result.branches
    n = len(branches)
    if n < 2 or not joint.assignment:
        return result
    per_branch: list[dict[int, tuple[float, float]]] = [
        {run: (k, e) for run, k, e in zip(b.run_numbers, b.k, b.k_err)} for b in branches
    ]
    new_k: list[list[float]] = [list(b.k) for b in branches]
    new_e: list[list[float]] = [list(b.k_err) for b in branches]
    index_of_run = [{run: i for i, run in enumerate(b.run_numbers)} for b in branches]
    for run, perm in joint.assignment.items():
        if len(perm) != n or not all(run in mapping for mapping in per_branch):
            continue
        for component, curve in enumerate(perm):
            target = index_of_run[curve].get(run)
            if target is None:
                continue
            k, e = per_branch[component][run]
            new_k[curve][target] = k
            new_e[curve][target] = e
    realigned = tuple(
        KnightBranch(
            name=b.name,
            component=b.component,
            kind=b.kind,
            subscript=b.subscript,
            x=b.x,
            k=tuple(new_k[i]),
            k_err=tuple(new_e[i]),
            run_numbers=b.run_numbers,
            included=b.included,
        )
        for i, b in enumerate(branches)
    )
    return KnightAnalysisResult(
        unit=result.unit,
        unit_label=result.unit_label,
        scale=result.scale,
        branches=realigned,
        crossings=result.crossings,
        skipped_points=result.skipped_points,
    )


def assignment_swap_positions(
    result: KnightAnalysisResult, joint: KnightJointFitState
) -> tuple[float, ...]:
    """Scan positions (midpoints) where the fitted assignment swaps curves.

    These are the crossings the joint fit actually resolved — a firmer signal
    than the raw proximity flags, so the window marks these when a fit is
    active. Points not covered by the assignment are skipped.
    """
    if not result.branches or not joint.assignment:
        return ()
    reference = result.branches[0]
    ordered = sorted(
        (
            (x, joint.assignment[run])
            for x, run in zip(reference.x, reference.run_numbers)
            if run in joint.assignment
        ),
        key=lambda item: item[0],
    )
    swaps = []
    for (x_left, perm_left), (x_right, perm_right) in zip(ordered, ordered[1:]):
        if perm_left != perm_right:
            swaps.append(0.5 * (x_left + x_right))
    return tuple(swaps)


@dataclass
class KnightAnalysisState:
    """Persisted window state: conversion config + source binding + view hints.

    Deliberately excludes the point snapshot (rebuilt from the source series on
    open — a saved copy could go stale against a refit). ``source_batch_id`` /
    ``source_group_id`` re-bind the window to its series; ``x_key`` pins the
    scan axis. ``fold_180`` and ``show_markers`` are plot-view preferences.
    ``joint`` carries the (optional) joint K(θ) fit; its run-keyed assignment
    stays valid across snapshot rebuilds.
    """

    config: KnightShiftConfig = field(default_factory=KnightShiftConfig)
    source_batch_id: str | None = None
    source_group_id: str | None = None
    x_key: str = "angle"
    fold_180: bool = False
    show_markers: bool = True
    joint: KnightJointFitState | None = None

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "source_batch_id": self.source_batch_id,
            "source_group_id": self.source_group_id,
            "x_key": str(self.x_key),
            "fold_180": bool(self.fold_180),
            "show_markers": bool(self.show_markers),
            "joint": self.joint.to_dict() if self.joint is not None else None,
        }

    @classmethod
    def from_dict(cls, data: object) -> KnightAnalysisState:
        if not isinstance(data, dict):
            return cls()
        batch_id = data.get("source_batch_id")
        group_id = data.get("source_group_id")
        return cls(
            config=KnightShiftConfig.from_dict(data.get("config")),
            source_batch_id=str(batch_id) if batch_id else None,
            source_group_id=str(group_id) if group_id else None,
            x_key=str(data.get("x_key") or "angle"),
            fold_180=bool(data.get("fold_180", False)),
            show_markers=bool(data.get("show_markers", True)),
            joint=KnightJointFitState.from_dict(data.get("joint")),
        )


def migrate_legacy_state(fit_parameters_state: object) -> KnightAnalysisState | None:
    """Lift a pre-window trend-panel Knight-shift block into the new state.

    Projects saved before the analysis window stored the conversion config under
    ``fit_parameters_state["knight_shift"]`` (with the scan axis in
    ``x_axis_key``). Returns ``None`` when there is nothing to migrate (missing
    block or conversion never enabled) so callers can skip writing an empty
    state; the legacy block itself is left untouched for older app versions.
    """
    if not isinstance(fit_parameters_state, dict):
        return None
    legacy = fit_parameters_state.get("knight_shift")
    if not isinstance(legacy, dict) or not legacy.get("enabled"):
        return None
    config = KnightShiftConfig.from_dict(legacy)
    x_key = str(fit_parameters_state.get("x_axis_key") or "angle")
    active_group = fit_parameters_state.get("active_group_id")
    return KnightAnalysisState(
        config=config,
        source_group_id=str(active_group) if active_group else None,
        x_key=x_key,
        joint=_migrate_legacy_joint_fit(fit_parameters_state.get("joint_fit"), config),
    )


def _migrate_legacy_joint_fit(
    legacy: object, config: KnightShiftConfig
) -> KnightJointFitState | None:
    """Lift a legacy trend-panel ``joint_fit`` block into a joint-fit state.

    The run-keyed assignment (the durable, valuable part) migrates exactly. The
    per-curve parameters are lifted where present, but the *unit* they were
    fitted in is recorded as the legacy config's unit — for an ``auto`` unit
    that never matches a concrete display unit, so migrated curves render as
    stale ("re-run to refresh") rather than risking a wrongly-scaled overlay.
    """
    if not isinstance(legacy, dict):
        return None
    raw_assignment = legacy.get("assignment")
    if not isinstance(raw_assignment, dict) or not raw_assignment:
        return None
    assignment: dict[int, tuple[int, ...]] = {}
    for run, perm in raw_assignment.items():
        try:
            assignment[int(run)] = tuple(int(c) for c in perm)
        except (TypeError, ValueError):
            continue
    if not assignment:
        return None

    curves: list[KnightJointCurve] = []
    raw_curves = legacy.get("curves")
    if isinstance(raw_curves, dict):
        for trace in legacy.get("traces") or raw_curves.keys():
            entry = raw_curves.get(str(trace))
            ranges = entry.get("ranges") if isinstance(entry, dict) else None
            first = ranges[0] if isinstance(ranges, list) and ranges else None
            if not isinstance(first, dict):
                continue
            result = first.get("result") if isinstance(first.get("result"), dict) else {}
            uncertainties = (
                result.get("uncertainties") if isinstance(result.get("uncertainties"), dict) else {}
            )
            parameters = []
            for param in first.get("parameters") or []:
                if isinstance(param, dict) and "name" in param:
                    name = str(param["name"])
                    parameters.append(
                        (
                            name,
                            _finite_or_nan(param.get("value")),
                            _finite_or_nan(uncertainties.get(name, float("nan"))),
                        )
                    )
            curves.append(
                KnightJointCurve(
                    branch_name=str(trace),
                    parameters=tuple(parameters),
                    chi_squared=_finite_or_nan(result.get("chi_squared", float("nan"))),
                    reduced_chi_squared=_finite_or_nan(
                        result.get("reduced_chi_squared", float("nan"))
                    ),
                    n_points=int(result.get("n_points", 0) or 0),
                    success=bool(result.get("success", False)),
                )
            )

    return KnightJointFitState(
        model_name=str(legacy.get("model_name") or "KnightAnisotropy"),
        unit=config.unit.value,
        converged=True,
        message="Migrated from a saved trend-panel joint fit.",
        assignment=assignment,
        curves=tuple(curves),
    )


def snapshot_from_rows(
    rows: Sequence[object],
    *,
    x_values: Sequence[float],
    x_key: str,
    x_label: str,
    components: Sequence[tuple[str, str]],
    source_label: str = "",
    batch_id: str | None = None,
    group_id: str | None = None,
) -> KnightAnalysisInput:
    """Build a snapshot from trend-row-shaped objects.

    ``rows`` are duck-typed against the trend panel's row records (attributes
    ``run_number``, ``run_label``, ``field``, ``values``, ``errors``,
    ``covariance``, ``include_in_trend``); ``x_values`` supplies the resolved
    (unfolded) abscissa for each row, letting the caller keep ownership of the
    axis resolution. Rows and abscissae must align.
    """
    if len(rows) != len(x_values):
        raise ValueError("rows and x_values must have the same length")
    points = []
    for row, x in zip(rows, x_values):
        covariance = getattr(row, "covariance", None)
        points.append(
            KnightPoint(
                run_number=int(getattr(row, "run_number", 0)),
                run_label=str(getattr(row, "run_label", "")),
                x=_finite_or_nan(x),
                field_gauss=_finite_or_nan(getattr(row, "field", float("nan"))),
                values=dict(getattr(row, "values", {}) or {}),
                errors=dict(getattr(row, "errors", {}) or {}),
                covariance=covariance if isinstance(covariance, Mapping) else None,
                include=bool(getattr(row, "include_in_trend", True)),
            )
        )
    return KnightAnalysisInput(
        x_key=str(x_key),
        x_label=str(x_label),
        components=tuple((str(n), str(k)) for n, k in components),
        points=tuple(points),
        source_label=str(source_label),
        batch_id=batch_id,
        group_id=group_id,
    )
