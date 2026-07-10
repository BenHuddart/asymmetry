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
from typing import TYPE_CHECKING

import numpy as np

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

if TYPE_CHECKING:
    from asymmetry.core.fitting.angular_assignment import AngularAssignmentResult
    from asymmetry.core.fitting.experiment_design import NextPointSuggestion

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


#: Demagnetization factor N along the applied field for standard sample shapes
#: (SI convention, ΣN = 1). ``custom`` defers to the user-entered value.
SAMPLE_SHAPE_DEMAG_FACTORS: dict[str, float | None] = {
    "sphere": 1.0 / 3.0,
    "plate_parallel": 0.0,
    "plate_perpendicular": 1.0,
    "cylinder_axial": 0.0,
    "cylinder_transverse": 0.5,
    "custom": None,
}


@dataclass
class KnightCorrection:
    """Lorentz/demagnetizing-field correction to the measured shift.

    Applies Amato & Morenzoni Eq. 5.60: ``K_µ = K_exp − (1/3 − N)·χ``, with
    ``N`` the demagnetization factor along the applied field (SI convention,
    sphere = 1/3 — for which Lorentz and demagnetizing fields cancel and the
    correction vanishes) and ``chi`` the *volume* susceptibility in SI
    dimensionless units (multiply a CGS emu/cm³ value by 4π). The correction is
    a constant offset per branch, exact for an ellipsoidal sample whose shape
    is fixed relative to the field; for a rotating non-spheroidal sample N
    itself varies with angle, which this scalar form does not capture.
    """

    enabled: bool = False
    shape: str = "sphere"
    custom_n: float = 1.0 / 3.0
    chi_volume_si: float = 0.0

    def demag_factor(self) -> float:
        factor = SAMPLE_SHAPE_DEMAG_FACTORS.get(self.shape)
        return float(self.custom_n) if factor is None else float(factor)

    def offset(self) -> float:
        """The additive correction to a dimensionless K fraction (Eq. 5.60)."""
        if not self.enabled:
            return 0.0
        offset = -(1.0 / 3.0 - self.demag_factor()) * float(self.chi_volume_si)
        return offset if math.isfinite(offset) else 0.0

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "shape": str(self.shape),
            "custom_n": float(self.custom_n),
            "chi_volume_si": float(self.chi_volume_si),
        }

    @classmethod
    def from_dict(cls, data: object) -> KnightCorrection:
        if not isinstance(data, dict):
            return cls()
        shape = str(data.get("shape") or "sphere")
        if shape not in SAMPLE_SHAPE_DEMAG_FACTORS:
            shape = "sphere"
        return cls(
            enabled=bool(data.get("enabled", False)),
            shape=shape,
            custom_n=_finite_or_nan(data.get("custom_n", 1.0 / 3.0)),
            chi_volume_si=_finite_or_nan(data.get("chi_volume_si", 0.0)),
        )


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
    analysis_input: KnightAnalysisInput,
    config: KnightShiftConfig,
    correction: KnightCorrection | None = None,
) -> KnightAnalysisResult:
    """Derive the Knight-shift branches and crossings for a snapshot + config.

    Pure: the snapshot is not mutated, and calling twice with the same inputs
    gives the same result. A disabled config yields an empty result (no
    branches, no crossings) with the unit still resolved so axis labels stay
    stable while the user toggles the conversion. ``correction`` (optional)
    applies the Lorentz/demagnetizing offset of Eq. 5.60 to every shift; the
    offset is common to all branches, so crossings and branch ordering are
    unaffected. K uncertainties do not include a χ uncertainty.
    """
    components = selected_components(analysis_input, config) if config.enabled else ()
    correction_offset = correction.offset() if correction is not None else 0.0

    # First pass computes the dimensionless fractions so AUTO can pick a unit
    # from the full set before any branch is materialised.
    per_component: dict[str, list[tuple[KnightPoint, float, float]]] = {}
    for name, kind in components:
        rows: list[tuple[KnightPoint, float, float]] = []
        for point in sorted(analysis_input.points, key=lambda p: p.x):
            if not math.isfinite(point.x) or name not in point.values:
                continue
            k, sigma_k = _point_shift(point, name, kind, config)
            rows.append((point, k + correction_offset, sigma_k))
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


def _parse_joint_covariance(
    raw: object,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]] | None:
    """Parse a ``{"names": [...], "matrix": [[...]]}`` covariance block.

    Tolerant: returns ``None`` for a missing/legacy key or any malformed shape
    (non-square, row length mismatch, non-numeric entry) rather than raising.
    Non-finite numeric entries (``nan``/``inf``) are preserved as-is — they are
    numbers, just not usable ones; only a non-numeric entry invalidates.
    """
    if not isinstance(raw, dict):
        return None
    raw_names = raw.get("names")
    raw_matrix = raw.get("matrix")
    if not isinstance(raw_names, list) or not isinstance(raw_matrix, list):
        return None
    n = len(raw_names)
    if n == 0 or len(raw_matrix) != n:
        return None
    try:
        names = tuple(str(name) for name in raw_names)
    except (TypeError, ValueError):
        return None
    rows: list[tuple[float, ...]] = []
    for row in raw_matrix:
        if not isinstance(row, list) or len(row) != n:
            return None
        try:
            rows.append(tuple(float(v) for v in row))
        except (TypeError, ValueError):
            return None
    return names, tuple(rows)


@dataclass(frozen=True)
class KnightJointCurve:
    """One fitted physical curve of a joint K(θ) fit.

    ``branch_name`` is the ``K[...]`` trace this curve occupies after
    realignment. ``parameters`` are ``(name, value, error)`` triples in the fit
    unit (see :attr:`KnightJointFitState.unit`). ``covariance`` mirrors
    :attr:`ParameterModelFitResult.covariance` — ``(names, matrix)`` for the
    curve's free parameters, in the same order as ``names``. It is already in
    the display unit: :func:`run_joint_fit` scales the values into the display
    unit *before* fitting, so the Minuit covariance it reads back is already
    unit² in that display unit — no extra scaling is applied here. ``None``
    when the underlying fit produced no covariance (HESSE failed/didn't run)
    or the curve predates this field (legacy project).
    """

    branch_name: str
    parameters: tuple[tuple[str, float, float], ...]
    chi_squared: float
    reduced_chi_squared: float
    n_points: int
    success: bool
    covariance: tuple[tuple[str, ...], tuple[tuple[float, ...], ...]] | None = None

    def to_dict(self) -> dict:
        data: dict = {
            "branch_name": self.branch_name,
            "parameters": [[n, v, e] for n, v, e in self.parameters],
            "chi_squared": float(self.chi_squared),
            "reduced_chi_squared": float(self.reduced_chi_squared),
            "n_points": int(self.n_points),
            "success": bool(self.success),
        }
        if self.covariance is not None:
            names, matrix = self.covariance
            data["covariance"] = {
                "names": list(names),
                "matrix": [list(row) for row in matrix],
            }
        return data

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
            covariance=_parse_joint_covariance(data.get("covariance")),
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
    #: Lorentz/demag offset (fraction units) active when the fit ran — like
    #: ``unit``, a bookkeeping value: a changed correction shifts every K, so
    #: the fitted curves go stale (the assignment does not — a common offset
    #: cannot reorder branches).
    correction_offset: float = 0.0
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
            "correction_offset": float(self.correction_offset),
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
            correction_offset=_finite_or_nan(data.get("correction_offset", 0.0)),
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
    correction_offset: float = 0.0,
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
        covariance = None
        if fit.covariance is not None:
            cov_names, cov_matrix = fit.covariance
            covariance = (tuple(cov_names), tuple(tuple(row) for row in cov_matrix))
        curves.append(
            KnightJointCurve(
                branch_name=branch.name,
                parameters=parameters,
                chi_squared=float(fit.chi_squared),
                reduced_chi_squared=float(fit.reduced_chi_squared),
                n_points=int(fit.n_points) or len(angles),
                success=bool(fit.success),
                covariance=covariance,
            )
        )
    return KnightJointFitState(
        model_name=model_name,
        max_iter=int(max_iter),
        unit=result.unit.value,
        correction_offset=float(correction_offset),
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


# ── Next-angle BED bridges (Phase 4) ─────────────────────────────────────────
#
# Thin, GUI-free adapters that assemble the Knight-specific series inputs and
# delegate the acquisition math to ``experiment_design`` (the multi-series IG
# sum, the labelled model-discrimination sum, and the Hungarian set-matching
# assignment-discrimination utility), returning the shared ``NextPointSuggestion``
# so the GUI plumbing stays uniform. Curve parameters/covariance are in the
# display unit (``run_joint_fit`` scales before fitting); the branch traces are
# fractions, so per-curve noise is scaled by ``result.scale``.


def _curve_series(
    result: KnightAnalysisResult, joint: KnightJointFitState
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-curve realigned ``(angles, values, errors)`` in the display unit.

    Reconstructs each fitted curve's continuous trace from the joint fit's
    run-keyed assignment over exactly the points that entered the fit
    (``_joint_fit_matrices`` — shared, included points), scaling the fraction
    values/errors into the display unit. Curve index ``m`` corresponds to
    ``result.branches[m]`` (the order ``run_joint_fit`` fitted).
    """
    runs, angles, values, errors = _joint_fit_matrices(result)
    scale = result.scale
    n_curves = len(result.branches)
    angles_arr = np.asarray(angles, dtype=float)
    per_curve: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for curve in range(n_curves):
        curve_values: list[float] = []
        curve_errors: list[float] = []
        for point, run in enumerate(runs):
            perm = joint.assignment.get(run)
            component = perm.index(curve) if perm is not None and len(perm) == n_curves else curve
            curve_values.append(values[point][component] * scale)
            curve_errors.append(errors[point][component] * scale)
        per_curve.append(
            (
                angles_arr,
                np.asarray(curve_values, dtype=float),
                np.asarray(curve_errors, dtype=float),
            )
        )
    return per_curve


def suggest_next_angle(
    result: KnightAnalysisResult,
    joint: KnightJointFitState,
    *,
    x_min: float,
    x_max: float,
    target: tuple[str, str] | None = None,
    sigma_goal: float | None = None,
    n_candidates: int = 257,
) -> NextPointSuggestion:
    """Suggest the next scan angle that best constrains the joint K(θ) fit (§3.1).

    Builds one series per fitted curve (model from ``joint.model_name``,
    parameters/covariance from the curve, empirical noise from its realigned
    trace) and delegates to
    :func:`~asymmetry.core.fitting.experiment_design.suggest_next_point_multi`.
    ``target`` names ``(branch_name, param_name)`` for a c-optimal solve (with
    ``sigma_goal``); ``None`` gives the D-optimal information-gain sum. Curves
    without stored covariance degrade with a warning naming the branch; when
    *every* curve lacks covariance the suggestion is empty with a "re-run the
    joint fit" warning (legacy fits predate stored covariance).
    """
    from asymmetry.core.fitting.experiment_design import (
        SeriesSpec,
        _empty_suggestion,
        suggest_next_point_multi,
    )
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel
    from asymmetry.core.fitting.parameters import Parameter, ParameterSet

    target_param = target[1] if target is not None else None
    if len(result.branches) < 2 or len(joint.curves) < 2:
        return _empty_suggestion(
            target_param, ["The joint K(θ) fit needs at least two curves to suggest a next angle."]
        )
    if all(curve.covariance is None for curve in joint.curves):
        return _empty_suggestion(
            target_param,
            [
                "No stored fit covariance — re-run the joint fit to enable "
                "next-angle suggestions (legacy fits predate stored covariance)."
            ],
        )

    branch_index = {branch.name: i for i, branch in enumerate(result.branches)}
    series_data = _curve_series(result, joint)
    model = ParameterCompositeModel([joint.model_name])
    specs: list[SeriesSpec] = []
    labels: list[str] = []
    for curve in joint.curves:
        index = branch_index.get(curve.branch_name)
        if index is None:
            continue
        angles_i, _values_i, errors_i = series_data[index]
        params = ParameterSet([Parameter(name, value) for name, value, _err in curve.parameters])
        covariance = None
        if curve.covariance is not None:
            names, matrix = curve.covariance
            covariance = (list(names), [list(row) for row in matrix])
        specs.append(
            SeriesSpec(
                model=model,
                parameters=params,
                covariance=covariance,
                x_data=angles_i,
                y_err=errors_i,
            )
        )
        labels.append(curve.branch_name)

    target_spec: tuple[int, str] | None = None
    if target is not None:
        branch_name, param_name = target
        try:
            target_spec = (labels.index(branch_name), param_name)
        except ValueError:
            return _empty_suggestion(
                param_name, [f"Target branch {branch_name!r} is not among the fitted curves."]
            )

    return suggest_next_point_multi(
        specs,
        x_min,
        x_max,
        target=target_spec,
        sigma_goal=sigma_goal,
        n_candidates=n_candidates,
        labels=labels,
    )


def joint_fit_aic_inputs(joint: KnightJointFitState) -> tuple[float, int]:
    """``(total_chi_squared, n_curves·n_params)`` for AIC-weighting a joint fit.

    Feeds :func:`~asymmetry.core.fitting.experiment_design.aic_weights` so the
    GUI can rank an aligned (lead) fit against a misalignment (alternative) fit
    (§3.2). ``n_params`` is per-curve free-parameter count (all curves share the
    model), multiplied by the number of curves.
    """
    n_curves = len(joint.curves)
    n_params = len(joint.curves[0].parameters) if joint.curves else 0
    return float(joint.total_chi_squared), n_curves * n_params


def suggest_model_discriminating_angle(
    result: KnightAnalysisResult,
    joint_lead: KnightJointFitState,
    joint_alt: KnightJointFitState,
    *,
    x_min: float,
    x_max: float,
    n_candidates: int = 257,
) -> NextPointSuggestion:
    """Suggest the angle best distinguishing an aligned vs misaligned fit (§3.2).

    Curve identity is preserved: curves are matched across the two joint states by
    ``branch_name`` (an error if the branch sets differ), so the utility is the
    *labelled* disagreement sum

        U(θ) = Σ_m [f_m^lead(θ) − f_m^alt(θ)]² / 2σ_m²(θ)

    with no matching step. σ_m is the lead fit's realigned per-curve noise (display
    unit). When the two fits agree within noise everywhere the standard
    "agree within noise" warning is carried (as in
    :func:`~asymmetry.core.fitting.experiment_design.suggest_discriminating_point`).
    """
    from asymmetry.core.fitting.experiment_design import (
        _DISCRIMINATION_NOISE_FLOOR,
        NextPointSuggestion,
        _candidate_grid,
        _empirical_sigma,
        _empty_suggestion,
    )
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

    lead_by_name = {curve.branch_name: curve for curve in joint_lead.curves}
    alt_by_name = {curve.branch_name: curve for curve in joint_alt.curves}
    if not lead_by_name or set(lead_by_name) != set(alt_by_name):
        return _empty_suggestion(
            None, ["The lead and alternative joint fits cover different branches; cannot compare."]
        )
    if len(result.branches) < 2:
        return _empty_suggestion(None, ["The joint K(θ) fit needs at least two curves."])

    branch_index = {branch.name: i for i, branch in enumerate(result.branches)}
    if any(name not in branch_index for name in lead_by_name):
        return _empty_suggestion(None, ["Fitted curves do not match the analysis branches."])
    series_data = _curve_series(result, joint_lead)
    lead_model = ParameterCompositeModel([joint_lead.model_name])
    alt_model = ParameterCompositeModel([joint_alt.model_name])

    warnings: list[str] = []
    reference_name = next(iter(lead_by_name))
    ref_angles, _ref_values, ref_errors = series_data[branch_index[reference_name]]
    grid = _candidate_grid(ref_angles, ref_errors, x_min, x_max, n_candidates, warnings)
    if grid is None:
        return _empty_suggestion(None, warnings)
    candidates, _x_meas, _sigma_new, extrapolated = grid

    utility = np.zeros_like(candidates)
    sigma_stack: list[np.ndarray] = []
    max_variance = 0.0
    for name in lead_by_name:
        angles_i, _values_i, errors_i = series_data[branch_index[name]]
        noise = _empirical_sigma(angles_i, errors_i, [])
        if noise is None:
            continue
        x_meas, sigma_meas = noise
        sigma = np.interp(candidates, x_meas, sigma_meas)
        sigma_stack.append(sigma)
        variance = sigma**2
        max_variance = max(max_variance, float(np.max(variance)))
        lead_values = {n: v for n, v, _e in lead_by_name[name].parameters}
        alt_values = {n: v for n, v, _e in alt_by_name[name].parameters}
        f_lead = np.nan_to_num(
            np.asarray(lead_model.function(candidates, **lead_values), dtype=float)
        )
        f_alt = np.nan_to_num(np.asarray(alt_model.function(candidates, **alt_values), dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            pair = (f_lead - f_alt) ** 2 / (2.0 * variance)
        utility = utility + np.nan_to_num(pair, nan=0.0, posinf=0.0, neginf=0.0)

    sigma_new = (
        np.mean(np.vstack(sigma_stack), axis=0) if sigma_stack else np.zeros_like(candidates)
    )
    relative_floor = _DISCRIMINATION_NOISE_FLOOR * max(max_variance, 1.0)
    if not np.any(utility > relative_floor):
        warnings.append(
            "candidate models agree within noise everywhere in this range — no discriminating point"
        )
        return NextPointSuggestion(
            x_candidates=candidates,
            utility=utility,
            extrapolated=extrapolated,
            best_x=float("nan"),
            target=None,
            sigma_new=sigma_new,
            warnings=tuple(warnings),
        )

    best_index = int(np.argmax(utility))
    return NextPointSuggestion(
        x_candidates=candidates,
        utility=utility,
        extrapolated=extrapolated,
        best_x=float(candidates[best_index]),
        target=None,
        sigma_new=sigma_new,
        warnings=tuple(warnings),
    )


def suggest_assignment_discriminating_angle(
    result: KnightAnalysisResult,
    outcome: AngularAssignmentResult,
    *,
    x_min: float,
    x_max: float,
    n_candidates: int = 257,
) -> NextPointSuggestion:
    """Suggest the angle best resolving competing per-angle assignments (§3.3).

    ``outcome`` is an :class:`~asymmetry.core.fitting.angular_assignment.\
AngularAssignmentResult` carrying near-degenerate runner-up labellings
    (``keep_alternatives > 0``). The utility is the elementwise **max** over
    alternatives of the Hungarian set-matching divergence between the winner's and
    each alternative's predicted value sets (leader-vs-all — the TAS-AI lock-in
    lesson): it is ~0 where the labellings coincide (e.g. at a crossing) and peaks
    where they imply genuinely different curve sets. With no alternatives the
    suggestion is empty ("no near-degenerate assignments to discriminate").

    Per-curve noise comes from the winner's realigned trace errors
    (``outcome.curve_errors``); because the utility is a value²/σ² ratio it is
    invariant to the common display-unit scale, so ``result`` is used only to
    confirm the curve/branch counts agree.
    """
    from asymmetry.core.fitting.experiment_design import (
        _DISCRIMINATION_NOISE_FLOOR,
        NextPointSuggestion,
        _candidate_grid,
        _empirical_sigma,
        _empty_suggestion,
        set_matching_divergence,
    )
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

    n_curves = len(outcome.curves)
    if n_curves < 2 or len(result.branches) < 2:
        return _empty_suggestion(None, ["The joint K(θ) fit needs at least two curves."])
    if not outcome.alternatives:
        return _empty_suggestion(None, ["No near-degenerate assignments to discriminate."])

    model = ParameterCompositeModel([outcome.model_name])
    angles = np.asarray(outcome.angles, dtype=float)
    warnings: list[str] = []
    err0 = (
        np.asarray(outcome.curve_errors[0], dtype=float)
        if outcome.curve_errors
        else np.zeros_like(angles)
    )
    grid = _candidate_grid(angles, err0, x_min, x_max, n_candidates, warnings)
    if grid is None:
        return _empty_suggestion(None, warnings)
    candidates, _x_meas, _sigma_new, extrapolated = grid

    sigma_per_curve: list[np.ndarray] = []
    missing_sigma = False
    for i in range(n_curves):
        errors_i = (
            np.asarray(outcome.curve_errors[i], dtype=float)
            if i < len(outcome.curve_errors)
            else np.zeros_like(angles)
        )
        noise = _empirical_sigma(angles, errors_i, [])
        if noise is None:
            missing_sigma = True
            sigma_per_curve.append(np.ones_like(candidates))
        else:
            x_meas, sigma_meas = noise
            sigma_per_curve.append(np.interp(candidates, x_meas, sigma_meas))
    if missing_sigma:
        warnings.append(
            "Some curves have no usable error bars; their divergence is weighted uniformly."
        )

    winner = [(model, curve.parameters) for curve in outcome.curves]
    utility = np.zeros_like(candidates)
    for alternative in outcome.alternatives:
        if len(alternative.curves) != n_curves:
            continue
        alt_hypothesis = [(model, curve.parameters) for curve in alternative.curves]
        utility = np.maximum(
            utility,
            set_matching_divergence(winner, alt_hypothesis, sigma_per_curve, candidates),
        )

    sigma_new = np.mean(np.vstack(sigma_per_curve), axis=0)
    max_variance = max(float(np.max(sigma**2)) for sigma in sigma_per_curve)
    relative_floor = _DISCRIMINATION_NOISE_FLOOR * max(max_variance, 1.0)
    if not np.any(utility > relative_floor):
        warnings.append(
            "the competing assignments predict the same value set within noise "
            "everywhere in this range — no discriminating point"
        )
        return NextPointSuggestion(
            x_candidates=candidates,
            utility=utility,
            extrapolated=extrapolated,
            best_x=float("nan"),
            target=None,
            sigma_new=sigma_new,
            warnings=tuple(warnings),
        )

    best_index = int(np.argmax(utility))
    return NextPointSuggestion(
        x_candidates=candidates,
        utility=utility,
        extrapolated=extrapolated,
        best_x=float(candidates[best_index]),
        target=None,
        sigma_new=sigma_new,
        warnings=tuple(warnings),
    )


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
    correction: KnightCorrection = field(default_factory=KnightCorrection)
    source_batch_id: str | None = None
    source_group_id: str | None = None
    x_key: str = "angle"
    fold_180: bool = False
    show_markers: bool = True
    rescale_errors: bool = False
    joint: KnightJointFitState | None = None

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "correction": self.correction.to_dict(),
            "source_batch_id": self.source_batch_id,
            "source_group_id": self.source_group_id,
            "x_key": str(self.x_key),
            "fold_180": bool(self.fold_180),
            "show_markers": bool(self.show_markers),
            "rescale_errors": bool(self.rescale_errors),
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
            correction=KnightCorrection.from_dict(data.get("correction")),
            source_batch_id=str(batch_id) if batch_id else None,
            source_group_id=str(group_id) if group_id else None,
            x_key=str(data.get("x_key") or "angle"),
            fold_180=bool(data.get("fold_180", False)),
            show_markers=bool(data.get("show_markers", True)),
            rescale_errors=bool(data.get("rescale_errors", False)),
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
