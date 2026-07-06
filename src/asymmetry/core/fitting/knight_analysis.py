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


@dataclass
class KnightAnalysisState:
    """Persisted window state: conversion config + source binding + view hints.

    Deliberately excludes the point snapshot (rebuilt from the source series on
    open — a saved copy could go stale against a refit). ``source_batch_id`` /
    ``source_group_id`` re-bind the window to its series; ``x_key`` pins the
    scan axis. ``fold_180`` and ``show_markers`` are plot-view preferences.
    """

    config: KnightShiftConfig = field(default_factory=KnightShiftConfig)
    source_batch_id: str | None = None
    source_group_id: str | None = None
    x_key: str = "angle"
    fold_180: bool = False
    show_markers: bool = True

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "source_batch_id": self.source_batch_id,
            "source_group_id": self.source_group_id,
            "x_key": str(self.x_key),
            "fold_180": bool(self.fold_180),
            "show_markers": bool(self.show_markers),
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
