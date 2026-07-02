"""Fit-wizard scoping: narrow the candidate component set for a run.

The fit wizard trials fits against many built-in components. On a given run most
of them are physically irrelevant — a vortex-lattice component makes no sense in
zero field, a muonium four-frequency form is meaningless in a longitudinal run.
This module turns a run's *applied-field geometry* (and, for muonium, the field
*regime*) plus an optional physics-class preset into a concrete list of
in-scope components, with a specific human-readable reason recorded for every
component that is dropped.

Design rules honoured here:

* Field geometry is **never** inferred from the field magnitude — only from the
  recorded geometry token (see ``docs/porting/field-geometry/`` and
  :func:`asymmetry.core.io.base.field_direction_from_text`). The *muonium field
  regime* (low-/high-TF) is a separate, magnitude-based refinement that only
  ever narrows the muonium sub-family within an already-TF run.
* User-registered components (``physics_classes == {CUSTOM}``) match every
  scope and are never silently hidden — the wizard must never drop the user's
  own function behind their back.
* Envelopes (``GENERIC_RELAXATION``) and ``BACKGROUND`` survive every named
  preset, so a composite always has a relaxation envelope and a constant to
  reach for.

This module is Qt-free: it imports only the standard library, the scoping tags,
the component registry, and (for the dataset convenience wrappers) the pure-core
:class:`~asymmetry.core.data.dataset.MuonDataset`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.fitting.component_tags import (
    ALL_GEOMETRIES,
    ComputationalCost,
    FieldGeometry,
    PhysicsClass,
    geometry_from_field_direction,
)
from asymmetry.core.fitting.composite import COMPONENTS, ComponentDefinition

# --- muonium field-regime thresholds ------------------------------------
#
# In a transverse field the muonium sub-family that applies depends on the field
# *magnitude* relative to the hyperfine coupling: at low field the two satellite
# frequencies of ``MuoniumLowTF`` are resolved; at high field the intratriplet
# ``nu_12``/``nu_34`` pair of ``MuoniumHighTF`` dominates. These thresholds gate
# only the muonium sub-family *within* a TF run — they never decide geometry.

#: Above this TF field (gauss), the low-TF two-satellite form is no longer valid.
MUONIUM_LOW_TF_MAX_GAUSS: float = 150.0
#: Below this TF field (gauss), the high-TF intratriplet forms are not yet valid.
MUONIUM_HIGH_TF_MIN_GAUSS: float = 1500.0

# --- cost ordering ------------------------------------------------------
#
# ``ComputationalCost`` is a ``str``-Enum, so a naive ``<=`` compares members
# alphabetically ("cheap" < "expensive" < "moderate") — wrong. Compare through
# an explicit rank instead.
_COST_RANK: dict[ComputationalCost, int] = {
    ComputationalCost.CHEAP: 0,
    ComputationalCost.MODERATE: 1,
    ComputationalCost.EXPENSIVE: 2,
}

#: Rough per-component fit-count weighting for the GUI screening estimate. These
#: are display-only Stage-1/Stage-2 hints, not an exact schedule.
_COST_FIT_WEIGHT: dict[ComputationalCost, int] = {
    ComputationalCost.CHEAP: 3,
    ComputationalCost.MODERATE: 5,
    ComputationalCost.EXPENSIVE: 8,
}

#: Chemical-formula fluorine token: an uppercase ``F`` that begins an element
#: (followed by a stoichiometry digit, a non-lowercase char, or end-of-string).
#: Matches ``PbF2``/``CaF2``/``LiF``/``NaF``; rejects ``Fe``/``FeSe``/``Fer``.
#: Case-sensitive on purpose — a lowercase ``f`` is never the fluorine element.
_FLUORINE_TOKEN = re.compile(r"F(?=[0-9]|[^a-z]|$)")


class WizardScopePreset(str, Enum):
    """Named physics-class scope for the fit wizard."""

    AUTO = "auto"
    ZF_STATIC_MAGNETISM = "zf-static-magnetism"
    TF_KNIGHT_PRECESSION = "tf-knight-precession"
    TF_SUPERCONDUCTOR = "tf-superconductor"
    LF_DYNAMICS = "lf-dynamics"
    FLUORIDE_FMUF = "fluoride-fmuf"
    MUONIUM_RADICAL = "muonium-radical"
    ALL = "all"


@dataclass(frozen=True)
class ScopeQuery:
    """A concrete geometry/physics/cost filter over the component registry."""

    geometries: frozenset[FieldGeometry]
    physics_classes: frozenset[PhysicsClass]
    #: Cost cap; ``None`` for no cap. Ordering is CHEAP < MODERATE < EXPENSIVE.
    max_cost: ComputationalCost | None = None


#: Short human-readable descriptions of the static presets, used as the
#: inference note when the wizard is run on a named (non-Auto) preset.
_PRESET_NOTES: dict[WizardScopePreset, str] = {
    WizardScopePreset.ZF_STATIC_MAGNETISM: (
        "zero-field static-magnetism preset — ZF magnetism, envelopes, background"
    ),
    WizardScopePreset.TF_KNIGHT_PRECESSION: (
        "transverse-field precession preset — TF magnetism, envelopes, background"
    ),
    WizardScopePreset.TF_SUPERCONDUCTOR: (
        "transverse-field superconductor preset — TF vortex/magnetism, envelopes, background"
    ),
    WizardScopePreset.LF_DYNAMICS: (
        "longitudinal-field dynamics preset — LF dynamics/magnetism, envelopes, background"
    ),
    WizardScopePreset.FLUORIDE_FMUF: (
        "fluoride F-mu-F preset — ZF/LF molecular, envelopes, background"
    ),
    WizardScopePreset.MUONIUM_RADICAL: (
        "muonium/radical preset — muonium families across all geometries, envelopes, background"
    ),
    WizardScopePreset.ALL: "all component families (no scope restriction)",
}


#: Static physics-class scopes for every non-Auto preset. ``GENERIC_RELAXATION``
#: and ``BACKGROUND`` appear in each so envelopes and ``Constant`` always survive.
#: ``AUTO`` deliberately has no entry — it is resolved by :func:`infer_auto_query`.
PRESET_QUERIES: dict[WizardScopePreset, ScopeQuery] = {
    WizardScopePreset.ZF_STATIC_MAGNETISM: ScopeQuery(
        geometries=frozenset({FieldGeometry.ZF}),
        physics_classes=frozenset(
            {PhysicsClass.MAGNETISM, PhysicsClass.GENERIC_RELAXATION, PhysicsClass.BACKGROUND}
        ),
    ),
    WizardScopePreset.TF_KNIGHT_PRECESSION: ScopeQuery(
        geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset(
            {PhysicsClass.MAGNETISM, PhysicsClass.GENERIC_RELAXATION, PhysicsClass.BACKGROUND}
        ),
    ),
    WizardScopePreset.TF_SUPERCONDUCTOR: ScopeQuery(
        geometries=frozenset({FieldGeometry.TF}),
        physics_classes=frozenset(
            {
                PhysicsClass.SUPERCONDUCTIVITY,
                PhysicsClass.MAGNETISM,
                PhysicsClass.GENERIC_RELAXATION,
                PhysicsClass.BACKGROUND,
            }
        ),
    ),
    WizardScopePreset.LF_DYNAMICS: ScopeQuery(
        geometries=frozenset({FieldGeometry.LF}),
        physics_classes=frozenset(
            {
                PhysicsClass.DYNAMICS,
                PhysicsClass.MAGNETISM,
                PhysicsClass.GENERIC_RELAXATION,
                PhysicsClass.BACKGROUND,
            }
        ),
    ),
    WizardScopePreset.FLUORIDE_FMUF: ScopeQuery(
        geometries=frozenset({FieldGeometry.ZF, FieldGeometry.LF}),
        physics_classes=frozenset(
            {PhysicsClass.MOLECULAR, PhysicsClass.GENERIC_RELAXATION, PhysicsClass.BACKGROUND}
        ),
    ),
    WizardScopePreset.MUONIUM_RADICAL: ScopeQuery(
        geometries=ALL_GEOMETRIES,
        physics_classes=frozenset(
            {PhysicsClass.MUONIUM, PhysicsClass.GENERIC_RELAXATION, PhysicsClass.BACKGROUND}
        ),
    ),
    WizardScopePreset.ALL: ScopeQuery(
        geometries=ALL_GEOMETRIES,
        physics_classes=frozenset(PhysicsClass),
    ),
}


@dataclass(frozen=True)
class WizardScope:
    """A user-facing wizard scope: a preset plus explicit include/exclude overrides."""

    preset: WizardScopePreset = WizardScopePreset.AUTO
    include_components: frozenset[str] = frozenset()
    exclude_components: frozenset[str] = frozenset()

    def to_payload(self) -> dict:
        """Return a plain, JSON-serialisable representation of this scope."""
        return {
            "version": 1,
            "preset": self.preset.value,
            "include": sorted(self.include_components),
            "exclude": sorted(self.exclude_components),
        }

    @classmethod
    def from_payload(cls, payload: object) -> WizardScope:
        """Rebuild a scope from :meth:`to_payload` output, tolerant of garbage.

        A non-mapping payload, or one with an unknown preset, degrades to the
        default (``AUTO``) scope rather than raising. Include/exclude entries are
        coerced element-wise to ``str``; missing keys become empty sets.
        """
        if not isinstance(payload, Mapping):
            return cls()
        raw_preset = payload.get("preset", WizardScopePreset.AUTO.value)
        try:
            preset = WizardScopePreset(raw_preset)
        except ValueError:
            preset = WizardScopePreset.AUTO
        include = _coerce_name_set(payload.get("include"))
        exclude = _coerce_name_set(payload.get("exclude"))
        return cls(preset=preset, include_components=include, exclude_components=exclude)


def _coerce_name_set(value: object) -> frozenset[str]:
    """Coerce an arbitrary payload entry into a ``frozenset[str]`` of names."""
    if value is None or isinstance(value, (str, bytes)):
        # A bare string is not a name list here; treat scalars as "no names".
        return frozenset()
    if isinstance(value, Iterable):
        return frozenset(str(item) for item in value)
    return frozenset()


@dataclass(frozen=True)
class ExcludedComponent:
    """A component dropped from scope, with the specific reason why."""

    name: str
    reason: str


@dataclass(frozen=True)
class ScopeResolution:
    """The concrete outcome of resolving a :class:`WizardScope` against a run."""

    scope: WizardScope
    query: ScopeQuery
    effective_preset: WizardScopePreset
    inference_note: str
    #: Included component names, in registry order.
    included_components: tuple[str, ...] = ()
    #: Excluded components with reasons. Query/regime drops come first in
    #: registry order; any user-excluded names are appended after them.
    excluded_components: tuple[ExcludedComponent, ...] = ()

    @property
    def included_set(self) -> frozenset[str]:
        return frozenset(self.included_components)


def infer_auto_query(
    field_direction: str,
    field_gauss: float | None,
    sample_text: str = "",
) -> tuple[ScopeQuery, WizardScopePreset, str, tuple[ExcludedComponent, ...]]:
    """Infer a scope query from a run's recorded geometry (never its magnitude).

    Returns the query, the effective preset the outcome most resembles, a
    human-readable note naming the geometry source, and any muonium
    field-regime exclusions (TF only, and only when ``field_gauss`` is known).

    Geometry is taken solely from ``field_direction`` via
    :func:`geometry_from_field_direction`. When the geometry is unrecorded the
    query is a superset of every other outcome (all geometries × all classes),
    so the wizard never regresses on metadata-poor data.
    """
    geometry = geometry_from_field_direction(field_direction)
    exclusions: tuple[ExcludedComponent, ...] = ()

    if geometry is FieldGeometry.ZF:
        # All classes: molecular / muonium ZF physics can't be ruled out here.
        query = ScopeQuery(frozenset({FieldGeometry.ZF}), frozenset(PhysicsClass))
        effective = WizardScopePreset.ZF_STATIC_MAGNETISM
        note = "run geometry: zero field — screening ZF families"
    elif geometry is FieldGeometry.LF:
        query = ScopeQuery(frozenset({FieldGeometry.LF}), frozenset(PhysicsClass))
        effective = WizardScopePreset.LF_DYNAMICS
        note = "run geometry: longitudinal field — screening LF families"
    elif geometry is FieldGeometry.TF:
        query = ScopeQuery(frozenset({FieldGeometry.TF}), frozenset(PhysicsClass))
        effective = WizardScopePreset.TF_KNIGHT_PRECESSION
        note = "run geometry: transverse field — screening TF families"
        exclusions = _muonium_regime_exclusions(field_gauss)
    else:
        query = ScopeQuery(ALL_GEOMETRIES, frozenset(PhysicsClass))
        effective = WizardScopePreset.ALL
        note = "field geometry not recorded — screening all component families"

    if _FLUORINE_TOKEN.search(sample_text or ""):
        note += "; sample name suggests fluorine — F-mu-F candidates will be prioritised"

    return query, effective, note, exclusions


def _muonium_regime_exclusions(field_gauss: float | None) -> tuple[ExcludedComponent, ...]:
    """Muonium sub-family exclusions for a TF run of a known field magnitude.

    ``MuoniumTF`` (the exact four-frequency form) is never excluded. Returns an
    empty tuple when the field is unknown.
    """
    if field_gauss is None:
        return ()
    excluded: list[ExcludedComponent] = []
    if field_gauss > MUONIUM_LOW_TF_MAX_GAUSS:
        excluded.append(
            ExcludedComponent(
                "MuoniumLowTF",
                f"low-TF muonium form invalid above {MUONIUM_LOW_TF_MAX_GAUSS:g} G "
                f"(run field {field_gauss:g} G)",
            )
        )
    if field_gauss < MUONIUM_HIGH_TF_MIN_GAUSS:
        reason = (
            f"high-TF muonium form invalid below {MUONIUM_HIGH_TF_MIN_GAUSS:g} G "
            f"(run field {field_gauss:g} G)"
        )
        excluded.append(ExcludedComponent("MuoniumHighTF", reason))
        excluded.append(ExcludedComponent("MuoniumHighTFAniso", reason))
    return tuple(excluded)


def _component_exclusion_reason(
    definition: ComponentDefinition,
    query: ScopeQuery,
    preset: WizardScopePreset,
) -> str | None:
    """Return why *definition* is out of scope for *query*, or ``None`` if in scope.

    Checked in a fixed order so the reason is the most specific applicable one:
    frequency-domain first, then user-CUSTOM ubiquity, then geometry, physics,
    and finally cost.
    """
    if definition.domain != "time":
        return "frequency-domain component; the wizard fits time spectra"
    if definition.physics_classes == frozenset({PhysicsClass.CUSTOM}):
        return None  # user components match every query
    if not (definition.field_geometries & query.geometries):
        geoms = "/".join(sorted(g.value for g in query.geometries))
        return f"geometry '{geoms}' outside the component's applicable geometries"
    if not (definition.physics_classes & query.physics_classes):
        classes = "/".join(sorted(c.value for c in definition.physics_classes))
        return f"physics class '{classes}' outside the '{preset.value}' preset"
    if query.max_cost is not None and _COST_RANK[definition.cost] > _COST_RANK[query.max_cost]:
        return f"cost '{definition.cost.value}' above the '{query.max_cost.value}' cap"
    return None


def resolve_scope(
    scope: WizardScope,
    *,
    field_direction: str = "",
    field_gauss: float | None = None,
    sample_text: str = "",
    components: Mapping[str, ComponentDefinition] | None = None,
) -> ScopeResolution:
    """Resolve a :class:`WizardScope` against a run into concrete in/out lists.

    Picks the query (a static preset's, or Auto-inferred from the run), walks
    the component registry in order recording a specific reason for every drop,
    applies Auto's muonium regime exclusions, then applies the user's
    include/exclude overrides (exclude wins over include for the same name).
    Unknown override names are ignored for inclusion but appended to the note.
    """
    registry = COMPONENTS if components is None else components

    if scope.preset is WizardScopePreset.AUTO:
        query, effective, note, auto_exclusions = infer_auto_query(
            field_direction, field_gauss, sample_text
        )
    else:
        query = PRESET_QUERIES[scope.preset]
        effective = scope.preset
        note = _PRESET_NOTES[scope.preset]
        auto_exclusions = ()

    auto_excluded_names = {exc.name: exc for exc in auto_exclusions}

    included: list[str] = []
    excluded: list[ExcludedComponent] = []
    for name, definition in registry.items():
        reason = _component_exclusion_reason(definition, query, effective)
        if reason is None and name in auto_excluded_names:
            reason = auto_excluded_names[name].reason
        if reason is None:
            included.append(name)
        else:
            excluded.append(ExcludedComponent(name, reason))

    # Apply overrides. Exclude beats include for the same name.
    include_names = scope.include_components - scope.exclude_components
    unknown: list[str] = []

    if include_names:
        known_excluded = {exc.name for exc in excluded}
        resurrect = {n for n in include_names if n in known_excluded}
        excluded = [exc for exc in excluded if exc.name not in resurrect]
        # Re-insert resurrected names in registry order.
        included = [n for n in registry if n in included or n in resurrect]
        unknown.extend(sorted(n for n in include_names if n not in registry))

    if scope.exclude_components:
        drop = {n for n in scope.exclude_components if n in registry}
        if drop:
            already = {exc.name for exc in excluded}
            excluded.extend(
                ExcludedComponent(n, "excluded by user")
                for n in registry
                if n in drop and n not in already
            )
            included = [n for n in included if n not in drop]
        unknown.extend(sorted(n for n in scope.exclude_components if n not in registry))

    if unknown:
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        ordered = [n for n in unknown if not (n in seen or seen.add(n))]
        note += "; unknown component in overrides: " + ", ".join(ordered)

    return ScopeResolution(
        scope=scope,
        query=query,
        effective_preset=effective,
        inference_note=note,
        included_components=tuple(included),
        excluded_components=tuple(excluded),
    )


def _dataset_geometry_text(dataset: MuonDataset) -> str:
    """Best geometry token for a dataset: ``field_direction`` then ``field_state``."""
    metadata = dataset.metadata or {}
    for key in ("field_direction", "field_state"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def _dataset_sample_text(dataset: MuonDataset) -> str:
    """Best sample/title text for a dataset (first non-empty of title/sample)."""
    metadata = dataset.metadata or {}
    for key in ("title", "sample"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def resolve_scope_for_dataset(dataset: MuonDataset, scope: WizardScope) -> ScopeResolution:
    """Resolve *scope* for a single dataset, reading geometry/field/sample from it."""
    return resolve_scope(
        scope,
        field_direction=_dataset_geometry_text(dataset),
        field_gauss=dataset.field,
        sample_text=_dataset_sample_text(dataset),
    )


def resolve_scope_for_datasets(
    datasets: Iterable[MuonDataset], scope: WizardScope
) -> ScopeResolution:
    """Resolve *scope* across several datasets, unioning the in-scope set.

    A component is included if it is in scope for **any** dataset; a component is
    excluded only if it is excluded for **every** dataset (one representative
    reason is kept, prefixed ``"all runs: "``). The effective preset and note
    come from the first dataset with a recorded geometry, else the ALL fallback.
    The reported ``query`` is the first-resolved one — representative only.
    """
    resolutions = [resolve_scope_for_dataset(dataset, scope) for dataset in datasets]
    if not resolutions:
        query, effective, note, _ = infer_auto_query("", None, "")
        return ScopeResolution(
            scope=scope,
            query=query,
            effective_preset=effective,
            inference_note=note,
        )

    included_any: set[str] = set()
    for resolution in resolutions:
        included_any |= resolution.included_set

    # A component is excluded only if excluded in every resolution.
    exclude_reason: dict[str, str] = {}
    excluded_in_all: set[str] | None = None
    for resolution in resolutions:
        names = {exc.name for exc in resolution.excluded_components}
        for exc in resolution.excluded_components:
            exclude_reason.setdefault(exc.name, exc.reason)
        excluded_in_all = names if excluded_in_all is None else (excluded_in_all & names)
    excluded_in_all = excluded_in_all or set()
    excluded_in_all -= included_any

    # Preserve registry order for both lists.
    included = tuple(n for n in COMPONENTS if n in included_any)
    excluded = tuple(
        ExcludedComponent(n, "all runs: " + exclude_reason[n])
        for n in COMPONENTS
        if n in excluded_in_all
    )

    # Effective preset/note from the first dataset with a known geometry.
    effective = WizardScopePreset.ALL
    note = "field geometry not recorded — screening all component families"
    for dataset_res in resolutions:
        if dataset_res.effective_preset is not WizardScopePreset.ALL:
            effective = dataset_res.effective_preset
            note = dataset_res.inference_note
            break

    return ScopeResolution(
        scope=scope,
        query=resolutions[0].query,
        effective_preset=effective,
        inference_note=note,
        included_components=included,
        excluded_components=excluded,
    )


def estimate_screening_cost(resolution: ScopeResolution) -> tuple[int, int]:
    """Rough ``(candidates, fits)`` estimate for a resolved scope.

    ``candidates`` is the count of included time-domain components. ``fits`` is a
    display-only weighted sum (cheap:3, moderate:5, expensive:8) approximating
    the Stage-1/Stage-2 screening load — the exact numbers are for a GUI label,
    not a schedule.
    """
    candidates = 0
    fits = 0
    for name in resolution.included_components:
        definition = COMPONENTS.get(name)
        if definition is None or definition.domain != "time":
            continue
        candidates += 1
        fits += _COST_FIT_WEIGHT[definition.cost]
    return candidates, fits
