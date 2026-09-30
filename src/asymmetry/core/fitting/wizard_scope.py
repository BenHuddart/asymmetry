"""Fit-wizard scoping: narrow the candidate component set for a run.

The fit wizard trials fits against many built-in components. On a given run most
of them are physically irrelevant — a vortex-lattice component makes no sense in
zero field, a muonium four-frequency form is meaningless in a longitudinal run.
This module turns a run's *applied-field geometry* (and, for muonium, the field
*regime*) plus a :class:`WizardScope` — the physics classes looked for, the
user's include/exclude overrides and the slow-model switch — into a concrete
list of in-scope components, with a specific human-readable reason recorded for
every component that is dropped. :func:`describe_scope` renders the same
resolution as a typed :class:`ScopeView` for the model family picker.

Design rules honoured here:

* Field geometry is **never** inferred from the field magnitude — only from the
  recorded geometry token (see ``docs/porting/field-geometry/`` and
  :func:`asymmetry.core.io.base.field_direction_from_text`). The *muonium field
  regime* (low-/high-TF) is a separate, magnitude-based refinement that only
  ever narrows the muonium sub-family within an already-TF run.
* A scope never carries a geometry: geometry always comes from the runs. The
  user may answer the direction only on a run whose file records none
  (:func:`set_user_field_direction`), and the answer is marked as theirs.
* User-registered components (``physics_classes == {CUSTOM}``) match every
  scope and are never silently hidden — the wizard must never drop the user's
  own function behind their back.
* Envelopes (``GENERIC_RELAXATION``) and ``BACKGROUND`` survive every physics
  choice, so a composite always has a relaxation envelope and a constant to
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
    coerce_physics_classes,
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

# --- B≈0 geometry-label override ----------------------------------------
#
# Real ISIS runs are sometimes recorded with a TF/LF beamline label while the
# applied-field *setpoint* sits at (or near) zero — the run is physically a ZF
# measurement on a TF-capable beamline (see docs/porting/field-geometry/,
# "MUSR00044991.nxs: magnetic_field_state='TF' at magnetic_field=0 G"). Trusting
# the label alone excludes ZF-only families (F-mu-F, Kubo-Toyabe) exactly on the
# data that needs them.
#
# This is deliberately distinct from the "geometry is never inferred from field
# magnitude" invariant honoured elsewhere in this module: that rule forbids
# *guessing a TF/LF direction* from |B| (a TF run can legitimately sit at 0 G,
# so |B|==0 must never be read as "this was actually LF"). Reading the recorded
# setpoint and concluding "there is no applied field" is a different claim —
# it says nothing about which direction a nonzero field would have pointed.
# The override therefore only ever *widens* scope (adds the ZF family list on
# top of the labelled one); it never removes or overrides the recorded label,
# because the label may still be correct about which magnet/hardware was live.
ZERO_FIELD_MAX_GAUSS: float = 2.0

#: A component whose screening fits are expected to take longer than this per run is slow.
SLOW_SECONDS_PER_RUN: float = 5.0


@dataclass(frozen=True)
class FitTimeEstimates:
    """Expected screening-fit seconds per run for the components timed on this computer.

    Built by :meth:`~asymmetry.core.fitting.fit_time_store.FitTimeStore.estimates`;
    a component absent from ``seconds_per_run`` has never been timed here.
    Design: ``docs/plans/measured-fit-times.md`` (D4).
    """

    seconds_per_run: Mapping[str, float]

    def is_slow(self, name: str, definition: ComponentDefinition) -> bool:
        """Over :data:`SLOW_SECONDS_PER_RUN` once timed; the registry's expensive tier until then."""
        seconds = self.seconds_per_run.get(name)
        if seconds is None:
            return definition.cost is ComputationalCost.EXPENSIVE
        return seconds > SLOW_SECONDS_PER_RUN


#: The judgement before anything is timed: slow means the registry's expensive tier.
UNTIMED = FitTimeEstimates({})

#: Chemical-formula fluorine token: an uppercase ``F`` that begins an element
#: (followed by a stoichiometry digit, a non-lowercase char, or end-of-string).
#: Matches ``PbF2``/``CaF2``/``LiF``/``NaF``; rejects ``Fe``/``FeSe``/``Fer``.
#: Case-sensitive on purpose — a lowercase ``f`` is never the fluorine element.
#: An ``F`` followed by ``=`` is excluded: ISIS run titles carry the applied
#: field as ``F=<gauss>`` (``nickel T=100 F=0``), which is not a formula. So is
#: an ``F`` after ``T``, ``L`` or ``Z`` — the geometry tokens ``TF60G``,
#: ``LF100``, ``ZF`` — and no element symbol is one of those letters, so no
#: formula loses its fluorine to the rule.
_FLUORINE_TOKEN = re.compile(r"(?<![TLZ])F(?=[0-9]|[^a-z=]|$)")


class EffortTier(str, Enum):
    """User-facing global-fit-wizard effort level.

    Originally bound a four-position slider to different search engines and
    knobs. **Every tier now resolves to the separable role-search engine** (see
    :mod:`asymmetry.core.fitting.global_fit_wizard` ``_EFFORT_TIER_SEARCH_ENGINE``):
    that search takes the all-local assignment straight from the per-run fits,
    ranks every sharing pattern with a full-covariance surrogate, and walks
    backward elimination with one warm fit per step, so it costs O(P) coupled
    fits per template instead of O(2^P) — the honest answer at every tier rather
    than a coarser one, which leaves the control a single "Optimize" mode. The
    enum and its serialised payload are retained so a future *scope-based*
    quick-look tier can be added without a schema/UI change; the exhaustive
    wavefront (the harness referee) and the heuristic engines stay reachable
    behind the low-level ``search_engine`` string.
    """

    LOW = "low"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    EXHAUSTIVE = "exhaustive"


#: Default effort tier — the separable role-search engine, now the only
#: user-facing mode (see ``EffortTier``). Persisted/legacy payloads default here.
DEFAULT_EFFORT_TIER = EffortTier.EXHAUSTIVE

#: Short human-readable labels for the effort control. Every tier now runs the
#: separable engine, so the visible control surfaces one honest "Optimize" mode.
EFFORT_TIER_LABELS: dict[EffortTier, str] = {
    EffortTier.LOW: "Low (screening-grade)",
    EffortTier.BALANCED: "Balanced (recommended)",
    EffortTier.THOROUGH: "Thorough",
    EffortTier.EXHAUSTIVE: "Optimize",
}

#: One-line descriptions surfaced as tooltips next to the control.
EFFORT_TIER_DESCRIPTIONS: dict[EffortTier, str] = {
    EffortTier.LOW: (
        "Screening-grade heuristic engine (retained behind the search_engine "
        "seam; not a user-facing tier)."
    ),
    EffortTier.BALANCED: (
        "Heuristic search engine (retained behind the search_engine seam; not a user-facing tier)."
    ),
    EffortTier.THOROUGH: ("Separable role search (same engine as Optimize)."),
    EffortTier.EXHAUSTIVE: (
        "Separable role search over all promotable parameters: the all-local "
        "answer comes from the per-run fits, a full-covariance surrogate ranks "
        "every sharing pattern, and backward elimination fits the exact path. "
        "It is the single honest optimisation mode."
    ),
}


def effort_tier_to_payload(tier: EffortTier) -> str:
    """Serialise an :class:`EffortTier` to its plain string value."""
    return tier.value


def effort_tier_from_payload(payload: object) -> EffortTier:
    """Rebuild an :class:`EffortTier` from a payload, tolerant of garbage.

    An unrecognised or missing value degrades to :data:`DEFAULT_EFFORT_TIER`
    rather than raising.
    """
    if isinstance(payload, EffortTier):
        return payload
    try:
        return EffortTier(payload)
    except ValueError:
        return DEFAULT_EFFORT_TIER


@dataclass(frozen=True)
class ScopeQuery:
    """A concrete geometry/physics filter over the component registry."""

    geometries: frozenset[FieldGeometry]
    physics_classes: frozenset[PhysicsClass]


#: Classes every scope keeps, so a composite always has an envelope and a constant.
ALWAYS_IN_SCOPE: frozenset[PhysicsClass] = frozenset(
    {PhysicsClass.GENERIC_RELAXATION, PhysicsClass.BACKGROUND}
)

#: The version :meth:`WizardScope.to_payload` writes and :meth:`WizardScope.from_payload` reads.
SCOPE_PAYLOAD_VERSION: int = 2

_SCOPE_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {"version", "physics", "include", "exclude", "skip_slow"}
)


@dataclass(frozen=True)
class WizardScope:
    """What the wizard screens: the physics looked for, overrides and the slow switch.

    An empty ``physics`` means every class. The scope carries no geometry — that
    always comes from the runs' recorded field direction.
    """

    physics: frozenset[PhysicsClass] = frozenset()
    include_components: frozenset[str] = frozenset()
    exclude_components: frozenset[str] = frozenset()
    #: Leave out the models :meth:`FitTimeEstimates.is_slow` judges slow.
    skip_slow: bool = False

    @property
    def physics_classes(self) -> frozenset[PhysicsClass]:
        """Every class for an empty ``physics``, else it plus :data:`ALWAYS_IN_SCOPE`."""
        return self.physics | ALWAYS_IN_SCOPE if self.physics else frozenset(PhysicsClass)

    def to_payload(self) -> dict:
        """Return a plain, JSON-serialisable representation of this scope."""
        return {
            "version": SCOPE_PAYLOAD_VERSION,
            "physics": sorted(physics_class.value for physics_class in self.physics),
            "include": sorted(self.include_components),
            "exclude": sorted(self.exclude_components),
            "skip_slow": self.skip_slow,
        }

    @classmethod
    def from_payload(cls, payload: object) -> WizardScope:
        """Parse :meth:`to_payload` output; raise :class:`ValueError` on anything else."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"a scope payload must be a mapping, got {payload!r}")
        if set(payload) != _SCOPE_PAYLOAD_KEYS:
            raise ValueError(
                f"a scope payload has exactly the keys {sorted(_SCOPE_PAYLOAD_KEYS)}, "
                f"got {sorted(payload)}"
            )
        if payload["version"] != SCOPE_PAYLOAD_VERSION:
            raise ValueError(
                f"a scope payload is version {SCOPE_PAYLOAD_VERSION}, got {payload['version']!r}"
            )
        if not isinstance(payload["skip_slow"], bool):
            raise ValueError(f"a scope's skip_slow is a bool, got {payload['skip_slow']!r}")
        return cls(
            physics=coerce_physics_classes(_payload_names(payload, "physics")),
            include_components=frozenset(_payload_names(payload, "include")),
            exclude_components=frozenset(_payload_names(payload, "exclude")),
            skip_slow=payload["skip_slow"],
        )


def _payload_names(payload: Mapping, key: str) -> list[str]:
    """The list of strings under *key*, or :class:`ValueError`."""
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"a scope's {key!r} is a list of strings, got {value!r}")
    return value


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
    #: Human-readable notes: the geometry source, then any fluorine, muonium
    #: regime, physics, slow-model and unknown-override notes.
    notes: tuple[str, ...]
    #: Included component names, in registry order.
    included_components: tuple[str, ...] = ()
    #: Excluded components with reasons. Query/regime drops come first in
    #: registry order; any user-excluded names are appended after them.
    excluded_components: tuple[ExcludedComponent, ...] = ()

    @property
    def included_set(self) -> frozenset[str]:
        return frozenset(self.included_components)

    @property
    def inference_note(self) -> str:
        """The notes as one ``"; "``-joined line."""
        return "; ".join(self.notes)


_GEOMETRY_NOTES: dict[FieldGeometry, str] = {
    FieldGeometry.ZF: "run geometry: zero field — screening ZF families",
    FieldGeometry.LF: "run geometry: longitudinal field — screening LF families",
    FieldGeometry.TF: "run geometry: transverse field — screening TF families",
}

_UNRECORDED_NOTE = "field geometry not recorded — screening all component families"


def infer_run_geometries(
    field_direction: str,
    field_gauss: float | None,
    sample_text: str = "",
) -> tuple[frozenset[FieldGeometry], tuple[str, ...], tuple[ExcludedComponent, ...]]:
    """Infer the geometries to screen from a run's recorded geometry (never its magnitude).

    Returns the geometries, human-readable notes (the geometry source first),
    and any muonium field-regime exclusions (TF only, and only when
    ``field_gauss`` is known). Geometry is taken solely from ``field_direction``
    via :func:`geometry_from_field_direction`; an unrecorded geometry screens
    all three, so the wizard never regresses on metadata-poor data. A TF/LF
    label at a setpoint within :data:`ZERO_FIELD_MAX_GAUSS` of zero also
    screens ZF.
    """
    geometry = geometry_from_field_direction(field_direction)
    exclusions: tuple[ExcludedComponent, ...] = ()
    if geometry is None:
        geometries = ALL_GEOMETRIES
        notes = [_UNRECORDED_NOTE]
    else:
        widened = (
            geometry is not FieldGeometry.ZF
            and field_gauss is not None
            and abs(field_gauss) <= ZERO_FIELD_MAX_GAUSS
        )
        geometries = frozenset({geometry, FieldGeometry.ZF}) if widened else frozenset({geometry})
        note = _GEOMETRY_NOTES[geometry]
        if widened:
            note += (
                f"; applied-field setpoint {field_gauss:g} G is within "
                f"{ZERO_FIELD_MAX_GAUSS:g} G of zero — widening to include ZF families"
            )
        notes = [note]
        if geometry is FieldGeometry.TF:
            exclusions = _muonium_regime_exclusions(field_gauss)
            if exclusions:
                notes.append(
                    f"transverse field {field_gauss:g} G — muonium forms outside "
                    "their field regime are left out"
                )

    if _FLUORINE_TOKEN.search(sample_text or ""):
        notes.append("sample name suggests fluorine — F-mu-F candidates will be prioritised")

    return geometries, tuple(notes), exclusions


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


def _joined(values: Iterable[Enum]) -> str:
    return "/".join(sorted(value.value for value in values))


def _component_exclusion_reason(
    name: str,
    definition: ComponentDefinition,
    query: ScopeQuery,
    scope: WizardScope,
    fit_times: FitTimeEstimates,
) -> str | None:
    """Return why *definition* is out of scope for *query*, or ``None`` if in scope.

    Checked in a fixed order so the reason is the most specific applicable one:
    frequency-domain first, then geometry, physics, and finally slowness. An
    untagged user component matches every geometry and physics choice, but the
    slow-model switch still applies to it.
    """
    if definition.domain != "time":
        return "frequency-domain component; the wizard fits time spectra"
    if definition.physics_classes != frozenset({PhysicsClass.CUSTOM}):
        if not (definition.field_geometries & query.geometries):
            return (
                f"applies in {_joined(definition.field_geometries)}, "
                f"not the runs' {_joined(query.geometries)} geometry"
            )
        if not (definition.physics_classes & query.physics_classes):
            return (
                f"physics class '{_joined(definition.physics_classes)}' is not looked for "
                f"({_joined(scope.physics)})"
            )
    if scope.skip_slow and fit_times.is_slow(name, definition):
        return "slow model; slow models are left out"
    return None


def resolve_scope(
    scope: WizardScope,
    *,
    field_direction: str = "",
    field_gauss: float | None = None,
    sample_text: str = "",
    components: Mapping[str, ComponentDefinition] | None = None,
    fit_times: FitTimeEstimates = UNTIMED,
) -> ScopeResolution:
    """Resolve a :class:`WizardScope` against a run into concrete in/out lists.

    Takes the geometries from the run and the physics classes from the scope,
    leaves out what *fit_times* judges slow when the scope skips slow models,
    walks the component registry in order recording a specific reason for
    every drop, applies the muonium regime exclusions, then applies the user's
    include/exclude overrides (exclude wins over include for the same name).
    Unknown override names are ignored for inclusion but named in the notes.
    """
    registry = COMPONENTS if components is None else components

    geometries, geometry_notes, regime_exclusions = infer_run_geometries(
        field_direction, field_gauss, sample_text
    )
    query = ScopeQuery(geometries, scope.physics_classes)
    notes = list(geometry_notes)
    if scope.physics:
        notes.append(f"looking for {', '.join(sorted(c.value for c in scope.physics))}")
    if scope.skip_slow:
        notes.append("slow models left out")

    regime_reasons = {exc.name: exc.reason for exc in regime_exclusions}

    included: list[str] = []
    excluded: list[ExcludedComponent] = []
    for name, definition in registry.items():
        reason = _component_exclusion_reason(name, definition, query, scope, fit_times)
        if reason is None:
            reason = regime_reasons.get(name)
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
                ExcludedComponent(n, "switched off by you")
                for n in registry
                if n in drop and n not in already
            )
            included = [n for n in included if n not in drop]
        unknown.extend(sorted(n for n in scope.exclude_components if n not in registry))

    if unknown:
        # Preserve order, drop duplicates.
        notes.append("unknown component in overrides: " + ", ".join(dict.fromkeys(unknown)))

    return ScopeResolution(
        scope=scope,
        query=query,
        notes=tuple(notes),
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


def dataset_field_geometry(dataset: MuonDataset) -> FieldGeometry | None:
    """The run's field geometry, recorded or answered by the user; ``None`` when unknown."""
    return geometry_from_field_direction(_dataset_geometry_text(dataset))


#: ``metadata["field_direction_source"]`` on a direction the user answered.
_USER_SOURCE = "user"

#: The loader vocabulary (:func:`~asymmetry.core.io.base.field_direction_from_text`), ZF/LF/TF.
FIELD_DIRECTION_TEXT: dict[FieldGeometry, str] = {
    FieldGeometry.ZF: "Zero field",
    FieldGeometry.LF: "Longitudinal",
    FieldGeometry.TF: "Transverse",
}


def _answered_by_user(metadata: Mapping) -> bool:
    return metadata.get("field_direction_source") == _USER_SOURCE


def set_user_field_direction(
    datasets: Iterable[MuonDataset], geometry: FieldGeometry | None
) -> None:
    """Answer the field direction on every run whose file records none.

    Any earlier user answer is withdrawn first, so ``None`` ("Not recorded")
    leaves each run as its file had it. A direction the file records
    (``field_direction`` or ``field_state``) is never touched. The dataset and
    its run carry the same values, as the project load path applies them.
    """
    for dataset in datasets:
        metadatas = [dataset.metadata] + ([] if dataset.run is None else [dataset.run.metadata])
        for metadata in metadatas:
            if _answered_by_user(metadata):
                del metadata["field_direction"], metadata["field_direction_source"]
        if geometry is not None and dataset_field_geometry(dataset) is None:
            for metadata in metadatas:
                metadata["field_direction"] = FIELD_DIRECTION_TEXT[geometry]
                metadata["field_direction_source"] = _USER_SOURCE


def user_field_direction_overrides(dataset: MuonDataset) -> dict[str, str]:
    """The user's direction answer as project ``metadata_overrides``; empty when there is none."""
    if not _answered_by_user(dataset.metadata):
        return {}
    return {key: dataset.metadata[key] for key in ("field_direction", "field_direction_source")}


def restore_user_field_direction(dataset: MuonDataset, saved: str) -> None:
    """Re-apply a saved direction answer; a direction the file now records wins."""
    geometry = geometry_from_field_direction(saved)
    if geometry is None:
        raise ValueError(
            f"a saved field-direction answer is one of {sorted(FIELD_DIRECTION_TEXT.values())}, "
            f"got {saved!r}"
        )
    set_user_field_direction([dataset], geometry)


def _dataset_sample_text(dataset: MuonDataset) -> str:
    """Best sample/title text for a dataset (first non-empty of title/sample)."""
    metadata = dataset.metadata or {}
    for key in ("title", "sample"):
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


def dataset_suggests_fluorine(dataset: MuonDataset) -> bool:
    """Whether the run's sample/title text carries a chemical-formula fluorine.

    The same case-sensitive ``F``-element token the scope notes use to prioritise
    F-mu-F candidates (:data:`_FLUORINE_TOKEN`); surfaced as a boolean so the fit
    wizard can *promote* the fmuf family on a fluorine sniff, not merely annotate
    the scope note.  Matches ``CaF2``/``NaF``/``LiF``; rejects ``Fe``/``FeSe``.
    """
    return bool(_FLUORINE_TOKEN.search(_dataset_sample_text(dataset) or ""))


def resolve_scope_for_dataset(
    dataset: MuonDataset, scope: WizardScope, fit_times: FitTimeEstimates = UNTIMED
) -> ScopeResolution:
    """Resolve *scope* for a single dataset, reading geometry/field/sample from it."""
    return resolve_scope(
        scope,
        field_direction=_dataset_geometry_text(dataset),
        field_gauss=dataset.field,
        sample_text=_dataset_sample_text(dataset),
        fit_times=fit_times,
    )


def resolve_scope_for_datasets(
    datasets: Iterable[MuonDataset],
    scope: WizardScope,
    fit_times: FitTimeEstimates = UNTIMED,
) -> ScopeResolution:
    """Resolve *scope* across several datasets, unioning the in-scope set.

    A component is included if it is in scope for **any** dataset; a component is
    excluded only if it is excluded for **every** dataset (one representative
    reason is kept). The notes come from the first
    dataset with a recorded geometry, else the first dataset. The reported
    ``query`` is the first-resolved one — representative only. No datasets
    resolve to nothing in scope.
    """
    datasets = list(datasets)
    resolutions = [resolve_scope_for_dataset(dataset, scope, fit_times) for dataset in datasets]
    if not resolutions:
        geometries, notes, _ = infer_run_geometries("", None)
        return ScopeResolution(
            scope=scope,
            query=ScopeQuery(geometries, scope.physics_classes),
            notes=notes,
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
        ExcludedComponent(n, exclude_reason[n]) for n in COMPONENTS if n in excluded_in_all
    )

    representative = next(
        (
            resolution
            for dataset, resolution in zip(datasets, resolutions, strict=True)
            if dataset_field_geometry(dataset) is not None
        ),
        resolutions[0],
    )

    return ScopeResolution(
        scope=scope,
        query=resolutions[0].query,
        notes=representative.notes,
        included_components=included,
        excluded_components=excluded,
    )


# --- the typed view the model family picker renders -----------------------


#: The family every user-registered time component is listed under, after the built-ins.
USER_FAMILY_TITLE = "Your functions"

#: Family cards in picker display order: registry ``category`` (or
#: :data:`USER_FAMILY_TITLE`) → (display title, one-line blurb).
FAMILY_TEXT: dict[str, tuple[str, str]] = {
    "Relaxation": ("Relaxation", "Monotonic loss of polarisation — the usual first guess."),
    "Kubo-Toyabe": (
        "Kubo–Toyabe",
        "Random static or fluctuating fields: the ⅓ tail, LF decoupling.",
    ),
    "Oscillation": ("Oscillation", "Coherent precession: ordered magnets, superconductors."),
    "Muonium": ("Muonium", "Muon bound to an electron."),
    "Nuclear dipolar": ("Nuclear dipolar", "Muon bound to F or H nuclei — F–μ–F and relatives."),
    "Background": ("Background", "Always included."),
    USER_FAMILY_TITLE: (USER_FAMILY_TITLE, "Functions you defined."),
}


@dataclass(frozen=True)
class RecordedGeometry:
    """How the runs' files record their applied-field direction, and the user's answers."""

    #: ``(geometry, run count)`` for every geometry some run's file records, in ZF/TF/LF order.
    counts: tuple[tuple[FieldGeometry, int], ...]
    #: ``(geometry, run count)`` for every direction the user answered, in ZF/TF/LF order.
    answered: tuple[tuple[FieldGeometry, int], ...]
    #: Runs whose file records no direction, answered or not.
    unrecorded: int

    @property
    def editable(self) -> bool:
        """The direction can be answered: at least one run's file records none."""
        return self.unrecorded > 0

    @property
    def answer(self) -> FieldGeometry | None:
        """The user's direction when every run the file leaves open carries it, else ``None``."""
        return next((geometry for geometry, n in self.answered if n == self.unrecorded), None)


def _geometry_counts(
    geometries: list[FieldGeometry | None],
) -> tuple[tuple[FieldGeometry, int], ...]:
    return tuple(
        (geometry, geometries.count(geometry))
        for geometry in FieldGeometry
        if geometry in geometries
    )


@dataclass(frozen=True)
class ScopeComponent:
    """One time-domain component as the picker shows it."""

    name: str
    label: str
    use_when: str
    description: str
    geometries: frozenset[FieldGeometry]
    #: :meth:`FitTimeEstimates.is_slow` for these runs.
    slow: bool
    #: Expected screening-fit seconds per run on this computer; ``None`` until timed.
    estimated_seconds: float | None
    #: Its geometry and field regime fit at least one run.
    applies: bool
    included: bool
    #: Why it is out of scope; empty when included.
    reason: str


@dataclass(frozen=True)
class ScopeFamily:
    """One family card: its display title, blurb and components in registry order."""

    title: str
    blurb: str
    components: tuple[ScopeComponent, ...]


@dataclass(frozen=True)
class ScopeView:
    """Everything the model family picker renders for one scope and set of runs."""

    scope: WizardScope
    geometry: RecordedGeometry
    families: tuple[ScopeFamily, ...]
    notes: tuple[str, ...]

    @property
    def included_count(self) -> int:
        return sum(c.included for family in self.families for c in family.components)

    @property
    def applicable_count(self) -> int:
        return sum(c.applies for family in self.families for c in family.components)

    @property
    def screens_a_model(self) -> bool:
        """Something besides the background constant is included."""
        background, _ = FAMILY_TEXT["Background"]
        return any(
            c.included
            for family in self.families
            if family.title != background
            for c in family.components
        )

    @property
    def slow_included(self) -> tuple[ScopeComponent, ...]:
        """The included slow components, in display order."""
        return tuple(
            c for family in self.families for c in family.components if c.included and c.slow
        )


def describe_scope(
    datasets: Iterable[MuonDataset],
    scope: WizardScope,
    fit_times: FitTimeEstimates = UNTIMED,
) -> ScopeView:
    """Resolve *scope* across *datasets* into the typed view the picker renders.

    Inclusion and reasons come from :func:`resolve_scope_for_datasets`;
    ``applies`` is inclusion under the unrestricted scope, so a component that
    does not fit the runs' geometry is still listed with its reason. *fit_times*
    is the judgement the wizard run must be given too, so the models tagged slow
    are exactly those **Leave out slow models** leaves out.
    """
    datasets = list(datasets)
    resolution = resolve_scope_for_datasets(datasets, scope, fit_times)
    applicable = resolve_scope_for_datasets(datasets, WizardScope()).included_set
    reasons = {exc.name: exc.reason for exc in resolution.excluded_components}

    recorded = [
        dataset_field_geometry(dataset)
        for dataset in datasets
        if not _answered_by_user(dataset.metadata)
    ]
    answered = [
        dataset_field_geometry(dataset)
        for dataset in datasets
        if _answered_by_user(dataset.metadata)
    ]
    geometry = RecordedGeometry(
        counts=_geometry_counts(recorded),
        answered=_geometry_counts(answered),
        unrecorded=recorded.count(None) + len(answered),
    )

    members: dict[str, list[ScopeComponent]] = {key: [] for key in FAMILY_TEXT}
    for name, definition in COMPONENTS.items():
        if definition.domain != "time":
            continue
        key = USER_FAMILY_TITLE if definition.user else definition.category
        members[key].append(
            ScopeComponent(
                name=name,
                label=definition.label,
                use_when=definition.use_when,
                description=definition.description,
                geometries=definition.field_geometries,
                slow=fit_times.is_slow(name, definition),
                estimated_seconds=fit_times.seconds_per_run.get(name),
                applies=name in applicable,
                included=name in resolution.included_set,
                reason=reasons.get(name, ""),
            )
        )

    return ScopeView(
        scope=scope,
        geometry=geometry,
        families=tuple(
            ScopeFamily(*FAMILY_TEXT[key], tuple(components))
            for key, components in members.items()
            if components
        ),
        notes=resolution.notes,
    )
