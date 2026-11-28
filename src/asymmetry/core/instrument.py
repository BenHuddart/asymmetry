"""Instrument layout definitions for muon spectrometers.

This module provides static geometric descriptions of the detector arrangements
for HiFi, EMU, MuSR, PSI FLAME, PSI HAL-9500, and PSI GPS (in three variants — the
6-detector PSI-BIN layout ``GPS``, the 11-detector ROOT sub-detector layout
``GPS-RD``, and the 15-detector ROOT layout ``GPS-RD15`` that exports the combined
counters *and* the sub-detectors, all shown to the user as "GPS"), along with
standard grouping presets, plus a preset-free generic ring
(:func:`generic_layout`) for instruments it does not know.  The data here is used by the interactive detector
layout editor
(:class:`~asymmetry.gui.windows.detector_layout_dialog.DetectorLayoutDialog`)
but has no GUI dependencies and can be used independently.

PSI declarations distinguish acquired readouts, physical affiliations and drawing
occurrences. Their sources and unresolved assumptions are published with each profile.

Detector IDs are always **1-based** in this module, matching the instrument
manual conventions.  Conversion to 0-based indices for internal computation
is the responsibility of the caller.
"""

from __future__ import annotations

import re
from typing import Final

from asymmetry.core.instruments.catalog import PROFILES
from asymmetry.core.instruments.layout import (
    GENERIC_INSTRUMENT,
    PROJECTION_TINTS,
    TRANSVERSE_PROJECTION_TINTS,
    AsymmetryProjection,
    BankLayout,
    DetectorSegment,
    GroupDefinition,
    InstrumentLayout,
    PresetGrouping,
    ReferenceArrow,
    generic_layout,
)
from asymmetry.core.instruments.recognition import canonical_layout_name, recognize_layout
from asymmetry.core.instruments.resolution import (
    InstrumentContext,
    assess_instrument,
    layout_key_for_selection,
)
from asymmetry.core.instruments.resolve import layout_from_profile
from asymmetry.core.transform.grouping import resolve_group_indices

__all__ = [
    "DetectorSegment",
    "BankLayout",
    "GroupDefinition",
    "AsymmetryProjection",
    "PresetGrouping",
    "PresetLayoutMismatchError",
    "ReferenceArrow",
    "InstrumentLayout",
    "INSTRUMENT_NAMES",
    "GENERIC_INSTRUMENT",
    "CANONICAL_VECTOR_AXES",
    "PROJECTION_TINTS",
    "TRANSVERSE_PROJECTION_TINTS",
    "derive_projection_pairs",
    "projection_memberships",
    "get_instrument_layout",
    "generic_layout",
    "layout_detector_labels",
    "detector_label_key",
    "preset_grouping_for_run",
    "recommend_grouping_preset",
    "recommend_grouping_preset_for_run",
    "instrument_display_name",
    "instrument_choices_for",
    "variant_for_histograms",
    "detect_instrument",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


#: Fixed semantic frame tints for the canonical EMU vector projections.  Chosen
#: away from the common run-colour palette so they read as chrome, not data.

#: Fixed semantic frame tints for the transverse-field dual-grouping projections
#: (MuSR / HiFi).  Like :data:`PROJECTION_TINTS` these are muted and chosen away
#: from both the run-colour palette and the EMU vector tints, since the two
#: instruments are never shown together.  ``Top-Bottom`` keeps the same rose tint
#: on MuSR and HiFi for cross-instrument consistency.


# ---------------------------------------------------------------------------
# Projection derivation
# ---------------------------------------------------------------------------

#: Canonical EMU vector-polarization axes, in payload order. The single source
#: of truth shared by the grouping dialog (per-projection alpha table ordering
#: and canonical detection) and the main window's alpha resolver, so the axis
#: set never drifts between them. The per-axis alpha table displays them P_z
#: first (``reversed``).
CANONICAL_VECTOR_AXES: Final[tuple[str, ...]] = ("P_x", "P_y", "P_z")

#: Legacy canonical vector group names → axis label, used to infer projections
#: for grouping payloads saved before projections were declared explicitly.
_LEGACY_VECTOR_AXIS_NAMES: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "P_x": (("px left",), ("px right",)),
    "P_y": (("py top", "py up"), ("py bottom", "py down")),
    "P_z": (("pz forward",), ("pz backward",)),
}


def derive_projection_pairs(
    groups: dict | None,
    group_names: dict | None = None,
    projections: object = None,
) -> dict[str, tuple[int, int]]:
    """Return ordered ``{label: (forward_gid, backward_gid)}`` for a grouping.

    This is the single source of truth for resolving a multi-projection grouping
    into its forward/backward pairs, shared by the grouping dialog and the main
    window.

    Resolution order:

    1. An explicit ``projections`` declaration — a sequence of
       :class:`AsymmetryProjection` or plain
       ``{"label", "forward_group", "backward_group"}`` dicts.  Only projections
       whose group IDs exist and are non-empty in ``groups`` are returned, in
       declaration order.
    2. Fallback: the legacy canonical EMU vector group names in ``group_names``,
       so groupings saved before projections were declared still resolve.  This
       path is all-or-nothing (it returns ``{}`` unless all three axes resolve),
       matching the original behaviour.
    """
    if not isinstance(groups, dict) or not groups:
        return {}

    norm_groups: dict[int, object] = {}
    for gid, members in groups.items():
        try:
            norm_groups[int(gid)] = members
        except (TypeError, ValueError):
            continue

    def _has(gid: int) -> bool:
        return gid in norm_groups and bool(norm_groups.get(gid))

    if projections:
        pairs: dict[str, tuple[int, int]] = {}
        for proj in projections:
            if isinstance(proj, AsymmetryProjection):
                label, fwd, bwd = proj.label, proj.forward_group, proj.backward_group
            elif isinstance(proj, dict):
                label = proj.get("label")
                fwd = proj.get("forward_group")
                bwd = proj.get("backward_group")
            else:
                continue
            try:
                fwd_i, bwd_i = int(fwd), int(bwd)
            except (TypeError, ValueError):
                continue
            if label and _has(fwd_i) and _has(bwd_i):
                pairs[str(label)] = (fwd_i, bwd_i)
        return pairs

    names = group_names if isinstance(group_names, dict) else {}
    by_name: dict[str, int] = {}
    for gid, name in names.items():
        try:
            by_name[str(name).strip().lower()] = int(gid)
        except (TypeError, ValueError):
            continue

    def _find(candidates: tuple[str, ...]) -> int | None:
        for cand in candidates:
            gid = by_name.get(cand)
            if gid is not None and _has(gid):
                return gid
        return None

    legacy_pairs: dict[str, tuple[int, int]] = {}
    for label, (fwd_names, bwd_names) in _LEGACY_VECTOR_AXIS_NAMES.items():
        fwd = _find(fwd_names)
        bwd = _find(bwd_names)
        if fwd is None or bwd is None:
            return {}
        legacy_pairs[label] = (fwd, bwd)
    return legacy_pairs


def projection_memberships(groupings: list[dict]) -> dict[str, list[int]]:
    """Place several runs' groupings on shared projection subplots.

    Returns ordered ``{label: [index into groupings]}``. A grouping resolving two
    or more projections joins each of them. A single-pair grouping measures one
    polarization component, so it joins the projection whose forward/backward
    groups hold exactly its own detectors on the same instrument — EMU
    ``Longitudinal`` is the ``Vector Polarization`` ``P_z`` pair, GPS
    ``Longitudinal`` the WEP ``FB`` pair. ``{}`` when no grouping has projections
    or a single-pair grouping matches none: the runs then share one plain pane.
    """
    resolved = [
        derive_projection_pairs(g.get("groups"), g.get("group_names"), g.get("projections"))
        for g in groupings
    ]
    detectors: dict[tuple[str, frozenset[int], frozenset[int]], str] = {}
    memberships: dict[str, list[int]] = {}
    for index, (grouping, pairs) in enumerate(zip(groupings, resolved)):
        if len(pairs) < 2:
            continue
        instrument = str(grouping.get("instrument") or "")
        for label, (fwd, bwd) in pairs.items():
            groups = grouping["groups"]
            key = (
                instrument,
                frozenset(resolve_group_indices(groups, fwd)),
                frozenset(resolve_group_indices(groups, bwd)),
            )
            detectors.setdefault(key, label)
            memberships.setdefault(label, []).append(index)
    if not memberships:
        return {}
    for index, (grouping, pairs) in enumerate(zip(groupings, resolved)):
        if len(pairs) >= 2:
            continue
        groups = grouping.get("groups") or {}
        key = (
            str(grouping.get("instrument") or ""),
            frozenset(resolve_group_indices(groups, grouping.get("forward_group"))),
            frozenset(resolve_group_indices(groups, grouping.get("backward_group"))),
        )
        label = detectors.get(key)
        if label is None:
            return {}
        memberships[label].append(index)
    return {label: sorted(members) for label, members in memberships.items()}


# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------

#: Registry keys of all supported instrument layouts.  ``"GPS"`` (6-detector
#: PSI-BIN), ``"GPS-RD"`` (11-detector ROOT sub-detectors) and ``"GPS-RD15"``
#: (15-detector ROOT export carrying the combined counters *and* the
#: sub-detectors) are three variants of the one physical GPS instrument; they
#: share the display name "GPS" and are collapsed to a single dropdown entry by
#: :func:`~asymmetry.core.instrument.instrument_choices_for`.
_DISPLAY_ORDER: Final[tuple[str, ...]] = (
    "HiFi",
    "MuSR",
    "EMU",
    "FLAME",
    "HAL",
    "GPS",
    "GPS-RD",
    "GPS-RD15",
)


_LAYOUT_KEYS = tuple(
    config.layout_key or profile.identity.layout_key
    for profile in PROFILES
    for config in profile.detectors.acquisitions
)
INSTRUMENT_NAMES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*[name for name in _DISPLAY_ORDER if name in _LAYOUT_KEYS], *_LAYOUT_KEYS))
)


#: Layout name of :func:`generic_layout`. It is not a registry key: the layout
#: depends on the run's detector count, so callers build it per run.


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LAYOUTS: dict[str, InstrumentLayout] | None = None


def _build_registry() -> dict[str, InstrumentLayout]:
    return {
        config.layout_key or profile.identity.layout_key: layout_from_profile(profile, config.key)
        for profile in PROFILES
        for config in profile.detectors.acquisitions
    }


def get_instrument_layout(name: str) -> InstrumentLayout:
    """Return the :class:`InstrumentLayout` for *name*.

    Parameters
    ----------
    name:
        One of the names in :data:`~asymmetry.core.instrument.INSTRUMENT_NAMES` (case-sensitive).

    Raises
    ------
    KeyError
        If *name* is not a known instrument.
    """
    global _LAYOUTS
    if _LAYOUTS is None:
        _LAYOUTS = _build_registry()
    if name in _LAYOUTS:
        return _LAYOUTS[name]
    resolved = canonical_layout_name(name)
    if resolved is not None:
        return _LAYOUTS[resolved]
    return _LAYOUTS[name]


def recommend_grouping_preset(layout: InstrumentLayout, field_direction: str | None) -> str | None:
    """Recommend a grouping preset for *layout* given the run's field geometry.

    A pure helper the Grouping / Detector-Layout dialog uses to nudge the user
    away from a grouping that washes out the signal — never to auto-apply one.

    Geometry selects a recommendation only. Applying a preset remains an explicit
    user action and preserves its declared channel memberships and leg order.

    Parameters
    ----------
    layout:
        The instrument layout whose presets are the recommendation candidates.
    field_direction:
        The applied-field geometry from run metadata
        (``metadata["field_direction"]``): ``"Transverse"``, ``"Longitudinal"``,
        ``"Zero field"``, or an empty/unknown string.  These values are produced
        by
        :meth:`asymmetry.core.io.nexus.NeXusMuonLoader._field_direction_from_state`
        and the PSI free-text classifier
        :func:`asymmetry.core.io.base.field_direction_from_text`.

    Returns
    -------
    str or None
        The name of a preset in ``layout.presets`` to recommend, or ``None``
        when no nudge applies: non-transverse geometry (longitudinal / zero
        field), an unknown/absent field direction, or no suitable transverse
        preset on this instrument.  For transverse data a single forward/backward
        pair (a preset with no declared projections, e.g. GPS
        ``Spin-rotated (B+U/F+D)``) is preferred over a multi-projection
        vector/WEP preset, so the recommendation is directly usable for the basic
        time-domain fit the longitudinal default would collapse.

    Notes
    -----
    This is a *recommendation* only.  Whether to surface the nudge — i.e. whether
    the current preset already matches the recommendation — is the caller's
    decision; this helper does not know the current preset.
    """
    if not isinstance(field_direction, str) or field_direction.strip().casefold() != "transverse":
        # No nudge for longitudinal / zero-field / unknown geometries.
        return None

    # Candidate transverse presets: name carries a spin-rotated, transverse, or
    # (HAL-9500) octant marker. The longitudinal default never matches these
    # tokens, so it is excluded automatically. HAL's "Per-octant" preset is
    # itself a high-field grouping (each azimuthal wedge is a combined F+B
    # opposed pair), so it counts as a transverse candidate alongside
    # "Transverse (opposed pairs)".
    transverse = [
        name
        for name in layout.presets
        if "rotat" in name.lower() or "transverse" in name.lower() or "octant" in name.lower()
    ]
    if not transverse:
        return None

    # Prefer a single forward/backward pair (no declared projections): it drops
    # straight into the ordinary time-domain asymmetry fit that the longitudinal
    # default washes out. Multi-projection presets (vector / WEP) are a second
    # choice. Declaration order breaks ties within each tier.
    single_pair = [name for name in transverse if not layout.presets[name].projections]
    return (single_pair or transverse)[0]


def recommend_grouping_preset_for_run(
    *,
    n_histograms: int,
    metadata: dict | None,
    source_file: str | None = None,
    current_preset: str | None = None,
    instrument_context: InstrumentContext | None = None,
) -> str | None:
    """Recommend a transverse grouping preset for a *loaded run*, or ``None``.

    A run-level convenience over :func:`recommend_grouping_preset`: it resolves
    the run's acquired instrument layout and suppresses the nudge when the
    run's *current* grouping already is the
    recommended preset. Where the bare :func:`recommend_grouping_preset` is
    layout-only and preset-agnostic, this folds in the run's field geometry and
    current preset so a caller can decide in one call whether to surface the
    "the current grouping washes out the precession" hint — used by the plot and
    Fourier panels, which (unlike the grouping dialog) do not already hold a
    resolved layout or the in-dialog draft preset name.

    Parameters
    ----------
    n_histograms:
        Number of detector histograms in the run (drives layout detection and
        the BIN/ROOT variant choice).
    metadata:
        Run metadata; ``metadata["field_direction"]`` supplies the geometry and
        the instrument and histogram labels drive acquisition matching.
    source_file:
        Optional source path, a final instrument-detection hint.
    current_preset:
        The grouping preset already applied to the run
        (``run.grouping["grouping_preset"]``), or ``None`` when unknown/custom.

    instrument_context:
        The loaded run's effective instrument choice, including conflicts.

    Returns
    -------
    str or None
        The name of a preset to recommend, or ``None`` when the run is not
        transverse-field, no instrument layout resolves, the layout has no
        suitable transverse preset, or ``current_preset`` already equals the
        recommendation.
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    count = int(n_histograms or 0)
    selection = (
        instrument_context.selection
        if instrument_context is not None
        else assess_instrument(count, metadata=metadata, source_file=source_file).selection
    )
    if selection is None:
        return None
    name = layout_key_for_selection(selection, count)
    if name is None:
        return None
    layout = get_instrument_layout(name)
    recommended = recommend_grouping_preset(layout, metadata.get("field_direction"))
    if recommended is None or recommended == current_preset:
        return None
    return recommended


def detect_instrument(n_histograms, *, metadata=None, source_file=None) -> str | None:
    """Return the public recognized layout name; map compatibility is assessed separately."""
    selected = assess_instrument(n_histograms, metadata=metadata, source_file=source_file).selection
    if selected is None:
        return None
    profile = next(item for item in PROFILES if item.identity.key == selected.profile_key)
    if not profile.detectors.acquisitions:
        return None
    return (
        layout_key_for_selection(selected, int(n_histograms))
        or recognize_layout(n_histograms, metadata=metadata, source_file=source_file)
        or profile.identity.layout_key
    )


def instrument_display_name(name: str) -> str:
    """Return the user-facing dropdown name for a layout registry key.

    Variant keys (e.g. ``"GPS-RD"``) map to their shared display name
    (``"GPS"``); every other key maps to itself.
    """
    profile = next((item for item in PROFILES if item.identity.key == name), None)
    if profile is not None:
        return profile.identity.name
    try:
        return get_instrument_layout(name).display
    except KeyError:
        return name


def _variant_families() -> dict[str, list[str]]:
    """Map each display name to its registry keys, in ``INSTRUMENT_NAMES`` order.

    Two layouts that share a ``display_name`` are variants of one physical
    instrument (e.g. the 6-detector ``"GPS"``, the 11-detector ``"GPS-RD"`` and
    the 15-detector ``"GPS-RD15"`` all display "GPS"); the first key listed is the
    family default.  The family
    relationship is derived from the layouts themselves, so adding a variant only
    requires giving it the shared ``display_name`` — no separate registry to keep
    in sync.
    """
    families: dict[str, list[str]] = {}
    for key in INSTRUMENT_NAMES:
        families.setdefault(instrument_display_name(key), []).append(key)
    return families


def instrument_choices_for(active_name: str | None = None) -> list[tuple[str, str]]:
    """Return ``[(display_name, registry_key)]`` for the instrument dropdown.

    Variant families (e.g. the 6-detector PSI-BIN ``"GPS"``, the 11-detector ROOT
    ``"GPS-RD"`` and the 15-detector ROOT ``"GPS-RD15"``) collapse to a single
    entry under their shared display name, so the user only ever sees one "GPS".
    The variant exposed for a family
    is the one whose key equals *active_name* (the currently-loaded layout);
    otherwise the family default (first registry key) is used.
    """
    choices: list[tuple[str, str]] = []
    for display, keys in _variant_families().items():
        if len(keys) == 1:
            choices.append((display, keys[0]))
        else:
            chosen = active_name if active_name in keys else keys[0]
            choices.append((display, chosen))
    return choices


def variant_for_histograms(name: str, n_histograms: int) -> str:
    """Return the registry key of *name*'s variant whose detector count fits the run.

    Layout variants that share a display name (e.g. GPS BIN with 6 detectors vs
    the GPS ROOT exports with 11 or 15) are distinguished by detector count.
    Given a layout *name* and a run's ``n_histograms``, return the sibling variant
    whose ``n_detectors`` equals that count, so the layout always matches the
    data.  Returns *name* unchanged when it has no sibling, the count is unknown,
    or no sibling fits.

    A returned key whose ``n_detectors`` still disagrees with the run (no sibling
    fitted) is a *mismatch*, not a silent fallback: turning a preset into a
    concrete grouping through :func:`preset_grouping_for_run` rejects it rather
    than mapping the preset's detector ids onto the wrong histograms.
    """
    keys = _variant_families().get(instrument_display_name(name), [name])
    if len(keys) <= 1 or not n_histograms:
        return name
    for key in keys:
        try:
            if get_instrument_layout(key).n_detectors == int(n_histograms):
                return key
        except KeyError:
            continue
    return name


# ---------------------------------------------------------------------------
# Preset -> concrete grouping (validated against the run)
# ---------------------------------------------------------------------------


class PresetLayoutMismatchError(ValueError):
    """A grouping preset does not fit the run it is being applied to.

    Raised by :func:`preset_grouping_for_run` when the layout supplying the preset
    describes a different detector arrangement than the run actually carries, so
    the preset's detector ids would land on the wrong histograms.  This is always
    a resolution bug in the caller (usually a stale or hand-picked layout variant
    where :func:`variant_for_histograms` should have been used), never something
    the data can be at fault for — hence a hard error rather than a silent
    best-effort grouping.
    """


#: Detector-label spellings that mean the same counter.  Instrument layouts use
#: the manual's full words while data files use the terse export spellings, so
#: the two are compared through this map.
_DETECTOR_LABEL_ALIASES: Final[dict[str, str]] = {
    "forw": "forward",
    "fwd": "forward",
    "back": "backward",
    "bwd": "backward",
    "righ": "right",
}


def detector_label_key(label: object) -> str:
    """Return the comparison key for one detector label.

    Case/punctuation and documented aliases share one key (Forward / Forw).
    """
    compact = re.sub(r"[^a-z0-9]+", "", str(label).lower())
    return _DETECTOR_LABEL_ALIASES.get(compact, compact)


def layout_detector_labels(layout: InstrumentLayout) -> dict[int, str]:
    """Return ``{detector_id: label}`` for the detectors *layout* names.

    One entry per clickable detector that declares a label (label-free banks such
    as HiFi/MuSR/EMU return an empty mapping).  This is the layout's *expectation*
    of what each histogram index holds, and is what
    :func:`preset_grouping_for_run` checks against the run's own labels.
    """
    return {seg.detector_id: seg.label for seg in layout.active_segments if seg.label is not None}


def preset_grouping_for_run(
    layout: InstrumentLayout,
    preset_name: str,
    *,
    n_histograms: int,
    labels: list[str] | None = None,
) -> dict[int, list[int]]:
    """Resolve a preset into ``{group_id: [detector_ids]}`` for a specific run.

    The supported way to turn an :class:`InstrumentLayout` preset into a concrete
    grouping, because it *checks the layout against the run first*.  Detector ids
    in this module are positional (detector *N* → histogram ``N − 1``), so a
    layout that describes a different arrangement than the run carries produces a
    grouping that is wrong without being empty, invalid, or in any other way
    detectable downstream.  That is exactly how the 15-histogram GPS ROOT export
    used to be grouped with the 11-histogram ``GPS-RD`` layout, silently summing
    the Down counter into the "up" group.

    Two checks, both raising :class:`PresetLayoutMismatchError`:

    1. **Wrong variant.**  Another variant of the same physical instrument (a
       sibling sharing *layout*'s display name) has exactly *n_histograms*
       detectors while *layout* does not — i.e. :func:`variant_for_histograms`
       would have picked a different layout and the caller did not ask it.
    2. **Labels do not line up.**  The layout's expected label for a detector id
       (:func:`layout_detector_labels`) disagrees with the run's own label at that
       position.  Spelling differences between the manual and the export are
       absorbed by :func:`detector_label_key`, so a surviving disagreement means
       the ids address different counters.

    Membership is authored here. Reduction rejects absent required channels;
    a partial file cannot trim a preset or replace it implicitly.

    Parameters
    ----------
    layout:
        The layout supplying the preset.  Resolve it with
        :func:`variant_for_histograms` when the instrument has variants.
    preset_name:
        A key of ``layout.presets``.
    n_histograms:
        Number of histograms in the run.  ``0``/unknown skips check 1.
    labels:
        The run's per-histogram labels in histogram order (``run.grouping[
        "histogram_labels"]`` / ``metadata["histogram_labels"]``), if known.
        Fewer labels than the layout has detectors skips check 2.

    Returns
    -------
    dict
        ``{group_id: [detector_ids]}`` with 1-based detector ids, sorted, empty
        groups dropped. Required channels stay authored; reduction checks availability.

    Raises
    ------
    KeyError
        If *preset_name* is not a preset of *layout*.
    PresetLayoutMismatchError
        If the layout does not describe this run (see above).
    """
    preset = layout.presets[preset_name]

    count = int(n_histograms or 0)
    if count > 0 and layout.n_detectors != count:
        fitted = variant_for_histograms(layout.name, count)
        if fitted != layout.name:
            raise PresetLayoutMismatchError(
                f"Layout {layout.name!r} describes {layout.n_detectors} detectors but the "
                f"run has {count} histograms, so preset {preset_name!r} would map its "
                f"detector ids onto the wrong histograms — use the {fitted!r} variant instead."
            )

    expected = layout_detector_labels(layout)
    actual = list(labels or [])
    if expected and len(actual) >= layout.n_detectors:
        conflicts: list[str] = []
        for det_id in sorted(
            {int(d) for gdef in preset.groups.values() for d in gdef.detector_ids}
        ):
            if not 1 <= det_id <= len(actual):
                continue
            want = expected.get(det_id)
            if want is None:
                continue
            got = actual[det_id - 1]
            if detector_label_key(want) != detector_label_key(got):
                conflicts.append(f"detector {det_id}: layout says {want!r}, run says {got!r}")
        if conflicts:
            raise PresetLayoutMismatchError(
                f"Layout {layout.name!r} does not match the run's detector labels, so preset "
                f"{preset_name!r} would group the wrong counters — " + "; ".join(conflicts) + "."
            )

    groups: dict[int, list[int]] = {}
    for gid, gdef in preset.groups.items():
        detector_ids = sorted(int(d) for d in gdef.detector_ids)
        if detector_ids:
            groups[int(gid)] = detector_ids
    return groups
