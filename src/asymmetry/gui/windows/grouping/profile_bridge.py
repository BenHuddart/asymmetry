"""Draft-profile / payload bridging helpers for the grouping editor.

The grouping dialog edits an in-memory :class:`~asymmetry.core.project.profiles.GroupingProfile`
(the *draft*) through the historical per-control form, which reads and writes a
flat ``run.grouping`` *payload* dict. This module is the seam between the two:

* :func:`payload_from_profile_for_preview` resolves a draft against a preview run
  so the form controls (which still speak "payload") can be seeded from the draft
  plus the preview run's per-run facts.
* :func:`profile_from_form_payload` lifts the shareable settings back out of the
  payload the form built into an updated draft (preserving the draft's name /
  fingerprint / active flag).
* :func:`preset_payload` turns an instrument :class:`PresetGrouping` into the
  groups / names / forward / backward / projections a form apply needs, and
  :func:`payload_matches_preset` reports whether a payload still equals a named
  preset (so a drifted draft can drop its stale ``grouping_preset``).

Keeping this logic out of the ~2.7k-line dialog module makes the preset-drift and
draft-isolation rules testable in isolation.
"""

from __future__ import annotations

from typing import Any

from asymmetry.core.instrument import InstrumentLayout, PresetGrouping
from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    profile_from_payload,
    resolve_effective_grouping,
)


def payload_from_profile_for_preview(profile: GroupingProfile, preview_run) -> dict[str, Any]:
    """Resolve *profile* against *preview_run* into a full grouping payload.

    This is exactly :func:`resolve_effective_grouping`, exposed under a name that
    documents its dialog use: seed the payload-speaking form controls from the
    draft profile merged with the preview run's own per-run facts (t0, good-bin
    window, file deadtime, period tables). Changing the preview run therefore
    only changes the per-run facts, never the draft's shareable settings.
    """
    return resolve_effective_grouping(profile, preview_run)


def profile_from_form_payload(
    payload: dict[str, Any],
    *,
    name: str,
    fingerprint: ProfileFingerprint,
    active: bool = True,
) -> GroupingProfile:
    """Lift the shareable settings out of a form-built *payload* into a profile.

    Thin wrapper over :func:`profile_from_payload` that names the dialog use: the
    form's ``_current_grouping_payload`` output is turned back into the draft
    profile on every edit and at apply time. Per-run/file-derived keys in the
    payload are ignored (they come back from the run at resolve time).
    """
    return profile_from_payload(payload, name, fingerprint, active=active)


def preset_payload(instrument: InstrumentLayout, preset_name: str) -> dict[str, Any] | None:
    """Return the groups/names/slots/projections a preset assigns, or ``None``.

    The returned dict has the payload keys the grouping form consumes when a
    preset is applied to the draft: ``groups`` (1-based detector id lists),
    ``group_names``, ``forward_group``, ``backward_group``, ``grouping_preset``,
    and ``projections``. ``None`` when the instrument has no such preset.
    """
    preset = instrument.presets.get(preset_name)
    if preset is None:
        return None
    groups: dict[int, list[int]] = {}
    group_names: dict[int, str] = {}
    for gid, gdef in preset.groups.items():
        if not gdef.detector_ids:
            continue
        groups[int(gid)] = sorted(int(d) for d in gdef.detector_ids)
        if gdef.name:
            group_names[int(gid)] = str(gdef.name)
    return {
        "groups": groups,
        "group_names": group_names,
        "forward_group": int(preset.forward_group),
        "backward_group": int(preset.backward_group),
        "grouping_preset": preset_name,
        "instrument": instrument.name,
        "projections": [dict(p.to_payload()) for p in preset.projections],
    }


def _preset_groups(preset: PresetGrouping) -> dict[int, set[int]]:
    return {
        int(gid): {int(d) for d in gdef.detector_ids}
        for gid, gdef in preset.groups.items()
        if gdef.detector_ids
    }


def payload_matches_preset(
    payload: dict[str, Any], instrument: InstrumentLayout, preset_name: str
) -> bool:
    """Return whether *payload* still equals the named preset of *instrument*.

    Compares the same fields the detector-layout editor uses for its
    ``(Current: …)`` status: the group→detector-id sets, the group names, and the
    forward/backward slot ids. When any differs the draft has *drifted* from the
    preset and the caller should clear the stale ``grouping_preset`` marker.
    Detector ids in *payload* are 1-based (as the form stores them).
    """
    preset = instrument.presets.get(preset_name)
    if preset is None:
        return False

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, dict):
        return False
    current_groups: dict[int, set[int]] = {}
    for key, values in raw_groups.items():
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        ids = {int(v) for v in values if _is_int(v)}
        if ids:
            current_groups[gid] = ids
    if current_groups != _preset_groups(preset):
        return False

    raw_names = payload.get("group_names")
    current_names = (
        {int(k): str(v) for k, v in raw_names.items() if _is_int(k) and str(v).strip()}
        if isinstance(raw_names, dict)
        else {}
    )
    preset_names = {int(gid): str(gdef.name) for gid, gdef in preset.groups.items() if gdef.name}
    if current_names != preset_names:
        return False

    try:
        if int(payload.get("forward_group", -1)) != int(preset.forward_group):
            return False
        if int(payload.get("backward_group", -1)) != int(preset.backward_group):
            return False
    except (TypeError, ValueError):
        return False
    return True


def _is_int(value: object) -> bool:
    try:
        int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return True


__all__ = [
    "payload_from_profile_for_preview",
    "profile_from_form_payload",
    "preset_payload",
    "payload_matches_preset",
]
