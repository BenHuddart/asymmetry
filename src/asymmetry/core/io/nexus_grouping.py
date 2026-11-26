"""NeXus detector declarations, period indexing, and supported grouping defaults."""

import warnings
from dataclasses import dataclass

import numpy as np

from asymmetry.core.instruments.catalog import profile_by_key
from asymmetry.core.instruments.defaults import absent_file_grouping_default, file_grouping_default
from asymmetry.core.instruments.layout import PresetGrouping
from asymmetry.core.instruments.resolution import resolve_instrument


@dataclass(frozen=True)
class GroupingSelection:
    groups: dict[int, list[int]]
    pair: tuple[int, int] | None

    @property
    def payload(self) -> dict:
        return {
            "groups": self.groups,
            **(
                {"forward_group": self.pair[0], "backward_group": self.pair[1]} if self.pair else {}
            ),
        }


def nexus_grouping(
    values, n_detectors: int, default: PresetGrouping | None = None
) -> GroupingSelection:
    """Present tables retain their memberships; absence alone permits a default."""
    if values is None:
        if default is not None:
            return GroupingSelection(
                {gid: list(group.detector_ids) for gid, group in default.groups.items()},
                (default.forward_group, default.backward_group),
            )
        split = max(1, n_detectors // 2)
        return GroupingSelection(
            {
                1: list(range(1, split + 1)),
                2: list(range(split + 1, n_detectors + 1)) or list(range(1, n_detectors + 1)),
            },
            (1, 2),
        )
    table = np.asarray(values)
    if table.shape != (n_detectors,) or not np.issubdtype(table.dtype, np.integer):
        raise ValueError("Grouping declaration must contain one integer per acquired detector")
    groups = {}
    for ordinal, group in enumerate(table, start=1):
        if group > 0:
            groups.setdefault(int(group), []).append(ordinal)
    if not groups:
        raise ValueError("Grouping declaration contains no positive detector group")
    ordered = sorted(groups)
    return GroupingSelection(groups, tuple(ordered[:2]) if len(ordered) >= 2 else None)


def nexus_period_grouping(values, n_periods: int, n_detectors: int):
    """Decode exact shared-detector or period-major histogram declarations.

    NeXus muon V1, Cottrell Rev. 8, notes 4 and 6: grouping indexes
    histogram_number, including switching-state doubling.
    """
    if values is None:
        return [None] * n_periods
    table = np.asarray(values)
    if table.shape == (n_detectors,):
        return [table] * n_periods
    if table.shape in {(n_periods * n_detectors,), (n_periods, n_detectors)}:
        return list(table.reshape(n_periods, n_detectors))
    raise ValueError(
        "Grouping declaration must contain one integer per acquired detector or histogram"
    )


def grouping_membership(grouping: dict) -> tuple:
    """Compare detector membership and ordered legs independently of presentation."""
    return (
        tuple(
            sorted(
                (int(gid), tuple(sorted(members)))
                for gid, members in grouping.get("groups", {}).items()
            )
        ),
        grouping.get("forward_group"),
        grouping.get("backward_group"),
    )


class NexusGroupingWarning(UserWarning):
    """An invalid legacy declaration was replaced by a documented preset on import."""


def _integer_scalar(value, expected: int) -> bool:
    array = np.asarray(value)
    return (
        array.shape in {(), (1,)}
        and np.issubdtype(array.dtype, np.integer)
        and array.item() == expected
    )


def legacy_zero_grouping_declaration(
    entry, histogram_group, counts, grouping, switching, orientation, read_optional
) -> dict | None:
    """Recognise the verified single-period V1 layout before period table reshaping."""
    table = np.asarray(grouping)
    if (
        counts.ndim != 2
        or table.shape != (counts.shape[0],)
        or not np.issubdtype(table.dtype, np.integer)
        or np.any(table != 0)
        or not _integer_scalar(switching, 1)
    ):
        return None
    node = histogram_group["grouping"]
    available = node.attrs.get("available")
    if not _integer_scalar(available, 1):
        return None

    detector_count = read_optional(entry, "instrument/detector/number")
    if not _integer_scalar(detector_count, counts.shape[0]):
        return None
    period_type = read_optional(entry, "instrument/beam/period_type")
    if not _integer_scalar(period_type, 1):
        return None
    ordinals = read_optional(histogram_group, "histogram_number")
    if ordinals is not None:
        indices = np.asarray(ordinals)
        if not np.issubdtype(indices.dtype, np.integer) or not np.array_equal(
            indices, np.arange(1, len(table) + 1)
        ):
            return None
    for path in ("instrument/name", "instrument/detector/orientation"):
        declaration = np.asarray(read_optional(entry, path))
        if declaration.shape not in {(), (1,)} or not isinstance(declaration.item(), (str, bytes)):
            return None
    canonical = orientation.strip().casefold()
    if canonical not in {"l", "longitudinal"}:
        return None
    return {"values": table.tolist(), "available": int(np.asarray(available).item())}


def apply_nexus_grouping(run, table, legacy_zero_declaration, *, replay=None) -> None:
    """Resolve the file grouping, then capture its scientific and source attribution."""
    n_detectors = len(run.histograms)
    context = resolve_instrument(run)
    identity = context.selection
    default = None
    fallback_reason = None
    if table is None and replay is None:
        default = absent_file_grouping_default(identity, run.metadata)
    elif legacy_zero_declaration is not None and context.observed.selection is not None:
        observed = context.observed.selection
        profile = profile_by_key(observed.profile_key)
        instrument = profile.identity.name
        labels = {label.casefold() for label in (instrument, *profile.recognition.aliases)}
        if run.metadata["instrument"].casefold() in labels:
            default = file_grouping_default(observed, run.metadata, "legacy_v1_zero_grouping")
        if default is not None:
            identity = observed
            run.grouping["nexus_grouping_declaration"] = legacy_zero_declaration
            fallback_reason = "nexus_v1_all_zero"
            table = None
            if replay is None:
                message = (
                    "The file’s detector-grouping table contains only zeros. "
                    f"Applied the standard {instrument} forward/backward grouping."
                )
                run.metadata["nexus_grouping_warning"] = message
                warnings.warn(message, NexusGroupingWarning, stacklevel=2)

    run.grouping.update(nexus_grouping(table, n_detectors, default).payload)
    if default is not None:
        run.grouping["grouping_preset"] = default.name
        run.grouping["grouping_provenance"] = {
            "origin": "profile",
            "profile_key": identity.profile_key,
            "profile_revision": identity.profile_revision,
            "acquisition": identity.acquisition,
            **({"fallback_reason": fallback_reason} if fallback_reason is not None else {}),
        }
