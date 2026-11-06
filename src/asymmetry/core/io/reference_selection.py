"""Source-bound selection of an acquired reference, before correction arithmetic."""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from asymmetry.core.data.dataset import Run
from asymmetry.core.io.periods import (
    acquired_period_histograms,
    reference_mapping_requires_replay,
    select_period_grouping,
    selected_period_index,
    source_run_of,
)
from asymmetry.core.project.reference_restore import restore_reference_selection


def reference_source(source: str) -> str:
    """Normalise a declared path without reading acquisition data."""
    return str(Path(source).expanduser().resolve()) if source else ""


def reference_selection(run: Run) -> dict:
    """Declare the acquired period currently selected in a loaded run."""
    index = selected_period_index(run.grouping)
    return {
        "source_file": run.source_file,
        "run_number": source_run_of(run),
        "period_number": index + 1 if index is not None else run.metadata.get("period_number", 1),
    }


def reference_observations(run: Run, mode: str | None = None) -> frozenset[tuple[str, int, int]]:
    """Acquired selections represented by a live single-period or G±R view."""
    from asymmetry.core.utils.constants import PeriodMode

    if run.metadata.get("raw_period_histograms_available") is False:
        return frozenset()
    selection = reference_selection(run)
    active = mode or run.grouping.get("period_mode")
    if run.grouping.get("period_histograms") and active in {
        PeriodMode.GREEN_MINUS_RED,
        PeriodMode.GREEN_PLUS_RED,
    }:
        periods = (1, 2)
    else:
        index = selected_period_index(run.grouping, active)
        periods = (index + 1 if index is not None else selection["period_number"],)
    return frozenset(
        (reference_source(run.source_file) or f"memory:{id(run)}", selection["run_number"], period)
        for period in periods
    )


def select_reference_run(run: Run, payload: dict) -> Run | None:
    """Match the declared source, acquisition, and period before consuming counts."""
    source = str(payload.get("source_file") or "")
    if source and reference_source(source) != reference_source(run.source_file):
        return None
    selection = restore_reference_selection(payload, run)
    number = selection.get("run_number")
    if number is not None and int(number) != source_run_of(run):
        return None
    held = run.grouping.get("period_histograms")
    if "period_number" not in selection and held is not None and len(held) > 1:
        raise ValueError("the reference acquired period is ambiguous; declare a period")
    if run.metadata.get("raw_period_histograms_available") is False:
        return None
    period = selection.get("period_number", run.metadata.get("period_number", 1))
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise ValueError("reference acquired period must be a positive integer")
    if held is not None:
        if period > len(held):
            return None
        histograms = acquired_period_histograms(run.grouping, period - 1)
        if histograms is None:
            raise ValueError(
                "the reference acquired period has no raw histograms; reload its source"
            )
        # Reference resolution reads acquired counts; it never reduces a preview
        # or copies detector arrays on the GUI thread.
        grouping = select_period_grouping(run.grouping, period - 1)
        for key in ("period_histograms", "period_reduced", "period_mode"):
            grouping.pop(key, None)
        return replace(
            run,
            histograms=histograms,
            grouping=grouping,
            metadata=dict(run.metadata, source_run_number=source_run_of(run), period_number=period),
        )
    if period != run.metadata.get("period_number", 1):
        return None
    return run


def resolve_loaded_reference(runs: Iterable[Run], payload: dict) -> Run | None:
    """Prefer a live selected view of the same acquisition; reject ambiguity."""
    if reference_mapping_requires_replay(payload.get("period_mapping")):
        raise ValueError("a mapped reference is not one acquired period; select an acquired period")
    matches, seen, mapped = {}, set(), set()
    for run in runs:
        if id(run) in seen:
            continue
        seen.add(id(run))
        selected = select_reference_run(run, payload)
        if selected is None:
            continue
        selection = reference_selection(selected)
        key = (
            reference_source(selected.source_file) or id(run),
            selection["run_number"],
            selection["period_number"],
        )
        if reference_mapping_requires_replay(
            selected.grouping.get("period_mapping", selected.metadata.get("period_mapping"))
        ):
            mapped.add(key)
            continue
        direct, held = matches.setdefault(key, ([], []))
        (direct if selected is run else held).append(selected)
    candidates = [run for direct, held in matches.values() for run in (direct or held)]
    if len(candidates) > 1 or len(matches.keys() | mapped) > 1:
        raise ValueError(
            "the reference acquisition/period is ambiguous; declare a unique source and period"
        )
    if not candidates and mapped:
        raise ValueError("a mapped reference is not one acquired period; select an acquired period")
    return candidates[0] if candidates else None
