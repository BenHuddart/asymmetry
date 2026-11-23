"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.instruments.resolution import (
    InstrumentConflictError,
    RecordedInstrumentChoice,
    resolve_instrument,
)
from asymmetry.core.instruments.source import assess_source, source_for_run
from asymmetry.core.io.base import BaseLoader, LoaderRegistry, LoadResult
from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.periods import (
    PeriodMode,
    period_count,
    period_labels,
    select_period,
)
from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.root import RootLoader
from asymmetry.core.io.run_preparation import ReplayState, prepare_legacy_loaded_run
from asymmetry.core.io.run_range import resolve_run_range, scan_run_files
from asymmetry.core.io.unit_report import unit_report as unit_report
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.deadtime import file_deadtime_table
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest
from asymmetry.core.transform.reference_timing import ReferenceTiming, reference_timing_request

# Register built-in loaders
LoaderRegistry.register(NexusLoader)
LoaderRegistry.register(PsiLoader)
LoaderRegistry.register(RootLoader)


def load(
    filepath: str,
    fmt: str | None = None,
    period: int | str | PeriodMode | None = None,
    *,
    checkpoint=None,
    progress=None,
    replay: ReplayState | None = None,
) -> LoadResult:
    """Load a μSR data file and return a :class:`~asymmetry.core.data.dataset.MuonDataset`.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    fmt : str, optional
        Force a specific format (e.g. ``"nxs"``).  If *None*, the format
        is auto-detected from the file extension.
    period : int or str or PeriodMode, optional
        For period-mode (multi-period) files, select a single period and
        return just that :class:`~asymmetry.core.data.dataset.MuonDataset`. Accepts a 1-based period number
        or, for two-period files, a ``"red"``/``"green"`` label. When *None*
        (default) the historical behaviour is preserved: single/two-period
        files return one :class:`~asymmetry.core.data.dataset.MuonDataset` and 3+ period files a ``list``.
        See :mod:`asymmetry.core.io.periods`.

    checkpoint : callable, optional
        Cooperative cancellation callback, called between hash chunks and around
        decoding by the built-in loaders. Raising aborts the load.
    progress : callable, optional
        Receives bytes read and total bytes for each of the two whole-file hash
        passes bracketing built-in decoding. Decoding itself is format-specific.

    Raises
    ------
    ValueError
        If ``period`` is out of range or an unknown label.
    """
    loader = LoaderRegistry.get_loader(filepath, fmt=fmt)
    callbacks = {
        key: value
        for key, value in (("checkpoint", checkpoint), ("progress", progress), ("replay", replay))
        if value is not None
    }
    from inspect import Parameter, signature

    try:
        parameters = signature(loader.load, follow_wrapped=False).parameters
    except (TypeError, ValueError):
        supported = set()
    else:
        supported = {
            key
            for key, parameter in parameters.items()
            if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }
        if any(p.kind is Parameter.VAR_KEYWORD for p in parameters.values()):
            supported |= {"checkpoint", "progress"}
    callbacks = {key: value for key, value in callbacks.items() if key in supported}
    opaque_replay = "replay" not in supported
    opaque_checkpoint = checkpoint is not None and "checkpoint" not in callbacks
    if opaque_checkpoint:
        checkpoint()
    result = loader.load(filepath, **callbacks)
    if opaque_checkpoint:
        checkpoint()
    for dataset in result if isinstance(result, list) else [result]:
        run = dataset.run
        if run is not None:
            if opaque_replay:
                prepare_legacy_loaded_run(run, replay)
            run.grouping.pop("instrument", None)
            context = resolve_instrument(run)
            if context.requires_choice:
                dataset.metadata["correction_error"] = str(InstrumentConflictError(context))
            if run.instrument_choice is None and context.selection is not None:
                run.instrument_choice = RecordedInstrumentChoice(
                    context.selection, context.observed.evidence
                )
            if run.source_assessment is None:
                run.source_assessment = source_for_run(run)
    if (
        period is None
        and replay is not None
        and not isinstance(result, list)
        and result.run is not None
    ):
        originals = result.run.grouping.get("period_inputs", ())
        matches = [index for index, original in enumerate(originals) if replay.selects(original)]
        if len(matches) == 1:
            period = matches[0] + 1
    if period is None:
        return result
    return select_period(result, period)


def load_background_run(payload: dict, *, checkpoint=None) -> MuonDataset:
    """Load exactly the declared reference acquisition and acquired period."""
    from asymmetry.core.instruments.source import SourceAssessment
    from asymmetry.core.io.reference_selection import reference_selection, resolve_loaded_reference

    if not isinstance(payload, dict) or not payload.get("source_file"):
        raise ValueError("background_run payload has no source_file")
    timing = payload.get("timing")
    if timing is not None and not isinstance(timing, dict):
        raise ValueError("Reference timing must be a source-bound request")
    source = SourceAssessment.from_dict(timing["source"]) if timing and "source" in timing else None
    number = payload.get("run_number", payload.get("legacy_run_number"))
    replay = ReplayState(
        source=source,
        run_number=int(number) if number is not None else None,
        period_number=payload.get("period_number"),
    )
    result = load(
        str(payload["source_file"]),
        replay=replay,
        **({"checkpoint": checkpoint} if checkpoint is not None else {}),
    )
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    candidates = result if isinstance(result, list) else [result]
    selected = resolve_loaded_reference(
        (ds.run for ds in candidates if ds.run is not None), payload
    )
    if selected is None or not selected.histograms:
        raise ValueError(
            "the declared reference acquisition/period is unavailable in the source file"
        )
    dataset = (
        select_period(result, reference_selection(selected)["period_number"])
        if isinstance(result, list) or result.run.grouping.get("period_histograms") is not None
        else result
    )
    return replace(dataset, run=selected, metadata=selected.metadata)


def reference_file_identity(payload: dict | None) -> tuple[int, int, int, int] | None:
    """Read external reference identity without loading counts or validating settings.

    Ordinary writes change size or mtime; replacement also changes inode.
    Tools preserving all these fields must explicitly invalidate cached spectra.
    """
    if not isinstance(payload, dict) or not payload.get("source_file"):
        return None
    try:
        status = Path(str(payload["source_file"])).stat()
    except (OSError, ValueError):
        return None
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def resolve_background_reference(
    payload: dict | None,
    *,
    sample_exposure: ReferenceExposure = ReferenceExposure(),
    datasets: Iterable[MuonDataset] = (),
    cache: dict | None = None,
    allow_file: bool = True,
    reference_loader=None,
) -> BackgroundReference | None:
    """Resolve declared source/period, preferring live inputs over disk snapshots.

    Number-only requests must be unambiguous. Advisory ``allow_file=False``
    resolution uses loaded inputs only, including when a disk cache exists.
    File cache entries retain source-stat invalidation and consumed selection.
    """
    from asymmetry.core.io.reference_selection import (
        reference_selection,
        reference_source,
        resolve_loaded_reference,
    )

    if not isinstance(payload, dict):
        raise ValueError("no reference is recorded")
    source_file = str(payload.get("source_file") or "")
    if (
        not source_file
        and payload.get("run_number") is None
        and payload.get("legacy_run_number") is None
    ):
        raise ValueError("no reference is recorded")
    scale_request = ReferenceScaleRequest.from_payload(payload)
    reference_run = resolve_loaded_reference(
        (ds.run for ds in datasets if ds.run is not None), payload
    )
    live = reference_run is not None
    if reference_run is None:
        if not allow_file:
            return None
        if not source_file:
            raise ValueError("the reference is not loaded and no source file is recorded")
        file_identity = reference_file_identity(payload)
        if cache is not None:
            reference_run = resolve_loaded_reference(
                (
                    cached
                    for (path, _, _, identity), cached in cache.items()
                    if path == reference_source(source_file) and identity == file_identity
                ),
                payload,
            )
        if reference_run is None:
            loader = load_background_run if reference_loader is None else reference_loader
            reference_run = loader(payload).run
            if cache is not None and reference_run is not None:
                selection = reference_selection(reference_run)
                source_key = (
                    reference_source(reference_run.source_file),
                    selection["run_number"],
                    selection["period_number"],
                    file_identity,
                )
                cache[source_key] = reference_run

    if reference_run is None or not reference_run.histograms:
        raise ValueError("the reference has no histograms")

    from asymmetry.core.data.acquisition import ContentIdentity
    from asymmetry.core.project.reference_restore import complete_reference_restore
    from asymmetry.core.transform.corrections import CorrectionUnavailableError

    expected = ContentIdentity.from_dict(payload.get("content_identity"))
    if expected is not None and expected != reference_run.source_binding.observed:
        raise CorrectionUnavailableError(
            "Reference source differs from its recorded content; explicit replacement is required"
        )
    complete_reference_restore(payload, reference_run)
    scaling = scale_request.resolve(
        sample_exposure, ReferenceExposure.from_grouping(reference_run.grouping)
    )

    timing_grouping = dict(reference_run.grouping)
    previous = payload.get("timing")
    if previous is not None:
        if not isinstance(previous, dict) or set(previous) - {"source"} not in (
            {"t0_policy"},
            {"effective_detector_t0_bins"},
        ):
            raise ValueError("Reference timing request requires one policy or absolute table")
        if "t0_policy" in previous and not isinstance(previous["t0_policy"], dict):
            raise ValueError("Reference timing requires an explicit policy mode")
        if not live:
            for key in ("t0_policy", "effective_detector_t0_bins"):
                timing_grouping.pop(key, None)
            timing_grouping.update(previous)
    fresh_live = live and "timing" not in payload
    request = reference_timing_request(
        timing_grouping,
        source_for_run(reference_run) if fresh_live else assess_source(reference_run.metadata),
        previous=previous if previous is not None else None if fresh_live else {},
    )
    timing_grouping["source"] = request["source"]
    timing = ReferenceTiming.from_grouping(
        reference_run.histograms, timing_grouping, reference_run.metadata
    )
    if live or payload.get("timing") is not None:
        payload["timing"] = timing.request()
    return BackgroundReference(
        histograms=list(reference_run.histograms),
        timing=timing,
        instrument_context=resolve_instrument(reference_run),
        file_dead_time_us=tuple(file_deadtime_table(reference_run.grouping)),
        scaling=scaling,
        source_file=reference_run.source_file or source_file,
        content_identity=reference_run.source_binding.observed,
        deadtime_normaliser=good_frames(reference_run.grouping, default=0.0) or None,
        run_number=reference_selection(reference_run)["run_number"],
        period_number=reference_selection(reference_run)["period_number"],
    )


def background_reference_for_grouping(
    reference: BackgroundReference | None, grouping: dict
) -> BackgroundReference | None:
    """Adapt sample-period exposure, retaining Manual intent and acquisition facts."""
    if reference is None:
        return None
    payload = grouping.get("background_run", {"scale_request": reference.scaling.request.to_dict()})
    scaling = ReferenceScaleRequest.from_payload(payload).resolve(
        ReferenceExposure.from_grouping(grouping), reference.scaling.reference
    )
    return replace(reference, scaling=scaling)


__all__ = [
    "unit_report",
    "BaseLoader",
    "LoadResult",
    "LoaderRegistry",
    "MuonDataset",
    "NexusLoader",
    "PeriodMode",
    "PsiLoader",
    "RootLoader",
    "load",
    "resolve_run_range",
    "scan_run_files",
    "period_count",
    "period_labels",
    "select_period",
    "load_background_run",
    "resolve_background_reference",
    "background_reference_for_grouping",
    "BackgroundReference",
]
