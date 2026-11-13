"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset
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
from asymmetry.core.io.run_range import resolve_run_range, scan_run_files
from asymmetry.core.transform.corrections import BackgroundReference
from asymmetry.core.transform.deadtime import file_deadtime_table
from asymmetry.core.transform.grouping import good_frames
from asymmetry.core.transform.reference_scale import ReferenceExposure, ReferenceScaleRequest
from asymmetry.core.transform.reference_timing import ReferenceTiming

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
) -> LoadResult:
    """Load a μSR data file and return a :class:`MuonDataset`.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    fmt : str, optional
        Force a specific format (e.g. ``"nxs"``).  If *None*, the format
        is auto-detected from the file extension.
    period : int or str or PeriodMode, optional
        For period-mode (multi-period) files, select a single period and
        return just that :class:`MuonDataset`. Accepts a 1-based period number
        or, for two-period files, a ``"red"``/``"green"`` label. When *None*
        (default) the historical behaviour is preserved: single/two-period
        files return one :class:`MuonDataset` and 3+ period files a ``list``.
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
        for key, value in (("checkpoint", checkpoint), ("progress", progress))
        if value is not None
    }
    result = loader.load(filepath, **callbacks)
    if period is None:
        return result
    return select_period(result, period)


def load_background_run(payload: dict) -> MuonDataset:
    """Load exactly the declared reference acquisition and acquired period."""
    from asymmetry.core.io.reference_selection import reference_selection, resolve_loaded_reference

    if not isinstance(payload, dict) or not payload.get("source_file"):
        raise ValueError("background_run payload has no source_file")
    result = load(str(payload["source_file"]))
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
            reference_run = load_background_run(payload).run
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

    from asymmetry.core.project.reference_restore import complete_reference_restore

    complete_reference_restore(payload, reference_run)
    scaling = scale_request.resolve(
        sample_exposure, ReferenceExposure.from_grouping(reference_run.grouping)
    )

    timing_grouping = dict(reference_run.grouping)
    if not live and payload.get("timing") is not None:
        request = payload["timing"]
        if not isinstance(request, dict) or set(request) not in (
            {"t0_policy"},
            {"effective_detector_t0_bins"},
        ):
            raise ValueError("Reference timing request requires one policy or absolute table")
        if "t0_policy" in request and not isinstance(request["t0_policy"], dict):
            raise ValueError("Reference timing requires an explicit policy mode")
        for key in ("t0_policy", "effective_detector_t0_bins"):
            timing_grouping.pop(key, None)
        timing_grouping.update(request)
    timing = ReferenceTiming.from_grouping(
        reference_run.histograms, timing_grouping, reference_run.metadata
    )
    if live or payload.get("timing") is not None:
        payload["timing"] = timing.request()
    return BackgroundReference(
        histograms=list(reference_run.histograms),
        timing=timing,
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
