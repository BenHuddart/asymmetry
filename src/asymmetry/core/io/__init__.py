"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np

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

# Register built-in loaders
LoaderRegistry.register(NexusLoader)
LoaderRegistry.register(PsiLoader)
LoaderRegistry.register(RootLoader)


def load(
    filepath: str,
    fmt: str | None = None,
    period: int | str | PeriodMode | None = None,
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

    Raises
    ------
    ValueError
        If ``period`` is out of range or an unknown label.
    """
    loader = LoaderRegistry.get_loader(filepath, fmt=fmt)
    result = loader.load(filepath)
    if period is None:
        return result
    return select_period(result, period)


def load_background_run(payload: dict) -> MuonDataset:
    """Load exactly the declared reference acquisition and acquired period."""
    from asymmetry.core.io.reference_selection import reference_selection, resolve_loaded_reference

    if not isinstance(payload, dict) or not payload.get("source_file"):
        raise ValueError("background_run payload has no source_file")
    result = load(str(payload["source_file"]))
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


def _grouping_good_frames(source: object) -> float | None:
    """Positive ``good_frames`` from a run/dataset grouping, else ``None``."""
    return good_frames(getattr(source, "grouping", None), default=0.0) or None


def reference_frame_exposure(
    live_reference: float | None,
    sample_good_frames: float | None,
    payload: dict,
) -> tuple[float, float | None, float | None, str]:
    """Sample/reference good-frame ratio, without mixing measurement epochs.

    A *live* ratio is used only when BOTH the sample and reference live frame
    counts are available; otherwise the payload's snapshot ratio (both counts
    recorded together at project-save time), then the explicit payload ``scale``
    snapshot, then ``1.0`` — never a live count divided by a stale snapshot from
    a different exposure, which would silently skew the subtraction.
    """
    live_sample = (
        float(sample_good_frames)
        if sample_good_frames is not None and float(sample_good_frames) > 0.0
        else None
    )
    sample, reference = live_sample, live_reference
    source = "live_frames"
    if live_sample is not None and live_reference:
        scale = live_sample / float(live_reference)
    elif (
        payload.get("good_frames_sample") is not None
        and payload.get("good_frames_reference") is not None
    ):
        try:
            sample = float(payload["good_frames_sample"])
            reference = float(payload["good_frames_reference"])
        except (TypeError, ValueError):
            raise ValueError(
                "reference background frame counts must be finite and positive"
            ) from None
        if not np.isfinite(sample) or not np.isfinite(reference) or sample <= 0 or reference <= 0:
            raise ValueError("reference background frame counts must be finite and positive")
        scale = sample / reference
        source = "saved_frames"
    else:
        sample, reference = None, None
        source = "explicit_scale" if "scale" in payload else "unity_assumption"
        try:
            scale = float(payload.get("scale", 1.0))
        except (TypeError, ValueError):
            raise ValueError("reference background scale must be finite and positive") from None
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference background scale must be finite and positive")
    return scale, sample, reference, source


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
    sample_good_frames: float | None = None,
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
    reference_run = resolve_loaded_reference(
        (ds.run for ds in datasets if ds.run is not None), payload
    )
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
    scale, sample_frames, reference_frames, scale_source = reference_frame_exposure(
        _grouping_good_frames(reference_run), sample_good_frames, payload
    )

    return BackgroundReference(
        histograms=list(reference_run.histograms),
        file_dead_time_us=tuple(file_deadtime_table(reference_run.grouping)),
        scale=float(scale),
        source_file=reference_run.source_file or source_file,
        good_frames_sample=sample_frames,
        good_frames_reference=reference_frames,
        scale_source=scale_source,
        run_number=reference_selection(reference_run)["run_number"],
        period_number=reference_selection(reference_run)["period_number"],
    )


def background_reference_for_grouping(
    reference: BackgroundReference | None, grouping: dict
) -> BackgroundReference | None:
    """Use selected live exposure with a reference snapshot from the same request.

    Saved frame pairs retain their epoch; explicit scales and unity assumptions
    remain authored choices. Only a live-frame snapshot supplies a live denominator.
    """
    if reference is None or reference.scale_source != "live_frames":
        return reference
    scale, sample, frames, source = reference_frame_exposure(
        reference.good_frames_reference,
        good_frames(grouping, default=0.0) or None,
        grouping.get("background_run", {}),
    )
    return replace(
        reference,
        scale=scale,
        good_frames_sample=sample,
        good_frames_reference=frames,
        scale_source=source,
    )


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
