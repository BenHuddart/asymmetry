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
    """Load the reference run named by a ``background_run`` grouping payload.

    The payload (written by the grouping dialog and persisted with the
    grouping) carries ``source_file`` and ``run_number``. Multi-period files
    resolve to their first period, matching the red-period default of the
    reduction path. Raises ``ValueError`` when the payload is unusable so
    callers can surface the problem instead of silently skipping the
    subtraction.
    """
    if not isinstance(payload, dict):
        raise ValueError("background_run payload must be a dict")
    source_file = str(payload.get("source_file", "") or "")
    if not source_file:
        raise ValueError("background_run payload has no source_file")
    result = load(source_file)
    dataset = result[0] if isinstance(result, list) else result
    if dataset.run is None or not dataset.run.histograms:
        raise ValueError(f"Background run {source_file!r} has no histograms")
    return dataset


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
    """Resolve a ``background_run`` grouping payload to a :class:`BackgroundReference`.

    This is the single home for reference-run resolution so the same logic
    serves the GUI, scripted core reductions, and the grouped Fourier path
    (previously it lived only in ``MainWindow`` and ``reference_run`` silently
    no-op'd everywhere else).

    Resolution order: match the reference run number against the already-loaded
    ``datasets`` (reuse an open run); otherwise load the payload's
    ``source_file`` (cached by requested run, path, and file identity, so a
    batch apply loads the reference **once**, not once per sample dataset). The
    scale is sample/reference good frames, falling back to the payload snapshots
    and finally ``1.0``. With ``allow_file=False``, an unloaded reference returns
    ``None`` without disk I/O, for advisory previews.

    Raises
    ------
    ValueError
        With a human-readable message when the reference cannot be resolved, so
        callers can surface it. (Loader I/O failures propagate as ``OSError``.)
    """
    if not isinstance(payload, dict):
        raise ValueError("no reference is recorded")

    source_file = str(payload.get("source_file", "") or "")
    run_number = payload.get("run_number")
    run_number = None if run_number is None else int(run_number)

    reference_run = None
    # A hit for the same requested run and source skips the registry scan and
    # the loader, so a batch apply sharing one reference resolves it once
    # instead of re-scanning the registry for every sample dataset.
    source_key = (source_file, run_number, reference_file_identity(payload))
    if cache is not None and source_file and source_key in cache:
        reference_run = cache[source_key]

    if reference_run is None and run_number is not None:
        for dataset in datasets:
            run = getattr(dataset, "run", None)
            try:
                matches = run is not None and int(dataset.run_number) == int(run_number)
            except (TypeError, ValueError, AttributeError):
                matches = False
            if matches:
                reference_run = run
                break

    if reference_run is None:
        if not allow_file:
            return None
        if not source_file:
            raise ValueError("the reference is not loaded and no source file is recorded")
        # Cache only loader-loaded references (an in-registry run is reused live
        # each call so it tracks in-memory edits to that run).
        reference_run = load_background_run(payload).run
        if cache is not None and source_file:
            cache[source_key] = reference_run

    if reference_run is None or not reference_run.histograms:
        raise ValueError("the reference has no histograms")

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
        run_number=None if run_number is None else int(run_number),
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
