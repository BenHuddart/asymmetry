"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

from collections.abc import Iterable
from dataclasses import dataclass

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
    grouping = getattr(source, "grouping", None)
    if isinstance(grouping, dict):
        try:
            value = float(grouping.get("good_frames", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    return None


@dataclass
class BackgroundReference:
    """A resolved ``background_run`` reference: source histograms + frame scale.

    ``histograms`` are the reference run's raw (pre-deadtime) histograms — the
    caller applies the *same* deadtime treatment as the sample and groups/aligns
    them before :func:`subtract_scaled_counts` (study divergence D6). ``scale``
    is the good-frame ratio sample/reference (WiMDA's exposure scale).
    """

    histograms: list
    scale: float
    run_number: int | None = None


def resolve_background_reference(
    payload: dict | None,
    *,
    sample_good_frames: float | None = None,
    datasets: Iterable[MuonDataset] = (),
    cache: dict[str, object] | None = None,
) -> BackgroundReference:
    """Resolve a ``background_run`` grouping payload to a :class:`BackgroundReference`.

    This is the single home for reference-run resolution so the same logic
    serves the GUI, scripted core reductions, and the grouped Fourier path
    (previously it lived only in ``MainWindow`` and ``reference_run`` silently
    no-op'd everywhere else).

    Resolution order: match the reference run number against the already-loaded
    ``datasets`` (reuse an open run); otherwise load the payload's
    ``source_file`` (cached per source path in ``cache`` when provided, so a
    batch apply loads the reference **once**, not once per sample dataset). The
    scale is sample/reference good frames, falling back to the payload snapshots
    and finally ``1.0``.

    Raises
    ------
    ValueError
        With a human-readable message when the reference cannot be resolved, so
        callers can surface it. (Loader I/O failures propagate as ``OSError``.)
    """
    if not isinstance(payload, dict):
        raise ValueError("no reference is recorded")

    reference_run = None
    run_number = payload.get("run_number")
    if run_number is not None:
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
        source_file = str(payload.get("source_file", "") or "")
        if not source_file:
            raise ValueError("the reference is not loaded and no source file is recorded")
        if cache is not None and source_file in cache:
            reference_run = cache[source_file]
        else:
            reference_run = load_background_run(payload).run
            if cache is not None:
                cache[source_file] = reference_run

    if reference_run is None or not reference_run.histograms:
        raise ValueError("the reference has no histograms")

    if sample_good_frames is None or float(sample_good_frames) <= 0.0:
        sample_good_frames = payload.get("good_frames_sample")
    reference_frames = _grouping_good_frames(reference_run) or payload.get("good_frames_reference")
    try:
        scale = float(sample_good_frames) / float(reference_frames)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            scale = float(payload.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    return BackgroundReference(
        histograms=list(reference_run.histograms),
        scale=float(scale),
        run_number=None if run_number is None else int(run_number),
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
    "period_count",
    "period_labels",
    "select_period",
    "load_background_run",
    "resolve_background_reference",
    "BackgroundReference",
]
