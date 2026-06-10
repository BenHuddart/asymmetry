"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

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
]
