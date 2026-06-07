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
]
