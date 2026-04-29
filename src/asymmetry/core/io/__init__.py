"""File I/O loaders (plugin-based).

Use the :func:`load` convenience function to auto-detect the format::

    from asymmetry.core.io import load
    dataset = load("run12345.nxs")
"""

from asymmetry.core.io.base import BaseLoader, LoaderRegistry, LoadResult
from asymmetry.core.io.nexus import NexusLoader
from asymmetry.core.io.psi import PsiLoader
from asymmetry.core.io.root import RootLoader

# Register built-in loaders
LoaderRegistry.register(NexusLoader)
LoaderRegistry.register(PsiLoader)
LoaderRegistry.register(RootLoader)


def load(filepath: str, fmt: str | None = None) -> LoadResult:
    """Load a μSR data file and return a :class:`MuonDataset`.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    fmt : str, optional
        Force a specific format (e.g. ``"nxs"``).  If *None*, the format
        is auto-detected from the file extension.
    """
    loader = LoaderRegistry.get_loader(filepath, fmt=fmt)
    return loader.load(filepath)


__all__ = [
    "BaseLoader",
    "LoadResult",
    "LoaderRegistry",
    "NexusLoader",
    "PsiLoader",
    "RootLoader",
    "load",
]
