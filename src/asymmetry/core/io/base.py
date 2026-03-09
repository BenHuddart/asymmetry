"""Abstract base class and registry for data-file loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset


class BaseLoader(ABC):
    """Interface that every file-format loader must implement."""

    #: File extensions this loader claims (e.g. ``[".wim"]``).
    extensions: list[str] = []

    #: Short human-readable name (e.g. ``"WiMDA (.wim)"``).
    format_name: str = ""

    @abstractmethod
    def load(self, filepath: str) -> MuonDataset:
        """Read *filepath* and return a :class:`MuonDataset`."""
        ...


class LoaderRegistry:
    """Central registry of available file-format loaders."""

    _loaders: dict[str, type[BaseLoader]] = {}

    @classmethod
    def register(cls, loader_cls: type[BaseLoader]) -> None:
        for ext in loader_cls.extensions:
            cls._loaders[ext.lower()] = loader_cls

    @classmethod
    def get_loader(cls, filepath: str, fmt: str | None = None) -> BaseLoader:
        if fmt is not None:
            ext = f".{fmt.lstrip('.')}"
        else:
            ext = Path(filepath).suffix.lower()

        loader_cls = cls._loaders.get(ext)
        if loader_cls is None:
            known = ", ".join(sorted(cls._loaders)) or "(none)"
            raise ValueError(
                f"No loader registered for extension {ext!r}.  "
                f"Known extensions: {known}"
            )
        return loader_cls()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(cls._loaders)
