"""Abstract base class and registry for data-file loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset

LoadResult = MuonDataset | list[MuonDataset]


class BaseLoader(ABC):
    """Interface that every file-format loader must implement."""

    #: File extensions this loader claims (e.g. ``[".nxs"]``).
    extensions: list[str] = []

    #: Short human-readable name (e.g. ``"ISIS NeXus (.nxs/.nexus)"``).
    format_name: str = ""

    @abstractmethod
    def load(self, filepath: str) -> LoadResult:
        """Read *filepath* and return one or more :class:`MuonDataset` objects.

        Most formats return a single dataset. Multi-period NeXus files may
        return multiple datasets (one per period).
        """
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
                f"No loader registered for extension {ext!r}.  Known extensions: {known}"
            )
        return loader_cls()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(cls._loaders)

    @classmethod
    def file_dialog_filter(cls) -> str:
        """Build a Qt file-dialog filter string from registered loaders.

        Returns
        -------
        str
            Filter string suitable for ``QFileDialog``, including an aggregate
            "All Supported Files" entry, one entry per loader format, and a
            trailing "All files (*)" fallback.
        """
        if not cls._loaders:
            return "All files (*)"

        unique: dict[type[BaseLoader], list[str]] = {}
        for ext, loader_cls in cls._loaders.items():
            unique.setdefault(loader_cls, []).append(ext)

        all_exts = sorted({ext for exts in unique.values() for ext in exts})
        all_patterns = " ".join(f"*{ext}" for ext in all_exts)
        parts = [f"All Supported Files ({all_patterns})"]

        for loader_cls, exts in sorted(
            unique.items(), key=lambda item: item[0].format_name.lower()
        ):
            patterns = " ".join(f"*{ext}" for ext in sorted(exts))
            label = loader_cls.format_name or f"Data files ({patterns})"
            if "(" in label and ")" in label:
                parts.append(label)
            else:
                parts.append(f"{label} ({patterns})")

        parts.append("All files (*)")
        return ";;".join(parts)
