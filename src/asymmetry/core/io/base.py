"""Abstract base class and registry for data-file loaders."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from asymmetry.core.data.dataset import MuonDataset

LoadResult = MuonDataset | list[MuonDataset]

# Explicit applied-field geometry tags as they appear in PSI free-text fields
# (run comment / setup / title). PSI files carry no structured field-state code
# like ISIS NeXus ``sample/magnetic_field_state``; the experimenter records the
# geometry implicitly, conventionally as a ``TF``/``LF``/``ZF`` tag in the run
# comment — e.g. ``"FeSe 9p4 TF100"`` or ``"Y124 TF150G"`` — which musrfit treats
# as the setup (``SetSetup(GetComment())``). The bare tag may be followed by the
# field magnitude and unit (``TF150G``), so it is accepted when followed by a
# digit or a word boundary; an optional leading ``w`` admits "weak TF" (``wTF``).
# The leading ``\b`` keeps it from firing on substrings of sample names
# ("half" → no ``LF``, "tffactor" → no ``TF``).
_FIELD_DIRECTION_TAGS: dict[str, re.Pattern[str]] = {
    "Transverse": re.compile(r"\b(?:transverse|w?tf(?=\d|\b))", re.IGNORECASE),
    "Longitudinal": re.compile(r"\b(?:longitudinal|w?lf(?=\d|\b))", re.IGNORECASE),
    "Zero field": re.compile(r"\b(?:zero[\s-]?field|zf(?=\d|\b))", re.IGNORECASE),
}


def field_direction_from_text(*texts: object) -> str:
    """Classify the applied-field geometry from PSI free-text fields.

    PSI ``.bin``/``.mdu``/``.root`` files carry no structured field-state code,
    so the geometry is read from free text (run comment, setup, title) and
    **only** from an explicit, unambiguous ``TF``/``LF``/``ZF`` /
    transverse/longitudinal/zero-field token.  This mirrors the policy the NeXus
    loader and the field-geometry study settled on (see
    ``docs/porting/field-geometry/``): never infer geometry from the field
    *magnitude* (a TF run can sit at 0 G) nor from detector/sample *orientation*
    (a build-axis that reads "L" regardless of the applied field).

    Parameters
    ----------
    *texts:
        Free-text fields to scan, in any order (``None``/empty are ignored).

    Returns
    -------
    str
        ``"Transverse"``, ``"Longitudinal"``, ``"Zero field"``, or ``""`` when no
        token is present or two conflicting tokens appear (ambiguous → unknown,
        rather than a misleading guess).
    """
    blob = " ".join(str(t) for t in texts if t)
    matched = {label for label, pattern in _FIELD_DIRECTION_TAGS.items() if pattern.search(blob)}
    return next(iter(matched)) if len(matched) == 1 else ""


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
