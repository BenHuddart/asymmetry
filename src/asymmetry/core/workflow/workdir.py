"""The work directory: an analysis session held on disk, not in a process.

Every workflow command reads and writes ``<folder>/.asymmetry/``::

    manifest.json          # asymmetry version, folder, settings, run list
    survey.json            # output of `survey`
    reduced/<run>.npz      # time, asymmetry, error
    reduced/<run>.json     # run metadata + settings + digest
    wizard/<run>.json      # screening payload: recommendation, narrative, recipe
    recipes/<name>.json    # a fit recipe (model + parameters + window)
    series/<name>.json     # per-run results, trend table, quality flags
    plots/*.png            # headless PNGs written by --plot

so a later command can pick up a reduced spectrum without reloading and
re-reducing the file, and an agent has state between invocations without a
long-lived process. The JSON written here is the single source of truth for
those later commands.

A reduced entry is keyed on a **digest** of everything that determines its
numbers: the source file's identity (size, mtime and the SHA-256 of its first
mebibyte), the resolved grouping payload, and the reduction settings. A cached
entry whose digest no longer matches is stale and is recomputed, never
trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry import __version__
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.workflow.jsonio import write_json as _write_json
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.reduction import ReductionSettings

#: Schema version stamped into every file the work directory writes.
SCHEMA = 1

#: Default work-directory name inside a data folder.
WORKDIR_NAME = ".asymmetry"

#: Bytes of the source file hashed into its identity. A muon run file's header
#: and the first detectors' counts live here, so a file edited in place is
#: caught without reading a multi-megabyte histogram block.
_FILE_HASH_BYTES = 1024 * 1024


def _canonical(value: Any) -> Any:
    """A JSON-safe, order-stable rendering of an arbitrary grouping value.

    Grouping payloads carry numpy arrays and scalars, tuples, and integer dict
    keys. This maps all of them onto JSON primitives so the digest sees the
    values themselves rather than a memory address or a truncated ``repr``.
    """
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Identity of a source file: size, mtime and a hash of its leading bytes."""
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(_FILE_HASH_BYTES))
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_head": digest.hexdigest(),
    }


def reduction_digest(
    *,
    source_file: str | Path,
    grouping: dict[str, Any],
    settings: ReductionSettings,
) -> str:
    """The digest a reduced entry is keyed on."""
    payload = {
        "file": file_fingerprint(source_file),
        "grouping": _canonical(grouping),
        "settings": settings.to_dict(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReducedEntry:
    """The JSON sidecar of one reduced run."""

    run_number: int
    digest: str
    source_file: str
    n_points: int
    settings: ReductionSettings
    run: dict[str, Any]
    alpha: float
    deadtime_mode: str
    forward_group: int
    backward_group: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict (round-trips via :meth:`from_dict`)."""
        return {
            "schema": SCHEMA,
            "asymmetry_version": __version__,
            "run_number": self.run_number,
            "digest": self.digest,
            "source_file": self.source_file,
            "n_points": self.n_points,
            "settings": self.settings.to_dict(),
            "run": dict(self.run),
            "alpha": self.alpha,
            "deadtime_mode": self.deadtime_mode,
            "forward_group": self.forward_group,
            "backward_group": self.backward_group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReducedEntry:
        """Reconstruct an entry from :meth:`to_dict` output."""
        return cls(
            run_number=int(data["run_number"]),
            digest=str(data["digest"]),
            source_file=str(data["source_file"]),
            n_points=int(data["n_points"]),
            settings=ReductionSettings.from_dict(data["settings"]),
            run=dict(data["run"]),
            alpha=float(data["alpha"]),
            deadtime_mode=str(data["deadtime_mode"]),
            forward_group=int(data["forward_group"]),
            backward_group=int(data["backward_group"]),
        )


class WorkDir:
    """Read/write access to one ``.asymmetry/`` session directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout -------------------------------------------------------------

    @classmethod
    def for_folder(cls, folder: str | Path, root: str | Path | None = None) -> WorkDir:
        """The work directory for a data *folder* (``<folder>/.asymmetry``), or *root*."""
        return cls(Path(folder) / WORKDIR_NAME if root is None else Path(root))

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def survey_path(self) -> Path:
        return self.root / "survey.json"

    @property
    def reduced_dir(self) -> Path:
        return self.root / "reduced"

    @property
    def wizard_dir(self) -> Path:
        return self.root / "wizard"

    @property
    def recipes_dir(self) -> Path:
        return self.root / "recipes"

    @property
    def series_dir(self) -> Path:
        return self.root / "series"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def ensure(self) -> None:
        """Create the directory layout."""
        for directory in (
            self.reduced_dir,
            self.wizard_dir,
            self.recipes_dir,
            self.series_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- survey -------------------------------------------------------------

    def write_survey(self, payload: dict[str, Any]) -> Path:
        """Write ``survey.json`` and return its path."""
        self.ensure()
        _write_json(
            self.survey_path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload
        )
        return self.survey_path

    def read_survey(self) -> dict[str, Any]:
        """The stored survey payload; :class:`FileNotFoundError` when none was written."""
        return json.loads(self.survey_path.read_text(encoding="utf-8"))

    # -- manifest -----------------------------------------------------------

    def write_manifest(
        self,
        *,
        folder: str | Path,
        settings: ReductionSettings,
        runs: list[int],
    ) -> Path:
        """Record the session's provenance: version, folder, settings, run list."""
        self.ensure()
        _write_json(
            self.manifest_path,
            {
                "schema": SCHEMA,
                "asymmetry_version": __version__,
                "folder": str(folder),
                "settings": settings.to_dict(),
                "runs": [int(run) for run in runs],
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )
        return self.manifest_path

    def read_manifest(self) -> dict[str, Any]:
        """The stored manifest; :class:`FileNotFoundError` when none was written."""
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    # -- reduced spectra ----------------------------------------------------

    def _reduced_paths(self, run_number: int) -> tuple[Path, Path]:
        return (
            self.reduced_dir / f"{int(run_number)}.npz",
            self.reduced_dir / f"{int(run_number)}.json",
        )

    def write_reduced(self, dataset: MuonDataset, entry: ReducedEntry) -> Path:
        """Store *dataset*'s arrays plus its *entry* sidecar; return the ``.npz`` path."""
        self.ensure()
        array_path, json_path = self._reduced_paths(entry.run_number)
        np.savez(
            array_path,
            time=np.asarray(dataset.time, dtype=np.float64),
            asymmetry=np.asarray(dataset.asymmetry, dtype=np.float64),
            error=np.asarray(dataset.error, dtype=np.float64),
        )
        _write_json(json_path, entry.to_dict())
        return array_path

    def entry(self, run_number: int) -> ReducedEntry:
        """The sidecar for a reduced run; :class:`KeyError` when it is not stored."""
        _, json_path = self._reduced_paths(run_number)
        if not json_path.exists():
            raise KeyError(f"Run {run_number} has not been reduced into {self.root}")
        return ReducedEntry.from_dict(json.loads(json_path.read_text(encoding="utf-8")))

    def reduced(self, run_number: int) -> MuonDataset:
        """The stored reduced dataset; :class:`KeyError` when it is not stored.

        The returned dataset carries the recorded run metadata but no
        :class:`~asymmetry.core.data.dataset.Run` — the raw counts are not
        cached, so anything that needs histograms must reload the file.
        """
        stored = self.entry(run_number)
        array_path, _ = self._reduced_paths(run_number)
        with np.load(array_path) as arrays:
            return MuonDataset(
                time=arrays["time"],
                asymmetry=arrays["asymmetry"],
                error=arrays["error"],
                metadata=dict(stored.run),
            )

    def is_current(self, run_number: int, digest: str) -> bool:
        """Whether the stored reduction of *run_number* was made with *digest*."""
        _, json_path = self._reduced_paths(run_number)
        if not json_path.exists():
            return False
        return self.entry(run_number).digest == digest

    def reduced_runs(self) -> list[int]:
        """Run numbers with a stored reduction, ascending (empty before any write)."""
        return sorted(int(path.stem) for path in self.reduced_dir.glob("*.json"))

    # -- screening ----------------------------------------------------------

    def wizard_path(self, run_number: int) -> Path:
        """Where the screening payload for *run_number* is stored."""
        return self.wizard_dir / f"{int(run_number)}.json"

    def write_wizard(self, run_number: int, payload: dict[str, Any]) -> Path:
        """Write ``wizard/<run>.json`` and return its path."""
        self.ensure()
        path = self.wizard_path(run_number)
        _write_json(path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload)
        return path

    def read_wizard(self, run_number: int) -> dict[str, Any]:
        """The stored screening payload; :class:`KeyError` when there is none."""
        path = self.wizard_path(run_number)
        if not path.exists():
            raise KeyError(f"Run {run_number} has not been screened into {self.root}")
        return json.loads(path.read_text(encoding="utf-8"))

    def screened_runs(self) -> list[int]:
        """Run numbers with a stored screening, ascending."""
        return sorted(int(path.stem) for path in self.wizard_dir.glob("*.json"))

    # -- recipes ------------------------------------------------------------

    def recipe_path(self, name: str) -> Path:
        """Where the recipe called *name* is stored."""
        return self.recipes_dir / f"{name}.json"

    def write_recipe(self, name: str, recipe: FitRecipe) -> Path:
        """Write ``recipes/<name>.json`` and return its path."""
        self.ensure()
        path = self.recipe_path(name)
        _write_json(path, recipe.to_dict())
        return path

    def read_recipe(self, name: str) -> FitRecipe:
        """The stored recipe; :class:`KeyError` when there is none by that name."""
        path = self.recipe_path(name)
        if not path.exists():
            raise KeyError(f"No recipe {name!r} in {self.recipes_dir}")
        return FitRecipe.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def recipe_names(self) -> list[str]:
        """The names of every stored recipe, sorted."""
        return sorted(path.stem for path in self.recipes_dir.glob("*.json"))

    # -- series -------------------------------------------------------------

    def series_path(self, name: str) -> Path:
        """Where the series called *name* is stored."""
        return self.series_dir / f"{name}.json"

    def write_series(self, name: str, payload: dict[str, Any]) -> Path:
        """Write ``series/<name>.json`` and return its path."""
        self.ensure()
        path = self.series_path(name)
        _write_json(path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload)
        return path

    def read_series(self, name: str) -> dict[str, Any]:
        """The stored series payload; :class:`KeyError` when there is none."""
        path = self.series_path(name)
        if not path.exists():
            raise KeyError(f"No series {name!r} in {self.series_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def series_names(self) -> list[str]:
        """The names of every stored series, sorted."""
        return sorted(path.stem for path in self.series_dir.glob("*.json"))


__all__ = [
    "SCHEMA",
    "WORKDIR_NAME",
    "ReducedEntry",
    "WorkDir",
    "file_fingerprint",
    "reduction_digest",
]
