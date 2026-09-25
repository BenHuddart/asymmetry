"""The work directory: an analysis session held on disk, not in a process.

Every workflow command reads and writes ``./asymmetry-work/`` — a visible
directory in the *project* the analysis is being done in, never in the data
folder, which is routinely a read-only share or archive::

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

**One work directory holds one folder's runs of one instrument.** Everything
under it is keyed on the run number alone, so two folders whose run numbers
overlap — or two instruments sharing run numbers in one folder — would
overwrite each other's spectra, recipes and series in a single directory. The
manifest records the :class:`RunSelection` the session was opened for, and
:meth:`WorkDir.bind` — which every command goes through before it reads or
writes anything — refuses a directory that holds a different one. A directory
with no manifest yet is unclaimed; the first ``survey`` or ``reduce`` writes the
binding.

A reduced entry is keyed on a **digest** of everything that determines its
numbers: the identity of every source file (size, mtime and the SHA-256 of the
whole file — one file, or each member of a co-add), the resolved grouping payload, the reduction settings and the schema the
sidecar was written under. A cached entry whose digest no longer matches is
stale and is recomputed, never trusted; one written under an older schema is
refused until ``reduce`` rewrites it.

Names a caller chooses — a recipe's, a series' — become path components under
this directory, so every path built from one goes through :func:`safe_name`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry import __version__
from asymmetry.core.data.dataset import MuonDataset
from asymmetry.core.io.run_range import ScanRunFilesResult, scan_run_files
from asymmetry.core.workflow.jsonio import write_json as _write_json
from asymmetry.core.workflow.recipe import FitRecipe
from asymmetry.core.workflow.reduction import ReductionSettings

#: Schema version stamped into every file the work directory writes. 2: a
#: reduced sidecar's run record carries ``sample_temperature_logged``, and every
#: stored series its ``trend`` and ``trend_fits``. 3: reduction settings carry
#: the pair, background range and t0/t_good offsets. 4: the manifest records the
#: instrument whose runs the session holds, and a reduced sidecar the runs it
#: co-adds.
SCHEMA = 4

#: Default work-directory name, resolved against the current directory. Not
#: hidden: an analyst who has to find a plot, delete a stale session or put a
#: recipe under version control should see it in a file listing.
WORKDIR_NAME = "asymmetry-work"

#: Chunk the source file is read in while hashing — a buffer size, not a limit
#: on what is hashed (:func:`file_fingerprint` reads to the end).
_FILE_HASH_CHUNK = 1024 * 1024

#: Characters a name may never contain, because each one would make it
#: something other than a single path component under this directory.
_NAME_FORBIDDEN = ("/", "\\", "\0")


def instrument_name(prefix: str) -> str:
    """The instrument a run file's prefix names; ``emu`` and ``EMU`` are one across eras."""
    return prefix.upper()


def describe_instruments(entries: list[tuple[str, int, Path]]) -> str:
    """``"EMU (58 files), MUSR (9 files)"`` for :func:`scan_run_files` *entries*."""
    counts = Counter(instrument_name(prefix) for prefix, _run_number, _path in entries)
    return ", ".join(
        f"{name} ({count} file{'s' if count != 1 else ''})"
        for name, count in sorted(counts.items())
    )


@dataclass(frozen=True)
class RunSelection:
    """The run files a session reads: one folder, and one instrument's files in it.

    ``instrument`` is an :func:`instrument_name`, or ``None`` for every file in
    the folder. A folder holding two instruments whose run numbers collide
    needs one named, because everything stored is keyed on the run number.
    """

    folder: Path
    instrument: str | None

    def matches(self, prefix: str) -> bool:
        """Whether a file with this *prefix* is one of the selected runs."""
        return self.instrument is None or instrument_name(prefix) == self.instrument

    def scan(self) -> ScanRunFilesResult:
        """:func:`scan_run_files` on the folder, keeping the selected files only.

        :class:`ValueError` when the folder is not a directory, and when the
        instrument names none of the files it holds — listing the instruments
        it does hold, since the name is the user's to correct.
        """
        found = scan_run_files(self.folder)
        entries = [entry for entry in found.entries if self.matches(entry[0])]
        if found.entries and not entries:
            raise ValueError(
                f"{self.folder} holds no {self.instrument} run files; it holds "
                f"{describe_instruments(found.entries)}."
            )
        return ScanRunFilesResult(entries=entries, truncated=found.truncated)

    def __str__(self) -> str:
        return str(self.folder) if self.instrument is None else f"{self.folder} ({self.instrument})"


class WorkDirMismatchError(Exception):
    """A work directory was asked to serve runs that are not the ones it holds.

    Carries the directory and both selections so a caller can phrase the
    message in its own vocabulary; :meth:`str` is already a complete sentence
    naming them.
    """

    def __init__(self, *, root: Path, bound: RunSelection, requested: RunSelection) -> None:
        self.root = root
        self.bound = bound
        self.requested = requested
        super().__init__(
            f"{root} belongs to {bound}; for {requested} pass --workdir {WORKDIR_NAME}-<name>"
        )


def safe_name(name: str) -> str:
    """*name*, checked to be usable as one path component; :class:`ValueError` if not.

    A recipe's or a series' name is chosen by whoever ran the command and then
    interpolated into a path under the work directory. This is the one place
    that decides what a name may be: non-empty, no ``/``, ``\\`` or NUL, not
    ``.`` or ``..``, and no leading ``.`` (which would hide the file and, at
    the front of ``..``, walk out of the directory). The message names the
    offending value so a caller can put it in front of the user unchanged.
    """
    if not name:
        raise ValueError("A name must not be empty.")
    for character in _NAME_FORBIDDEN:
        if character in name:
            shown = "NUL" if character == "\0" else repr(character)
            raise ValueError(f"Name {name!r} contains {shown}; it must be a single path component.")
    if name in (".", ".."):
        raise ValueError(f"Name {name!r} is a directory reference, not a name.")
    if name.startswith("."):
        raise ValueError(f"Name {name!r} starts with '.'; that is a hidden path, not a name.")
    return name


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
    """Identity of a source file: size, mtime and the SHA-256 of the whole file.

    The whole file, not a leading slice: a slice leaves a file edited in place
    beyond it — same size, same mtime, different counts — fingerprinting
    identical, and this module's contract is that a stale cache entry is never
    trusted. Reading to the end costs little on the sizes involved: on this
    machine the test fixture's 41 kB run hashes in ~0.4 ms and a 5 MiB file
    (the scale of a real ISIS run) in ~3 ms, against seconds to reduce one.
    """
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_FILE_HASH_CHUNK), b""):
            digest.update(chunk)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def reduction_digest(
    *,
    source_files: Sequence[str | Path],
    grouping: dict[str, Any],
    settings: ReductionSettings,
) -> str:
    """The digest a reduced entry is keyed on: every source file it was reduced from."""
    payload = {
        "schema": SCHEMA,
        "files": [file_fingerprint(path) for path in source_files],
        "grouping": _canonical(grouping),
        "settings": settings.to_dict(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReducedEntry:
    """The JSON sidecar of one reduced run.

    ``members`` names the runs a co-add summed, the first of them
    ``run_number`` itself; it is empty for a run reduced alone.
    """

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
    members: list[int]

    @property
    def source_runs(self) -> list[int]:
        """The runs whose files this entry was reduced from."""
        return self.members or [self.run_number]

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
            "members": list(self.members),
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
            members=[int(run) for run in data["members"]],
        )


class WorkDir:
    """Read/write access to one ``asymmetry-work/`` session directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- layout -------------------------------------------------------------

    @classmethod
    def default(cls, root: str | Path | None = None) -> WorkDir:
        """``<cwd>/asymmetry-work``, or *root* when one was named.

        Resolved against the current directory — the project the analysis is
        being done in — and never against the data folder, which is often a
        read-only share or archive and is not where an analyst keeps work.
        """
        return cls(Path.cwd() / WORKDIR_NAME if root is None else Path(root))

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
    def scans_dir(self) -> Path:
        return self.root / "scans"

    @property
    def spectra_dir(self) -> Path:
        return self.root / "spectra"

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
            self.scans_dir,
            self.spectra_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- binding ------------------------------------------------------------

    @property
    def selection(self) -> RunSelection | None:
        """The runs this session holds, or ``None`` while it is unclaimed."""
        if not self.manifest_path.exists():
            return None
        manifest = self.read_manifest()
        # Schema 3 predates instruments: such a session held every run in its folder.
        instrument = manifest["instrument"] if manifest["schema"] >= 4 else None
        return RunSelection(Path(str(manifest["folder"])).resolve(), instrument)

    def bind(self, folder: str | Path, instrument: str | None) -> RunSelection:
        """The runs this session serves for *folder* and *instrument*; raise if it holds others.

        The single gate every command passes before it touches the directory,
        so no command can mix two data folders, or two instruments' runs, into
        one cache (see the module docstring). Both paths are resolved, so the
        same folder named relatively and absolutely is the same folder. The
        answer is the narrower selection of the two: a session holding every
        run in its folder takes on the instrument a command names, and a
        command naming none reads the instrument the session holds. Raises
        :class:`WorkDirMismatchError` when the manifest names a different
        folder or a different instrument.
        """
        requested = RunSelection(Path(folder).resolve(), instrument)
        bound = self.selection
        if bound is None:
            return requested
        if bound.folder != requested.folder or (
            bound.instrument is not None
            and instrument is not None
            and bound.instrument != instrument
        ):
            raise WorkDirMismatchError(root=self.root, bound=bound, requested=requested)
        return RunSelection(bound.folder, bound.instrument if instrument is None else instrument)

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
        selection: RunSelection,
        *,
        settings: ReductionSettings | None = None,
        runs: list[int] | None = None,
    ) -> Path:
        """Record the session's provenance: version, runs selected, settings, run list.

        This is also where the directory is **bound** to its folder — as an
        absolute, resolved path — and instrument: ``survey`` writes it with
        neither settings nor runs to claim a fresh directory, ``reduce`` writes
        all three, and a later ``survey`` leaves what ``reduce`` recorded in
        place rather than erasing the reduction's provenance.
        """
        self.ensure()
        stored = self.read_manifest() if self.manifest_path.exists() else {}
        _write_json(
            self.manifest_path,
            {
                "schema": SCHEMA,
                "asymmetry_version": __version__,
                "folder": str(selection.folder.resolve()),
                "instrument": selection.instrument,
                "settings": stored.get("settings") if settings is None else settings.to_dict(),
                "runs": stored.get("runs", []) if runs is None else [int(run) for run in runs],
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
        """The sidecar for a reduced run.

        :class:`KeyError` when it is not stored, or was written under an older
        schema whose run record lacks fields later commands read.
        """
        _, json_path = self._reduced_paths(run_number)
        if not json_path.exists():
            raise KeyError(f"Run {run_number} has not been reduced into {self.root}")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA:
            raise KeyError(
                f"Run {run_number} was reduced into {self.root} by an older asymmetry "
                f"(work-directory schema {data['schema']}, now {SCHEMA}); reduce it again."
            )
        return ReducedEntry.from_dict(data)

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
        """Whether the stored reduction of *run_number* was made with *digest*.

        Reads the raw sidecar: the schema is part of the digest, so an entry
        written under an older schema is simply not current.
        """
        _, json_path = self._reduced_paths(run_number)
        if not json_path.exists():
            return False
        return json.loads(json_path.read_text(encoding="utf-8"))["digest"] == digest

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
        """Where the recipe called *name* is stored.

        Raises :class:`ValueError` for a *name* that is not one path component
        (see :func:`safe_name`), so ``write_recipe``/``read_recipe`` cannot be
        talked into a path outside ``recipes/``.
        """
        return self.recipes_dir / f"{safe_name(name)}.json"

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
        """Where the series called *name* is stored.

        Raises :class:`ValueError` for a *name* that is not one path component
        (see :func:`safe_name`).
        """
        return self.series_dir / f"{safe_name(name)}.json"

    def series_plot_dir(self, name: str) -> Path:
        """``plots/<name>/`` — where a series' per-run fit PNGs go.

        Raises :class:`ValueError` for a *name* that is not one path component
        (see :func:`safe_name`); the plot paths are built here rather than in
        the command modules so every one of them is checked.
        """
        return self.plots_dir / safe_name(name)

    def trend_plot_path(self, name: str, param_name: str) -> Path:
        """``plots/<name>-trend-<param>.png`` — a series' trend PNG for one parameter.

        Raises :class:`ValueError` for a *name* that is not one path component
        (see :func:`safe_name`). *param_name* comes from the fitted model, not
        from a path the user typed.
        """
        return self.plots_dir / f"{safe_name(name)}-trend-{param_name}.png"

    def write_series(self, name: str, payload: dict[str, Any]) -> Path:
        """Write ``series/<name>.json`` and return its path."""
        self.ensure()
        path = self.series_path(name)
        _write_json(path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload)
        return path

    def read_series(self, name: str) -> dict[str, Any]:
        """The stored series payload.

        :class:`KeyError` when there is none, or it was written under an older
        schema (before it carried its trend and ``trend_fits``).
        """
        path = self.series_path(name)
        if not path.exists():
            raise KeyError(f"No series {name!r} in {self.series_dir}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA:
            raise KeyError(
                f"Series {name!r} was written by an older asymmetry (work-directory schema "
                f"{data['schema']}, now {SCHEMA}); fit it again."
            )
        return data

    def series_names(self) -> list[str]:
        """The names of every stored series, sorted."""
        return sorted(path.stem for path in self.series_dir.glob("*.json"))

    # -- integral scans and frequency spectra ------------------------------

    def scan_path(self, name: str) -> Path:
        """Where an integral field scan called *name* is stored."""
        return self.scans_dir / f"{safe_name(name)}.json"

    def write_scan(self, name: str, payload: dict[str, Any]) -> Path:
        """Write ``scans/<name>.json`` and return its path."""
        self.ensure()
        path = self.scan_path(name)
        _write_json(path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload)
        return path

    def read_scan(self, name: str) -> dict[str, Any]:
        """Read a stored integral scan; raise :class:`KeyError` when absent."""
        path = self.scan_path(name)
        if not path.exists():
            raise KeyError(f"No scan {name!r} in {self.scans_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def spectrum_paths(self, name: str) -> tuple[Path, Path]:
        """The ``(.npz, .json)`` paths for a named frequency spectrum."""
        stem = safe_name(name)
        return self.spectra_dir / f"{stem}.npz", self.spectra_dir / f"{stem}.json"

    def write_spectrum(
        self,
        name: str,
        *,
        frequency: np.ndarray,
        real: np.ndarray,
        magnitude: np.ndarray,
        payload: dict[str, Any],
    ) -> tuple[Path, Path]:
        """Store a frequency spectrum's arrays and JSON provenance."""
        self.ensure()
        array_path, json_path = self.spectrum_paths(name)
        np.savez(
            array_path,
            frequency=np.asarray(frequency, dtype=np.float64),
            real=np.asarray(real, dtype=np.float64),
            magnitude=np.asarray(magnitude, dtype=np.float64),
        )
        _write_json(json_path, {"schema": SCHEMA, "asymmetry_version": __version__} | payload)
        return array_path, json_path


__all__ = [
    "SCHEMA",
    "WORKDIR_NAME",
    "ReducedEntry",
    "RunSelection",
    "WorkDir",
    "WorkDirMismatchError",
    "describe_instruments",
    "file_fingerprint",
    "instrument_name",
    "reduction_digest",
    "safe_name",
]
