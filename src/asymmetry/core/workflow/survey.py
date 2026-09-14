"""Survey a folder of run files: one row per run, plus the experiment's shape.

This is the first step of the scriptable session façade (see
``docs/plans/agent-cli-skill.md``). It answers the question an analyst asks
before touching any data: *what is in this directory?* — which instrument,
which runs, at which temperatures and fields, is there an alpha-calibration
run, and do the runs form a temperature scan or a field scan.

Everything here is metadata: the loaders are used for their parsing, never for
their numerics, and no reduction is performed. The result is
JSON-serialisable via :meth:`FolderSurvey.to_dict` so a CLI or an agent can
consume it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from asymmetry.core.data.calibration import (
    best_calibration_run_index,
    classify_tf_calibration_run,
)
from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.component_tags import geometry_from_field_direction

#: Timestamp spellings the loaders hand us: ISO-8601 from the ISIS NeXus
#: headers and the PSI ``dd-MMM-yy HH:MM:SS`` run header form. External file
#: text, so an unparsed value simply yields no duration.
_TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%b-%y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S",
)

#: Decimal places the scan grouping rounds temperature/field to before using
#: them as a group key, so 350 K and 350.0 K land in the same scan.
_SCAN_KEY_DECIMALS = 3


def _parse_timestamp(text: object) -> datetime | None:
    """Parse a run-header timestamp, or ``None`` when the file records none."""
    raw = str(text or "").strip()
    if not raw:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def run_geometry(metadata: dict[str, Any]) -> str | None:
    """The applied-field geometry of a run: ``"ZF"``/``"TF"``/``"LF"`` or ``None``.

    A recorded field magnitude of exactly zero means zero field, whatever the
    file's field-state code says — ISIS stamps ``TF`` on the zero-field runs of
    a scan that began with a weak-TF calibration, and a survey that echoed that
    stamp would mislabel the whole scan. Otherwise the loader's structured
    geometry (``field_direction``, else the raw ``field_state`` token) decides,
    and a run that records neither reports ``None`` rather than a guess.
    """
    field = metadata.get("field")
    if field is not None and float(field) == 0.0:
        return "ZF"
    geometry = geometry_from_field_direction(
        str(metadata.get("field_direction") or metadata.get("field_state") or "")
    )
    return None if geometry is None else geometry.value


def run_facility(metadata: dict[str, Any]) -> str:
    """The facility a run came from.

    PSI and MusrRoot files record it; the ISIS NeXus headers do not, so a
    ``nexus_version`` marker (only the ISIS loaders set it) reads as ISIS.
    """
    facility = str(metadata.get("facility") or "").strip()
    if facility:
        return facility
    return "ISIS" if metadata.get("nexus_version") else ""


def has_file_deadtime(run: Run) -> bool:
    """Whether this run's file carries usable per-detector deadtime values.

    The same source and test the grouping window's ``from_file`` deadtime mode
    uses: a ``dead_time_us`` table (the loaders' key; ``deadtime_loaded_us`` is
    its legacy alias) that covers every detector and is not all zeros. A file
    that records the table but leaves it at zero carries no deadtime.
    """
    grouping = run.grouping
    values = grouping.get("dead_time_us")
    if not isinstance(values, (list, tuple)):
        values = grouping.get("deadtime_loaded_us")
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) < max(1, len(run.histograms)):
        return False
    return any(float(value) != 0.0 for value in values)


@dataclass(frozen=True)
class RunRow:
    """One run's metadata as the survey reports it."""

    run_number: int
    file: str
    prefix: str
    instrument: str
    facility: str
    title: str
    sample: str | None
    temperature: float | None
    field: float | None
    field_direction: str
    geometry: str | None
    n_histograms: int
    n_points: int
    bin_width_us: float
    start_time: str | None
    duration_s: float | None
    has_file_deadtime: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "file": self.file,
            "prefix": self.prefix,
            "instrument": self.instrument,
            "facility": self.facility,
            "title": self.title,
            "sample": self.sample,
            "temperature": self.temperature,
            "field": self.field,
            "field_direction": self.field_direction,
            "geometry": self.geometry,
            "n_histograms": self.n_histograms,
            "n_points": self.n_points,
            "bin_width_us": self.bin_width_us,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "has_file_deadtime": self.has_file_deadtime,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    """A run the weak-TF classifier flagged as a possible alpha calibration."""

    run_number: int
    field_gauss: float | None
    reason: str
    best: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "field_gauss": self.field_gauss,
            "reason": self.reason,
            "best": self.best,
        }


@dataclass(frozen=True)
class ScanGroup:
    """A set of runs that vary one quantity with the other two held fixed.

    ``axis`` is the varying quantity (``"temperature"`` or ``"field"``);
    ``geometry`` plus the *other* quantity (``field`` for a temperature scan,
    ``temperature`` for a field scan) are what the group holds fixed.
    ``runs`` and ``values`` are parallel and ordered along the axis.
    """

    axis: str
    geometry: str | None
    temperature: float | None
    field: float | None
    runs: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "axis": self.axis,
            "geometry": self.geometry,
            "temperature": self.temperature,
            "field": self.field,
            "runs": list(self.runs),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class FolderSurvey:
    """Everything :func:`survey_folder` found in one directory."""

    folder: str
    runs: list[RunRow]
    calibration_candidates: list[CalibrationCandidate]
    best_calibration_run: int | None
    scans: list[ScanGroup]
    #: ``True`` when the directory held more entries than the scan cap, so
    #: ``runs`` may be missing files that exist (see ``scan_run_files``).
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "folder": self.folder,
            "truncated": self.truncated,
            "runs": [row.to_dict() for row in self.runs],
            "calibration_candidates": [c.to_dict() for c in self.calibration_candidates],
            "best_calibration_run": self.best_calibration_run,
            "scans": [scan.to_dict() for scan in self.scans],
        }

    def row(self, run_number: int) -> RunRow:
        """The row for *run_number*, or :class:`KeyError` when it is not here."""
        for row in self.runs:
            if row.run_number == run_number:
                return row
        raise KeyError(f"Run {run_number} is not in the survey of {self.folder}")


def build_run_row(
    dataset: MuonDataset,
    *,
    path: Path,
    prefix: str,
    run_number: int,
) -> RunRow:
    """Describe one loaded dataset as a :class:`RunRow`."""
    run = dataset.run
    metadata = run.metadata
    started = str(metadata.get("started") or "") or None
    start = _parse_timestamp(started)
    stop = _parse_timestamp(metadata.get("stopped"))
    duration = (stop - start).total_seconds() if start is not None and stop is not None else None
    sample = str(metadata.get("sample") or "").strip() or None
    return RunRow(
        run_number=run_number,
        file=path.name,
        prefix=prefix,
        instrument=str(metadata.get("instrument") or ""),
        facility=run_facility(metadata),
        title=str(metadata.get("title") or ""),
        sample=sample,
        temperature=dataset.temperature,
        field=dataset.field,
        field_direction=str(metadata.get("field_direction") or metadata.get("field_state") or ""),
        geometry=run_geometry(metadata),
        n_histograms=len(run.histograms),
        n_points=dataset.n_points,
        bin_width_us=float(run.histograms[0].bin_width),
        start_time=started,
        duration_s=duration,
        has_file_deadtime=has_file_deadtime(run),
    )


def _scan_groups(rows: list[RunRow]) -> list[ScanGroup]:
    """Group *rows* into temperature scans and field scans.

    A temperature scan is every run sharing a (geometry, field) pair, ordered
    by temperature; a field scan every run sharing a (geometry, temperature)
    pair, ordered by field. A group qualifies only when it holds at least two
    runs *at two different axis values*: a single run is not a scan, and
    neither are three zero-field runs all at 350 K, which would otherwise be
    reported as a "field scan" from 0 G to 0 G.
    """
    scans: list[ScanGroup] = []
    for axis, held in (("temperature", "field"), ("field", "temperature")):
        buckets: dict[tuple[str, float], list[RunRow]] = {}
        for row in rows:
            axis_value = getattr(row, axis)
            held_value = getattr(row, held)
            if axis_value is None or held_value is None:
                continue
            key = (row.geometry or "", round(float(held_value), _SCAN_KEY_DECIMALS))
            buckets.setdefault(key, []).append(row)
        for (geometry, held_value), members in buckets.items():
            ordered = sorted(members, key=lambda row: float(getattr(row, axis)))
            axis_values = [float(getattr(row, axis)) for row in ordered]
            if len(ordered) < 2 or len(set(axis_values)) < 2:
                continue
            scans.append(
                ScanGroup(
                    axis=axis,
                    geometry=geometry or None,
                    temperature=held_value if held == "temperature" else None,
                    field=held_value if held == "field" else None,
                    runs=[row.run_number for row in ordered],
                    values=axis_values,
                )
            )
    scans.sort(key=lambda scan: (scan.axis, scan.runs[0]))
    return scans


def survey_folder(folder: str | Path) -> FolderSurvey:
    """Load every run file in *folder* and report what the experiment contains.

    Raises :class:`ValueError` when *folder* is not a directory (from
    :func:`asymmetry.core.io.run_range.scan_run_files`).
    """
    from asymmetry.core.io import load, scan_run_files

    folder = Path(folder)
    found = scan_run_files(folder)

    rows: list[RunRow] = []
    metadatas: list[dict[str, Any] | None] = []
    for prefix, run_number, path in found.entries:
        result = load(str(path))
        # A multi-period file loads as a list; the survey describes its first
        # period, the same one the reduction default selects.
        dataset = result[0] if isinstance(result, list) else result
        rows.append(build_run_row(dataset, path=path, prefix=prefix, run_number=run_number))
        metadatas.append(dataset.run.metadata)

    best_index = best_calibration_run_index(metadatas)
    candidates: list[CalibrationCandidate] = []
    for index, (row, metadata) in enumerate(zip(rows, metadatas)):
        verdict = classify_tf_calibration_run(metadata)
        if not verdict.is_candidate:
            continue
        candidates.append(
            CalibrationCandidate(
                run_number=row.run_number,
                field_gauss=verdict.field_gauss,
                reason=verdict.reason,
                best=index == best_index,
            )
        )

    return FolderSurvey(
        folder=str(folder),
        runs=rows,
        calibration_candidates=candidates,
        best_calibration_run=None if best_index is None else rows[best_index].run_number,
        scans=_scan_groups(rows),
        truncated=found.truncated,
    )


__all__ = [
    "CalibrationCandidate",
    "FolderSurvey",
    "RunRow",
    "ScanGroup",
    "build_run_row",
    "has_file_deadtime",
    "run_facility",
    "run_geometry",
    "survey_folder",
]
