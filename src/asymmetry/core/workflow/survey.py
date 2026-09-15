"""Survey a folder of run files: one row per run, plus the experiment's shape.

This is the first step of the scriptable session façade (see
``docs/plans/agent-cli-skill.md``). It answers the question an analyst asks
before touching any data: *what is in this directory?* — which instrument,
which runs, at which temperatures and fields, is there an alpha-calibration
run, and do the runs form a temperature scan or a field scan.

Most of it is metadata: the loaders are used for their parsing, never for
their numerics. The one exception is :func:`precession_evidence`, which
*measures* whether a run at a recorded non-zero field precesses at that
field's Larmor frequency — because the metadata cannot be trusted to say so.
ISIS EMU files from 2024 record no field state at all, and other ISIS files
stamp ``TF`` on longitudinal decoupling runs, so a folder holding a perfectly
good transverse-field run would otherwise report no calibration candidate and
an agent would be left guessing. The result is JSON-serialisable via
:meth:`FolderSurvey.to_dict` so a CLI or an agent can consume it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.core.data.calibration import (
    WEAK_TF_FIELD_RANGE_GAUSS,
    best_calibration_run_index,
    classify_tf_calibration_run,
)
from asymmetry.core.data.dataset import MuonDataset, Run
from asymmetry.core.fitting.component_tags import geometry_from_field_direction
from asymmetry.core.fitting.fit_wizard import fingerprint_spectrum
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
from asymmetry.core.workflow.reduction import ReductionSettings, reduce_run

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

#: Dominant-FFT SNR a line must clear before the survey will call it precession
#: at all. Measured across the muon-school corpus: genuine weak-transverse-field
#: runs at 20 G and 100 G score 89–418, while the longitudinal decoupling runs
#: ISIS stamps ``Transverse`` at 40–120 G score about 3. Ten sits in the empty
#: gap between the two populations.
PRECESSION_SNR_FLOOR = 10.0

#: How far the dominant line may sit from the Larmor frequency of the recorded
#: field and still be called Larmor precession, as a fraction of that frequency.
#: On the corpus the true transverse-field runs land 3–14 % high (the FFT bin is
#: coarse and the recorded field is nominal), while the runs that must be
#: rejected — ordered magnets precessing in their own internal field — are off by
#: factors of 4 to 40. A quarter separates the two with room on both sides.
LARMOR_FREQUENCY_TOLERANCE = 0.25

#: The three measured outcomes, plus ``None`` for "could not be measured".
PRECESSION_STATES = ("larmor", "other", "none")

#: Where :func:`resolve_row_geometry` got a run's geometry from. (The screening
#: layer has its own, wider list — it can also be told by a person.)
ROW_GEOMETRY_SOURCES = ("field", "measured", "refuted", "file", "none")


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


@dataclass(frozen=True)
class PrecessionEvidence:
    """What the spectrum itself says about precession in the recorded field.

    ``state`` is ``"larmor"`` (a strong line at the Larmor frequency of the
    recorded field — the run really is transverse), ``"other"`` (a strong line
    somewhere else, which is a magnet precessing in its own internal field),
    ``"none"`` (no line worth the name), or ``None`` when no measurement was
    possible — then ``note`` says why.
    """

    state: str | None
    #: Dominant line (MHz) when one was found, else ``None``.
    frequency_mhz: float | None
    #: SNR of the dominant line, or ``None`` when nothing was measured.
    snr: float | None
    #: γ_μ/2π × B for the recorded field, or ``None`` when no field is recorded.
    larmor_mhz: float | None
    #: Why no measurement was made; empty when one was.
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "state": self.state,
            "frequency_mhz": self.frequency_mhz,
            "snr": self.snr,
            "larmor_mhz": self.larmor_mhz,
            "note": self.note,
        }

    def describe(self) -> str:
        """One plain sentence, for a person reading a command's output."""
        if self.state is None:
            return f"at the Larmor frequency: not measured — {self.note}"
        if self.state == "larmor":
            return (
                f"at the Larmor frequency: yes — {self.frequency_mhz:.3f} MHz "
                f"(SNR {self.snr:.0f}) against a Larmor {self.larmor_mhz:.3f} MHz"
            )
        if self.state == "other":
            return (
                f"at the Larmor frequency: no — the strongest line is "
                f"{self.frequency_mhz:.3f} MHz (SNR {self.snr:.0f}), not the "
                f"Larmor {self.larmor_mhz:.3f} MHz"
            )
        return (
            f"at the Larmor frequency: no — no line above SNR "
            f"{PRECESSION_SNR_FLOOR:.0f} (strongest SNR {self.snr:.0f}) against "
            f"a Larmor {self.larmor_mhz:.3f} MHz"
        )


def _nyquist_mhz(dataset: MuonDataset) -> float:
    """The record's Nyquist frequency, from the spacing of its time axis."""
    time = np.asarray(dataset.time, dtype=float)
    return 0.5 / float(time[1] - time[0])


def precession_evidence(dataset: MuonDataset, field: float | None) -> PrecessionEvidence:
    """Does *dataset* precess at the Larmor frequency of its recorded *field*?

    *dataset* is the run reduced to asymmetry (default
    :class:`~asymmetry.core.workflow.reduction.ReductionSettings`); *field* is
    the applied field in Gauss the file recorded for it. The measurement is
    :func:`~asymmetry.core.fitting.fit_wizard.fingerprint_spectrum` — the same
    spectral reading the fit wizard shortlists candidate models from — compared
    against γ_μ/2π × B under :data:`PRECESSION_SNR_FLOOR` and
    :data:`LARMOR_FREQUENCY_TOLERANCE`.

    A run at zero (or unrecorded) field has no Larmor frequency to look for, and
    a run whose Larmor frequency is above the record's Nyquist frequency could
    not show the line even if it were there; both report ``state=None`` with the
    reason in ``note`` rather than a verdict the data cannot support.
    """
    if field is None:
        return PrecessionEvidence(
            state=None,
            frequency_mhz=None,
            snr=None,
            larmor_mhz=None,
            note="the file records no applied field, so there is no Larmor frequency to look for",
        )
    larmor_mhz = field_gauss_to_frequency_mhz(abs(float(field)))
    if larmor_mhz == 0.0:
        return PrecessionEvidence(
            state=None,
            frequency_mhz=None,
            snr=None,
            larmor_mhz=0.0,
            note="the applied field is zero, so there is no Larmor precession to look for",
        )
    nyquist_mhz = _nyquist_mhz(dataset)
    if larmor_mhz > nyquist_mhz:
        return PrecessionEvidence(
            state=None,
            frequency_mhz=None,
            snr=None,
            larmor_mhz=larmor_mhz,
            note=(
                f"the Larmor frequency of {abs(float(field)):g} G is {larmor_mhz:.2f} MHz, "
                f"above this record's Nyquist frequency of {nyquist_mhz:.2f} MHz"
            ),
        )

    fingerprint = fingerprint_spectrum(dataset)
    snr = float(fingerprint.dominant_fft_snr)
    if not (fingerprint.oscillatory_hint and snr >= PRECESSION_SNR_FLOOR):
        return PrecessionEvidence(
            state="none",
            frequency_mhz=None,
            snr=snr,
            larmor_mhz=larmor_mhz,
            note="",
        )
    frequency_mhz = float(fingerprint.dominant_fft_frequency_mhz)
    matches = abs(frequency_mhz / larmor_mhz - 1.0) <= LARMOR_FREQUENCY_TOLERANCE
    return PrecessionEvidence(
        state="larmor" if matches else "other",
        frequency_mhz=frequency_mhz,
        snr=snr,
        larmor_mhz=larmor_mhz,
        note="",
    )


def resolve_row_geometry(
    metadata: dict[str, Any], evidence: PrecessionEvidence
) -> tuple[str | None, str]:
    """A run's ``(geometry, source)`` with the spectrum allowed to have a say.

    In order:

    ``"field"``
        A recorded field of exactly zero is zero field and settles it.
    ``"measured"``
        Precession at the Larmor frequency of the recorded field *is* a
        transverse field, whatever the file says or fails to say — the case
        :func:`run_geometry` cannot reach, because ISIS EMU files record no
        field state at all.
    ``"refuted"``
        The spectrum was read and holds no line at all, so the applied field is
        not precessing the muon and the file's ``TF`` stamp is contradicted.
        Reporting ``None`` here is the honest answer: the run is a longitudinal
        measurement, or a transverse one whose line is not resolvable, and the
        file's claim is not evidence for either. (A line at *some other*
        frequency — ``"other"`` — is an internal field in an ordered or
        broadened state, which is perfectly consistent with a transverse applied
        field, so it refutes nothing and the file's token stands.)
    ``"file"``
        Nothing was measured, or an internal line was; the file's own token
        decides.
    ``"none"``
        Nothing decides.
    """
    field = metadata.get("field")
    if field is not None and float(field) == 0.0:
        return "ZF", "field"
    if evidence.state == "larmor":
        return "TF", "measured"
    if evidence.state == "none":
        return None, "refuted"
    geometry = run_geometry(metadata)
    return (None, "none") if geometry is None else (geometry, "file")


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
    #: One of :data:`ROW_GEOMETRY_SOURCES` — how ``geometry`` was decided.
    geometry_source: str
    #: What the spectrum says about precession in the recorded field.
    precession: PrecessionEvidence
    #: Detector-bank orientation as the file records it. Never a geometry: a
    #: bank orientation says nothing about which way the applied field points
    #: (see ``docs/porting/field-geometry/``). It is reported because it is
    #: often the only clue to the sample environment a metadata-poor file gives.
    detector_orientation: str
    #: The run's free-text note (NeXus ``notes``, which the loaders surface as
    #: ``comment``), where an experimenter records what a run actually was.
    notes: str
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
            "geometry_source": self.geometry_source,
            # Flattened alongside the metadata columns rather than nested: this
            # payload is a table of runs, and an agent reads `precession` in the
            # same glance as `geometry`.
            "precession": self.precession.state,
            "precession_frequency_mhz": self.precession.frequency_mhz,
            "precession_snr": self.precession.snr,
            "precession_larmor_mhz": self.precession.larmor_mhz,
            "precession_note": self.precession.note,
            "detector_orientation": self.detector_orientation,
            "notes": self.notes,
            "n_histograms": self.n_histograms,
            "n_points": self.n_points,
            "bin_width_us": self.bin_width_us,
            "start_time": self.start_time,
            "duration_s": self.duration_s,
            "has_file_deadtime": self.has_file_deadtime,
        }


@dataclass(frozen=True)
class CalibrationCandidate:
    """A run that could serve as the weak-TF alpha calibration.

    ``source`` is ``"measured"`` when the spectrum was seen precessing at the
    Larmor frequency of the recorded field, and ``"metadata"`` when no such
    measurement was possible and the file's own transverse-field evidence
    (:func:`~asymmetry.core.data.calibration.classify_tf_calibration_run`) is
    all there is. ``snr`` is the measured line's SNR, ``None`` for a metadata
    candidate.
    """

    run_number: int
    field_gauss: float | None
    reason: str
    source: str
    snr: float | None
    best: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "field_gauss": self.field_gauss,
            "reason": self.reason,
            "source": self.source,
            "snr": self.snr,
            "best": self.best,
        }


@dataclass(frozen=True)
class ScanGroup:
    """A set of runs on one instrument that vary one quantity, holding another.

    ``axis`` is the varying quantity (``"temperature"`` or ``"field"``);
    ``instrument`` plus the *other* quantity (``field`` for a temperature scan,
    ``temperature`` for a field scan) are what the group holds fixed.
    ``runs`` and ``values`` are parallel and ordered along the axis.

    Geometry is a *property* of the group, not part of its key. A physical scan
    is one scan even when the survey can only resolve the geometry of part of it
    — a magnet's transverse-field scan resolves above its transition and not
    below — so ``geometry`` is the members' single agreed geometry, and ``None``
    with a ``geometry_note`` when they disagree.
    """

    axis: str
    instrument: str
    geometry: str | None
    #: How the members' geometries break down when they do not agree (e.g.
    #: ``"TF measured on 12 of 21 runs; 9 unresolved"``); empty when they do.
    geometry_note: str
    temperature: float | None
    field: float | None
    runs: list[int]
    values: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "axis": self.axis,
            "instrument": self.instrument,
            "geometry": self.geometry,
            "geometry_note": self.geometry_note,
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
    precession: PrecessionEvidence,
) -> RunRow:
    """Describe one loaded dataset as a :class:`RunRow`.

    *precession* comes from :func:`precession_evidence` on the *reduced* record
    — the caller has it, this function does not reduce anything — and decides
    the row's geometry when the file's own metadata cannot.
    """
    run = dataset.run
    metadata = run.metadata
    started = str(metadata.get("started") or "") or None
    start = _parse_timestamp(started)
    stop = _parse_timestamp(metadata.get("stopped"))
    duration = (stop - start).total_seconds() if start is not None and stop is not None else None
    sample = str(metadata.get("sample") or "").strip() or None
    geometry, geometry_source = resolve_row_geometry(metadata, precession)
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
        geometry=geometry,
        geometry_source=geometry_source,
        precession=precession,
        detector_orientation=str(metadata.get("detector_orientation") or "").strip(),
        # The NeXus loaders read the ``notes`` node into ``comment``; PSI and
        # MusrRoot also record a ``comment``. Either spelling is the same field.
        notes=str(metadata.get("notes") or metadata.get("comment") or "").strip(),
        n_histograms=len(run.histograms),
        n_points=dataset.n_points,
        bin_width_us=float(run.histograms[0].bin_width),
        start_time=started,
        duration_s=duration,
        has_file_deadtime=has_file_deadtime(run),
    )


def _group_geometry(members: list[RunRow]) -> tuple[str | None, str]:
    """A scan's ``(geometry, note)``: the members' agreed geometry, or a tally.

    Geometry is not part of a scan's identity, so a group whose members do not
    all resolve the same way is still one scan — it reports no geometry and says
    how the members broke down, which is what an analyst then has to reason
    about (a transverse-field scan through a magnetic transition resolves above
    it and not below).
    """
    geometries = {row.geometry for row in members}
    if len(geometries) == 1:
        return geometries.pop(), ""

    total = len(members)
    parts: list[str] = []
    for geometry in sorted(value for value in geometries if value is not None):
        matching = [row for row in members if row.geometry == geometry]
        measured = " measured" if all(r.geometry_source == "measured" for r in matching) else ""
        parts.append(f"{geometry}{measured} on {len(matching)} of {total} runs")
    unresolved = sum(1 for row in members if row.geometry is None)
    if unresolved:
        parts.append(f"{unresolved} unresolved")
    return None, "; ".join(parts)


def _scan_groups(rows: list[RunRow]) -> list[ScanGroup]:
    """Group *rows* into temperature scans and field scans.

    A temperature scan is every run sharing an (instrument, field) pair, ordered
    by temperature; a field scan every run sharing an (instrument, temperature)
    pair, ordered by field. A group qualifies only when it holds at least two
    runs *at two different axis values*: a single run is not a scan, and
    neither are three zero-field runs all at 350 K, which would otherwise be
    reported as a "field scan" from 0 G to 0 G. A field scan needs two
    different *non-zero* fields on top of that.

    The instrument is in the key because two instruments in one folder are two
    campaigns and must never merge; geometry is **not**, because it is measured
    per run and a scan that resolves only in part is still one scan (see
    :func:`_group_geometry`).
    """
    scans: list[ScanGroup] = []
    for axis, held in (("temperature", "field"), ("field", "temperature")):
        buckets: dict[tuple[str, float], list[RunRow]] = {}
        for row in rows:
            axis_value = getattr(row, axis)
            held_value = getattr(row, held)
            if axis_value is None or held_value is None:
                continue
            key = (row.instrument, round(float(held_value), _SCAN_KEY_DECIMALS))
            buckets.setdefault(key, []).append(row)
        for (instrument, held_value), members in buckets.items():
            ordered = sorted(members, key=lambda row: float(getattr(row, axis)))
            axis_values = [float(getattr(row, axis)) for row in ordered]
            if len(ordered) < 2 or len(set(axis_values)) < 2:
                continue
            if axis == "field" and len({value for value in axis_values if value != 0.0}) < 2:
                # A zero-field run beside a *single* field run is a run and its
                # reference, not a field scan — a field scan varies the field.
                # Geometry used to keep those apart; with it out of the key, a
                # magnet's fine ZF re-scan and its TF scan share temperatures
                # run for run, and every such pair would be reported as a
                # spurious "0 to 100 G field scan".
                continue
            geometry, geometry_note = _group_geometry(ordered)
            scans.append(
                ScanGroup(
                    axis=axis,
                    instrument=instrument,
                    geometry=geometry,
                    geometry_note=geometry_note,
                    temperature=held_value if held == "temperature" else None,
                    field=held_value if held == "field" else None,
                    runs=[row.run_number for row in ordered],
                    values=axis_values,
                )
            )
    scans.sort(key=lambda scan: (scan.axis, scan.runs[0]))
    return scans


def calibration_verdict(
    metadata: dict[str, Any] | None, field: float | None, evidence: PrecessionEvidence
) -> tuple[str | None, str]:
    """``(source, reason)`` for using one run as the alpha calibration.

    Two sources, and the measurement outranks the file. A run seen precessing at
    the Larmor frequency of a recorded field inside
    :data:`~asymmetry.core.data.calibration.WEAK_TF_FIELD_RANGE_GAUSS` is a
    candidate (``"measured"``) however poorly the file describes itself — that
    is the ISIS EMU file recording no field state at all. A run whose precession
    *could* be measured and was not found is not a candidate however confidently
    the file claims ``TF`` — that is the longitudinal decoupling run ISIS
    mislabels. Only where no measurement was possible (no recorded field, or a
    Larmor frequency above the record's Nyquist) does
    :func:`~asymmetry.core.data.calibration.classify_tf_calibration_run` decide
    alone (``"metadata"``). ``source`` is ``None`` when the run will not do, and
    ``reason`` says why either way.
    """
    lo, hi = WEAK_TF_FIELD_RANGE_GAUSS
    if evidence.state is None:
        verdict = classify_tf_calibration_run(metadata)
        return ("metadata" if verdict.is_candidate else None), verdict.reason
    gauss = abs(float(field))
    if evidence.state != "larmor":
        return None, f"no precession at the Larmor frequency of the recorded {gauss:g} G"
    if not (lo <= gauss <= hi):
        return None, (
            f"precession at the Larmor frequency, but the recorded {gauss:g} G is "
            f"outside the weak-TF window [{lo:.0f}, {hi:.0f}] G"
        )
    return "measured", (
        f"precession at the Larmor frequency of the recorded {gauss:g} G (SNR {evidence.snr:.0f})"
    )


def _calibration_candidates(
    rows: list[RunRow], metadatas: list[dict[str, Any] | None]
) -> tuple[list[CalibrationCandidate], int | None]:
    """The runs that could calibrate alpha, and the best of them.

    Each run is judged by :func:`calibration_verdict`. The best candidate is the
    strongest measured line; with no measured candidate at all it falls back to
    the metadata classifier's own ranking, which prefers a field near the middle
    of the weak-TF window.
    """
    candidates: list[CalibrationCandidate] = []
    # Runs the classifier is allowed to speak for, masked to ``None`` elsewhere
    # so ``best_calibration_run_index`` ranks only those.
    unmeasured: list[dict[str, Any] | None] = []
    for row, metadata in zip(rows, metadatas):
        source, reason = calibration_verdict(metadata, row.field, row.precession)
        unmeasured.append(metadata if source == "metadata" else None)
        if source is None:
            continue
        candidates.append(
            CalibrationCandidate(
                run_number=row.run_number,
                field_gauss=row.field,
                reason=reason,
                source=source,
                snr=row.precession.snr if source == "measured" else None,
                best=False,
            )
        )

    measured = [candidate for candidate in candidates if candidate.source == "measured"]
    if measured:
        best_run = max(measured, key=lambda candidate: candidate.snr).run_number
    else:
        index = best_calibration_run_index(unmeasured)
        best_run = None if index is None else rows[index].run_number

    candidates = [
        replace(candidate, best=candidate.run_number == best_run) for candidate in candidates
    ]
    return candidates, best_run


def survey_folder(folder: str | Path) -> FolderSurvey:
    """Load every run file in *folder* and report what the experiment contains.

    Every run is also reduced under the default
    :class:`~asymmetry.core.workflow.reduction.ReductionSettings` so its
    precession can be measured (see :func:`precession_evidence`); the file is
    loaded once and that one :class:`~asymmetry.core.data.dataset.Run` is
    reduced, never re-read.

    Raises :class:`ValueError` when *folder* is not a directory (from
    :func:`asymmetry.core.io.run_range.scan_run_files`).
    """
    from asymmetry.core.io import load, scan_run_files

    folder = Path(folder)
    found = scan_run_files(folder)
    settings = ReductionSettings()

    rows: list[RunRow] = []
    metadatas: list[dict[str, Any] | None] = []
    for prefix, run_number, path in found.entries:
        result = load(str(path))
        # A multi-period file loads as a list; the survey describes its first
        # period, the same one the reduction default selects.
        dataset = result[0] if isinstance(result, list) else result
        precession = precession_evidence(reduce_run(dataset.run, settings), dataset.field)
        rows.append(
            build_run_row(
                dataset,
                path=path,
                prefix=prefix,
                run_number=run_number,
                precession=precession,
            )
        )
        metadatas.append(dataset.run.metadata)

    candidates, best_run = _calibration_candidates(rows, metadatas)

    return FolderSurvey(
        folder=str(folder),
        runs=rows,
        calibration_candidates=candidates,
        best_calibration_run=best_run,
        scans=_scan_groups(rows),
        truncated=found.truncated,
    )


__all__ = [
    "LARMOR_FREQUENCY_TOLERANCE",
    "PRECESSION_SNR_FLOOR",
    "PRECESSION_STATES",
    "ROW_GEOMETRY_SOURCES",
    "CalibrationCandidate",
    "FolderSurvey",
    "PrecessionEvidence",
    "RunRow",
    "ScanGroup",
    "build_run_row",
    "calibration_verdict",
    "has_file_deadtime",
    "precession_evidence",
    "resolve_row_geometry",
    "run_facility",
    "run_geometry",
    "survey_folder",
]
