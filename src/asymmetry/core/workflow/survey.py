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
from asymmetry.core.fitting.fit_wizard import (
    MIN_CYCLES_IN_EFFECTIVE_WINDOW,
    fingerprint_spectrum,
)
from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
from asymmetry.core.io.nexus import active_series_mean
from asymmetry.core.workflow.reduction import (
    ReductionSettings,
    estimate_alpha_for_run,
    reduce_run,
)
from asymmetry.core.workflow.workdir import RunSelection, instrument_name

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
ROW_GEOMETRY_SOURCES = ("field", "measured", "coils", "refuted", "file", "none")

#: One field component must exceed the other this many times over before the
#: logged coils name a geometry (see :func:`coil_geometry`).
COIL_DOMINANCE = 10.0

#: HiFi's Z coil carries up to this much field cancelling the stray axial field
#: (7–12 G on the transverse runs checked), so that much of a Z reading is not an
#: applied field.
COIL_Z_COMPENSATION_GAUSS = 12.0


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
        return _spontaneous_precession(fingerprint_spectrum(dataset))
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
    frequency_mhz = float(fingerprint.dominant_fft_frequency_mhz)
    snr = float(fingerprint.dominant_fft_snr)
    matches = abs(frequency_mhz / larmor_mhz - 1.0) <= LARMOR_FREQUENCY_TOLERANCE
    # A dominant "line" that completes under two cycles in the record, away
    # from the Larmor frequency, is the relaxation leaking into the lowest bins,
    # not precession. The damped-line scan — the one that reaches heavily
    # damped lines the Hann-windowed FFT is blind to — speaks instead.
    resolved = fingerprint.dominant_fft_cycles_in_window >= MIN_CYCLES_IN_EFFECTIVE_WINDOW
    if not (fingerprint.oscillatory_hint and snr >= PRECESSION_SNR_FLOOR and (resolved or matches)):
        if not fingerprint.has_damped_line_candidate:
            return PrecessionEvidence(
                state="none",
                frequency_mhz=None,
                snr=snr,
                larmor_mhz=larmor_mhz,
                note="",
            )
        frequency_mhz = float(fingerprint.damped_line_frequency_mhz)
        snr = float(fingerprint.damped_line_snr)
        matches = abs(frequency_mhz / larmor_mhz - 1.0) <= LARMOR_FREQUENCY_TOLERANCE
    return PrecessionEvidence(
        state="larmor" if matches else "other",
        frequency_mhz=frequency_mhz,
        snr=snr,
        larmor_mhz=larmor_mhz,
        note="",
    )


def _spontaneous_precession(fingerprint: Any) -> PrecessionEvidence:
    """A zero-field run's own line: precession in an internal field, or none.

    The same reading as the transverse case (see :func:`precession_evidence`):
    a resolved dominant line, else the damped-line scan's, else nothing. There
    is no Larmor frequency to compare with, so any line is ``"other"`` — a
    spontaneous internal field, the signature of static magnetic order.
    """
    if (
        fingerprint.oscillatory_hint
        and fingerprint.dominant_fft_snr >= PRECESSION_SNR_FLOOR
        and fingerprint.dominant_fft_cycles_in_window >= MIN_CYCLES_IN_EFFECTIVE_WINDOW
    ):
        frequency, snr = fingerprint.dominant_fft_frequency_mhz, fingerprint.dominant_fft_snr
    elif fingerprint.has_damped_line_candidate:
        frequency, snr = fingerprint.damped_line_frequency_mhz, fingerprint.damped_line_snr
    else:
        return PrecessionEvidence(
            state="none",
            frequency_mhz=None,
            snr=None,
            larmor_mhz=0.0,
            note="zero field: no spontaneous line — no static order resolved in this record",
        )
    return PrecessionEvidence(
        state="other",
        frequency_mhz=float(frequency),
        snr=float(snr),
        larmor_mhz=0.0,
        note="zero field: spontaneous precession in an internal field",
    )


def coil_geometry(metadata: dict[str, Any]) -> str | None:
    """``"LF"``/``"TF"`` from the run's own logged field-coil readbacks, else ``None``.

    HiFi logs every coil it drives (``Field_Main``, ``Field_Z`` along the beam;
    ``Field_X``, ``Field_Y`` across it). The axial field is ``Field_Main +
    Field_Z`` and the transverse ``hypot(Field_X, Field_Y)``, each the mean over
    the run (:func:`~asymmetry.core.io.nexus.active_series_mean`). Axial above
    :data:`COIL_DOMINANCE` times the transverse is LF; transverse above that
    many times the axial field left after :data:`COIL_Z_COMPENSATION_GAUSS` is
    TF; anything between, or a file that does not log all four coils, is
    ``None``.
    """
    series = metadata.get("nexus_time_series", {})
    main, z, x, y = (
        active_series_mean(series.get(name))
        for name in ("Field_Main", "Field_Z", "Field_X", "Field_Y")
    )
    if main is None or z is None or x is None or y is None:
        return None
    axial = abs(main + z)
    transverse = float(np.hypot(x, y))
    if axial > COIL_DOMINANCE * transverse:
        return "LF"
    if transverse > COIL_DOMINANCE * max(axial - COIL_Z_COMPENSATION_GAUSS, 0.0):
        return "TF"
    return None


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
    ``"coils"``
        The run's logged field-coil readbacks (:func:`coil_geometry`) name the
        geometry. They are the run's own measurement of the field it applied,
        so they outrank both a silent spectrum (a transverse line can be too
        damped to resolve) and the file's field-state stamp, which HiFi sets to
        ``TF`` on its longitudinal runs; they rank below measured precession.
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
    coils = coil_geometry(metadata)
    if coils is not None:
        return coils, "coils"
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
    #: The setpoint, and the measured sample temperature when the file logs one.
    temperature: float | None
    sample_temperature_logged: float | None
    #: Where the loader read ``sample_temperature_logged`` from (a log path, or a
    #: PSI header sensor), ``None`` when there is no reading.
    sample_temperature_log_source: str | None
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
    #: Gross events in the default (first) period, summed over raw histograms.
    total_events: int
    bin_width_us: float
    start_time: str | None
    duration_s: float | None
    has_file_deadtime: bool
    #: Number of independently selectable acquisition periods in the file.
    n_periods: int = 1

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
            "sample_temperature_logged": self.sample_temperature_logged,
            "sample_temperature_log_source": self.sample_temperature_log_source,
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
            "n_periods": self.n_periods,
            "n_points": self.n_points,
            "total_events": self.total_events,
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
    #: The run file's prefix, which names its instrument where two share run numbers.
    prefix: str
    field_gauss: float | None
    reason: str
    source: str
    snr: float | None
    #: This run's own forward/backward balance
    #: (:func:`~asymmetry.core.workflow.reduction.estimate_alpha_for_run`).
    alpha: float
    best: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "run_number": self.run_number,
            "prefix": self.prefix,
            "field_gauss": self.field_gauss,
            "reason": self.reason,
            "source": self.source,
            "snr": self.snr,
            "alpha": self.alpha,
            "best": self.best,
        }


#: A logged sample temperature this far from its setpoint — in kelvin *and* as a
#: fraction of the setpoint — is a different temperature, not thermometer
#: scatter: a cryostat still cooling, or a sensor offset that moves a transition.
TEMPERATURE_DEPARTURE_K = 0.3
TEMPERATURE_DEPARTURE_FRACTION = 0.01


def departs(setpoint: float | None, logged: float | None) -> bool:
    """Whether a logged sample temperature departs from its setpoint (see above)."""
    if setpoint is None or logged is None:
        return False
    offset = abs(logged - setpoint)
    return offset > TEMPERATURE_DEPARTURE_K and offset > TEMPERATURE_DEPARTURE_FRACTION * abs(
        setpoint
    )


def temperature_departures(rows: list[RunRow]) -> list[int]:
    """Runs whose logged sample temperature departs from the setpoint (see above)."""
    return [
        row.run_number for row in rows if departs(row.temperature, row.sample_temperature_logged)
    ]


#: Relative change in alpha between consecutive calibration candidates that
#: marks a step — a sample change, a moved detector or a second instrument —
#: rather than the scatter of one setup's estimates (a few percent).
ALPHA_STEP_TOLERANCE = 0.10


@dataclass(frozen=True)
class AlphaStep:
    """Alpha changing between two consecutive calibration candidates, in run order."""

    before_run: int
    after_run: int
    alpha_before: float
    alpha_after: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "before_run": self.before_run,
            "after_run": self.after_run,
            "alpha_before": self.alpha_before,
            "alpha_after": self.alpha_after,
        }


def alpha_steps(candidates: list[CalibrationCandidate]) -> list[AlphaStep]:
    """Every place, in run order, where alpha moves by more than :data:`ALPHA_STEP_TOLERANCE`.

    A step means no single calibration run serves the folder: each block of runs
    between steps needs a calibration run from inside it. Run order is taken
    one instrument at a time, so two instruments sharing run numbers are never
    interleaved.
    """
    ordered = sorted(
        candidates,
        key=lambda candidate: (instrument_name(candidate.prefix), candidate.run_number),
    )
    return [
        AlphaStep(
            before_run=first.run_number,
            after_run=second.run_number,
            alpha_before=first.alpha,
            alpha_after=second.alpha,
        )
        for first, second in zip(ordered, ordered[1:])
        if abs(second.alpha / first.alpha - 1.0) > ALPHA_STEP_TOLERANCE
    ]


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
    #: The run note a field scan's members share — part of a field scan's
    #: identity, since one sample at one temperature is often scanned several
    #: ways; empty for a temperature scan, whose identity it is not.
    notes: str = ""

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
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FolderSurvey:
    """Everything :func:`survey_folder` found in one directory."""

    folder: str
    runs: list[RunRow]
    calibration_candidates: list[CalibrationCandidate]
    best_calibration_run: int | None
    #: Where alpha changes between candidates; empty when one alpha serves all.
    alpha_steps: list[AlphaStep]
    #: Runs whose logged sample temperature departs from the setpoint.
    temperature_departures: list[int]
    scans: list[ScanGroup]
    #: ``True`` when the directory held more entries than the scan cap, so
    #: ``runs`` may be missing files that exist (see ``scan_run_files``).
    truncated: bool
    #: The forward/backward groups precession was measured on; ``None`` is
    #: each file's own pair.
    pair: tuple[str, str] | None
    #: Temperature scans left out of ``scans`` as cross-sections of a grid of
    #: longer field scans.
    cross_sections: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return {
            "folder": self.folder,
            "truncated": self.truncated,
            "pair": None if self.pair is None else list(self.pair),
            "cross_sections": self.cross_sections,
            "runs": [row.to_dict() for row in self.runs],
            "calibration_candidates": [c.to_dict() for c in self.calibration_candidates],
            "best_calibration_run": self.best_calibration_run,
            "alpha_steps": [step.to_dict() for step in self.alpha_steps],
            "temperature_departures": list(self.temperature_departures),
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
    n_periods: int = 1,
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
    total_events = round(
        sum(
            float(np.sum(np.asarray(histogram.counts, dtype=float))) for histogram in run.histograms
        )
    )
    return RunRow(
        run_number=run_number,
        file=path.name,
        prefix=prefix,
        instrument=str(metadata.get("instrument") or ""),
        facility=run_facility(metadata),
        title=str(metadata.get("title") or ""),
        sample=sample,
        temperature=dataset.temperature,
        sample_temperature_logged=dataset.sample_temperature_logged,
        sample_temperature_log_source=metadata.get("sample_temperature_log_source"),
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
        n_periods=int(n_periods),
        n_points=dataset.n_points,
        total_events=total_events,
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


def _scan_groups(rows: list[RunRow]) -> tuple[list[ScanGroup], int]:
    """Group *rows* into temperature scans and field scans.

    A temperature scan is every run sharing an (instrument, field) pair, ordered
    by temperature. A field scan is every run sharing an instrument,
    temperature, run note and period count, ordered by field (see
    :func:`_field_scan_members` for how one is cut into repeats and cleared of
    calibration runs).

    A group qualifies only when it holds at least two runs *at two different
    axis values*: a single run is not a scan, and neither are three zero-field
    runs all at 350 K, which would otherwise be reported as a "field scan" from
    0 G to 0 G. A field scan needs two different *non-zero* fields on top of
    that. A temperature scan every run of which sits in a longer field scan is
    a cross-section of a grid of field scans, not a measurement of its own; it
    is left out and counted, and the count returned beside the scans.

    The instrument is in the key because two instruments in one folder are two
    campaigns and must never merge; geometry is **not**, because it is measured
    per run and a scan that resolves only in part is still one scan (see
    :func:`_group_geometry`).
    """
    groups: list[tuple[str, str, float, str, list[RunRow]]] = []
    by_field: dict[tuple, list[RunRow]] = {}
    by_temperature: dict[tuple, list[RunRow]] = {}
    # Stretches of consecutive runs at one temperature, counted per instrument
    # and keyed by file: two instruments in one folder interleave in run order.
    stretch: dict[str, int] = {}
    current: dict[str, float] = {}
    count = 0
    for row in rows:
        if row.temperature is None or row.field is None:
            continue
        temperature = round(float(row.temperature), _SCAN_KEY_DECIMALS)
        if current.get(row.instrument) != temperature:
            count, current[row.instrument] = count + 1, temperature
        stretch[row.file] = count
        field = round(float(row.field), _SCAN_KEY_DECIMALS)
        by_field.setdefault((row.instrument, field), []).append(row)
        key = (row.instrument, temperature, row.notes, row.n_periods)
        by_temperature.setdefault(key, []).append(row)
    for (instrument, field), members in by_field.items():
        groups.append(("temperature", instrument, field, "", members))
    for (instrument, temperature, notes, _periods), members in by_temperature.items():
        for scan in _field_scan_members(members, stretch):
            groups.append(("field", instrument, temperature, notes, scan))

    scans: list[ScanGroup] = []
    for axis, instrument, held_value, notes, members in groups:
        ordered = sorted(members, key=lambda row: float(getattr(row, axis)))
        axis_values = [float(getattr(row, axis)) for row in ordered]
        if len(ordered) < 2 or len(set(axis_values)) < 2:
            continue
        if axis == "field" and len({value for value in axis_values if value != 0.0}) < 2:
            # A zero-field run beside a *single* field run is a run and its
            # reference, not a field scan — a field scan varies the field.
            continue
        geometry, geometry_note = _group_geometry(ordered)
        scans.append(
            ScanGroup(
                axis=axis,
                instrument=instrument,
                geometry=geometry,
                geometry_note=geometry_note,
                temperature=held_value if axis == "field" else None,
                field=held_value if axis == "temperature" else None,
                runs=[row.run_number for row in ordered],
                values=axis_values,
                notes=notes,
            )
        )
    longest_field_scan: dict[tuple[str, int], int] = {}
    for scan in scans:
        if scan.axis == "field":
            for run in scan.runs:
                key = (scan.instrument, run)
                longest_field_scan[key] = max(longest_field_scan.get(key, 0), len(scan.runs))
    cross_sections = [
        scan
        for scan in scans
        if scan.axis == "temperature"
        and all(
            longest_field_scan.get((scan.instrument, run), 0) > len(scan.runs) for run in scan.runs
        )
    ]
    scans = [scan for scan in scans if scan not in cross_sections]
    scans.sort(key=lambda scan: (scan.axis, scan.runs[0]))
    return [_note_shared_unresolved(scan, scans, rows) for scan in scans], len(cross_sections)


def _field_scan_members(members: list[RunRow], stretch: dict[str, int]) -> list[list[RunRow]]:
    """The field scans in *members*, runs sharing an instrument, temperature, note and period count.

    Two cuts. A run at a field its scan already holds, taken after the cryostat
    visited another temperature, starts a repeat of the scan — a field point
    re-measured in the same visit (a return sweep) does not, and neither does a
    new field after a detour (a scan measured alternately at two temperatures).
    And a minority of runs precessing at their Larmor frequency among runs that
    do not are transverse calibrations taken beside a longitudinal scan, and are
    not part of it; where they are the majority the scan is transverse, and runs
    too slow or too fast to show a line belong to it. *stretch* numbers each
    file's stretch of consecutive runs at one temperature.
    """
    scans: list[list[RunRow]] = []
    for row in members:
        field = round(float(row.field), _SCAN_KEY_DECIMALS)
        if scans:
            fields = {round(float(member.field), _SCAN_KEY_DECIMALS) for member in scans[-1]}
            if field not in fields or stretch[row.file] == stretch[scans[-1][-1].file]:
                scans[-1].append(row)
                continue
        scans.append([row])
    result: list[list[RunRow]] = []
    for scan in scans:
        larmor = [row for row in scan if row.precession.state == "larmor"]
        if 2 * len(larmor) < len(scan):
            scan = [row for row in scan if row.precession.state != "larmor"]
        result.append(scan)
    return result


def _note_shared_unresolved(
    scan: ScanGroup, scans: list[ScanGroup], rows: list[RunRow]
) -> ScanGroup:
    """*scan* with the few members that look like another scan's points named.

    A temperature scan resolved as transverse on at least three runs in four,
    whose remaining runs show no line and also sit in a field scan with no
    transverse line of its own, is most likely holding that field scan's
    (longitudinal) points — which a precession model then fails to fit. A grid
    of fields by temperatures, where every run belongs to both kinds of scan,
    does not qualify: its scans are not mostly resolved. Only *scan*'s own
    instrument is consulted, since another may share its run numbers.
    """
    geometry = {row.run_number: row.geometry for row in rows if row.instrument == scan.instrument}
    unresolved = [run for run in scan.runs if geometry[run] is None]
    if scan.axis != "temperature" or not unresolved or 4 * len(unresolved) > len(scan.runs):
        return scan
    longitudinal_like = {
        run
        for other in scans
        if other.instrument == scan.instrument
        and other.axis == "field"
        and all(geometry[member] != "TF" for member in other.runs)
        for run in other.runs
    }
    shared = [run for run in unresolved if run in longitudinal_like]
    if not shared:
        return scan
    runs = ", ".join(str(run) for run in shared)
    return replace(
        scan,
        geometry_note=(
            f"{scan.geometry_note}; run{'s' if len(shared) > 1 else ''} {runs} also "
            f"belong{'' if len(shared) > 1 else 's'} to a field scan with no transverse line "
            f"at this temperature — likely its points, not this scan's"
        ),
    )


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
    rows: list[RunRow], metadatas: list[dict[str, Any] | None], alphas: dict[str, float]
) -> tuple[list[CalibrationCandidate], int | None]:
    """The runs that could calibrate alpha, and the best of them.

    Each run is judged by :func:`calibration_verdict`. The best candidate is the
    strongest measured line; with no measured candidate at all it falls back to
    the metadata classifier's own ranking, which prefers a field near the middle
    of the weak-TF window. *alphas* is keyed by file name, since two instruments
    in one folder can share a run number.
    """
    candidates: list[tuple[RunRow, CalibrationCandidate]] = []
    # Runs the classifier is allowed to speak for, masked to ``None`` elsewhere
    # so ``best_calibration_run_index`` ranks only those.
    unmeasured: list[dict[str, Any] | None] = []
    for row, metadata in zip(rows, metadatas):
        source, reason = calibration_verdict(metadata, row.field, row.precession)
        unmeasured.append(metadata if source == "metadata" else None)
        if source is None:
            continue
        candidates.append(
            (
                row,
                CalibrationCandidate(
                    run_number=row.run_number,
                    prefix=row.prefix,
                    field_gauss=row.field,
                    reason=reason,
                    source=source,
                    snr=row.precession.snr if source == "measured" else None,
                    alpha=alphas[row.file],
                    best=False,
                ),
            )
        )

    measured = [pair for pair in candidates if pair[1].source == "measured"]
    if measured:
        best_row = max(measured, key=lambda pair: pair[1].snr)[0]
    else:
        index = best_calibration_run_index(unmeasured)
        best_row = None if index is None else rows[index]

    return (
        [replace(candidate, best=row is best_row) for row, candidate in candidates],
        None if best_row is None else best_row.run_number,
    )


def _run_holding_subfolders(folder: Path) -> list[tuple[str, int]]:
    """Immediate sub-folders of *folder* that hold run files, with their counts.

    One level only, matching :func:`survey_folder`'s own non-recursive scan: a
    sub-folder's own sub-folders are not inspected.
    """
    from asymmetry.core.io.run_range import scan_run_files

    holders: list[tuple[str, int]] = []
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir():
            continue
        count = len(scan_run_files(entry).entries)
        if count:
            holders.append((entry.name, count))
    return holders


def survey_folder(
    folder: str | Path, *, pair: tuple[str, str] | None = None, instrument: str | None = None
) -> FolderSurvey:
    """Load every run file in *folder* — or *instrument*'s — and report what it contains.

    Every run is also reduced under the default
    :class:`~asymmetry.core.workflow.reduction.ReductionSettings` — on *pair*
    when one is named — so its precession can be measured (see
    :func:`precession_evidence`); the file is loaded once and that one
    :class:`~asymmetry.core.data.dataset.Run` is reduced, never re-read.

    Two instruments in one folder may share run numbers, so everything the
    survey measures per run is keyed by file, and scans never mix instruments.

    Raises :class:`ValueError` when *folder* is not a directory or *instrument*
    names none of its files (from :meth:`RunSelection.scan`), or when it holds
    no run files of its own but its immediate sub-folders do — naming them with
    their run counts, so the caller can point at one instead of at a folder
    that merely contains the experiment.
    """
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import period_count

    folder = Path(folder)
    found = RunSelection(folder, instrument).scan()
    if not found.entries:
        holders = _run_holding_subfolders(folder)
        if holders:
            listing = ", ".join(
                f"{name} ({count} run{'s' if count != 1 else ''})" for name, count in holders
            )
            raise ValueError(
                f"{folder} holds no run files directly; its sub-folders do: {listing}. "
                "Survey one of them instead."
            )
    settings = ReductionSettings(pair=pair)

    rows: list[RunRow] = []
    metadatas: list[dict[str, Any] | None] = []
    alphas: dict[str, float] = {}
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
                n_periods=len(result) if isinstance(result, list) else period_count(result),
            )
        )
        metadatas.append(dataset.run.metadata)
        if calibration_verdict(dataset.run.metadata, dataset.field, precession)[0] is not None:
            alphas[path.name] = estimate_alpha_for_run(dataset.run, settings).alpha

    candidates, best_run = _calibration_candidates(rows, metadatas, alphas)

    scans, cross_sections = _scan_groups(rows)
    return FolderSurvey(
        folder=str(folder),
        runs=rows,
        calibration_candidates=candidates,
        best_calibration_run=best_run,
        alpha_steps=alpha_steps(candidates),
        temperature_departures=temperature_departures(rows),
        scans=scans,
        truncated=found.truncated,
        pair=pair,
        cross_sections=cross_sections,
    )


__all__ = [
    "ALPHA_STEP_TOLERANCE",
    "LARMOR_FREQUENCY_TOLERANCE",
    "PRECESSION_SNR_FLOOR",
    "PRECESSION_STATES",
    "ROW_GEOMETRY_SOURCES",
    "AlphaStep",
    "CalibrationCandidate",
    "FolderSurvey",
    "PrecessionEvidence",
    "RunRow",
    "ScanGroup",
    "TEMPERATURE_DEPARTURE_FRACTION",
    "TEMPERATURE_DEPARTURE_K",
    "alpha_steps",
    "build_run_row",
    "calibration_verdict",
    "coil_geometry",
    "departs",
    "has_file_deadtime",
    "precession_evidence",
    "resolve_row_geometry",
    "run_facility",
    "run_geometry",
    "survey_folder",
    "temperature_departures",
]
