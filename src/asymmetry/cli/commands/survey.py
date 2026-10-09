"""``asymmetry survey`` — what is in this folder of runs?"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._output import (
    UserError,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._reduction import add_pair_argument, parse_pair
from asymmetry.cli._runs import range_text, run_clashes, run_spec
from asymmetry.cli._workdir import WORKDIR_NAME, add_workdir_argument, workdir_for

if TYPE_CHECKING:
    from asymmetry.core.workflow.survey import ScanGroup


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``survey`` subcommand."""
    parser = subparsers.add_parser(
        "survey",
        help="List the runs in a folder with their metadata, scans and calibration runs",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    add_pair_argument(parser)
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="write survey.json into")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Survey the folder, store the result in the work directory, and report it."""
    from asymmetry.core.workflow.survey import survey_folder

    folder = Path(args.folder)

    # Resolved before the folder is read: surveying measures precession on
    # every run, and a work directory that belongs to another folder should
    # say so at once rather than after that.
    workdir, selection = workdir_for(folder, args.workdir, args.instrument)

    try:
        survey = survey_folder(folder, pair=parse_pair(args.pair), instrument=selection.instrument)
    except ValueError as exc:
        raise UserError(str(exc)) from None
    # The first command run against a fresh directory is normally this one, so
    # this is where the session is usually claimed for its data folder.
    workdir.write_manifest(selection)
    survey_path = workdir.write_survey(survey.to_dict())

    if args.json:
        emit_json(payload(survey=survey.to_dict(), survey_path=str(survey_path)))
        return

    print(_render(survey, survey_path))


def scan_label(scan: ScanGroup) -> str:
    """One line naming a scan: axis, instrument, geometry, held value, periods, span and end runs."""
    held = f"B = {scan.field:g} G" if scan.axis == "temperature" else f"T = {scan.temperature:g} K"
    geometry = scan.geometry or ("mixed geometry" if scan.geometry_note else "unknown geometry")
    unit = "K" if scan.axis == "temperature" else "G"
    instrument = f"{scan.instrument}, " if scan.instrument else ""
    notes = f', notes "{scan.notes}"' if scan.notes else ""
    notes += (
        f", sample{'s' if len(scan.samples) > 1 else ''} "
        + ", ".join(f'"{sample}"' for sample in scan.samples)
        if scan.samples
        else ""
    )
    if scan.n_periods > 1:
        notes += f", {scan.n_periods} periods" + (" (red/green)" if scan.n_periods == 2 else "")
    # The members in run order: axis order need not be run order, and its two
    # endpoints read as a run range that leaves runs out.
    return (
        f"{scan.axis} scan, {instrument}{geometry}, {held}{notes}: {len(scan.runs)} runs, "
        f"{scan.values[0]:g} to {scan.values[-1]:g} {unit} ({range_text(sorted(scan.runs))})"
    )


def _composition_lines(sets, rows, scan: ScanGroup, folder: str, options: str) -> list[str]:
    """The composition series a scan crossing samples holds, with the commands that fit one."""
    from asymmetry.core.workflow.survey import sample_name

    if not sets:
        return []
    unit = "K" if scan.axis == "temperature" else "G"
    shown = max(sets, key=lambda found: len(found.runs))
    supplied = ",".join(
        f"{run}={'<value>' if value is None else f'{value:g}'}"
        for run, value in zip(shown.runs, shown.values, strict=True)
    )
    name = f"composition-{shown.setpoint:g}"
    others = [found.setpoint for found in sets if found is not shown]
    lines = [
        f"      composition: at {shown.setpoint:g} {unit} these runs differ only in the "
        f"sample, and two or more name its composition — a series along composition, "
        f"whatever the logged temperatures do (state their spread; the setpoint is shared):"
    ]
    lines.extend(
        f"        {run} {sample_name(rows[run])!r}"
        + (f" — notes {rows[run].notes!r}" if rows[run].notes else "")
        for run in shown.runs
    )
    lines.append(
        f"        asymmetry fit-series {folder} --runs {run_spec(shown.runs)} --recipe <recipe> "
        f"--order concentration --x {supplied} --name {name}{options}"
    )
    lines.append(
        f"        then asymmetry trend {folder} --series {name} --model Linear --param <rate>"
        f"{options} — a rate linear in concentration is a rate constant (its slope). Give "
        f"each <value> from the run's title and notes (a pure solvent is 0), and drop a run "
        f"neither gives one for."
    )
    if others:
        lines.append(
            "        The same holds at "
            + ", ".join(f"{setpoint:g}" for setpoint in others)
            + f" {unit}: one series per setpoint gives the rate constant against temperature."
        )
    return lines


#: A survey of more runs than this prints its findings before the run table.
_LONG_SURVEY_RUNS = 100

#: Suffixes on the ``geom`` column naming a geometry the file's stamp did not decide.
_GEOMETRY_MARKS = {"measured": "*", "coils": "+"}


def _departure_blocks(survey) -> list[tuple[str, list[int], float, float]]:
    """Departing runs in consecutive blocks of similar offset: ``(instrument, runs, min, max)``.

    Consecutive in run order within one instrument, with an offset within 0.5 K
    or 20 % of the block's last one — so a cryostat that sat 6 K warm for
    fifteen runs reads as its own block, not as the far end of one range.
    """
    from asymmetry.core.workflow.survey import departs
    from asymmetry.core.workflow.workdir import instrument_name

    blocks: list[tuple[str, list[tuple[int, float]]]] = []
    previous: str | None = None
    for row in sorted(survey.runs, key=lambda row: (instrument_name(row.prefix), row.run_number)):
        if not departs(row.temperature, row.sample_temperature_logged):
            previous = None
            continue
        instrument = instrument_name(row.prefix)
        offset = row.sample_temperature_logged - row.temperature
        if previous == instrument and abs(offset - blocks[-1][1][-1][1]) <= max(
            0.5, 0.2 * abs(blocks[-1][1][-1][1])
        ):
            blocks[-1][1].append((row.run_number, offset))
        else:
            blocks.append((instrument, [(row.run_number, offset)]))
        previous = instrument
    return [
        (
            instrument,
            [run for run, _ in block],
            min(o for _, o in block),
            max(o for _, o in block),
        )
        for instrument, block in blocks
    ]


def _run_label(prefix: str, run_number: int, clashes: dict[int, list[Path]]) -> str:
    """A run as the survey names it: with its instrument where the number is shared."""
    from asymmetry.core.workflow.workdir import instrument_name

    return f"{instrument_name(prefix)} {run_number}" if run_number in clashes else str(run_number)


def _selection_options(survey, instrument: str, clashes: dict[int, list[Path]]) -> str:
    """The ``--instrument``/``--pair`` a command on this folder needs to select what the survey did."""
    options = f" --instrument {instrument}" if clashes else ""
    return options + (f" --pair {'/'.join(survey.pair)}" if survey.pair else "")


def _render(survey, survey_path: Path) -> str:
    """The human-readable survey: the run table, then candidates and scans."""
    from asymmetry.core.io.psi import PSI_HEADER_SAMPLE_SENSOR
    from asymmetry.core.workflow.survey import NOTES_PLACEHOLDER, composition_sets
    from asymmetry.core.workflow.workdir import instrument_name

    clashes = run_clashes([(row.prefix, row.run_number, Path(row.file)) for row in survey.runs])
    headers = [
        "run",
        "T/K",
        "T log/K",
        "B/G",
        "geom",
        "prec",
        "orient",
        "hist",
        "periods",
        "points",
        "events",
        "dt",
        "title",
        "notes",
    ]
    rows = [
        [
            _run_label(row.prefix, row.run_number, clashes),
            format_number(row.temperature, 2),
            format_number(row.sample_temperature_logged, 2),
            format_number(row.field, 2),
            (row.geometry or "-") + _GEOMETRY_MARKS.get(row.geometry_source, ""),
            # An `other` line's frequency is itself evidence: an internal
            # field, a muonium line, or a sub-cycle artefact near 0.1 MHz.
            (row.precession.state or "-")
            + (f"@{row.precession.frequency_mhz:.3g}" if row.precession.state == "other" else ""),
            row.detector_orientation or "-",
            str(row.n_histograms),
            str(row.n_periods),
            str(row.n_points),
            str(row.total_events),
            "yes" if row.has_file_deadtime else "no",
            row.title,
            row.notes or "-",
        ]
        for row in survey.runs
    ]
    instruments = sorted({row.instrument for row in survey.runs if row.instrument})
    header = [
        f"{len(survey.runs)} run(s) in {survey.folder}"
        + (f" — {', '.join(instruments)}" if instruments else ""),
        "",
    ]
    table = [render_table(headers, rows) if rows else "(no run files found)", ""]
    if rows:
        table.append(
            "prec: precession measured"
            + (f" on the {'/'.join(survey.pair)} pair" if survey.pair else "")
            + " against the Larmor frequency of the recorded field "
            "— larmor / other@<MHz> (a different line, at that frequency) / none / - "
            "(not measurable). "
            "geom*: geometry measured from that precession rather than read from the file; "
            "geom+: geometry read from the run's logged field-coil readbacks (axial against "
            "transverse), not its field-state stamp."
        )
        if any(
            row.sample_temperature_log_source == PSI_HEADER_SAMPLE_SENSOR for row in survey.runs
        ):
            table.append(
                "T log (PSI): header sensor 1, an unlabelled sensor inferred to be the sample's "
                "because it tracks the sample on the runs checked; reported only when steady and "
                "within a factor of two of the setpoint."
            )
        table.append("")
    # A long run table would bury the findings under it — and past a few
    # hundred lines, cut them off — so after it come the notes and scans;
    # for a large folder it goes last.
    long = len(rows) > _LONG_SURVEY_RUNS
    lines = header if long else header + table
    if survey.other_pair is not None:
        other = survey.other_pair
        lines.append(
            f"PAIR: no run precesses on the file's own detector pair, but run "
            f"{other.run_number} does on {other.forward}/{other.backward} "
            f"({other.precession.describe()}), the pair across the field. Survey again with "
            f"--pair {other.forward}/{other.backward}, and pass the same --pair to alpha, "
            f"reduce and integral-scan."
        )
        lines.append("")
    if clashes:
        shared = sorted(
            {instrument_name(row.prefix) for row in survey.runs if row.run_number in clashes}
        )
        lines.append(
            f"RUN NUMBERS COLLIDE: {' and '.join(shared)} share {len(clashes)} run number(s) "
            f"in this folder (e.g. {min(clashes)}), shown with their instrument. Every "
            "other command keys its work directory on the run number, so on this folder each "
            f"needs --instrument NAME (the file prefix, in any case: {', '.join(shared)})."
        )
        lines.append("")
    from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz
    from asymmetry.core.workflow.survey import MUONIUM_MHZ_PER_G, muonium_runs

    muonium = muonium_runs(survey.runs)
    if muonium:
        listed = run_spec(sorted(row.run_number for row in muonium), separator=", ")
        lines.append(
            f"MUONIUM: runs {listed} precess near {MUONIUM_MHZ_PER_G:.4g} MHz per gauss of their "
            f"field (the other@ line), muonium's triplet precession in a weak transverse field, "
            f"not the bare muon's {field_gauss_to_frequency_mhz(1.0):.4g} MHz/G: screen them "
            f"with --scope muonium-radical and report the muonium line, its amplitude against "
            f"field, and where it is lost."
        )
        lines.append("")
    if survey.temperature_departures:
        lines.append(
            "TEMPERATURE: the logged sample temperature (T log) and the setpoint (T/K) "
            "disagree on these runs, by the offset shown (T log - T/K). Decide which to trust "
            "for each block and say why: a logged value the apparatus could reach (a cryostat "
            "still cooling or parked elsewhere, a block several kelvin off its neighbours) is a "
            "different measurement — order it with --order sample_temperature_logged and never "
            "quote those runs at their setpoints; only a value no sample here could have had "
            "(hundreds of kelvin off, a liquid logged above its boiling point) points to the "
            "sensor. The scans below are grouped by setpoint:"
        )
        for instrument, runs, lo, hi in _departure_blocks(survey):
            span = f"{lo:+.2f} K" if abs(hi - lo) < 0.005 else f"{lo:+.2f} to {hi:+.2f} K"
            listed = run_spec(runs, separator=", ")
            shown = f"{instrument} {listed}" if clashes else listed
            lines.append(f"  {shown}: {span}")
        lines.append("")
    for repeat in survey.repeats:
        condition = f"{repeat.field:g} G, {repeat.temperature:g} K" + (
            f', notes "{repeat.notes}"' if repeat.notes else ""
        )
        runs = range_text(repeat.runs)
        if clashes:
            runs = f"{repeat.instrument} {runs}"
        if repeat.co_add:
            lines.append(
                f"REPEATS: {runs} repeat one condition ({condition}) — co-add them for "
                f"statistics before a spectrum: asymmetry reduce {shlex.quote(survey.folder)} "
                f"--runs {run_spec(repeat.runs)} --coadd"
                f"{_selection_options(survey, repeat.instrument, clashes)} "
                f"--workdir {WORKDIR_NAME}-coadd"
            )
        else:
            lines.append(
                f"UNRECORDED SCAN: {runs} record one condition ({condition}), but their note "
                f"names a scan: each run is a point in a quantity the files do not record. Do "
                f"not co-add them; find what was stepped in the logbook or the brief."
            )
        lines.append("")
    if survey.truncated:
        lines.append(
            "WARNING: the folder holds more entries than the scan cap; runs may be missing."
        )
        lines.append("")

    if survey.calibration_candidates:
        lines.append(
            "Alpha-calibration candidates (alpha on raw counts: deadtime off, background none; "
            "`reduce --alpha-from` re-measures under its own corrections, and applies that):"
        )
        for candidate in survey.calibration_candidates:
            marker = " (best)" if candidate.best else ""
            # The SNR of a measured candidate is already in its reason.
            lines.append(
                f"  run {_run_label(candidate.prefix, candidate.run_number, clashes)}{marker} "
                f"[{candidate.source}] "
                f"alpha {candidate.alpha:.4f}: {candidate.reason}"
            )
        for step in survey.alpha_steps:
            lines.append(
                f"  ALPHA STEP between runs {step.before_run} and {step.after_run} "
                f"({step.alpha_before:.4f} -> {step.alpha_after:.4f}): no single run calibrates "
                f"this folder; reduce each block of runs with a calibration run from inside it."
            )
    else:
        lines.append(
            "Alpha-calibration candidates: none — alpha cannot be measured from this folder."
        )
    lines.append("")

    if survey.scans:
        lines.append("Scans:")
        for scan in survey.scans:
            lines.append(f"  {scan_label(scan)}")
            if scan.geometry_note:
                lines.append(f"      geometry: {scan.geometry_note}")
            if len(scan.samples) > 1:
                rows = {
                    row.run_number: row for row in survey.runs if row.instrument == scan.instrument
                }
                lines.extend(
                    _composition_lines(
                        composition_sets(scan, rows),
                        rows,
                        scan,
                        shlex.quote(survey.folder),
                        _selection_options(survey, scan.instrument, clashes),
                    )
                )
            if scan.axis == "field" and scan.n_periods == 2:
                lines.append(
                    f"      red/green: a two-period scan is measured in the green - red "
                    f"difference, not the summed periods — asymmetry integral-scan "
                    f"{shlex.quote(survey.folder)} --runs {run_spec(scan.runs)}"
                    f"{_selection_options(survey, scan.instrument, clashes)} "
                    f"--period green-red; with a field step between the periods (differential "
                    f"ALC) fit --model LorentzianLCRPair with its dB held at the value that "
                    f"command's Next line gives (the printed red - green offset, negated)."
                )
        if survey.cross_sections:
            lines.append(
                f"  ({survey.cross_sections} temperature scan(s) through the field scans' points "
                "not listed: each is a cross-section of the field scans above)"
            )
    else:
        lines.append("Scans: none — no two runs share a geometry and a held quantity.")
    lines.append("")

    if survey.notes_scans:
        lines.append(
            f"NOTES SCANS: each set of runs below holds one temperature and field while its "
            f"notes or title step a number the files do not record ({NOTES_PLACEHOLDER} in "
            f"the text). It is a separate measurement of that quantity, not a point of a "
            f"temperature or field scan above: reduce its runs, screen one, and fit the series "
            f"against that quantity, not against temperature."
        )
        for scan in survey.notes_scans:
            runs = range_text(sorted(scan.runs))
            if clashes:
                runs = f"{scan.instrument} {runs}"
            start = min(scan.runs)
            folder = shlex.quote(survey.folder)
            options = f" --instrument {scan.instrument}" if clashes else ""
            supplied = ",".join(
                f"{run}={value:g}" for run, value in sorted(zip(scan.runs, scan.values))
            )
            lines.append(
                f"  {runs} ({scan.temperature:g} K, {scan.field:g} G), {scan.source} "
                f'"{scan.template}": {scan.quantity} from {scan.values[0]:g} to '
                f"{scan.values[-1]:g} on {len(scan.runs)} runs"
            )
            lines.append(f"      asymmetry wizard {folder} --run {start}{options}")
            lines.append(
                f"      asymmetry fit-series {folder} --runs {run_spec(scan.runs)} "
                f"--recipe wizard-{start} --order {scan.quantity} --x {supplied} "
                f"--start {start}{options}"
            )
        lines.append("")
    if long:
        lines.extend(["Runs:", *table])
    lines.append(f"Survey written to {survey_path}")
    return "\n".join(lines)
