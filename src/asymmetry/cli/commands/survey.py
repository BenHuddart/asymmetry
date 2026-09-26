"""``asymmetry survey`` — what is in this folder of runs?"""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import (
    UserError,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._reduction import add_pair_argument, parse_pair
from asymmetry.cli._runs import run_clashes
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


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
    if not folder.is_dir():
        raise UserError(f"{folder} does not exist or is not a directory.")

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


def _run_list(runs: list[int]) -> str:
    """``[1, 2, 3, 7]`` as ``"1-3, 7"``."""
    spans: list[list[int]] = []
    for run in sorted(runs):
        if spans and run == spans[-1][-1] + 1:
            spans[-1].append(run)
        else:
            spans.append([run])
    return ", ".join(f"{span[0]}-{span[-1]}" if len(span) > 1 else str(span[0]) for span in spans)


def _run_label(prefix: str, run_number: int, clashes: dict[int, list[Path]]) -> str:
    """A run as the survey names it: with its instrument where the number is shared."""
    from asymmetry.core.workflow.workdir import instrument_name

    return f"{instrument_name(prefix)} {run_number}" if run_number in clashes else str(run_number)


def _render(survey, survey_path: Path) -> str:
    """The human-readable survey: the run table, then candidates and scans."""
    from asymmetry.core.io.psi import PSI_HEADER_SAMPLE_SENSOR
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
    lines = [
        f"{len(survey.runs)} run(s) in {survey.folder}"
        + (f" — {', '.join(instruments)}" if instruments else ""),
        "",
        render_table(headers, rows) if rows else "(no run files found)",
        "",
    ]
    if rows:
        lines.append(
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
            lines.append(
                "T log (PSI): header sensor 1, an unlabelled sensor inferred to be the sample's "
                "because it tracks the sample on the runs checked; reported only when steady and "
                "within a factor of two of the setpoint."
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
    if survey.temperature_departures:
        lines.append(
            "TEMPERATURE: the logged sample temperature (T log) and the setpoint (T/K) "
            "disagree on these runs, by the offset shown (T log - T/K). Decide which to trust "
            "for each block and say why: a block sitting at a different temperature from its "
            "neighbours (a cryostat still cooling or parked elsewhere) is a different "
            "measurement — order it with --order sample_temperature_logged; a steady offset "
            "a sample could not have had (a liquid logged above its boiling point) points to "
            "the sensor. The scans below are grouped by setpoint:"
        )
        for instrument, runs, lo, hi in _departure_blocks(survey):
            span = f"{lo:+.2f} K" if abs(hi - lo) < 0.005 else f"{lo:+.2f} to {hi:+.2f} K"
            shown = f"{instrument} {_run_list(runs)}" if clashes else _run_list(runs)
            lines.append(f"  {shown}: {span}")
        lines.append("")
    if survey.truncated:
        lines.append(
            "WARNING: the folder holds more entries than the scan cap; runs may be missing."
        )
        lines.append("")

    if survey.calibration_candidates:
        lines.append("Alpha-calibration candidates:")
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
            held = (
                f"B = {scan.field:g} G"
                if scan.axis == "temperature"
                else f"T = {scan.temperature:g} K"
            )
            geometry = scan.geometry or (
                "mixed geometry" if scan.geometry_note else "unknown geometry"
            )
            unit = "K" if scan.axis == "temperature" else "G"
            instrument = f"{scan.instrument}, " if scan.instrument else ""
            # Runs are listed in axis order, which need not be run order, so
            # the endpoints are shown with an arrow rather than as a range.
            notes = f', notes "{scan.notes}"' if scan.notes else ""
            lines.append(
                f"  {scan.axis} scan, {instrument}{geometry}, {held}{notes}: "
                f"{len(scan.runs)} runs, "
                f"{scan.values[0]:g} to {scan.values[-1]:g} {unit} "
                f"(run {scan.runs[0]} -> {scan.runs[-1]})"
            )
            if scan.geometry_note:
                lines.append(f"      geometry: {scan.geometry_note}")
        if survey.cross_sections:
            lines.append(
                f"  ({survey.cross_sections} temperature scan(s) through the field scans' points "
                "not listed: each is a cross-section of the field scans above)"
            )
    else:
        lines.append("Scans: none — no two runs share a geometry and a held quantity.")

    lines.append("")
    lines.append(f"Survey written to {survey_path}")
    return "\n".join(lines)
