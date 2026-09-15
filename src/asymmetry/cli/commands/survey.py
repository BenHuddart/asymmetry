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


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``survey`` subcommand."""
    parser = subparsers.add_parser(
        "survey",
        help="List the runs in a folder with their metadata, scans and calibration runs",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory to write survey.json into (default: <folder>/.asymmetry)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Survey the folder, store the result in the work directory, and report it."""
    from asymmetry.core.workflow.survey import survey_folder
    from asymmetry.core.workflow.workdir import WorkDir

    folder = Path(args.folder)
    if not folder.is_dir():
        raise UserError(f"{folder} does not exist or is not a directory.")

    survey = survey_folder(folder)
    workdir = WorkDir.for_folder(folder, args.workdir)
    survey_path = workdir.write_survey(survey.to_dict())

    if args.json:
        emit_json(payload(survey=survey.to_dict(), survey_path=str(survey_path)))
        return

    print(_render(survey, survey_path))


def _render(survey, survey_path: Path) -> str:
    """The human-readable survey: the run table, then candidates and scans."""
    headers = [
        "run",
        "T/K",
        "B/G",
        "geom",
        "prec",
        "orient",
        "hist",
        "points",
        "dt",
        "title",
        "notes",
    ]
    rows = [
        [
            str(row.run_number),
            format_number(row.temperature, 2),
            format_number(row.field, 2),
            # A trailing * marks a geometry the spectrum decided, not the file.
            (row.geometry or "-") + ("*" if row.geometry_source == "measured" else ""),
            row.precession.state or "-",
            row.detector_orientation or "-",
            str(row.n_histograms),
            str(row.n_points),
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
            "prec: precession measured against the Larmor frequency of the recorded field "
            "— larmor / other (a different line) / none / - (not measurable). "
            "geom*: geometry measured from that precession rather than read from the file."
        )
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
                f"  run {candidate.run_number}{marker} [{candidate.source}]: {candidate.reason}"
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
            lines.append(
                f"  {scan.axis} scan, {instrument}{geometry}, {held}: "
                f"{len(scan.runs)} runs, "
                f"{scan.values[0]:g} to {scan.values[-1]:g} {unit} "
                f"(run {scan.runs[0]} -> {scan.runs[-1]})"
            )
            if scan.geometry_note:
                lines.append(f"      geometry: {scan.geometry_note}")
    else:
        lines.append("Scans: none — no two runs share a geometry and a held quantity.")

    lines.append("")
    lines.append(f"Survey written to {survey_path}")
    return "\n".join(lines)
