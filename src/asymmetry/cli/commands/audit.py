"""``asymmetry audit`` — which numbers in a draft summary no command printed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._numbers import unstated_relations, unsupported_laws, unverified_numbers
from asymmetry.cli._output import UserError, emit_json, payload
from asymmetry.cli._runs import range_text
from asymmetry.cli._workdir import OUTPUT_LOG, WORKDIR_NAME

if TYPE_CHECKING:
    from asymmetry.core.workflow.survey import ScanGroup


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``audit`` subcommand."""
    parser = subparsers.add_parser(
        "audit",
        help="List the numbers in a draft summary that no command's output printed",
    )
    parser.add_argument("draft", help="The draft summary (a text or Markdown file)")
    parser.add_argument(
        "--workdir",
        action="append",
        default=None,
        help=(
            f"Work directory whose {OUTPUT_LOG} to check against (repeatable; default: "
            f"every ./{WORKDIR_NAME}* directory here)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Print each surveyed scan no fit covers, then each unverified number with its line."""
    from asymmetry.core.workflow.survey import ScanGroup
    from asymmetry.core.workflow.workdir import WorkDir

    draft = Path(args.draft)
    if not draft.is_file():
        raise UserError(f"No draft at {draft}.")
    roots = (
        [Path(root) for root in args.workdir]
        if args.workdir
        else sorted(Path.cwd().glob(f"{WORKDIR_NAME}*"))
    )
    logs = [root / OUTPUT_LOG for root in roots if (root / OUTPUT_LOG).is_file()]
    if not logs:
        raise UserError(
            f"No {OUTPUT_LOG} in {', '.join(str(root) for root in roots) or 'this directory'}; "
            f"run the analysis commands from this directory first."
        )
    log_text = "\n".join(log.read_text(encoding="utf-8") for log in logs)
    text = draft.read_text(encoding="utf-8")
    found = unverified_numbers(text, log_text)
    laws = unsupported_laws(text, log_text)
    relations = unstated_relations(text, log_text)
    workdirs = [WorkDir(root) for root in roots]
    fitted = set().union(*(workdir.fitted_runs() for workdir in workdirs))
    calibration = set().union(*(workdir.alpha_calibration_runs() for workdir in workdirs))
    surveys = [workdir.read_survey() for workdir in workdirs if workdir.survey_path.is_file()]
    unfitted = [
        (survey["folder"], scan, sorted(set(scan.runs) - fitted))
        for survey in surveys
        for scan in (ScanGroup(**entry) for entry in survey["scans"])
        if set(scan.runs) - fitted
    ]
    unfitted_notes = [
        (scan, sorted(set(scan["runs"]) - fitted))
        for survey in surveys
        for scan in survey["notes_scans"]
        if set(scan["runs"]) - fitted
    ]

    if args.json:
        emit_json(
            payload(
                logs=[str(log) for log in logs],
                unfitted_scans=[
                    scan.to_dict() | {"unfitted_runs": runs} for _, scan, runs in unfitted
                ],
                unfitted_notes_scans=[
                    scan | {"unfitted_runs": runs} for scan, runs in unfitted_notes
                ],
                unsupported_laws=[{"law": law, "phrase": phrase} for law, phrase in laws],
                unstated_relations=relations,
                unverified=[
                    {"text": entry.text, "line_number": entry.line_number, "line": entry.line}
                    for entry in found
                ],
            )
        )
        return
    if unfitted:
        print(_unfitted_report(unfitted, calibration))
    if unfitted_notes:
        print(
            "Scans the run notes define (survey NOTES SCANS) with runs no fit holds — fit each "
            "against its own quantity with the fit-series command the survey printed:"
        )
        for scan, runs in unfitted_notes:
            print(
                f"  {scan['instrument']} {range_text(scan['runs'])}, {scan['source']} "
                f'"{scan["template"]}" ({scan["quantity"]}): not fitted: {range_text(runs)}'
            )
    for law, phrase in laws:
        print(
            f"The draft says {phrase!r}, but every {law} fit this session printed LAW NOT "
            f"ESTABLISHED: describe that trend in plain words instead."
        )
    for message in relations:
        print(message)
    if not found and not laws and not relations:
        when = "Once every scan above is fitted, send" if unfitted or unfitted_notes else "Now send"
        print(
            f"No unprinted numbers found in {draft}. {when} its text as your whole final "
            f"message, starting at its title — the user sees neither this output nor the "
            f"file, and the reply says nothing about this check."
        )
        return
    if not found:
        return
    print(
        f"{len(found)} number(s) in {draft} appear in no logged command output — "
        f"arithmetic, a conversion, or a value from memory. Remove each, quote the printed "
        f"value instead, or say the relation in words:"
    )
    for entry in found:
        print(f"  line {entry.line_number}: {entry.text!r} in: {entry.line}")


#: A scan of at most this many runs is listed on one shared line: a setpoint's
#: two or three fields, a calibration pair — rarely the experiment's question.
_SHORT_SCAN_RUNS = 3


def _unfitted_report(
    unfitted: list[tuple[str, ScanGroup, list[int]]],
    calibration: set[int],
) -> str:
    """The surveyed scans whose runs no stored fit holds, with what to do about each."""
    from asymmetry.cli.commands.survey import scan_label

    lines = [
        "Scans the survey found with runs that no fit, fit-series, fit-global or integral-scan "
        "fitted. Each scan is a measurement:"
    ]
    long_scans = [entry for entry in unfitted if len(entry[1].runs) > _SHORT_SCAN_RUNS]
    listed = {run for _, _, runs in long_scans for run in runs}
    short = sorted(
        {run for _, scan, runs in unfitted if len(scan.runs) <= _SHORT_SCAN_RUNS for run in runs}
        - listed
    )
    for folder, scan, runs in long_scans:
        lines.append(f"  {scan_label(scan)}")
        calibrators = sorted(calibration & set(runs))
        verdict = (
            "never fitted" if len(runs) == len(scan.runs) else f"not fitted: {range_text(runs)}"
        )
        if len(scan.samples) > 1:
            lines.append(
                f"      {verdict}. It crosses samples ({', '.join(scan.samples)}): fit the runs "
                f"of each sample that form a measurement, or say in the summary why not."
            )
        elif calibrators and scan.axis == "temperature":
            start = calibrators[0]
            lines.append(
                f"      {verdict}. Alpha was measured on {range_text(calibrators)}, and that "
                f"does not account for the scan: its runs are a temperature scan of the line's "
                f"width and envelope shape, which fit-series weighs run by run. Fit it:"
            )
            lines.append(f"        asymmetry wizard {folder} --run {start}")
            lines.append(
                f"        asymmetry fit-series {folder} --runs {','.join(map(str, runs))} "
                f"--recipe wizard-{start} --order temperature --start {start}"
            )
        else:
            lines.append(
                f"      {verdict}. Fit it — a scan crossing a transition needs a series on each "
                f"side. A run counts once a fit was tried on it, failed or not: a survey 'none' "
                f"means no Fourier line, not no signal, so fit it before calling it unusable "
                f"and report what the fit shows."
            )
    if short:
        lines.append(
            f"  and short scans of 2-{_SHORT_SCAN_RUNS} runs, not fitted: {range_text(short)} "
            f"— fit them where they bear on the question."
        )
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
