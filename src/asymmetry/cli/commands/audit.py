"""``asymmetry audit`` — which numbers in a draft summary no command printed."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from asymmetry.cli._numbers import unstated, unsupported_laws, unverified_numbers
from asymmetry.cli._output import UserError, emit_json, payload
from asymmetry.cli._runs import range_text, run_spec
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
    relations = unstated(text, log_text)
    unpaired = _correlations_without_lines(roots)
    untested = _untested_doublets(roots)
    untrended = _untrended_series(roots, log_text)
    dips = _unfitted_dips(roots)
    unfitted, unfitted_notes = unfitted_scans(roots)

    if args.json:
        emit_json(
            payload(
                logs=[str(log) for log in logs],
                unfitted_scans=[
                    entry.scan.to_dict() | {"unfitted_runs": entry.runs} for entry in unfitted
                ],
                unfitted_notes_scans=[
                    scan | {"unfitted_runs": runs} for _, scan, runs in unfitted_notes
                ],
                unsupported_laws=[{"law": law, "phrase": phrase} for law, phrase in laws],
                unstated=relations,
                unverified=[
                    {"text": entry.text, "line_number": entry.line_number, "line": entry.line}
                    for entry in found
                ],
            )
        )
        return
    if unfitted:
        print(_unfitted_report(unfitted))
    if unfitted_notes:
        print(
            "Scans the run notes define (survey NOTES SCANS) with runs no fit holds — fit each "
            "against its own quantity with the fit-series command the survey printed:"
        )
        for folder, scan, runs in unfitted_notes:
            print(
                f"  {scan['instrument']} {range_text(scan['runs'])}, {scan['source']} "
                f'"{scan["template"]}" ({scan["quantity"]}): not fitted: {range_text(runs)}'
            )
            values = dict(zip(scan["runs"], scan["values"], strict=True))
            x = " --x " + ",".join(f"{run}={values[run]:g}" for run in runs)
            print("\n".join(_fit_commands(folder, runs, scan["quantity"], start=runs[0], x=x)))
    for law, phrase in laws:
        print(
            f"The draft says {phrase!r}, but every {law} fit this session printed LAW NOT "
            f"ESTABLISHED: describe that trend in plain words instead."
        )
    for message in relations:
        print(message)
    for root, run in unpaired:
        print(
            f"Run {run} in {root} has a correlation spectrum but no plain transform: its peak "
            f"is the sum of two lines that only the plain FFT shows. Run asymmetry fourier "
            f"<folder> --run {run} --window none --workdir {root} and report those lines."
        )
    for root, run, frequency in untested:
        print(
            f"Run {run} in {root}: fourier found a line at {frequency:.6g} MHz in a field of "
            f"tesla order and printed a two-line test, but no fit here has two lines. Run that "
            f"test (asymmetry fourier <folder> --run {run} --window none --workdir {root} prints "
            f"it) before calling the line single."
        )
    if untrended:
        print(
            "Series no trend command has read: "
            + ", ".join(f"{name} ({root})" for root, name in untrended)
            + ". Run asymmetry trend <folder> --series NAME --workdir ROOT on each: its table "
            "and notes say what the series shows (a shift, a change of shape, a law to fit)."
        )
    for root, name, window in dips:
        print(
            f"integral-scan {name} ({root}) announced another dip near {window['centre']:g} that no "
            f"fit holds: fit it with the same options and --model 'LorentzianLCR + Linear' "
            f"--xmin {window['x_min']:g} --xmax {window['x_max']:g} --workdir {root}, and report "
            f"what it shows."
        )
    # A short scan (a calibration pair, a setpoint's two or three fields) is
    # listed for the summary to account for; only a longer one holds the reply.
    held_scans = [entry for entry in unfitted if len(entry.scan.runs) > _SHORT_SCAN_RUNS]
    held = held_scans or unfitted_notes or laws or relations or unpaired or untested
    held = held or untrended or dips
    if not found and held:
        print(
            f"No unprinted numbers found in {draft}, but do not reply yet: act on each item "
            f"above (a measurement left unfitted or unread is a result missing from the "
            f"summary), add what each shows to {draft}, and run audit again."
        )
        return
    if not found:
        # Agents retype a passed draft and add to it; the text to send is printed whole.
        print(
            f"No unprinted numbers found in {draft}. Now send its text as your whole final "
            f"message: copy everything between the two marker lines below exactly — do not "
            f"retype, shorten or add to it (edit the file and audit again instead). The user "
            f"sees neither this output nor the file, and the reply says nothing about this "
            f"check."
        )
        print(f"{_REPLY_MARK} BEGIN {_REPLY_MARK}\n{text.strip()}\n{_REPLY_MARK} END {_REPLY_MARK}")
        return
    print(
        f"{len(found)} number(s) in {draft} appear in no logged command output — "
        f"arithmetic, a conversion, or a value from memory. Remove each, quote the printed "
        f"value instead, or say the relation in words:"
    )
    for entry in found:
        print(f"  line {entry.line_number}: {entry.text!r} in: {entry.line}")
    if held:
        print("And do not reply yet: act on each item listed above as well, then run audit again.")
    else:
        print(
            f"Then run audit again: reply only once it prints 'No unprinted numbers found in "
            f"{draft}'."
        )


#: The marker framing the passed draft, which the reply copies.
_REPLY_MARK = "-----"

#: A scan of at most this many runs is listed on one shared line: a setpoint's
#: two or three fields, a calibration pair — rarely the experiment's question.
_SHORT_SCAN_RUNS = 3


class _UnfittedScan(NamedTuple):
    """A surveyed scan with runs no fit holds, and its own directory's calibrators and lines."""

    folder: str
    scan: ScanGroup
    runs: list[int]
    calibration: set[int]
    lineless: set[int]


def unfitted_scans(
    roots: list[Path],
) -> tuple[list[_UnfittedScan], list[tuple[str, dict, list[int]]]]:
    """The surveyed scans, and the scans the run notes define, with runs no fit holds."""
    from asymmetry.core.workflow.survey import ScanGroup
    from asymmetry.core.workflow.workdir import WorkDir

    # Run numbers identify runs only within one data folder, so each survey is
    # held against the fits and calibrators of every work directory on its folder.
    workdirs = [workdir for workdir in map(WorkDir, roots) if workdir.survey_path.is_file()]
    surveys = [workdir.read_survey() for workdir in workdirs]
    fitted: dict[str, set[int]] = {}
    calibrators: dict[str, set[int]] = {}
    for workdir, survey in zip(workdirs, surveys, strict=True):
        fitted.setdefault(survey["folder"], set()).update(workdir.fitted_runs())
        calibrators.setdefault(survey["folder"], set()).update(workdir.alpha_calibration_runs())
    surveyed = [
        (survey, fitted[survey["folder"]], calibrators[survey["folder"]])
        for survey in {survey["folder"]: survey for survey in surveys}.values()
    ]
    unfitted = [
        _UnfittedScan(
            survey["folder"],
            scan,
            sorted(set(scan.runs) - fitted),
            calibration,
            {int(run["run_number"]) for run in survey["runs"] if run["precession"] == "none"},
        )
        for survey, fitted, calibration in surveyed
        for scan in (ScanGroup(**entry) for entry in survey["scans"])
        if set(scan.runs) - fitted
    ]
    unfitted_notes = [
        (survey["folder"], scan, sorted(set(scan["runs"]) - fitted))
        for survey, fitted, _ in surveyed
        for scan in survey["notes_scans"]
        if set(scan["runs"]) - fitted
    ]
    return unfitted, unfitted_notes


def still_unfitted(root: Path) -> str | None:
    """One line naming the measurements in *root* no fit holds yet, for a result command's close."""
    unfitted, unfitted_notes = unfitted_scans([root])
    names = [
        f"{entry.scan.instrument} {range_text(entry.runs)} ({entry.scan.axis})"
        for entry in unfitted
        if len(entry.scan.runs) > _SHORT_SCAN_RUNS
    ] + [
        f"{scan['instrument']} {range_text(runs)} ({scan['quantity']})"
        for _, scan, runs in unfitted_notes
    ]
    if not names:
        return None
    return (
        f"Still unfitted: {len(names)} measurement(s) the survey found — {'; '.join(names)}. "
        f"Fit each before you write the summary: the audit holds the reply until they are."
    )


def _unfitted_report(unfitted: list[_UnfittedScan]) -> str:
    """The surveyed scans whose runs no stored fit holds, with what to do about each."""
    from asymmetry.cli.commands.survey import scan_label

    lines = [
        "Scans the survey found with runs that no fit, fit-series, fit-global or integral-scan "
        "fitted. Each scan is a measurement:"
    ]
    long_scans = [entry for entry in unfitted if len(entry.scan.runs) > _SHORT_SCAN_RUNS]
    listed = {(entry.folder, run) for entry in long_scans for run in entry.runs}
    short = sorted(
        {
            run
            for entry in unfitted
            if len(entry.scan.runs) <= _SHORT_SCAN_RUNS
            for run in entry.runs
            if (entry.folder, run) not in listed
        }
    )
    for folder, scan, runs, calibration, lineless in long_scans:
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
            lines.append(
                f"      {verdict}. Alpha was measured on {range_text(calibrators)}, and that "
                f"does not account for the scan: its runs are a temperature scan of the line's "
                f"width and envelope shape, which fit-series weighs run by run. Fit it:"
            )
            lines.extend(_fit_commands(folder, runs, scan.axis, start=calibrators[0]))
        else:
            lines.append(
                f"      {verdict}. Fit it — a scan crossing a transition needs a series on each "
                f"side. A run counts once a fit was tried on it, failed or not: a survey 'none' "
                f"means no Fourier line, not no signal, so fit it before calling it unusable "
                f"and report what the fit shows:"
            )
            if set(runs) <= lineless:
                lines.append(
                    "        The survey finds no line in these runs, so a precession model does not "
                    "describe them: fit their relaxation."
                )
                lines.append(
                    f"        asymmetry recipe {shlex.quote(folder)} --expression 'Exponential + "
                    f"Constant' --run {runs[0]} --name relax-{runs[0]}"
                )
                lines.append(
                    f"        asymmetry fit-series {shlex.quote(folder)} --runs {run_spec(runs)} "
                    f"--recipe relax-{runs[0]} --order {scan.axis} --name relax-{runs[0]}"
                )
            elif scan.axis == "field" and scan.geometry != "TF":
                lines.append(
                    f"        asymmetry integral-scan {shlex.quote(folder)} --runs "
                    f"{run_spec(runs)} --plot  (then --model for the curve it shows)"
                )
            else:
                lines.extend(_fit_commands(folder, runs, scan.axis, start=runs[0]))
    if short:
        lines.append(
            f"  and short scans of 2-{_SHORT_SCAN_RUNS} runs, not fitted: {range_text(short)} "
            f"— fit them where they bear on the question."
        )
    return "\n".join(lines)


def _untested_doublets(roots: list[Path]) -> list[tuple[Path, int, float]]:
    """``(work directory, run, MHz)`` for a tesla-field line no two-line fit there tested.

    A fit tests the pair only when two of its frequencies lie within
    :data:`~asymmetry.cli.commands.fourier.PAIR_SPLIT` of each other; a wizard recipe whose second line is the
    first's harmonic does not.
    """
    from asymmetry.cli.commands.fourier import close_pair, tesla_field_lines

    found = []
    for root in roots:
        fits = [
            json.loads(path.read_text(encoding="utf-8"))["fit"]
            for path in (root / "fits").glob("*.json")
        ]
        fitted = [fit["parameters"] for fit in fits if fit["success"]] + [
            row
            for path in (root / "series").glob("*.json")
            for row in json.loads(path.read_text(encoding="utf-8")).get("trend", {"rows": []})[
                "rows"
            ]
            if "failed" not in row["flags"]
        ]
        if any(close_pair(values) is not None for values in fitted):
            continue
        for path in sorted((root / "spectra").glob("*.json")):
            spectrum = json.loads(path.read_text(encoding="utf-8"))
            found.extend(
                (root, int(spectrum["run"]), peak["frequency_mhz"])
                # Spectra stored before field_gauss was recorded carry no field.
                for peak in tesla_field_lines({"field_gauss": None} | spectrum)[:1]
            )
    return found


def _unfitted_dips(roots: list[Path]) -> list[tuple[Path, str, dict]]:
    """``(work directory, scan, window)`` for each dip a scan's fit announced and no fit holds."""
    from asymmetry.cli.commands.integral_scan import holds_dip

    found = []
    for root in roots:
        payloads = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "scans").glob("*.json"))
        }
        found.extend(
            (root, name, window)
            for name, payload in payloads.items()
            if payload["fit"] is not None
            for window in payload["fit"]["next_dip_windows"]
            if not holds_dip(
                (
                    other["fit"]
                    for other in payloads.values()
                    if other["runs"] == payload["runs"] and other["fit"] is not None
                ),
                window,
            )
        )
    return found


def _untrended_series(roots: list[Path], log_text: str) -> list[tuple[Path, str]]:
    """``(work directory, name)`` for each ``fit-series`` series no logged ``trend`` command read."""
    from asymmetry.core.workflow.workdir import WorkDir

    return [
        (root, name)
        for root in roots
        for name in WorkDir(root).fit_series()
        if re.search(
            rf"^\$ asymmetry trend .*--series {re.escape(name)}(?:\s|$)",
            log_text,
            re.MULTILINE,
        )
        is None
    ]


def _correlations_without_lines(roots: list[Path]) -> list[tuple[Path, int]]:
    """``(work directory, run)`` for each correlation spectrum without a plain one beside it."""
    found = []
    for root in roots:
        axes: dict[str, set[int]] = {"hyperfine_coupling": set(), "frequency": set()}
        for path in sorted((root / "spectra").glob("*.json")):
            spectrum = json.loads(path.read_text(encoding="utf-8"))
            axes[spectrum["axis"]].add(int(spectrum["run"]))
        found.extend((root, run) for run in sorted(axes["hyperfine_coupling"] - axes["frequency"]))
    return found


def _fit_commands(
    folder: str, runs: list[int], order: str, *, start: int, x: str = ""
) -> list[str]:
    """The ``wizard`` and ``fit-series`` commands that fit *runs* as one series."""
    return [
        f"        asymmetry wizard {shlex.quote(folder)} --run {start}",
        f"        asymmetry fit-series {shlex.quote(folder)} --runs {run_spec(runs)} "
        f"--recipe wizard-{start} --order {order}{x} --start {start}",
    ]


__all__ = ["add_parser", "run"]
