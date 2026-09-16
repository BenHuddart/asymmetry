"""``asymmetry reduce`` — forward/backward asymmetry for a set of runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from asymmetry.cli._output import (
    UserError,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._runs import resolve_run, resolve_runs
from asymmetry.cli._workdir import add_workdir_argument, workdir_for

#: Points averaged to report the initial asymmetry A(0).
_A0_POINTS = 5


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``reduce`` subcommand."""
    parser = subparsers.add_parser(
        "reduce",
        help="Reduce runs to forward/backward asymmetry and cache them in the work directory",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument(
        "--runs",
        required=True,
        help="Run numbers, e.g. '17294-17296,17300'",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha to reduce with")
    parser.add_argument(
        "--alpha-from",
        type=int,
        default=None,
        dest="alpha_from",
        help="Estimate alpha on this run (a weak-TF calibration run) and use it",
    )
    parser.add_argument(
        "--deadtime",
        choices=["off", "from_file"],
        default="off",
        help="Deadtime correction (default: off, matching the GUI's fresh-run default)",
    )
    parser.add_argument("--rebin", type=int, default=1, help="Merge this many bins (default: 1)")
    parser.add_argument(
        "--tmin", type=float, default=None, help="Discard points below this time/µs"
    )
    parser.add_argument(
        "--tmax", type=float, default=None, help="Discard points above this time/µs"
    )
    parser.add_argument(
        "--period",
        default=None,
        metavar="RED|GREEN|N",
        help=(
            "Select one period from a multi-period file. The common two-period "
            "labels are red (period 1) and green (period 2)"
        ),
    )
    parser.add_argument(
        "--plot", action="store_true", help="Write plots/reduced-<run>.png for each run"
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="write into")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Reduce every named run, caching each result in the work directory."""
    from asymmetry.cli import plots
    from asymmetry.core.io import load
    from asymmetry.core.io.periods import period_count, select_period
    from asymmetry.core.workflow.reduction import (
        ReductionSettings,
        estimate_alpha_for_run,
        reduce_run,
        resolve_reduction_grouping,
    )
    from asymmetry.core.workflow.survey import build_run_row, precession_evidence
    from asymmetry.core.workflow.workdir import ReducedEntry, reduction_digest

    folder = Path(args.folder)
    if args.alpha is not None and args.alpha_from is not None:
        raise UserError("Pass either --alpha or --alpha-from, not both.")
    if args.plot:
        plots.require_matplotlib()

    if args.alpha_from is not None:
        alpha_path = resolve_run(folder, args.alpha_from)
        alpha_result = load(str(alpha_path))
        try:
            alpha_dataset = (
                select_period(alpha_result, args.period)
                if args.period is not None
                else (alpha_result[0] if isinstance(alpha_result, list) else alpha_result)
            )
        except (TypeError, ValueError) as exc:
            raise UserError(f"Run {args.alpha_from}: {exc}") from None
        alpha = estimate_alpha_for_run(alpha_dataset.run).alpha
        alpha_source = f"estimated:{args.alpha_from}"
    elif args.alpha is not None:
        alpha, alpha_source = float(args.alpha), "user"
    else:
        alpha, alpha_source = 1.0, "assumed"

    try:
        settings = ReductionSettings(
            alpha=alpha,
            alpha_source=alpha_source,
            deadtime=args.deadtime,
            rebin=args.rebin,
            t_min=args.tmin,
            t_max=args.tmax,
            period=args.period,
        )
    except ValueError as exc:
        # ReductionSettings owns the vocabulary the CLI accepts; a value it
        # rejects is the user's, so it exits 1 with a message, not 2.
        raise UserError(str(exc)) from None

    targets = resolve_runs(folder, args.runs)
    workdir = workdir_for(folder, args.workdir)
    # Written before the first spectrum, not after the last: the manifest is
    # what binds the directory to this data folder, so a reduction interrupted
    # part-way still leaves a directory that says whose runs are in it.
    workdir.write_manifest(
        folder=folder,
        settings=settings,
        runs=[run_number for run_number, _prefix, _path in targets],
    )

    entries: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    for run_number, prefix, path in targets:
        result = load(str(path))
        try:
            dataset_in = (
                select_period(result, args.period)
                if args.period is not None
                else (result[0] if isinstance(result, list) else result)
            )
        except (TypeError, ValueError) as exc:
            raise UserError(f"Run {run_number}: {exc}") from None
        source_run = dataset_in.run
        grouping = resolve_reduction_grouping(source_run, settings)
        digest = reduction_digest(source_file=path, grouping=grouping, settings=settings)

        if workdir.is_current(run_number, digest):
            dataset = workdir.reduced(run_number)
            entry = workdir.entry(run_number)
            recomputed = False
        else:
            dataset = reduce_run(source_run, settings)
            row = build_run_row(
                dataset_in,
                path=path,
                prefix=prefix,
                run_number=run_number,
                # Measured on the record this command actually produced, so a
                # rebinned or trimmed reduction is judged on what it reduced to.
                precession=precession_evidence(dataset, dataset_in.field),
                n_periods=period_count(dataset_in),
            )
            entry = ReducedEntry(
                run_number=run_number,
                digest=digest,
                source_file=str(path),
                n_points=dataset.n_points,
                settings=settings,
                run=row.to_dict(),
                alpha=float(grouping["alpha"]),
                deadtime_mode=str(grouping["deadtime_mode"]),
                forward_group=int(grouping["forward_group"]),
                backward_group=int(grouping["backward_group"]),
            )
            workdir.write_reduced(dataset, entry)
            recomputed = True

        entries.append(_entry_payload(entry, dataset, recomputed=recomputed))

        if args.plot:
            plot_paths.append(
                plots.plot_reduced(
                    dataset.time,
                    dataset.asymmetry,
                    dataset.error,
                    run_number=run_number,
                    temperature=entry.run.get("temperature"),
                    field=entry.run.get("field"),
                    title=entry.run.get("title", ""),
                    out_path=workdir.plots_dir / f"reduced-{run_number}.png",
                )
            )

    if args.json:
        emit_json(
            payload(
                folder=str(folder),
                workdir=str(workdir.root),
                settings=settings.to_dict(),
                entries=entries,
                plots=[str(path) for path in plot_paths],
            )
        )
        return

    print(_render(entries, settings, workdir.root, plot_paths))


def _entry_payload(entry, dataset, *, recomputed: bool) -> dict[str, Any]:
    """One row of the reduce output: the entry's provenance plus curve statistics."""
    asymmetry = np.asarray(dataset.asymmetry, dtype=np.float64)
    error = np.asarray(dataset.error, dtype=np.float64)
    data = entry.to_dict()
    data["recomputed"] = recomputed
    data["a0_percent"] = float(np.mean(asymmetry[:_A0_POINTS])) if asymmetry.size else None
    data["mean_error_percent"] = float(np.mean(error)) if error.size else None
    return data


def _render(
    entries: list[dict[str, Any]], settings, workdir_root: Path, plot_paths: list[Path]
) -> str:
    """The human-readable reduce table plus the settings line."""
    headers = ["run", "T/K", "B/G", "points", "A(0)/%", "err/%", "alpha", "deadtime"]
    rows = [
        [
            str(entry["run_number"]),
            format_number(entry["run"]["temperature"], 2),
            format_number(entry["run"]["field"], 2),
            str(entry["n_points"]),
            format_number(entry["a0_percent"], 3),
            format_number(entry["mean_error_percent"], 3),
            f"{entry['alpha']:.4f}",
            entry["deadtime_mode"],
        ]
        for entry in entries
    ]
    reused = sum(1 for entry in entries if not entry["recomputed"])
    lines = [
        render_table(headers, rows),
        "",
        f"alpha {settings.alpha:.4f} ({settings.alpha_source}), "
        f"deadtime {settings.deadtime}, background {settings.background}, "
        f"rebin {settings.rebin}, period {settings.period or 'default'}",
        f"{len(entries)} run(s) reduced into {workdir_root}"
        + (f" ({reused} reused from cache)" if reused else ""),
    ]
    if plot_paths:
        lines.append(f"Plots written: {', '.join(str(path) for path in plot_paths)}")
    return "\n".join(lines)
