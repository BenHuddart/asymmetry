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
from asymmetry.cli._reduction import add_reduction_arguments, describe, reduction_settings
from asymmetry.cli._runs import coadd_note, range_text, resolve_runs
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
    parser.add_argument(
        "--coadd",
        action="store_true",
        help=(
            "Sum the named runs' counts (the GUI's co-add) and reduce the sum as one run, "
            "stored under the first run's number; that run's own reduction is replaced, "
            "so keep both in separate --workdir directories"
        ),
    )
    add_reduction_arguments(parser)
    parser.add_argument("--rebin", type=int, default=1, help="Merge this many bins (default: 1)")
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Discard points below this time/µs from the stored reduction every later fit uses",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help=(
            "Discard points above this time/µs from the stored reduction every later fit "
            "uses; to zoom the plot only, use --plot-tmax"
        ),
    )
    parser.add_argument(
        "--plot-tmax",
        dest="plot_tmax",
        type=float,
        default=None,
        help="Draw the reduced PNG only up to this time/µs; the stored reduction keeps it all",
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
    from asymmetry.core.io.periods import period_count
    from asymmetry.core.workflow.reduction import (
        load_reduction_source,
        reduce_run,
        resolve_reduction_grouping,
    )
    from asymmetry.core.workflow.survey import build_run_row, precession_evidence
    from asymmetry.core.workflow.workdir import ReducedEntry, reduction_digest

    folder = Path(args.folder)
    if args.plot:
        plots.require_matplotlib()
    workdir, selection = workdir_for(folder, args.workdir, args.instrument)
    settings = reduction_settings(
        args, selection, rebin=args.rebin, t_min=args.tmin, t_max=args.tmax
    )

    targets = resolve_runs(selection, args.runs)
    if args.coadd and len(targets) < 2:
        raise UserError(f"--coadd sums two or more runs; {args.runs!r} names one run file.")
    # One reduction per run, or one of the co-add, keyed on its first member.
    reductions = (
        [
            (
                targets[0][0],
                targets[0][1],
                [path for _run, _prefix, path in targets],
                [run for run, _prefix, _path in targets],
            )
        ]
        if args.coadd
        else [(run_number, prefix, [path], []) for run_number, prefix, path in targets]
    )
    # Written before the first spectrum, not after the last: the manifest is
    # what binds the directory to this data folder, so a reduction interrupted
    # part-way still leaves a directory that says whose runs are in it.
    workdir.write_manifest(
        selection,
        settings=settings,
        runs=[run_number for run_number, _prefix, _paths, _members in reductions],
    )

    entries: list[dict[str, Any]] = []
    plot_paths: list[Path] = []
    for run_number, prefix, paths, members in reductions:
        try:
            dataset_in = load_reduction_source(paths, settings.period)
            grouping = resolve_reduction_grouping(dataset_in.run, settings)
        except (TypeError, ValueError) as exc:
            source = f"Co-add of {range_text(members)}" if members else f"Run {run_number}"
            raise UserError(f"{source}: {exc}") from None
        digest = reduction_digest(source_files=paths, grouping=grouping, settings=settings)

        if workdir.is_current(run_number, digest):
            dataset = workdir.reduced(run_number)
            entry = workdir.entry(run_number)
            recomputed = False
        else:
            try:
                dataset = reduce_run(dataset_in.run, settings)
            except ValueError as exc:
                raise UserError(str(exc)) from None
            row = build_run_row(
                dataset_in,
                path=paths[0],
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
                source_file=str(paths[0]),
                n_points=dataset.n_points,
                settings=settings,
                run=row.to_dict(),
                alpha=float(grouping["alpha"]),
                deadtime_mode=str(grouping["deadtime_mode"]),
                forward_group=int(grouping["forward_group"]),
                backward_group=int(grouping["backward_group"]),
                members=members,
            )
            workdir.write_reduced(dataset, entry)
            recomputed = True

        entries.append(_entry_payload(entry, dataset, recomputed=recomputed))

        if args.plot:
            plot_paths.append(
                plots.plot_reduced(
                    *_plot_window(dataset, args.plot_tmax),
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


def _plot_window(dataset, plot_tmax: float | None):
    """``(time, asymmetry, error)`` up to *plot_tmax* for the PNG, the whole record without."""
    shown = dataset if plot_tmax is None else dataset.time_range(None, plot_tmax)
    return shown.time, shown.asymmetry, shown.error


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
        *(
            coadd_note(entry["run_number"], entry["members"])
            for entry in entries
            if entry["members"]
        ),
        describe(settings),
        f"{len(entries)} run(s) reduced into {workdir_root}"
        + (f" ({reused} reused from cache)" if reused else ""),
    ]
    if plot_paths:
        lines.append(f"Plots written: {', '.join(str(path) for path in plot_paths)}")
    return "\n".join(lines)
