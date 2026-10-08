"""``asymmetry fit-series`` — fit one recipe across a scan, chained."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from asymmetry.cli._axis import add_axis_arguments, axis_from_arguments
from asymmetry.cli._output import (
    UserError,
    checked_name,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._recipes import add_recipe_arguments, load_recipe, recipe_with_overrides
from asymmetry.cli._runs import (
    present_runs,
    range_text,
    reduced_datasets,
    run_spec,
    window_note,
)
from asymmetry.cli._workdir import add_workdir_argument, workdir_for
from asymmetry.cli.commands.fit import HELD_BACKGROUND


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``fit-series`` subcommand."""
    parser = subparsers.add_parser(
        "fit-series",
        help="Fit a recipe across a scan of reduced runs, chained along the scan order",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument(
        "--runs",
        required=True,
        help="Run numbers, e.g. '17294-17322'",
    )
    add_recipe_arguments(parser, free=False)
    add_axis_arguments(parser, default=None)
    parser.add_argument("--tmin", type=float, default=None, help="Fit only above this time / µs")
    parser.add_argument("--tmax", type=float, default=None, help="Fit only below this time / µs")
    parser.add_argument(
        "--global",
        dest="global_params",
        default="",
        metavar="P,Q",
        help=(
            "Parameters held identical across every run. The batch is "
            "block-separable, so these are pinned at their recipe value and not "
            "fitted; everything else is free per run"
        ),
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="RUN",
        help=(
            "Chain outward from this run in both directions, instead of from "
            "the first run in scan order. Screen the run with the clearest "
            "structure ('asymmetry wizard --run N'), then start the series "
            "there ('--start N'): every fit then warm-starts from a neighbour "
            "nearer the run the recipe describes"
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name to store the series under (default: series-<recipe stem>)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Write plots/<name>/<run>.png per run and "
            "plots/<name>-trend-<param>.png per free parameter"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Fit every named run with the recipe and store the series."""
    from asymmetry.cli import plots
    from asymmetry.core.workflow.series import fit_series

    if args.plot:
        plots.require_matplotlib()

    folder = Path(args.folder)
    workdir, selection = workdir_for(folder, args.workdir, args.instrument)

    recipe = recipe_with_overrides(load_recipe(workdir, args.recipe), fix=args.fix)
    if args.tmin is not None or args.tmax is not None:
        recipe = recipe.with_window(t_min=args.tmin, t_max=args.tmax)
    global_params = [name.strip() for name in args.global_params.split(",") if name.strip()]
    # ``args.name is None`` — not falsy — is "no --name given": an explicit
    # empty one is a name that cannot be used, and says so rather than
    # quietly becoming the default.
    default_name = f"series-{Path(args.recipe).stem}"
    name = checked_name(default_name if args.name is None else args.name, flag="--name")

    datasets = reduced_datasets(workdir, present_runs(selection, args.runs))

    # Both of these are the user's choices, checked before a minute of fitting
    # starts so a typo or an unlabelled run exits 1 rather than 2.
    unknown = sorted(set(global_params) - set(recipe.parameter_names))
    if unknown:
        raise UserError(
            f"--global names {', '.join(unknown)}, which {recipe.expression!r} does not "
            f"have (it has {', '.join(recipe.parameter_names)})."
        )
    if args.start is not None and args.start not in datasets:
        raise UserError(
            f"--start {args.start} is not in the series "
            f"(it holds {', '.join(str(run) for run in sorted(datasets))})."
        )
    axis = axis_from_arguments(args, datasets)

    outcome = fit_series(
        datasets,
        recipe,
        axis=axis,
        global_params=global_params,
        start_run=args.start,
        name=name,
    )
    # Additive: so that a later `trend --plot` on this series (a separate
    # invocation, with no recipe in hand) can rebuild the model curve.
    series_payload = outcome.to_dict() | {
        "kind": "series",
        "recipe": recipe.to_dict(),
        "trend_fits": {},
    }
    series_path = workdir.write_series(name, series_payload)

    plot_paths: list[Path] = []
    if args.plot:
        for entry in outcome.results:
            run_number = entry["run"]
            plot_paths.append(
                plots.plot_fit(
                    datasets[run_number].time,
                    datasets[run_number].asymmetry,
                    datasets[run_number].error,
                    model_function=recipe.model().function,
                    parameters=entry["parameters"],
                    t_min=recipe.t_min,
                    t_max=recipe.t_max,
                    run_number=run_number,
                    expression=outcome.expression,
                    out_path=workdir.series_plot_dir(name) / f"{run_number}.png",
                )
            )
        for param_name in outcome.free_params:
            plot_paths.append(
                plots.plot_trend(
                    outcome.trend.rows,
                    param_name=param_name,
                    order_key=outcome.order_key,
                    out_path=workdir.trend_plot_path(name, param_name),
                    title=f"{name} — {outcome.expression}: {param_name}",
                )
            )

    if args.json:
        emit_json(
            payload(
                series=outcome.to_dict(),
                series_path=str(series_path),
                plots=[str(path) for path in plot_paths],
            )
        )
        return

    print(_render(outcome, series_path, plot_paths))
    held = [entry["run"] for entry in outcome.results if entry["background_held"]]
    if held:
        print(
            f"NOTE: on {range_text(held)} {HELD_BACKGROUND} Say in the summary that it was "
            f"held there."
        )
    from asymmetry.core.workflow.series import envelope_change, lineless_end

    for note in (window_note(workdir, sorted(datasets)), envelope_change(outcome.trend)):
        if note is not None:
            print(note)
    lineless = lineless_end(outcome.trend)
    if lineless:
        folder_arg = shlex.quote(args.folder)
        order_key = outcome.order_key
        runs = sorted(int(row["key"]) for row in lineless)
        middle = lineless[len(lineless) // 2]["key"]
        supplied = (
            ""
            if args.x is None
            else " --x " + ",".join(f"{row['key']}={row['x']:g}" for row in lineless)
        )
        upper = lineless[-1] is outcome.trend.rows[-1]
        # A line also vanishes where the muon's local field passes through zero,
        # well inside the ordered phase, so the warm end is only *likely* paramagnetic.
        side = (
            "above an ordering transition the paramagnetic side (a line also vanishes where "
            "the local field at the muon passes through zero, so check before naming a "
            f"transition), and the physics is a relaxation rate to report against {order_key}"
            if upper and order_key in ("temperature", "sample_temperature_logged")
            else f"where the physics is a relaxation rate to report against {order_key}"
        )
        print(
            f"NOTE: the survey finds no precession in {range_text(runs)} ({order_key} "
            f"{lineless[0]['x']:g}–{lineless[-1]['x']:g}, the {'high' if upper else 'low'}-"
            f"{order_key} end of this scan): the side of the transition with no precession, {side}. "
            f"This precession model has no line to describe there, flagged or not, so its "
            f"values on these runs are not results. Fit them with a relaxation-only recipe:\n"
            f"  asymmetry recipe {folder_arg} --expression 'Exponential + Constant' "
            f"--run {middle} --name {name}-relax\n"
            f"  asymmetry fit-series {folder_arg} --runs {run_spec(runs)} --recipe "
            f"{name}-relax --order {order_key}{supplied} --name {name}-relax\n"
            f"(or screen run {middle} with the wizard for the relaxation shape first). Only "
            f"where this series still holds a frequency on a weak line the survey missed (a "
            f"free envelope width can swallow one), refit those runs with the width held "
            f"(--fix) at a value from the runs that do precess."
        )
    from asymmetry.cli.commands.wizard import runner_up_name

    runner_up = runner_up_name(Path(args.recipe).stem)
    misfit = [
        entry["run"] for entry in outcome.results if _MISFIT_FLAGS & set(entry["quality_flags"])
    ]
    if 2 * len(misfit) > len(outcome.results) and runner_up in workdir.recipe_names():
        supplied = "" if args.x is None else f" --x {shlex.quote(args.x)}"
        start = "" if args.start is None else f" --start {args.start}"
        print(
            f"NOTE: {len(misfit)} of {len(outcome.results)} runs are flagged failed or "
            f"amplitude_exceeds_data, so {outcome.expression} does not describe this scan. "
            f"The screening that wrote {args.recipe} also wrote its runner-up, "
            f"{workdir.read_recipe(runner_up).expression}, which that run hardly told apart: "
            f"fit the scan with it and keep whichever series describes the runs.\n"
            f"  asymmetry fit-series {shlex.quote(args.folder)} --runs {args.runs} --recipe "
            f"{runner_up} --order {outcome.order_key}{supplied}{start} --name {name}-alt"
        )
    print(
        f"Next: asymmetry trend {shlex.quote(args.folder)} --series {name} — the trend "
        f"table, and the law it calls for."
    )
    if args.order == "temperature":
        from asymmetry.core.workflow.survey import departs

        departing = sorted(
            run
            for run, dataset in datasets.items()
            if departs(
                dataset.metadata.get("temperature"),
                dataset.metadata.get("sample_temperature_logged"),
            )
        )
        if departing:
            print(
                f"NOTE: ordered by the setpoint, but the logged sample temperature departs on "
                f"{len(departing)} of these runs ({', '.join(str(run) for run in departing)}). "
                f"Decide which axis the physics follows — a parameter that is smooth against "
                f"one and not the other says which — and refit with --order "
                f"sample_temperature_logged if it is the logged one."
            )


#: Flags that say a run's fit ran away rather than measured the physics.
_MISFIT_FLAGS = frozenset({"failed", "amplitude_exceeds_data"})


def _render(outcome, series_path: Path, plot_paths: list[Path] | None = None) -> str:
    """The human-readable per-run table plus the seeding that was used."""
    compared = "envelope" in outcome.trend.columns
    headers = [
        "run",
        outcome.order_key,
        "chi2_red",
        "verdict",
        *(["envelope (dchi2 rival-own)"] if compared else []),
        "flags",
    ]
    rows = []
    for entry in outcome.results:
        quality = entry["quality"]
        rows.append(
            [
                str(entry["run"]),
                format_number(entry["x"], 3),
                format_number(entry["reduced_chi_squared"], 3),
                "unknown" if quality is None else quality["verdict"],
                *(
                    [
                        "-"
                        if entry["envelope"]["preferred"] is None
                        else f"{entry['envelope']['preferred']} "
                        f"({format_number(entry['envelope']['delta_chi2'], 3)})"
                    ]
                    if compared
                    else []
                ),
                ", ".join(entry["quality_flags"]) or "-",
            ]
        )

    flagged = sum(1 for entry in outcome.results if entry["quality_flags"])
    lines = [
        f"{outcome.name} — {outcome.expression}, {len(outcome.results)} run(s) "
        f"ordered by {outcome.order_key}"
        + (f", chained outward from run {outcome.start_run}" if outcome.start_run else ""),
        "",
        render_table(headers, rows),
        "",
    ]
    for branch in outcome.branches:
        label = f"seeding ({branch.direction}, {len(branch.runs)} run(s))"
        lines.append(
            f"{label}: {branch.seeding_used}"
            + (f" — {branch.seeding_reason}" if branch.seeding_reason else "")
        )
    if outcome.reseeded_runs:
        lines.append("reseeded : " + ", ".join(str(run) for run in outcome.reseeded_runs))
    if outcome.global_params:
        lines.append(f"pinned   : {', '.join(outcome.global_params)}")
    lines.append(f"flagged  : {flagged} of {len(outcome.results)} run(s) — none were dropped")
    lines.append(f"Series written to {series_path}")
    if plot_paths:
        lines.append(f"{len(plot_paths)} plot(s) written under {plot_paths[0].parent.parent}")
    return "\n".join(lines)
