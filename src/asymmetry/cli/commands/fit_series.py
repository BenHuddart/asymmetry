"""``asymmetry fit-series`` — fit one recipe across a scan, chained."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import UserError, emit_json, format_number, payload, render_table
from asymmetry.cli._recipes import add_recipe_arguments, load_recipe, recipe_with_overrides
from asymmetry.cli._runs import parse_run_spec, reduced_datasets


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
    parser.add_argument(
        "--order",
        choices=["temperature", "field", "run"],
        required=True,
        help="Scan quantity the series is ordered and trended along",
    )
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
        "--name",
        default=None,
        help="Name to store the series under (default: series-<recipe stem>)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory to read and write (default: <folder>/.asymmetry)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Fit every named run with the recipe and store the series."""
    from asymmetry.core.workflow.series import fit_series, order_values
    from asymmetry.core.workflow.workdir import WorkDir

    folder = Path(args.folder)
    workdir = WorkDir.for_folder(folder, args.workdir)

    recipe = recipe_with_overrides(load_recipe(workdir, args.recipe), fix=args.fix)
    global_params = [name.strip() for name in args.global_params.split(",") if name.strip()]
    name = args.name or f"series-{Path(args.recipe).stem}"

    datasets = reduced_datasets(workdir, parse_run_spec(args.runs))

    # Both of these are the user's choices, checked before a minute of fitting
    # starts so a typo or an unlabelled run exits 1 rather than 2.
    unknown = sorted(set(global_params) - set(recipe.parameter_names))
    if unknown:
        raise UserError(
            f"--global names {', '.join(unknown)}, which {recipe.expression!r} does not "
            f"have (it has {', '.join(recipe.parameter_names)})."
        )
    try:
        order_values(datasets, args.order)
    except ValueError as exc:
        raise UserError(str(exc)) from None

    outcome = fit_series(
        datasets,
        recipe,
        order_key=args.order,
        global_params=global_params,
        name=name,
    )
    series_path = workdir.write_series(name, outcome.to_dict())

    if args.json:
        emit_json(payload(series=outcome.to_dict(), series_path=str(series_path)))
        return

    print(_render(outcome, series_path))


def _render(outcome, series_path: Path) -> str:
    """The human-readable per-run table plus the seeding that was used."""
    headers = ["run", outcome.order_key, "chi2_red", "verdict", "flags"]
    rows = []
    for entry in outcome.results:
        quality = entry["quality"]
        rows.append(
            [
                str(entry["run"]),
                format_number(entry["x"], 3),
                format_number(entry["reduced_chi_squared"], 3),
                "unknown" if quality is None else quality["verdict"],
                ", ".join(entry["quality_flags"]) or "-",
            ]
        )

    flagged = sum(1 for entry in outcome.results if entry["quality_flags"])
    lines = [
        f"{outcome.name} — {outcome.expression}, {len(outcome.results)} run(s) "
        f"ordered by {outcome.order_key}",
        "",
        render_table(headers, rows),
        "",
        f"seeding  : {outcome.seeding_used}"
        + (f" ({outcome.seeding_reason})" if outcome.seeding_reason else ""),
    ]
    if outcome.reseeded_runs:
        lines.append("reseeded : " + ", ".join(str(run) for run in outcome.reseeded_runs))
    if outcome.global_params:
        lines.append(f"pinned   : {', '.join(outcome.global_params)}")
    lines.append(f"flagged  : {flagged} of {len(outcome.results)} run(s) — none were dropped")
    lines.append(f"Series written to {series_path}")
    return "\n".join(lines)
