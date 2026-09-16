"""``asymmetry fit-global`` — a true coupled multi-run fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import (
    UserError,
    checked_name,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._recipes import add_recipe_arguments, load_recipe, recipe_with_overrides
from asymmetry.cli._runs import parse_run_spec, reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fit-global", help="Fit multiple reduced runs simultaneously with shared parameters"
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--runs", required=True, help="Runs in one simultaneous-fit group")
    add_recipe_arguments(parser, free=True)
    parser.add_argument(
        "--shared", required=True, metavar="P,Q", help="Parameters fitted once across all runs"
    )
    parser.add_argument(
        "--field-param",
        action="append",
        default=[],
        metavar="NAME",
        help="Set this parameter from each run's field and hold it (repeatable)",
    )
    parser.add_argument(
        "--strategy",
        choices=["joint", "profiled", "least_squares"],
        default="joint",
    )
    parser.add_argument("--name", default="global-fit", help="Stored fit name")
    parser.add_argument("--plot", action="store_true", help="Write one fitted plot per run")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from asymmetry.cli import plots
    from asymmetry.core.workflow.global_fit import fit_global

    if args.plot:
        plots.require_matplotlib()
    folder = Path(args.folder)
    workdir = workdir_for(folder, args.workdir)
    recipe = recipe_with_overrides(
        load_recipe(workdir, args.recipe),
        fix=args.fix,
        free=args.free,
    )
    datasets = reduced_datasets(workdir, parse_run_spec(args.runs))
    shared = [name.strip() for name in args.shared.split(",") if name.strip()]
    name = checked_name(args.name, flag="--name")
    try:
        outcome = fit_global(
            datasets,
            recipe,
            shared_params=shared,
            field_params=[item.strip() for item in args.field_param if item.strip()],
            strategy=args.strategy,
        )
    except (KeyError, ValueError) as exc:
        raise UserError(str(exc)) from None

    # The stored series carries its own path and its per-run plots', so a
    # coupled fit read back from series/<name>.json names the same artefacts
    # the CLI reports. The fit is written before the plots are drawn, so an
    # expensive joint fit survives a failure in the (cheap) plotting step.
    plot_paths: list[Path] = (
        [workdir.series_plot_dir(name) / f"{result['run']}.png" for result in outcome.results]
        if args.plot
        else []
    )
    artefacts = {
        "series_path": str(workdir.series_path(name)),
        "plots": [str(path) for path in plot_paths],
    }
    stored = (
        outcome.to_dict() | {"name": name, "recipe": recipe.to_dict(), "kind": "global"} | artefacts
    )
    workdir.write_series(name, stored)
    if args.plot:
        for result, plot_path in zip(outcome.results, plot_paths, strict=True):
            run_number = result["run"]
            plots.plot_fit(
                datasets[run_number].time,
                datasets[run_number].asymmetry,
                datasets[run_number].error,
                model_function=recipe.model().function,
                parameters=result["parameters"],
                t_min=recipe.t_min,
                t_max=recipe.t_max,
                run_number=run_number,
                expression=outcome.expression,
                out_path=plot_path,
            )
    result_payload = outcome.to_dict() | {"name": name} | artefacts
    if args.json:
        emit_json(payload(global_fit=result_payload))
        return
    print(_render(result_payload))


def _render(outcome: dict) -> str:
    rows = [
        [
            str(result["run"]),
            format_number(result["reduced_chi_squared"], 3),
            "unknown" if result["quality"] is None else result["quality"]["verdict"],
            ", ".join(result["quality_flags"]) or "-",
        ]
        for result in outcome["results"]
    ]
    shared = ", ".join(
        f"{name}={format_number(value, 6)}"
        + (
            f" ± {format_number(outcome['shared_uncertainties'][name], 6)}"
            if name in outcome["shared_uncertainties"]
            else ""
        )
        for name, value in outcome["shared"].items()
    )
    lines = [
        f"{outcome['name']} — {outcome['expression']}, {len(rows)} simultaneously fitted run(s)",
        f"shared: {shared or '-'}",
        "",
        render_table(["run", "chi2_red", "verdict", "flags"], rows),
        "",
        f"Fit written to {outcome['series_path']}",
    ]
    if outcome["plots"]:
        lines.append(f"{len(outcome['plots'])} plot(s) written")
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
