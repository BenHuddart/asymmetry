"""``asymmetry fit`` — fit one reduced run with a recipe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from asymmetry.cli._output import emit_json, format_number, payload, render_table
from asymmetry.cli._recipes import add_recipe_arguments, load_recipe, recipe_with_overrides
from asymmetry.cli._runs import reduced_datasets


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``fit`` subcommand."""
    parser = subparsers.add_parser(
        "fit",
        help="Fit one reduced run with a recipe",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Run number to fit")
    add_recipe_arguments(parser)
    parser.add_argument("--tmin", type=float, default=None, help="Fit only above this time/µs")
    parser.add_argument("--tmax", type=float, default=None, help="Fit only below this time/µs")
    parser.add_argument("--plot", action="store_true", help="Write plots/fit-<run>.png")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory to read (default: <folder>/.asymmetry)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Fit the named run and report the parameter table and quality verdict."""
    from asymmetry.cli import plots
    from asymmetry.core.workflow.series import fit_one
    from asymmetry.core.workflow.workdir import WorkDir

    if args.plot:
        plots.require_matplotlib()

    folder = Path(args.folder)
    workdir = WorkDir.for_folder(folder, args.workdir)
    dataset = reduced_datasets(workdir, [args.run])[args.run]

    recipe = recipe_with_overrides(load_recipe(workdir, args.recipe), fix=args.fix, free=args.free)
    if args.tmin is not None or args.tmax is not None:
        recipe = recipe.with_window(t_min=args.tmin, t_max=args.tmax)

    result = fit_one(dataset, recipe)

    plot_path = None
    if args.plot:
        plot_path = plots.plot_fit(
            dataset.time,
            dataset.asymmetry,
            dataset.error,
            model_function=recipe.model().function,
            parameters=result["parameters"],
            t_min=recipe.t_min,
            t_max=recipe.t_max,
            run_number=args.run,
            expression=recipe.expression,
            out_path=workdir.plots_dir / f"fit-{args.run}.png",
        )

    if args.json:
        emit_json(
            payload(
                fit=result,
                expression=recipe.expression,
                recipe=recipe.to_dict(),
                plots=[] if plot_path is None else [str(plot_path)],
            )
        )
        return

    print(_render(result, recipe, plot_path))


def _render(result: dict[str, Any], recipe, plot_path: Path | None = None) -> str:
    """The human-readable fit report: parameters, then χ² and flags."""
    free = set(result["free_params"])
    uncertainties = result["uncertainties"]
    headers = ["parameter", "value", "error", "state"]
    rows = []
    for name, value in result["parameters"].items():
        error = uncertainties.get(name)
        rows.append(
            [
                name,
                f"{value:.6g}",
                "-" if error is None else f"{error:.3g}",
                "free" if name in free else "fixed",
            ]
        )

    quality = result["quality"]
    verdict = "unknown" if quality is None else quality["verdict"]
    flags = result["quality_flags"]
    at_bound = result["params_at_bound"]
    lines = [
        f"Run {result['run']} — {recipe.expression}",
        "",
        render_table(headers, rows),
        "",
        f"chi2_red : {format_number(result['reduced_chi_squared'], 4)}  ({verdict})",
        f"converged: {'yes' if result['success'] else 'NO'}",
    ]
    if at_bound:
        lines.append(f"at bound : {', '.join(at_bound)}")
    lines.append(f"flags    : {', '.join(flags) if flags else 'none'}")
    if plot_path is not None:
        lines.append(f"Plot written to {plot_path}")
    return "\n".join(lines)
