"""``asymmetry recipe`` — write a fit recipe for a model the physics calls for."""

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
from asymmetry.cli._recipes import parse_fix
from asymmetry.cli._runs import reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``recipe`` subcommand."""
    parser = subparsers.add_parser(
        "recipe",
        help="Write a fit recipe for a model expression, bypassing the wizard",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument(
        "--expression",
        required=True,
        help="Time-domain model, e.g. 'Oscillatory * Exponential + Constant'",
    )
    parser.add_argument("--name", required=True, help="Name to store the recipe under")
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Seed amplitudes, phase, background, applied field and Larmor frequency from this reduced run",
    )
    parser.add_argument(
        "--initial",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Starting value (repeatable)",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Hold a parameter at a value (repeatable)",
    )
    parser.add_argument("--tmin", type=float, default=None, help="Fit window start / µs")
    parser.add_argument("--tmax", type=float, default=None, help="Fit window end / µs")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="write into")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Build the recipe, store it, and print every parameter it carries."""
    from asymmetry.core.workflow.recipe import FitRecipe

    workdir = workdir_for(Path(args.folder), args.workdir)
    name = checked_name(args.name, flag="--name")
    dataset = None if args.run is None else reduced_datasets(workdir, [args.run])[args.run]
    try:
        recipe = FitRecipe.from_expression(
            args.expression, dataset=dataset, t_min=args.tmin, t_max=args.tmax
        ).with_overrides(initial=parse_fix(args.initial, flag="--initial"), fix=parse_fix(args.fix))
    except KeyError as exc:
        raise UserError(str(exc.args[0])) from None
    except ValueError as exc:
        raise UserError(str(exc)) from None
    recipe_path = workdir.write_recipe(name, recipe)

    if args.json:
        emit_json(payload(recipe=recipe.to_dict(), recipe_name=name, recipe_path=str(recipe_path)))
        return

    rows = [
        [
            parameter.name,
            format_number(parameter.value, 6),
            format_number(parameter.min, 4),
            format_number(parameter.max, 4),
            "fixed" if parameter.fixed else "free",
        ]
        for parameter in recipe.parameters
    ]
    print(
        "\n".join(
            [
                f"{name} — {recipe.expression}"
                + ("" if args.run is None else f", seeded from run {args.run}"),
                "",
                render_table(["parameter", "start", "min", "max", "state"], rows),
                "",
                f"Recipe written to {recipe_path}; fit it with --recipe {name}",
            ]
        )
    )


__all__ = ["add_parser", "run"]
