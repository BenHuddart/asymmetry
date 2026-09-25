"""``asymmetry fit-global`` — a true coupled multi-run fit, or a batch of them."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

from asymmetry.cli._axis import (
    MEMBER_ORDER_DEFAULT,
    add_axis_arguments,
    axis_from_arguments,
    member_axis,
)
from asymmetry.cli._output import (
    UserError,
    checked_name,
    emit_json,
    format_number,
    payload,
    render_table,
    render_trend,
)
from asymmetry.cli._recipes import add_recipe_arguments, load_recipe, recipe_with_overrides
from asymmetry.cli._runs import parse_run_spec, reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fit-global", help="Fit multiple reduced runs simultaneously with shared parameters"
    )
    parser.add_argument("folder", help="Directory holding the run files")
    runs = parser.add_mutually_exclusive_group(required=True)
    runs.add_argument("--runs", help="Runs in one simultaneous-fit group")
    runs.add_argument(
        "--groups",
        metavar="RUNS;RUNS;...",
        help=(
            "Several simultaneous-fit groups, e.g. '101,102,103;104,105,106': each is "
            "fitted and stored as <name>-<i> (i from 1), and <name> stores the groups' "
            "shared parameters as a trend along --group-order"
        ),
    )
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
    add_axis_arguments(parser, default="run")
    add_axis_arguments(
        parser,
        default=MEMBER_ORDER_DEFAULT,
        noun="group",
        plural="groups of --groups",
        prefix="group-",
        example="1=0,2=0.25,3=0.5",
    )
    parser.add_argument("--name", default="global-fit", help="Stored fit name")
    parser.add_argument("--plot", action="store_true", help="Write one fitted plot per run")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from asymmetry.cli import plots

    if args.groups is None and (
        args.group_x is not None or args.group_order != MEMBER_ORDER_DEFAULT
    ):
        raise UserError("--group-order and --group-x order a batch of --groups.")
    if args.plot:
        plots.require_matplotlib()
    folder = Path(args.folder)
    workdir, _selection = workdir_for(folder, args.workdir, args.instrument)
    recipe = recipe_with_overrides(
        load_recipe(workdir, args.recipe),
        fix=args.fix,
        free=args.free,
    )
    shared = [name.strip() for name in args.shared.split(",") if name.strip()]
    name = checked_name(args.name, flag="--name")
    options = {
        "shared_params": shared,
        "field_params": [item.strip() for item in args.field_param if item.strip()],
        "strategy": args.strategy,
    }
    if args.groups is None:
        _fit_group(args, workdir, recipe, name, options)
    else:
        _fit_batch(args, workdir, recipe, name, options)


def _fit_group(args, workdir, recipe, name: str, options: dict[str, Any]) -> None:
    """One simultaneous fit of ``--runs``, stored and reported."""
    from asymmetry.core.workflow.global_fit import fit_global

    datasets = reduced_datasets(workdir, parse_run_spec(args.runs))
    axis = axis_from_arguments(args, datasets)
    try:
        outcome = fit_global(datasets, recipe, axis=axis, **options)
    except (KeyError, ValueError) as exc:
        raise UserError(str(exc)) from None
    result_payload = _store_group(workdir, name, outcome, recipe, datasets, plot=args.plot)
    if args.json:
        emit_json(payload(global_fit=result_payload))
        return
    print(_render(result_payload))
    # A decoupling triplet shares its dynamics; a scan over many fields asks how
    # the rate changes with field, which a shared rate cannot show.
    if axis.name == "field" and len(outcome.results) > 3:
        print(
            f"NOTE: {len(outcome.results)} runs along field with "
            f"{', '.join(options['shared_params'])} shared. "
            f"If the question is how the relaxation changes with field (decoupling, "
            f"Redfield), fit a single-rate recipe per run with fit-series --order field "
            f"and then trend --model Redfield instead."
        )


def _fit_batch(args, workdir, recipe, name: str, options: dict[str, Any]) -> None:
    """One simultaneous fit per group of ``--groups``, and the batch trend, stored and reported."""
    from asymmetry.core.workflow.global_fit import fit_global_batch
    from asymmetry.core.workflow.workdir import series_digest

    specs = [spec.strip() for spec in args.groups.split(";")]
    if not all(specs):
        raise UserError(f"--groups {args.groups!r} has an empty group between semicolons.")
    groups = {
        f"{name}-{index}": reduced_datasets(workdir, parse_run_spec(spec))
        for index, spec in enumerate(specs, start=1)
    }
    every_run: dict[int, Any] = {}
    for datasets in groups.values():
        repeated = sorted(set(every_run) & set(datasets))
        if repeated:
            raise UserError(f"Run(s) {', '.join(map(str, repeated))} are in two groups.")
        every_run.update(datasets)
    axis = axis_from_arguments(args, every_run)
    batch_axis = member_axis(
        args.group_order,
        args.group_x,
        groups,
        key=lambda text: f"{name}-{int(text)}",
        flag="--group-x",
        noun="group",
    )
    try:
        batch = fit_global_batch(groups, recipe, axis=axis, batch_axis=batch_axis, **options)
    except (KeyError, ValueError) as exc:
        raise UserError(str(exc)) from None

    group_payloads = [
        _store_group(workdir, member, outcome, recipe, groups[member], plot=args.plot)
        for member, outcome in batch.groups.items()
    ]
    stored = batch.to_dict() | {
        "name": name,
        "kind": "global-batch",
        "shared_params": options["shared_params"],
        "recipe": recipe.to_dict(),
        "members": [
            {
                "series": member,
                "fit": None,
                "digest": series_digest(workdir.read_series(member), None),
            }
            for member in batch.groups
        ],
        "series_path": str(workdir.series_path(name)),
        "trend_fits": {},
    }
    workdir.write_series(name, stored)
    if args.json:
        emit_json(payload(global_batch=stored | {"groups": group_payloads}))
        return
    for group_payload in group_payloads:
        print(_render(group_payload))
        print()
    print(
        "\n".join(
            [
                f"{name} — {batch.expression}, {len(batch.groups)} group(s) ordered by "
                f"{batch.trend.order_key}; each row is one group's shared parameters",
                "",
                render_trend(batch.trend),
                "",
                f"Batch written to {stored['series_path']}",
                f"Next: asymmetry trend {shlex.quote(args.folder)} --series {name} --model "
                f"<law> --param <shared parameter> — or a law per group on its run-local "
                f"parameters (--series {name}-<i>), then trend --from-fits across the groups.",
            ]
        )
    )


def _store_group(workdir, name: str, outcome, recipe, datasets, *, plot: bool) -> dict[str, Any]:
    """Write one simultaneous fit as the ``global`` series *name*, with its plots.

    The stored series carries its own path and its per-run plots', so a coupled
    fit read back from series/<name>.json names the same artefacts the CLI
    reports. The fit is written before the plots are drawn, so an expensive
    joint fit survives a failure in the (cheap) plotting step. Returns the
    reported payload.
    """
    from asymmetry.cli import plots

    plot_paths: list[Path] = (
        [workdir.series_plot_dir(name) / f"{result['run']}.png" for result in outcome.results]
        if plot
        else []
    )
    artefacts = {
        "series_path": str(workdir.series_path(name)),
        "plots": [str(path) for path in plot_paths],
    }
    workdir.write_series(
        name,
        outcome.to_dict()
        | {"name": name, "recipe": recipe.to_dict(), "kind": "global", "trend_fits": {}}
        | artefacts,
    )
    if plot:
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
    return outcome.to_dict() | {"name": name} | artefacts


def _render(outcome: dict) -> str:
    rows = [
        [
            str(result["run"]),
            format_number(result["x"], 3),
            *(
                format_number(result["parameters"][name], 4)
                + " ± "
                + format_number(result["uncertainties"].get(name), 4)
                for name in outcome["free_params"]
            ),
            format_number(result["reduced_chi_squared"], 3),
            "unknown" if result["quality"] is None else result["quality"]["verdict"],
            ", ".join(result["quality_flags"]) or "-",
        ]
        for result in outcome["results"]
    ]
    headers = [
        "run",
        outcome["order_key"],
        *outcome["free_params"],
        "chi2_red",
        "verdict",
        "flags",
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
        render_table(headers, rows),
        "",
        f"Fit written to {outcome['series_path']}",
    ]
    if outcome["plots"]:
        lines.append(f"{len(outcome['plots'])} plot(s) written")
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
