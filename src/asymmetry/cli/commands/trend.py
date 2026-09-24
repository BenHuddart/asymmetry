"""``asymmetry trend`` — the parameter trend of a stored series, optionally fitted."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from asymmetry.cli._output import (
    UserError,
    checked_name,
    emit_json,
    format_number,
    payload,
    render_table,
)
from asymmetry.cli._recipes import parse_fix
from asymmetry.cli._runs import parse_run_spec
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``trend`` subcommand."""
    parser = subparsers.add_parser(
        "trend",
        help="Print, export or fit the parameter trend of a stored series",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--series", required=True, help="Name the series was stored under")
    parser.add_argument("--csv", default=None, help="Also write the table to this CSV file")
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Write plots/<series>-trend-<param>.png for every free parameter "
            "(with --model, only for --param, with the fitted curve)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="EXPR",
        help="Fit this parameter-vs-x expression to --param, e.g. 'OrderParameter', 'Redfield'",
    )
    parser.add_argument("--param", default=None, help="The trend column --model is fitted to")
    parser.add_argument("--xmin", type=float, default=None, help="Fit range start, in x units")
    parser.add_argument("--xmax", type=float, default=None, help="Fit range end, in x units")
    parser.add_argument(
        "--initial",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Model start value (repeatable)",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Hold a model parameter at this value (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        metavar="RUNS",
        help="Leave these runs out of the fit (every other run with a value enters)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Read the stored series, report its trend table and fit it when asked."""
    from asymmetry.core.workflow.series import TrendTable

    fit_options = [
        flag
        for flag, value in (
            ("--param", args.param),
            ("--xmin", args.xmin),
            ("--xmax", args.xmax),
            ("--initial", args.initial or None),
            ("--fix", args.fix or None),
            ("--exclude", args.exclude),
        )
        if value is not None
    ]
    if args.model is None and fit_options:
        raise UserError(f"{', '.join(fit_options)} require --model EXPR.")
    if args.model is not None and args.param is None:
        raise UserError("--model needs --param NAME, the trend column to fit.")
    if args.plot:
        from asymmetry.cli import plots

        plots.require_matplotlib()

    workdir = workdir_for(Path(args.folder), args.workdir)
    name = checked_name(args.series, flag="--series")
    stored = workdir.series_names()
    if name not in stored:
        known = ", ".join(stored) if stored else "none yet — run 'asymmetry fit-series' first"
        raise UserError(f"No series {name!r} in {workdir.series_dir} (it holds: {known}).")

    try:
        series = workdir.read_series(name)
    except KeyError as exc:
        raise UserError(exc.args[0]) from None
    trend = TrendTable(**series["trend"])

    fit = None
    if args.model is not None:
        from asymmetry.core.workflow.trend_fit import fit_trend

        try:
            fit = fit_trend(
                trend,
                args.param,
                args.model,
                x_min=args.xmin,
                x_max=args.xmax,
                initial=parse_fix(args.initial, flag="--initial"),
                fixed=parse_fix(args.fix),
                exclude=parse_run_spec(args.exclude) if args.exclude else (),
            ).to_dict()
        except ValueError as exc:
            raise UserError(str(exc)) from None
        series["trend_fits"][args.param] = fit
        workdir.write_series(name, series)

    csv_path = None
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(trend.to_csv(), encoding="utf-8")

    plot_paths: list[Path] = []
    if args.plot:
        plot_paths = _plots(workdir, name, series, trend, fit)

    if args.json:
        emit_json(
            payload(
                name=series["name"],
                expression=series["expression"],
                trend=trend.to_dict(),
                fit=fit,
                csv_path=None if csv_path is None else str(csv_path),
                plots=[str(path) for path in plot_paths],
            )
        )
        return

    print(_render(series, trend, fit, csv_path, plot_paths))


def _plots(workdir, name: str, series: dict[str, Any], trend, fit: dict | None) -> list[Path]:
    """One trend plot per free parameter, or the fitted one with its curve."""
    from asymmetry.cli import plots
    from asymmetry.core.fitting.parameter_models import ParameterCompositeModel

    if fit is None:
        return [
            plots.plot_trend(
                trend.rows,
                param_name=param_name,
                order_key=trend.order_key,
                out_path=workdir.trend_plot_path(name, param_name),
                title=f"{series['name']} — {series['expression']}: {param_name}",
            )
            for param_name in series["free_params"]
        ]
    fitted_x = [row["x"] for row in trend.rows if row["run"] in fit["runs"]]
    return [
        plots.plot_trend(
            trend.rows,
            param_name=fit["param"],
            order_key=trend.order_key,
            out_path=workdir.trend_plot_path(name, fit["param"]),
            title=f"{series['name']}: {fit['param']} — {fit['expression']}",
            model=(
                ParameterCompositeModel.from_expression(fit["expression"]).function,
                fit["parameters"],
                (min(fitted_x), max(fitted_x)),
            ),
        )
    ]


def _render(
    series: dict[str, Any],
    trend,
    fit: dict | None,
    csv_path: Path | None,
    plot_paths: list[Path],
) -> str:
    """The human-readable trend table, then the fit when there is one."""
    rows = [[_cell(row[column]) for column in trend.columns] for row in trend.rows]
    lines = [
        f"{series['name']} — {series['expression']}, ordered by {trend.order_key}",
        "",
        render_table(trend.columns, rows),
    ]
    if fit is not None:
        lines.extend(["", *_render_fit(fit, series["free_params"])])
    else:
        lines.extend(["", *_law_hints(series["name"], trend.order_key, series["free_params"])])
    if csv_path is not None:
        lines.extend(["", f"Trend written to {csv_path}"])
    if plot_paths:
        lines.append(f"Plots written: {', '.join(str(path) for path in plot_paths)}")
    return "\n".join(lines)


#: Base names of relaxation rates and of precession frequencies.
_RATE_BASES = frozenset({"Lambda", "lambda", "sigma", "Delta", "nu"})
_FREQUENCY_BASES = frozenset({"frequency", "freq", "B_int", "field"})

#: Axes a trend is read along from the files; anything else was supplied.
_FILE_AXES = frozenset({"temperature", "sample_temperature_logged", "field", "run"})


def _law_hints(name: str, order_key: str, free_params: list[str]) -> list[str]:
    """Which trend law this series' axis and parameters call for (Step 6a).

    A field-ordered rate is the Redfield question and needs one rate; a
    frequency against temperature is an order parameter; a rate against a
    supplied quantity (a concentration) is a linear rate law.
    """
    by_base: dict[str, list[str]] = {}
    for param in free_params:
        by_base.setdefault(re.sub(r"_\d+$", "", param), []).append(param)
    rates = [p for base, ps in by_base.items() if base in _RATE_BASES for p in ps]
    frequencies = [p for base, ps in by_base.items() if base in _FREQUENCY_BASES for p in ps]
    command = f"asymmetry trend <folder> --series {name} --model"
    hints: list[str] = []
    if order_key == "field" and rates:
        hints.append(
            f"A relaxation rate against field is the decoupling question: {command} "
            f"Redfield --param {rates[0]} (--fix m=2 for the textbook form)."
        )
        siblings = by_base[re.sub(r"_\d+$", "", rates[0])]
        if len(siblings) > 1:
            hints.append(
                f"  This series splits the rate between {', '.join(siblings)}: Redfield "
                f"describes one rate, so refit the series with a single-rate recipe first."
            )
    if order_key in ("temperature", "sample_temperature_logged") and frequencies:
        hints.append(
            f"A precession frequency against temperature is an order parameter: {command} "
            f"OrderParameter --param {frequencies[0]} (fit below the transition)."
        )
    if order_key not in _FILE_AXES and rates:
        hints.append(
            f"A rate against a supplied {order_key} is a rate law: {command} Linear "
            f"--param {rates[0]}."
        )
    if not hints:
        hints.append(
            "If the experiment asks for a number this trend encodes — a transition "
            "temperature, a correlation time, an activation energy — fit the law for it: "
            f"{command} <law> --param <column> (Step 6a of the skill)."
        )
    return ["Next — the law this trend calls for:", *hints]


def _render_fit(fit: dict[str, Any], free_params: list[str]) -> list[str]:
    """The fit block: model, range, parameters with errors, and what was left out."""
    span = (
        ""
        if fit["x_min"] is None and fit["x_max"] is None
        else f" over {format_number(fit['x_min'], 4)} .. {format_number(fit['x_max'], 4)}"
    )
    lines = [
        f"Fit of {fit['expression']} to {fit['param']} against {fit['order_key']}{span}: "
        f"{fit['n_points']} point(s), "
        + (
            f"chi2_red {format_number(fit['reduced_chi_squared'], 3)}"
            if fit["success"]
            else f"FAILED — {fit['message']}"
        ),
    ]
    # A χ²ᵣ above 1 says the points scatter more than their own errors allow;
    # the fit's errors scaled by √χ²ᵣ are the honest ones then, so they lead.
    scale = max(1.0, fit["reduced_chi_squared"]) ** 0.5 if fit["success"] else 1.0

    def error(name: str, factor: float) -> str:
        if name in fit["fixed"]:
            return "fixed"
        return format_number(fit["uncertainties"].get(name, 0.0) * factor, 6) + (
            " (at bound)" if name in fit["params_at_bound"] else ""
        )

    rows = [
        [name, format_number(value, 6), error(name, scale)]
        + ([error(name, 1.0)] if scale > 1.0 else [])
        for name, value in fit["parameters"].items()
    ]
    headers = ["parameter", "value"] + (
        [f"error (x sqrt(chi2_red) = {scale:.3g})", "unscaled error"] if scale > 1.0 else ["error"]
    )
    lines.append(render_table(headers, rows))
    undetermined = [
        name
        for name, value in fit["parameters"].items()
        if name not in fit["fixed"]
        and not 0.0 < abs(fit["uncertainties"].get(name, 0.0)) < abs(value)
    ]
    if not fit["success"] or fit["params_at_bound"] or undetermined:
        reasons = (
            (["the fit did not converge"] if not fit["success"] else [])
            + [f"{name} is at a bound" for name in fit["params_at_bound"]]
            + [f"{name}'s error is missing, zero or as large as its value" for name in undetermined]
        )
        lines.append(
            f"LAW NOT ESTABLISHED ({'; '.join(reasons)}): {fit['expression']} does not describe "
            f"this trend. Describe the trend in plain words and do not use this law's physics."
        )
    base = re.sub(r"_\d+$", "", fit["param"])
    siblings = [
        name for name in free_params if name != fit["param"] and re.sub(r"_\d+$", "", name) == base
    ]
    if siblings:
        lines.append(
            f"NOTE: {fit['param']} is one of several {base} components in this series' model "
            f"(also {', '.join(siblings)}). Check it is the component the law describes — "
            f"the one physical rate or line — and not one of two that together describe it."
        )
    if fit["excluded"]:
        lines.append(
            "left out: "
            + "; ".join(f"{entry['run']} ({entry['reason']})" for entry in fit["excluded"])
        )
    if fit["flagged"]:
        lines.append(
            "flagged but fitted: "
            + "; ".join(f"{entry['run']} ({', '.join(entry['flags'])})" for entry in fit["flagged"])
        )
    return lines


def _cell(value: Any) -> str:
    """Render one trend cell for the terminal table."""
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
