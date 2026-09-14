"""``asymmetry trend`` — the parameter trend of a stored series."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from asymmetry.cli._output import UserError, emit_json, payload, render_table


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``trend`` subcommand."""
    parser = subparsers.add_parser(
        "trend",
        help="Print (or export) the parameter trend of a stored series",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--series", required=True, help="Name the series was stored under")
    parser.add_argument("--csv", default=None, help="Also write the table to this CSV file")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write plots/<series>-trend-<param>.png for every free parameter",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory to read (default: <folder>/.asymmetry)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Read the stored series and report its trend table."""
    from asymmetry.core.workflow.series import TrendTable
    from asymmetry.core.workflow.workdir import WorkDir

    if args.plot:
        from asymmetry.cli import plots

        plots.require_matplotlib()

    workdir = WorkDir.for_folder(Path(args.folder), args.workdir)
    stored = workdir.series_names()
    if args.series not in stored:
        known = ", ".join(stored) if stored else "none yet — run 'asymmetry fit-series' first"
        raise UserError(f"No series {args.series!r} in {workdir.series_dir} (it holds: {known}).")

    series = workdir.read_series(args.series)
    trend = TrendTable(**series["trend"])

    csv_path = None
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(trend.to_csv(), encoding="utf-8")

    plot_paths: list[Path] = []
    if args.plot:
        from asymmetry.core.workflow.recipe import FitRecipe

        # `fit-series` stores the recipe it fitted with so a later, separate
        # `trend --plot` (no recipe in hand otherwise) can rebuild the model —
        # here, just its expression, for the same plot titles `fit-series
        # --plot` itself writes.
        recipe = FitRecipe.from_dict(series["recipe"])
        for param_name in series["free_params"]:
            plot_paths.append(
                plots.plot_trend(
                    trend.rows,
                    param_name=param_name,
                    order_key=trend.order_key,
                    out_path=workdir.plots_dir / f"{args.series}-trend-{param_name}.png",
                    title=f"{series['name']} — {recipe.expression}: {param_name}",
                )
            )

    if args.json:
        emit_json(
            payload(
                name=series["name"],
                expression=series["expression"],
                trend=trend.to_dict(),
                csv_path=None if csv_path is None else str(csv_path),
                plots=[str(path) for path in plot_paths],
            )
        )
        return

    print(_render(series, trend, csv_path, plot_paths))


def _render(
    series: dict[str, Any], trend, csv_path: Path | None, plot_paths: list[Path] | None = None
) -> str:
    """The human-readable trend table."""
    rows = [[_cell(row[column]) for column in trend.columns] for row in trend.rows]
    lines = [
        f"{series['name']} — {series['expression']}, ordered by {trend.order_key}",
        "",
        render_table(trend.columns, rows),
    ]
    if csv_path is not None:
        lines.extend(["", f"Trend written to {csv_path}"])
    if plot_paths:
        lines.append(f"Plots written: {', '.join(str(path) for path in plot_paths)}")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Render one trend cell for the terminal table."""
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
