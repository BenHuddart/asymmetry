"""``asymmetry integral-scan`` — build and optionally fit an ALC/QLCR scan."""

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
from asymmetry.cli._reduction import add_reduction_arguments, describe, reduction_settings
from asymmetry.cli._runs import resolve_runs
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "integral-scan",
        help="Build an integral-asymmetry scan and optionally fit a field-scan model",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--runs", required=True, help="Run numbers in the scan")
    parser.add_argument("--name", default="integral-scan", help="Stored scan name")
    add_reduction_arguments(parser, background=False)
    parser.add_argument("--tmin", type=float, default=None, help="Integration-window start / µs")
    parser.add_argument("--tmax", type=float, default=None, help="Integration-window end / µs")
    parser.add_argument("--method", choices=["integral", "differential"], default="integral")
    parser.add_argument("--order", choices=["field", "temperature", "run"], default="field")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional field-scan expression, e.g. 'LorentzianLCR + Cubic'",
    )
    parser.add_argument(
        "--initial",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fit start (repeatable)",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Fixed fit value (repeatable)",
    )
    parser.add_argument(
        "--xmin", type=float, default=None, help="Fit only points at or above this x"
    )
    parser.add_argument(
        "--xmax", type=float, default=None, help="Fit only points at or below this x"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        metavar="MODEL",
        help="Fit and subtract this baseline model first",
    )
    parser.add_argument(
        "--baseline-regions",
        default=None,
        metavar="LO:HI,...",
        help="Non-resonant x ranges used by --baseline",
    )
    parser.add_argument("--plot", action="store_true", help="Write plots/<name>.png")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="write into")
    parser.set_defaults(func=run)


def _regions(text: str | None) -> list[tuple[float, float]]:
    if text is None:
        return []
    regions: list[tuple[float, float]] = []
    for token in text.split(","):
        lo_text, separator, hi_text = token.strip().partition(":")
        if not separator:
            raise UserError(f"--baseline-regions entry {token!r} is not LO:HI.")
        try:
            lo, hi = float(lo_text), float(hi_text)
        except ValueError:
            raise UserError(f"--baseline-regions entry {token!r} is not numeric.") from None
        if lo >= hi:
            raise UserError(f"--baseline-regions entry {token!r} must have LO < HI.")
        regions.append((lo, hi))
    return regions


def run(args: argparse.Namespace) -> None:
    from asymmetry.cli import plots
    from asymmetry.core.io import load
    from asymmetry.core.workflow.integral_scan import (
        build_integral_scan,
        field_scan_payload,
        fit_integral_scan,
    )
    from asymmetry.core.workflow.reduction import reduction_source

    if args.baseline is not None and args.baseline_regions is None:
        raise UserError("--baseline requires --baseline-regions LO:HI,...")
    if args.baseline is None and args.baseline_regions is not None:
        raise UserError("--baseline-regions requires --baseline MODEL.")
    fit_only_options = []
    if args.initial:
        fit_only_options.append("--initial")
    if args.fix:
        fit_only_options.append("--fix")
    if args.baseline is not None:
        fit_only_options.extend(["--baseline", "--baseline-regions"])
    if args.xmin is not None or args.xmax is not None:
        fit_only_options.append("--xmin/--xmax")
    if args.model is None and fit_only_options:
        names = ", ".join(dict.fromkeys(fit_only_options))
        raise UserError(f"{names} require --model MODEL.")
    if args.plot:
        plots.require_matplotlib()

    folder = Path(args.folder)
    name = checked_name(args.name, flag="--name")
    workdir, selection = workdir_for(folder, args.workdir, args.instrument)
    targets = resolve_runs(selection, args.runs)

    settings = reduction_settings(args, selection)
    datasets = []
    for run_number, _prefix, path in targets:
        try:
            datasets.append(reduction_source(load(str(path)), settings.period))
        except (TypeError, ValueError) as exc:
            raise UserError(f"Run {run_number}: {exc}") from None
    try:
        scan = build_integral_scan(
            datasets,
            settings,
            t_min=args.tmin,
            t_max=args.tmax,
            method=args.method,
            order_key=args.order,
        )
    except ValueError as exc:
        raise UserError(str(exc)) from None
    if scan.n_points == 0:
        reasons = "; ".join(f"{run}: {reason}" for run, reason in scan.excluded)
        raise UserError(f"No runs contributed to the integral scan. {reasons}")

    fit_scan = scan
    fit_payload = None
    model = None
    if args.model is not None:
        try:
            fit_scan, fit_payload = fit_integral_scan(
                scan,
                args.model,
                initial=parse_fix(args.initial, flag="--initial"),
                fixed=parse_fix(args.fix),
                baseline_model=args.baseline,
                baseline_regions=_regions(args.baseline_regions),
                x_min=args.xmin,
                x_max=args.xmax,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UserError(str(exc)) from None
        from asymmetry.core.fitting.field_scan import as_composite_model

        model = as_composite_model(args.model)

    result_payload = {
        "name": name,
        "folder": str(folder),
        "runs": [run for run, _prefix, _path in targets],
        "settings": settings.to_dict(),
        "t_min": args.tmin,
        "t_max": args.tmax,
        "scan": field_scan_payload(scan),
        "fit_scan": field_scan_payload(fit_scan) if fit_payload is not None else None,
        "fit": fit_payload,
    }
    # The stored scan carries its own path and its plot's, so an agent reading
    # scans/<name>.json back finds the same artefacts the CLI reports. Both are
    # known before either is produced, and the scan is written first so an
    # expensive integral scan survives a failure in the (cheap) plotting step.
    plot_path = workdir.plots_dir / f"{name}.png" if args.plot else None
    result_payload["scan_path"] = str(workdir.scan_path(name))
    result_payload["plot"] = None if plot_path is None else str(plot_path)
    workdir.write_scan(name, result_payload)

    if plot_path is not None:
        plots.plot_scan(
            fit_scan.x,
            fit_scan.value,
            fit_scan.error,
            expression=args.model,
            model_function=None if model is None else model.function,
            parameters=None if fit_payload is None else fit_payload["parameters"],
            x_label=fit_scan.x_label,
            y_label=fit_scan.y_label,
            title=name,
            out_path=plot_path,
        )

    if args.json:
        emit_json(payload(**result_payload))
        return
    print(_render(result_payload, settings))


def _render(result: dict, settings) -> str:
    points = result["scan"]["points"]
    rows = [
        [
            str(point["run"]),
            format_number(point["x"], 3),
            format_number(point["value"], 6),
            format_number(point["error"], 6),
        ]
        for point in points
    ]
    lines = [
        f"{result['name']} — {len(points)} point(s), ordered by {result['scan']['order_key']}",
        "",
        render_table(["run", "x", "integral A", "error"], rows),
        "",
        describe(settings),
    ]
    if result["fit"] is not None:
        fit = result["fit"]
        # A failed fit is reported, not raised: the scan is worth keeping, and
        # where the parameters ended up says which component ran away.
        verdict = "" if fit["success"] else f" — FAILED ({fit['message'] or 'no message'})"
        lines.extend(
            [
                f"fit: {fit['expression']}, chi2_red "
                f"{format_number(fit['reduced_chi_squared'], 3)}{verdict}",
                "parameters: "
                + ", ".join(
                    f"{name}={format_number(value, 6)}" for name, value in fit["parameters"].items()
                ),
            ]
        )
    lines.append(f"Scan written to {result['scan_path']}")
    if result["plot"] is not None:
        lines.append(f"Plot written to {result['plot']}")
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
