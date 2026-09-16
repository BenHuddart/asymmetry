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
from asymmetry.cli._runs import resolve_run, resolve_runs
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "integral-scan",
        help="Build an integral-asymmetry scan and optionally fit a field-scan model",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--runs", required=True, help="Run numbers in the scan")
    parser.add_argument("--name", default="integral-scan", help="Stored scan name")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed detector balance")
    parser.add_argument(
        "--alpha-from", type=int, default=None, dest="alpha_from", help="Estimate alpha on this run"
    )
    parser.add_argument(
        "--period", default=None, metavar="RED|GREEN|N", help="Select one acquisition period"
    )
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


def _selected(load_result, period: str | None, run_number: int):
    from asymmetry.core.io.periods import select_period

    try:
        if period is not None:
            return select_period(load_result, period)
        return load_result[0] if isinstance(load_result, list) else load_result
    except (TypeError, ValueError) as exc:
        raise UserError(f"Run {run_number}: {exc}") from None


def run(args: argparse.Namespace) -> None:
    from asymmetry.cli import plots
    from asymmetry.core.io import load
    from asymmetry.core.workflow.integral_scan import (
        build_integral_scan,
        field_scan_payload,
        fit_integral_scan,
    )
    from asymmetry.core.workflow.reduction import estimate_alpha_for_run

    if args.alpha is not None and args.alpha_from is not None:
        raise UserError("Pass either --alpha or --alpha-from, not both.")
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
    if args.model is None and fit_only_options:
        names = ", ".join(dict.fromkeys(fit_only_options))
        raise UserError(f"{names} require --model MODEL.")
    if args.plot:
        plots.require_matplotlib()

    folder = Path(args.folder)
    targets = resolve_runs(folder, args.runs)
    name = checked_name(args.name, flag="--name")
    workdir = workdir_for(folder, args.workdir)

    if args.alpha_from is not None:
        path = resolve_run(folder, args.alpha_from)
        calibration = _selected(load(str(path)), args.period, args.alpha_from)
        alpha = estimate_alpha_for_run(calibration.run).alpha
        alpha_source = f"estimated:{args.alpha_from}"
    elif args.alpha is not None:
        alpha, alpha_source = float(args.alpha), "user"
    else:
        alpha, alpha_source = 1.0, "assumed"

    datasets = [
        _selected(load(str(path)), args.period, run_number) for run_number, _prefix, path in targets
    ]
    try:
        scan = build_integral_scan(
            datasets,
            alpha=alpha,
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
                initial=parse_fix(args.initial),
                fixed=parse_fix(args.fix),
                baseline_model=args.baseline,
                baseline_regions=_regions(args.baseline_regions),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UserError(str(exc)) from None
        if not fit_payload["success"]:
            raise UserError(
                f"Integral-scan fit failed: {fit_payload['message'] or 'unknown error'}"
            )
        from asymmetry.core.fitting.field_scan import as_composite_model

        model = as_composite_model(args.model)

    result_payload = {
        "name": name,
        "folder": str(folder),
        "runs": [run for run, _prefix, _path in targets],
        "alpha": alpha,
        "alpha_source": alpha_source,
        "period": args.period,
        "t_min": args.tmin,
        "t_max": args.tmax,
        "scan": field_scan_payload(scan),
        "fit_scan": field_scan_payload(fit_scan) if fit_payload is not None else None,
        "fit": fit_payload,
    }
    scan_path = workdir.write_scan(name, result_payload)

    plot_path = None
    if args.plot:
        plot_path = plots.plot_scan(
            fit_scan.x,
            fit_scan.value,
            fit_scan.error,
            expression=args.model,
            model_function=None if model is None else model.function,
            parameters=None if fit_payload is None else fit_payload["parameters"],
            x_label=fit_scan.x_label,
            y_label=fit_scan.y_label,
            title=name,
            out_path=workdir.plots_dir / f"{name}.png",
        )

    result_payload["scan_path"] = str(scan_path)
    result_payload["plot"] = None if plot_path is None else str(plot_path)
    if args.json:
        emit_json(payload(**result_payload))
        return
    print(_render(result_payload))


def _render(result: dict) -> str:
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
        f"alpha {result['alpha']:.4f} ({result['alpha_source']}), period {result['period'] or 'default'}",
    ]
    if result["fit"] is not None:
        fit = result["fit"]
        lines.extend(
            [
                f"fit: {fit['expression']}, chi2_red {format_number(fit['reduced_chi_squared'], 3)}",
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
