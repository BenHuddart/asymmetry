"""``asymmetry integral-scan`` — build and optionally fit an ALC/QLCR scan."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
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
from asymmetry.cli._reduction import (
    add_reduction_arguments,
    deadtime_note,
    describe,
    reduction_settings,
)
from asymmetry.cli._runs import range_text, resolve_runs
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "integral-scan",
        help="Build an integral-asymmetry scan and optionally fit a field-scan model",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--runs", required=True, help="Run numbers in the scan")
    parser.add_argument("--name", default="integral-scan", help="Stored scan name")
    add_reduction_arguments(parser)
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
    from asymmetry.core.io.periods import period_count
    from asymmetry.core.workflow.integral_scan import (
        build_integral_scan,
        contradicted_tf_stamps,
        field_scan_payload,
        fit_integral_scan,
        period_field_offset_gauss,
    )
    from asymmetry.core.workflow.reduction import GREEN_MINUS_RED, reduction_source

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

    # The red period's field less the green's, from the Hall probe each run logged.
    offset = (
        period_field_offset_gauss([dataset.run for dataset in datasets])
        if settings.period == GREEN_MINUS_RED
        else None
    )

    fit_scan = scan
    fit_payload = None
    model = None
    fixed = parse_fix(args.fix)
    if args.model is not None:
        try:
            fit_scan, fit_payload = fit_integral_scan(
                scan,
                args.model,
                initial=parse_fix(args.initial, flag="--initial"),
                fixed=fixed,
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
        "period_field_offset": (
            None if offset is None else {"gauss": offset[0], "runs": offset[1]}
        ),
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

    offset_names = (
        []
        if model is None
        else [
            model.component_param_name(index, "dB")
            for index, component in enumerate(model.components)
            if "dB" in component.param_names
        ]
    )
    free_offsets = [name for name in offset_names if name not in fixed]
    # Without --period a two-period run reduces as its periods summed.
    summed = (
        []
        if settings.period is not None
        else [
            run_number
            for (run_number, _prefix, _path), dataset in zip(targets, datasets, strict=True)
            if period_count(dataset) == 2
        ]
    )
    from asymmetry.core.workflow.survey import run_geometry

    stamped_tf = {
        run_number: abs(float(dataset.metadata["field"]))
        for (run_number, _prefix, _path), dataset in zip(targets, datasets, strict=True)
        if run_geometry(dataset.metadata) == "TF"
    }
    stored = {
        path.stem: stored_fit
        for path in sorted(workdir.scans_dir.glob("*.json"))
        if path.stem != name
        and (stored_fit := json.loads(path.read_text(encoding="utf-8"))["fit"]) is not None
    }
    elsewhere = [
        value
        for stored_fit in stored.values()
        if stored_fit["success"]
        for parameter, value in stored_fit["parameters"].items()
        if parameter.split("_")[0] == "B0"
    ]
    notes = _notes(
        result_payload, free_offsets, summed, contradicted_tf_stamps(scan, stamped_tf), elsewhere
    )
    if fit_payload is not None:
        notes.extend(_line_comparisons(fit_payload, stored))
    from asymmetry.core.workflow.survey import has_file_deadtime

    notes.extend(
        deadtime_note(settings, any(has_file_deadtime(dataset.run) for dataset in datasets))
    )
    if args.json:
        emit_json(payload(**result_payload, notes=notes))
        return
    print(_render(result_payload, settings, notes))


def _render(result: dict, settings, notes: list[str]) -> str:
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
    offset = result["period_field_offset"]
    if offset is not None:
        lines.append(
            f"period field offset (red - green): {format_number(offset['gauss'], 2)} G, "
            f"mean of {offset['runs']} run(s)"
        )
    if result["fit"] is not None:
        fit = result["fit"]
        # A failed fit is reported, not raised: the scan is worth keeping, and
        # where the parameters ended up says which component ran away.
        at_bound = (
            f"; at a bound: {', '.join(fit['params_at_bound'])}" if fit["params_at_bound"] else ""
        )
        verdict = "" if fit["success"] else f" — FAILED ({fit['message']}{at_bound})"
        from asymmetry.core.fitting.field_scan import as_composite_model

        model = as_composite_model(fit["expression"])
        # Fields and couplings carry their unit; an amplitude's stored unit is
        # not that of an integral asymmetry, so it prints none.
        units = {
            model.component_param_name(index, local): info.unit
            for index, component in enumerate(model.components)
            for local, info in component.param_info.items()
            if info.unit in ("G", "MHz")
        }
        # A held parameter has no error; one pinned on a bound is not determined.
        rows = [
            [
                name,
                format_number(value, 6),
                units.get(name, ""),
                (
                    "fixed"
                    if name in fit["fixed"]
                    else format_number(fit["uncertainties"][name], 6)
                    if fit["success"]
                    else "-"
                )
                + (" (at bound)" if name in fit["params_at_bound"] else ""),
            ]
            for name, value in fit["parameters"].items()
        ]
        lines.extend(
            [
                f"fit: {fit['expression']}, chi2_red "
                f"{format_number(fit['reduced_chi_squared'], 3)}{verdict}",
                render_table(["parameter", "value", "unit", "error"], rows),
            ]
        )
    lines.extend(notes)
    lines.append(f"Scan written to {result['scan_path']}")
    if result["plot"] is not None:
        lines.append(f"Plot written to {result['plot']}")
    return "\n".join(lines)


def _notes(
    result: dict,
    free_offsets: list[str],
    summed: list[int],
    longitudinal: list[int],
    elsewhere: list[float],
) -> list[str]:
    """Every NOTE and Next line the scan calls for — printed, and kept in --json.

    *elsewhere* holds the line centres other stored scans in the work directory fitted.
    """
    fit = result["fit"]
    offset = result["period_field_offset"]
    lines: list[str] = []
    repeats = _repeated_points(result["scan"]["points"])
    if repeats:
        lines.append(
            f"NOTE: the scan measures {result['scan']['order_key']} "
            + "; ".join(
                f"{x:g} more than once (runs {', '.join(str(p['run']) for p in group)}: "
                + ("they differ by more than three errors" if differ else "they agree")
                + ")"
                for x, group, differ in repeats
            )
            + ". Say so: a return pass or repeated points, and whether the curve came back to "
            "itself."
        )
    if longitudinal:
        lines.append(
            f"NOTE: the files stamp {range_text(longitudinal)} TF, but at fields of a kilogauss "
            f"and more a transverse field precesses the polarisation through many periods "
            f"within the window, so its integral asymmetry would sit near zero; these keep a "
            f"large one. They are longitudinal and the stamp is wrong: keep them in the scan, "
            f"and say so — with this reason — in the summary."
        )
    if offset is not None and free_offsets:
        # A differential pair's dB is the green field less the red: the offset, negated.
        fixes = " ".join(
            f"--fix {name}={format_number(-offset['gauss'], 2)}" for name in free_offsets
        )
        lines.append(
            f"Next: the red period sat {format_number(-offset['gauss'], 2)} G below the green; "
            f"with the pair offset free the fit is degenerate, so refit with {fixes}."
        )
    if fit is not None and fit["resonance_windows"]:
        lines.append(_failed_fit_next(fit))
    elif fit is not None:
        lines.extend(_poor_fit_note(fit, elsewhere))
    if summed:
        lines.append(
            f"NOTE: {range_text(summed)} are two-period (red/green) runs, and without --period "
            f"this scan summed both periods, blurring the red/green contrast they were taken "
            f"for. Measure their difference: rerun with --period green-red; with a field step "
            f"between the periods (differential ALC) fit --model LorentzianLCRPair with its dB "
            f"held at the value that command's Next line gives (the printed red - green "
            f"offset, negated)."
        )
    if fit is not None and any(
        term.strip() in ("LorentzianLCR", "GaussianLCR") for term in fit["expression"].split("+")
    ):
        lines.append(
            "NOTE: no radical ALC or hyperfine model is available: these resonance fields "
            "are not converted into muon or proton couplings or site assignments — say so, "
            "rather than only that none were quoted."
        )
    return lines


#: A converged fit this far above its errors has left structure unfitted.
_POOR_SCAN_FIT = 2.0

#: A windowed line's amplitude this many errors from zero is a resonance, not
#: noise, when its fit stays within this chi2_red: a background step read as a
#: dip leaves the residuals far above their errors.
_RESOLVED_DEPTH = 5.0
_RESOLVED_FIT = 4.0

#: Centres or widths closer than this many combined errors are the same.
_DISTINCT_ERRORS = 2.0

#: Half-widths of a window around a fitted line: room for both flanks and some
#: background on each side.
_WINDOW_WIDTHS = 4.0


def _resolved_lines(fit: dict, fit_limit: float = _RESOLVED_FIT) -> list[str]:
    """The ``B0`` of each windowed line with both flanks in range, deep, fitted within *fit_limit*."""
    from asymmetry.core.workflow.integral_scan import DIP_FLANK_WIDTHS

    if fit["x_min"] is None and fit["x_max"] is None:
        return []
    if not fit["success"] or fit["params_at_bound"] or fit["reduced_chi_squared"] > fit_limit:
        return []
    low, high = fit["x_range"]
    resolved = []
    for name, centre in fit["parameters"].items():
        if name.split("_")[0] != "B0":
            continue
        width = abs(fit["parameters"][name.replace("B0", "Bwid", 1)])
        amplitude = name.replace("B0", "f", 1)
        if (
            low <= centre - DIP_FLANK_WIDTHS * width
            and centre + DIP_FLANK_WIDTHS * width <= high
            and abs(fit["parameters"][amplitude])
            >= _RESOLVED_DEPTH * fit["uncertainties"][amplitude]
        ):
            resolved.append(name)
    return resolved


def _line_comparisons(fit: dict, stored: dict[str, dict]) -> list[str]:
    """Each resolved line beside the same line resolved in another stored scan, with directions."""

    def described(scan_fit: dict, line: str) -> tuple[float, float, float, float]:
        """Centre, error, width, error — errors scaled by √chi2_red where it exceeds 1."""
        width = line.replace("B0", "Bwid", 1)
        scale = math.sqrt(max(scan_fit["reduced_chi_squared"], 1.0))
        return (
            scan_fit["parameters"][line],
            scale * scan_fit["uncertainties"][line],
            abs(scan_fit["parameters"][width]),
            scale * scan_fit["uncertainties"][width],
        )

    def direction(a: float, a_err: float, b: float, b_err: float, words: tuple[str, ...]) -> str:
        if abs(a - b) <= _DISTINCT_ERRORS * math.hypot(a_err, b_err):
            return words[2]
        return words[0] if a > b else words[1]

    notes = []
    # A poor chi2_red only widens the errors here: the comparison is of two lines
    # each fitted with both flanks, not a verdict that either is a resonance.
    for line in _resolved_lines(fit, math.inf):
        centre, centre_err, width, width_err = described(fit, line)
        for other, other_fit in stored.items():
            for other_line in _resolved_lines(other_fit, math.inf):
                c2, c2_err, w2, w2_err = described(other_fit, other_line)
                if abs(centre - c2) > width + w2:
                    continue
                notes.append(
                    f"COMPARE: this line at {centre:g} ± {centre_err:.3g} (width {width:g} ± "
                    f"{width_err:.3g}) and scan {other}'s at {c2:g} ± {c2_err:.3g} (width {w2:g} ± "
                    f"{w2_err:.3g}), errors scaled by √chi2_red, are one line in two scans: this "
                    f"one is "
                    + direction(
                        width,
                        width_err,
                        w2,
                        w2_err,
                        ("broader", "narrower", "of a width these errors cannot tell apart"),
                    )
                    + " and "
                    + direction(
                        centre,
                        centre_err,
                        c2,
                        c2_err,
                        (
                            "higher in field",
                            "lower in field",
                            "at a field these errors cannot tell apart",
                        ),
                    )
                    + ". Report both and that direction — a width or field changing between "
                    "conditions is the physics (motional narrowing, a changing coupling)."
                )
    return notes


def _repeated_points(points: list[dict]) -> list[tuple[float, list[dict], bool]]:
    """``(x, points, differ)`` for each x measured more than once, *differ* past three errors."""
    groups: dict[float, list[dict]] = {}
    for point in points:
        groups.setdefault(point["x"], []).append(point)
    return [
        (
            x,
            group,
            any(
                abs(a["value"] - b["value"]) > _DISTINCT_REPEAT * math.hypot(a["error"], b["error"])
                for a in group
                for b in group
            ),
        )
        for x, group in sorted(groups.items())
        if len(group) > 1
    ]


#: Repeated points further apart than this many combined errors did not come back.
_DISTINCT_REPEAT = 3.0


def _poor_fit_note(fit: dict, elsewhere: Sequence[float] = ()) -> list[str]:
    """Notes on what a converged resonance fit left out or cannot vouch for.

    A dip another stored scan already fitted (a centre in *elsewhere*) is not announced again.
    """
    from asymmetry.core.workflow.integral_scan import DIP_FLANK_WIDTHS

    lines = {name: value for name, value in fit["parameters"].items() if name.split("_")[0] == "B0"}
    if not lines:
        return []
    notes = []
    low, high = fit["x_range"]
    # Only a chosen --xmin/--xmax window can cut a line's flank off; a whole
    # scan narrower than its line just leaves the width unmeasured.
    windowed = fit["x_min"] is not None or fit["x_max"] is not None
    resolved = _resolved_lines(fit)
    for name, centre in lines.items() if windowed else ():
        width = abs(fit["parameters"][name.replace("B0", "Bwid", 1)])
        if centre - DIP_FLANK_WIDTHS * width < low or centre + DIP_FLANK_WIDTHS * width > high:
            notes.append(
                f"NOTE: the line at {centre:g} (width {width:g}) runs off the fitted range "
                f"{low:g}–{high:g}: without data rising again on both sides it may be a step "
                f"or the background's edge, not a resonance — or a broad shape stretched over "
                f"a narrower dip. Look at the plot: refit a narrower window around the dip's "
                f"minimum, or a wider one if the dip itself runs to the edge, before reporting "
                f"it."
            )
        elif name in resolved:
            amplitude = name.replace("B0", "f", 1)
            depth = abs(fit["parameters"][amplitude]) / fit["uncertainties"][amplitude]
            notes.append(
                f"RESONANCE: the line at {centre:g} ± {fit['uncertainties'][name]:g} (width "
                f"{width:g}) sits inside the window {low:g}–{high:g} with data on both flanks, "
                f"its amplitude {depth:.1f} errors from zero: a resolved resonance — report "
                f"its centre and width"
                + (
                    f", with errors understated by the chi2_red of "
                    f"{format_number(fit['reduced_chi_squared'], 3)}."
                    if fit["reduced_chi_squared"] > _POOR_SCAN_FIT
                    else "."
                )
            )
    unfitted = [
        window
        for window in fit["next_dip_windows"]
        if not any(
            window["x_min"] <= centre <= window["x_max"] for centre in [*lines.values(), *elsewhere]
        )
    ]
    if unfitted:
        notes.append(
            "NOTE: the scan holds another dip this fit does not include, in "
            + "; ".join(f"{w['x_min']:g}–{w['x_max']:g}" for w in unfitted)
            + ": fit it on its own local background — "
            + "; ".join(
                f"--model 'LorentzianLCR + Linear' --xmin {w['x_min']:g} --xmax {w['x_max']:g}"
                for w in unfitted
            )
            + " — and report every dip the scan shows."
        )
    if fit["reduced_chi_squared"] > _POOR_SCAN_FIT and len(resolved) < len(lines):
        notes.append(
            f"NOTE: the fit converged at chi2_red {format_number(fit['reduced_chi_squared'], 3)}: "
            "over a long range the background may rise or step where no polynomial can "
            "follow — a fit that cannot is not a result, and the summary should say that is "
            "why — or the range holds more dips than the model. Look at the plot (--plot) "
            "and fit one resonance per --xmin/--xmax window on its own local background"
            + (
                ": "
                + "; ".join(
                    f"--model 'LorentzianLCR + Linear' --xmin "
                    f"{max(low, centre - _WINDOW_WIDTHS * width):.0f}"
                    f" --xmax {min(high, centre + _WINDOW_WIDTHS * width):.0f}"
                    for centre, width in (
                        (centre, abs(fit["parameters"][name.replace("B0", "Bwid", 1)]))
                        for name, centre in lines.items()
                    )
                )
                + "."
                if not windowed
                else "."
            )
        )
    return notes


def _failed_fit_next(fit: dict) -> str:
    """What to try after a failed resonance fit: the scan's own dips, and a window each."""
    windows = fit["resonance_windows"]
    dips = ", ".join(f"{window['parameter']} {window['centre']:g}" for window in windows)
    # A hand-given centre outside its dip's window started on another feature.
    moved = [
        window
        for window in windows
        if window["parameter"] in fit["initial"]
        and not window["x_min"] <= fit["initial"][window["parameter"]] <= window["x_max"]
    ]
    text = f"Next: the scan's own largest dips are at {dips}. "
    if moved:
        starts = " ".join(f"--initial {w['parameter']}={w['centre']:g}" for w in windows)
        text += f"The fit started away from them: refit with {starts}, or "
    else:
        text += (
            "The fit already started each centre there, so the usual cause is the "
            "background: across a long scan it rises or steps where no polynomial can "
            "follow, and the resonances cannot be fitted on it together — say so, and "
        )
    return (
        text
        + "fit one resonance per window on its own local background: "
        + "; ".join(
            f"--model '{window['component']} + Linear' --xmin {window['x_min']:g} "
            f"--xmax {window['x_max']:g}"
            for window in windows
        )
        + "."
    )


__all__ = ["add_parser", "run"]
