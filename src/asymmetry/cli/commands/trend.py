"""``asymmetry trend`` — the parameter trend of a stored series, optionally fitted.

With ``--from-fits`` it first builds that series: one stored law's parameter
from each of several series, against their temperature or supplied values.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asymmetry.cli._axis import MEMBER_ORDER_DEFAULT, add_axis_arguments, member_axis
from asymmetry.cli._output import (
    UserError,
    checked_name,
    emit_json,
    format_number,
    payload,
    render_table,
    render_trend,
)
from asymmetry.cli._recipes import parse_fix
from asymmetry.cli._runs import parse_run_spec, reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``trend`` subcommand."""
    parser = subparsers.add_parser(
        "trend",
        help="Print, export or fit the parameter trend of a stored series",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument(
        "--series",
        required=True,
        help="Name the series was stored under (with --from-fits, the name to store it under)",
    )
    parser.add_argument(
        "--from-fits",
        default=None,
        metavar="SERIES,...",
        help=(
            "Build --series from these stored series: each one's trend-fit parameter "
            "--param (e.g. a rate constant fitted per temperature), against --order"
        ),
    )
    parser.add_argument(
        "--fit",
        default=None,
        metavar="PARAM[:EXPR]",
        help=(
            "With --from-fits, the members' trend fit to read, by the column it was "
            "fitted to (or column:expression); needed only when a member holds several"
        ),
    )
    add_axis_arguments(
        parser,
        default=MEMBER_ORDER_DEFAULT,
        noun="series",
        plural="--from-fits series",
        example="scan-1=0,scan-2=0.25",
    )
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
    parser.add_argument(
        "--param",
        default=None,
        help=(
            "The trend column --model is fitted to; with --from-fits, the members' "
            "trend-fit parameter the built series tabulates (and --model fits)"
        ),
    )
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
        metavar="KEYS",
        help=(
            "Leave these rows out of the fit — runs, or member series for a trend built "
            "from other series (every other row with a value enters)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Read (or build) the stored series, report its trend table and fit it when asked."""
    from asymmetry.core.workflow.series import TrendTable
    from asymmetry.core.workflow.workdir import DERIVED_SERIES_KINDS

    if args.from_fits is None and (
        args.fit is not None or args.x is not None or args.order != MEMBER_ORDER_DEFAULT
    ):
        raise UserError("--fit, --order and --x build a trend --from-fits.")
    if args.from_fits is not None and args.param is None:
        raise UserError("--from-fits needs --param NAME, the trend-fit parameter to tabulate.")
    fit_options = [
        flag
        for flag, value in (
            ("--param", args.param if args.from_fits is None else None),
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

    workdir, _selection = workdir_for(Path(args.folder), args.workdir, args.instrument)
    name = checked_name(args.series, flag="--series")
    series = (
        _stored_series(workdir, name)
        if args.from_fits is None
        else _series_from_fits(args, workdir, name)
    )
    trend = TrendTable(**series["trend"])

    fit = None
    if args.model is not None:
        from asymmetry.core.workflow.trend_fit import fit_trend, trend_fit_key

        if args.exclude is None:
            exclude: list[str] = []
        elif series["kind"] in DERIVED_SERIES_KINDS:
            exclude = [key.strip() for key in args.exclude.split(",")]
        else:
            exclude = [str(run) for run in parse_run_spec(args.exclude)]
        try:
            fit = fit_trend(
                trend,
                args.param,
                args.model,
                x_min=args.xmin,
                x_max=args.xmax,
                initial=parse_fix(args.initial, flag="--initial"),
                fixed=parse_fix(args.fix),
                exclude=exclude,
            ).to_dict()
        except ValueError as exc:
            raise UserError(str(exc)) from None
        series["trend_fits"][trend_fit_key(args.param, args.model)] = fit
    if args.from_fits is not None or fit is not None:
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


def _stored_series(workdir, name: str) -> dict[str, Any]:
    """The series stored as *name*; a user error naming what is stored when it cannot be read."""
    stored = workdir.series_names()
    if name not in stored:
        known = ", ".join(stored) if stored else "none yet — run 'asymmetry fit-series' first"
        raise UserError(f"No series {name!r} in {workdir.series_dir} (it holds: {known}).")
    try:
        return workdir.read_series(name)
    except KeyError as exc:
        raise UserError(exc.args[0]) from None


def _series_from_fits(args: argparse.Namespace, workdir, name: str) -> dict[str, Any]:
    """The ``fit-trend`` series ``--from-fits`` names: one row per member series."""
    from asymmetry.core.workflow.trend_fit import fit_trend_table
    from asymmetry.core.workflow.workdir import DERIVED_SERIES_KINDS, series_digest

    names = list(dict.fromkeys(item.strip() for item in args.from_fits.split(",") if item.strip()))
    if name in names:
        raise UserError(f"--series {name} is the trend being built; it cannot be a member too.")
    members = {member: _stored_series(workdir, member) for member in names}
    fit_keys = {member: _chosen_fit(member, members[member], args.fit) for member in names}
    if args.x is None:
        runless = [member for member in names if members[member]["kind"] in DERIVED_SERIES_KINDS]
        if runless:
            raise UserError(
                f"{', '.join(runless)} hold no runs to read {args.order} from; give every "
                f"member's value with --order NAME --x SERIES=VALUE,..."
            )
        groups = {
            member: reduced_datasets(
                workdir, [result["run"] for result in members[member]["results"]]
            )
            for member in names
        }
    else:
        # Supplied values are keyed by member alone; no run is read.
        groups = {member: {} for member in names}
    axis = member_axis(args.order, args.x, groups, key=str, flag="--x", noun="series")
    try:
        trend = fit_trend_table(
            {member: members[member]["trend_fits"][fit_keys[member]] for member in names},
            args.param,
            axis,
        )
    except ValueError as exc:
        raise UserError(str(exc)) from None
    return {
        "name": name,
        "kind": "fit-trend",
        "expression": f"{args.param} of {', '.join(sorted(set(fit_keys.values())))}",
        "order_key": axis.name,
        "free_params": [args.param],
        "members": [
            {
                "series": member,
                "fit": fit_keys[member],
                "digest": series_digest(members[member], fit_keys[member]),
            }
            for member in names
        ],
        "trend": trend.to_dict(),
        "trend_fits": {},
    }


def _chosen_fit(member: str, series: dict[str, Any], choice: str | None) -> str:
    """The key of *member*'s trend fit ``--fit`` names: its only one when *choice* is None."""
    fits = series["trend_fits"]
    matching = [key for key, fit in fits.items() if choice is None or choice in (key, fit["param"])]
    if len(matching) == 1:
        return matching[0]
    held = ", ".join(fits) if fits else "none"
    if not fits:
        advice = f"fit a law to it first (trend --series {member} --model EXPR --param NAME)"
    elif matching:
        advice = "name one with --fit PARAM:EXPR"
    else:
        advice = f"none is --fit {choice}"
    raise UserError(f"Series {member} holds trend fits: {held}; {advice}.")


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
    fitted_x = [row["x"] for row in trend.rows if row["key"] in fit["keys"]]
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
    lines = [
        f"{series['name']} — {series['expression']}, ordered by {trend.order_key}",
        "",
        render_trend(trend),
    ]
    from asymmetry.core.workflow.series import envelope_change

    # What the trend itself shows is printed whether or not a law is fitted
    # to it: a law answers one question, these name the others.
    change = envelope_change(trend)
    readings = [
        *([change] if change is not None else []),
        *_rate_steps(trend, series["free_params"]),
        *_phase_drift(trend, series["free_params"]),
        *_frequency_response(series, trend),
    ]
    if fit is not None:
        lines.extend(
            [
                "",
                *_render_fit(
                    fit,
                    series["free_params"],
                    _gap_law_steps(series["name"], trend, fit, plot_paths),
                ),
                *readings,
            ]
        )
    else:
        lines.extend(
            [
                "",
                *readings,
                *_law_hints(series["name"], trend, series["free_params"]),
                *_doublet_hint(series, trend),
                *_weight_shift(series, trend),
            ]
        )
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


#: A frequency whose values span less than this fraction of their median sits
#: at a fixed field (the applied one) along the scan: not an order parameter.
_HELD_FREQUENCY_SPREAD = 0.10


def _held_frequency(trend, param: str) -> float | None:
    """The median of *param* when it holds steady along the scan, else ``None``."""
    import statistics

    values = [
        row[param]
        for row in trend.rows
        if row[param] is not None and not {"failed", "frequency_unresolved"} & set(row["flags"])
    ]
    if len(values) < 2:
        return None
    median = statistics.median(values)
    return median if (max(values) - min(values)) < _HELD_FREQUENCY_SPREAD * abs(median) else None


#: A width or rate whose two blocks along the scan differ by more than this
#: many combined errors has changed there, however small the change looks.
_STEP_SIGNIFICANCE = 5.0

#: A row further than this many combined errors from an end's level has left it.
_LEVEL_DEPARTURE = 3.0


#: Flags that say a row's values are not a measurement (``large_rel_err`` is
#: carried by its own error bar and stays).
_UNRELIABLE = frozenset(
    {
        "failed",
        "frequency_unresolved",
        "bound_pinned",
        "spurious_reseeded",
        "amplitude_exceeds_data",
    }
)


def _measured(trend, *params: str) -> list[dict[str, Any]]:
    """The rows, in scan order, that fitted every one of *params* reliably, with an error."""
    return sorted(
        (
            row
            for row in trend.rows
            if all(row[param] is not None and row[f"{param}_err"] for param in params)
            and not _UNRELIABLE & set(row["flags"])
        ),
        key=lambda row: row["x"],
    )


def _weighted_mean(rows, param: str) -> tuple[float, float]:
    """The error-weighted mean of *param* over *rows*, and its error."""
    weights = [1.0 / row[f"{param}_err"] ** 2 for row in rows]
    mean = sum(w * row[param] for w, row in zip(weights, rows)) / sum(weights)
    return mean, 1.0 / math.sqrt(sum(weights))


def _held_level(block: list[dict[str, Any]], param: str) -> tuple[float, list[dict[str, Any]]]:
    """The level at *block*'s far end, and the rows from that end that still hold it.

    *block* runs from an end of the scan inward. The level is the weighted mean
    of its outer half (two rows at least); walking inward, the first row further
    from it than ``_LEVEL_DEPARTURE`` combined errors has left it.
    """
    outer = max(2, len(block) // 2)
    level, error = _weighted_mean(block[:outer], param)
    held = next(
        (
            index
            for index in range(outer, len(block))
            if abs(block[index][param] - level)
            > _LEVEL_DEPARTURE * math.hypot(block[index][f"{param}_err"], error)
        ),
        len(block),
    )
    return level, block[:held]


def _rate_steps(trend, free_params: list[str]) -> list[str]:
    """A note per width or rate that steps along the scan, with where it leaves each level.

    The step is the split of the scan into two contiguous blocks (two runs or
    more each) whose error-weighted means differ most, in combined errors. Its
    place is bracketed by where the parameter leaves each end's level (see
    :func:`_held_level`), so a gradual change is not reported as a midpoint.
    """
    notes = []
    axis = trend.order_key
    # An amplitude steps along a supplied or field axis (a range curve, a
    # steering scan); along temperature its changes are the physics' own.
    stepping = _RATE_BASES | (
        set() if trend.order_key in ("temperature", "sample_temperature_logged") else {"A"}
    )
    for param in free_params:
        if re.sub(r"_\d+$", "", param) not in stepping:
            continue
        rows = _measured(trend, param)
        splits = []
        for k in range(2, len(rows) - 1):
            (low, low_err), (high, high_err) = (
                _weighted_mean(rows[:k], param),
                _weighted_mean(rows[k:], param),
            )
            splits.append((abs(high - low) / math.hypot(low_err, high_err), k))
        if not splits:
            continue
        significance, k = max(splits)
        if significance <= _STEP_SIGNIFICANCE:
            continue
        low_level, low_rows = _held_level(rows[:k], param)
        high_level, high_rows = _held_level(rows[k:][::-1], param)
        low_edge, high_edge = low_rows[-1]["x"], high_rows[-1]["x"]
        notes.append(
            f"NOTE: {param} changes along the scan ({significance:.0f}x the combined error "
            f"between the weighted means of its two sides). It leaves its low-{axis} level "
            f"({format_number(low_level, 4)} over {low_rows[0]['x']:g}–{low_edge:g}) above "
            f"{low_edge:g} and its high-{axis} level ({format_number(high_level, 4)} over "
            f"{high_edge:g}–{high_rows[0]['x']:g}) below {high_edge:g}: the change lies between "
            f"{low_edge:g} and {high_edge:g}, and an onset is where it leaves a level, not a "
            f"midpoint. Report it and that span: a small step in a width or rate is often the "
            f"physics (a transition, an onset), even when the parameter you expected to move "
            f"did not."
        )
    return notes


#: A phase whose straight line in field explains less of its variation than
#: this is not a timing offset (φ = 2π f Δt is exactly linear in f).
_LINEAR_SHARE = 0.9


#: A timing offset is a few bins: a phase slope implying more than this (µs)
#: is another effect, not t0.
_MAX_T0_OFFSET_US = 1.0


def _phase_drift(trend, free_params: list[str]) -> list[str]:
    """The t0 note for a phase that runs linearly with field along a field scan."""
    from asymmetry.core.fitting.spectral import field_gauss_to_frequency_mhz

    if trend.order_key != "field":
        return []
    offsets: dict[str, float] = {}
    for param in (p for p in free_params if re.sub(r"_\d+$", "", p) == "phase"):
        # The line's own fitted frequency (same component suffix) sets how fast
        # its phase runs; without one, the bare muon's γ times the field does.
        line = "frequency" + param.removeprefix("phase")
        own_line = line in free_params
        rows = _measured(trend, param, *([line] if own_line else []))
        if len(rows) < 3:
            continue
        f = [row[line] if own_line else field_gauss_to_frequency_mhz(row["x"]) for row in rows]
        weights = [1.0 / row[f"{param}_err"] ** 2 for row in rows]
        f_mean = sum(w * x for w, x in zip(weights, f)) / sum(weights)
        y_mean = sum(w * row[param] for w, row in zip(weights, rows)) / sum(weights)
        spread = sum(w * (x - f_mean) ** 2 for w, x in zip(weights, f))
        slope = (
            sum(w * (x - f_mean) * (row[param] - y_mean) for w, x, row in zip(weights, f, rows))
            / spread
        )
        scatter = sum(w * (row[param] - y_mean) ** 2 for w, row in zip(weights, rows))
        # φ = φ0 - 2π f Δt for a signal arriving Δt (µs) after t0: slope -2π Δt
        # in rad/MHz. |slope| / its error 1/sqrt(spread) is the significance.
        offset = -slope / (2.0 * math.pi)
        if (
            abs(slope) * math.sqrt(spread) > _STEP_SIGNIFICANCE
            and slope**2 * spread >= _LINEAR_SHARE * scatter
            and abs(offset) <= _MAX_T0_OFFSET_US
        ):
            offsets[param] = offset
    if not offsets:
        return []
    late = next(iter(offsets.values())) > 0.0
    verb = "changes" if len(offsets) == 1 else "change"
    return [
        f"NOTE: {', '.join(offsets)} {verb} linearly with field along the scan (Δt = "
        + ", ".join(f"{1e3 * offset:+.3g} ns" for offset in offsets.values())
        + " from the slope). A phase that runs linearly with field — that is, with frequency, "
        f"φ = 2π f Δt — is a timing offset between t0 and the muons' arrival, not the sample. "
        f"It {'falls' if late else 'rises'} with field, so t0 sits "
        f"{'before' if late else 'after'} the arrival: re-reduce with asymmetry reduce "
        f"<folder> --runs <runs> --t0-offset <bins> ({'positive' if late else 'negative'}, "
        f"|Δt| over the bin width) and refit; the phase should then hold steady."
    ]


#: The frequency must rise by at least this factor along the scan for a falling
#: amplitude to be the instrument's response: a line held near one field
#: (a superconductor's, a Knight-shifted one) loses amplitude for other reasons.
_RESPONSE_FREQUENCY_SPAN = 1.5


def _frequency_response(series: dict[str, Any], trend) -> list[str]:
    """A note per precession amplitude that falls as its frequency rises along the scan."""
    from asymmetry.core.fitting.composite import CompositeModel

    free = series["free_params"]
    frequencies = [p for p in free if re.sub(r"_\d+$", "", p) in _FREQUENCY_BASES]
    if not frequencies or not any(re.sub(r"_\d+$", "", p) == "A" for p in free):
        return []
    names = CompositeModel.from_expression(series["expression"]).param_names
    notes = []
    for frequency in frequencies:
        # A product term lists its amplitude first: the line's amplitude is the
        # last one before its frequency.
        amplitude = next(
            (
                n
                for n in reversed(names[: names.index(frequency)])
                if re.sub(r"_\d+$", "", n) == "A"
            ),
            None,
        )
        if amplitude not in free:
            continue
        rows = _measured(trend, frequency, amplitude)
        if len(rows) < 2:
            continue
        first, last = rows[0], rows[-1]

        def change(param: str) -> float:
            return (last[param] - first[param]) / math.hypot(
                first[f"{param}_err"], last[f"{param}_err"]
            )

        if (
            change(frequency) <= _STEP_SIGNIFICANCE
            or change(amplitude) >= -_STEP_SIGNIFICANCE
            or last[frequency] < _RESPONSE_FREQUENCY_SPAN * first[frequency]
        ):
            continue
        halved = next((row for row in rows if row[amplitude] <= first[amplitude] / 2.0), None)
        notes.append(
            f"NOTE: {amplitude} falls from {first[amplitude]:.4g} to {last[amplitude]:.4g} while "
            f"{frequency} rises from {first[frequency]:.4g} to {last[frequency]:.4g} "
            f"along the scan"
            + (
                ""
                if halved is None
                else f", halving by {frequency} {halved[frequency]:.4g} "
                f"({trend.order_key} {halved['x']:g})"
            )
            + ". A precession amplitude that falls as the frequency rises is the instrument's "
            "frequency response — the muon pulse's width at a pulsed source, or the time "
            "binning — not the sample losing its signal: report it as that, and compare "
            "amplitudes only between runs at similar frequencies."
        )
    return notes


#: A held frequency whose ends differ by more than this many combined errors
#: has moved: a shift worth reporting, not scatter.
_SHIFT_SIGNIFICANCE = 5.0


def _frequency_shift(trend, param: str) -> tuple[float, float, float] | None:
    """The first and last values of a held frequency that still moved, and the shift's error."""
    rows = _measured(trend, param)
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    error = math.hypot(first[f"{param}_err"], last[f"{param}_err"])
    if abs(last[param] - first[param]) <= _SHIFT_SIGNIFICANCE * error:
        return None
    return first[param], last[param], error


#: A held line above this frequency (MHz) sits in a field of tesla order,
#: where inequivalent sites or sublattices split it by about the resolution.
HIGH_FIELD_LINE_MHZ = 100.0


def _doublet_hint(series: dict[str, Any], trend) -> list[str]:
    """The two-line fit to try when a held line may be an unresolved pair."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.workflow.series import RIVAL_ENVELOPES

    free = series["free_params"]
    frequencies = [p for p in free if re.sub(r"_\d+$", "", p) in _FREQUENCY_BASES]
    if len(frequencies) != 1 or trend.order_key not in (
        "temperature",
        "sample_temperature_logged",
    ):
        return []
    held = _held_frequency(trend, frequencies[0])
    terms = series["expression"].split(" + ")
    lines = [index for index, term in enumerate(terms) if "Oscillatory" in term]
    if held is None or len(lines) != 1:
        return []
    decided = (
        [row for row in trend.rows if row["envelope"] in RIVAL_ENVELOPES]
        if "envelope" in trend.columns
        else []
    )
    cold_exponential = bool(decided) and min(decided, key=lambda row: row["x"])["envelope"] == (
        "Exponential"
    )
    if held < HIGH_FIELD_LINE_MHZ and not cold_exponential:
        return []
    two_line = " + ".join([*terms[: lines[0] + 1], terms[lines[0]], *terms[lines[0] + 1 :]])
    new_frequencies = [
        name
        for name in CompositeModel.from_expression(two_line).param_names
        if re.sub(r"_\d+$", "", name) in _FREQUENCY_BASES
    ]
    coldest = min(
        (row for row in trend.rows if row[frequencies[0]] is not None), key=lambda r: r["x"]
    )
    rates = [p for p in free if re.sub(r"_\d+$", "", p) in _RATE_BASES and coldest[p]]
    # Two lines split by about their width fit as one broadened line.
    split = coldest[rates[0]] / math.pi if rates else 0.0
    initial = "".join(
        f" --initial {name}={format_number(held + sign * split / 2.0, 7)}"
        for name, sign in zip(new_frequencies, (1.0, -1.0))
    )
    return [
        f"A held line can still be two: inequivalent muon sites or magnetic sublattices "
        f"split it by as little as the FFT resolution, and an unresolved pair fits as one "
        f"line whose envelope turns exponential. Before reading the relaxation as the whole "
        f"story — or the sample as unordered — fit the coldest run with two lines and "
        f"compare chi2_red: asymmetry recipe <folder> --run {coldest['key']} --name two-line "
        f"--expression '{two_line}'{initial}, then asymmetry fit <folder> --run "
        f"{coldest['key']} --recipe two-line. A field distribution beyond two lines needs "
        f"MaxEnt or a multi-group analysis, which this CLI does not do: say so under Not done."
    ]


#: The share of the relaxing amplitude one term must gain or lose between the
#: scan's two ends before the relaxation is called a different shape.
_WEIGHT_SHIFT = 0.5


def _weight_shift(series: dict[str, Any], trend) -> list[str]:
    """A NOTE when the relaxing amplitude moves between differently shaped terms along the scan."""
    from asymmetry.core.fitting.composite import CompositeModel
    from asymmetry.core.workflow.workdir import DERIVED_SERIES_KINDS

    terms = series["expression"].split(" + ")
    if series["kind"] in DERIVED_SERIES_KINDS or any("*" in term for term in terms):
        return []
    model = CompositeModel.from_expression(series["expression"])
    # (shape, amplitude, rate) per relaxing term: a term whose rate is zero
    # within its error relaxes nothing, whatever amplitude it carries.
    shaped = [
        (
            component.name,
            mapping["A"],
            next(name for local, name in mapping.items() if local != "A"),
        )
        for component, mapping in zip(model.components, model.parameter_mapping(), strict=True)
        if component.name != "Constant" and "A" in mapping and len(mapping) == 2
    ]
    if len({shape for shape, _, _ in shaped}) < 2:
        return []
    columns = [name for _, amp, rate in shaped for name in (amp, rate, f"{rate}_err")]
    rows = [row for row in trend.rows if all(row[name] is not None for name in columns)]
    if len(rows) < 4:
        return []
    end = max(2, len(rows) // 4)

    def share(block: list[dict[str, Any]], shape: str) -> float:
        weights = {
            amp: sum(abs(row[amp]) for row in block if row[rate] > 2.0 * row[f"{rate}_err"])
            for _, amp, rate in shaped
        }
        total = sum(weights.values())
        return (
            sum(weights[amp] for name, amp, _ in shaped if name == shape) / total if total else 0.0
        )

    for shape in dict.fromkeys(name for name, _, _ in shaped):
        low, high = share(rows[:end], shape), share(rows[-end:], shape)
        if abs(high - low) >= _WEIGHT_SHIFT:
            first, last = rows[end - 1]["x"], rows[-end]["x"]
            return [
                f"NOTE: the relaxation changes shape along the scan: the {shape} terms carry "
                f"{low:.0%} of the relaxing amplitude over the first {end} runs ({trend.order_key} "
                f"up to {first:g}) and {high:.0%} over the last {end} (from {last:g}). Report that "
                f"change of shape and where it happens, whatever the flags say about the "
                f"individual terms."
            ]
    return []


def _law_hints(name: str, trend, free_params: list[str]) -> list[str]:
    """Which trend law this series' axis and parameters call for (Step 6a).

    A field-ordered rate is the Redfield question and needs one rate; a
    frequency falling with temperature is an order parameter, one held steady
    is the applied field's and leaves the relaxation as the physics; a rate
    against a supplied quantity (a concentration) is a linear rate law.
    """
    order_key = trend.order_key
    by_base: dict[str, list[str]] = {}
    for param in free_params:
        by_base.setdefault(re.sub(r"_\d+$", "", param), []).append(param)
    rates = [p for base, ps in by_base.items() if base in _RATE_BASES for p in ps]
    frequencies = [p for base, ps in by_base.items() if base in _FREQUENCY_BASES for p in ps]
    command = f"asymmetry trend <folder> --series {name} --model"
    hints: list[str] = []
    # A precessing line's rate is its width (TF), not a Redfield relaxation (LF).
    if order_key == "field" and rates and not frequencies:
        hints.append(
            f"A relaxation rate against field is the decoupling question: {command} "
            f"Redfield --param {rates[0]} (--fix m=2 for the textbook form)."
        )
        siblings = by_base[re.sub(r"_\d+$", "", rates[0])]
        if len(siblings) > 1:
            hints.append(
                f"  This series splits the rate between {', '.join(siblings)}: Redfield "
                f"describes one exponential rate, so refit the series first with "
                f"asymmetry recipe <folder> --expression 'Exponential + Constant' --run <run> "
                f"--name single-rate, then fit-series --recipe single-rate. Flags in that "
                f"series are caveats on the law, not a reason to skip it."
            )
    if order_key in ("temperature", "sample_temperature_logged") and frequencies:
        held = _held_frequency(trend, frequencies[0])
        if held is None:
            hints.append(
                f"A precession frequency against temperature is an order parameter: {command} "
                f"OrderParameter --param {frequencies[0]} (fit below the transition)."
            )
        else:
            shift = _frequency_shift(trend, frequencies[0])
            widths = [p for p in rates if re.sub(r"_\d+$", "", p) == "sigma"] or rates
            # A line pulled below its normal-state frequency on cooling while its
            # width grows is the vortex lattice's diamagnetic shift and field
            # distribution: the width is a superfluid-density measure.
            if shift is not None and shift[0] < shift[1] and widths:
                cold = _measured(trend, widths[0])
                if len(cold) >= 2 and cold[0][widths[0]] > cold[-1][widths[0]]:
                    hints.append(
                        f"{frequencies[0]} falls below its warm value on cooling while "
                        f"{widths[0]} grows: if the sample is a superconductor, that is its "
                        f"diamagnetic shift and vortex lattice — fit the width with a gap law and "
                        f"say whether it describes it: {command} SC_SWave --param {widths[0]}, "
                        f"then SC_DWave."
                    )
            if shift is not None:
                hints.append(
                    f"{frequencies[0]} moves from {format_number(shift[0], 5)} to "
                    f"{format_number(shift[1], 5)} MHz, a shift of "
                    f"{format_number(shift[1] - shift[0], 5)} ± {format_number(shift[2], 5)} "
                    f"MHz, while staying near one field: a shift of the line (a Knight shift, "
                    f"or a superconductor's diamagnetic shift below Tc). Report it."
                )
            hints.append(
                f"{frequencies[0]} stays near {format_number(held, 4)} MHz along the scan (within 10 %): the "
                f"line follows a fixed field, so its frequency is not an order parameter — which "
                f"says nothing for or against order in the sample. The physics is in the "
                f"relaxation — its rate"
                + (f" ({', '.join(rates)})" if rates else "")
                + " and its shape"
                + (
                    ": see the envelope column, where fit-series compared a Gaussian and an "
                    "exponential envelope run by run."
                    if "envelope" in trend.columns
                    else " (fit the series with a Gaussian and with an exponential envelope)."
                )
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


#: Prefactors and offsets of the trend components (Arrhenius and
#: CriticalDivergence ``a``/``c``, Linear's intercept ``b``): an undetermined
#: one says nothing about whether the law's physical parameters are.
_NUISANCE_BASES = frozenset({"a", "b", "c"})


#: A law's shape exponent the points often cannot fix, with the value to hold
#: it at and what it sets; a free fit of one must also come out positive. Near
#: Tc the order parameter's (1 - (T/Tc)^alpha)^beta reduces to a power of
#: (Tc - T) whatever alpha is, so alpha trades off against y0; Redfield's m is
#: 2 for the Lorentzian spectral density of an exponential correlation.
_SHAPE_EXPONENTS: dict[str, tuple[str, float, str]] = {
    "alpha": ("OrderParameter", 1.0, "the curve's shape far below the transition"),
    "m": ("Redfield", 2.0, "the spectral density's fall-off (2 for a Lorentzian)"),
}


#: The axis each law is a law in; fitted against another, its parameters mean nothing.
_LAW_AXES: dict[str, frozenset[str]] = {
    "Redfield": frozenset({"field"}),
    **dict.fromkeys(
        ("OrderParameter", "Arrhenius", "CriticalDivergence"),
        frozenset({"temperature", "sample_temperature_logged"}),
    ),
}


def _render_fit(
    fit: dict[str, Any], free_params: list[str], law_steps: Sequence[str] = ()
) -> list[str]:
    """The fit block: model, range, parameters with errors, verdict and what was left out.

    *law_steps* — what this law asks next — follow the verdict and replace the generic
    advice to hold an undetermined parameter.
    """
    lo, hi = fit["x_fitted"]
    lines = [
        f"Fit of {fit['expression']} to {fit['param']} against {fit['order_key']}, over the "
        f"points' span {format_number(lo, 4)} .. {format_number(hi, 4)}: "
        f"{fit['n_points']} point(s), "
        + (
            f"chi2_red {format_number(fit['reduced_chi_squared'], 3)}"
            if fit["success"]
            else f"FAILED — {fit['message']}"
        ),
    ]
    from asymmetry.core.workflow.trend_fit import error_scale

    # The scaled errors are the honest ones (see error_scale): they lead, and
    # the verdict below is judged on them.
    scale = error_scale(fit)

    def error(name: str, factor: float) -> str:
        if name in fit["fixed"]:
            return "fixed"
        return format_number(fit["uncertainties"].get(name, 0.0) * factor, 6) + (
            " (at bound)" if name in fit["params_at_bound"] else ""
        )

    rows = [
        [name, format_number(value, 6), error(name, scale)]
        + ([error(name, 1.0)] if scale > 1.0 else [])
        + [fit["units"].get(name) or ""]
        for name, value in fit["parameters"].items()
    ]
    headers = (
        ["parameter", "value"]
        + (
            [f"error (x sqrt(chi2_red) = {scale:.3g})", "unscaled error"]
            if scale > 1.0
            else ["error"]
        )
        + ["unit"]
    )
    lines.append(render_table(headers, rows))
    physical = [
        name
        for name in fit["parameters"]
        if name not in fit["fixed"] and re.sub(r"_\d+$", "", name) not in _NUISANCE_BASES
    ]
    undetermined = [
        name
        for name in physical
        if not 0.0 < abs(fit["uncertainties"].get(name, 0.0)) * scale < abs(fit["parameters"][name])
    ]
    pinned = [name for name in fit["params_at_bound"] if name in physical]
    misplaced = [
        law
        for law, axes in _LAW_AXES.items()
        if law in fit["expression"] and fit["order_key"] not in axes
    ]
    shapes = [
        (name, value, meaning)
        for name, (law, value, meaning) in _SHAPE_EXPONENTS.items()
        if law in fit["expression"] and name in physical and law not in misplaced
    ]
    negative = [name for name, _, _ in shapes if fit["parameters"][name] <= 0.0]
    if not fit["success"] or pinned or undetermined or negative:
        reasons = (
            (["the fit did not converge"] if not fit["success"] else [])
            + [f"{name} is at a bound" for name in pinned]
            + [f"{name} is not positive, which the law does not allow" for name in negative]
            + [
                f"{name}'s scaled error is missing, zero or as large as its value"
                for name in undetermined
            ]
        )
        lines.append(
            f"LAW NOT ESTABLISHED ({'; '.join(reasons)}): {fit['expression']} does not describe "
            f"this trend. Describe the trend in plain words and do not use this law's physics."
        )
        lines.extend(law_steps)
        determined = [name for name in physical if name not in undetermined + pinned]
        if shapes:
            lines.append(
                "Next: "
                + "; ".join(
                    f"{name} sets {meaning} — refit with --fix {name}={format_number(value, 3)}"
                    for name, value, meaning in shapes
                )
                + ", and report the law with that value stated as fixed."
            )
        elif fit["success"] and determined and undetermined and not law_steps:
            lines.append(
                f"Next: {', '.join(determined)} {'is' if len(determined) == 1 else 'are'} "
                f"determined and {', '.join(undetermined)} not. Hold "
                f"{', '.join(undetermined)} at a textbook value with --fix and refit; then "
                f"report {', '.join(determined)} with the fixed value stated."
            )
    else:
        lines.append(
            "Converged, with its physical parameters determined: report them with the "
            "errors in the first error column"
            + (
                f" and the chi2_red ({format_number(fit['reduced_chi_squared'], 3)}); if the "
                "curve visibly misses points on the plot, say the law describes the trend "
                "only roughly — but do not withhold the values."
                if scale > 1.0
                else "."
            )
        )
        lines.extend(law_steps)
        if fit["expression"].strip() == "Linear" and fit["order_key"] not in _FILE_AXES:
            lines.append(
                f"The slope m is the change in {fit['param']} per unit {fit['order_key']}: for a "
                f"rate against a concentration it is the rate constant. Report m with its "
                f"error, in {fit['param']}'s unit per unit {fit['order_key']}."
            )
    for law in misplaced:
        lines.append(
            f"NOTE: {law} is a law in {' or '.join(sorted(_LAW_AXES[law]))}, and this trend is "
            f"against {fit['order_key']}: its parameters have no physical meaning here. Describe "
            f"the trend in plain words, or fit it against the axis the law is written in."
        )
    if fit["turning_point"] is not None:
        lines.append(
            f"NOTE: the fitted points turn through an extremum near {fit['order_key']} = "
            f"{format_number(fit['turning_point'], 4)}. A law that only rises or only falls, "
            f"fitted across it, averages two regimes: say so, and fit each side with "
            f"--xmin/--xmax if the physics differs."
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
            + "; ".join(f"{entry['key']} ({entry['reason']})" for entry in fit["excluded"])
        )
    if fit["flagged"]:
        lines.append(
            "flagged but fitted: "
            + "; ".join(f"{entry['key']} ({', '.join(entry['flags'])})" for entry in fit["flagged"])
        )
    return lines


def _gap_law_steps(name: str, trend, fit: dict[str, Any], plot_paths: list[Path]) -> list[str]:
    """What a superconducting gap law's fit asks next: the normal state, a verdict, its rivals."""
    from asymmetry.core.fitting.parameter_models import (
        SUPERCONDUCTING_GAP_LAWS,
        ParameterCompositeModel,
    )

    model = ParameterCompositeModel.from_expression(fit["expression"])
    gaps = [
        (index, law)
        for index, law in enumerate(model.component_names)
        if law in SUPERCONDUCTING_GAP_LAWS
    ]
    if not gaps:
        return []
    index, law = gaps[0]
    tc = fit["parameters"][model.component_param_name(index, "Tc")]
    width = model.component_param_name(index, SUPERCONDUCTING_GAP_LAWS[law])
    warm = sum(row["x"] > tc for row in trend.rows if row["key"] in fit["keys"])
    excluded = [entry["key"] for entry in fit["excluded"] if entry["reason"] == "excluded"]

    def command(expression: str, *, warm_side: bool = False) -> str:
        """``trend`` with *expression* on the same points — past any --xmax when *warm_side*."""
        ranges = (("--xmin", fit["x_min"]), ("--xmax", None if warm_side else fit["x_max"]))
        fixes = [held for held in fit["fixed"] if not (warm_side and held == width)]
        return (
            f"asymmetry trend <folder> --series {name} --model "
            + (f"'{expression}'" if " " in expression else expression)
            + f" --param {fit['param']}"
            + "".join(f" {flag} {value:g}" for flag, value in ranges if value is not None)
            + "".join(f" --fix {held}={fit['parameters'][held]:g}" for held in fixes)
            + (f" --exclude {','.join(excluded)}" if excluded else "")
        )

    lines: list[str] = []
    if warm < 2:
        coverage = (
            f"NOTE: {warm} of the fitted points {'lies' if warm == 1 else 'lie'} above the "
            f"fitted Tc = {format_number(tc, 4)}. {law} settles at {width} above Tc, so only "
            f"normal-state points determine {width}; with fewer than two it is not set by the "
            f"data."
        )
        if fit["x_max"] is not None:
            return [
                f"{coverage} Next: refit with the warm points in, rather than holding {width} "
                f"at a chosen value — no --xmax, or one above the transition: "
                f"{command(fit['expression'], warm_side=True)}. The verdict on "
                f"the law, and its comparison with the other gap laws, follow from that refit."
            ]
        lines.append(
            f"{coverage} The scan holds no more above Tc: hold {width} at the width measured "
            f"in the normal state with --fix {width}=VALUE, and say it was held."
        )
    plot = str(plot_paths[0]) if plot_paths else "the plot (rerun with --plot)"
    lines.append(
        f"Verdict due: say in the report whether {law} describes {fit['param']}(T) — yes when "
        f"chi2_red ({format_number(fit['reduced_chi_squared'], 3)}) is near 1 and the curve on "
        f"{plot} follows the points below and above Tc, no otherwise."
    )
    if law == "SC_SWave":
        nodal = re.sub(r"\bSC_SWave\b", "SC_DWave", fit["expression"])
        lines.append(
            f"SC_SWave is the fully gapped law; a gap with line nodes is its rival. Fit SC_DWave "
            f"to the same points and compare chi2_red: {command(nodal)}."
        )
    return lines
