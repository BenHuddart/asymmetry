"""``asymmetry fourier`` — frequency spectrum and quantitative peak table."""

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
from asymmetry.cli._runs import reduced_datasets, resolve_run
from asymmetry.cli._workdir import add_workdir_argument, workdir_for
from asymmetry.cli.commands.trend import HIGH_FIELD_LINE_MHZ


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fourier", help="Transform a reduced run and report resolved frequency peaks"
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Reduced run number")
    parser.add_argument(
        "--name",
        default=None,
        help="Stored spectrum name (default: run-<N>, or run-<N>-correlation)",
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help=(
            "Muoniated-radical correlation spectrum: pair the radical lines of the run's "
            "forward and backward groups (reloaded from the run files, the co-add's members "
            "for a co-add, on the file's own bins) onto the hyperfine coupling axis; peaks "
            "are couplings A_mu / MHz, --fmin/--fmax bound the coupling"
        ),
    )
    parser.add_argument(
        "--correlation-field",
        type=float,
        default=None,
        dest="correlation_field",
        metavar="GAUSS",
        help="Transverse field the line pairs are computed at (default: the run's own field)",
    )
    parser.add_argument(
        "--correlation-order",
        type=int,
        default=2,
        dest="correlation_order",
        help="Ratio-penalty order of the correlation function (default: 2, as WiMDA)",
    )
    parser.add_argument(
        "--window",
        choices=["none", "hann", "cosine", "gaussian", "lorentzian"],
        default="none",
    )
    parser.add_argument("--padding", type=int, default=4, help="Zero-padding factor (default: 4)")
    parser.add_argument("--tmin", type=float, default=None, help="Transform-window start / µs")
    parser.add_argument("--tmax", type=float, default=None, help="Transform-window end / µs")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase rotation / degrees")
    parser.add_argument(
        "--filter-tau",
        type=float,
        default=1.5,
        help="Time constant for the lorentzian/gaussian window / µs",
    )
    parser.add_argument("--fmin", type=float, default=0.0, help="Lowest reported frequency / MHz")
    parser.add_argument("--fmax", type=float, default=None, help="Highest reported frequency / MHz")
    parser.add_argument("--peaks", type=int, default=6, help="Maximum peaks to report")
    parser.add_argument("--plot", action="store_true", help="Write plots/<name>.png")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from asymmetry.cli import plots
    from asymmetry.core.workflow.fourier import (
        CorrelationSettings,
        FourierSettings,
        correlation_spectrum,
        fourier_spectrum,
    )

    if args.plot:
        plots.require_matplotlib()
    folder = Path(args.folder)
    workdir, selection = workdir_for(folder, args.workdir, args.instrument)
    dataset = reduced_datasets(workdir, [args.run])[args.run]
    default_name = f"run-{args.run}-correlation" if args.correlation else f"run-{args.run}"
    name = checked_name(args.name if args.name is not None else default_name, flag="--name")
    try:
        settings = FourierSettings(
            window=args.window,
            padding_factor=args.padding,
            t_min=args.tmin,
            t_max=args.tmax,
            phase_degrees=args.phase,
            filter_time_constant_us=args.filter_tau,
            f_min=args.fmin,
            f_max=args.fmax,
            max_peaks=args.peaks,
        )
        if args.correlation:
            entry = workdir.entry(args.run)
            field = args.correlation_field
            if field is None:
                if entry.run["field"] is None:
                    raise UserError(
                        f"Run {args.run} records no applied field; name the field the radical "
                        "lines precess in with --correlation-field."
                    )
                field = abs(entry.run["field"])
            outcome = correlation_spectrum(
                _reduced_counts(workdir, selection, entry),
                dataset.time,
                settings,
                CorrelationSettings(field_gauss=field, order=args.correlation_order),
                group_ids=[entry.forward_group, entry.backward_group],
            )
        else:
            outcome = fourier_spectrum(dataset, settings)
    except ValueError as exc:
        raise UserError(str(exc)) from None

    # The stored metadata names the plot, so a spectrum read back from
    # spectra/<name>.json points at the same PNG the CLI reports.
    plot_path = workdir.plots_dir / f"{name}.png" if args.plot else None
    result = outcome.to_dict() | {
        "name": name,
        "run": args.run,
        "field_gauss": workdir.entry(args.run).run["field"],
        "plot": None if plot_path is None else str(plot_path),
    }
    array_path, metadata_path = workdir.write_spectrum(
        name,
        frequency=outcome.frequency,
        real=outcome.real,
        magnitude=outcome.magnitude,
        payload=result,
    )
    if plot_path is not None:
        plots.plot_spectrum(
            outcome.frequency,
            outcome.real,
            outcome.magnitude,
            run_number=args.run,
            coupling=args.correlation,
            out_path=plot_path,
        )
    result |= {"array_path": str(array_path), "metadata_path": str(metadata_path)}
    if args.json:
        emit_json(payload(**result))
        return
    print(_render(result))


def _reduced_counts(workdir, selection, entry):
    """The run *entry* was reduced from, reloaded with the grouping its reduction resolved.

    The same files (a co-add's members), period and settings as ``reduce``,
    so the correlation spectrum sees the detectors, deadtime, t0, good window
    and background the stored curve was made with. Raises :class:`UserError`
    when a file changed since the reduction, which the stored curve no longer
    describes.
    """
    from dataclasses import replace

    from asymmetry.core.workflow.reduction import (
        GREEN_MINUS_RED,
        load_reduction_source,
        resolve_reduction_grouping,
    )
    from asymmetry.core.workflow.workdir import reduction_digest

    if entry.settings.period == GREEN_MINUS_RED:
        raise UserError(
            f"Run {entry.run_number} was reduced to the green − red difference; the "
            "correlation spectrum transforms one period's counts, so reduce it with "
            "--period red or --period green."
        )
    paths = [resolve_run(selection, run) for run in entry.source_runs]
    source = load_reduction_source(paths, entry.settings.period)
    grouping = resolve_reduction_grouping(source.run, entry.settings)
    digest = reduction_digest(source_files=paths, grouping=grouping, settings=entry.settings)
    if not workdir.is_current(entry.run_number, digest):
        raise UserError(
            f"Run {entry.run_number}'s files changed since it was reduced; run 'asymmetry "
            "reduce' on it again first."
        )
    return replace(source.run, grouping=grouping)


#: Tabulated peaks closer than this many resolution elements are a pair the
#: transform barely separates.
_CLOSE_PEAKS = 2


def _render(result: dict) -> str:
    peaks = result["peak_analysis"]["peaks"]
    coupling = result["axis"] == "hyperfine_coupling"
    axis = "A_mu/MHz" if coupling else "frequency/MHz"
    rows = [
        [
            format_number(peak["frequency_mhz"], 6),
            format_number(peak["amplitude"], 5),
            format_number(peak["width_mhz"], 5),
            format_number(peak["snr"], 2),
        ]
        for peak in peaks
    ]
    title = (
        f"Run {result['run']} correlation spectrum at {result['correlation']['field_gauss']:g} G"
        if coupling
        else f"Run {result['run']} Fourier spectrum"
    )
    lines = [
        f"{title} — FFT, window {result['settings']['window']}, {result['n_points']} bins, "
        f"resolution {result['resolution_mhz']:.6g} MHz, band "
        f"{result['frequency_min_mhz']:.6g}–{result['frequency_max_mhz']:.6g} of "
        f"{result['full_band_mhz'][0]:.6g}–{result['full_band_mhz'][1]:.6g} MHz",
        "",
        render_table([axis, "amplitude", "width/MHz", "SNR"], rows)
        if rows
        else "No peaks passed the detection threshold.",
        *(
            [
                "Strongest maxima in the band — candidates, not detections; confirm one "
                "with a time-domain fit before calling it a line:",
                *(
                    f"  {entry['frequency_mhz']:.4f} MHz  "
                    f"(height {entry['height_over_noise']:.1f}x the noise floor)"
                    for entry in result["candidate_maxima"]
                ),
            ]
            if result["candidate_maxima"]
            else []
        ),
        "",
        f"Spectrum written to {result['array_path']}",
        f"Provenance written to {result['metadata_path']}",
    ]
    if result["plot"] is not None:
        lines.append(f"Plot written to {result['plot']}")
    if result["outside_band"]:
        lines.append(
            "NOTE: the transform also holds lines outside this band — "
            + ", ".join(
                f"{entry['frequency_mhz']:.6g} MHz (SNR {entry['snr']:.0f})"
                for entry in result["outside_band"]
            )
            + ". --fmin/--fmax hid them from this table; widen the band (or drop --fmax) "
            "before saying a line is absent."
        )
    resolution = result["resolution_mhz"]
    close = sorted(peak["frequency_mhz"] for peak in peaks)
    pairs = [
        (low, high)
        for low, high in zip(close, close[1:])
        if high - low <= _CLOSE_PEAKS * resolution
    ]
    if pairs and not coupling:
        lines.append(
            "NOTE: "
            + "; ".join(f"{low:.6g} and {high:.6g} MHz" for low, high in pairs)
            + f" lie within {_CLOSE_PEAKS} resolution elements of each other: two lines "
            "the FFT barely separates. Report both frequencies (a splitting, not one line), "
            "and fit them in the time domain with two lines started there."
        )
    from asymmetry.core.fitting.knight_shift import larmor_frequency_mhz

    paired = {frequency for pair in pairs for frequency in pair}
    larmor = (
        None if result["field_gauss"] is None else larmor_frequency_mhz(abs(result["field_gauss"]))
    )
    # A diamagnetic line in a field of tesla order, not a radical's hyperfine line.
    high = [
        peak
        for peak in peaks
        if larmor is not None
        and larmor >= HIGH_FIELD_LINE_MHZ
        and abs(peak["frequency_mhz"] - larmor) <= 0.1 * larmor
        and peak["frequency_mhz"] not in paired
    ]
    if high and not coupling:
        line = max(high, key=lambda peak: peak["snr"])
        centre, width = line["frequency_mhz"], line["width_mhz"]
        lines.append(
            f"NOTE: the line at {centre:.6g} MHz (width {width:.4g} MHz, "
            f"{width / resolution:.1f} resolution elements) sits in a field of tesla order, "
            f"where inequivalent muon sites or magnetic sublattices split a line by about the "
            f"resolution, so one FFT peak can hold two. Fit two lines started either side of it "
            f"and compare chi2_red with one before reporting a single line: asymmetry recipe "
            f"<folder> --run {result['run']} --name two-line --expression 'Oscillatory * "
            f"Exponential + Oscillatory * Exponential + Constant' --initial "
            f"frequency_1={centre + width / 2:.6g} --initial frequency_3={centre - width / 2:.6g}"
            f", then asymmetry fit <folder> --run {result['run']} --recipe two-line."
        )
    if not coupling:
        lines.append(
            "This is an FFT: maximum-entropy (MaxEnt) spectra and multi-group "
            "field-distribution analysis are not available here — where the field "
            "distribution is part of the question, say so under Not done."
        )
    if coupling:
        lines.append(
            f"The correlation peak is the muon hyperfine coupling A_mu = nu_1 + nu_2, the sum "
            f"of the radical's two precession lines. Next: those lines themselves — "
            f"asymmetry fourier <folder> --run {result['run']} with the same window and time "
            f"range and no --fmax (both lie below A_mu) — and report them with the "
            f"transform, window and resolution beside A_mu."
        )
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
