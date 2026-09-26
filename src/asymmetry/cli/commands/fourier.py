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
        f"{title} — {result['n_points']} bins, resolution {result['resolution_mhz']:.6g} MHz",
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
    return "\n".join(lines)


__all__ = ["add_parser", "run"]
