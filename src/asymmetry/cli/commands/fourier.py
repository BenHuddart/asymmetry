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
from asymmetry.cli._runs import reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fourier", help="Transform a reduced run and report resolved frequency peaks"
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Reduced run number")
    parser.add_argument("--name", default=None, help="Stored spectrum name (default: run-<N>)")
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
    from asymmetry.core.workflow.fourier import FourierSettings, fourier_spectrum

    if args.plot:
        plots.require_matplotlib()
    folder = Path(args.folder)
    workdir, _selection = workdir_for(folder, args.workdir, args.instrument)
    dataset = reduced_datasets(workdir, [args.run])[args.run]
    name = checked_name(args.name if args.name is not None else f"run-{args.run}", flag="--name")
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
            out_path=plot_path,
        )
    result |= {"array_path": str(array_path), "metadata_path": str(metadata_path)}
    if args.json:
        emit_json(payload(**result))
        return
    print(_render(result))


def _render(result: dict) -> str:
    peaks = result["peak_analysis"]["peaks"]
    rows = [
        [
            format_number(peak["frequency_mhz"], 6),
            format_number(peak["amplitude"], 5),
            format_number(peak["width_mhz"], 5),
            format_number(peak["snr"], 2),
        ]
        for peak in peaks
    ]
    lines = [
        f"Run {result['run']} Fourier spectrum — {result['n_points']} bins, resolution {result['resolution_mhz']:.6g} MHz",
        "",
        render_table(["frequency/MHz", "amplitude", "width/MHz", "SNR"], rows)
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
