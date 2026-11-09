"""The reduction options, declared and turned into settings once.

``reduce``, ``alpha`` and ``integral-scan`` all reduce runs, so they take the
same spellings for the same choices — the detector pair, deadtime, background,
t0 and t_good offsets, the period and the detector balance — and build one
:class:`~asymmetry.core.workflow.reduction.ReductionSettings` from them here.
The settings object owns the vocabulary; this module only parses the command
line into it and turns its :class:`ValueError` into a one-line
:class:`UserError`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from asymmetry.cli._output import UserError
from asymmetry.cli._runs import resolve_run, run_spec

if TYPE_CHECKING:
    from asymmetry.core.workflow.reduction import ReductionSettings
    from asymmetry.core.workflow.workdir import RunSelection, WorkDir

#: The command-line spelling of the green − red period difference.
GREEN_RED = "green-red"


def add_reduction_arguments(parser: argparse.ArgumentParser, *, alpha: bool = True) -> None:
    """Declare the reduction options on *parser*.

    ``alpha`` adds ``--alpha``/``--alpha-from`` (a command that measures alpha
    has neither).
    """
    # An option a command does not declare still reads as its default, so the
    # settings are built from one namespace shape.
    parser.set_defaults(alpha=None, alpha_from=None)
    if alpha:
        parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha to reduce with")
        parser.add_argument(
            "--alpha-from",
            type=int,
            default=None,
            dest="alpha_from",
            help=(
                "Estimate alpha on this run (a weak-TF calibration run), with the same "
                "pair, deadtime, background and t0, and use it"
            ),
        )
    parser.add_argument(
        "--deadtime",
        choices=["off", "from_file"],
        default="off",
        help="Deadtime correction (default: off, matching the GUI's fresh-run default)",
    )
    add_calibration_arguments(parser)
    add_pair_argument(parser)
    parser.add_argument(
        "--background",
        default="none",
        metavar="none|tail_fit|range[:FIRST:LAST]",
        help=(
            "Background subtraction: tail_fit fits a flat rate under the late-time "
            "decay; range averages an exact inclusive acquisition interval. "
            "Use range:FIRST:LAST in zero-based bins; absent bounds require a usable pre-t0 default. "
            "Default: none"
        ),
    )
    parser.add_argument(
        "--t0-offset",
        type=int,
        default=0,
        dest="t0_offset",
        metavar="BINS",
        help="Shift every detector's file t0 by this many bins (signed; default 0)",
    )
    parser.add_argument(
        "--t-good-offset",
        type=int,
        default=None,
        dest="t_good_offset",
        metavar="BINS",
        help="Signed first-bin offset from effective t0 (default: the file's)",
    )
    parser.add_argument(
        "--period",
        default=None,
        metavar=f"RED|GREEN|N|{GREEN_RED}",
        help=(
            "Select one period from a multi-period file (red is period 1, green "
            f"period 2), or {GREEN_RED} for the difference of a two-period run"
        ),
    )


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose shared explicit operations without changing no-option behaviour."""
    from asymmetry.core.fitting.tf_calibration import TFEnvelope
    from asymmetry.core.workflow.calibration import CALIBRATION_METHODS

    parser.add_argument(
        "--calibration-method",
        choices=(*CALIBRATION_METHODS, "automatic"),
        default=None,
        help="Explicit calibration operation (omitted: TF calibration fit)",
    )
    parser.add_argument(
        "--fit-beta",
        action="store_true",
        help="Fit beta with alpha (requires tf_calibration_fit; otherwise beta is fixed at 1)",
    )
    parser.add_argument(
        "--calibration-envelope",
        choices=[e.value for e in TFEnvelope],
        default=None,
        help="Constrain TF to one envelope (omitted: compare all four)",
    )
    parser.add_argument(
        "--calibration-period",
        default=None,
        metavar="RED|GREEN|N",
        help="Acquired calibration source period, independent of the target --period",
    )
    parser.add_argument(
        "--calibration-frequency",
        type=float,
        default=None,
        metavar="MHZ",
        help="TF frequency seed in MHz (omitted: source field)",
    )


def calibration_options(args: argparse.Namespace):
    """Parse once at the shared calibration request boundary."""
    from asymmetry.core.workflow.calibration import (
        CalibrationOptions,
        explicit_calibration_options,
    )

    return CalibrationOptions(
        method=explicit_calibration_options().method
        if args.calibration_method is None
        else None
        if args.calibration_method == "automatic"
        else args.calibration_method,
        fit_beta=args.fit_beta,
        envelope=args.calibration_envelope,
        period=args.calibration_period,
        frequency_mhz=args.calibration_frequency,
    )


def add_pair_argument(parser: argparse.ArgumentParser) -> None:
    """Declare ``--pair``, which the survey takes on its own."""
    parser.add_argument(
        "--pair",
        default=None,
        metavar="FWD/BWD",
        help=(
            "Forward and backward groups by name or id, e.g. 'Up/Down' or '3/4' "
            "(default: the file's own pair; `asymmetry info` lists the groups)"
        ),
    )


def parse_pair(text: str | None) -> tuple[str, str] | None:
    """``--pair FWD/BWD`` as two group labels."""
    if text is None:
        return None
    labels = [label.strip() for label in text.split("/")]
    if len(labels) != 2:
        raise UserError(f"--pair takes FWD/BWD, got {text!r}.")
    return labels[0], labels[1]


def _background(text: str) -> dict[str, Any]:
    mode, _separator, bins = text.partition(":")
    if not bins:
        return {"background": mode}
    first, _separator, last = bins.partition(":")
    try:
        return {"background": mode, "background_range": (int(first), int(last))}
    except ValueError:
        raise UserError(f"--background {text!r}: the range is FIRST:LAST in bins.") from None


def reduction_settings(
    args: argparse.Namespace, selection: RunSelection, **window: Any
) -> ReductionSettings:
    """The settings the reduction options in *args* ask for.

    *window* carries the command's own ``rebin``/``t_min``/``t_max``. With
    ``--alpha-from`` the calibration run — one of *selection*'s, so of the same
    instrument — is reduced under the same settings, so alpha balances the
    spectra it will be applied to.
    """
    from asymmetry.core.io import load
    from asymmetry.core.workflow.calibration import calibrate_run
    from asymmetry.core.workflow.reduction import (
        ALPHA_ESTIMATED_PREFIX,
        GREEN_MINUS_RED,
        ReductionSettings,
    )

    alpha, alpha_from = args.alpha, args.alpha_from
    if alpha is not None and alpha_from is not None:
        raise UserError("Pass either --alpha or --alpha-from, not both.")
    try:
        settings = ReductionSettings(
            alpha=1.0 if alpha is None else float(alpha),
            alpha_source="assumed" if alpha is None else "user",
            deadtime=args.deadtime,
            pair=parse_pair(args.pair),
            t0_offset_bins=args.t0_offset,
            t_good_offset_bins=args.t_good_offset,
            period=GREEN_MINUS_RED if args.period == GREEN_RED else args.period,
            **_background(args.background),
            **window,
        )
        options = calibration_options(args)
        if alpha_from is None:
            if args.command != "alpha" and (
                args.calibration_method is not None
                or options.period is not None
                or args.fit_beta
                or args.calibration_envelope is not None
                or args.calibration_frequency is not None
            ):
                raise ValueError("Explicit calibration options require --alpha-from")
            return settings
        loaded = load(str(resolve_run(selection, alpha_from)))
        estimate = calibrate_run(loaded, settings, options)
        if estimate.alpha is None:
            raise ValueError(estimate.to_dict()["message"])
    except (TypeError, ValueError) as exc:
        # ReductionSettings owns the vocabulary the CLI accepts; a value it
        # rejects is the user's, so it exits 1 with a message, not 2.
        raise UserError(str(exc)) from None
    if options.method is not None:
        return replace(settings, alpha=estimate.result)
    return replace(
        settings, alpha=estimate.alpha, alpha_source=f"{ALPHA_ESTIMATED_PREFIX}{alpha_from}"
    )


def deadtime_note(settings: ReductionSettings, carries_deadtime: bool) -> list[str]:
    """The NOTE for a reduction that leaves off the deadtimes its files carry."""
    if settings.deadtime != "off" or not carries_deadtime:
        return []
    return [
        "NOTE: these files carry per-detector deadtimes and this reduction leaves deadtime "
        "off. Pass --deadtime from_file before fitting further — it matters most at a pulsed "
        "source's high early count rates — and say in the summary which you used."
    ]


def alpha_step_note(workdir: WorkDir, alpha_from: int | None, runs: list[int]) -> list[str]:
    """The NOTE for one ``--alpha-from`` applied across an ALPHA STEP the survey printed."""
    if alpha_from is None or not workdir.survey_path.is_file():
        return []
    survey = workdir.read_survey()
    # Run numbers order runs only within one instrument; a step between two
    # instruments separates campaigns, which their own calibrations cover.
    instruments: dict[int, set[str]] = {}
    for row in survey["runs"]:
        instruments.setdefault(row["run_number"], set()).add(row["instrument"])
    notes = []
    for step in survey["alpha_steps"]:
        before, after = step["before_run"], step["after_run"]
        shared = instruments.get(before, set()) & instruments.get(after, set())
        if not shared & instruments.get(alpha_from, set()):
            continue
        across = [
            run
            for run in runs
            if shared & instruments.get(run, set())
            and ((alpha_from <= before and run >= after) or (alpha_from >= after and run <= before))
        ]
        if across:
            notes.append(
                f"NOTE: --alpha-from {alpha_from} calibrates one side of the survey's ALPHA STEP "
                f"between runs {before} and {after} ({step['alpha_before']:.4f} -> "
                f"{step['alpha_after']:.4f}), but {'runs' if len(across) > 1 else 'run'} "
                f"{run_spec(across)} {'lie' if len(across) > 1 else 'lies'} on the other side. "
                f"Reduce them again with --alpha-from a calibration run from their own block; "
                f"their amplitudes under this alpha are not results."
            )
    return notes


def describe(settings: ReductionSettings) -> str:
    """The one-line account of *settings* printed under a reduction's table."""
    background = settings.background
    if settings.background_range is not None:
        background += f" {settings.background_range[0]}:{settings.background_range[1]}"
    from asymmetry.core.workflow.reduction import ALPHA_ESTIMATED_PREFIX

    alpha = f"alpha {settings.effective_alpha:.4f} ({settings.effective_alpha_source}"
    if settings.alpha_source.startswith(ALPHA_ESTIMATED_PREFIX) and (
        settings.deadtime != "off" or settings.background != "none"
    ):
        alpha += (
            " on these corrected counts — survey measures raw counts, so their "
            "value differs; this is the one applied"
        )
    if settings.quick_calibration is not None:
        estimate = settings.quick_calibration.estimate
        error = (
            "error unavailable"
            if estimate.alpha_error is None
            else f"+/- {estimate.alpha_error:.3g}"
        )
        alpha = (
            f"alpha {estimate.alpha:.6g} {error} ({settings.effective_alpha_source}, beta 1 fixed"
        )
    if settings.calibration is not None:
        result = settings.calibration.selected
        error = (
            "error unavailable" if result.alpha_error is None else f"+/- {result.alpha_error:.3g}"
        )
        beta = (
            "1 fixed"
            if result.beta_fixed
            else (
                f"{result.beta:.6g} "
                + (
                    "error unavailable"
                    if result.beta_error is None
                    else f"+/- {result.beta_error:.3g}"
                )
            )
        )
        alpha = f"alpha {result.alpha:.6g} {error} (run {result.source_run_number}, {result.model}, beta {beta}; conditional errors"
    parts = [
        alpha + ")",
        f"deadtime {settings.deadtime}",
        f"background {background}",
        f"pair {'/'.join(settings.pair) if settings.pair else 'file'}",
        f"t0 offset {settings.t0_offset_bins}",
        "t_good offset "
        + ("file" if settings.t_good_offset_bins is None else str(settings.t_good_offset_bins)),
        f"rebin {settings.rebin}",
        f"period {settings.period or 'default'}",
    ]
    return ", ".join(parts)


__all__ = [
    "GREEN_RED",
    "add_pair_argument",
    "add_reduction_arguments",
    "deadtime_note",
    "describe",
    "parse_pair",
    "reduction_settings",
]
