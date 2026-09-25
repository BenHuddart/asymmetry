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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from asymmetry.cli._output import UserError
from asymmetry.cli._runs import resolve_run

if TYPE_CHECKING:
    from asymmetry.core.workflow.reduction import ReductionSettings

#: The command-line spelling of the green − red period difference.
GREEN_RED = "green-red"


def add_reduction_arguments(
    parser: argparse.ArgumentParser, *, alpha: bool = True, background: bool = True
) -> None:
    """Declare the reduction options on *parser*.

    ``alpha`` adds ``--alpha``/``--alpha-from`` (a command that measures alpha
    has neither); ``background`` adds ``--background`` (the integral scan does
    not subtract one).
    """
    # An option a command does not declare still reads as its default, so the
    # settings are built from one namespace shape.
    parser.set_defaults(alpha=None, alpha_from=None, background="none")
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
    add_pair_argument(parser)
    if background:
        parser.add_argument(
            "--background",
            default="none",
            metavar="none|tail_fit|range[:FIRST:LAST]",
            help=(
                "Background subtraction: tail_fit fits a flat rate under the late-time "
                "decay (pulsed sources); range averages a pre-t0 bin range (continuous "
                "sources; default range 0.1·t0–0.6·t0). Default: none"
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
        help="First good bin this many bins after the effective t0 (default: the file's)",
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


def reduction_settings(args: argparse.Namespace, folder: Path, **window: Any) -> ReductionSettings:
    """The settings the reduction options in *args* ask for.

    *window* carries the command's own ``rebin``/``t_min``/``t_max``. With
    ``--alpha-from`` the calibration run is reduced under the same settings, so
    alpha balances the spectra it will be applied to.
    """
    from asymmetry.core.io import load
    from asymmetry.core.workflow.reduction import (
        GREEN_MINUS_RED,
        ReductionSettings,
        estimate_alpha_for_run,
        reduction_source,
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
        if alpha_from is None:
            return settings
        calibration = reduction_source(load(str(resolve_run(folder, alpha_from))), settings.period)
        estimate = estimate_alpha_for_run(calibration.run, settings)
    except (TypeError, ValueError) as exc:
        # ReductionSettings owns the vocabulary the CLI accepts; a value it
        # rejects is the user's, so it exits 1 with a message, not 2.
        raise UserError(str(exc)) from None
    return replace(settings, alpha=estimate.alpha, alpha_source=f"estimated:{alpha_from}")


def describe(settings: ReductionSettings) -> str:
    """The one-line account of *settings* printed under a reduction's table."""
    background = settings.background
    if settings.background_range is not None:
        background += f" {settings.background_range[0]}:{settings.background_range[1]}"
    parts = [
        f"alpha {settings.alpha:.4f} ({settings.alpha_source})",
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
    "describe",
    "parse_pair",
    "reduction_settings",
]
