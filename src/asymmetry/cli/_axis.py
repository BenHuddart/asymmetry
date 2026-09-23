"""The scan axis a series or simultaneous fit is ordered and trended along."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import TYPE_CHECKING

from asymmetry.cli._output import UserError

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow.series import ScanAxis

#: Spelled out rather than read from ``series.ORDER_KEYS`` so that building the
#: parser does not import the fitting engine; a test pins the two together.
ORDER_HELP = (
    "Quantity the runs are ordered and trended along: temperature (the "
    "setpoint), sample_temperature_logged, field or run, read from the files; "
    "or any other name, whose value for every run you give with --x"
)


def add_axis_arguments(parser: argparse.ArgumentParser, *, default: str | None) -> None:
    """Declare ``--order`` and ``--x``; ``default=None`` makes ``--order`` required."""
    parser.add_argument(
        "--order",
        required=default is None,
        default=default,
        metavar="QUANTITY",
        help=ORDER_HELP,
    )
    parser.add_argument(
        "--x",
        default=None,
        metavar="RUN=VALUE,...",
        help=(
            "Per-run values of a quantity the files do not record, e.g. "
            "'--order concentration --x 78251=0,78279=0.25,78277=0.5'"
        ),
    )


def axis_from_arguments(
    args: argparse.Namespace, datasets_by_run: Mapping[int, MuonDataset]
) -> ScanAxis:
    """Resolve ``--order``/``--x`` against the runs being fitted."""
    from asymmetry.core.workflow.series import scan_axis, supplied_axis

    try:
        if args.x is None:
            return scan_axis(datasets_by_run, args.order)
        return supplied_axis(args.order, _parse_values(args.x), datasets_by_run)
    except ValueError as exc:
        raise UserError(str(exc)) from None


def _parse_values(text: str) -> dict[int, float]:
    """Turn ``"101=0,102=0.5"`` into ``{101: 0.0, 102: 0.5}``."""
    values: dict[int, float] = {}
    for token in text.split(","):
        run_text, separator, value_text = token.strip().partition("=")
        if not separator:
            raise UserError(f"--x entry {token.strip()!r} is not RUN=VALUE.")
        try:
            run, value = int(run_text), float(value_text)
        except ValueError:
            raise UserError(f"--x entry {token.strip()!r} is not RUN=VALUE.") from None
        if run in values:
            raise UserError(f"--x gives run {run} twice.")
        values[run] = value
    return values


__all__ = ["ORDER_HELP", "add_axis_arguments", "axis_from_arguments"]
