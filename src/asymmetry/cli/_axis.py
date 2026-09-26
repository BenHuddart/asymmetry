"""The scan axis a series, a simultaneous fit or a batch of them is ordered along."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, TypeVar

from asymmetry.cli._output import UserError

if TYPE_CHECKING:
    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.workflow.series import ScanAxis

Member = TypeVar("Member", int, str)

#: The axis of a batch of groups or of a trend across series, when none is named:
#: each member's setpoint averaged over its runs.
MEMBER_ORDER_DEFAULT = "temperature"


def order_help(noun: str, plural: str, flag: str) -> str:
    """``--order`` help for an axis over *plural*, whose supplied values go in *flag*.

    Spelled out rather than read from ``series.ORDER_KEYS`` so that building the
    parser does not import the fitting engine; a test pins the two together.
    """
    return (
        f"Quantity the {plural} are ordered and trended along: temperature (the "
        f"setpoint), sample_temperature_logged, field or run, read from the files"
        + ("" if noun == "run" else f" and averaged over the runs of each {noun}")
        + f"; or any other name, whose value for every {noun} you give with {flag}"
    )


ORDER_HELP = order_help("run", "runs", "--x")


def add_axis_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: str | None,
    noun: str = "run",
    plural: str = "runs",
    prefix: str = "",
    example: str = "101=0,102=0.25,103=0.5",
) -> None:
    """Declare ``--<prefix>order`` and ``--<prefix>x``; ``default=None`` makes the order required."""
    parser.add_argument(
        f"--{prefix}order",
        required=default is None,
        default=default,
        metavar="QUANTITY",
        help=order_help(noun, plural, f"--{prefix}x")
        + ("" if default is None else f" (default: {default})"),
    )
    parser.add_argument(
        f"--{prefix}x",
        default=None,
        metavar=f"{noun.upper()}=VALUE,...",
        help=(
            f"Per-{noun} values of a quantity the files do not record, e.g. "
            f"'--{prefix}order concentration --{prefix}x {example}'"
        ),
    )


def member_axis(
    order: str,
    values: str | None,
    groups: Mapping[Member, Mapping[int, MuonDataset]],
    *,
    key: Callable[[str], Member],
    flag: str,
    noun: str,
) -> ScanAxis[Member]:
    """Resolve an order and its supplied *values* against *groups* of runs.

    With no *values*, each group's mean of the file quantity *order* over its
    runs; otherwise the ``KEY=VALUE`` list from *flag*, each key parsed by
    *key* into a member of *groups*.
    """
    from asymmetry.core.workflow.series import group_axis, supplied_axis

    try:
        if values is None:
            return group_axis(groups, order)
        return supplied_axis(order, _parse_values(values, key, flag, noun), groups, noun=noun)
    except ValueError as exc:
        raise UserError(str(exc)) from None


def axis_from_arguments(
    args: argparse.Namespace, datasets_by_run: Mapping[int, MuonDataset]
) -> ScanAxis[int]:
    """Resolve ``--order``/``--x`` against the runs being fitted."""
    return member_axis(
        args.order,
        args.x,
        {run: {run: dataset} for run, dataset in datasets_by_run.items()},
        key=int,
        flag="--x",
        noun="run",
    )


def _parse_values(
    text: str, key: Callable[[str], Member], flag: str, noun: str
) -> dict[Member, float]:
    """Turn ``"101=0,102=0.5"`` into ``{key("101"): 0.0, key("102"): 0.5}``."""
    shape = f"{noun.upper()}=VALUE"
    values: dict[Member, float] = {}
    for token in text.split(","):
        key_text, separator, value_text = token.strip().partition("=")
        if not separator:
            raise UserError(f"{flag} entry {token.strip()!r} is not {shape}.")
        try:
            member, value = key(key_text.strip()), float(value_text)
        except ValueError:
            raise UserError(f"{flag} entry {token.strip()!r} is not {shape}.") from None
        if member in values:
            raise UserError(f"{flag} gives {noun} {member} twice.")
        values[member] = value
    return values


__all__ = [
    "MEMBER_ORDER_DEFAULT",
    "ORDER_HELP",
    "add_axis_arguments",
    "axis_from_arguments",
    "member_axis",
]
