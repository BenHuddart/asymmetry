"""Naming a recipe, and overriding it, on the command line.

``--recipe`` takes either a path or a bare name resolved against the work
directory's ``recipes/``, and ``--fix``/``--free`` edit the recipe the fit
starts from without touching the stored file. The parsing is CLI vocabulary, so
it lives here; the editing itself is
:meth:`~asymmetry.core.workflow.recipe.FitRecipe.with_overrides`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from asymmetry.cli._output import UserError, checked_name

if TYPE_CHECKING:
    from asymmetry.core.workflow.recipe import FitRecipe
    from asymmetry.core.workflow.workdir import WorkDir


def add_recipe_arguments(parser: argparse.ArgumentParser, *, free: bool = True) -> None:
    """Declare ``--recipe`` and the override flags on a fitting subcommand."""
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe file, or the name of one in the work directory's recipes/",
    )
    parser.add_argument(
        "--fix",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Hold a parameter at a value (repeatable)",
    )
    if free:
        parser.add_argument(
            "--free",
            action="append",
            default=[],
            metavar="NAME",
            help="Release a parameter the recipe holds (repeatable)",
        )


def load_recipe(workdir: WorkDir, reference: str) -> FitRecipe:
    """The recipe *reference* names: a path, or a name under ``recipes/``.

    Raises :class:`UserError` naming the recipes the work directory does hold
    when the reference resolves to nothing, and for a bare name that is not
    usable as one (see :func:`~asymmetry.cli._output.checked_name`) — a name is
    interpolated into a path under ``recipes/``, so it has to stay inside it.
    """
    from asymmetry.core.workflow.recipe import FitRecipe

    path = Path(reference)
    if path.suffix == ".json" or path.exists():
        if not path.is_file():
            raise UserError(f"Recipe file {path} does not exist.")
        return FitRecipe.from_dict(json.loads(path.read_text(encoding="utf-8")))

    reference = checked_name(reference, flag="--recipe")
    stored = workdir.recipe_names()
    if reference not in stored:
        known = ", ".join(stored) if stored else "none yet — run 'asymmetry wizard' first"
        raise UserError(f"No recipe {reference!r} in {workdir.recipes_dir} (it holds: {known}).")
    return workdir.read_recipe(reference)


def parse_fix(entries: list[str], *, flag: str = "--fix") -> dict[str, float]:
    """Turn ``["A0=20", "phase=0"]`` into ``{"A0": 20.0, "phase": 0.0}``.

    *flag* names the option the entries came from, for the error message.
    """
    fixed: dict[str, float] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not separator or not name.strip():
            raise UserError(f"{flag} {entry!r} is not NAME=VALUE.")
        try:
            fixed[name.strip()] = float(value)
        except ValueError:
            raise UserError(f"{flag} {entry!r} does not name a number.") from None
    return fixed


def recipe_with_overrides(
    recipe: FitRecipe,
    *,
    fix: list[str],
    free: list[str] = (),
) -> FitRecipe:
    """Apply this invocation's ``--fix``/``--free`` entries to *recipe*."""
    try:
        return recipe.with_overrides(fix=parse_fix(fix), free=[name.strip() for name in free])
    except KeyError as exc:
        # The recipe owns the parameter vocabulary; a name it does not carry is
        # the user's typo, so it exits 1 with the message rather than 2.
        raise UserError(str(exc.args[0])) from None


__all__ = [
    "add_recipe_arguments",
    "load_recipe",
    "parse_fix",
    "recipe_with_overrides",
]
