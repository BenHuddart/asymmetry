"""``asymmetry wizard`` — screen one reduced run against candidate models."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import UserError, emit_json, format_number, payload, render_table
from asymmetry.cli._runs import reduced_datasets

#: Candidates listed in the human-readable table.
_TOP_CANDIDATES = 5


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``wizard`` subcommand."""
    parser = subparsers.add_parser(
        "wizard",
        help="Screen a reduced run against the fit wizard's candidate models",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Run number to screen")
    parser.add_argument(
        "--geometry",
        choices=["ZF", "TF", "LF"],
        default=None,
        help=(
            "Applied-field geometry, overriding what the file records "
            "(ISIS stamps TF on zero-field runs and some files record nothing)"
        ),
    )
    # Not an argparse ``choices`` list: reading the wizard's preset vocabulary
    # means importing the fitting package, and the parser is built on *every*
    # invocation — including ``--help`` and the commands that never fit
    # anything. The value is checked in :func:`run`, which names every preset.
    parser.add_argument(
        "--scope",
        default="auto",
        metavar="PRESET",
        help="Candidate-family scope preset (default: auto, from the run's geometry)",
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Work directory to read and write (default: <folder>/.asymmetry)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Screen the run, store the payload and the recipe, and report both."""
    from asymmetry.core.workflow.screen import SCOPE_PRESETS, screen_run
    from asymmetry.core.workflow.workdir import WorkDir

    if args.scope not in SCOPE_PRESETS:
        raise UserError(
            f"Unknown scope preset {args.scope!r}; expected one of {', '.join(SCOPE_PRESETS)}."
        )

    folder = Path(args.folder)
    workdir = WorkDir.for_folder(folder, args.workdir)
    dataset = reduced_datasets(workdir, [args.run])[args.run]

    result = screen_run(
        dataset,
        geometry=args.geometry,
        scope_preset=args.scope,
        run_number=args.run,
    )
    wizard_path = workdir.write_wizard(args.run, result.to_dict())
    recipe_name = f"wizard-{args.run}"
    recipe_path = (
        None if result.recipe is None else workdir.write_recipe(recipe_name, result.recipe)
    )

    if args.json:
        emit_json(
            payload(
                screen=result.to_dict(),
                wizard_path=str(wizard_path),
                recipe_name=None if recipe_path is None else recipe_name,
                recipe_path=None if recipe_path is None else str(recipe_path),
            )
        )
        return

    print(_render(result, wizard_path, recipe_path))


def _render(result, wizard_path: Path, recipe_path: Path | None) -> str:
    """The human-readable screening report."""
    geometry = result.geometry or "unknown"
    lines = [
        f"Run {result.run_number} — geometry {geometry} (from {result.geometry_source}), "
        f"scope {result.scope_preset}",
    ]
    if result.scope_note:
        lines.append(f"  scope: {result.scope_note}")
    lines.append("")

    if result.recommended_key is None:
        lines.append("Recommendation: none — the wizard found no candidate it would stand behind.")
    else:
        lines.append(f"Recommendation: {result.recommended_key}")
        lines.append(f"  {result.summary}")
    lines.append(f"  confidence : {result.confidence}")
    lines.append(f"  verdict    : {result.verdict}")
    if result.caveat:
        lines.append(f"  caveat     : {result.caveat}")
    lines.append("")

    headers = ["", "key", "title", "category", "AICc", "chi2_red", "params"]
    rows = [
        [
            "*" if candidate.is_recommended else ("~" if candidate.is_comparable else ""),
            candidate.key,
            candidate.title,
            candidate.category,
            format_number(candidate.aicc, 1),
            format_number(candidate.chi2_red, 3),
            str(candidate.parameter_count),
        ]
        for candidate in result.candidates[:_TOP_CANDIDATES]
    ]
    lines.append(render_table(headers, rows))
    lines.append("(* recommended, ~ comparable)")
    lines.append("")

    lines.append(result.narrative.rstrip())
    lines.append("")
    lines.append(f"Screening written to {wizard_path}")
    if recipe_path is None:
        lines.append("No recipe written — there is no recommended model to fit.")
    else:
        lines.append(f"Recipe written to {recipe_path}")
    return "\n".join(lines)
