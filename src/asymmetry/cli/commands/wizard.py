"""``asymmetry wizard`` — screen one reduced run against candidate models."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import replace
from pathlib import Path

from asymmetry.cli._output import UserError, emit_json, format_number, payload, render_table
from asymmetry.cli._runs import reduced_datasets
from asymmetry.cli._workdir import add_workdir_argument, workdir_for

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
            "Applied-field geometry, overriding the survey's and the file's "
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
    parser.add_argument(
        "--include",
        default="",
        metavar="C,D",
        help=(
            "Time-domain components to add to the scope's families, e.g. "
            "'Oscillatory' for a line in an LF run"
        ),
    )
    parser.add_argument(
        "--exclude",
        default="",
        metavar="C,D",
        help="Components to drop from the scope, e.g. 'VortexLattice,VortexLatticePowder'",
    )
    parser.add_argument("--tmin", type=float, default=None, help="Screen only above this time / µs")
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Screen only below this time / µs (the recipe keeps the window)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Write plots/wizard-<run>.png of data + recommendation"
    )
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    add_workdir_argument(parser, purpose="read and write")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Screen the run, store the payload and the recipe, and report both."""
    from asymmetry.cli import plots
    from asymmetry.core.workflow.screen import SCOPE_PRESETS, screen_run

    if args.scope not in SCOPE_PRESETS:
        raise UserError(
            f"Unknown scope preset {args.scope!r}; expected one of {', '.join(SCOPE_PRESETS)}."
        )
    if args.plot:
        plots.require_matplotlib()

    folder = Path(args.folder)
    workdir = workdir_for(folder, args.workdir)
    dataset = reduced_datasets(workdir, [args.run])[args.run]
    if args.tmin is not None or args.tmax is not None:
        dataset = dataset.time_range(args.tmin, args.tmax)

    try:
        result = screen_run(
            dataset,
            geometry=args.geometry,
            survey_geometry=_survey_geometry(workdir, args.run),
            scope_preset=args.scope,
            include=_names(args.include),
            exclude=_names(args.exclude),
            run_number=args.run,
        )
    except ValueError as exc:
        raise UserError(str(exc)) from None
    if result.recipe is not None and (args.tmin is not None or args.tmax is not None):
        result = replace(result, recipe=result.recipe.with_window(t_min=args.tmin, t_max=args.tmax))
    wizard_path = workdir.write_wizard(args.run, result.to_dict())
    recipe_name = f"wizard-{args.run}"
    recipe_path = (
        None if result.recipe is None else workdir.write_recipe(recipe_name, result.recipe)
    )

    plot_path = None
    plot_note = None
    if args.plot:
        if result.recipe is None:
            plot_note = "no recommendation to plot"
        else:
            recipe = result.recipe
            plot_path = plots.plot_fit(
                dataset.time,
                dataset.asymmetry,
                dataset.error,
                model_function=recipe.model().function,
                parameters={p.name: p.value for p in recipe.parameters},
                t_min=recipe.t_min,
                t_max=recipe.t_max,
                run_number=args.run,
                expression=recipe.expression,
                out_path=workdir.plots_dir / f"wizard-{args.run}.png",
            )

    if args.json:
        emit_json(
            payload(
                screen=result.to_dict(),
                wizard_path=str(wizard_path),
                recipe_name=None if recipe_path is None else recipe_name,
                recipe_path=None if recipe_path is None else str(recipe_path),
                plots=[] if plot_path is None else [str(plot_path)],
                plot_note=plot_note,
            )
        )
        return

    from asymmetry.core.fitting.fit_wizard import effective_window_duration

    duration_us = effective_window_duration(dataset)
    print(_render(args.folder, duration_us, result, wizard_path, recipe_path, plot_path, plot_note))


def _names(text: str) -> list[str]:
    """``"A, B"`` as ``["A", "B"]``."""
    return [name.strip() for name in text.split(",") if name.strip()]


def _survey_geometry(workdir, run_number: int) -> str | None:
    """The geometry the folder's survey resolved for *run_number*, if surveyed.

    The survey may have *measured* it from Larmor precession, which is the only
    source that can speak for a file recording no field state; so when a survey
    exists in the work directory its reading beats this one dataset's metadata.
    ``None`` when the folder was never surveyed, the run is not in the survey,
    or the survey could not decide either.
    """
    if not workdir.survey_path.exists():
        return None
    for row in workdir.read_survey()["runs"]:
        if row["run_number"] == run_number:
            return row["geometry"]
    return None


def _render(
    folder: str,
    duration_us: float,
    result,
    wizard_path: Path,
    recipe_path: Path | None,
    plot_path: Path | None,
    plot_note: str | None,
) -> str:
    """The human-readable screening report."""
    geometry = result.geometry or "unknown"
    lines = [
        f"Run {result.run_number} — geometry {geometry} (from {result.geometry_source}), "
        f"scope {result.scope_preset}"
        + "".join(f" +{name}" for name in result.scope_include)
        + "".join(f" -{name}" for name in result.scope_exclude),
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

    # The spectral evidence and the fitted values are what an analyst reads
    # first: a precession frequency found here is a finding even when the
    # recommendation is not the model the scan ends up fitted with.
    from asymmetry.core.fitting.fit_wizard import MIN_CYCLES_IN_EFFECTIVE_WINDOW

    # A "line" completing under MIN_CYCLES_IN_EFFECTIVE_WINDOW cycles in the
    # record is relaxation leaking into the lowest bins, not precession (the
    # survey's rule too), so it is neither printed nor offered as a seed.
    peaks = [
        peak
        for peak in result.recommendation["peak_analysis"]["peaks"]
        if peak["frequency_mhz"] * duration_us >= MIN_CYCLES_IN_EFFECTIVE_WINDOW
    ]
    lines.append(
        "Spectral lines: "
        + (
            ", ".join(f"{peak['frequency_mhz']:.4g} MHz (SNR {peak['snr']:.1f})" for peak in peaks)
            if peaks
            else "none detected"
        )
    )
    fitted_lines = (
        []
        if result.recipe is None
        else [
            parameter.value
            for parameter in result.recipe.parameters
            if parameter.name.startswith("frequency")
        ]
    )
    unfitted = [
        peak["frequency_mhz"]
        for peak in peaks
        if not any(abs(value / peak["frequency_mhz"] - 1.0) < 0.1 for value in fitted_lines)
    ]
    if unfitted:
        lines.append(
            "A detected line the recommendation does not fit is still a candidate. To test "
            "one in the time domain, start a recipe at it and check the fitted amplitude "
            "against its error:"
        )
        lines.append(
            f"  asymmetry recipe {shlex.quote(folder)} --run {result.run_number} --name line-{result.run_number} "
            f'--expression "Oscillatory * Exponential + Constant" '
            f"--initial frequency={unfitted[0]:.4g}"
        )
    if result.recipe is not None:
        lines.append(
            "Recommended fit: "
            + ", ".join(
                f"{parameter.name}={parameter.value:.4g}" + (" (fixed)" if parameter.fixed else "")
                for parameter in result.recipe.parameters
            )
        )
    lines.append("")

    lines.append(result.narrative.rstrip())
    lines.append("")
    lines.append(f"Screening written to {wizard_path}")
    if recipe_path is None:
        lines.append("No recipe written — there is no recommended model to fit.")
    else:
        lines.append(f"Recipe written to {recipe_path}")
    if plot_path is not None:
        lines.append(f"Plot written to {plot_path}")
    elif plot_note is not None:
        lines.append(f"No plot written — {plot_note}.")
    return "\n".join(lines)
