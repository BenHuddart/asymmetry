"""``asymmetry alpha`` — measure the detector balance on a calibration run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from asymmetry.cli._output import UserError, emit_json, payload
from asymmetry.cli._reduction import (
    add_reduction_arguments,
    calibration_options,
    reduction_settings,
)
from asymmetry.cli._runs import add_instrument_argument, resolve_run


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``alpha`` subcommand."""
    parser = subparsers.add_parser(
        "alpha",
        help="Estimate the forward/backward balance alpha from one run",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Run number to estimate alpha on")
    add_reduction_arguments(parser, alpha=False)
    add_instrument_argument(parser)
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Estimate alpha on the named run and report it with a suitability verdict."""
    from asymmetry.core.fitting.tf_calibration import TFCalibrationSelection
    from asymmetry.core.io import load
    from asymmetry.core.workflow.calibration import calibrate_run, calibration_source
    from asymmetry.core.workflow.reduction import (
        reduce_run,
    )
    from asymmetry.core.workflow.survey import calibration_verdict, precession_evidence
    from asymmetry.core.workflow.workdir import RunSelection

    selection = RunSelection(Path(args.folder), args.instrument)
    settings = reduction_settings(args, selection)
    path = resolve_run(selection, args.run)
    try:
        loaded = load(str(path))
        options = calibration_options(args)
        estimate = calibrate_run(loaded, settings, options)
        dataset = calibration_source(loaded, settings, options)
        # The run is loaded anyway, so say plainly whether it precesses at the
        # Larmor frequency of its recorded field — the thing that makes a run
        # usable for alpha, and the thing the file's own TF stamp does not settle.
        evidence = precession_evidence(
            reduce_run(dataset.run, replace(settings, period=None)), dataset.field
        )
    except (TypeError, ValueError) as exc:
        raise UserError(f"Run {args.run}: {exc}") from None

    # The same two-source rule the survey's candidate list uses, so the two
    # commands use the same advisory calibration recommendation.
    source, reason = calibration_verdict(dataset.run.metadata, dataset.field, evidence)
    warning = (
        None
        if source is not None
        else (
            f"run {args.run} is not a weak-transverse-field calibration run "
            f"({reason}); the estimate balances this run's own counts, "
            "which a relaxing or magnetically ordered sample will bias"
        )
    )

    if args.json:
        emit_json(
            payload(
                alpha=estimate.to_dict(),
                file=path.name,
                is_calibration_candidate=source is not None,
                calibration_source=source,
                reason=reason,
                precession=evidence.to_dict(),
                warning=warning,
            )
        )
        return

    print(f"Run {estimate.run_number} ({path.name})")
    summary = estimate.to_dict()
    for name in ("alpha", "beta"):
        value = summary.get(name)
        error = summary.get(f"{name}_error")
        text = "unavailable" if value is None else f"{value:.6g}"
        text += f" +/- {error:.3g}" if error is not None else " (error unavailable)"
        if name == "beta" and summary.get("beta_fixed"):
            text = "1 (fixed)"
        print(f"  {name:16}: {text}")
    if summary.get("model"):
        print(f"  model           : {summary['model']} (conditional covariance errors)")
    if summary["message"]:
        print(f"  diagnostic      : {summary['message']}")
    if estimate.record is not None:
        candidates = (
            estimate.record.result.candidates
            if isinstance(estimate.record.result, TFCalibrationSelection)
            else (estimate.record.result,)
        )
        for candidate in candidates:
            if not candidate.ok:
                print(f"  {candidate.envelope.value}: {candidate.message}")
        domain = candidates[0].fit.domains[0]
        print(f"  deadtime        : {domain.deadtime.requested} -> {domain.deadtime.status.value}")
        for label, outcome in domain.background_preflight:
            print(f"  background {label}: {outcome.requested} -> {outcome.status.value}")
        print("  TF backgrounds  : fitted in the count model")
    else:
        corrections = estimate.result.corrections
        for outcome in (corrections.deadtime, corrections.background):
            print(f"  {outcome.name:16}: {outcome.requested} -> {outcome.status.value}")
    print(f"  method          : {estimate.method}")
    print(f"  forward group   : {estimate.forward_group}")
    print(f"  backward group  : {estimate.backward_group}")
    print(f"  calibration run : {'yes' if source is not None else 'no'} — {reason}")
    print(f"  precession      : {evidence.describe()}")
    if warning is not None:
        print(f"  WARNING: {warning}")
