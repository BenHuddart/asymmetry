"""``asymmetry alpha`` — measure the detector balance on a calibration run."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import UserError, emit_json, payload
from asymmetry.cli._reduction import add_reduction_arguments, reduction_settings
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
    from asymmetry.core.io import load
    from asymmetry.core.workflow.reduction import (
        estimate_alpha_for_run,
        reduce_run,
        reduction_source,
    )
    from asymmetry.core.workflow.survey import calibration_verdict, precession_evidence
    from asymmetry.core.workflow.workdir import RunSelection

    selection = RunSelection(Path(args.folder), args.instrument)
    settings = reduction_settings(args, selection)
    path = resolve_run(selection, args.run)
    try:
        dataset = reduction_source(load(str(path)), settings.period)
        estimate = estimate_alpha_for_run(dataset.run, settings)
        # The run is loaded anyway, so say plainly whether it precesses at the
        # Larmor frequency of its recorded field — the thing that makes a run
        # usable for alpha, and the thing the file's own TF stamp does not settle.
        evidence = precession_evidence(reduce_run(dataset.run, settings), dataset.field)
    except (TypeError, ValueError) as exc:
        raise UserError(f"Run {args.run}: {exc}") from None

    # The same two-source rule the survey's candidate list uses, so the two
    # commands can never disagree about whether a run will calibrate alpha.
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
    print(f"  alpha           : {estimate.alpha:.4f}")
    print(f"  method          : {estimate.method}")
    print(f"  forward group   : {estimate.forward_group}")
    print(f"  backward group  : {estimate.backward_group}")
    print(f"  calibration run : {'yes' if source is not None else 'no'} — {reason}")
    print(f"  precession      : {evidence.describe()}")
    if warning is not None:
        print(f"  WARNING: {warning}")
