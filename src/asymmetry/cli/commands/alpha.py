"""``asymmetry alpha`` — measure the detector balance on a calibration run."""

from __future__ import annotations

import argparse
from pathlib import Path

from asymmetry.cli._output import emit_json, payload
from asymmetry.cli._runs import resolve_run


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare the ``alpha`` subcommand."""
    parser = subparsers.add_parser(
        "alpha",
        help="Estimate the forward/backward balance alpha from one run",
    )
    parser.add_argument("folder", help="Directory holding the run files")
    parser.add_argument("--run", type=int, required=True, help="Run number to estimate alpha on")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable payload")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Estimate alpha on the named run and report it with a suitability verdict."""
    from asymmetry.core.data.calibration import classify_tf_calibration_run
    from asymmetry.core.io import load
    from asymmetry.core.workflow.reduction import estimate_alpha_for_run

    path = resolve_run(Path(args.folder), args.run)
    result = load(str(path))
    dataset = result[0] if isinstance(result, list) else result
    estimate = estimate_alpha_for_run(dataset.run)

    verdict = classify_tf_calibration_run(dataset.run.metadata)
    warning = (
        None
        if verdict.is_candidate
        else (
            f"run {args.run} is not a weak-transverse-field calibration run "
            f"({verdict.reason}); the estimate balances this run's own counts, "
            "which a relaxing or magnetically ordered sample will bias"
        )
    )

    if args.json:
        emit_json(
            payload(
                alpha=estimate.to_dict(),
                file=path.name,
                is_calibration_candidate=verdict.is_candidate,
                reason=verdict.reason,
                warning=warning,
            )
        )
        return

    print(f"Run {estimate.run_number} ({path.name})")
    print(f"  alpha           : {estimate.alpha:.4f}")
    print(f"  method          : {estimate.method}")
    print(f"  forward group   : {estimate.forward_group}")
    print(f"  backward group  : {estimate.backward_group}")
    print(f"  calibration run : {'yes' if verdict.is_candidate else 'no'} — {verdict.reason}")
    if warning is not None:
        print(f"  WARNING: {warning}")
