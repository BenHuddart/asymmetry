#!/usr/bin/env python3
"""Run a wave of agent evaluations in parallel, one ``run_eval.py`` per dataset.

A pass of the evaluation loop is a named set of (dataset, rubric) cases run
side by side into one output root, ``<out>/<case>/``. Each case is an ordinary
:mod:`run_eval` run; this script only fans them out, waits, prints each case's
exit code, wall time and cost, and then deletes the per-run dataset copies
(``data/``, ``workdir/``) — scoring reads the summaries and transcripts, and the
copies of a few passes are enough to fill a disk (on macOS, iCloud then
offloads the corpus itself, which then reads as empty).

Score the results with ``scoring_brief.md`` beside this file.

Usage::

    python tools/agent_eval/run_wave.py --set trend-fit --out /tmp/evals/pass16a
    python tools/agent_eval/run_wave.py --set tier-a --parallel 3 --out /tmp/evals/regress
    python tools/agent_eval/run_wave.py --case plateau-redfield --case sn-critical-field \\
        --out /tmp/evals/pass16b
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = Path.home() / "Documents" / "WiMDA muon school"

#: Each rubric's data folder, relative to the corpus root.
CASES: dict[str, str] = {
    "plateau-redfield": "Magnetism/Dynamics in a magnetic plateau system/Data",
    "sn-critical-field": "Superconductivity/Critical fields in Sn/Data",
    "euo-psi": "Magnetism/Magnetic ordering in EuO/data",
    "maleic-mu-kinetics": "Chemistry/Muonium reaction with maleic acid/Data",
    "ferromagnetic-nickel": "Magnetism/Ferromagnetic nickel/Data",
    "fmuf-ptfe": "Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data",
    "spin-glass-ymnal": "Magnetism/Spin Glass YMnAl/data",
    "high-tc-cuprate": "Superconductivity/A high-Tc cuprate/Data",
    "copper-diffusion": "Nuclear magnetism and ionic motion/Muon diffusion and QLCR in copper/Data",
    "molecular-antiferromagnet": "Magnetism/A molecular antiferromagnet/Data",
    "spin-peierls": "Magnetism/A spin-Peierls transition/Data",
    "afm-high-tf-mdu": "Magnetism/AFM transition in high TF/data",
    "alc-tcnq": "Chemistry/ALC resonance in TCNQ/Data",
    "ionic-motion-llz": "Nuclear magnetism and ionic motion/Ionic motion in a solid electrolyte/Data",
    "photo-musr-silicon": "Semiconductors/Photo-muSR in silicon/Data",
    "cds-fourier": "Semiconductors/Shallow donor state in cadmium sulphide/Data",
}

#: Named waves: the trend-fit cases the 2026-09-23/24 loop iterated on, the
#: Tier A regression set, and the two hold-outs never tuned against.
SETS: dict[str, tuple[str, ...]] = {
    "trend-fit": ("plateau-redfield", "sn-critical-field", "euo-psi", "maleic-mu-kinetics"),
    "tier-a": ("ferromagnetic-nickel", "fmuf-ptfe", "spin-glass-ymnal", "high-tc-cuprate"),
    "hold-out": ("copper-diffusion", "molecular-antiferromagnet"),
}


def run_case(case: str, corpus: Path, out: Path, model: str) -> tuple[str, int]:
    """One ``run_eval.py`` for *case*, its console output kept in ``<out>/<case>.log``."""
    with (out / f"{case}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "run_eval.py"),
                "--data",
                str(corpus / CASES[case]),
                "--rubric",
                case,
                "--model",
                model,
                "--out",
                str(out / case),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return case, completed.returncode


def report(case: str, returncode: int, out: Path) -> str:
    """One line: the case, its exit code, and the wall time and cost it recorded."""
    cost_path = out / case / "cost.json"
    if not cost_path.is_file():
        return f"{case}: exit {returncode}, no cost.json (see {out / f'{case}.log'})"
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    return f"{case}: exit {returncode}, {cost['wall_seconds']:.0f} s, ${cost['total_cost_usd']:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        choices=sorted(SETS),
        help="A named wave of cases (repeatable)",
    )
    parser.add_argument(
        "--case", action="append", default=[], choices=sorted(CASES), help="A case (repeatable)"
    )
    parser.add_argument("--out", required=True, help="Output root; one directory per case")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="The corpus root")
    parser.add_argument("--model", default="sonnet", help="Model alias for the agent")
    parser.add_argument("--parallel", type=int, default=4, help="Cases run at once")
    parser.add_argument("--keep-data", action="store_true", help="Keep the per-run dataset copies")
    args = parser.parse_args(argv)

    cases = list(dict.fromkeys([*(case for name in args.set for case in SETS[name]), *args.case]))
    if not cases:
        parser.error("name a --set or at least one --case")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus).expanduser()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(lambda case: run_case(case, corpus, out, args.model), cases))
    for case, returncode in results:
        print(report(case, returncode, out))
        if not args.keep_data:
            for copy in ("data", "workdir"):
                shutil.rmtree(out / case / copy, ignore_errors=True)
    return max(returncode for _, returncode in results)


if __name__ == "__main__":
    sys.exit(main())
