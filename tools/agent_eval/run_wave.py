#!/usr/bin/env python3
"""Run a wave of Claude Code or Codex evaluations in parallel.

A pass of the evaluation loop is a named set of (dataset, rubric) cases run
side by side into one output root, ``<out>/<case>/``. Each case is an ordinary
single-dataset runner invocation; this script fans them out, waits, prints
each case's exit code, wall time and cost, and deletes the per-run dataset
copies (``data/``, ``workdir/``) — scoring reads the summaries and transcripts, and the
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
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

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
    "lifeas-psi": "Superconductivity/LiFeAs/data",
    "trsb-re6zr": "Superconductivity/TRSB/data",
    "basics": "Basics/data",
    "corannulene-alc": "Chemistry/Molecular dynamics of corannulene/data",
    "benzene-high-tf": "Chemistry/Muon spectroscopy of benzene/data/High TF rotation",
    "benzene-rf": "Chemistry/Muon spectroscopy of benzene/data/RF resonance",
    "benzene-repolarisation": "Chemistry/Muon spectroscopy of benzene/data/Repolarisation",
    "benzene-alc": "Chemistry/Muon spectroscopy of benzene/data/ALC resonance",
}

#: Named waves: the trend-fit cases the 2026-09-23/24 loop iterated on, the
#: Tier A regression set, and the two hold-outs never tuned against.
SETS: dict[str, tuple[str, ...]] = {
    "trend-fit": ("plateau-redfield", "sn-critical-field", "euo-psi", "maleic-mu-kinetics"),
    "tier-a": ("ferromagnetic-nickel", "fmuf-ptfe", "spin-glass-ymnal", "high-tc-cuprate"),
    "hold-out": ("copper-diffusion", "molecular-antiferromagnet"),
    "workflow": ("alc-tcnq", "ionic-motion-llz", "photo-musr-silicon", "cds-fourier"),
    "tier-b": ("spin-peierls", "afm-high-tf-mdu"),
    "corpus-2026": (
        "lifeas-psi",
        "trsb-re6zr",
        "basics",
        "corannulene-alc",
        "benzene-high-tf",
        "benzene-rf",
        "benzene-repolarisation",
        "benzene-alc",
    ),
}

SETS["all"] = tuple(CASES)
SETS["codex-gate"] = (*SETS["tier-a"], *SETS["workflow"])
SETS["codex-expansion"] = tuple(case for case in CASES if case not in SETS["codex-gate"])
RUNNERS = {"claude": "run_eval.py", "codex": "run_codex_eval.py"}


def run_case(case: str, corpus: Path, out: Path, model: str | None, agent: str) -> tuple[str, int]:
    """Run *case* on the selected host, keeping its console output beside the results."""
    with (out / f"{case}.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / RUNNERS[agent]),
                "--data",
                str(corpus / CASES[case]),
                "--rubric",
                case,
                *(["--model", model] if model is not None else []),
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
    label = f"{cost['agent']}/{cost['model']}"
    price = cost.get("total_cost_usd")
    billing = f", ${price:.2f}" if price is not None else ""
    return f"{case}: {label}, exit {returncode}, {cost['wall_seconds']:.0f} s{billing}"


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
    parser.add_argument(
        "--corpus",
        default=os.environ.get("ASYMMETRY_CORPUS_ROOT"),
        help="The corpus root (default: $ASYMMETRY_CORPUS_ROOT)",
    )
    parser.add_argument(
        "--agent", choices=sorted(RUNNERS), default="claude", help="Evaluation host"
    )
    parser.add_argument("--model", help="Model (default: sonnet for Claude, gpt-6-luna for Codex)")
    parser.add_argument("--parallel", type=int, default=1, help="Cases run at once (default: 1)")
    parser.add_argument("--keep-data", action="store_true", help="Keep the per-run dataset copies")
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error("--parallel must be positive")

    cases = list(dict.fromkeys([*(case for name in args.set for case in SETS[name]), *args.case]))
    if not cases:
        parser.error("name a --set or at least one --case")
    if args.corpus is None:
        parser.error("pass --corpus or set ASYMMETRY_CORPUS_ROOT")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus).expanduser()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(
            pool.map(lambda case: run_case(case, corpus, out, args.model, args.agent), cases)
        )
    for case, returncode in results:
        print(report(case, returncode, out))
        if not args.keep_data:
            for copy in ("data", "workdir"):
                shutil.rmtree(out / case / copy, ignore_errors=True)
    return max(returncode for _, returncode in results)


if __name__ == "__main__":
    sys.exit(main())
