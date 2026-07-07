#!/usr/bin/env python
"""Load/reduction hot-path benchmark for the Asymmetry performance programme.

This script captures a re-runnable, pre-optimisation baseline for the core
load-and-reduce pipeline so a multi-PR optimisation effort can be gated on
concrete numbers. It imports from :mod:`asymmetry.core` only — no Qt, no
matplotlib, no ``asymmetry.gui`` — so it stays a pure-engine measurement.

What it times
-------------
* **Real-file loads** — ``asymmetry.core.io.load`` through the loader registry
  for each ``--files`` path (records histogram count, bin count, wall time, RSS).
* **Reduction stages** — each of the five hot-path stages, timed *separately*,
  on the first real file **and** on synthetic ROOT-scale runs:

  1. ``resolve_effective_grouping`` (auto-detect t0 policy),
  2. ``apply_grouping_aligned`` over a half-split of the detectors,
  3. ``prepare_histograms_with_deadtime``,
  4. ``reduce_grouped_asymmetry``,
  5. ``build_grouped_time_domain_datasets``.

* **Synthetic runs** at ROOT-instrument scale on a ``(n_det, n_bins)`` grid.
  Each synthetic scenario is skipped (and the skip reported) when its estimated
  peak (float64 counts × 3 for transient copies) would exceed 50 % of physical
  RAM.

Isolation and repeats
---------------------
Every scenario is measured in a **fresh subprocess** (the script re-executes
itself with ``--scenario NAME --json``) so peak-RSS high-water marks never
contaminate one another. Each scenario is repeated ``--repeats`` times (default
3), once per subprocess; the parent reports the **median** wall time and the
**max** peak RSS across repeats.

Example
-------
::

    .venv/bin/python tools/perf_benchmark.py \\
        --files run_a.mdu run_b.mdu run_c.bin run_d.nxs \\
        --output baseline_perf.json \\
        --repeats 3

The child (subprocess) form is an implementation detail but is stable::

    .venv/bin/python tools/perf_benchmark.py --scenario 'load|0' --json \\
        --files run_a.mdu ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

import numpy as np

from asymmetry.core.data.dataset import Histogram, MuonDataset, Run
from asymmetry.core.fitting.grouped_time_domain import build_grouped_time_domain_datasets
from asymmetry.core.io import load
from asymmetry.core.project.profiles import (
    GroupingProfile,
    ProfileFingerprint,
    T0Policy,
    resolve_effective_grouping,
)
from asymmetry.core.transform.deadtime import prepare_histograms_with_deadtime
from asymmetry.core.transform.grouping import (
    apply_grouping_aligned,
    common_t0_for_groups,
    effective_group_indices,
)
from asymmetry.core.transform.reduce import reduce_grouped_asymmetry

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Reduction stages timed separately, in pipeline order.
STAGES: tuple[str, ...] = (
    "resolve_grouping",
    "apply_grouping",
    "deadtime",
    "reduce",
    "grouped_time_domain",
)

#: Synthetic ROOT-scale grid: ``(n_detectors, n_bins)``.
SYNTHETIC_GRID: tuple[tuple[int, int], ...] = (
    (32, 2_000_000),
    (128, 1_000_000),
    (128, 4_000_000),
)

#: Transient-copy multiplier for the synthetic memory-safety estimate.
SYNTHETIC_PEAK_FACTOR = 3.0

#: Fraction of physical RAM a synthetic scenario's estimated peak may not exceed.
RAM_SAFETY_FRACTION = 0.5

#: Deadtime injected into synthetic/real groupings so the correction path
#: actually executes (µs, one per histogram).
DEADTIME_US = 0.01

#: Muon lifetime (µs) used to shape synthetic decay histograms.
TAU_MU_US = 2.197


# --------------------------------------------------------------------------- #
# Platform helpers
# --------------------------------------------------------------------------- #


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in **bytes**.

    ``resource.getrusage`` reports ``ru_maxrss`` in **bytes** on macOS but in
    **kibibytes** on Linux, so the platform is normalised explicitly — the
    committed baseline (captured on macOS) and CI (Linux) must be comparable.
    """
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if sys.platform == "darwin" else ru * 1024)


def physical_ram_bytes() -> int:
    """Return total physical RAM in bytes (``hw.memsize`` on macOS)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def git_head() -> str:
    """Return the current commit SHA, or ``"unknown"``.

    Read-only (no index lock), so it is safe even while another process stages
    changes in the same checkout. Any failure degrades to ``"unknown"``.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def machine_info() -> dict[str, Any]:
    """Collect host machine facts recorded alongside the results."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": physical_ram_bytes(),
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- #
# Synthetic run + profile construction
# --------------------------------------------------------------------------- #


def build_synthetic_run(n_det: int, n_bins: int) -> Run:
    """Build a Run of ``n_det`` realistically-shaped decay histograms.

    Each histogram is a flat pre-t0 baseline, a sharp rise at t0, then an
    exponential muon decay. Per-detector t0 jitter (a few bins) exercises the
    alignment path in :func:`apply_grouping_aligned`. The counts are float64
    so the memory estimate matches reality.
    """
    bin_width = 0.016  # µs (16 ns), typical of a continuous-source TDC
    base_t0 = 128
    idx = np.arange(n_bins, dtype=np.float64)
    histograms: list[Histogram] = []
    for det in range(n_det):
        t0 = base_t0 + (det % 5)
        rel = (idx - t0) * bin_width
        counts = np.where(
            idx >= t0,
            1000.0 * np.exp(-np.maximum(rel, 0.0) / TAU_MU_US),
            5.0,
        )
        histograms.append(
            Histogram(
                counts=counts,
                bin_width=bin_width,
                t0_bin=t0,
                good_bin_start=base_t0,
                good_bin_end=n_bins - 1,
            )
        )
    run = Run(
        run_number=999_000 + n_det,
        histograms=histograms,
        metadata={"instrument": "SYNTH", "title": f"synthetic {n_det}x{n_bins}"},
        grouping={
            "t0_bin": base_t0,
            "first_good_bin": base_t0,
            "last_good_bin": n_bins - 1,
            "good_frames": 1000.0,
        },
    )
    return run


def make_profile(n_hist: int, instrument: str) -> GroupingProfile:
    """Build a two-group profile (forward/backward half-split), auto-detect t0.

    Detector numbers are **1-based** (the grouping convention
    :func:`resolve_group_indices` decodes). ``included_groups`` is set
    explicitly so ``resolve_effective_grouping`` emits the key the grouped
    time-domain build requires.
    """
    half = max(1, n_hist // 2)
    forward = list(range(1, half + 1))
    backward = list(range(half + 1, n_hist + 1))
    return GroupingProfile(
        name="perf-benchmark",
        fingerprint=ProfileFingerprint(instrument=instrument, histogram_count=n_hist),
        groups={1: forward, 2: backward},
        group_names={1: "Forward", 2: "Backward"},
        included_groups={1: True, 2: True},
        forward_group=1,
        backward_group=2,
        t0_policy=T0Policy(mode="auto_detect"),
    )


def _inject_deadtime(grouping: dict[str, Any], n_hist: int) -> None:
    """Add a nonzero per-detector deadtime table so the correction runs."""
    grouping["dead_time_us"] = [DEADTIME_US] * n_hist


def _run_for_scenario(spec: str, files: list[str]) -> tuple[Run, str]:
    """Resolve a reduction scenario's Run + a human-readable dataset label."""
    kind, payload, _stage = _parse_scenario(spec)
    if kind == "reduce_real":
        result = load(files[int(payload)])
        dataset = result[0] if isinstance(result, list) else result
        if dataset.run is None:
            raise ValueError(f"{files[int(payload)]!r} loaded without a source run")
        return dataset.run, os.path.basename(files[int(payload)])
    n_det, n_bins = (int(v) for v in payload.split("x"))
    return build_synthetic_run(n_det, n_bins), payload


# --------------------------------------------------------------------------- #
# Scenario parsing / enumeration
# --------------------------------------------------------------------------- #


def _parse_scenario(spec: str) -> tuple[str, str, str]:
    """Split a ``kind|payload[|stage]`` scenario id into its parts."""
    parts = spec.split("|")
    kind = parts[0]
    payload = parts[1] if len(parts) > 1 else ""
    stage = parts[2] if len(parts) > 2 else ""
    return kind, payload, stage


def enumerate_scenarios(files: list[str]) -> list[str]:
    """Build the full scenario id list for a given ``--files`` set."""
    scenarios: list[str] = [f"load|{i}" for i in range(len(files))]
    if files:
        scenarios += [f"reduce_real|0|{stage}" for stage in STAGES]
    for n_det, n_bins in SYNTHETIC_GRID:
        scenarios += [f"reduce_synth|{n_det}x{n_bins}|{stage}" for stage in STAGES]
    return scenarios


# --------------------------------------------------------------------------- #
# Child: run one scenario once, emit JSON on stdout
# --------------------------------------------------------------------------- #


def _time_call(fn) -> tuple[float, Any]:
    """Time a zero-arg callable, returning ``(wall_ms, result)``."""
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result


def run_scenario_once(spec: str, files: list[str]) -> dict[str, Any]:
    """Execute one scenario a single time and return a measurement dict.

    Setup (loading / synthetic construction / grouping resolution) is performed
    but **not** timed; only the target stage call is measured. ``peak_rss_bytes``
    is read after the call so it reflects the setup + target high-water mark for
    this isolated process.
    """
    kind, payload, stage = _parse_scenario(spec)

    if kind == "load":
        path = files[int(payload)]
        wall_ms, result = _time_call(lambda: load(path))
        dataset = result[0] if isinstance(result, list) else result
        run = dataset.run
        n_hist = len(run.histograms) if run is not None else 0
        n_bins = run.histograms[0].n_bins if run is not None and run.histograms else 0
        return {
            "wall_ms": wall_ms,
            "peak_rss_bytes": peak_rss_bytes(),
            "n_histograms": n_hist,
            "n_bins": n_bins,
            "dataset": os.path.basename(path),
        }

    # Reduction scenarios (real or synthetic) --------------------------------
    run, label = _run_for_scenario(spec, files)
    n_hist = len(run.histograms)
    n_bins = run.histograms[0].n_bins if run.histograms else 0
    instrument = str(run.grouping.get("instrument") or run.metadata.get("instrument") or "")
    profile = make_profile(n_hist, instrument)

    # Stage 1 is the resolve call itself; the others treat it as setup.
    if stage == "resolve_grouping":
        wall_ms, _ = _time_call(lambda: resolve_effective_grouping(profile, run))
        peak = peak_rss_bytes()
    else:
        grouping = resolve_effective_grouping(profile, run)
        _inject_deadtime(grouping, n_hist)
        forward_idx = effective_group_indices(grouping, 1, n_histograms=n_hist)
        backward_idx = effective_group_indices(grouping, 2, n_histograms=n_hist)
        alpha = float(grouping.get("alpha", 1.0))

        if stage == "apply_grouping":
            common_t0 = common_t0_for_groups(run.histograms, forward_idx, backward_idx)
            wall_ms, _ = _time_call(
                lambda: apply_grouping_aligned(run.histograms, forward_idx, common_t0_bin=common_t0)
            )
        elif stage == "deadtime":
            wall_ms, _ = _time_call(
                lambda: prepare_histograms_with_deadtime(run.histograms, grouping, True)
            )
        elif stage == "reduce":
            wall_ms, _ = _time_call(
                lambda: reduce_grouped_asymmetry(
                    histograms=run.histograms,
                    grouping=grouping,
                    forward_idx=forward_idx,
                    backward_idx=backward_idx,
                    alpha=alpha,
                    use_deadtime=True,
                    deadtime_mode="file",
                    use_background=False,
                )
            )
        elif stage == "grouped_time_domain":
            reduction = reduce_grouped_asymmetry(
                histograms=run.histograms,
                grouping=grouping,
                forward_idx=forward_idx,
                backward_idx=backward_idx,
                alpha=alpha,
                use_deadtime=True,
                deadtime_mode="file",
                use_background=False,
            )
            run.grouping = grouping
            dataset = MuonDataset(
                time=reduction.time,
                asymmetry=reduction.asymmetry,
                error=reduction.error,
                metadata=dict(run.metadata),
                run=run,
            )
            wall_ms, _ = _time_call(lambda: build_grouped_time_domain_datasets(dataset))
        else:
            raise ValueError(f"unknown stage {stage!r}")
        peak = peak_rss_bytes()

    return {
        "wall_ms": wall_ms,
        "peak_rss_bytes": peak,
        "n_histograms": n_hist,
        "n_bins": n_bins,
        "dataset": label,
    }


# --------------------------------------------------------------------------- #
# Parent: subprocess orchestration + aggregation
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioResult:
    """Aggregated result for one scenario across repeats."""

    scenario: str
    kind: str
    stage: str
    status: str  # "ok" | "skipped" | "failed"
    dataset: str = ""
    n_histograms: int = 0
    n_bins: int = 0
    median_wall_ms: float | None = None
    max_rss_bytes: int | None = None
    reason: str = ""
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "kind": self.kind,
            "stage": self.stage,
            "status": self.status,
            "dataset": self.dataset,
            "n_histograms": self.n_histograms,
            "n_bins": self.n_bins,
            "median_wall_ms": self.median_wall_ms,
            "max_rss_bytes": self.max_rss_bytes,
            "reason": self.reason,
            "runs": self.runs,
        }


def _synthetic_skip_reason(spec: str, ram_bytes: int) -> str | None:
    """Return a skip reason if a synthetic scenario would blow the RAM budget."""
    kind, payload, _stage = _parse_scenario(spec)
    if kind != "reduce_synth" or ram_bytes <= 0:
        return None
    n_det, n_bins = (int(v) for v in payload.split("x"))
    est_peak = n_det * n_bins * 8 * SYNTHETIC_PEAK_FACTOR
    budget = ram_bytes * RAM_SAFETY_FRACTION
    if est_peak > budget:
        return (
            f"estimated peak {est_peak / 1e9:.1f} GB exceeds "
            f"{RAM_SAFETY_FRACTION:.0%} of {ram_bytes / 1e9:.1f} GB RAM"
        )
    return None


def _invoke_child(spec: str, files: list[str], timeout: float) -> dict[str, Any]:
    """Run one scenario in a subprocess, returning its parsed JSON dict.

    Raises on nonzero exit or unparseable stdout; the stderr text is attached to
    the exception message so the retry logic can spot ImportError/SyntaxError
    from a mid-edit source tree.
    """
    cmd = [sys.executable, os.path.abspath(__file__), "--scenario", spec, "--json"]
    if files:
        cmd += ["--files", *files]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}: {proc.stderr.strip()[-500:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"unparseable stdout: {exc}: {proc.stdout.strip()[:200]}"
            f" | stderr: {proc.stderr.strip()[-300:]}"
        ) from exc


def _is_transient_error(message: str) -> bool:
    """Whether an error looks like a mid-edit import/syntax failure worth retrying."""
    lowered = message.lower()
    return "importerror" in lowered or "syntaxerror" in lowered or "modulenotfound" in lowered


def run_repeats(
    spec: str, files: list[str], repeats: int, timeout: float
) -> tuple[list[dict[str, Any]], str]:
    """Run a scenario ``repeats`` times, retrying transient import/syntax deaths.

    Returns ``(runs, error)``: ``runs`` holds the per-repeat measurement dicts
    (empty on failure) and ``error`` is a message string when the scenario
    ultimately failed, else ``""``.
    """
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        attempt = 0
        while True:
            attempt += 1
            try:
                runs.append(_invoke_child(spec, files, timeout))
                break
            except subprocess.TimeoutExpired:
                return runs, f"timed out after {timeout:.0f}s"
            except RuntimeError as exc:
                message = str(exc)
                if _is_transient_error(message) and attempt <= 3:
                    print(
                        f"  [retry {attempt}/3] transient failure on {spec}: {message[:120]}",
                        file=sys.stderr,
                    )
                    time.sleep(30)
                    continue
                return runs, message
    return runs, ""


def aggregate(spec: str, runs: list[dict[str, Any]], error: str) -> ScenarioResult:
    """Fold per-repeat measurements into a single :class:`ScenarioResult`."""
    kind, _payload, stage = _parse_scenario(spec)
    if error or not runs:
        return ScenarioResult(
            scenario=spec, kind=kind, stage=stage, status="failed", reason=error or "no runs"
        )
    walls = [float(r["wall_ms"]) for r in runs]
    rss = [int(r["peak_rss_bytes"]) for r in runs]
    first = runs[0]
    return ScenarioResult(
        scenario=spec,
        kind=kind,
        stage=stage,
        status="ok",
        dataset=str(first.get("dataset", "")),
        n_histograms=int(first.get("n_histograms", 0)),
        n_bins=int(first.get("n_bins", 0)),
        median_wall_ms=float(median(walls)),
        max_rss_bytes=max(rss),
        runs=runs,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def _fmt_mb(value: int | None) -> str:
    return "—" if value is None else f"{value / 1e6:,.1f}"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the results payload as a human-readable markdown report."""
    machine = payload["machine"]
    lines: list[str] = []
    lines.append("# Asymmetry load/reduction baseline\n")
    lines.append(f"- Generated: {payload['generated']}")
    lines.append(f"- Commit: `{payload['git_head']}`")
    lines.append(f"- Repeats: {payload['repeats']} (median wall, max RSS)")
    lines.append(
        f"- Machine: {machine['platform']} | {machine['cpu_count']} CPU | "
        f"{machine['ram_bytes'] / 1e9:.1f} GB RAM | "
        f"Python {machine['python']} | numpy {machine['numpy']}"
    )
    lines.append("")

    header = "| Scenario | Dataset | n_hist | n_bins | Median wall (ms) | Max RSS (MB) | Status |"
    sep = "|---|---|---:|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)
    for res in payload["results"]:
        label = res["scenario"]
        status = res["status"]
        if status != "ok":
            note = f"{status}: {res['reason']}" if res["reason"] else status
            lines.append(
                f"| `{label}` | {res['dataset'] or '—'} | "
                f"{res['n_histograms'] or '—'} | {res['n_bins'] or '—'} | — | — | {note} |"
            )
            continue
        lines.append(
            f"| `{label}` | {res['dataset']} | {res['n_histograms']} | "
            f"{res['n_bins']:,} | {_fmt_ms(res['median_wall_ms'])} | "
            f"{_fmt_mb(res['max_rss_bytes'])} | ok |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def run_parent(args: argparse.Namespace) -> int:
    """Enumerate + drive all scenarios, then print markdown and write JSON."""
    files = args.files or []
    for path in files:
        if not os.path.exists(path):
            print(f"warning: input file not found: {path}", file=sys.stderr)
    ram_bytes = physical_ram_bytes()
    scenarios = enumerate_scenarios(files)

    results: list[ScenarioResult] = []
    for spec in scenarios:
        skip = _synthetic_skip_reason(spec, ram_bytes)
        kind, _payload, stage = _parse_scenario(spec)
        if skip is not None:
            print(f"[skip] {spec}: {skip}", file=sys.stderr)
            results.append(
                ScenarioResult(scenario=spec, kind=kind, stage=stage, status="skipped", reason=skip)
            )
            continue
        print(f"[run ] {spec}", file=sys.stderr)
        runs, error = run_repeats(spec, files, args.repeats, args.timeout)
        result = aggregate(spec, runs, error)
        if result.status != "ok":
            print(f"       -> {result.status}: {result.reason}", file=sys.stderr)
        results.append(result)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": git_head(),
        "repeats": args.repeats,
        "machine": machine_info(),
        "results": [r.to_dict() for r in results],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.output}", file=sys.stderr)

    print(render_markdown(payload))
    return 0


def run_child(args: argparse.Namespace) -> int:
    """Execute a single scenario once and print its JSON dict on stdout."""
    measurement = run_scenario_once(args.scenario, args.files or [])
    sys.stdout.write(json.dumps(measurement))
    sys.stdout.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Real data files to load/reduce (first is used for real reduction).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON results file (parent mode).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per scenario (default 3).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-subprocess timeout in seconds (default 600).",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Internal: run a single scenario in child mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Internal: child mode marker (emit JSON on stdout).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenario is not None:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
