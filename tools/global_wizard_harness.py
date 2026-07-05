#!/usr/bin/env python3
"""Golden-verdict regression harness for the global-fit wizard (PR 1).

This is the acceptance gate every later global-fit-wizard efficiency PR is
measured against. It freezes *Exhaustive* wizard verdicts on a small set of
known-ground-truth synthetic cases (plus optional real-corpus cases) and, for
any candidate wizard configuration, reports per case:

    verdict-agreement % vs the frozen baseline, IC gap on disagreement,
    agreement vs planted truth (synthetics), wall time, Minuit fit / feval
    counts, and a status in {OK, TIMEOUT, ERROR}.

Design notes (see ``docs/porting/global-fit-wizard-efficiency/test-data.md``):

* **Core-import-only.** This module imports only ``asymmetry.core`` — no Qt, no
  matplotlib — so it is fast and headless.
* **Freeze once, diff per run.** ``--freeze`` runs full Exhaustive across the
  case set a single time and writes ``baseline/*.json`` (compact verdicts + ICs
  + fit counts + provenance). Every later run loads that baseline and runs only
  the candidate config, diffing against the cached verdicts.
* **Under-10-min guard.** Each case runs in its *own* subprocess with a per-case
  timeout and a hard overall wall guard. On a per-case breach the case reports
  ``TIMEOUT`` and the harness continues — it never blocks. The wizard opens its
  own ``ProcessPoolExecutor`` internally, so the case subprocess makes itself a
  session/process-group leader and on timeout the *whole group* is reaped in one
  shot (``os.killpg`` on POSIX, ``taskkill /F /T`` on Windows) — the grandchild
  pool workers die with it, no orphans.
* **Effort tiers.** ``--tier {low,balanced,thorough,exhaustive}`` selects a
  wizard-configuration callable. The tiers do not exist in the wizard yet (they
  arrive in PR 5); for now every tier aliases the current Exhaustive behaviour,
  so the freeze/diff plumbing and case set are ready without blocking on the
  tiers themselves.

macOS uses the ``spawn`` start method, which re-imports this module in every
child, so everything heavy lives behind ``if __name__ == "__main__"`` and the
case payloads are plain picklable objects.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Paths / provenance
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_DIR = _REPO_ROOT / "docs" / "porting" / "global-fit-wizard-efficiency" / "baseline"
_BASELINE_PATH = _BASELINE_DIR / "exhaustive_baseline.json"

#: Schema version for the frozen baseline artifact. Bump on breaking changes.
BASELINE_SCHEMA_VERSION = 1

#: Env var that opts real-corpus cases in. Synthetics-only by default so the
#: harness runs headless in CI without the data checkout.
CORPUS_ENV_VAR = "ASYMMETRY_WIZARD_HARNESS_CORPUS"

#: IC-gap tolerance (metric units) below which a verdict disagreement is
#: considered "inside the robustness delta" — a soft signal, not a hard fail.
DEFAULT_IC_GAP_TOLERANCE = 2.0

TIERS = ("low", "balanced", "thorough", "exhaustive")


# --------------------------------------------------------------------------- #
# Synthetic case definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyntheticCase:
    """A synthetic global-fit case with planted ground-truth parameter roles.

    ``components``/``operators`` define the :class:`CompositeModel` used both to
    generate the data and (via ``template_keys``) to bound the wizard's template
    shortlist so Exhaustive stays inside the time budget while still exercising
    the same role-search code paths. ``planted_roles`` maps every model
    parameter to its intended verdict, ``"global"`` (identical across all G
    members) or ``"local"`` (varies along a smooth scan).
    """

    case_id: str
    components: tuple[str, ...]
    operators: tuple[str, ...]
    template_keys: tuple[str, ...]
    n_groups: int
    global_values: dict[str, float]
    #: local param name -> the G planted values (one per group, along the scan)
    local_scans: dict[str, tuple[float, ...]]
    planted_roles: dict[str, str]
    noise_level: float = 0.004
    n_points: int = 120
    seed: int = 12345
    description: str = ""

    @property
    def n_params(self) -> int:
        return len(self.global_values) + len(self.local_scans)

    def param_names(self) -> tuple[str, ...]:
        return tuple(self.global_values) + tuple(self.local_scans)


def synthetic_cases() -> tuple[SyntheticCase, ...]:
    """Return the frozen synthetic case set.

    Coverage (never P=7): pure-global, pure-local, mixed, correlated-pair, and
    near-degenerate — with G in {3, 4} and P in {3, 4, 5}. P=5/G=3 exercises the
    same paths as the P=7 monster at a fraction of the wall cost.
    """

    return (
        # --- pure-global: every param shared, no scan (P=3, G=3) --------------
        SyntheticCase(
            case_id="pure_global_exp_g3",
            components=("Exponential", "Constant"),
            operators=("+",),
            template_keys=("exp_constant",),
            n_groups=3,
            global_values={"A_1": 0.20, "Lambda": 0.35, "A_bg": 0.010},
            local_scans={},
            planted_roles={"A_1": "global", "Lambda": "global", "A_bg": "global"},
            seed=1001,
            description="All parameters shared; identical across the series.",
        ),
        # --- pure-local: rate varies strongly along the scan (P=3, G=4) -------
        SyntheticCase(
            case_id="pure_local_exp_g4",
            components=("Exponential", "Constant"),
            operators=("+",),
            template_keys=("exp_constant",),
            n_groups=4,
            global_values={},
            local_scans={
                "A_1": (0.24, 0.20, 0.16, 0.12),
                "Lambda": (0.15, 0.30, 0.55, 0.90),
                "A_bg": (0.005, 0.010, 0.015, 0.020),
            },
            planted_roles={"A_1": "local", "Lambda": "local", "A_bg": "local"},
            seed=1002,
            description="Every parameter drifts along the scan; nothing shared.",
        ),
        # --- mixed: shared amplitude/bg, local relaxation rate (P=3, G=4) -----
        SyntheticCase(
            case_id="mixed_exp_g4",
            components=("Exponential", "Constant"),
            operators=("+",),
            template_keys=("exp_constant",),
            n_groups=4,
            global_values={"A_1": 0.20, "A_bg": 0.010},
            local_scans={"Lambda": (0.15, 0.28, 0.50, 0.85)},
            planted_roles={"A_1": "global", "Lambda": "local", "A_bg": "global"},
            # Seed chosen for comfortable residual runs-test margin on every
            # group (max |z| ~0.76, threshold 2.0). Forcing A_1/A_bg global
            # leaves less freedom to absorb a group's noise draw, so a borderline
            # draw (max |z| ~2.37) can trip the continuity gate and null the
            # winner — a whiter draw keeps the mixed verdict stable cross-platform.
            seed=11003,
            description="Shared amplitude + background; only the rate scans.",
        ),
        # --- correlated-pair: two components with related, both-local rates
        #     (P=5, G=3) — stresses E/F/G interaction handling ----------------
        SyntheticCase(
            case_id="correlated_pair_biexp_g3",
            components=("Exponential", "Gaussian", "Constant"),
            operators=("+", "+"),
            template_keys=("exp_gaussian_constant", "exp_constant", "gaussian_constant"),
            n_groups=3,
            global_values={"A_1": 0.14, "A_2": 0.12, "A_bg": 0.010},
            local_scans={
                # correlated pair: both broaden together along the scan
                "Lambda": (0.20, 0.40, 0.75),
                "sigma": (0.18, 0.30, 0.48),
            },
            planted_roles={
                "A_1": "global",
                "Lambda": "local",
                "A_2": "global",
                "sigma": "local",
                "A_bg": "global",
            },
            seed=1004,
            description="Two dynamic rates that scan together (correlated pair).",
        ),
        # --- near-degenerate: two exponentials with near-equal rates
        #     (P=5, G=3) — stresses identifiability demotion -----------------
        SyntheticCase(
            case_id="near_degenerate_biexp_g3",
            components=("Exponential", "Exponential", "Constant"),
            operators=("+", "+"),
            template_keys=("biexp_constant", "exp_constant"),
            n_groups=3,
            global_values={
                "A_1": 0.13,
                "Lambda_1": 0.30,
                "A_2": 0.12,
                "Lambda_2": 0.31,  # tightly near-degenerate with Lambda_1
                "A_bg": 0.010,
            },
            local_scans={},
            # NOTE: these planted roles describe the *generative* biexp model.
            # The correct Exhaustive verdict is the *demotion* to a single
            # exponential (``exp_constant``): the near-degenerate biexp fit fails
            # the residual/continuity gate (unstable covariance), so no biexp
            # candidate wins. This case's "ground truth" is the demotion itself,
            # which is why it is excluded from the planted-truth recovery check.
            planted_roles={
                "A_1": "global",
                "Lambda_1": "global",
                "A_2": "global",
                "Lambda_2": "global",
                "A_bg": "global",
            },
            seed=1005,
            description=(
                "Two near-equal exponential rates (near-degenerate); the correct "
                "verdict is demotion to a single exponential."
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Synthetic data generation (runs in the case subprocess)
# --------------------------------------------------------------------------- #


def _build_case_datasets(case: SyntheticCase) -> list[Any]:
    """Materialise the ``MuonDataset`` list for a synthetic case.

    Deferred core imports keep this module import-light for the CLI parent and
    for the always-on CI test (which never generates data).
    """

    from asymmetry.core.data.dataset import MuonDataset
    from asymmetry.core.fitting.composite import CompositeModel

    model = CompositeModel(list(case.components), operators=list(case.operators))
    time_axis = np.linspace(0.0, 8.0, case.n_points)
    datasets: list[Any] = []
    for group_index in range(case.n_groups):
        params: dict[str, float] = dict(case.global_values)
        for name, scan in case.local_scans.items():
            params[name] = float(scan[group_index])
        clean = model.function(time_axis, **params)
        # Deterministic per-group noise, pinned to the case seed.
        rng = np.random.default_rng(case.seed * 1000 + group_index)
        noisy = clean + rng.normal(0.0, case.noise_level, size=time_axis.shape)
        run_number = case.seed * 10 + group_index
        datasets.append(
            MuonDataset(
                time=time_axis,
                asymmetry=noisy,
                error=np.full_like(time_axis, case.noise_level),
                metadata={
                    "run_number": run_number,
                    "field": 50.0 * (group_index + 1),
                    "temperature": 5.0,
                    "run_label": str(run_number),
                },
            )
        )
    return datasets


# --------------------------------------------------------------------------- #
# Tier -> wizard-configuration callables
# --------------------------------------------------------------------------- #
#
# A tier maps to a callable that runs the wizard with a particular configuration
# and returns the ``GlobalFitWizardRecommendation`` plus its instrumentation
# dict. Today every tier aliases the current Exhaustive behaviour (PR 5 adds the
# real tier policy); the seam is what PR 1 delivers.


def _run_wizard_exhaustive(
    datasets: list[Any],
    *,
    template_keys: tuple[str, ...],
) -> tuple[Any, dict[str, object]]:
    from asymmetry.core.fitting.global_fit_wizard import (
        build_global_fit_wizard_recommendation,
    )

    instrumentation: dict[str, object] = {}
    recommendation = build_global_fit_wizard_recommendation(
        datasets,
        instrumentation=instrumentation,
        selected_template_keys=template_keys or None,
    )
    return recommendation, instrumentation


def _run_wizard_sleeper(
    datasets: list[Any],
    *,
    template_keys: tuple[str, ...],
) -> tuple[Any, dict[str, object]]:
    """Module-level config that overruns any sane per-case timeout.

    Registered under the ``_timeout_probe`` tier so the ``slow`` test can prove
    the real subprocess kill path (``run_case_isolated`` -> TIMEOUT). It must be
    module-level (not a closure) so the spawned child re-imports and finds it.
    """

    time.sleep(3600.0)
    raise AssertionError("sleeper should have been killed by the timeout guard")


# Every user-facing tier currently aliases exhaustive. PR 5 replaces the aliases
# with real tier-configuration callables; the harness contract does not change.
# ``_timeout_probe`` is a test-only tier for exercising the real kill path.
TIER_CONFIGS: dict[str, Callable[..., tuple[Any, dict[str, object]]]] = {
    "low": _run_wizard_exhaustive,
    "balanced": _run_wizard_exhaustive,
    "thorough": _run_wizard_exhaustive,
    "exhaustive": _run_wizard_exhaustive,
    "_timeout_probe": _run_wizard_sleeper,
}


# --------------------------------------------------------------------------- #
# Verdict extraction
# --------------------------------------------------------------------------- #


def _extract_verdict(recommendation: Any, case: SyntheticCase) -> dict[str, object]:
    """Reduce a recommendation to the compact, diffable verdict fields.

    The verdict is ``(recommended_key, per-parameter role map)`` plus the ICs
    of the winning assessment. Because the harness runs the *real* wizard (no
    monkeypatched template restriction), the winning template key is part of the
    verdict — correlated-pair / near-degenerate cases can flip the template, not
    just roles.
    """

    assessment = recommendation.recommended_assessment
    if assessment is None:
        return {
            "recommended_key": recommendation.recommended_key,
            "template_key": None,
            "roles": {},
            "aic": None,
            "aicc": None,
            "bic": None,
            "n_params": None,
        }

    global_names = set(assessment.global_param_names)
    local_names = set(assessment.local_param_names)
    roles: dict[str, str] = {}
    for name in case.param_names():
        if name in local_names:
            roles[name] = "local"
        elif name in global_names:
            roles[name] = "global"
        else:
            # Parameter absent from the winning template (e.g. a template flip).
            roles[name] = "absent"

    return {
        # Full selection key (includes the g=/l= role suffix) for debugging.
        "recommended_key": recommendation.recommended_key,
        # The winning *template* key alone (before the role suffix) — the
        # template-flip signal, compared independently of the role map so a
        # flip is not double-counted against the per-parameter roles.
        "template_key": assessment.template.key,
        "roles": roles,
        "aic": _finite_or_none(assessment.aic),
        "aicc": _finite_or_none(assessment.aicc),
        "bic": _finite_or_none(assessment.bic),
        "n_params": assessment.parameter_count,
    }


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Case runner (executed inside an isolated subprocess)
# --------------------------------------------------------------------------- #


def _case_worker(case: SyntheticCase, tier: str, out_queue: mp.Queue[dict[str, object]]) -> None:
    """Run one case's wizard config and push a compact result over the queue.

    Runs in a *child* process so a hung fit can be killed without taking the
    harness down. Only picklable primitives cross the boundary.

    On POSIX the child makes itself a new **session/process-group leader**
    (``os.setsid``) *before* the wizard spawns its ``ProcessPoolExecutor``, so
    every grandchild pool worker inherits this group and the parent can reap the
    whole group with a single ``os.killpg`` on timeout. (``mp.Process`` has no
    ``start_new_session`` flag, so the child sets the group itself.)
    """

    _become_group_leader()
    try:
        datasets = _build_case_datasets(case)
        config = TIER_CONFIGS[tier]
        start = time.perf_counter()
        recommendation, instrumentation = config(datasets, template_keys=case.template_keys)
        wall_s = time.perf_counter() - start
        verdict = _extract_verdict(recommendation, case)
        counters = instrumentation.get("counters", {})
        if not isinstance(counters, dict):
            counters = {}
        out_queue.put(
            {
                "status": "OK",
                "verdict": verdict,
                "wall_s": wall_s,
                "minuit_fits": int(counters.get("global_fit_calls", 0) or 0),
                "minuit_fevals": int(counters.get("minuit_function_calls", 0) or 0),
                "exact_fit_invocations": int(counters.get("exact_fit_invocations", 0) or 0),
            }
        )
    except Exception as exc:  # noqa: BLE001 — report, never crash the harness
        import traceback

        out_queue.put(
            {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def run_case_isolated(
    case: SyntheticCase,
    tier: str,
    *,
    timeout_s: float,
) -> dict[str, object]:
    """Run one case in its own session/process group; kill the group on timeout.

    The wizard opens its own ``ProcessPoolExecutor`` inside the child, so a bare
    ``child.terminate()`` would orphan the grandchild pool workers. The child
    makes itself a session/group leader (:func:`_become_group_leader`), so on
    timeout the *whole group* is reaped in one shot — ``os.killpg`` on POSIX,
    ``taskkill /F /T`` on Windows (which has no process groups / ``pgrep``).
    Returns a report dict; on breach ``status == "TIMEOUT"`` and the harness
    continues.
    """

    ctx = mp.get_context("spawn")
    out_queue: mp.Queue[dict[str, object]] = ctx.Queue()
    proc = ctx.Process(target=_case_worker, args=(case, tier, out_queue), daemon=False)
    proc.start()

    proc.join(timeout_s)
    if proc.is_alive():
        _kill_process_group(proc.pid)
        proc.join(10.0)
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)
        _drain_queue(out_queue)
        return {"status": "TIMEOUT", "wall_s": timeout_s}

    # The child has exited, but its queue-feeder thread may still be flushing the
    # payload: use a short blocking get rather than get_nowait so a slow delivery
    # is not mis-reported as "no result".
    try:
        result = out_queue.get(timeout=30.0)
    except Exception:  # noqa: BLE001 — empty/broken queue after a clean exit
        result = {"status": "ERROR", "error": "case subprocess produced no result"}
    _drain_queue(out_queue)
    return result


def _drain_queue(out_queue: mp.Queue[dict[str, object]]) -> None:
    """Best-effort close the queue and join its feeder so no threads linger."""

    try:
        out_queue.close()
        out_queue.join_thread()
    except Exception:  # noqa: BLE001
        pass


def _become_group_leader() -> None:
    """Make the current process a new session/process-group leader (POSIX).

    Called at the top of the child worker so the wizard's pool workers inherit
    the group and can be reaped group-wide. No-op on Windows (no ``setsid``).
    """

    setsid = getattr(os, "setsid", None)
    if setsid is None:
        return
    try:
        setsid()
    except OSError:
        pass


def _kill_process_group(pid: int) -> None:
    """Kill *pid* and every process in its group / tree, cross-platform."""

    if sys.platform == "win32":
        # Windows has no process groups or pgrep; taskkill /T kills the tree.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return

    import signal as signal_module

    # The child called setsid(), so its pgid == its pid; signal the whole group.
    for sig_name in ("SIGTERM", "SIGKILL"):
        sig = getattr(signal_module, sig_name, None)
        if sig is None:
            continue
        try:
            os.killpg(os.getpgid(pid), sig)
        except (OSError, ProcessLookupError):
            # Fall back to signalling the single pid if the group is already gone.
            try:
                os.kill(pid, sig)
            except (OSError, ProcessLookupError):
                pass
        if sig_name == "SIGTERM":
            time.sleep(0.5)


# --------------------------------------------------------------------------- #
# Baseline: freeze + load + provenance
# --------------------------------------------------------------------------- #


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _case_provenance(case: SyntheticCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "components": list(case.components),
        "operators": list(case.operators),
        "template_keys": list(case.template_keys),
        "n_groups": case.n_groups,
        "n_params": case.n_params,
        "seed": case.seed,
        "planted_roles": dict(case.planted_roles),
        "description": case.description,
    }


def freeze_baseline(
    cases: Sequence[SyntheticCase],
    *,
    timeout_s: float,
    generation_date: str,
) -> dict[str, object]:
    """Run full Exhaustive across *cases* once and return the baseline payload."""

    entries: dict[str, object] = {}
    for case in cases:
        report = run_case_isolated(case, "exhaustive", timeout_s=timeout_s)
        entries[case.case_id] = {
            "provenance": _case_provenance(case),
            "verdict": report.get("verdict"),
            "minuit_fits": report.get("minuit_fits"),
            "minuit_fevals": report.get("minuit_fevals"),
            "status": report.get("status"),
            "wall_s": report.get("wall_s"),
        }

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "generation_date": generation_date,
        "tier": "exhaustive",
        "cases": entries,
    }


def load_baseline(path: Path = _BASELINE_PATH) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_baseline(payload: dict[str, object], path: Path = _BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# Diff / report
# --------------------------------------------------------------------------- #


def compare_verdicts(
    candidate: dict[str, object] | None,
    baseline_verdict: dict[str, object] | None,
) -> tuple[float, list[str]]:
    """Return (agreement fraction over params, list of disagreeing params).

    Agreement counts a parameter as matching when its role is identical. The
    winning template key is compared too: a template flip counts as *every*
    parameter disagreeing (roles become meaningless across templates).
    """

    if candidate is None or baseline_verdict is None:
        return 0.0, ["<missing>"]

    cand_roles = dict(candidate.get("roles", {}) or {})
    base_roles = dict(baseline_verdict.get("roles", {}) or {})
    param_names = sorted(set(cand_roles) | set(base_roles))
    if not param_names:
        return 1.0, []

    template_flip = candidate.get("template_key") != baseline_verdict.get("template_key")
    disagreements: list[str] = []
    matches = 0
    for name in param_names:
        if template_flip or cand_roles.get(name) != base_roles.get(name):
            disagreements.append(name)
        else:
            matches += 1
    if template_flip:
        disagreements.insert(0, "<template-flip>")
    return matches / len(param_names), disagreements


def _agreement_vs_planted(verdict: dict[str, object] | None, case: SyntheticCase) -> float | None:
    if verdict is None:
        return None
    roles = dict(verdict.get("roles", {}) or {})
    names = case.param_names()
    if not names:
        return None
    matches = sum(1 for name in names if roles.get(name) == case.planted_roles.get(name))
    return matches / len(names)


def _ic_gap(
    candidate: dict[str, object] | None,
    baseline_verdict: dict[str, object] | None,
) -> float | None:
    """AICc (fallback AIC) gap between candidate and baseline winners."""

    if candidate is None or baseline_verdict is None:
        return None

    def metric(v: dict[str, object]) -> float | None:
        for key in ("aicc", "aic"):
            value = v.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
        return None

    cand = metric(candidate)
    base = metric(baseline_verdict)
    if cand is None or base is None:
        return None
    return abs(cand - base)


@dataclass
class CaseReport:
    case_id: str
    tier: str
    status: str
    agree_pct_vs_frozen: float | None
    ic_gap_on_disagreement: float | None
    agree_pct_vs_planted_truth: float | None
    wall_s: float | None
    minuit_fits: int | None
    minuit_fevals: int | None
    #: True when a disagreement's IC gap sits within the robustness tolerance
    #: (a "soft" disagreement, per verification-plan.md); None when there is no
    #: disagreement or the gap is unknown.
    ic_gap_within_tolerance: bool | None = None
    disagreements: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "tier": self.tier,
            "status": self.status,
            "agree_pct_vs_frozen": self.agree_pct_vs_frozen,
            "ic_gap_on_disagreement": self.ic_gap_on_disagreement,
            "ic_gap_within_tolerance": self.ic_gap_within_tolerance,
            "agree_pct_vs_planted_truth": self.agree_pct_vs_planted_truth,
            "wall_s": self.wall_s,
            "minuit_fits": self.minuit_fits,
            "minuit_fevals": self.minuit_fevals,
            "disagreements": self.disagreements,
            "error": self.error,
        }


def run_harness(
    cases: Sequence[SyntheticCase],
    tier: str,
    *,
    baseline: dict[str, object] | None,
    per_case_timeout_s: float,
    overall_wall_s: float,
    ic_gap_tolerance: float = DEFAULT_IC_GAP_TOLERANCE,
) -> dict[str, object]:
    """Run the candidate *tier* over *cases*, diffing against *baseline*.

    A hard overall wall guard stops launching further cases once the budget is
    exhausted; already-launched cases still honour their per-case timeout.

    On a verdict disagreement the report flags whether the IC gap sits within
    ``ic_gap_tolerance`` — the "inside the robustness delta" signal that later
    PRs (greedy/Q, Balanced) are allowed to trip without failing acceptance.
    """

    baseline_cases = {}
    if baseline is not None:
        baseline_cases = baseline.get("cases", {}) or {}

    reports: list[CaseReport] = []
    start = time.perf_counter()
    for case in cases:
        elapsed = time.perf_counter() - start
        remaining = overall_wall_s - elapsed
        if remaining <= 0:
            reports.append(
                CaseReport(
                    case_id=case.case_id,
                    tier=tier,
                    status="TIMEOUT",
                    agree_pct_vs_frozen=None,
                    ic_gap_on_disagreement=None,
                    agree_pct_vs_planted_truth=None,
                    wall_s=None,
                    minuit_fits=None,
                    minuit_fevals=None,
                    error="overall wall budget exhausted before launch",
                )
            )
            continue

        case_timeout = min(per_case_timeout_s, remaining)
        result = run_case_isolated(case, tier, timeout_s=case_timeout)
        status = str(result.get("status", "ERROR"))
        verdict = result.get("verdict") if status == "OK" else None
        base_entry = baseline_cases.get(case.case_id, {}) if baseline_cases else {}
        base_verdict = base_entry.get("verdict") if isinstance(base_entry, dict) else None

        agree_frozen: float | None = None
        disagreements: list[str] = []
        ic_gap: float | None = None
        gap_within_tolerance: bool | None = None
        if status == "OK" and base_verdict is not None:
            agree_frozen, disagreements = compare_verdicts(verdict, base_verdict)
            if disagreements:
                ic_gap = _ic_gap(verdict, base_verdict)
                if ic_gap is not None:
                    gap_within_tolerance = ic_gap <= ic_gap_tolerance

        reports.append(
            CaseReport(
                case_id=case.case_id,
                tier=tier,
                status=status,
                agree_pct_vs_frozen=agree_frozen,
                ic_gap_on_disagreement=ic_gap,
                ic_gap_within_tolerance=gap_within_tolerance,
                agree_pct_vs_planted_truth=(
                    _agreement_vs_planted(verdict, case) if status == "OK" else None
                ),
                wall_s=_as_float(result.get("wall_s")),
                minuit_fits=_as_int(result.get("minuit_fits")),
                minuit_fevals=_as_int(result.get("minuit_fevals")),
                disagreements=disagreements,
                error=str(result.get("error")) if result.get("error") else None,
            )
        )

    return _roll_up(reports, tier=tier, total_wall_s=time.perf_counter() - start)


def _roll_up(reports: Sequence[CaseReport], *, tier: str, total_wall_s: float) -> dict[str, object]:
    frozen = [r.agree_pct_vs_frozen for r in reports if r.agree_pct_vs_frozen is not None]
    planted = [
        r.agree_pct_vs_planted_truth for r in reports if r.agree_pct_vs_planted_truth is not None
    ]
    total_fits = sum(r.minuit_fits or 0 for r in reports)
    total_fevals = sum(r.minuit_fevals or 0 for r in reports)
    return {
        "tier": tier,
        "cases": [r.to_dict() for r in reports],
        "rollup": {
            "n_cases": len(reports),
            "n_ok": sum(1 for r in reports if r.status == "OK"),
            "n_timeout": sum(1 for r in reports if r.status == "TIMEOUT"),
            "n_error": sum(1 for r in reports if r.status == "ERROR"),
            "mean_agree_vs_frozen": (sum(frozen) / len(frozen)) if frozen else None,
            "min_agree_vs_frozen": min(frozen) if frozen else None,
            "mean_agree_vs_planted": (sum(planted) / len(planted)) if planted else None,
            "total_wall_s": total_wall_s,
            "total_minuit_fits": total_fits,
            "total_minuit_fevals": total_fevals,
        },
    }


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


# --------------------------------------------------------------------------- #
# Real corpus cases (env-gated) — DEFERRED
# --------------------------------------------------------------------------- #
#
# The env gate is defined so the contract is stable, but the corpus cases are
# NOT yet wired: unlike the synthetics they have no planted ground truth and
# need their own frozen-baseline mode generated from the local WiMDA Muon School
# data checkout (see docs/testing/ and the ``project_testing_corpus`` memory).
# That second baseline mode is deferred to a later global-fit-wizard PR. Until
# then the harness runs synthetics-only regardless of the env var; setting it
# emits a one-line notice rather than silently implying support.
#
# TODO(global-fit-wizard PRs 2-5): add ~1-2 real G=3 biexp-class corpus cases
# behind CORPUS_ENV_VAR, with a corpus-baseline freeze mode.


def corpus_enabled() -> bool:
    """Whether the (deferred) real-corpus cases were opted in via the env var."""

    return bool(os.environ.get(CORPUS_ENV_VAR))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _print_report(report: dict[str, object]) -> None:
    rollup = report.get("rollup", {})
    print(f"\nGlobal-fit wizard harness — tier={report.get('tier')}")
    print("-" * 72)
    header = (
        f"{'case_id':<28} {'status':<8} {'frozen%':>8} {'planted%':>9} "
        f"{'wall_s':>8} {'fits':>6} {'fevals':>8}"
    )
    print(header)
    for case in report.get("cases", []):
        frozen = case.get("agree_pct_vs_frozen")
        planted = case.get("agree_pct_vs_planted_truth")
        wall = case.get("wall_s")
        print(
            f"{case['case_id']:<28} {case['status']:<8} "
            f"{_pct(frozen):>8} {_pct(planted):>9} "
            f"{_fmt(wall):>8} {_fmt_int(case.get('minuit_fits')):>6} "
            f"{_fmt_int(case.get('minuit_fevals')):>8}"
        )
        if case.get("disagreements"):
            gap = case.get("ic_gap_on_disagreement")
            within = case.get("ic_gap_within_tolerance")
            tag = ""
            if within is True:
                tag = ", within robustness delta"
            elif within is False:
                tag = ", OUTSIDE robustness delta"
            print(
                f"    disagreements: {', '.join(case['disagreements'])}  (IC gap={_fmt(gap)}{tag})"
            )
        if case.get("error"):
            print(f"    error: {case['error']}")
    print("-" * 72)
    print(
        f"rollup: {rollup.get('n_ok')}/{rollup.get('n_cases')} OK, "
        f"{rollup.get('n_timeout')} timeout, {rollup.get('n_error')} error | "
        f"mean frozen%={_pct(rollup.get('mean_agree_vs_frozen'))} "
        f"min={_pct(rollup.get('min_agree_vs_frozen'))} | "
        f"mean planted%={_pct(rollup.get('mean_agree_vs_planted'))} | "
        f"wall={_fmt(rollup.get('total_wall_s'))}s "
        f"fits={rollup.get('total_minuit_fits')} "
        f"fevals={rollup.get('total_minuit_fevals')}"
    )


def _pct(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.0f}%"
    return "-"


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "-"


def _fmt_int(value: object) -> str:
    if isinstance(value, (int, float)):
        return str(int(value))
    return "-"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Golden-verdict regression harness for the global-fit wizard.",
    )
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default="exhaustive",
        help="Effort tier to run (all currently alias exhaustive; PR 5 adds real tiers).",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Diff the candidate run against the frozen baseline.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Regenerate the frozen Exhaustive baseline (runs full Exhaustive once).",
    )
    parser.add_argument(
        "--per-case-timeout",
        type=float,
        default=180.0,
        help="Per-case wall timeout in seconds (breach => TIMEOUT, harness continues).",
    )
    parser.add_argument(
        "--overall-wall",
        type=float,
        default=540.0,
        help="Hard overall wall guard in seconds (< 10 min by default).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable report as JSON on stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = list(synthetic_cases())

    if corpus_enabled():
        print(
            f"note: {CORPUS_ENV_VAR} is set, but real-corpus cases are deferred; "
            "running synthetics only.",
            file=sys.stderr,
        )

    if args.freeze:
        generation_date = time.strftime("%Y-%m-%d", time.gmtime())
        print(
            f"Freezing Exhaustive baseline over {len(cases)} cases "
            f"(this is the once-offline cost)..."
        )
        payload = freeze_baseline(
            cases,
            timeout_s=args.per_case_timeout,
            generation_date=generation_date,
        )
        write_baseline(payload)
        case_entries = [entry for entry in payload["cases"].values() if isinstance(entry, dict)]
        non_ok = [entry for entry in case_entries if entry.get("status") != "OK"]
        n_ok = len(case_entries) - len(non_ok)
        print(
            f"Wrote baseline to {_BASELINE_PATH} "
            f"({n_ok}/{len(cases)} cases OK, git {payload['git_sha'][:8]})."
        )
        if non_ok:
            # A non-OK case means the baseline is incomplete: signal failure so
            # an incomplete baseline is not committed by accident (it is still
            # written for inspection).
            statuses = ", ".join(
                f"{entry.get('provenance', {}).get('case_id', '?')}={entry.get('status')}"
                for entry in non_ok
            )
            print(
                f"error: {len(non_ok)} case(s) did not complete cleanly ({statuses}); "
                "the frozen baseline is INCOMPLETE — do not commit it.",
                file=sys.stderr,
            )
            return 1
        return 0

    baseline: dict[str, object] | None = None
    if args.compare_baseline:
        if not _BASELINE_PATH.exists():
            print(
                f"error: no frozen baseline at {_BASELINE_PATH}; run --freeze first.",
                file=sys.stderr,
            )
            return 2
        baseline = load_baseline()
        version = baseline.get("schema_version")
        if version != BASELINE_SCHEMA_VERSION:
            print(
                f"error: frozen baseline schema_version={version!r} does not match "
                f"the harness (expected {BASELINE_SCHEMA_VERSION}); re-freeze with "
                "--freeze before comparing.",
                file=sys.stderr,
            )
            return 2

    report = run_harness(
        cases,
        args.tier,
        baseline=baseline,
        per_case_timeout_s=args.per_case_timeout,
        overall_wall_s=args.overall_wall,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
