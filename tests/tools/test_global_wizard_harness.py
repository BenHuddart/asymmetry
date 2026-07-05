"""Tests for the global-fit wizard golden-verdict regression harness.

The always-on tests here are FAST and deterministic: they never run the real
549 s Exhaustive wizard. They exercise the freeze/diff plumbing, the report
schema, the planted-truth math, and the TIMEOUT-continues guard by injecting
fake tier callables, and they diff the *committed* frozen baseline for
verdict-level self-consistency (verdicts + bounded IC gap only — never exact
fit counts, which differ across BLAS/platform).

The real Exhaustive-vs-baseline 100 % self-check is marked ``slow`` (full tier
only) so it stays out of the always-on shard.
"""

from __future__ import annotations

import json
import time

import pytest

import tools.global_wizard_harness as harness

# Unmarked tests are auto-marked ``unit`` by tests/conftest.py, so the fast
# tests below land in the fast/standard tiers; the real-wizard self-check at the
# bottom carries an explicit ``slow`` marker to stay out of both.


# --------------------------------------------------------------------------- #
# Case set / generator invariants
# --------------------------------------------------------------------------- #


def test_case_set_matrix_stays_inside_budget() -> None:
    """Cases cover the required matrix and never the P=7 monster."""

    cases = harness.synthetic_cases()
    case_ids = {case.case_id for case in cases}
    # The five archetypes the spec requires.
    for expected in (
        "pure_global",
        "pure_local",
        "mixed",
        "correlated_pair",
        "near_degenerate",
    ):
        assert any(expected in cid for cid in case_ids), expected

    for case in cases:
        assert case.n_groups in (3, 4), case.case_id
        assert case.n_params in (3, 4, 5), (case.case_id, case.n_params)
        assert case.n_params != 7


def test_planted_roles_cover_every_parameter() -> None:
    for case in harness.synthetic_cases():
        assert set(case.planted_roles) == set(case.param_names())
        # Globals have a single planted value; locals scan across all groups.
        for name, scan in case.local_scans.items():
            assert case.planted_roles[name] == "local"
            assert len(scan) == case.n_groups
        for name in case.global_values:
            assert case.planted_roles[name] == "global"


def test_generated_datasets_are_deterministic() -> None:
    """The generator is seeded, so datasets are byte-identical run-to-run."""

    case = next(c for c in harness.synthetic_cases() if c.case_id.startswith("pure_local"))
    first = harness._build_case_datasets(case)
    second = harness._build_case_datasets(case)
    assert len(first) == case.n_groups
    for ds_a, ds_b in zip(first, second, strict=True):
        assert (ds_a.asymmetry == ds_b.asymmetry).all()
        assert ds_a.run_number == ds_b.run_number


# --------------------------------------------------------------------------- #
# Frozen baseline: shape + provenance + self-consistency
# --------------------------------------------------------------------------- #


def test_frozen_baseline_exists_and_is_well_formed() -> None:
    payload = harness.load_baseline()
    assert payload["schema_version"] == harness.BASELINE_SCHEMA_VERSION
    assert isinstance(payload["git_sha"], str) and payload["git_sha"]
    assert isinstance(payload["generation_date"], str) and payload["generation_date"]
    assert payload["tier"] == "exhaustive"

    baseline_cases = payload["cases"]
    frozen_ids = set(baseline_cases)
    case_ids = {case.case_id for case in harness.synthetic_cases()}
    assert frozen_ids == case_ids, "baseline drift vs the in-code case set"

    for case_id, entry in baseline_cases.items():
        assert entry["status"] == "OK", case_id
        verdict = entry["verdict"]
        assert verdict is not None
        assert isinstance(verdict["roles"], dict) and verdict["roles"]
        assert verdict["template_key"]


def test_frozen_baseline_recovers_planted_truth_where_expected() -> None:
    """The frozen Exhaustive verdicts should match planted roles on the
    non-adversarial cases (pure-global, pure-local, mixed)."""

    payload = harness.load_baseline()
    by_id = {c.case_id: c for c in harness.synthetic_cases()}
    # near_degenerate is deliberately excluded: its correct verdict is a
    # demotion, not its generative (biexp) planted roles.
    recovery_prefixes = ("pure_global", "pure_local", "mixed", "correlated_pair")
    for case_id, entry in payload["cases"].items():
        case = by_id[case_id]
        if not any(case_id.startswith(prefix) for prefix in recovery_prefixes):
            continue
        agree = harness._agreement_vs_planted(entry["verdict"], case)
        assert agree == pytest.approx(1.0), (case_id, entry["verdict"]["roles"])


# --------------------------------------------------------------------------- #
# Diff / comparison math (no wizard call)
# --------------------------------------------------------------------------- #


def test_compare_verdicts_full_agreement() -> None:
    verdict = {
        "template_key": "exp_constant",
        "roles": {"A_1": "global", "Lambda": "local", "A_bg": "global"},
    }
    agree, disagreements = harness.compare_verdicts(verdict, dict(verdict))
    assert agree == pytest.approx(1.0)
    assert disagreements == []


def test_compare_verdicts_role_disagreement() -> None:
    base = {
        "template_key": "exp_constant",
        "roles": {"A_1": "global", "Lambda": "local", "A_bg": "global"},
    }
    cand = {
        "template_key": "exp_constant",
        "roles": {"A_1": "global", "Lambda": "global", "A_bg": "global"},
    }
    agree, disagreements = harness.compare_verdicts(cand, base)
    assert agree == pytest.approx(2 / 3)
    assert disagreements == ["Lambda"]


def test_compare_verdicts_template_flip_counts_all_params() -> None:
    base = {
        "template_key": "exp_constant",
        "roles": {"A_1": "global", "Lambda": "local"},
    }
    cand = {
        "template_key": "gaussian_constant",
        "roles": {"A_1": "global", "Lambda": "local"},
    }
    agree, disagreements = harness.compare_verdicts(cand, base)
    assert agree == pytest.approx(0.0)
    assert "<template-flip>" in disagreements


def test_ic_gap_computes_absolute_aicc_difference() -> None:
    cand = {"aicc": 12.0, "aic": 10.0}
    base = {"aicc": 10.5, "aic": 9.0}
    assert harness._ic_gap(cand, base) == pytest.approx(1.5)


def test_disagreement_flags_ic_gap_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role disagreement whose IC gap is inside the tolerance is flagged as a
    soft ("within robustness delta") disagreement, not a hard failure."""

    case = harness.synthetic_cases()[0]

    def _fake_isolated(c, tier, *, timeout_s):  # noqa: ANN001, ARG001
        # Candidate flips one role but sits only 1.0 IC unit off the baseline.
        roles = dict(c.planted_roles)
        first = next(iter(roles))
        roles[first] = "local" if roles[first] == "global" else "global"
        return {
            "status": "OK",
            "verdict": {
                "template_key": c.template_keys[0],
                "roles": roles,
                "aicc": 101.0,
            },
            "wall_s": 0.0,
            "minuit_fits": 1,
            "minuit_fevals": 1,
        }

    monkeypatch.setattr(harness, "run_case_isolated", _fake_isolated)
    baseline = {
        "cases": {
            case.case_id: {
                "verdict": {
                    "template_key": case.template_keys[0],
                    "roles": dict(case.planted_roles),
                    "aicc": 100.0,
                }
            }
        }
    }
    report = harness.run_harness(
        [case],
        "exhaustive",
        baseline=baseline,
        per_case_timeout_s=30.0,
        overall_wall_s=60.0,
        ic_gap_tolerance=2.0,
    )
    entry = report["cases"][0]
    assert entry["agree_pct_vs_frozen"] < 1.0
    assert entry["ic_gap_on_disagreement"] == pytest.approx(1.0)
    assert entry["ic_gap_within_tolerance"] is True


# --------------------------------------------------------------------------- #
# TIMEOUT / ERROR aggregation (fast: run_case_isolated stubbed in-process)
# --------------------------------------------------------------------------- #
#
# These stay in-process by replacing ``run_case_isolated`` itself, so they never
# spawn a real wizard. The spawn boundary re-imports the module in the child, so
# a monkeypatched TIER_CONFIGS entry or a local closure would NOT cross into the
# child — hence the real kill path is proven separately in the ``slow`` test at
# the bottom, using a module-level sleeper config the child can re-import.


def test_timeout_status_aggregates_and_harness_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A case reporting TIMEOUT does not stop later cases; the roll-up counts
    every timeout. Aggregation is exercised in-process (no spawn)."""

    def _fake_isolated(case, tier, *, timeout_s):  # noqa: ANN001, ARG001
        return {"status": "TIMEOUT", "wall_s": timeout_s}

    monkeypatch.setattr(harness, "run_case_isolated", _fake_isolated)
    cases = harness.synthetic_cases()[:2]
    report = harness.run_harness(
        cases,
        "exhaustive",
        baseline=None,
        per_case_timeout_s=1.0,
        overall_wall_s=30.0,
    )
    assert [c["status"] for c in report["cases"]] == ["TIMEOUT", "TIMEOUT"]
    assert report["rollup"]["n_timeout"] == 2
    assert report["rollup"]["n_cases"] == 2


def test_error_status_is_captured_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_isolated(case, tier, *, timeout_s):  # noqa: ANN001, ARG001
        return {"status": "ERROR", "error": "synthetic explosion"}

    monkeypatch.setattr(harness, "run_case_isolated", _fake_isolated)
    cases = harness.synthetic_cases()[:1]
    report = harness.run_harness(
        cases,
        "exhaustive",
        baseline=None,
        per_case_timeout_s=30.0,
        overall_wall_s=60.0,
    )
    case = report["cases"][0]
    assert case["status"] == "ERROR"
    assert "synthetic explosion" in (case["error"] or "")
    assert report["rollup"]["n_error"] == 1


def test_fake_tier_run_produces_full_report_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fast stub exercises the whole diff -> report path and the report
    carries every field the spec's schema requires."""

    baseline_by_id = {c.case_id: c for c in harness.synthetic_cases()}

    # Stub run_case_isolated so the test stays in-process and fast while still
    # driving the diff + roll-up against a matching synthetic baseline.
    def _fake_isolated(case, tier, *, timeout_s):  # noqa: ANN001, ARG001
        roles = dict(case.planted_roles)
        template_key = case.template_keys[0]
        return {
            "status": "OK",
            "verdict": {
                "template_key": template_key,
                "recommended_key": f"{template_key}|planted",
                "roles": roles,
                "aic": 100.0,
                "aicc": 101.0,
                "bic": 105.0,
                "n_params": case.n_params,
            },
            "wall_s": 0.01,
            "minuit_fits": 7,
            "minuit_fevals": 123,
        }

    monkeypatch.setattr(harness, "run_case_isolated", _fake_isolated)

    # Build a synthetic baseline that matches the injected verdicts exactly.
    baseline = {
        "schema_version": harness.BASELINE_SCHEMA_VERSION,
        "git_sha": "deadbeef",
        "generation_date": "2026-01-01",
        "tier": "exhaustive",
        "cases": {
            case_id: {
                "provenance": harness._case_provenance(case),
                "verdict": {
                    "template_key": case.template_keys[0],
                    "recommended_key": f"{case.template_keys[0]}|planted",
                    "roles": dict(case.planted_roles),
                    "aic": 100.0,
                    "aicc": 101.0,
                    "bic": 105.0,
                    "n_params": case.n_params,
                },
                "minuit_fits": 7,
                "minuit_fevals": 123,
                "status": "OK",
                "wall_s": 0.01,
            }
            for case_id, case in baseline_by_id.items()
        },
    }

    report = harness.run_harness(
        harness.synthetic_cases(),
        "exhaustive",
        baseline=baseline,
        per_case_timeout_s=30.0,
        overall_wall_s=60.0,
    )

    required = {
        "case_id",
        "tier",
        "status",
        "agree_pct_vs_frozen",
        "ic_gap_on_disagreement",
        "ic_gap_within_tolerance",
        "agree_pct_vs_planted_truth",
        "wall_s",
        "minuit_fits",
        "minuit_fevals",
    }
    for case in report["cases"]:
        assert required <= set(case)
        assert case["status"] == "OK"
        assert case["agree_pct_vs_frozen"] == pytest.approx(1.0)
    assert report["rollup"]["mean_agree_vs_frozen"] == pytest.approx(1.0)
    assert report["rollup"]["min_agree_vs_frozen"] == pytest.approx(1.0)
    # The whole report round-trips through JSON (it is what the CLI emits).
    json.loads(json.dumps(report))


def test_overall_wall_guard_stops_launching_further_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the overall wall budget is spent, remaining cases report TIMEOUT
    without being launched."""

    calls: list[str] = []

    def _fake_isolated(case, tier, *, timeout_s):  # noqa: ANN001, ARG001
        calls.append(case.case_id)
        return {
            "status": "OK",
            "verdict": {"template_key": "x", "roles": {}, "aicc": 1.0},
            "wall_s": 0.0,
            "minuit_fits": 0,
            "minuit_fevals": 0,
        }

    monkeypatch.setattr(harness, "run_case_isolated", _fake_isolated)
    # A zero overall wall means no case may launch.
    report = harness.run_harness(
        harness.synthetic_cases(),
        "exhaustive",
        baseline=None,
        per_case_timeout_s=30.0,
        overall_wall_s=0.0,
    )
    assert calls == []
    assert all(c["status"] == "TIMEOUT" for c in report["cases"])


# --------------------------------------------------------------------------- #
# Slow: the real Exhaustive-vs-baseline self-check (full tier / opt-in only)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_subprocess_kill_reports_timeout_and_continues() -> None:
    """Prove the *real* kill path: a module-level sleeper config overruns its
    per-case timeout, ``run_case_isolated`` kills the child (and its process
    tree), the case reports TIMEOUT, and the next case still runs.

    This is the spec's under-10-min guard acceptance bar. It runs the real
    subprocess machinery, so it is ``slow`` (full tier only)."""

    cases = harness.synthetic_cases()[:2]
    start = time.perf_counter()
    report = harness.run_harness(
        cases,
        "_timeout_probe",  # module-level sleeper; visible in the spawned child
        baseline=None,
        per_case_timeout_s=2.0,
        overall_wall_s=30.0,
    )
    elapsed = time.perf_counter() - start
    assert [c["status"] for c in report["cases"]] == ["TIMEOUT", "TIMEOUT"]
    assert report["rollup"]["n_timeout"] == 2
    # Both cases were killed near their timeout rather than running to 3600 s.
    assert elapsed < 30.0


@pytest.mark.slow
def test_exhaustive_reproduces_frozen_baseline_at_full_agreement() -> None:
    """PR 1 acceptance: current Exhaustive reproduces the frozen baseline at
    100 % verdict agreement. Runs the real wizard, so it is ``slow`` (full tier
    only) and kept out of the always-on shard."""

    baseline = harness.load_baseline()
    report = harness.run_harness(
        harness.synthetic_cases(),
        "exhaustive",
        baseline=baseline,
        per_case_timeout_s=180.0,
        overall_wall_s=540.0,
    )
    for case in report["cases"]:
        assert case["status"] == "OK", (case["case_id"], case.get("error"))
        assert case["agree_pct_vs_frozen"] == pytest.approx(1.0), (
            case["case_id"],
            case["disagreements"],
        )
    assert report["rollup"]["min_agree_vs_frozen"] == pytest.approx(1.0)
