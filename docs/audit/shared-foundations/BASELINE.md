# Shared-Foundations Audit — Baseline

Captured 2026-07-03 on branch `audit/shared-foundations` @ `cd014da`
(tree identical to `main` @ `3d2359a`), in this worktree's `.venv`
(Python 3.12.13, numpy 2.2.6), `QT_QPA_PLATFORM=offscreen`.

This is the **arithmetic baseline** for the Phase 4 test-reorg parity check.
Phase 4 must collect the same numbers **plus** whatever tests this audit adds
(tracked in the "Tests added by the audit" section below).

## Collected test counts (`pytest --collect-only`)

| Tier | Marker expression | Collected | Deselected | Total items |
|---|---|---|---|---|
| **standard** (validate) | `not slow and not integration` | **4059** | 94 | 4153 |
| **full** | *(none)* | **4153** | 0 | 4153 |
| **fast** | `unit and not slow and not gui and not io and not integration` | **2362** | 1791 | 4153 |

## `validate` result (standard tier)

```
4046 passed, 12 skipped, 1 xfailed, 15 warnings in 234.81s (0:03:54)
structural: ok
lint: ok
```

4046 passed + 12 skipped + 1 xfailed = **4059** collected — consistent with the
standard-tier collection count above.

## `gui-smoke`

`python -m asymmetry.gui.app --smoke-test` → exit 0 (pass).

## Tests added by the audit

Running tally so Phase 4's parity check can subtract audit-added tests from the
new collection count. Update this as each phase adds characterization/regression
tests.

| Phase | Test file(s) | Tests added (net) |
|---|---|---|
| 0 | `test_axis_limit_field_characterization.py` (new), `test_fit_range_commit_roundtrip.py` (new), `test_wizard_result_caching.py` (new), `test_fit_parameters_panel.py` (+2) | **+35** |
| 1a | (field + AxisLimitControls — test edits were import/type repoints, net 0) | +0 |
| 1b | `test_mpl_canvas.py` (new — canvas factory coverage) | **+8** |
| 1c | `test_export_utils.py` (new — compile_gle wrapper) | **+3** |
| 1d | `test_fit_run_controls.py` (new — FitRunControls) | **+9** |
| 1 (Review A fix) | `test_axis_limit_controls.py` (new — AxisLimitControls) | **+6** |
| 2 (mechanical split) | `test_styles_tokens.py` no-hex guard parametrized over the 5 new `fit/` submodules | **+5** |
| 3A (wizard base) | `test_wizard_base.py` (new — WizardWindowBase) | **+4** |
| 3 (Review B2 fix) | `test_wizard_base.py` (+1 stale-error soft-lock regression) | **+1** |
| 4 (test reorg) | (pure file moves — 0 added) | +0 |
| 5 (harness rules) | `tests/tools/test_harness.py` (+10 — 3 rules × pass-clean + fire-on-violation) | **+10** |
| 7 (final-review fix) | `test_fit_wizard_window.py` (+1 — failed-refresh clears recommendation regression) | **+1** |

**Running total of audit-added tests: 82.** Full-tier collection = **4235** (= 4153 + 82). Standard-tier passed = 4127 (the +1 is integration-marked, excluded from standard).

**Phase 4 parity rule:** `new_full_collected == 4153 + Σ(tests added by audit)`.
Any shortfall is a silent dropped-collection regression and must be root-caused,
not accepted.

**Phase 4 confirmation:** `pytest --collect-only -q` post-reorganization reports
**4224 tests collected** — exact parity (0 dropped, 0 added by the move itself).
Standard-tier `validate` after the move: **4117 passed, 12 skipped, 1 xfailed**
in 89.23s — identical to the pre-move Phase-3 standard-tier result. `structural`
and `lint` (which exercises the 84 rewritten `pyproject.toml`
`per-file-ignores` keys) are both clean.
