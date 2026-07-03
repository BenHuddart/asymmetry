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

**Running total of audit-added tests: 46.** Full-tier collection after Phase 1c = **4199** (= 4153 + 46). Standard-tier passed = 4092 (was 4046 at baseline).

**Phase 4 parity rule:** `new_full_collected == 4153 + Σ(tests added by audit)`.
Any shortfall is a silent dropped-collection regression and must be root-caused,
not accepted.
