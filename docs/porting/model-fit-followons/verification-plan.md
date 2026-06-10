# Verification plan: Model-fit follow-ons

Acceptance is the harness ladder green at each phase
(`python tools/harness.py validate`, ~2 min; the **worktree** `.venv/bin/python`;
`QT_QPA_PLATFORM=offscreen` for GUI tests) plus the checks below. Oracle values
and datasets: [test-data.md](test-data.md). Standing rule: physical correctness
over WiMDA equivalence, divergences documented with both behaviours.

## Phase 1 — arbitrary X column (+ effective-variance x-uncertainty)

1. **x-key encoding** round-trips (`param:<name>` preserved through
   `_normalize_x_key`, `_serialize/_deserialize_model_fits`; legacy default
   honoured) — test-data §1.1, §1.4.
2. **Scope degrade**: `component_names_for_x("param:*")` is common-only — §1.2.
3. **param-vs-param on real data**: EuO λ vs ν runs and uses ν as x — §1.3.
4. **Effective variance**: σ_x=0 byte-identical to OLS (§2.1); constant-σ_eff
   closed-form match and error inflation (§2.2); finite-difference slope sanity
   (§2.3). Toggle default OFF; `use_x_errors` round-trips (legacy → False).
5. **Plot/GLE**: x-label and horizontal x-bars render for a `param:*` x
   (offscreen GUI test); GLE export emits the x-error column.
6. **Core purity**: no Qt/matplotlib pulled into `asymmetry.core` (structural
   harness).
7. `validate` + `docs` green → milestone commit.

## Phase 2 — cross-group error modes + windows

1. **Core honesty**: `global_fit_parameter_model` honours `windows` (point
   counts, §3.1) and `error_mode`/`error_value` (σ changes, SCATTER rescales
   global **and** local σ, ν<1 indeterminate — §3.2).
2. **Degenerate equality**: two-identical-groups cross-group fit equals the
   single-series fit (§3.3) — proves the path reduces correctly.
3. **GUI un-hidden & wired**: `CrossGroupFitDialog` now shows the error-mode
   combo, value field and "+ Window" controls; the previously-asserting
   `test_cross_group_dialog_hides_unsupported_controls` is inverted; a new test
   shows the dialog drives the backend (windows change point counts; mode
   changes σ). Verdict renders (n_points populated) and is suppressed for
   none/scatter.
4. **Config persistence**: error_mode/error_value/windows round-trip through
   dialog config and the panel's cross-group-config serialiser; legacy loads
   cleanly — §3.4.
5. No double-masking: when windows are present the dialog does not also pre-slice
   by `[x_min,x_max]` (one masking site).
6. `validate` green → milestone commit.

## Phase 3 — results-table recursion

1. **Local-param rows**: new *Model fit results* series with one row per group,
   correct values/errors/coordinate/`origin` — §4.1.
2. **Globals present** per the chosen representation — §4.2.
3. **Recursion round-trip**: a second trend fit on the derived series produces a
   finite result — §4.3 (the headline acceptance for item 3).
4. **Persistence**: derived series + `_FitRow.origin` round-trip; legacy rows
   load with `origin=None` — §4.4.
5. **Overwrite, not duplicate**: re-running the same cross-group fit replaces its
   results series — §4.5.
6. `validate` + `docs` green → milestone commit; update study README/this file
   with outcomes.

## Phase 4 — STRETCH: quadrature combinator (⊕)

Only if Phases 1–3 are green with budget left. Acceptance: `⊕` parses in the
composite grammar, the builder dialog offers it, `sqrt(f²+g²)` evaluates
correctly (oracle: `PowerLaw ⊕ Constant` equals `PowerLawQuadBG`), GLE export
emits it, and round-trips through `ParameterCompositeModel.to_dict/from_dict`.
If not reached: record the design state in implementation-options.md and stop —
it stays a follow-on.

## End state

- Full `validate` green after each phase; milestone commit per phase; **no push,
  no PR** (standing instruction) until explicitly asked.
- Suite ≥ 1823 passed and rising; user-guide `parameter_trending.rst` extended
  (result-first prose, rendered math, uncertainties as 0.23(1), APS reference
  lists, a "when to use this" register per feature) and `docs` builds clean.
- Study updated with final decisions and verification outcomes; new follow-ons
  recorded. Then `/code-review` at high effort on the branch diff; fix confirmed
  findings before reporting.
