# Test data & the golden-verdict regression harness (PR 1 spec)

The harness is the acceptance gate for PRs 2–5: it freezes Exhaustive verdicts
and, for any candidate wizard configuration, reports **verdict-agreement %, IC
gap on disagreement, wall, and fit count**. This file is the buildable spec for
PR 1.

## Why a bespoke harness (not just pytest)

The current Exhaustive path is **549 s for one real G=3 case**, so you cannot
run full Exhaustive on every invocation. The harness must (a) freeze the
baseline once and cache it, (b) run only the *candidate* configuration per
invocation, and (c) stay under **~10 minutes wall** with a hard timeout guard.

## Keeping it under 10 minutes — the design

1. **Freeze the baseline once, cache it.** Run full Exhaustive on the case set a
   single time, serialise a **compact extracted-verdict snapshot** — the
   per-parameter global/local role map, the winning template key, the ICs, and
   the Minuit fit/feval counts — to `baseline/*.json`, commit them. (This is a
   compact extractor, **not** `serialize_global_fit_wizard_recommendation`: the
   full serializer embeds `fitted_curves_by_run` / `component_curves_by_run`,
   large per-run float arrays that do not diff stably across BLAS/platforms, so
   the baseline would be huge and fragile. Storing only the diffable fields
   keeps the frozen artifact small and cross-platform stable.) Every later run
   loads the frozen baseline and runs **only the candidate config**, diffing
   against the cached verdicts. Re-freeze only when the winning-model physics
   changes — guard with a git-SHA + case-hash check that *warns on drift* rather
   than silently re-running Exhaustive.

2. **Small cases only — the matrix, not realism.** Target the code paths, not
   corpus fidelity:
   - **4–5 synthetic cases with known ground truth**: G∈{3,4}, P∈{3,4,5}.
     **Never the P=7 / 364 s monster** — P=5/G=3 is ~40 s Exhaustive and
     exercises the same paths. One case each for:
     - pure-global (all params shared),
     - pure-local (all params free),
     - mixed (some of each),
     - a **correlated-pair** case (stresses E/F/G interaction handling —
       the one place greedy/Q are expected to be able to disagree),
     - a **near-degenerate template** case (stresses J identifiability demotion).
   - **1–2 real small cases** (G=3, capped to a biexp-class model) as a reality
     anchor.
   - Total frozen-baseline cost, paid **once offline**: ~5–8 min. Acceptable
     because it is off the per-run path.

3. **Per-run budget.** Each candidate config runs the same 5–7 cases at its own
   tier (Low/Balanced are seconds; Thorough tens of seconds). Wrap the whole run
   in a **hard 10-min wall guard**: each case runs under a per-case timeout; on
   breach the case reports `TIMEOUT` and the harness continues — it **never
   blocks**. The Exhaustive-tier self-check runs only the P≤5 cases so even
   Exhaustive-vs-baseline stays < 10 min.

4. **Determinism.** Pin RNG seeds for synthetics and any multi-start jitter;
   assert reproducible fit counts, so a regression in fit count is itself a
   signal (e.g. an escalation misfire in PR 2).

## Synthetic generator

Plant ground-truth roles by construction: pick globals (identical value across
all G members) and locals (values drawn along a smooth scan), add **additive
Gaussian noise in asymmetry space at a fixed per-point sigma** (matching the
uniform `error` array the datasets carry), with **pinned per-group RNG seeds**
so every draw is reproducible, and record the intended verdict per parameter.
This gives the harness a *second* correctness signal beyond
"agrees with frozen Exhaustive": "recovers planted truth". Keep the generator in
the harness module, not in `core` (study scaffolding, not shipped behaviour).

## Report schema

Per case:
```
{ case_id, tier, verdicts,
  agree_pct_vs_frozen, ic_gap_on_disagreement,
  agree_pct_vs_planted_truth,          # synthetic cases only
  wall_s, minuit_fits, minuit_fevals, status }   # status in {OK, TIMEOUT, ERROR}
```
Plus a roll-up (mean/min agreement, total wall, total fits). Flags: `--tier
{low,balanced,thorough,exhaustive}` and `--compare-baseline`.

## Baseline artifacts

- `docs/porting/global-fit-wizard-efficiency/baseline/*.json` — frozen verdicts +
  ICs + fit counts, with provenance (git SHA, case params, generation date).

## Acceptance for PR 1 itself

Running the harness on **current main Exhaustive** must reproduce the frozen
baseline at **100 % agreement** — the harness agrees with itself. This is the
sanity check that the freeze/diff plumbing is correct before any optimisation
PR leans on it.

## Real corpus note

Real small cases should draw from the WiMDA Muon School corpus (see the
project's `docs/testing/` guide and the `project_testing_corpus` memory), G=3,
capped to a biexp-class model to stay inside the time budget. Gate real-corpus
cases behind an env var (mirroring existing env-gated corpus tests) so the
harness runs synthetics-only in CI/headless without the data checkout.
