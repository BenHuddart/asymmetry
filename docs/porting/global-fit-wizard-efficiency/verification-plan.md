# Verification plan

Every PR after PR 1 is measured by the golden-verdict harness (`test-data.md`).
This file lists the per-PR acceptance bars and the cross-cutting things to
validate *before trusting* each technique.

## Per-PR acceptance bars

| PR | Bar |
|---|---|
| **1 harness** | Reproduces frozen Exhaustive at 100 % agreement on all cases; full run < 10 min; hard timeout guard proven (a deliberately slow case reports `TIMEOUT`, harness continues). |
| **2 free wins** | Verdict agreement vs frozen baseline = **100 %** (A/B are exact; C/D exact-to-argmax). Report ≥7× wall reduction on the P=5 case; fit-count drop consistent with escalation. |
| **3 engine G** | Profiled vs joint: parameter values within fit tolerance, IC within <0.5 on every case; VarPro values+errors match current; G-sweep shows ~linear scaling; verdicts unchanged (100 %). Default strategy stays `joint` until this passes. |
| **4 search+surrogate** | Balanced-path agreement ≥95 %; on every disagreement the IC gap is within the robustness delta; surrogate-rank-of-winner ≤ K on all cases; greedy recovers planted truth on non-interaction synthetics. |
| **5 slider** | Each tier runs end-to-end through the GUI; Balanced ≥95 % verdict agreement with Exhaustive; Exhaustive reproduces the frozen baseline; decimation-tier ICs are internally consistent (no mixed-binning leaderboard). |

## Cross-cutting: validate before trusting

1. **Cache completeness under sparse searches (E/F/G).** The robustness/gate
   layer compares against the winner's single-flip neighbourhood. Greedy/Q/
   surrogate leave a sparse cache, so the flip-neighbourhood must be explicitly
   fitted. This is the most likely silent breakage in PR 4 — assert the cache
   contains all P flips of the winner before the verdict step runs.
2. **Step-hint propagation survives Hesse removal (C).** If the curvature/step
   hints depend on Hesse output, substitute Minuit's running EDM covariance.
   Verify fit counts don't balloon (lost warm-start quality) after C lands.
3. **Surrogate rank-of-winner ≤ K on every harness case (G).** Report it
   explicitly; if it exceeds K anywhere, grow K or widen the verify set.
4. **Escalation fire-rate and verdict-neutrality (D).** Count how often
   escalation fires; confirm no anomaly restart ever changes a verdict on the
   exact-tier cases (expected: none). A verdict flip here means the certificate
   ε is too loose.
5. **AICc/BIC n-consistency under decimation (K).** Never mix decimated and
   full-resolution ICs on one leaderboard; gate decimation on the oscillatory
   fingerprint's Nyquist so binning never aliases a candidate frequency.
6. **Pre-tests never fix an invalid param (E).** Q-test must skip (→ ambiguous)
   any parameter whose single-fit error was flagged invalid or at-limit.
7. **Incumbent-first ordering for the layer bound (A).** The bound uses the
   incumbent, so interleave the wavefront to fit the Q-predicted best assignment
   first — otherwise the bound is toothless in the early layers.
8. **Profiled vs joint / VarPro vs current (L/M).** Must match
   values/errors/IC on the harness before the profiled path becomes any tier's
   default. Keep `joint` as the default strategy until signed off.

## Regression discipline

- Re-freeze the baseline only on a deliberate physics change, never to "make the
  harness pass" — a drop in agreement is the signal the harness exists to catch.
- Fit-count is a first-class assertion: an unexplained change in
  `minuit_fits`/`minuit_fevals` is a regression even if verdicts still agree.

## PR 5 rework outcome (slider collapse + cancel fix)

The PR 5 four-position effort slider (Low/Balanced/Thorough/Exhaustive) was
reworked after a design review. **Keep the plumbing, drop the approximation.**

- **Heuristic Low/Balanced were empirically dominated by bounded-exhaustive.**
  PR 2's exact bounds already made Exhaustive near-minimal (~1000 fits on the
  harness corpus) and 12-way parallel, while the heuristic driver is serial by
  construction. On real workloads the heuristic tiers were *slower* (up to ~15×)
  with no fit-count headroom left to reclaim. They were therefore removed from
  the user-facing slider, which now surfaces a single honest "Optimize —
  exhaustive" mode.
- **Every `EffortTier` now resolves to the exact bounded-wavefront engine**
  (`_EFFORT_TIER_SEARCH_ENGINE` maps all four values to `SEARCH_ENGINE_EXHAUSTIVE`),
  and the extra tier knobs (I portfolio cap, J identifiability demotion, K
  screening decimation) are gated on the *heuristic engine string*, never on
  `effort_tier` — so they are inert for every user-facing tier by construction.
  Evidence: `global_wizard_harness.py --tier {low,balanced,thorough,exhaustive}
  --compare-baseline` reports **100% agreement with the frozen baseline and
  identical fit counts (1262 fits, 205069 fevals) for all four tiers**; the only
  variation is process-pool scheduling noise in the wall time.
- **Heuristics retained behind the `search_engine` seam.** `_run_heuristic_search`,
  the Q/greedy/surrogate/decimation code, and `global_search/homogeneity.py` are
  kept reachable only via the low-level `search_engine="low"/"balanced"` string
  (the PR 4 seam) for future large-P use and regression coverage. The core
  test-suite still exercises them through that string, decoupled from `EffortTier`.
- **Cancel fix.** A cooperative `cancel_callback` was threaded through the core
  builders (`build_global_fit_wizard_recommendation`, the staged builder,
  `build_global_fit_wizard_screening_recommendation`, the phase-1 single-fit
  helper) and into `_run_exhaustive_wavefront_search`, which now checks it between
  templates, between Hamming layers, and before dispatching each layer's
  assignment fits. Under the process pool in-flight futures cannot be killed, so
  cancel stops scheduling further layers/templates and the `finally` block shuts
  the pool down. In-process serial calls also forward the callback into
  `engine.global_fit` for mid-fit abort. A focused test proves a truthy callback
  raises `FitCancelledError` promptly and runs zero fits.
