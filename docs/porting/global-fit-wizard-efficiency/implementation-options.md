# Implementation options: the five-PR phased plan

Legend: **[serial]** must land before dependents; **[parallel]** can run
concurrently in its own worktree. Model tier reflects difficulty/risk. Every PR
after PR 1 reports its `test-data.md` harness numbers in its description.

---

## PR 1 — Golden-verdict regression harness  **[serial — blocks all others]**  · Opus

- **Goal.** A fast, deterministic harness that freezes Exhaustive verdicts on a
  small corpus + known-ground-truth synthetics and, for any candidate wizard
  configuration, reports **verdict-agreement %, IC gap on disagreement, wall,
  and fit count**. This is the acceptance gate for PRs 2–5.
- **Techniques.** None (infrastructure) — but it *defines the contract* every
  later technique is measured against.
- **Files/functions.** New harness entrypoint under `tools/` (or
  `tests/tools/`), a cached `baseline/` of frozen Exhaustive verdicts, a small
  synthetic generator with known ground-truth roles. Reuses
  `serialize_/deserialize_global_fit_wizard_recommendation`
  (global_fit_wizard.py ~1843) for stable snapshots and
  `build_global_fit_wizard_recommendation` (~1379) as the entry.
- **Full harness spec is in `test-data.md`** (including the under-10-min design).
- **Acceptance for PR 1 itself:** running it on *current main Exhaustive*
  reproduces the frozen baseline at 100 % agreement (the harness agrees with
  itself).

---

## PR 2 — Free wins: exact bounds + fit-cost hygiene  **[parallel-A, after PR1]**  · Opus

- **Goal.** Land the accuracy-preserving speedups that apply at *all* tiers, so
  the baseline every tier is calibrated against already reflects them. Target
  **7–12× on the 549 s case with identical verdicts.**
- **Techniques.** A (layer bound), B (cross-template bound), C (strategy-0 /
  no-Hesse screening), D (escalation-on-anomaly).
- **Files/functions.**
  - `_run_exhaustive_wavefront_search` (~5497): insert the all-local anchor fit
    + per-layer bound check that halts layer enumeration (A). **Primary
    hotspot** — see §Conflicts.
  - `_build_global_fit_wizard_recommendation_staged` (~1418) /
    `_shortlist_template_keys` (~2339): cross-template incumbent bound (B),
    ordering templates by screen score and skipping dominated ones.
  - `_evaluate_attempt_variants` (~3493), `_assignment_attempt_variants` /
    `_trim_assignment_attempt_variants` (~4444–4511),
    `_staged_multi_local_assignment` (~3035): gate the multi-start/staged
    battery behind the monotonicity certificate; escalate on anomaly (D).
  - `engine.py` / `asymmetry_global.py::fit_global` (~174): plumb a
    `screening` fit-mode flag → strategy 0, loose EDM, `hesse=False` (C). Add a
    separate `final=True` path for winner + flip-neighbourhood.
  - IC helpers (`metric_value`, `parameter_count`, `additive_terms` ~154–185):
    expose `penalty(k)` cleanly for the bound; do not change their semantics.
- **Validate (call out in PR):** step-hint propagation survives Hesse removal
  (substitute running EDM covariance if it depends on it); Δ_margin in A ≥ the
  robustness layer's max delta; escalation never changes a verdict (harness:
  100 % agreement expected — these are exact/near-exact). See `verification-plan.md`.
- **Model tier.** **Opus** — innermost loop, monotonicity certificate, engine
  fit-mode contract; correctness-critical.

---

## PR 3 — Engine G-scaling: profiled locals + variable projection  **[parallel-B, after PR1]**  · Opus

- **Goal.** Fix the super-linear G cost at the engine level so *every* tier and
  the downstream executed global fit benefit. Independent PR, as Ben requested.
- **Techniques.** L (profiled/nested locals), M (variable projection).
- **Files/functions.**
  - `asymmetry_global.py::fit_global` (~174) — add a profiled objective: outer
    Minuit over globals only; inner per-dataset local solves (parallel over G).
    Keep the joint path as `strategy="joint"` and profiled as
    `strategy="profiled"` behind a flag; default stays joint until the harness
    signs off.
  - `engine.py` — VarPro: detect linear params (amplitudes, constant background)
    from model metadata; solve them by linear least-squares inside the residual;
    remove from the Minuit parameter vector (still counted in IC `k`).
  - `models.py` / `parameters.py` — mark which parameters are linear so VarPro
    finds them generically (do **not** hard-code component names).
- **Sequencing.** Fully parallel with PR 2 — different files (engine vs
  wavefront). The one shared file is `asymmetry_global.py::fit_global`: PR 2
  touches it for the screening fit-mode flag, PR 3 for the profiled strategy.
  **Coordinate:** pre-agree the `fit_global` signature by landing a tiny stub
  commit off PR 1's branch that adds both `screening: bool` and `strategy: str`
  params with no-op defaults; both PRs branch from it.
- **Validate.** Profiled vs joint must match parameter values within fit
  tolerance and IC to <0.5 on every harness case; VarPro values+errors vs
  current on the same; the harness G-sweep must show ~linear scaling.
- **Model tier.** **Opus** — the hardest numerics (profiling, VarPro
  weighting/bounds), highest silent-error risk.

---

## PR 4 — Search strategies + surrogate scoring  **[serial after PR2]**  · Opus

- **Goal.** Add the non-exhaustive search engines the slider needs at Low and
  Balanced. Depends on PR 2 because these share the wavefront/attempt code and
  the exact bounds (A/B) are their safety net.
- **Techniques.** E (Q pre-tests), F (greedy — wire up `_search_parameter_roles`),
  G (Wald surrogate), H (template racing).
- **Files/functions.**
  - `_search_parameter_roles` (~2653), `_best_forward_role_change` (~2805),
    `_forward_role_change_candidates` (~2857): activate greedy as a live path
    (F). **Ensure the winner's flip-neighbourhood is explicitly fitted** so the
    verdict/robustness layer has its delta comparisons (greedy leaves a sparse
    cache — the most likely silent breakage).
  - New helper near the shortlist/prefit code: Q-test (E) consuming the
    single-run prefit cache (`_single_run_prefit_cache_for` ~1513). Fixes tail
    roles, emits the ambiguous set.
  - New surrogate helper: Wald GLS collapse (G) from the all-local joint fit
    covariance; ranks subsets; feeds top-K into the real-fit path.
  - `_run_exhaustive_wavefront_search` (~5497) / staged driver (~1418): template
    racing (H) — layers 0–1 for all, advance top 1–2.
  - `_staged_local_search_settings` (~1363) / `_staged_orchestrator_config`
    (~1338): route which strategy is active.
- **Conflict hotspot.** PR 4 and PR 2 both edit `_run_exhaustive_wavefront_search`
  and the staged driver — **serial, PR 4 after PR 2**, no concurrency here.
- **Validate.** Harness verdict-agreement vs frozen baseline: greedy/Q may
  legitimately disagree on interaction-pair cases — the report must show the IC
  gap is within the robustness delta on disagreements, and
  surrogate-rank-of-winner ≤ K on all cases.
- **Model tier.** **Opus** — surrogate math + verdict-cache correctness.

---

## PR 5 — Effort slider: tiers + wiring + GUI  **[serial after PR4]**  · Sonnet

- **Goal.** Bind the techniques to Low / Balanced / Thorough / Exhaustive and
  surface the control. Pure orchestration + UI once the engines exist.
- **Techniques.** I (Low portfolio cap), J (identifiability demotion),
  K (screening decimation), plus the tier policy wiring E–H and A–D.
- **Tier policy (the deliverable contract):**

  | Tier | Search | V policy | Templates | Per-fit |
  |---|---|---|---|---|
  | **Low** | Q pre-tests (wide bands) → greedy on remainder | 1 warm attempt, escalate on failure | 2–3, P≤5, complexity prior (I), demote degenerate (J) | decimate 4× (K), no Hesse until verdict |
  | **Balanced** (default) | Q (conservative) → surrogate ranks ambiguous middle → verify top-K; bound A/B | escalation-on-anomaly (D); anchors ×2 | full shortlist, raced (H) | decimate 2×, full-res winner refit |
  | **Thorough** | full wavefront over ambiguous middle, exact bounds A/B only (no surrogate skipping) | (D), anchors full battery | full shortlist, raced, generous margins | full resolution |
  | **Exhaustive** | full 2^P, all params promotable (no pre-fixing), bounds with Δ ≥ all robustness deltas | current full multi-start × staged cycles | current 6-template shortlist, no racing | full resolution (+ engine PR 3 speedups) |

- **Balanced acceptance bar:** matches Exhaustive **verdicts** on ≥95 % of
  harness cases; where it differs the IC gap sits inside the robustness delta.
  **Exhaustive contract:** keeps (i) all params searchable, (ii) full
  flip-neighbourhood cache, (iii) full multi-start — it is the referee.
- **Files/functions.**
  - `wizard_scope.py` — add an `EffortTier` enum + payload (mirror
    `WizardScopePreset` ~106; serialise alongside scope).
  - `_staged_orchestrator_config` (~1338) / `_staged_local_search_settings`
    (~1363) — map tier → strategy knobs (which of E–K/A–D are on).
  - `build_global_fit_wizard_recommendation` (~1379) — accept + thread the tier.
  - Decimation (K) helper near dataset prep; demotion (J) near
    `_shortlist_template_keys` (~2339); complexity prior (I) in the IC penalty
    path.
  - GUI: `gui/windows/global_fit_wizard_window.py` +
    `gui/panels/fit/global_tab.py` — the slider control, default Balanced,
    "screening-grade" label at Low. Persist tier in the wizard payload.
- **Model tier.** **Sonnet** — mostly wiring, enum, GUI, config mapping; the
  hard algorithms are already built and harness-gated in PRs 2–4. Escalate a
  specific sub-question to Opus only if the tier↔knob mapping proves subtle.

---

## Parallelization & model-assignment map

```
                 PR1 (harness, Opus) ── blocks everything ──┐
                          │                                 │
        ┌─────────────────┴───────────────────┐            │
        ▼                                      ▼            │
  PR2 free-wins (Opus)                 PR3 engine G (Opus)  │  ← PR2 & PR3 PARALLEL
  bounds+Hesse+escalation              profiled locals+VarPro   (pre-agree fit_global sig)
        │                                      │            │
        └──────────────┬───────────────────────┘            │
                       ▼                                     │
             PR4 search+surrogate (Opus)  ← SERIAL after PR2 │
             (shares wavefront hotspot with PR2)             │
                       │                                     │
                       ▼                                     │
             PR5 slider+GUI (Sonnet) ── SERIAL after PR4 ────┘
```

- **Serial spine:** PR1 → PR2 → PR4 → PR5. PR3 branches off PR1 and rejoins
  before PR5.
- **Safe parallel pair:** PR2 (wavefront/attempt loop) and PR3 (engine numerics)
  touch mostly disjoint files. The single collision surface is
  `asymmetry_global.py::fit_global`; mitigate with the shared stub commit above.
- **Conflict hotspots in `global_fit_wizard.py` (5871 lines):**
  - `_run_exhaustive_wavefront_search` (~5497) — PR2 (bounds) **and** PR4
    (racing/greedy routing). Kept serial precisely to avoid a two-agent
    collision here.
  - `_staged_multi_local_assignment` (~3035) / `_evaluate_attempt_variants`
    (~3493) — PR2 only (escalation). PR4 stays in the greedy/surrogate/shortlist
    regions (~2339–2857) — a different span, low overlap once PR2 has merged.
  - `_staged_orchestrator_config` / `_staged_local_search_settings`
    (~1338–1378) — PR4 (strategy routing) then PR5 (tier mapping). Serial.
- **Model tiers:** PR1–PR4 Opus (correctness-critical numerics + loop surgery);
  PR5 Sonnet (wiring/GUI). This concentrates the hard reasoning where silent
  errors are expensive and lets the cheaper model do the mechanical assembly on
  a harness-proven foundation.

---

## Pseudocode for the tricky kernels

**Layer bound (A):**
```
fit all_local  -> chi2_floor            # one node, near-free via prefit cache
incumbent = best IC seen so far (start = all_local's IC)
for k in 0..P:                          # Hamming layer = #local params
    penalty_k = 2 * (n_global + k*G)    # AIC form; AICc/BIC analogous, monotone in k
    if chi2_floor + penalty_k > incumbent + delta_margin:
        break                           # no node in layer >= k can win
    for assignment in layer(k):
        fit; update incumbent
# delta_margin >= max robustness delta so gate-eligible near-misses still fit
```

**Homogeneity Q-test (E):**
```
for p in promotable_params:
    theta_g, sigma_g = per_dataset_estimate(p)      # from single-run prefit cache
    theta_bar = sum(theta_g/sigma_g^2) / sum(1/sigma_g^2)
    Q = sum((theta_g - theta_bar)^2 / sigma_g^2)     # ~ chi2_{G-1} under 'global'
    if any sigma_g invalid or p at limit: ambiguous.add(p); continue
    if p_value(Q, df=G-1) < p_local_thresh:  fixed_local.add(p)   # clearly varies
    elif p_value(Q) > p_global_thresh:       fixed_global.add(p)  # clearly constant
    else:                                    ambiguous.add(p)
# enumerate power set over 'ambiguous' only; band widths are the effort knob
```

**Wald quadratic surrogate (G):**
```
fit all_local jointly -> theta_hat (per param per dataset), covariance C
for subset S of params to globalise:
    # GLS collapse of S's per-dataset estimates onto a shared value:
    predicted_delta_chi2(S) = sum_{p in S} quadratic_penalty(theta_hat_p{.}, C_p)
    surrogate_IC(S) = chi2_all_local + predicted_delta_chi2(S) + penalty(|S| globalised)
rank subsets by surrogate_IC
real-fit top-K per layer; if realised order disagrees near top, grow K
```

**Profiled locals (L):**
```
def objective(globals):
    total = 0
    for g in datasets:                  # parallelisable over G
        # inner: small local-only fit with globals held fixed
        locals_g = minuit_solve(residual(dataset_g, globals, locals), locals_only)
        total += chi2_g(globals, locals_g)
    return total
outer = minuit_solve(objective, globals_only)   # tiny Hessian, no n_local*G blowup
# VarPro (M): inside residual(), solve linear params (amplitudes, bg) by lstsq,
# not by Minuit; keep them in IC k-count.
```

---

## 6. Post-delivery: the orchestrator bake-off (not in the 5 PRs)

`GlobalSearchOrchestrator` (relaxed → discrete → beam) is the highest-ceiling
path but unproven. **After** PR 5 ships, run it head-to-head against
bounded-wavefront on the harness (verdict agreement, wall, fit count). Promote it
to the Balanced engine *only* if it wins. Do not gate the slider release on it,
and do not spend a PR slot wiring it speculatively.
