# Study: global-fit-wizard-efficiency

**Umbrella:** performance / internal-refactor (not a reference-program port —
no WiMDA/musrfit/Mantid seam). **Status:** study → phased plan complete,
decision-ready. Ben has approved proceeding.

This study turns a set of vetted efficiency directions into a concrete +
phased plan to make the **global fit wizard role search** dramatically faster,
and to deliver a **Low / Balanced / Thorough / Exhaustive effort slider**. It
selects the techniques worth doing, drops the redundant/low-yield ones (with
reasons), and bundles the work into **exactly five independently-mergeable
PRs**, ordered for parallel subagent execution.

Companion files:
- `comparison.md` — selected-vs-dropped technique inventory with rationale.
- `implementation-options.md` — the 5-PR phased plan, parallelization/model map,
  conflict hotspots, and pseudocode for the tricky kernels.
- `test-data.md` — the golden-verdict regression harness spec (PR 1 deliverable).
- `verification-plan.md` — cross-cutting things to validate before trusting each
  technique, and the acceptance bars per tier.

---

## 1. Problem & benchmark recap

The wizard screens which physics parameters of a fit model should be **global**
(shared across a series of `G` datasets) vs **local** (free per dataset), using
an information criterion (AIC / AICc default / BIC). For `P` promotable
parameters it currently runs an **exhaustive "wavefront" over the full 2^P
power set** of global/local assignments, with no pruning between Hamming layers.
Each assignment triggers **V ≈ 30–80 individual Minuit fits** (multi-start
initial-value variants × 2–4 staged refinement cycles × optional simplex
rescue). The joint `global_fit` concatenates all `G` datasets into one Minuit
optimisation whose Hessian is ~O((n_global + n_local·G)²).

Measured cost (iminuit, real + synthetic):

| Axis | Observation |
|---|---|
| **Headline** | Real corpus, G=3, discovered model: **549 s**, 97 % in the wavefront. 19,562 fits, 9.1 M fevals, winning template P=7 (2⁷ power set). |
| **P sweep** (1 template, G=3) | exp+const P≈3 → 8 s; biexp+const P≈5 → 44 s; triexp+const P≈7 → 364 s. Wall grows ×5–8 per +2 params (worse than the ×4 node growth): the per-node fit multiplier V itself grows 31 → 43 → 77. |
| **G sweep** (biexp+const, node count flat at 33) | G=2 → 14 s; G=4 → 36 s; G=8 → 98 s; G=16 → 286 s. Per-fit cost ~O(G^1.4–1.5). |

**Cost model:** `wall ≈ Σ_templates [2^P] × [V ≈ 30–80] × [C(G,n) super-linear in G]`.
P is exponential (headline driver); G is super-linear; V is a large,
P-growing constant. Screening + portfolio + rerank are <3 % of wall.

The structural key exploited throughout is **nested-model monotonicity**:
making any parameter local strictly adds freedom, so χ² can only decrease down
the lattice. Almost every admissible bound and sanity check below falls out of
this.

## 2. Code map (verified against the tree)

- `core/fitting/global_fit_wizard.py` (5871 lines) — the whole wizard.
  Key regions: `_build_global_fit_wizard_recommendation_staged` (~1418),
  `_run_exhaustive_wavefront_search` (~5497), attempt-variant machinery
  `_assignment_attempt_variants`/`_trim_assignment_attempt_variants` (~4444–4511)
  and `_evaluate_attempt_variants` (~3493), staged cycles
  `_staged_multi_local_assignment` (~3035) + `_probe_assignment_candidate` (~2995),
  dormant greedy `_search_parameter_roles` (~2653) + `_best_forward_role_change`
  (~2805) + `_staged_globalization_assignment`, IC helpers around
  `metric_value`/`parameter_count`/`additive_terms` (~154–185), shortlisting
  `_shortlist_template_keys` (~2339), config `_staged_orchestrator_config`
  (~1338) / `_staged_local_search_settings` (~1363), serialisation
  `serialize_global_fit_wizard_recommendation` (~1843).
- `core/fitting/asymmetry_global.py` (348 lines) — `fit_global` (~174), the
  joint Minuit call; `_validate_parameter_partition`, `_resolve_initial_params`.
- `core/fitting/engine.py` — shared Minuit driver (strategy / Hesse / tolerance
  knobs live here).
- `core/fitting/global_search/orchestrator.py` (157 lines) — dormant
  `GlobalSearchOrchestrator` (relaxed → discrete → beam) + `GlobalSearchConfig`
  + `_approximate_candidate_score`. Zero live callers.
- `core/fitting/wizard_scope.py` — `WizardScopePreset` (~106), `WizardScope`,
  `resolve_scope`. The new `EffortTier` enum should mirror this pattern.
- GUI: `gui/windows/global_fit_wizard_window.py`, `gui/panels/fit/global_tab.py`.

## 3. Deliverables at a glance

1. **Golden-verdict regression harness** (PR 1, gate for everything). See
   `test-data.md`.
2. **Free wins** — exact bounds + fit-cost hygiene, all tiers (PR 2).
3. **Engine G-scaling** — profiled locals + variable projection (PR 3, Ben
   specifically wants this).
4. **Search strategies + surrogate** — greedy, Q pre-tests, Wald surrogate,
   racing (PR 4).
5. **Effort slider + GUI** — Low / Balanced / Thorough / Exhaustive (PR 5).

The 5-PR detail, sequencing, parallelization and model-tier assignments are in
`implementation-options.md`.
