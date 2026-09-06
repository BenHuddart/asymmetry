# Global Fit Wizard: transition breaks, phases, and a separable role search

Status: plan agreed with maintainer (2026-09-06); single PR on
`feat/global-wizard-transitions`, phased subagent work with a lead review gate
after every phase. Decisions that change during implementation are appended
under "Decisions recorded".

## Problem

A temperature (or field) series often crosses one or more transitions, and
the fit function that describes the runs on one side does not describe the
runs on the other. The Global Fit Wizard assumes one template and one
global/local role assignment for the whole ordered series, so on such a
series it can only (a) recommend the wrong single model, (b) flag a
"fingerprint jump" and call the result tentative, or (c) time out. Measured
on a real 29-run zero-field scan with two transitions (study of 2026-09-05):

- The series template portfolio is built from the *median* fingerprint and a
  half-of-the-runs vote on plain peak detection (`_scoped_series_templates`,
  `_series_multiplet_pattern_family_keys`), so the two heavily damped lines
  that describe every run in the low-temperature phase never enter the
  portfolio. The single-run Fit Wizard, whose fingerprint includes the
  damped-line scan, finds them in 10–20 s per run.
- The coupled role search (`_run_exhaustive_wavefront_search`) exhausts its
  180 s budget on an 11- or 14-run segment before finishing one
  shared-parameter fit and returns the all-local anchors, i.e. no sharing.
  About half of that time re-solves the all-local anchor as one joint
  98-parameter Minuit problem although phase 1 already holds every per-run
  fit; the rest is the joint solver's ~G² per-iteration cost on badly seeded
  all-global nodes.

## Design (settled)

**Mental model.** The series is partitioned into *phases* (contiguous runs
along the sweep axis). Within a phase one template and one global/local
assignment hold; a *break* between phases is a change of **structure**
(template and/or assignment), never merely of values. Each phase becomes a
data group nested under the series group, owning its own global-fit series.

1. **Breaks are structural only**, and the structure is the template
   *family* (oscillatory, relaxation, multi-rate, Kubo-Toyabe, …). Adjacent
   segments of the same family are one segment; the template within the
   family and the sharing pattern are priced into a segment's cost but are
   decided by the coupled fit within the phase, never by a break. A drifting
   global therefore has exactly two honest representations, global or local;
   it can never be approximated by a staircase of breaks.
2. **Minimum phase length L = 3** runs. A stub shorter than L is admitted only
   at either end of the series; it receives no coupled fit, is scored at its
   own per-run cost plus the usual break penalty, and is reported as
   "excluded from the global fit: looks like a different phase". Interior
   runs are never excised (a run that does not fit its neighbours is a
   per-run gate failure, which the existing verdicts already report).
3. **Partition cost** = Σ_s IC(best structure of segment s) + β per break,
   minimised exactly by dynamic programming over the run order ("exactly k
   breaks" recursion, so the whole penalty path comes out of one pass).
   The number of breaks is read off the path: the elbow is pre-selected and
   the path is shown, so the user can move the choice.
4. **Segment costs come in tiers**, each an admissible bound for the next:
   tier 1 (free) = per-run per-template IC sums from the phase-1 table (the
   all-local cost, a lower bound on every assignment of that template);
   tier 2 (closed form) = tier 1 plus the full-covariance Wald/GLS collapse
   cost of the best sharing pattern; tier 3 (exact) = the coupled role search
   on the segment. The partition search runs on tier 2; tier 3 runs only on
   the segments of the selected partition and of its one-run-shifted
   neighbours.
5. **The role search is separable, not exhaustive.** All-local comes from
   phase 1 and is never refitted jointly. A full-covariance surrogate scores
   every assignment at once. Backward elimination (globalise one parameter
   at a time from all-local, warm-started from the GLS collapse) is the exact
   path, then the winner's single-flip neighbourhood is fitted so verdicts
   and parameter-sharing diagnostics are exact. Coupled fits run with
   `strategy="profiled"` at the series' search resolution; the winner is
   refitted at full resolution with the joint solver for reporting. The
   exhaustive wavefront stays behind the `search_engine` seam as the harness
   referee. The new engine is the default for every effort tier.
6. **Phase 1 goes through the single-run wizard.** The series alphabet is the
   union of the per-run single-run recommendations' assessed templates (plus
   multiplet templates supported by *any* run), reusing the fit tabs' cached
   single-run recommendations when present. Every run then gets a score for
   every alphabet template at one common *series search resolution* via
   completion fits seeded from the per-run results.
7. **Phases are data groups nested under the series group** (`parent_group_id`),
   each owning one global-fit series; compact headers (ordinal + range) with
   an ⓘ popover for provenance and actions; phase colour on a 4 px stripe
   and swatch, never the row background; "Move to phase…" as the manual
   boundary override (marks both phase series stale); selecting the series
   header selects every phase's runs. The Fit Parameters panel shows the
   phase swatch on each phase's series button and, for an active phase
   series, shades the phase range and draws the boundaries. Cross-phase trend
   stitching and cross-segment parameter ties are deferred.
8. **v1 fits segments independently.** No heterogeneous (per-dataset model)
   global fit in this PR.

### Statistical detail

*Tier-2 surrogate.* For run r with all-local estimate θ_r and covariance C_r
(from the per-run `FitResult.covariance`; diagonal fallback from
`uncertainties` when the covariance is missing or not `covariance_accurate`),
the cost of globalising a subset S is the GLS collapse with the other
parameters profiled out: W_r = (C_r[S,S])⁻¹, θ̄_S = (Σ_r W_r)⁻¹ Σ_r W_r θ_{r,S},
Δχ²(S) = Σ_r (θ_{r,S} − θ̄_S)ᵀ W_r (θ_{r,S} − θ̄_S). Surrogate IC(S) =
Σ_r χ²_r + Δχ²(S) + penalty(k(S), n) with k(S) = |S| + (P − |S|)·G and the
same `_metric_penalty` the exact path uses. The collapse also gives the warm
start for the exact fit: shared values θ̄_S and per-run conditional locals
θ_{r,¬S} − C_r[¬S,S] C_r[S,S]⁻¹ (θ_{r,S} − θ̄_S). Ill-conditioned blocks
(condition number above 1e12, non-finite entries, parameters at a bound)
fall back to the diagonal form for that run; a parameter at a bound on any
run is never pre-fixed by the surrogate.

*Backward elimination.* From S = ∅ (all-local, exact from phase 1), repeat:
pick the parameter p ∉ S with the best surrogate IC(S ∪ {p}); fit S ∪ {p}
exactly (profiled, search resolution, warm start from the collapse); accept
if the exact IC improves on the incumbent by more than 0; else stop. The
child has fewer free parameters than the parent, so χ²_child ≥ χ²_parent − ε
(ε = `_WARM_CERTIFICATE_EPSILON`) is the monotonicity certificate; a
violation means the parent was mis-converged and the parent is refitted
from the child's values before continuing. If the warm fit fails to
converge, escalate to the existing multi-start battery for that node only.
When the elimination stops, fit the single-flip neighbourhood of the winner
(every parameter toggled, skipping cached nodes) at search resolution, then
refit the winner and its neighbourhood at full resolution with the joint
solver (`prefit_cache_override={}`; never mix resolutions on one
leaderboard). At most P + P + (P + 1) coupled fits per template; templates
are raced (the top three by phase-1 sum and surrogate-predicted best IC; the
existing cross-template incumbent bound prunes the rest).

*Series search resolution.* One rebin factor for the whole series:
min over runs of `fit_wizard.analysis_rebin_factor` (bandwidth-aware, uses
the detected lines including damped-scan lines), floored at 1. Every tier-1
completion fit and every search-phase coupled fit runs at this factor; IC
values on the leaderboard are therefore mutually comparable; the reported
winner is refitted at full resolution.

*Penalty path and elbow.* With F_k the minimum total cost with exactly k
breaks (k = 0 … ⌊G/L⌋ − 1) the marginal gains are g_k = F_{k−1} − F_k. A
break is admissible when g_k ≥ β_floor with β_floor = c·ln(N_total),
N_total the total number of fitted points in the series at the search
resolution, c = 2 by default (a modified-BIC-style floor). The pre-selected
partition has k* = the largest k with g_1 … g_k all admissible. The card
shows the whole path with g_k per row. `c` is a `PartitionConfig` field, not
a GUI control.

*Boundary estimate.* A break between runs a and b (adjacent in axis order)
is reported at (x_a + x_b)/2 ± (x_b − x_a)/2.

## Code map (from the 2026-09-06 reconnaissance)

- Core wizard: `src/asymmetry/core/fitting/global_fit_wizard.py`
  (`build_global_fit_wizard_screening_recommendation` :1376,
  `build_or_complete_single_fit_wizard_recommendations_for_global_portfolio`
  :1165 with the exact-key-match reuse gate at :1220–1229,
  `_build_single_fit_prescreen_assessments` :1576, dispatch :2287–2313,
  `_fit_exact_assignment` :4401, `_run_heuristic_search` :7325,
  `_fill_winner_flip_neighbourhood` :7045, `_decimated_datasets_for_search`
  :7135, `_refit_states_at_full_resolution` :7215,
  `_run_exhaustive_wavefront_search` :7611, constants :180–195 and :262–267).
- Single-run wizard: `src/asymmetry/core/fitting/fit_wizard.py`
  (`build_fit_wizard_recommendation` :2231,
  `build_fit_wizard_recommendation_for_templates` :2784 — no peak analysis,
  so multiplet seeds need a per-run `TemplateSeedContext`;
  `build_oscillatory_multiplet_templates` :1617; `analysis_rebin_factor`
  :5282; `_serialize_fit_result` :3535 drops covariance).
- Surrogate kernels: `src/asymmetry/core/fitting/global_search/homogeneity.py`
  (diagonal only today; no production callers outside the wizard).
- Engine: `src/asymmetry/core/fitting/engine.py` `FitEngine.global_fit`
  :1273 (`strategy="profiled"` :1925 — serial inner solves, no
  `initial_step_sizes`, no covariance on results; joint path emits per-run
  covariance :1846–1906); `MuonDataset.rebin` (`core/data/dataset.py:226`).
- Harness: `tools/global_wizard_harness.py` (no `--engine` flag yet; tiers
  → `_run_wizard_with_tier` :298).
- Project model: `core/representation/group.py` `DataGroup` (five fields),
  `project_model.py:144–260` group CRUD, `schema.py` v18 with the
  `_migrate_vN_to_vN+1` ladder (:242–322), `series.py` `FitSeries`.
- Browser: `gui/panels/data_browser.py` (`_rebuild_table` :1478 two-level
  loop, `_display_order`, `_add_group_header_row` :1514, `_add_dataset_row`
  :1814, context menus :3607/:3822, `_populate_send_to_group_menu` :3736,
  `_dataset_run_numbers_from_keys` :2864, `get_state` :4921).
- Series recording: `gui/mainwindow.py` `_resolve_batch_group` :12040,
  `_record_global_fit_batch` :11165, `_record_fit_series` :11081.
- Wizard window: `gui/windows/global_fit_wizard_window.py`
  (`_build_result_page` :545, `_populate_from_recommendation` :1372,
  `_start_selected_optimisation` :998, `_run_global_fit_wizard_analysis`
  :164, apply :2051/:2063); global tab apply
  `gui/panels/fit/global_tab.py:3546`.
- Trend panel: `gui/panels/fit_parameters_panel.py` (`_rebuild_group_buttons`
  :1618, `_series_to_plot` :5320 colour `C{idx}`, `_draw_single_series`
  :5361, `_build_gle_export` :6884).
- Tokens: `gui/styles/tokens.py` (`PROFILE_COLORS` :65 pattern),
  `gui/utils/profile_colors.py`.
- Docs: `docs/reference/global_fit_wizard.rst`, `gui_usage.rst` "Data
  groups", `project_files.rst` "Data groups and fit series",
  `parameter_trending.rst`; scenarios `docs/screenshots/scenarios/*.py`
  registered in `capture.py:297`.

## Phases

Each phase is one subagent task in its own worktree off the feature branch,
with a lead review gate before merge. Rules for every phase: no defensive
guards (make bad states impossible by construction; ask rather than hedge);
tests beside the behaviour; `python tools/harness.py test -- <focused>` while
iterating, `--tier fast` after core changes, affected GUI files focused;
nothing from the maintainer's private data (no paths, run numbers, sample
names, temperatures) in the repo.

### Phase A — series alphabet and phase 1 via the single-run wizard (core, Opus)

- `build_or_complete_single_fit_wizard_recommendations_for_global_portfolio`:
  a per-run table is built with `build_fit_wizard_recommendation(dataset,
  scope=…, user_frequencies_mhz=…)` (the full single-run path; runs are
  processed serially, each using its own pool, so nested pools never
  occur) and an existing recommendation for a run is reused when its scope
  signature matches (drop the exact-template-key equality gate).
- Series alphabet = union over runs of the templates the per-run
  recommendations *assessed* (stage 2, not null baselines), capped at 24 by
  keeping, in order, every template that wins or is comparable on any run,
  then by best per-run rank. Multiplet templates are included when any run
  produced one (`oscillatory{n}_*` keys are stable across runs).
- Series search resolution `series_rebin_factor(datasets, recommendations)`
  = min over runs of `analysis_rebin_factor` (≥ 1).
- Completion: for every (run, template) cell missing from the run's table,
  fit the template on the run's rebinned dataset with
  `build_fit_wizard_recommendation_for_templates`, passing a per-run seed
  context built from that run's `peak_analysis` (so multiplet seeds are
  meaningful) and, when a sibling run already fitted the same template, its
  fitted values as the first variant. Cells the run's own table already holds
  at a *different* rebin factor are refitted at the series factor from their
  own fitted values (one warm variant). Result: a complete per-run ×
  per-template table at one resolution, with `FitResult.covariance` kept in
  memory (persisted tables lose it; the surrogate's diagonal fallback covers
  that).
- `_build_single_fit_prescreen_assessments` consumes the completed table as
  before. Instrumentation records alphabet size, completion fit count and the
  series factor.
- Tests: `tests/core/test_global_fit_wizard.py` /
  `test_global_fit_wizard_scope.py` — replace the quorum-vote tests with
  any-run-support tests; a synthetic series where only 40 % of runs carry a
  damped multiplet must put the multiplet template in the alphabet and win it
  on those runs; completion cells exist for every (run, template); reuse of a
  cached single-run recommendation is pinned; the series factor is the min.
- GUI: `global_tab._existing_single_fit_recommendations_for_selected_runs`
  already feeds the window; make the window pass them through unchanged and
  write back generated ones (existing signal). No visual change in this
  phase.

### Phase B — surrogate kernel and partition search (core, Opus)

New module `src/asymmetry/core/fitting/global_search/surrogate.py`:
- `RunEstimate(values: dict[str, float], covariance: NDArray | None,
  covariance_names: tuple[str, ...], uncertainties: dict[str, float],
  at_bound: frozenset[str], chi_squared: float, n_points: int)` built from a
  `FitResult` by `run_estimate_from_fit_result`.
- `collapse_cost(estimates, subset) -> CollapseResult(delta_chi2, shared_values,
  conditional_locals_by_run)` implementing the GLS collapse above with the
  diagonal fallback rule.
- `surrogate_ic(estimates, subset, metric)` and
  `rank_assignments(estimates, free_names, metric) -> list[(subset, ic)]`
  over all 2^P subsets (P ≤ 12; above that, rank only single-parameter
  additions, which is all backward elimination needs).

New module `src/asymmetry/core/fitting/global_search/partition.py`:
- `PartitionConfig(min_segment=3, floor_coefficient=2.0, max_breaks=None)`.
- `SegmentCost` protocol: `cost(i, j) -> (ic, structure)`; a tier-1
  implementation over a `dict[run][template] -> ic` table and a tier-2
  implementation that adds the surrogate's best sharing per template
  (using `rank_assignments`). Missing/failed cells make that template
  infeasible on that segment.
- `partition_series(order, cost, config) -> PartitionPath`: exact DP with
  exactly-k breaks, end stubs allowed only at positions 0 and G with length
  < L (scored per-run best, no coupled fit, flagged `excluded`), adjacent
  equal-structure segments merged, `PartitionPath.solutions[k]` with
  `segments`, `total_ic`, `gain`, `admissible`, `selected_k`, and boundary
  estimates from the axis values.
- Serialisation of `PartitionPath`/`SeriesPartition` (plain dicts).
- Tests (`tests/core/test_global_search_partition.py`,
  `test_global_search_surrogate.py`): surrogate equals the exact quadratic
  on a linear model; diagonal fallback fires on singular covariance; planted
  template change at run k found with the elbow at 1 break; planted
  role-structure change found via tier 2; smooth drift of a global yields
  0 breaks; a single paramagnetic run at the top end is excluded, not
  absorbed; interior singleton never excised; gains are monotone
  non-increasing; boundary estimates.

### Phase C — separable role search engine (core, Opus; after B)

- New engine constant `SEARCH_ENGINE_SEPARABLE = "separable"` in
  `SEARCH_ENGINES`; `_EFFORT_TIER_SEARCH_ENGINE` maps every tier to it;
  `_DEFAULT_SEARCH_ENGINE` becomes it. Dispatch branch at :2287 calls
  `_run_separable_search`.
- `_run_separable_search(datasets, shortlisted_templates, template_contexts,
  prescreen_assessments, …)`: builds `RunEstimate`s from the prescreen
  `fit_results_by_run`; all-local node assembled without a fit (χ² sum, IC,
  diagnostics from the per-run results); backward elimination per template
  as specified above; racing over the top three templates; flip
  neighbourhood; full-resolution refit of winner + neighbourhood with
  `strategy="joint"`; assessments finalised exactly as
  `_finalise_heuristic_assessments` does (fixed names, parameter
  recommendations from the cache, assessment keys).
- Coupled fits in the search go through `_fit_exact_assignment` with a new
  `strategy` argument threaded to `global_fit`, a warm start from the
  collapse, one variant, escalation to the battery only on failure.
- Engine: `_global_fit_profiled` accepts `initial_step_sizes` for the outer
  problem (clamped like the joint path). Nothing else in the engine changes.
- Assignment tasks (the exact fits of a template's elimination path and
  flip neighbourhood) run on the existing spawn pool across templates and,
  later, across segments; the profiled inner loop stays serial.
- Harness: `--engine {separable,exhaustive}` option passing `search_engine=`;
  acceptance = ≥ 95 % verdict agreement with the frozen exhaustive baseline
  and every disagreement inside the IC-gap tolerance; fit counts reported.
- Tests: the `search_engine` seam tests in `tests/core/test_global_fit_wizard.py`
  gain separable-engine cases (planted global/local roles recovered,
  flip neighbourhood fitted, all-local never refitted jointly — pinned by
  the `exact_fit_invocations` counter, resolution never mixed on the
  leaderboard, certificate violation refits the parent); harness test
  updated for `--engine`.

### Phase C2 — per-phase optimisation and the partitioned recommendation (core, Opus; after B, C)

- `GlobalFitWizardRecommendation` gains `partition_path: PartitionPath | None`
  and `phase_assessments: dict[(k, segment_index), GlobalCandidateAssessment]`
  keyed by the path solution and segment; `recommended_key` and the card
  semantics stay for the single-segment case; a partitioned recommendation
  has `recommended_partition_k` and per-phase recommended keys.
- `build_global_fit_wizard_screening_recommendation` computes the tier-2
  partition path from the completed phase-1 table (cheap, always) and
  stores it.
- `build_global_fit_wizard_recommendation(..., partition_k=k)` runs the
  separable search per segment of solution k and of the one-run-shifted
  neighbours of each break (only where the shift keeps every segment ≥ L),
  re-scores the path rows it touched with exact ICs, and returns the merged
  recommendation. Segments are independent tasks on the pool.
- Serialisation/deserialisation and `merge_global_fit_wizard_recommendations`
  extended; `rerank_…` re-selects per phase.
- Tests: synthetic two-phase series (template change) end to end: the
  screening path has its elbow at 1 break, optimisation returns one
  assessment per phase with the planted roles, shifted-neighbour
  verification leaves the break in place; excluded stub carries no
  assessment; serialisation round-trips.

### Phase D1 — project model: nested phase groups, tokens (core/project, Sonnet)

- `DataGroup` gains `parent_group_id: str | None`, `phase_ordinal: int | None`,
  `phase_range: tuple[float, float] | None`, `phase_boundaries: dict`
  (lower/upper estimate ± half-gap, or `None` at the series ends),
  `phase_color: str | None`, `phase_provenance: dict` (wizard date, selected
  k, gains, model title, confidence). `ProjectModel` gains
  `create_phase_groups(parent_id, phases)`, `phase_groups_for(parent_id)`,
  `move_run_to_phase(run, phase_id)` (marks both affected series stale via
  the existing membership snapshot), `remove_data_group` cascading to
  phases (with the same keep-fits/delete-fits choice).
- Schema v19: additive migration (`parent_group_id` default `None`, phase
  fields default `None`), docstring paragraph, `_SUPPORTED_VERSIONS`,
  tests in `tests/project/` updated for the constant.
- `tokens.PHASE_COLORS` (cobalt #2F4DA0, aqua #2AA198, leaf #4F7F1E, gold
  #D9A21B, brick #B8432B) with dark-theme variants beside the existing
  light/dark token pattern, and `gui/utils/phase_colors.py` with
  `phase_color(ordinal)` and `soft_phase_background`.
- `browser_state.data_groups` view entries echo `parent_group_id`.

### Phase D2 — Data Browser: nested phases (GUI, Opus; after D1)

- `_rebuild_table` renders a third level: series header → phase sub-header
  (`_add_phase_header_row`: collapse chevron, swatch, "Phase I · 1.8 – 16.0 K",
  ⓘ button) → member rows with a 4 px phase stripe (a left-edge decoration
  in the row delegate, not a background tint). Excluded runs (members of the
  parent, of no phase) render under the parent after the phases, hatched
  stripe, italic, badge "excluded · looks like a different phase".
- `_display_order` and `_integrate_registry_into_display` exclude phase
  groups from the top level; sorting keeps phases in ordinal order and
  members in axis order within a phase.
- Selection: `_dataset_run_numbers_from_keys` resolves a series header to
  all phases' runs plus excluded runs; a phase header to its runs.
- Context menus: phase header → Collapse/Expand, Rename Phase, Fit this
  phase…, Show series from this phase, Ungroup (cascade rules from D1);
  member row inside a partitioned series → "Move to phase ▸" submenu (all
  phases + "Exclude from phases"), built like `_populate_send_to_group_menu`.
- ⓘ popover (`PhaseInfoPopover`, a frameless `QFrame` beside the header):
  model, fit state/confidence, shared parameters, boundary estimate, found
  by (date, k, gains), actions Fit this phase / Show series / Rename.
- `get_state`/`restore_state` carry phase groups; series highlight (amber)
  unchanged.
- Tests: `tests/gui/test_data_browser_phases.py` — rendering order,
  selection semantics, move-to-phase marks series stale, excluded row,
  popover contents, state round-trip.

### Phase D3 — wizard window: Transitions card, per-phase optimise, apply (GUI, Opus; after C2, D1)

- Result page gains a `TransitionsCard` between the series card and the
  shortlist: one row per path solution (breaks, boundaries as temperatures,
  gain, admissible marker), elbow row pre-selected, plain summary
  ("2 transitions found: 16.5 ± 0.5 K and 28.5 ± 0.5 K"); selecting a row
  recolours the series overlay by phase (replacing the axis gradient) and is
  what **Optimize phases** runs (`partition_k`). Excluded stubs are named.
- After per-phase optimisation the answer card summarises per phase
  (template, roles, confidence) and **Apply phases** creates the phase
  groups under the series group (D1 API), records one global-fit series per
  phase through the existing `global_fit_completed` seam (one emission per
  phase with the phase group bound), and binds the global tab to the first
  phase. The single-segment path (k = 0) is unchanged.
- Cache/persistence of the partitioned recommendation through the existing
  wizard cache entries.
- Tests: `tests/gui/test_global_fit_wizard_window.py` additions with a faked
  partitioned recommendation: card rows, elbow preselect, optimise call
  arguments, apply creates groups/series.

### Phase D4 — Fit Parameters panel: swatches, bands, boundaries (GUI, Sonnet; after D1)

- Series buttons of phase-owned series carry the phase swatch (colour from
  the owning `DataGroup.phase_color`, threaded through `load_representation_series`
  entries); the plot colour for such a series is the phase colour instead of
  `C{idx}`.
- When the active series belongs to a phase, `_draw_single_series` shades
  the phase range (`axvspan`, soft phase colour) and draws each boundary as
  `axvline` with an `axvspan` uncertainty band; GLE export mirrors both.
- Tests in `tests/gui/test_fit_parameters_panel.py` (swatch present, band and
  lines drawn for a phase series, absent for a plain series, GLE contains
  the band).

### Phase E — documentation and changelog (docs, Sonnet; after all)

- `docs/reference/global_fit_wizard.rst`: revise the "fit each run
  individually across a transition" paragraph; new sections "Transitions:
  phases and the penalty path" (quote the card's strings verbatim), "How the
  role search works" (separable search, referee), "Phases in the Data
  Browser and Fit Parameters panel"; update the effort-tier section (engine
  unchanged across tiers, now separable).
- `gui_usage.rst` "Data groups": phase sub-groups, ⓘ popover, Move to phase;
  `project_files.rst`: schema v19 fields; `parameter_trending.rst`: phase
  swatches, bands, boundaries.
- Screenshot scenarios: `global_fit_wizard_transitions` (synthetic two-phase
  series from `docs/screenshots/data/archetypes.py`, a new
  `make_two_phase_zf_tscan`), `data_browser_phases`, and an update to the
  trending scenario; registered in `capture.py`.
- `CHANGELOG.md` `[Unreleased]`: Added (transitions, phases), Changed
  (separable search default, series alphabet), Fixed (all-local anchor
  refit; timeout on real series).
- `docs/PLANS.md` entry; `docs/porting/global-fit-wizard-efficiency/README.md`
  gets a pointer to this plan's §5 outcome.

### Phase F — integration, gates, PR (lead)

- `python tools/harness.py validate`, `structural`, `lint`, `docs`.
- `tools/global_wizard_harness.py --engine separable --compare-baseline`.
- Private acceptance gate on the maintainer's series (scratchpad only):
  alphabet contains the two-line damped template; per-run screen wins it in
  the low-temperature phase; the elbow selects 2 breaks at the known
  transitions; per-phase optimisation finishes; end-to-end under 10 minutes
  cold on the laptop. Reported in the PR text as pass/fail with timings and
  nothing else.

## Gates

Each phase: focused tests green, `--tier fast` green for core phases, lead
review of the diff against this plan and the no-guards rule, then merge into
the feature branch. Phase F runs the full validation once.

## Decisions recorded

- 2026-09-06: structure-only breaks; L = 3; end stubs excluded; elbow
  pre-selected with the path visible; segments fitted independently in v1;
  separable search is the default engine; phases nest under the series group
  as real data groups; compact headers with ⓘ popover; excluded runs stay in
  their origin group; trend panel gets swatches, bands and boundaries; the
  five-colour phase palette above.
- 2026-09-06 (Phase C): the separable engine chooses its minimiser architecture
  per node rather than always profiling. `strategy="profiled"` is used once the
  node's free-parameter count (`n_global + n_local·G`) reaches 20, `"joint"`
  below it. §5 specified profiled throughout; profiled's saving is a smaller
  Hessian, and on a short series there is no large Hessian to save while its
  outer loop re-solves every dataset each iteration (measured on the harness's
  5-parameter, 3-run cases: several times the joint cost for the same verdict).
  The full-resolution refit of the winner and its flip-neighbourhood stays
  joint, as specified.
- 2026-09-06 (Phase C2): **the partition is scored with BIC**, whatever ranking
  metric the user chose. Per-run costs in the table are `chi2 + k·ln(n_run)` at
  the series search resolution and tier-2 segment costs use
  `SelectionMetric.BIC`. Reason, measured on the real 29-run scan: two damped
  lines plus a relaxation *nests* plain relaxation, so under AIC a structural
  change between them costs only 2 per extra parameter per run and tier 1 saw
  almost nothing; under BIC (`ln n ≈ 10` per parameter there) the same change
  scored ~100 per break and the path had a clean elbow. The ranking metric still
  decides which candidate wins *within* a phase.
- 2026-09-06 (Phase C2): **cell feasibility is "the fit converged"**
  (`FitResult.success`), not the single-run wizard's per-run disqualifiers
  ("amplitude consistent with zero", "rate unresolved"). Treating a disqualified
  cell as infeasible makes its template infeasible on every segment containing
  that run, which forced breaks around single weak runs *inside* a phase; handing
  the IC the run's real χ² let the partition cost decide, and gave the right
  answer.
- 2026-09-06 (Phase C2): **tier 2 walks greedily, not exhaustively**. On G = 29,
  24 templates, P ≤ 9 the `rank_assignments` enumeration took ~57 s, because the
  DP scores O(G²) windows for every template. Three changes bring a realistic
  table to ~0.4–1.5 s: `greedy_assignment` forward selection (`P(P+1)/2`
  collapses, provably the same subset when the parameters score separably);
  `OrderedCollapse`, which holds each subset's prefix sums so a *window* costs
  one small solve rather than a pass over its runs; and an exact per-window
  bound across templates (`Σχ² + penalty(P, n)` is a floor, so a template whose
  floor already loses is skipped). A degenerate table — every template
  statistically identical, parameters i.i.d. per run — still costs ~10 s; a real
  alphabet is not that.
- 2026-09-06 (Phase C2): **tier-3 verification covers the elbow's neighbours in
  `k` and in position** — solutions `k*−1, k*, k*+1` where they exist, plus each
  break of `k*` shifted one run either way where every phase stays ≥ L. Every
  distinct segment across those is fitted once, the rows they touch are re-scored
  with exact per-segment BICs, and `selected_k` is re-derived from the exact
  gains. `recommended_partition_k` follows the re-evaluated elbow only inside the
  verified window. A segment shorter than L never receives a coupled fit and an
  excluded stub keeps its per-run cost.
- 2026-09-06 (Phase C2): **segments run serially in the parent, templates fan out
  over one hoisted spawn pool**. Pools must not nest, so the only question is
  which level gets the workers, and the alphabet is always larger than the number
  of phases. Measured on the two-phase synthetic (8 verified segments): a pool
  opened per `_drain_separable_tasks` call cost 8.3 s in pool startup alone;
  hoisting one pool across every segment (`shared_executor`) brought it to 1.9 s;
  no pool at all — which is what a segment-level task would experience internally
  — was 1.0 s, because that synthetic has two templates and millisecond fits. The
  pool's cost is one fixed spawn while its benefit scales with alphabet size and
  fit cost, so the hoisted pool is the shipped arrangement.
- 2026-09-06 (Phase C2): the **series search resolution** is threaded into the
  per-phase searches as both `search_rebin_factor` and `prescreen_rebin_factor`,
  so each segment's all-local anchor is the phase-1 pre-screen's own per-run fits
  and costs no fit at all.
- 2026-09-07 (integration, private gate): **a break is a change of template
  family, not of template or sharing pattern.** With template-level structure
  keys the tier-2 path on the real series fragmented into seven admissible
  breaks — an osc2→osc1 switch inside the ordered phase, and sharing-pattern
  changes inside the relaxation phase — each worth hundreds of BIC units at 2.6
  million points. With family keys the same table gives a clean elbow at the
  physics. The DP now enforces the rule by construction (adjacent segments,
  end stubs included, must differ in structure key), replacing the post-DP
  merge; `series_template_families` supplies the map from the per-run family
  reports (multiplet templates → `oscillatory`).
- 2026-09-07 (integration): the screening pass no longer runs the serial
  "repair" of failed cells (30 minutes on the real series); coverage of a
  supplied table requires *converged* cells, so a failed cell is retried by the
  parallel completion pass with a sibling warm start instead.
- 2026-09-07 (integration): two source fixes from the private gate — a line
  whose period is under eight native bins cannot be protected by any rebin
  factor and no longer constrains it, and a detected line above the analysed
  record's Nyquist is never seeded (its frequency bounds inverted on the
  Nyquist clamp and Minuit refused the limit).
- 2026-09-07 (integration, private gate): **the partition is scored under a
  per-run penalty convention** — a local parameter pays `ln n_run`, a shared one
  `ln N_segment` — so an unshared segment costs exactly the sum of its runs'
  own BICs and tier 1 bounds tier 2 on one scale. The joint convention
  (`k·ln N_segment`) charged every local parameter an extra `ln G` for the
  company it kept, ~270 BIC units against a nine-parameter template over a
  16-run segment, and pushed the path towards short segments of simple
  templates (a "relaxation" description of the ordered phase). The exactly
  refitted rows are scored the same way (`_partition_bic`).
- 2026-09-07 (integration, private gate): **tier-3 verification is `k*` and
  `k*−1` only.** The full set (`k*±1` plus every one-run shift of each break)
  came to 18 distinct segments on the real series at several minutes each; the
  shifts are dropped in v1 (the boundary's ± half-gap already states the
  position uncertainty) and `k*+1` is not refitted (its surrogate gain was
  already below the floor). A warm node whose one fit fails now escalates to a
  *capped* battery (first seed, one staged cycle, no prefit-only fallback)
  instead of the full multi-start ladder.
- 2026-09-07 (integration, private gate): **phases are fitted and reported at
  the series search resolution**; the per-phase path performs no
  full-resolution refit. The refit of a 12-run phase's winner and flips on the
  native record cost minutes per node (a joint fit over ~1.1 M points, the warm
  refit escalating to simplex), and it put full-resolution rows next to
  search-resolution ones on the path. The exact rows are now scored on the
  analysed points, one scale along the whole path, and the fit a user applies
  from a phase seeds the Batch tab's own global fit on the native record. The
  series-wide (non-partitioned) search keeps its full-resolution refit, now
  warm-started from the winner.
- 2026-09-07 (Phase A2, phase-1 throughput): **stage 1 runs a few per-run
  analyses at once, warm-started along the sweep axis.** §Phase A had them
  serial "because each analysis uses the whole machine"; measured, it does not —
  peak detection, the damped-line scan, the fingerprint, the residual pass and
  the tier gating are serial with the analysis's own pool idle, ~15 s of the
  ~15 s per run on the real 29-run series. `_phase_one_concurrency` runs
  `min(4, cores // 2)` analyses on a parent thread pool with each call's
  `max_workers` divided down (`_phase_one_analysis_workers`), so the host still
  sees one machine's worth of fit workers and no pool nests; `concurrency == 1`
  keeps the old serial path (it is what a two-core CI runner gets). Each
  analysis is handed the nearest *finished* neighbour's fitted values per
  template as an extra first variant (`warm_start_by_template` on
  `build_fit_wizard_recommendation`, prepended by `_assess_candidate_template`
  exactly as the completion path already did), never a replacement, so the
  answer can only improve; the completion order therefore decides which
  neighbour seeds which run, and phase 1 is no longer bit-reproducible.
  Cold completion cells drop to a two-rung ladder. The alphabet also loses,
  before scoring, every candidate that some *one* other candidate beats on
  every run (`alphabet_bound_dropped_keys`): χ²_t,r ≥ IC_u,r on every run makes
  u's all-local cost an upper bound on the best structure of any segment and
  Σχ²_t a lower bound on t's, so t can never win. Comparing against each run's
  *best* IC instead would not be exact — the argmin may move from run to run
  while a segment is scored under one template (IC_u1 = (10, 100),
  IC_u2 = (100, 10), χ²_t = (11, 11): t clears both per-run minima and still
  wins the pair at 24 against 110). Measured on a synthetic 12-run,
  8 000-point, three-family series: 115–139 s before, 73–82 s after; the bound
  dropped nothing there (with three families no single candidate dominates
  every run), so its value is on series where one model leads throughout.
- 2026-09-07 (integration, private gate): **each phase's search owns its pool
  and a tripped budget terminates it.** A shared pool let a tripped phase's
  abandoned fits keep running, so every later phase's anchor tasks queued
  behind them and timed out with nothing done. The per-phase budget is 30
  minutes (a 12-run phase measured ~40 s per coupled fit, up to 18 fits per
  raced template); the series-wide 180 s backstop is unchanged.
