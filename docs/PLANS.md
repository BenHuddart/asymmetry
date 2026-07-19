# Execution Plans

This file tracks repo-local plans that are useful for agents. Keep short tasks
inline here; create a dedicated plan file only when a project spans multiple
subsystems or days.

## Active

### Multiple Grouping Profiles per Project (Explicit Run Assignment)

Status: design settled with maintainer (2026-07-19); delivered as a single PR
on a feature branch, with maintainer review at each milestone below before
the next begins

Generalise the grouping model from "one active profile per fingerprint +
per-run release" to multiple concurrently-used profiles with an explicit
run→profile assignment. Typical use: several samples in one project, each
with its own background/α/deadtime policies, compared side by side. The
existing `GroupingProfile` data model, `resolve_effective_grouping`, and the
digest/caching layer are reused unchanged — the work is concentrated in the
assignment/selection layer (`active_profile_for_run`, the `active` flag,
the dataset persistence branch, `ScopePanel`).

Settled decisions:

- New runs auto-assign to a per-fingerprint **default profile** (today's
  `active` flag, repurposed as default-for-new-runs); users reassign
  manually. Metadata-based auto-assign rules are deferred.
- Per-run overrides stay **full snapshots** but additionally record their
  base profile, so Reattach has an explicit target and the browser marker
  can name it. Delta-on-profile overrides were considered and rejected.
- Assignment UX lives in both the grouping window (ScopePanel "Assign
  to ▸" beside Release/Reattach) and the Data Browser (context menu,
  multi-select).

Milestones (each reviewed before the next begins):

1. **Core assignment model + schema v17** — every dataset persists an
   explicit `profile` name (released runs keep `profile` + full
   `grouping_overrides`); `active` reinterpreted as default-for-new-runs;
   assignment-aware resolution replaces implicit active-profile lookup at
   load/save call sites; profile rename rewrites dataset refs; deleting a
   profile with assigned runs is guarded; v16→v17 migration stamps
   inherited runs with their active profile's name (lossless).
2. **Grouping window** — profile-aware ScopePanel (assigned-profile column,
   "Assign to ▸"), editing-target strip and Reattach target the assigned
   profile, default-profile star in the selector, delete-with-runs
   reassignment prompt; Apply loop targets runs assigned to the edited
   profile.
3. **Data Browser + docs** — context-menu assignment with multi-select, a
   per-profile marker beside the existing ⊗ override marker, docs pass
   (`detector_grouping.rst`, `project_files.rst`, screenshot scenario),
   changelog.

Acceptance criteria:

- Schema v17: every dataset carries an explicit `profile`; v16 projects
  migrate losslessly and single-profile projects behave identically.
- Multiple profiles per fingerprint are usable concurrently; each run
  resolves through its assigned profile; grouping digest / reduction-cache
  behaviour is unchanged (assignment still resolves to `run.grouping`).
- Released runs reattach to their recorded base profile; renaming a
  profile rewrites all dataset refs in the same operation; deleting a
  profile with assigned runs prompts for reassignment.
- Docs and screenshot scenarios updated; `harness validate` green.

### Corrections-tab UX follow-ups (grouping dialog)

Status: complete — all three milestones (adaptive deadtime section →
section-overflow indicator pill → compare pager above the preview) plus a
saturation-readout fix and a post-review fit-restoration fix landed on
`feat/correction-order-alpha-estimation` (commits `8b52aa0`..`9f73250`,
2026-07-17). Plan, execution notes, and the deferred-idea record
(converging-ghost stepper, hold-to-swap) live in
[porting/correction-order-alpha-estimation/corrections-tab-ux-plan.md](porting/correction-order-alpha-estimation/corrections-tab-ux-plan.md).

### Bayesian Experimental Design: Suggest Next Point (Trending)

Status: complete — Phases 1–2 (core + dialog GUI) shipped in PR #203,
Phase 3 (model discrimination, cost weighting, user docs) in PR #205; only
the study's explicitly deferred refinements remain

During a live experiment, after a trend model fit, suggest the next x
(temperature/field) and event count that most constrains the trend model's
parameters — Laplace expected-information-gain acquisition with c-/D-optimality,
a target-precision event-count rule (counts, not time; a user-supplied count
rate converts for display), and closed-form design oracles as tests. A
numerical prototype validated the acquisition math and showed the raw
rank-one post-σ is ~5–10× optimistic near critical points, so the counting
recommendation is calibrated by a Monte-Carlo refit pass
([studies/prototype-results.md](studies/prototype-results.md)).

Full study, method, decision log, and the phased plan (core → dialog GUI →
cost/discrimination → deferred refinements) live in
[studies/bed-next-point-suggestion.md](studies/bed-next-point-suggestion.md).

Acceptance criteria (v1 = Phases 1–2):

- `ParameterModelFitResult` exposes the fit covariance (serialised with the
  project; legacy projects tolerated).
- `core/fitting/experiment_design.py::suggest_next_point` +
  `calibrate_suggestion` are GUI-free and pass the oracle tests (line →
  interval extremes, Arrhenius → 1/T extreme, order parameter → just below
  T_c, ranking honesty + MC-calibrated post-σ vs direct refit,
  unreachable-goal detection).
- Model Fit dialog shows the utility curve + suggestion on demand, with an
  ill-conditioned-fit warning banner; `harness validate` green.

### BED for Knight-Shift Angle Scans (Next Angle + Discrimination)

Status: study + GUI design settled with maintainer (2026-07-10);
awaiting execution-plan approval, then a single PR on a feature branch

Extend the shipped trending BED to the Knight-shift window's angle scans:
suggest the next rotation angle that (a) most constrains the joint K(θ)
fit parameters (Laplace information gain summed over curves — one run
feeds every branch), (b) best tests for rotation-axis misalignment
against a new first+second-harmonic `AngularFourier2` model (the current
two `ANGULAR_MODELS` are one function family, so discrimination needs
it), or (c) best resolves competing crossing assignments (min-cost
set-matching divergence between near-degenerate EM labellings).
Prerequisites: `KnightJointCurve` keeps the per-curve covariance, and the
θ0 canonicalisation fold transforms it exactly (retiring the documented
quadrature approximation).

Full study, method, MuRef survey (no BED and no new fit families to
port), and the phased plan live in
[studies/bed-next-angle-knight-shift.md](studies/bed-next-angle-knight-shift.md).

Also in scope (settled at GUI review): the trend-panel "Knight shift
window…" button is hidden unless the fitted model has Knight-convertible
components (the `Analysis` menu action stays unconditional); the
window's "Suggest next angle" section is a single mode-selector
`PanelSection` (refine / test misalignment / resolve assignment) with
goal + events conversion but no movement-cost grid.

Acceptance criteria (v1 = Phases 1–4 of the study):

- `KnightJointCurve` round-trips a `(names, matrix)` covariance; legacy
  project dicts load with `covariance=None` and the suggestion degrades
  with a warning, never an exception.
- `_canonicalize_theta0` applies the exact J Σ Jᵀ fold; marginal errors
  come from the transformed covariance.
- `AngularFourier2` is registered, angle-scoped, joint-fit eligible, and
  recovers synthetic tilt parameters.
- The three suggestion bridges pass the closed-form oracles (cos 2θ
  antinode/node geometry, vector-sum doubling, misalignment peak,
  assignment divergence zero-at-crossing) and `harness validate` is
  green with no Qt imports in core.

### DataGroup / FitSeries Unification (Option C)

Status: implemented on branch `feat/datagroup-fitseries-unification`
(Phases 1–5 — core model, browser, fit flows, carry-forward, docs — complete);
awaiting Phase 6 full validation and PR review

Make DataGroups the canonical vehicle for batch fits: a run-membered
`FitSeries` structurally belongs to a group (`group_id`), its membership is
live-derived (group members minus per-series exclusions) with staleness
marking, and results stay a snapshot. Runs may belong to multiple groups
(browser shows duplicated, marker-glyph rows; selection dedupes). Ad-hoc
batch fits auto-create reusable `kind="auto"` groups (red-family palette;
user groups stay blue; rename promotes auto → user). "Share with Group" is
retired in favour of refresh-unless-fitted carry-forward (runs with a
recorded fit are never auto-overwritten; everything else follows the latest
fitted function). `ProjectModel.data_groups` becomes the single source of
truth (the GUI mirror dataclass and save/load sync are deleted), and
`_data_group_id_for_runs` provenance guessing goes away.

Full decision log (D1–D9), code map, phased plan with per-phase agent
assignments and review checklists, and migration details live in
[studies/datagroup-fitseries-unification.md](studies/datagroup-fitseries-unification.md).

Acceptance criteria:

- Schema v15: series carry `group_id` / `excluded_run_numbers` /
  `last_fitted_members`; v14 projects (including pre-Phase-7 and
  duplicate-era saves) migrate tolerantly; group-less legacy series load as
  frozen (`group_id=None`) — no synthesized groups on migration.
- Every newly recorded run-membered series has a group (launched-from or
  auto-created/reused); re-running a group analysis replaces in place;
  detector-group series (`member_kind="groups"`) behaviour is unchanged.
- A run in two groups renders as marked duplicate rows and reaches
  plot/fit/co-add exactly once when selected via both.
- A run with a recorded fit result (single or batch member) is never
  auto-overwritten by carry-forward, across save/load; unfit runs refresh
  from the latest fitted function; "Share with Group" is gone.
- Docs + screenshot scenarios updated (auto vs user group colours, marked
  rows, group-bound Batch tab); changelog calls out the Share with Group
  removal; `harness validate` and `gui-smoke` green.

### Fit Function Builder Redesign (Option C)

Status: implemented on branch `feat/fit-builder-redesign`; awaiting live GUI
review before the PR opens

Replace the calculator-style expression dialogs with a two-panel builder:
searchable component library (left) + structured model-row list (right), with
fraction groups as visual containers and an "Edit as text" escape hatch.
Both `FitFunctionBuilderDialog` and `ParameterModelBuilderDialog` build on one
shared base in `gui/widgets/function_builder/`.

Core rework included: fraction groups expose n−1 free parameters named
`f_<Component>` (last term = remainder `max(0, 1−Σ)` computed in
`CompositeModel`); legacy `fraction_i` parameter values are migrated on
project load. Deep component search (name/alias/category/param/description,
ranked) lives GUI-free in `core/fitting/component_search.py`.

Acceptance criteria:

- Both builder dialogs use the shared library + row-list base; the old
  free-text-primary editor and text-selection Fractions gesture are gone.
- Fraction groups survive structural edits; the displayed model is never
  ambiguous (no hidden parentheses).
- Pre-rework `.asymp` projects load, display, and refit with migrated
  fraction values.
- `harness validate` passes; docs updated with GUI walkthrough screenshots.

### Harness Adoption

Status: in progress

- Maintain `AGENTS.md` as the short navigation map.
- Keep `docs/HARNESS.md`, `docs/QUALITY.md`, and this file indexed from
  `docs/INDEX.md`.
- Run `python tools/harness.py structural` for fast boundary validation.
- Run `python tools/harness.py validate` before larger handoffs when local
  dependencies are available.

Acceptance criteria:

- Agents can discover architecture, validation commands, and current quality
  risks from repo-local files.
- CI runs the same harness command used locally for structural, lint, and test
  validation.
- New repeated conventions are promoted into checks or docs.

## Deferred

### FFT Zero-Padding as a Display-Only Concern

Today zero padding is a transform parameter: the padded (sinc-interpolated,
correlated) samples ARE the stored spectrum, threaded through recipes,
staleness comparison, caching, fits, and moments. The statistically clean
architecture stores the spectrum at padding 1 (independent samples — what
fits, moments, and exports consume) and treats the smooth curve as a pure
rendering: the plot draws the sinc envelope (computed from the complex
spectrum at display time) with the independent samples as points on top.
This dissolves the correlated-samples problem instead of correcting for it
(the interim effective-sample-size correction — `error_oversampling`,
WiMDA's `dof := n div zpad - nv` made consistent — ships with the padding
work and covers the statistics until then). Non-trivial because the display
curve must be derived from the COMPLEX spectrum (interpolating a magnitude
display is wrong near its kinks), so the envelope has to come out of the
compute path, not the plot panel; recipes and the project schema carry the
padding field today and would need a migration story.

### GUI Journey Harness

Build a repeatable smoke workflow for the most important GUI paths:

- launch the app in offscreen mode
- load a small synthetic or fixture-backed dataset
- open grouping, fitting, Fourier, and project-save paths
- capture failures with enough context for an agent to reproduce locally

### Documentation Freshness Checks

Consider a structural check that verifies `docs/INDEX.md` references all
top-level developer docs and that public API docs build without warnings.

### Ruff Baseline Cleanup

Status: complete

`python tools/harness.py lint` now checks `src`, `tests`, and `tools`, and
`lint-all` remains as an alias for that full baseline.

### Loader Fixture Catalog

Create a small fixture index documenting which file-format assumptions are
covered by tests and which rely on synthetic data.
