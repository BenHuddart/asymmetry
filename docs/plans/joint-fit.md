# Joint fit: several series, different models, shared parameters

Status: implemented 2026-09-18 on `feat/joint-fit` (Phases 1–5 as agent commits with lead fix-ups; Phase 6 gate green: validate, gui-smoke, docs, structural, lint), PR open.

## Problem

A sample measured across a phase transition needs two fit functions: an
oscillating model in the magnetically ordered phase and a relaxing model in
the paramagnetic phase. Some parameters are the same physical quantity in
both (the initial asymmetry, the background, alpha, a phase), and the fit
should say so. Today the only way to share a parameter is the Batch tab's
Global role, which shares it across the runs of **one** series with **one**
model. Two series with different models cannot share anything: the user fits
them separately and hopes the shared quantity comes out the same.

The pieces are almost all in place. `FitEngine.global_fit` already solves one
coupled least-squares problem over several datasets with a global/local
partition and a third, subset-shared scope (`local_param_groups`). What it
cannot do is give each dataset its own model function, or share a column
between two differently named parameters. Series already carry model,
members, fit range and parameter roles, so "which data belong to which
function" is already answered by the series the user has made. The Global
Parameter Fit window already shows the shape of a menu-launched, undocked,
persisted, stale-aware analysis with its own registry submenu.

## Settled design (decision log)

Decisions taken with Ben on 2026-09-18 unless marked *lead*.

**D1 — Name.** The feature is a **joint fit**. A parameter held equal across
series is **shared**. The three scopes read: Local (per run), Global (per
series), Shared (across series). The words "linked" (WiMDA equality link
groups, the Links chip, the ⇄ badge) and "global" (per-series role; the
Global Parameter Fit feature) keep their existing meanings and are not
reused for this feature.

**D2 — A joint fit composes existing series.** The user never assigns runs
to functions inside the joint fit. Each member is a recorded `FitSeries`
(model, members via its data group and exclusions, fit range, parameter
table, seeding, co-add) made in the Batch tab as today. The joint fit adds
only the shared-parameter table and its own label. Per-series recipes are
edited in the Batch tab, never in the joint-fit window (this keeps D1 of the
series workflow: the Batch tab edits one series).

**D3 — Members never overlap and share a representation type.** A run may
belong to exactly one member series of a joint fit; the same data twice in
one cost is wrong. v1 admits only `member_kind == "runs"` series of the same
representation type as the active one, with a canonical model. The window
prevents overlap by construction (a series whose members intersect a ticked
series is disabled with the reason on its tooltip); the core raises on it.

**D4 — Only series-Global parameters can be shared.** A shared parameter
maps, in each contributing series, to a parameter whose role in that series
is Global. Local and Fixed rows are not offered. A shared parameter must be
contributed by at least two series. Equality only in v1: no affine or
expression relations across series.

**D5 — Sharing lives on the joint fit, not on the series' roles.**
`PARAM_ROLES` stays `("global", "local", "fixed")`; the Batch tab and every
series consumer are untouched. The joint fit record holds the shared table;
each member series additionally records `joint_fit_id` and
`shared_params: {local name: shared name}` for readers that render one series
at a time (chips, cards, results windows).

**D6 — Seed and bounds of a shared parameter.** Seeded from the row of the
first contributing series (in the joint fit's member order); editable in the
shared table (value, min, max). Every other seed, bound, fixed value,
file-pinned value and fit range comes from the member series' recipe,
resolved exactly as the Batch tab resolves it when running that series
(recipe values only: the Batch tab's inherited single-fit seeds and
Initial-values dialog precedence do not apply to a joint run).

**D7 — Autodetection proposes the default shared table.** A core function
proposes rows from the member models and roles:

- *Tier one, ticked by default*: the same full parameter name, the same unit,
  occurring exactly once in each model, with Global role in each series.
- *Tier two, offered unticked*: the same base name with a different component
  index, or the same component type where each model has exactly one instance
  of it, again unit-equal and Global in each series.

Anything ambiguous is left to the user. The table is re-proposed when the
member set changes; user-added and user-edited rows survive a re-proposal.
The docs must state that "initial asymmetry" is shareable only when the model
exposes it as one parameter (a fraction-group model with an overall scale);
a sum of component amplitudes is a constraint, not a shared parameter, and is
out of scope.

**D8 — Results are ordinary series results.** A joint run writes each member
series' `results_by_run` in place under its existing `batch_id` (no new
series: the recipes did not change, only the constraint), stamps
`joint_fit_id` and `shared_params` on each member, and records or updates the
`JointFit` record (shared values, uncertainties, the shared covariance block,
combined χ², dof and per-series χ²ᵣ). Each per-run `FitResult` carries the
shared value under the series' own parameter name, so every existing
consumer keeps working.

**D9 — Staleness is runtime, never persisted.** A joint fit is stale when any
member series is missing, when any member's `joint_fit_id` no longer points
at it (the series was re-run on its own, per D3 of the series workflow, and
its results no longer honour the constraint), or when any member is itself
stale (`FitSeries.is_stale`). Re-running a member series alone is allowed
and detaches it; the window shows the stale reason and a Refit button. The
Batch tab is never blocked.

**D10 — Delete.** Deleting a joint fit clears `joint_fit_id`/`shared_params`
on its members and leaves their results in place. Deleting a member series
removes it from the joint fit; a joint fit left with fewer than two members
is deleted with it. Both cascade in `ProjectModel`, not in the GUI.

**D11 — Menu-launched, undocked window.** Analysis → "New joint fit…" opens
`JointFitWindow`; Analysis → "Joint fits" is a submenu listing recorded joint
fits (mirroring "Global parameter fits"), opening one loads it into the
window. No new dock or tab in the main GUI. The window has one instance,
owned by `MainWindow`, and runs the fit on `TaskRunner` with the standard
cancel path. (*Added 2026-09-18, after reviewing the feature on a real
project: the default label built from each member's full fallback name
(`"<model> · <fit-range>[ · <group>]"`) was unreadable for models with long
names — `naming.default_joint_fit_label` now composes each member's *short*
name (`naming.joint_member_name`: its own label, else its data group's
name, else its model label), falling back to the member's full fallback
name only when two members' short names collide. The full member list moves
to the chip-rail joint section header's tooltip and the window's label
field tooltip, one line per member.*)

**D12 — Engine (*lead*).** `_build_coupled_global_problem` is generalised to
several *blocks* (one per series: datasets, model function, global/local
names, initial parameter sets, fit range, local group key) plus a shared
column map. Column order is shared columns, then each block's free non-shared
globals in block order, then locals as today. `global_fit` becomes the
one-block, no-shared caller and must stay byte-for-byte identical in fitted
values, χ² and result packing (the existing engine tests pin this). v1
supports the `"joint"` (Minuit) and `"least_squares"` strategies; the
`"profiled"` strategy raises `NotImplementedError` for joint fits. Affine
ties raise via `_reject_affine_ties` as on every coupled path; equality link
groups within a series behave exactly as in `global_fit`.

**D13 — Schema v22 (*lead*).** Top-level `joint_fits` list; series entries
gain optional `joint_fit_id` and `shared_params`. Migration v21→v22 sets
`joint_fits` to `[]`; series fields default when absent.

**D14 — Deferred.** Affine or expression relations across series; sharing a
per-run Local across series; `member_kind == "groups"` series and
count-domain costs; the `"profiled"` strategy; creating a joint fit from the
Global Fit Wizard's phase partition (`apply_wizard_phase_assessments` is the
future entry point); routing `fit_asymmetry_series` through the new builder;
exposing `fit_joint` through the agent CLI façade.

## Key code map (verified 2026-09-18; re-check anchors before editing)

| Concern | Location |
| --- | --- |
| Coupled problem, model wrapper, sparsity | `core/fitting/engine.py` `_CoupledGlobalProblem` (~710), `_build_coupled_global_problem` (~743; per-dataset evaluation ~840–875), `_coupled_jacobian_sparsity` (~910) |
| `global_fit` (validation ~1720–1790, `_local_group_key` ~1764, block-separable fast path ~1815, result packing ~2100), `_global_fit_least_squares` (~2210), `_global_fit_profiled` (~2439), `_reject_affine_ties` (~330) | `core/fitting/engine.py` |
| Asymmetry-domain wrapper and result bundle (pattern for `JointFitResult`) | `core/fitting/asymmetry_global.py` (`GlobalFitResult` ~47, `fit_global` ~174) |
| Chained block-separable series (untouched; its `FitLaunch` use is the model for the window's launch context) | `core/fitting/series.py` |
| Parameter identity across models | `core/fitting/parameter_carry.py` (`ParameterIdentity`, `ComponentParameter`, `GroupAmplitude`, `FractionWeight` ~38–104); `CompositeModel.parameter_identities` `core/fitting/composite.py` (~2148); `CompositeModel.function` (~2642) |
| Parameter names, units, base/index split | `core/fitting/parameters.py` (`ParamInfo` ~25, `split_parameter_name` ~77, `get_param_info` ~688, `Parameter` ~821, `ParameterSet` ~838) |
| `FitSeries` (fields ~106–200, `is_global` ~226, `effective_members` ~277, `is_stale` ~299, `recipe_identity` ~384, `to_dict`/`from_dict` ~437/462), `PARAM_ROLES` (~37) | `core/representation/series.py` |
| `ProjectModel` (`add_batch` ~87, `active_series` ~63–121, `series_for_group` ~265, `remove_batch` ~465, `to_dict`/`from_dict` ~548–577) | `core/representation/project_model.py` |
| Persisted study record to mirror (`to_dict`/tolerant `from_dict`, digest-based staleness) | `core/representation/global_fit_study.py` |
| Schema (`CURRENT_SCHEMA_VERSION` = 21 at ~243, version docstring ~97–130, `_migrate_v20_to_v21` ~1344, `global_fit_studies` default ~764) | `core/project/schema.py` |
| Analysis menu (~1195–1246): Global Parameter Fit action, "New global parameter fit…", studies submenu; `_rebuild_global_fit_studies_menu` (~14389), `_restore_global_fit_studies` (~14422) | `gui/mainwindow.py` |
| Window ownership pattern: `_global_parameter_fit_window` (~969), `_on_global_parameter_fit` / `_on_new_global_parameter_fit` (~10065–10090) | `gui/mainwindow.py` |
| Recording: `_record_fit_series` (~11826), `_open_series_for` (~11854), `_record_global_fit_batch` (~11961), `_on_global_fit_completed` (~13442), `_runs_by_number_for` (~11815) | `gui/mainwindow.py` |
| Series delete / open in Batch tab: `_on_series_delete_requested` (~11375), `_open_series_in_batch_tab` (~11527), `_on_series_open_requested` (~11553) | `gui/mainwindow.py` |
| Trend panel reload `_refresh_trend_panel` (~10907); fit dataset crop `_get_fit_dataset` (~15257); project write/collect/restore (~15719, ~16019, ~16435) | `gui/mainwindow.py` |
| Batch tab launch: `_run_global_fit` (~3135; recipe → engine inputs ~3160–3230), `_parse_parameter_configuration` (~2896), `_effective_initial_values_by_run` (~2986), `FitLaunch` (~385), `_on_fit_finished` (~4805), signals (~452–485) | `gui/panels/fit/global_tab.py` |
| File-pinned parameter values `_get_file_value_for_parameter` | `gui/panels/fit/tab_base.py` (~506) |
| Undocked results window to copy (layout, `TaskRunner`, `set_stale`, Refit, `get_state`/`restore_state`, `closeEvent`) | `gui/windows/global_parameter_fit_window.py` (`GlobalParameterFitWindow` ~151) |
| Shared widgets to reuse | `gui/widgets/fit_run_controls.py` (`FitRunControls`), `gui/widgets/panel_section.py`, `gui/styles/widgets.py` (`apply_param_table_style`, `make_section_header`), `gui/tasks.py` (`TaskRunner`) |
| Parameters panel: `_GroupFitData` (~522), chip-rail section layout (~711, `_rebuild_group_buttons` ~1945), `load_representation_series` (~1613), cards `_rebuild_cards` (~3966), series drawing `_draw_single_series`/`_draw_multi_series`/`_plot_series_param` (~6173–6382) | `gui/panels/fit_parameters_panel.py` |
| Existing tests to extend or mirror | `tests/core/test_asymmetry_global_fit.py`, `test_engine_least_squares_fit.py`, `test_engine_profiled_fit.py`, `test_parameter_carry.py`, `test_series_model.py`, `test_global_fit_study.py`; `tests/project/test_series_recipe_migration.py`, `test_project_save_atomic.py`; `tests/gui/test_global_parameter_fit_window.py`, `test_global_fit_studies.py`, `test_series_workflow.py`, `test_fit_parameters_panel.py` |
| Docs pages | `docs/reference/asymmetry_domain_global_fit.rst`, `fitting.rst` § "Global fitting", `parameter_trending.rst`, `gui_usage.rst`, `project_files.rst`, `docs/ARCHITECTURE.md`; screenshot registry `docs/screenshots/capture.py`, scenarios `docs/screenshots/scenarios/` (e.g. `global_fit_lfkt.py`) |

## Core API (Phase 1 contract)

New module `core/fitting/joint.py`, Qt-free:

```python
@dataclass(frozen=True)
class JointSeriesProblem:
    key: Hashable                      # caller's series id
    datasets: list[MuonDataset]        # unique run numbers within the problem
    model_fn: Callable[..., NDArray]
    global_params: list[str]
    local_params: list[str]
    initial_params: dict[int, ParameterSet]   # run number -> set
    t_min: float | None = None
    t_max: float | None = None
    local_param_groups: dict[str, dict[int, Hashable]] | None = None

@dataclass(frozen=True)
class SharedParameter:
    name: str                          # the shared column's name
    members: dict[Hashable, str]       # series key -> that series' parameter name
    value: float
    min: float = -inf
    max: float = inf

@dataclass
class JointFitResult:
    success: bool
    shared_parameters: ParameterSet
    shared_uncertainties: dict[str, float]
    shared_covariance: NDArray | None
    series_results: dict[Hashable, dict[int, FitResult]]
    series_global_parameters: dict[Hashable, ParameterSet]   # incl. shared values under local names
    series_reduced_chi_squared: dict[Hashable, float]
    chi_squared: float
    dof: int
    reduced_chi_squared: float
    message: str = ""

def fit_joint(problems, shared, *, strategy="least_squares", method="migrad",
              max_calls=10000, minos=False, fit_engine=None,
              cancel_callback=None) -> JointFitResult: ...

def suggest_shared_parameters(models: Sequence[CompositeModel],
                              roles: Sequence[Mapping[str, str]]) -> list[SharedSuggestion]: ...
```

`fit_joint` raises (never guards) on: fewer than two problems; a run number
present in two problems; a shared member naming a parameter that is not in
that problem's `global_params`, or that is fixed in every one of that
problem's initial sets; a shared parameter with fewer than two members; an
affine tie anywhere; `strategy="profiled"`. `SharedSuggestion` carries
`name`, `members`, `tier` (`"exact"` / `"candidate"`) and a one-line
`rationale`.

## Execution plan

Branch `feat/joint-fit` off `main`, one PR. Phases are strictly ordered; each
is one or more commits by one subagent working **in the main checkout on
this branch** (no worktrees: sequential phases share one tree, and worktree
agents import `asymmetry` from the wrong checkout unless `PYTHONPATH` is
set). Subagents cannot be messaged after launch, so each brief is complete
up front: this decision log, the phase's scope and checklist, the lean rules,
and the test commands.

**Lean rules for every agent brief.** Prevent bad states by construction and
raise on the rest; no new `hasattr`/`isinstance` guards on internal objects
and no fallbacks for states the design prevents — stop with a clear note
rather than guard. No Qt or matplotlib in `core`. Reuse `gui/widgets/`,
`gui/styles/` and `gui/tasks.py` foundations (`FitRunControls`,
`PanelSection`, `apply_param_table_style`, `TaskRunner`); no new `QThread`,
no `processEvents`, no worker signal connected to a bare lambda. No new
project-file payloads beyond D13. Tests beside the behaviour, focused; run
`python tools/harness.py test -- <files>` while iterating and `--tier fast`
before handing back; never `validate` mid-phase. Quote UI strings verbatim
when they reach the docs.

**Lead review gate (after every phase).** Diff read against the phase
checklist and the decision log; focused tests plus `--tier fast` (plus the
phase's named GUI files) green; `harness structural` and `lint` green.
Fix-ups land inside the phase. `validate` and `gui-smoke` run once, in
Phase 6.

### Phase 1 — Engine generalisation and `fit_joint` (agent: Opus)

Scope: `core/fitting/engine.py`, new `core/fitting/joint.py`, tests in
`tests/core/`. No GUI, no project model.

- Generalise `_build_coupled_global_problem` to blocks plus a shared column
  map (D12). Each dataset evaluates its block's `model_fn`; `dataset_columns`
  and `_coupled_jacobian_sparsity` include the shared columns; a dataset
  whose block pins a shared member (fixed) uses its own pinned value, as
  `global_fit` does for a pinned global today. Keep `_CoupledGlobalProblem`
  the one object both solvers consume.
- `global_fit` calls the builder with one block and no shared map. Its
  column order, fitted values, χ², dof and packed results must not change:
  `tests/core/test_asymmetry_global_fit.py`,
  `test_engine_least_squares_fit.py`, `test_engine_profiled_fit.py` and
  `test_fitting_engine.py` pass unmodified.
- Add `FitEngine.joint_fit(...)` (or a module-level solver in `joint.py`
  that reuses `_global_fit_least_squares` and the joint Minuit path through
  the shared problem object; the lead has no preference, pick the smaller
  diff). Result packing yields per-series `FitResult`s with the shared value
  and uncertainty under the series' own name, per-series global
  `ParameterSet`s, the shared block of the covariance, and combined
  χ²/dof.
- `fit_joint` and `suggest_shared_parameters` per the contract above.
  Suggestion rules per D7, built on `CompositeModel.parameter_identities`,
  `split_parameter_name`, `get_param_info(...).unit` and component names.

Checklist (tests): two series with different models sharing one parameter
recover a known shared value from synthetic data, and the shared σ is the
same in both series' results; the same problem with the shared map empty
reproduces two independent `global_fit` answers; overlap, non-Global
member, single-member share, tie and `profiled` each raise with a clear
message; a shared member fixed in one series leaves that series at its
pinned value while the others fit; `least_squares` and `joint` strategies
agree on values to solver tolerance; suggestions: `A_bg`/`alpha` land in
tier one, `A_1` vs `A_2` in tier two, a name present twice in one model is
not suggested, unit mismatch is not suggested; `global_fit` byte-for-byte
regression on an existing fixture.

### Phase 2 — Project model and schema v22 (agent: Sonnet)

Scope: new `core/representation/joint_fit.py`, `core/representation/series.py`,
`core/representation/project_model.py`, `core/project/schema.py`, tests in
`tests/core/` and `tests/project/`. No GUI.

- `JointFit` dataclass: `joint_id`, `label`, `rep_type`, `member_batch_ids`
  (ordered), `shared: list[dict]` (name, members `{batch_id: param}`,
  value, min, max), `result: dict | None` (shared values, uncertainties,
  covariance rows, χ², dof, reduced χ², per-series reduced χ², timestamp),
  `to_dict`/`from_dict` (tolerant on the way in like `GlobalFitStudy`:
  skip a malformed entry, never fail the project). `is_stale(model)` and
  `stale_reason(model)` per D9, computed, never stored.
- `FitSeries.joint_fit_id: str | None` and `shared_params: dict[str, str]`
  with serialisation; `recipe_identity()` ignores both (they are not part
  of what the series *is*).
- `ProjectModel.joint_fits: dict[str, JointFit]`; `add_joint_fit`,
  `remove_joint_fit` (clears member stamps, D10), `remove_batch` cascade
  (drop the member; delete the joint fit under two members, D10),
  `joint_fit_for_series(batch_id)`; `to_dict`/`from_dict` round-trip.
- Schema v22 (D13): version docstring entry, `_SUPPORTED_VERSIONS`,
  `_migrate_v21_to_v22`, `save_project`/`load_project` carry `joint_fits`.
  Default label `naming.default_joint_fit_label(series_labels)` →
  `"Joint: <A> + <B>"`.

Checklist (tests): round-trip of a project with one joint fit and two
stamped series; v21 file loads with `joint_fits == []` and no series stamps;
deleting a member series of a three-member joint fit keeps it with two, a
second delete removes it and clears the survivor's stamp; deleting the joint
fit leaves member results intact; `is_stale` is true after a member's
`joint_fit_id` is cleared and false on a fresh record.

### Phase 3 — Window, menu, launch and recording (agent: Opus)

Scope: new `gui/windows/joint_fit_window.py`, `gui/mainwindow.py` (menu,
window ownership, recording, delete cascade wiring, project collect/restore),
the recipe-to-engine extraction in `gui/panels/fit/global_tab.py`, tests in
`tests/gui/`.

- Extract the recipe → engine-inputs step of `_run_global_fit`
  (~3160–3230: per-run `ParameterSet`s with bounds, fixed and file-pinned
  values, global/local lists, fit range) into one function that takes a
  series recipe, its model and its datasets. `_run_global_fit` calls it; the
  joint window calls it per member (D6). No second copy.
- `JointFitWindow(QMainWindow)`, modelled on `GlobalParameterFitWindow`:
  - *Series* section: one checkable row per eligible series of the active
    representation type (label, model text, member count, status);
    ineligible rows disabled with the reason on the tooltip (overlap with a
    ticked series, group-membered, computed, other rep type) (D3).
  - *Shared parameters* section: table with columns Shared · Value · Min ·
    Max · one column per ticked series (combo of that series' Global
    parameters plus "—"); "Suggest" re-runs `suggest_shared_parameters`
    (tier one ticked, tier two unticked); "Add shared parameter…" and remove;
    user rows survive re-suggestion (D7). `apply_param_table_style`.
  - Footer: `FitRunControls` (Run joint fit / Stop), a results card
    (combined χ²ᵣ, shared values ± σ, per-series χ²ᵣ each with "Open in
    Batch tab"), stale banner with reason and Refit (D9). Label field.
  - Runs `fit_joint` on `TaskRunner` with the cancel callback; datasets are
    cropped via `_get_fit_dataset` with the series' own fit range; a
    `FitLaunch`-style frozen context is what the completion reads.
    `closeEvent` stops the runner. One window instance owned by
    `MainWindow` with a strong reference.
- `MainWindow`: Analysis → "New joint fit…" and the "Joint fits" submenu
  (rebuilt from `ProjectModel.joint_fits` on change, D11); completion
  handler writes each member series' results in place, stamps the members,
  records the `JointFit`, refreshes the trend panel and overlays (D8);
  `_record_fit_series` clears a series' stamps when a non-joint run records
  into it (D9); series delete goes through the `ProjectModel` cascade (D10);
  `collect_project_state`/restore carry `joint_fits` and the window's
  `get_state`/`restore_state` (open joint id only).

Checklist (tests, offscreen): ticking series A disables an overlapping
series C with a tooltip; the suggest button proposes `A_bg` ticked for two
models that both carry it; running records results on both series under
their existing ids, stamps them, and lists the joint fit in the submenu;
re-running series A from the Batch tab marks the joint fit stale with the
detach reason; project save and reload reopens the joint fit from the
submenu with its shared table; Stop cancels a running fit and leaves both
series' previous results in place.

### Phase 4 — Parameters panel: shared badge and joint section (agent: Sonnet)

Scope: `gui/panels/fit_parameters_panel.py`, the `load_representation_series`
call site in `mainwindow.py` `_refresh_trend_panel`, tests in
`tests/gui/test_fit_parameters_panel.py`.

- Chip rail: series belonging to one joint fit are grouped under a section
  titled with the joint fit's label, using the existing sectioned layout
  (~711, `_rebuild_group_buttons`), ahead of the data-group sections.
- A shared parameter's card and legend entry carry a "Shared" badge (a
  distinct glyph from the ⇄ link badge and the ƒ tie badge; tooltip names
  the joint fit and the other series). A series-Global parameter is not
  plotted today: the trend view hides it behind the "held constant" footer
  hint, and that stays so on a jointed series. A **shared** parameter is
  the one addition: it is drawn as a single flat line across the union of
  the member series' x extents, listed once in the legend under the joint
  fit's label (*corrected 2026-09-18 in Phase 4 review; the first draft
  assumed Global parameters already drew flat lines*).
  (*Withdrawn 2026-09-18, Decision A, after reviewing the feature on a real
  project: the exception above was itself a wrong premise. The panel's rule
  is that a series-Global parameter — shared or not — has no chip and no
  card; a shared parameter stays a Global parameter in every contributing
  series, so it gets no badge, no card, and no flat line either. The
  "held constant" footer hint is instead extended: a shared parameter's
  entry in it gains a suffix naming the joint fit, `A_1 — shared across
  joint fit "<label>"`. The chip-rail section grouping stays as built.*)
- `_GroupFitData` gains `joint_fit_id`/`shared_params` as derived display
  state, supplied by `load_representation_series` like `phase`, not
  serialised.

Checklist (tests): two stamped series render one section with two chips; a
shared parameter card shows the badge and one flat line spanning both
series; an unstamped series renders exactly as before (existing panel tests
unchanged). (*Superseded 2026-09-18 by the Decision A withdrawal above: the
badge/flat-line checks were replaced by "held constant" hint tests naming
the shared parameter's joint fit; the section-grouping check stands.*)

### Phase 5 — Docs, screenshots, changelog (agent: Sonnet)

- New page `docs/reference/joint_fit.rst` (registered in `reference/index.rst`
  beside `asymmetry_domain_global_fit`): when to use a joint fit versus one
  series' Global role, the three scopes, the window walk-through with UI
  strings quoted verbatim, autodetection tiers (D7) including the
  initial-asymmetry caveat, staleness and detach (D9), delete (D10), the
  core API with a `fit_joint` example, and the v1 limits (D14).
- Cross-references from `fitting.rst` § "Global fitting",
  `parameter_trending.rst` (shared badge, joint section),
  `gui_usage.rst` (Analysis menu), `project_files.rst` (v22),
  `docs/ARCHITECTURE.md` (module map: `core/fitting/joint.py`,
  `core/representation/joint_fit.py`, `gui/windows/joint_fit_window.py`).
- Screenshot scenario `joint_fit_window.py` (window with two series ticked
  and a suggested shared table, deterministic synthetic data) registered in
  `docs/screenshots/capture.py`; within the size budget; `harness
  structural` screenshot-drift check green.
- `CHANGELOG.md` `[Unreleased]`: Added (joint fit: window, menu, shared
  parameters, autodetection, persistence), Changed (project schema v22).

### Phase 6 — Validation and PR (lead)

`python tools/harness.py structural`, `lint`, `validate`, `gui-smoke`,
`docs`; open the PR with the decision log summary and the migration note;
propose the release bump afterwards per `RELEASING.md`.

## Risks / watch items

- **Engine regression.** Phase 1 rewrites the column layout of every
  coupled fit in the app. The existing engine tests are the gate; the lead
  additionally diffs a saved `global_fit` result on a real project before
  and after.
- **`mainwindow.py` blast radius.** Phase 3 touches recording, delete and
  project restore. The brief lists the exact functions; the review gate
  greps that no second recipe-to-engine copy exists and that stamps are
  written only in the joint completion handler and cleared only in
  `_record_fit_series`.
- **Percent scale.** Every member series' datasets must be on the same
  asymmetry scale (the engine docstring's fraction-vs-percent trap now spans
  series). `_get_fit_dataset` already yields the representation's scale;
  the window takes only same-rep-type series (D3), which closes this.
- **Wizard flows.** The Global Fit Wizard's phase apply path
  (`tests/gui/test_global_fit_wizard_phases_apply.py`) records series
  through `_record_fit_series`; the stamp-clearing change in Phase 3 must
  keep it green.
- **Grouped and count-domain surfaces** are excluded in v1 (D14); the
  window's eligibility rule is what keeps them out, and the docs say so.
