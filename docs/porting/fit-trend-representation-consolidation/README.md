# Fits & Parameter Trending Across Representations — Behaviour Review

**Status:** review / study pass (no code changes)
**Branch / worktree:** `feat/fit-trend-representation-review`
**Goal of the follow-on work:** consolidate fit + parameter-trending behaviour so
it is *consistent and intuitive across every representation* (time F-B asymmetry,
grouped/raw-count time domain, FFT, MaxEnt, integral/field scans).

This document summarises **how the current system actually behaves** and gives a
**navigation index** so the implementing agent can jump straight to the relevant
code. It answers four questions posed for the review:

1. How do **DataGroups** and **FitSeries** interact?
2. How do **single fits** and **batch fits** interact with each other?
3. How are **fit functions** shared between datasets?
4. (implicit, and the real target) How does all of the above behave **across
   representations**, and where is it inconsistent?

> Line numbers are accurate at the time of writing (main @ `25bab07`) but treat
> them as *approximate anchors* — prefer the named class/function. Anchors marked
> ✓ were read directly during this review; the rest come from focused agent
> sweeps and should be spot-checked before you rely on exact offsets.

---

## 0. The one thing to understand first: the Representation spine

Everything about fits and trends hangs off the **domain-representation model**
(`asymmetry.core.representation`, described in
[`docs/ARCHITECTURE.md` §3.5](../../ARCHITECTURE.md)). Internalise this and the
rest falls out.

- A **`Representation`** is a *recipe-driven view* of one `Run`
  (`core/representation/base.py` ✓). There are five types
  (`RepresentationType`): `TIME_FB_ASYMMETRY`, `TIME_GROUPS`,
  `TIME_MAXENT_RECON`, `FREQ_FFT`, `FREQ_MAXENT`. Each maps to a domain
  (`time`/`frequency`) via `DOMAIN_OF`.
- Each dataset owns **up to one representation per type**
  (`DatasetRepresentations`, `core/representation/container.py` ✓), keyed by run
  number inside `ProjectModel.datasets`.
- Each representation carries exactly **one primary `FitSlot`** (its "the most
  recent fit for this `(dataset, representation)` pair") plus optional
  **per-projection** fit slots for vector groupings (`projection_fits`,
  `P_x/P_y/P_z`, TF labels). `FitSlot` fields: `model`, `parameters`, `result`,
  `provenance` (`none`/`single`/`batch`/`global`/`wizard`), `batch_id`,
  `diverged`, `include_in_trend`, `ui_state` (base.py ✓, lines 54–128).
- Each representation also carries **one `trend_state`** dict (formalised by
  `TrendState`, `core/representation/trend_state.py` ✓): `x_key`,
  `selected_quantities`, `derived_params`, `model_fits`, `axes_state`. **Trend
  identity is `(representation, quantity)`** — a `lambda` trended in the time
  representation cannot collide with one in the frequency representation.
- **`ProjectModel`** (`core/representation/project_model.py` ✓) owns the
  per-run representation map **and** the top-level `batches: dict[str,
  FitSeries]`. It is the single in-memory authority for divergence, trend
  inclusion, batch de-duplication, and recompute-on-load.

**The pivotal mapping** (`MainWindow._active_representation_type`,
`gui/mainwindow.py:8262` ✓): the *active plot-workspace view* decides which
representation a fit/trend targets:

| Active view (`plot_workspace.active_view()`) | Representation type |
|---|---|
| `fb_asymmetry` | `TIME_FB_ASYMMETRY` |
| `integral_scan` | `TIME_FB_ASYMMETRY` *(shares the F-B slot; the scan itself is a `FitSeries`)* |
| `groups` | `TIME_GROUPS` |
| `raw_counts` | `TIME_GROUPS` *(uncorrected display of the same grouped slot)* |
| `frequency` | `FREQ_FFT` |
| `maxent` | `FREQ_MAXENT` |

So **"a fit"** is always scoped to a `(run, representation, [projection])`
triple, and switching views silently switches which fit slot and which trend
state you are looking at. This is the root cause of most cross-representation
"why did my fit/trend disappear?" confusion and is the primary consolidation
surface.

---

## 1. Terminology: three different things all called "group"

This is the single biggest source of confusion in the codebase and must be
fixed in the report's language before anything else. **Three unrelated concepts
share the word "group":**

| Name in code | What it is | Where it lives | Touches fitting? |
|---|---|---|---|
| **DataGroup** | A *browser-only* visual folder of **runs** (e.g. "T = 150 K"), collapsible, user-named. | `gui/panels/data_browser.py:136` (`@dataclass DataGroup`) | **No** — invisible to the engine. Only used to *name* a batch and pick launch members. |
| **Detector group** | The forward/backward/etc. detector grouping carried on a `Run` (`run.grouping`). Drives F-B reduction and the grouped time-domain fit. | `core/transform/grouping.py`, `run.grouping` | Yes — it *is* the physics reduction. |
| **`FitSeries(member_kind="groups")`** | A *batch fit* whose members are `(run, detector-group)` pairs, keyed by synthetic negative ints `-(run*1000 + group_idx)`. | `core/representation/series.py` ✓ | Yes — each member is a fit domain. |

**Key relationship:** A **DataGroup** never becomes a **FitSeries**. When a batch
is launched from a DataGroup header, the only thing that crosses over is the
*name* (`MainWindow._data_group_name_for_runs` → series label,
`mainwindow.py:9687` ✓/9623) and the *set of member runs*. The DataGroup is not
persisted into the series and is not consulted during the fit. See §2.

**DataGroup structure** (`data_browser.py:136` ✓ reported):
```python
@dataclass
class DataGroup:
    group_id: str          # UUID, persisted
    name: str              # "T = 150 K" | "Group 1" (auto or user-edited)
    member_run_numbers: list[int]
    collapsed: bool = False
```
Browser maintains `_groups: dict[str, DataGroup]`, `_run_to_group: dict[int,
str]`, `_display_order: list[int|str]`. CRUD at `data_browser.py:891–1064`.
Persisted under the browser state key `data_groups` (`get_state`/`restore_state`,
~lines 4152/4263). Purely a *view* concept — grouping never reaches
`ProjectModel`, `FitSeries`, or the engine.

**Angle column** (`ExtraColumn` with `is_angle=True`,
`data_browser.py:162–237`): the angle-dependence workflow adds a per-run numeric
column that becomes a **trend X-axis** option, but note it is **not** a
`FitSeries.order_key` value (which only supports `run`/`field`/`temperature`).
Angle is handled in the trend panel's plotting, not in series member ordering —
a latent inconsistency (see §5).

---

## 2. Single fits vs batch fits vs global fits

### 2a. The engine (one minimiser seam)

`core/fitting/engine.py`:
- **Single fit:** `FitEngine.fit(dataset, model_fn, parameters, t_min, t_max,
  method, minos, cost_factory, …) -> FitResult` (~line 553).
- **Global/joint fit:** `FitEngine.global_fit(datasets, model_fn, global_params,
  local_params, initial_params, …) -> (dict[key, FitResult], ParameterSet)`
  (~line 857). Concatenates all datasets into one cost and minimises once, with
  one Minuit parameter per free global shared across datasets.
- Both funnel through **`drive_minuit()`** (~line 332): Migrad/Simplex → HESSE →
  optional MINOS. Single seam, so error handling and MINOS behave identically
  across paths.
- **`FitResult`** (~line 495) is the transient result:
  `success, chi_squared, reduced_chi_squared, parameters, uncertainties,
  covariance, residuals, minos_errors, warnings`.
- **`CostFactory`** (~line 436): `GAUSSIAN_COST` (default, √-weighted LSQ) vs
  `POISSON_COST` (Cash statistic, count-domain). Passed through all paths.

### 2b. The batch/series algorithms (per domain — **not unified**)

There are **two separate "series fit" implementations**, one per time-domain
representation. This is a major consolidation target.

- **F-B asymmetry batches:** `core/fitting/series.py` ✓ (`fit_asymmetry_series`
  ~line 241). A *block-separable* chain: when every free parameter is Local, the
  joint objective factorises into N independent `FitEngine.fit` calls. Adds
  **chain seeding** (`seeding="chain"`, warm-start from previous *good* run,
  `_chain_seed` ~line 134) and **spurious-branch detect+reseed** (amplitude
  collapse / frequency discontinuity near T_C → reseed from the smooth trend,
  refit, keep the better attempt). Returns `AsymmetrySeriesResult`
  (`results: dict[int, FitResult]`, `seeding_used/reason`, `reseeded_runs`).
- **Grouped time-domain batches:** `core/fitting/grouped_time_domain.py`
  (`fit_grouped_series` ~line 934). Members are `(run, detector-group)` pairs.
  Two-tier classification: physics params are `global`/`local`/`fixed` *across
  runs*, while the **nuisance block** (`N0`, background, amplitude,
  `relative_phase`) is *always per-(run,group) local* and never trended.
  Dispatches `_fit_grouped_series_independent` (batch/chain) or
  `_fit_grouped_series_global`.

**Frequency/MaxEnt have no batch-series fitter of their own** — FFT "trends" are
built from per-run single fits recorded on the `FREQ_FFT` slot and aggregated
into a `FitSeries` at record time (see §3 of the FFT branch memory). This
asymmetry (time has rich chained batch fitting; frequency does not) is another
consolidation surface.

### 2c. GUI dispatch

- Single fit: `SingleFitTab` in `gui/panels/fit_panel.py` (~1946). Fit →
  `FitEngine.fit` → `_apply_single_fit_result` (~2827) updates the table, emits
  `fit_completed` → `MainWindow._on_fit_completed` (`mainwindow.py:8237` ✓) →
  `_record_single_fit_slot`.
- Batch/global: `GlobalFitTab` (~3052) / grouped path
  `_run_grouped_series_fit` (~5697). F-B dispatch decides series-vs-global by
  counting free global params (`fit_panel.py` ~4805): **0 free globals →
  `fit_asymmetry_series`** (chained batch); **≥1 → `engine.global_fit`**.

### 2d. Where results live, and how single/batch interact

**Storage is by `(run, representation)` FitSlot, plus a project-level
`FitSeries` for batches.** Recording paths (all in `mainwindow.py` ✓):

- `_record_single_fit_slot` (8956 ✓): writes the active representation's slot
  with `provenance="single"` and the full `ui_state` form payload (so the editor
  restores verbatim). Then calls `refresh_divergence()`.
- `_record_global_fit_batch` (8997 ✓): F-B batch/global → builds a `FitSeries`
  (`param_roles`, `canonical_model`, `results_by_run` each stamped with the run's
  T/B via `_dataset_trend_coords`), then `_record_fit_series`.
- `_record_grouped_fit_series` (9500 ✓): grouped batch → `FitSeries(member_kind
  ="groups")`. **A single-source-run grouped fit is deliberately *not* a series**
  (a one-point trend has nothing to trend): it is stored as an ordinary single
  slot with `provenance="single"`, returns `None`.
- `_record_fit_series` (8919 ✓): the shared multi→series core — sort members,
  `remove_superseded_batches` (de-dup identical re-runs), `add_batch`, write one
  pointer `FitSlot` per source run, `refresh_divergence`.

**Interaction rules (as built):**
- A **single fit and a batch fit are stored independently** on the *same slot*
  — but there is only **one** primary slot per `(run, representation)`, so the
  **last write wins**. Recording a batch overwrites each member run's slot
  (`provenance` → `batch`/`global`, `batch_id` set). A later single fit on that
  run overwrites it back to `provenance="single"` and **diverges** it from the
  batch (see §4). Per-projection single fits are *not* series members and never
  affect divergence (base.py `_fit_key`, mainwindow `_record_single_fit_slot`).
- **Carry-forward** (`fit_panel._carry_forward_single_fit_form` ~8115): moving to
  a new/unseen dataset inherits the *previous* model + seeds + bounds + fixed
  flags + link-groups but **clears uncertainties/result** so an unfit run never
  shows a stale result. Structure carries; values are per-dataset.
- **Inherited batch seeds** (`fit_panel._refresh_inherited_single_fit_defaults`
  ~3976): when a batch model matches each run's last single fit, the batch seeds
  Local params from that run's single result and Global params from the
  cross-run average. Precedence: inherited single-fit seed → table value → model
  default.
- **Re-fit as co-added / co-add** (`mainwindow` ~10445, `core/transform/combine`):
  combines selected runs into one synthetic dataset, single-fits it, and records
  a *separate computed row* — originals untouched.

---

## 3. How fit functions/models are shared between datasets

**Design principle: model *structure* is shared/reused; parameter *values* are
per-dataset.**

- **Registries (immutable, process-global):** `core/fitting/models.py`
  (`MODELS`, single-channel `ModelDefinition`s),
  `core/fitting/composite.py` (`COMPONENTS`, baseline-free
  `ComponentDefinition`s), `core/fitting/parameter_models.py`
  (`PARAMETER_MODEL_COMPONENTS`, the trend-domain models like `CriticalDivergence`,
  `OrderParameter`). All insertion goes through `registration.insert_definition`
  (name grammar + uniqueness). **A bare name does not identify its registry/domain**
  — the same string can exist in different registries (see ARCHITECTURE §4.3).
- **`CompositeModel`** (`composite.py` ~1349): built from component names +
  operators + parentheses + fraction groups. Resolves each name to its registered
  `ComponentDefinition`, flattens a unified `param_names`/`defaults`/`info`
  (aliasing repeated components `A`,`A_1`,`A_2`; fraction-group amplitudes;
  suppressed amplitudes under `*`/`/`). **`to_dict()`/`from_dict()` serialise the
  structure but not values** — this is exactly what a `FitSlot.model` and a
  `FitSeries.canonical_model` store.
- **`ParameterSet`/`Parameter`** (`parameters.py` ~700): per-dataset values,
  bounds, `fixed`, `expr` ties, and **`link_group`** (WiMDA equality links —
  parameters sharing a `link_group` id are constrained equal during minimisation)
  and `AffineTie` (offset / equal-spacing).
- **Sharing across a batch:** one `CompositeModel` (as `canonical_model`) is
  reused for every member; the *relationship* between members is entirely
  expressed by `FitSeries.param_roles` (`global`/`local`/`fixed`). "Is this a
  batch or a global fit?" is **derived from the roles**, not a separate object
  (`FitSeries.is_global()` ✓). Group series additionally carry the always-local
  `nuisance_params` block, excluded from `param_roles`.
- **Global/joint fits:** same single `model_fn`/`CompositeModel` for all
  datasets; `global_params` get one shared Minuit parameter, `local_params` get
  one per dataset (`asymmetry_global.fit_global`, `engine.global_fit`).
  Cross-group (parameter-domain) joint fits mirror this over detector groups /
  trend points (`gui/panels/cross_group_fit_dialog.py`,
  `gui/panels/knight_joint_fit_dialog.py`). The global-fit wizard scores which
  params should be global (`core/fitting/global_fit_wizard.py`).
- **Divergence = "does this member's stored model still match the series'
  canonical model?"** (`ProjectModel.refresh_divergence` ✓,
  `series.canonical_model_matches` ✓, compared on normalised `to_dict` form).

---

## 4. Divergence & trend inclusion (the glue that ties fits to trends)

`ProjectModel` (✓) reconciles per-run `FitSlot`s against each `FitSeries`:

- On every record/single fit, `refresh_divergence()` walks each batch. A member
  whose slot model no longer matches `canonical_model` is flagged `diverged` and
  **excluded from trending by default the first time** (`include_in_trend=False`).
  Re-matching re-includes it. A *manual* re-inclusion of a still-diverged member
  is preserved.
- Group series evaluate divergence at the **source-run** level (all synthetic
  members of a run diverge/reconverge together, `_refresh_group_series_divergence`
  ✓).
- **Computed series** (integral/field scans, co-adds — `canonical_model is None`,
  `FitSeries.is_computed` ✓) are skipped by divergence entirely and do not clear
  a run's fit state when deleted.
- `trend_runs_for_batch` / `set_member_trend_inclusion` (✓) drive which members
  the trend panel plots.

**Trend rendering is pull-based** (`MainWindow._refresh_trend_panel` 8638 ✓):
after any series-recording fit or representation change, it gathers *all
`FitSeries` for the active representation*, builds per-member row dicts
(`_build_series_rows` 8491 ✓ — collapses group series to one row per source run,
drops nuisances, resolves T/B from the *displayed* browser coordinate honouring
logged-T/B toggles), and calls `FitParametersPanel.load_representation_series`.
The panel shows a red "Series" button per `FitSeries`; selecting one decoratively
tints member rows in the browser (never a real Qt selection — ARCHITECTURE §3.5).

**Trend X-axis:** stamped per member at record time via `_dataset_trend_coords`
(8401 ✓) → `{field, temperature}` (or `None`, never `0`, for a missing axis).
`MuonDataset.temperature/.field` return `None` when absent (no 0 fallback), but
`Run.temperature` floors to `0.0` — a known trap that previously collapsed trends
to 0 (see memory `fit-series-trend-branch`). Secondary **trend model fits**
(fit a `CriticalDivergence`/`OrderParameter` to a parameter-vs-T curve) live in
`gui/panels/model_fit_dialog.py` (`ModelFitDialog`), seeded by
`parameter_models.suggest_trend_seeds` (physics-aware Tc/amplitude seeds), and
persist into `TrendState.model_fits[param_name]`.

---

## 5. Cross-representation behaviour — where it is *inconsistent* (the work)

This is the heart of the consolidation. Observed asymmetries between
representations, roughly in priority order:

1. **Two independent batch-series engines.** Time-FB uses
   `fitting/series.py` (chain + reseed); grouped uses
   `fitting/grouped_time_domain.py`; frequency has **none**. Chain seeding and
   spurious-run recovery therefore exist for F-B batches but not for grouped or
   frequency batches. Consolidation: a single series-fit orchestration that all
   representations feed, with per-domain kernels.

2. **"Group" overload.** DataGroup (runs) vs detector group vs
   `member_kind="groups"`. Nothing is wrong mechanically, but the shared word
   makes the UI and code hard to reason about. Consolidation: rename / clearly
   badge these in UI and docstrings; make DataGroup → batch launch an explicit,
   discoverable path.

3. **One primary `FitSlot` per `(run, representation)` → last-write-wins between
   single and batch.** A single fit silently overwrites (and diverges) a batch
   member; there is no "these are two different fits of the same data" model
   except via the diverge/exclude flag. Users can lose a batch result by
   idly single-fitting a member. Consolidation: decide whether single and batch
   fits should coexist per slot, or make the overwrite explicit/undoable.

4. **Angle trend axis is bolted onto the panel, not the series.**
   `FitSeries.order_key ∈ {run, field, temperature}` — angle is a browser column
   surfaced only in trend plotting. Ordering, member sorting, and persistence of
   an angle-ordered series are therefore second-class. Consolidation: promote
   angle (and arbitrary custom columns) to a first-class `order_key`/trend axis
   uniformly.

5. **Trend-coordinate provenance differs by path.** `MuonDataset.temperature`
   returns `None` for missing; `Run.temperature` floors to `0.0`; the browser's
   *displayed* coordinate (logged-T/B toggles) is what gets stamped. Several past
   bugs (trend collapse to 0) trace here. Consolidation: one coordinate-resolution
   helper used by every record path.

6. **Single-point special-casing is per-path.** Grouped single fits are demoted
   to a non-series slot; F-B single fits are always slots; a one-run "batch" is
   suppressed. The rule ("don't make a 1-point series") is correct but
   re-implemented in each recorder. Consolidation: centralise the
   "when does a fit become a series?" decision.

7. **`ui_state` (form-restore) is single-fit only.** Batch/global members carry
   no `ui_state`, so reopening a batch member's editor cannot restore the exact
   batch form the way a single fit restores. Consolidation: decide the restore
   contract for batch members.

None of these are bugs to fix blindly — each is a *design decision* to make
consistent. The next agent should treat this list as the consolidation backlog
and confirm each with the user before changing behaviour.

---

## 6. Navigation index (files, grouped by concern)

### Core — representation spine (read these first) ✓
- `src/asymmetry/core/representation/base.py` — `Representation`, `FitSlot`,
  `RepresentationType`, per-projection fit slots.
- `src/asymmetry/core/representation/series.py` — **`FitSeries`** (members,
  `param_roles`, `results_by_run`, `is_global`, `is_computed`, divergence,
  ordering).
- `src/asymmetry/core/representation/project_model.py` — **`ProjectModel`**
  (batches, `refresh_divergence`, `trend_runs_for_batch`,
  `remove_superseded_batches`, `recompute_all`, project (de)serialisation).
- `src/asymmetry/core/representation/container.py` — `DatasetRepresentations`.
- `src/asymmetry/core/representation/time.py` / `frequency.py` — the five
  representation `compute()` recipes.
- `src/asymmetry/core/representation/factory.py` — `REPRESENTATION_REGISTRY`.
- `src/asymmetry/core/representation/trend_state.py` — `TrendState`.

### Core — fitting
- `src/asymmetry/core/fitting/engine.py` — `FitEngine.fit`, `.global_fit`,
  `drive_minuit`, `FitResult`, `CostFactory`.
- `src/asymmetry/core/fitting/series.py` — F-B block-separable chained batch
  (`fit_asymmetry_series`, `_chain_seed`, spurious-branch recovery).
- `src/asymmetry/core/fitting/series_seeding.py` — `diagnose_series`,
  `suggest_series_seeds`, `recommend_series_seeding`.
- `src/asymmetry/core/fitting/grouped_time_domain.py` — grouped batch
  (`fit_grouped_series`, `_fit_grouped_series_independent/_global`,
  `_group_dataset_run_number`, `GroupedSeriesFitResult`).
- `src/asymmetry/core/fitting/asymmetry_global.py` — `fit_global`,
  `GlobalFitResult`.
- `src/asymmetry/core/fitting/global_fit_wizard.py` — global/local role scoring.
- `src/asymmetry/core/fitting/composite.py` — `CompositeModel`, `COMPONENTS`.
- `src/asymmetry/core/fitting/models.py` — `MODELS`, `ModelDefinition`.
- `src/asymmetry/core/fitting/parameters.py` — `Parameter`/`ParameterSet`,
  `link_group`, `AffineTie`.
- `src/asymmetry/core/fitting/parameter_models.py` — trend-domain models +
  `suggest_trend_seeds`.
- `src/asymmetry/core/fitting/registration.py` — registry insertion core.
- `src/asymmetry/core/data/dataset.py` — `MuonDataset`/`Run` and the
  `temperature`/`field` property semantics (§4 trap).
- `src/asymmetry/core/utils/constants.py` — `ORDER_KEYS`.

### GUI — recording & trend glue (MainWindow) ✓
- `gui/mainwindow.py::_active_representation_type` (8262) — view→representation.
- `::_on_fit_completed` (8237), `::_record_single_fit_slot` (8956).
- `::_record_global_fit_batch` (8997), `::_record_grouped_fit_series` (9500),
  `::_record_fit_series` (8919).
- `::_refresh_trend_panel` (8638), `::_build_series_rows` (8491).
- `::_dataset_trend_coords` (8401), `::_data_group_name_for_runs` (9687),
  `::_default_batch_series_label` (9710).

### GUI — panels/dialogs/windows
- `gui/panels/data_browser.py` — **`DataGroup`** (136), `ExtraColumn`/Angle
  (162), group CRUD (891–1064), `displayed_coordinate`, browser state persist.
- `gui/panels/fit_panel.py` — `SingleFitTab`, `GlobalFitTab`, carry-forward
  (~8115), inherited seeds (~3976), grouped-series launch (~5697).
- `gui/panels/fit_parameters_panel.py` — the **trend panel**
  (`load_representation_series`), X/Y selection, angle axis.
- `gui/panels/model_fit_dialog.py` — `ModelFitDialog` (secondary trend fits).
- `gui/panels/fit_function_builder.py` — model construction from expression.
- `gui/panels/fit_parameters_panel.py`, `cross_group_fit_dialog.py`,
  `knight_joint_fit_dialog.py` — cross-group / Knight-shift joint fits.
- `gui/windows/multi_group_fit_window.py`, `global_parameter_fit_window.py`,
  `multi_group_fit_window.py`, `fit_wizard_window.py`,
  `global_fit_wizard_window.py` — grouped/global/wizard fit surfaces.
- `gui/panels/alc_panel.py` — ALC/integral scan (a *computed* `FitSeries`).

### Docs
- `docs/ARCHITECTURE.md` §3.5 (representation model), §4.3 / §4.3.1 (time-domain
  fitting, grouped boundary), §4.3 registry-domain note.

---

## 7. Suggested next steps for the implementing agent

1. **Confirm the consolidation scope with the user** — §5 is a menu of design
   decisions, not a to-do list. Do not change behaviour before agreeing which
   inconsistencies to unify and which are intentional.
2. Reproduce each cross-representation behaviour live in the GUI (use the WiMDA
   Muon School corpus, `docs/testing/`) and capture the *current* UX for each of
   the five representations: single fit → batch fit → trend, then switch views.
3. Prefer a **core-first** unification (a single series-fit orchestration + one
   coordinate resolver + one "when is it a series?" rule), with GUI recorders
   collapsing onto it — matching the existing core/GUI split invariant.
4. Land behind the study-first workflow: keep this README as the study index and
   add `comparison.md` / `implementation-options.md` / `verification-plan.md`
   siblings as the design firms up (mirroring
   `docs/porting/asymmetry-error-propagation/`).

---

*Prepared as a read-only review. No source files were modified. All non-✓ line
anchors are agent-reported and should be re-verified before edits.*
