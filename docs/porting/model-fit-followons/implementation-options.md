# Implementation options & plan: Model-fit follow-ons

Branch: `feat/model-fit-followons` (off `origin/main` `8e36d34`), worktree
`/Users/bhuddart/Source/Asymmetry-model-fit-followons` with its **own**
`.venv` (numpy 2.2.x). Each phase ends with `python tools/harness.py validate`
green (worktree `.venv/bin/python`; GUI tests `QT_QPA_PLATFORM=offscreen`) and a
milestone commit. **No push, no PR** until Ben asks.

The study (README, comparison, test-data, verification-plan) is the source of
truth; this file records the chosen options and the cold-startable plan.

## Chosen options (decided 2026-06-10)

| # | Choice | Decision |
|---|---|---|
| A | Item 3 target & recursion | Same panel, new *Model fit results* `_GroupFitData` series; recursion via item 1's arbitrary-X. |
| B | Item 3 row sources | Cross-group **local** params (one row/group) **+ global** params; single-fit-range export → follow-on. |
| C | Item 1 x-uncertainty | Opt-in effective-variance (Orear/York), default OFF, only when x is a fitted param; horizontal x-bars always shown for `param:*` x. ODR rejected (no box bounds). |
| D | Item 3 globals representation | **Constant columns on every local row + one global-summary row** (χ²ᵣ + globals±err). Cross-fit global accumulation → follow-on. |
| E | Effective-variance scope | **Single-series path only** this pass; cross-group x-uncertainty → follow-on. |
| F | X-key encoding | `"param:<name>"` namespace; reserved keys `field`/`temperature`/`run` unchanged. |
| G | `_FitRow` extension | Add `origin: str \| None = None` only; reuse field/temperature slots for the derived-row coordinate. |
| H | Eff-var toggle granularity | Dialog-level (whole `ModelFitDialog`, like the error-mode combo); persisted as `use_x_errors` on `ParameterModelFit`. |
| I | Results-series identity | Keyed on `param_name + x_key + group set` (mirror `_cross_group_config_key`); re-running replaces, never duplicates. |

## Phase 1 — Arbitrary X column + effective-variance (panel + core)

### 1.1 Core (`core/fitting/parameter_models.py`)
1. `_run_parameter_model_minuit`: add `xerr: NDArray | None = None`. When `xerr`
   is None/all-zero → keep the current `iminuit.cost.LeastSquares` path
   (byte-identical). When finite σ_x present → build a **custom least-squares
   cost**: `resid = (y − f(x;θ)) / sqrt(σ_y² + slope² σ_x²)`, slope via central
   difference `(f(x+h)−f(x−h))/2h`, `h = max(|x|,1)·1e-6`; `chi_squared=m.fval`,
   `ndof=N−n_free`. Preserve `m.limits`, fixed params, result extraction.
2. `fit_parameter_model`: add `xerr: NDArray | None = None`; mask + window-slice
   it alongside x/y/e; pass through to `_run_parameter_model_minuit`. SCATTER and
   COLUMN-floor logic unchanged.
3. No change to `component_names_for_x` (already common-only for unknown keys);
   add a test pinning it for `param:*`.

### 1.2 Panel (`gui/panels/fit_parameters_panel.py`)
1. `_normalize_x_key` (`:131`): preserve `param:*` (return as-is); reserved keys
   unchanged; junk → `run`.
2. `_effective_x_key` (`:1700`): when the combo holds a parameter entry, return
   its `param:<name>` key (store the encoded key as the combo item's userData).
3. `_x_value` (`:3031`): `if x_key.startswith("param:")` → `row.values.get(name,
   nan)`. Add `_x_error(row, x_key)` → `row.errors.get(name, nan)` for bars/σ_x.
4. X-combo population: append trendable fitted-parameter entries (union of
   `row.values` keys for the active series, same set as the Y-selector), labelled
   with `get_param_info(name)` symbol, userData `param:<name>`. Rebuild when the
   Y-controls rebuild and on group/series switch. Preserve current selection
   across rebuilds.
5. Sampling `_sample_fit_range_curve` (`:3345`): `param:*` → force linspace (skip
   the `field` geomspace branch).
6. Labels: `_format_x_label_gle` (`:95`) and the MPL x-label (`~:2867`) → for
   `param:<name>` use `_format_gle_label(name)` / `_format_plot_label(name)`.
7. Plot x-bars: in `_refresh_plot`, when `x_key` is `param:*` draw horizontal
   error bars from `_x_error`; GLE export emits the x-error column.
8. `_group_variable_value_for_rows` (`:2325`): add a `param:*` branch → fall back
   to temperature, then field, then run-index coordinate.

### 1.3 Dialog (`gui/panels/model_fit_dialog.py`)
1. Accept `x_errors: NDArray | None` in `__init__`; store `self._xerr`.
2. Add an **"Account for x uncertainty"** checkbox in the data section, enabled
   only when `x_key.startswith("param:")` **and** `self._xerr` is finite;
   default unchecked. (Hidden when `_supports_error_modes` is False? No — it is
   independent; but cross-group dialog leaves it off this pass — gate on a new
   `_supports_x_errors = True` class flag, False on `CrossGroupFitDialog`.)
3. `_run_fit` (`:889`): pass `xerr=self._xerr if checkbox.isChecked() else None`
   to `fit_parameter_model`.
4. `_open_model_fit_dialog` (`:2174`): gather `x_errors` via `_x_error` and pass
   to the dialog.

### 1.4 Persistence
`use_x_errors: bool = False` on `ParameterModelFit`; serialise in
`_serialize_model_fits` (`:3132`) and `_serialize_local_model_fits`
(`global_parameter_fit_window.py:290`); read back leniently (legacy → False).
`x_key="param:*"` already serialises; the only fix is `_normalize_x_key`
preserving it (1.2.1).

### 1.5 Tests (`tests/test_model_fit_followons.py` + extend existing)
test-data §1.1–1.4, §2.1–2.3; offscreen GUI tests for combo population, x-label,
x-bars, the toggle gating; persistence round-trips incl. legacy. → milestone
commit "model-fit-followons: arbitrary-X param-vs-param + effective-variance".

## Phase 2 — Cross-group error modes + fit windows

### 2.1 Core (`core/fitting/parameter_models.py`)
1. `CrossGroupFitResult` (`:1479`): add `error_mode: str =
   ErrorMode.COLUMN.value`, `n_points: int = 0`.
2. `global_fit_parameter_model` (`:1881`): add `error_mode`, `error_value`,
   `windows` params. In `cost_function` and the `total_points` accounting,
   per group: `mask &= windows_mask(x, windows, None, None)` (or x_min/x_max if
   passed) and σ = `apply_error_mode(y, group.yerr, error_mode, error_value)`
   (None → unit weights). Populate `n_points` (the masked total). After the fit,
   SCATTER → rescale `global_unc` and every `local_unc[gid]` by `√(χ²/ν)`;
   ν<1 → empty dicts + message. Set `error_mode` on the result.

### 2.2 GUI (`gui/panels/cross_group_fit_dialog.py`)
1. Flip `_supports_error_modes`/`_supports_windows` to True; keep
   `_post_rebuild_ranges_ui` hiding the per-range active checkbox; set
   `_supports_x_errors = False`.
2. `_run_fit` (`:425`): stop pre-slicing by `[x_min,x_max]` **when windows are
   present** (avoid double-masking — core slices); pass
   `error_mode=self._error_mode()`, `error_value=self._error_value()`,
   `windows=fit_range.windows`.
3. `_collect_config`/`_apply_existing_config` (`:139`,`:212`): add
   `error_mode`/`error_value`/`windows` keys (windows `[[lo,hi]]`,
   `parse_fit_windows` on read); legacy → Column/no windows.

### 2.3 Tests
test-data §3.1–3.4; invert
`test_cross_group_dialog_hides_unsupported_controls` → asserts present+wired; new
backend-honesty test; degenerate two-identical-groups equality. → milestone
commit "model-fit-followons: cross-group error modes + fit windows".

## Phase 3 — Results-table recursion

### 3.1 Schema
`_FitRow` (`:143`): add `origin: str | None = None`; update the row
(de)serialiser inside `_serialize_group_fit_results`/`_deserialize_group_fit_results`
(`:1324`,`:1386`) — legacy rows → `origin=None`.

### 3.2 New series builder (`fit_parameters_panel.py`)
In `_apply_cross_group_fit_to_groups` (`:1273`) — after the existing overlay
attach — build a *Model fit results* `_GroupFitData`:
- One `_FitRow` per group: `values = {**global_values, **local_values[gid]}`,
  `errors = {**global_unc, **local_unc[gid]}`, coordinate =
  `group.group_variable_value` written into the slot matching the new series'
  `inferred_x_key` (the group-variable axis), synthetic negative unique
  `run_number`, `run_label = group.group_name`, `origin="cross_group"`.
- Plus one **global-summary** `_FitRow` (decision D): `values = global_values`,
  `errors = global_unc`, a distinct label ("globals"), `origin="cross_group_global"`,
  coordinate = NaN (excluded from group-variable trends; selectable via
  arbitrary-X). Carry `reduced_chi_squared` (store as a pseudo-value column
  `chi2r` or on the row — decide in code; keep it a normal `values` entry so it
  trends).
- Register under a synthetic batch id keyed per decision I (replace on re-run).
- Set the series' `inferred_x_key` to the group-variable axis; do **not** steal
  the active tab.

### 3.3 Tests
test-data §4.1–4.5, incl. the **recursion round-trip** (second trend fit on the
derived series) and series persistence + overwrite. → milestone commit
"model-fit-followons: cross-group outputs → trendable results series".

## Phase 4 — STRETCH: quadrature combinator (⊕)

Only if 1–3 green with budget left. `⊕` in `ParameterCompositeModel` grammar
(parser/`operators`, `function`, `formula_string`, `to_dict/from_dict`), builder
dialog (`fit_function_builder.py`), GLE export. Oracle: `PowerLaw ⊕ Constant` ==
`PowerLawQuadBG`. If not reached, record design state here and leave as follow-on.

## User docs (extend `docs/user_guide/parameter_trending.rst`)
Sections (result-first; rendered math; 0.23(1) uncertainties; APS reference
lists; "when to use this" register): **Trending one parameter against another**
(param-vs-param, when it's meaningful, scope degradation); **Accounting for
x-uncertainty** (effective-variance, Orear/York refs, when it matters, default
off, WiMDA divergence E1); **Error modes & windows in cross-group fits** (now
available); **Recursive trending** (model-fit outputs as a new series). Build
clean with `python tools/harness.py docs`.

## Follow-ons (recorded)
1. **Single-fit-range export** to the results table (deferred per decision B).
2. **Cross-group x-uncertainty** (effective variance in
   `global_fit_parameter_model`; toggle on `CrossGroupFitDialog`) — decision E.
3. **Cross-fit global accumulation** — one global-summary row per successive
   cross-group fit into a dedicated series for true global recursion (decision D).
4. **Generic quadrature combinator** if Phase 4 not reached.
5. **python-user-functions** (Wave B) generalises the registry — land after.

## Verification outcomes (2026-06-10)

Phases 1–3 implemented; full `validate` green after each
(2011 → 2017 → ~2030 passed). Docs build clean (only pre-existing
missing-screenshot warnings). Milestone commits per phase.

- **Phase 1**: effective variance is byte-identical to OLS at σ_x = 0 (tested),
  matches an independent scipy minimisation of the Orear/York cost, and
  inflates errors; param-vs-param runs on the real EuO trend; x-key encoding,
  combo, GLE x-column, dialog toggle and persistence all round-trip
  (`tests/test_model_fit_followons.py`).
- **Phase 2**: `global_fit_parameter_model` honours windows (point counts) and
  error modes (weighting), SCATTER rescales global+local σ, two-identical-groups
  equals the single-series fit, invalid windows fail soft; the cross-group
  dialog now exposes + wires the controls and round-trips config
  (`test_cross_group_fit_dialog.py` guard test inverted).
- **Phase 3**: cross-group outputs become a computed `FitSeries`; the recursion
  round-trip (a second trend fit on the derived series) succeeds; re-running
  replaces; the series survives a trend-panel refresh
  (`tests/test_model_fit_results_series.py`).

### Divergence from the planned Phase 3 design (recorded)

The plan envisaged injecting `_FitRow`s into a panel-only `_GroupFitData`
series with a new `_FitRow.origin` field. **Implemented instead as a first-class
model-less (`is_computed`) `FitSeries` recorded in `ProjectModel`** via
`MainWindow._record_model_fit_results_series`, because `load_representation_series`
rebuilds the panel's series from `ProjectModel` after every fit and on
representation switch — a panel-only series would be wiped. The ProjectModel
route makes the results series persist (project save/load), survive refreshes,
and appear like any other series. Consequences:

- `_FitRow.origin` was **not** added (provenance is the series label
  "Model fit: …"); no `_FitRow` schema change was needed.
- `_build_series_rows` now prefers summary-provided field/temperature/run_label,
  so computed members without a backing dataset carry their trend coordinate.
- Global params appear as constant columns on every group row **plus** a
  dedicated `globals` summary row carrying χ²ᵣ (decision D), the group row's
  orthogonal coordinate on the trend axis (NaN for the globals row).

## Authoring/agent notes
- Always verify `git branch --show-current` == `feat/model-fit-followons` before
  committing; never touch the hub checkout.
- `_FitRow.origin`, `ParameterModelFit.use_x_errors`, the cross-group config
  `error_mode`/`error_value`/`windows`, and the derived series are **all
  save/reload surfaces** — every one needs a legacy-load + round-trip test.
- Regression-safety: with the toggle off and reserved x-keys, every existing fit
  and every model-function-parity oracle must stay byte-identical.

---

# Second pass — finishing follow-ons A–D (2026-06-10, `feat/model-fit-finish`)

PR #38 (the first pass, items 1–3 + two fixes) **merged to `main`**
(`mergedAt 2026-06-10T19:14:04Z`). This second pass lands the four recorded
follow-ons that remain, after which the *model-fit-followons* project is DONE.

- Base: `origin/main` (contains PR #38). Worktree
  `/Users/bhuddart/Source/Asymmetry-model-fit-finish`, branch
  `feat/model-fit-finish`, with its **own** `.venv` (numpy 2.2.x).
- Verify `git branch --show-current == feat/model-fit-finish` before every
  commit; never touch the hub checkout.
- Each phase ends with full `python tools/harness.py validate` green (worktree
  `.venv/bin/python`; GUI tests `QT_QPA_PLATFORM=offscreen`) and a milestone
  commit. Baseline ≥ **2027 passed** and rising. **No push / no PR** until Ben
  asks.

## Scope ↔ recorded follow-ons

| This pass | = Recorded follow-on (above) | Decision deferred in #38 |
|---|---|---|
| **A** Generic quadrature combinator `⊕` | #4 | Phase 4 STRETCH not reached |
| **B** Single-fit-range export to results table | #1 | decision B |
| **C** Cross-group x-uncertainty | #2 | decision E |
| **D** Cross-fit global accumulation | #3 | decision D part (c) |

`python-user-functions` (#5) stays **out of scope** — Wave B owns the registry
generalisation; land after this. The `x2` second analytic variable stays
covered by cross-group fitting (not revisited).

## Confirmed decisions (attended checkpoint, 2026-06-10)

| # | Question | Decision | Rationale |
|---|---|---|---|
| J | **Implementation order** | **C → D → B → A** | Risk-ascending. C/D extend the already-shipped `global_fit_parameter_model` + `_record_model_fit_results_series` machinery (lowest marginal risk); B adds a new export path; **A (parser/grammar) last** — highest risk, isolated. |
| K | **`⊕` grammar scope** | **Parameter-vs-x grammar only** | `PowerLawQuadBG` and the whole motivating use case live in `ParameterCompositeModel`. The shared parser/tokenizer (`composite.py`) is touched minimally and the time-domain `CompositeModel` stays **byte-identical** and rejects `⊕`. Quadrature of two time-domain muon components has no established physical meaning and no WiMDA precedent — implementing it there would gold-plate an untested grammar. |
| L | **Item B per-range row x-coordinate** | **Range-window center** (`(lo+hi)/2` of `effective_range_bounds`, in the trend's native x-units) | A multi-range single fit (e.g. windows `[0,40]`,`[60,100]` → centers `20`,`80`) is immediately trendable; the center is a real physical x. One row per range, keyed per `(param, x_key)` like cross-group; re-running replaces. A single-range fit yields one row — recorded, trendable once a sibling range (or accumulation, follow-on) adds a second point. Fallback when bounds are open: actual masked-x min/max; final tiebreaker the range index. |

## Architectural facts established by the study (file:line, base `origin/main`)

- **Shared parser, single operator set.** `parse_component_expression` /
  `build_component_expression` and
  `_ALLOWED_OPERATORS = frozenset({"+","-","*","/"})` live in
  `core/fitting/composite.py:951,990,1141`. They are **imported by**
  `parameter_models.py:16-17` and used by `ParameterCompositeModel`
  (parameter grammar) **and** by `parse_composite_expression` /
  `CompositeModel` (time-domain). The tokenizer
  `_tokenize_component_expression` (`composite.py:956`) regex
  `[A-Za-z_][A-Za-z0-9_]*|[(){}+\-*/]` only knows ASCII operator chars.
- **Operator-button UI is shared but already parameterised.**
  `FunctionExpressionBuilderDialog` (`gui/widgets/function_expression_builder.py:217`)
  hard-codes the `+ - * / ( )` keypad but accepts an
  `extra_token_buttons: list[tuple[str,str]] | None` arg (`:227,288-296`).
  The **parameter** model builder is
  `ParameterModelBuilderDialog(FunctionExpressionBuilderDialog)`
  (`model_fit_dialog.py:281`, `super().__init__(... model_parser=`
  `ParameterCompositeModel.from_expression ...)` at `:296-305`); the
  **time-domain** builder is `FitFunctionBuilderDialog`
  (`fit_function_builder.py:68`). So the `⊕` keypad button is added by passing
  `extra_token_buttons=[("⊕", " ⊕ ")]` from `ParameterModelBuilderDialog` only.
- **Oracle subtlety (must encode in the test).** Registry `PowerLaw` is
  `a·|x|ⁿ + c` — it carries its **own additive `c`** (`parameter_models.py:339`).
  `PowerLawQuadBG` is `sqrt((a·|x|ⁿ)² + BG²)` — its inner term has **no `c`**
  (`:354`). Therefore `PowerLaw ⊕ Constant ≡ PowerLawQuadBG` holds **only with
  PowerLaw's `c` fixed at 0** and Constant's `c` = BG. The oracle test fixes
  `c=0`.
- **Single fit already holds a list of ranges.** `ParameterModelFit`
  (`:1448`) has `ranges: list[ModelFitRange]`; each `ModelFitRange` (`:1349`)
  carries its own `model`, `parameters`, `result: ParameterModelFitResult`
  (`:1332` — params/uncertainties/`reduced_chi_squared`/`n_points`/`error_mode`)
  and `windows`. `effective_range_bounds(range)` (`:1434`) gives the
  `(lo,hi)` envelope. So item B is "one trendable row per range".
- **Phase-3 recursion series mechanism (what C/D/B reuse).**
  `MainWindow._record_model_fit_results_series` (`mainwindow.py:6543`) builds a
  computed (`is_computed`) `FitSeries` from a `CrossGroupFitResult`: one member
  per group + a `globals` member, keyed by a deterministic
  `batch_id = "modelfit-<sha1(param::x_key::group_ids)>"` so re-running the
  same fit **replaces** (not duplicates). `_build_series_rows` (`:5466`)
  reads each member's summary `parameters`/`uncertainties`/`field`/
  `temperature`/`run_label`; a present-but-`None` coordinate → NaN (off-axis).
- **Cross-group core already takes the modes.** `global_fit_parameter_model`
  (`parameter_models.py:1957`) already accepts
  `error_mode`/`error_value`/`windows`; it has **no `xerr`** yet.
  `CrossGroupFitDialog._supports_x_errors = False`
  (`cross_group_fit_dialog.py:56`); the single dialog has
  `_supports_x_errors = True` and the effective-variance toggle
  (`model_fit_dialog.py:352,435,529,984`).

---

## Phase C — Cross-group x-uncertainty (decision E retired)

Thread effective-variance x-uncertainty through the cross-group path so that,
when x is a fitted parameter (`x_key == "param:*"`), each group's per-point σ_x
inflates its weight exactly as the single-series path already does.

### C.1 Core (`core/fitting/parameter_models.py`)
1. `global_fit_parameter_model` (`:1957`): add
   `xerr: Mapping[gid, NDArray] | None = None` (per-group x-error arrays,
   aligned to each group's stored x). When `xerr` is None/all-zero for a group →
   current weighting (byte-identical). When finite σ_x present → per-group
   residual `resid = (y − f(x;θ)) / sqrt(σ_eff²)`,
   `σ_eff² = σ_y² + slope²·σ_x²`, `slope = (f(x+h;θ) − f(x−h;θ))/2h`,
   `h = max(|x|,1)·1e-6` — the **same** central-difference estimator as
   `_run_parameter_model_minuit` (single-series). Reuse a shared helper so the
   two paths cannot diverge numerically (extract
   `_effective_sigma2(model, x, params, sigma_y, sigma_x)` and call it from
   both). The window mask + `apply_error_mode` (Phase 2) compose unchanged:
   σ_y comes from `apply_error_mode`, then σ_x inflates it.
2. NONE/SCATTER guard: x-uncertainty is meaningful only with a real σ_y scale,
   so **ignore `xerr` when `error_mode` ∈ {none, scatter}** (mirror the
   single-series guard) and note it in the result message — same rule the GUI
   toggle enforces.

### C.2 GUI (`gui/panels/cross_group_fit_dialog.py`)
1. Flip `_supports_x_errors = True` (`:56`). The inherited
   "Account for x uncertainty" checkbox (built in `model_fit_dialog.py:435`)
   then renders, enabled only when `x_key.startswith("param:")` **and** the
   per-group σ_x is finite, default unchecked, **and** disabled for
   none/scatter modes (add that condition to the enable predicate so C.1's
   guard and the UI agree).
2. `fit_parameters_panel._run_cross_group_model_fit` (`:2229`): gather per-group
   σ_x via the existing `_x_error(row, x_key)` (Phase 1) alongside the x arrays,
   and pass `xerr={gid: σ_x_array}` to `global_fit_parameter_model` when the
   toggle is on (else `None`).
3. Config: persist `use_x_errors: bool` in `_collect_config` /
   `_apply_existing_config` (`:139,212`) next to `error_mode`/`windows`; legacy
   → False. Reaches `.asymp` via the panel's cross-group-config serialiser.

### C.3 Tests
- σ_x = 0 (or toggle off) ⇒ cross-group result **byte-identical** to the
  Phase-2 path (regression oracle).
- Cross-group effective-variance on two identical groups equals the
  **single-series** effective-variance fit on one copy (≲1e-6) — proves C.1
  reuses the same numerics as the single path (decision E's whole point).
- Finite σ_x inflates global+local σ vs σ_x ignored.
- none/scatter ⇒ `xerr` ignored (guard honoured); message flags it.
- `use_x_errors` config round-trips through dialog + panel serialiser; legacy
  loads False.

→ milestone commit *"model-fit-finish: cross-group x-uncertainty (effective
variance)"*.

## Phase D — Cross-fit global accumulation (decision D part (c))

So that the *shared global parameters themselves* can be trended **across
successive cross-group fits**, accumulate one global-summary row per distinct
fit into a single dedicated series (separate from the per-fit results series
that Phase 3 already builds).

### D.1 The accumulation series (`mainwindow.py`)
In `_record_model_fit_results_series` (`:6543`), in addition to the existing
per-fit series, upsert into a **singleton** accumulator series:
- Fixed `batch_id = "modelfit-globals"` (one per representation type →
  `f"modelfit-globals-{rep_type.value}"`), `label = "Global summary"`,
  `is_computed`. It persists and **grows** across fits.
- One member per **distinct** cross-group fit, keyed by the same logical key as
  the per-fit series
  (`member_key = -(1_000_000 + int(sha1(param::x_key::group_ids), 16) % 1_000_000)`,
  negative + deterministic so re-running a fit **updates its single row** rather
  than appending). `parameters = {**global_vals, "chi2_r": χ²ᵣ}`,
  `uncertainties = global_unc`, `run_label = f"{param} vs {x_key}"`.
- **x that distinguishes successive fits.** Carry a monotonic per-series
  `accumulation_index` (1,2,3…, assigned in first-seen order, persisted on the
  member summary as a plain `param:` column `fit_index`) as the **default** x;
  values are also written into `field`/`temperature` as `None` (off the two
  physical axes) so the series defaults to trending the globals against
  `fit_index`. With arbitrary-X (Phase 1) the user can instead pick **any
  global parameter** as x → param-vs-param across fits (the real recursion this
  unlocks). The index is stable across reload because it is stored, not
  recomputed from `hash()`.
- Upsert semantics: load the existing accumulator series for the rep (if any),
  replace/append the one member for this fit, re-`add_batch`. Members preserve
  their stored `fit_index`; a genuinely new fit gets `max(existing)+1`.

### D.2 Persistence
The accumulator series persists for free as a computed `FitSeries`
(ProjectModel route). New summary keys (`fit_index`) are plain floats in
`parameters` → already serialised. Test: two successive cross-group fits →
2-row global-summary series; save→reload preserves both rows and their
`fit_index`; re-running fit #1 updates row 1 in place (still 2 rows).

### D.3 Tests
- Two distinct cross-group fits accumulate into a 2-member `modelfit-globals-*`
  series; each member carries that fit's globals + χ²ᵣ + `fit_index`.
- The accumulator series is **trendable**: a trend fit of a global parameter
  vs `fit_index` (and vs another global, via `param:*`) yields a finite result.
- Re-running an existing fit replaces its row (count unchanged); a new fit
  appends (count +1, `fit_index` = max+1).
- Save/reload round-trip preserves rows + indices.

→ milestone commit *"model-fit-finish: cross-fit global accumulation series"*.

## Phase B — Single-fit-range export to the results table (decision B retired)

Generalise the Phase-3 recursion so a **single-series** `ModelFitDialog` fit
also emits a trendable results series — one row per `ModelFitRange`.

### B.1 Find/confirm the single-fit completion hook
The single (non-cross-group) trend fit is applied in
`fit_parameters_panel` (the `ModelFitDialog` accept path that stores the
`ParameterModelFit` into the panel's model-fit registry). Identify that point
(parallel to `_run_cross_group_model_fit` → `_on_cross_group_fit_completed`)
and emit a signal carrying `(parameter_name, x_key, ParameterModelFit)` to the
main window, mirroring the cross-group signal. *(Confirm exact method names in
code during implementation — not yet pinned to a line.)*

### B.2 Record the series (`mainwindow.py`)
Add `_record_single_model_fit_results_series(parameter_name, x_key, fit)`:
- One member per range in `fit.ranges` whose `result.success`. For range `r`:
  `parameters = {**r.result.parameters_as_dict, "chi2_r": r.result.reduced_chi_squared}`,
  `uncertainties = r.result.uncertainties`, **x-coordinate = window center**
  `c = mean(effective_range_bounds(r))` written into the trend axis slot
  (the series' `inferred_x_key` = the fit's own x-key family, i.e. the same
  axis the single fit trended on — so the derived rows sit on a real x).
  `run_label = f"range {i}: [{lo:g},{hi:g}]"`. Open-bound fallback: use the
  masked-x min/max of the fitted data; final tiebreaker the range index `i`.
- Deterministic `batch_id = "modelfit-single-<sha1(param::x_key)>"`; re-running
  the same single fit **replaces** its series (one results series per
  `(param, x_key)` trace).
- Reuse `_build_series_rows` unchanged (members carry their own coordinates).

### B.3 Refactor note
`_record_model_fit_results_series` (cross-group) and the new single-fit recorder
share the FitSeries-assembly boilerplate (deterministic batch id, member
summaries, coordinate fields, `add_batch` + `_refresh_trend_panel`). Extract a
small `_build_results_series(batch_id, label, rep_type, members)` helper used by
both to keep them from drifting.

### B.4 Tests
- A single fit with **2 ranges** (two windows over one trace) → a
  `modelfit-single-*` series with 2 rows; each row's `values` = that range's
  fitted params + `chi2_r`, `errors` = its uncertainties, coordinate = the
  window center.
- **Recursion round-trip**: a second trend fit on the derived series (a param
  vs window-center) yields a finite result.
- Re-running the single fit replaces (no duplicate).
- Save/reload preserves the derived series.

→ milestone commit *"model-fit-finish: single-fit ranges → trendable results
series"*.

## Phase A — Generic quadrature combinator `⊕` (parameter grammar only)

`y = sqrt(f² + g²)` as a binary operator alongside `+ − * /`, **in the
parameter-vs-x grammar only** (decision K).

### A.1 Parser/grammar (`core/fitting/composite.py`)
1. `_tokenize_component_expression` (`:956`): extend the regex to recognise the
   `⊕` glyph as its own token —
   `r"[A-Za-z_][A-Za-z0-9_]*|⊕|[(){}+\-*/]"`. (Recognising it is harmless to the
   time-domain parser, which rejects it as a non-operator — see A.4.)
2. `parse_component_expression` (`:990`): add a keyword arg
   `allowed_operators: frozenset[str] = _ALLOWED_OPERATORS`. Replace the two
   `token in _ALLOWED_OPERATORS` tests (`:1013,1026`) with
   `token in allowed_operators`. Default behaviour is byte-identical (base set).
3. `build_component_expression` (`:1141`): no change — it emits whatever
   operator strings it is given; `⊕` flows through.

### A.2 Parameter grammar (`core/fitting/parameter_models.py`)
1. Define `_PARAMETER_ALLOWED_OPERATORS = _ALLOWED_OPERATORS | {"⊕"}`.
2. `ParameterCompositeModel.__init__` (`:963`): validate `operators` against
   `_PARAMETER_ALLOWED_OPERATORS`.
3. `from_expression` (`:1015`): pass
   `allowed_operators=_PARAMETER_ALLOWED_OPERATORS` into
   `parse_component_expression`.
4. **Precedence = same as `+`/`−`** (level 1), left-associative. Quadrature
   addition is associative & commutative (`(a⊕b)⊕c = √(a²+b²+c²)`), so a
   left-fold is exact; sharing level 1 keeps `PowerLaw ⊕ Constant + Linear` →
   `√(PL²+C²) + Linear`, which is the intended reading.
   - Flat `function()` (`:1084`): in the first pass `⊕` is a **level-1** op
     (the `else` branch alongside `+`/`−`: `reduced_ops.append(op)`), so it is
     not collapsed with `*`/`/`. In the final left-fold, add a branch
     `result = sqrt(result² + value²)` for `⊕`.
   - `_evaluate_parenthesized()` (`:1125`): `precedence("⊕") = 1`;
     `apply_top_operator` gains a `⊕` branch
     `value_stack.append(sqrt(lhs² + rhs²))`.
5. `formula_string()` (`:1062`) and `component_expression_string()`: `⊕` renders
   as the joined operator token (a recognisable symbol) — no further change;
   `to_dict`/`from_dict` already serialise arbitrary operator strings and
   re-validate via `__init__`, so `⊕` round-trips once `__init__` allows it.

### A.3 GUI + GLE
1. `ParameterModelBuilderDialog.__init__` (`model_fit_dialog.py:296`): pass
   `extra_token_buttons=[("⊕", " ⊕ ")]` → a `⊕` keypad button appears in the
   **parameter** model builder only (and in `CrossGroupFitDialog`, which
   subclasses it — fine: `⊕` is a parameter-grammar operator). The time-domain
   `FitFunctionBuilderDialog` does not pass it → no `⊕` there.
2. GLE export of the model formula already emits the expression string (which
   contains `⊕`); confirm the export path and add a test asserting `⊕` survives.

### A.4 Time-domain stays inert (verify)
`parse_composite_expression` (`composite.py:1058`) keeps using the base
`_ALLOWED_OPERATORS`; a `⊕` token there is treated as a non-operator and
**rejected** (UnknownComponentError / "Expected operator"). `CompositeModel`
behaviour is byte-identical. Add a test asserting the time-domain builder/parser
rejects `⊕` (the grammar boundary).

### A.5 Oracle + parser tests
- **Oracle**: `ParameterCompositeModel.from_expression("PowerLaw ⊕ Constant")`
  with PowerLaw `c` fixed at 0 and Constant `c = BG` equals
  `PARAMETER_MODEL_COMPONENTS["PowerLawQuadBG"].function(a,n,BG)` to ≲1e-12 over
  a grid (positive and negative x). Record the `c=0` requirement in the test
  docstring.
- **Parser**: precedence vs `+`/`*` (`A ⊕ B + C`, `A + B ⊕ C`, `A ⊕ B * C`),
  associativity (`A ⊕ B ⊕ C` = `√(A²+B²+C²)`), nesting/parentheses
  (`(A ⊕ B) * C`, `A ⊕ (B + C)`), `to_dict`/`from_dict` round-trip,
  `component_expression_string` round-trip, **`.asymp` round-trip** of a
  `ParameterModelFit` whose model uses `⊕` (legacy state without `⊕` still
  loads), and GLE emits `⊕`.

→ milestone commit *"model-fit-finish: quadrature combinator ⊕ in parameter
grammar"*.

## User docs (this pass)
Extend `docs/user_guide/parameter_trending.rst` and
`docs/user_guide/composite_models.rst` (result-first prose; rendered math;
`0.23(1)` uncertainties; APS reference lists; a "when to use this" register per
feature):
- **Quadrature combinator `⊕`** (composite_models.rst): `√(f²+g²)`, the
  `PowerLaw ⊕ Constant = PowerLawQuadBG` identity, when a quadrature background
  is the right model (a signal riding on an incoherent background floor),
  precedence note, parameter-grammar-only scope.
- **x-uncertainty in cross-group fits** (parameter_trending.rst): the
  single-series effective-variance section now applies to cross-group fits too;
  same Orear/York refs; default off; divergence E from WiMDA.
- **Trending model-fit outputs** (parameter_trending.rst): the single-fit
  range-center series and the global-summary accumulation series — "trend the
  outputs of trends".
Build clean with `python tools/harness.py docs`.

## Newly recorded follow-ons (this pass)
- **Single-fit accumulation across fits** (decision L's third option): an
  accumulator for single fits mirroring Phase D, so single-*range* fits become
  trendable across successive fits. Deferred — range-center already makes
  multi-range fits trendable; revisit if users want cross-fit single trends.
- **`⊕` in the time-domain grammar**: only if a physical use case emerges
  (none today); the parser is already parameterised to make it a small change.
- **python-user-functions** (Wave B) — unchanged.

## Divergence ledger (this pass; both behaviours stated)
| # | Topic | WiMDA | Asymmetry |
|---|---|---|---|
| E5 | Quadrature operator | none (no composite-of-components grammar; QuadBG is a fixed model) | `⊕ = √(f²+g²)` as a first-class parameter-grammar operator; `PowerLaw ⊕ Constant ≡ PowerLawQuadBG` |
| E6 | x-uncertainty in cross-group fits | x exact, no cross-group concept | optional effective variance (Orear/York), default off → identical to OLS; reuses the single-series estimator |
| E7 | Global-parameter recursion | second-level Model Fit Table is a text log, not trendable | shared globals accumulate into a trendable `Global summary` series across successive fits |

## Verification outcomes — second pass (2026-06-10, `feat/model-fit-finish`)

All four follow-ons implemented in the agreed order **C → D → B → A**; full
`python tools/harness.py validate` green after each (2109 → 2113 → 2117 → 2127
passed, 1 xfailed), `docs` builds clean (only pre-existing missing-screenshot /
`_static` warnings), milestone commit per phase.

- **Phase C** (`55d70d1`): extracted `_effective_variance_residual` as the
  single shared central-difference estimator and refactored the single-series
  cost onto it (byte-identical); `global_fit_parameter_model` gained
  `xerr: Mapping[gid, array]` composing with windows + error modes, ignored
  under None/Scatter. `ParameterGroupData.xerr` added.
  `CrossGroupFitDialog._supports_x_errors=True`, σ_x threaded through the
  window-slice/snapshot, `use_x_errors` persisted (dialog config + panel
  serialiser, legacy → off). Tests: σ_x=0 byte-identical to OLS; cross-group
  eff-var **equals the single-series eff-var on identical groups** (the shared
  estimator); σ_x inflates σ; None/Scatter ignore it; dialog exposes + gates +
  round-trips the toggle.
- **Phase D** (`eb7733b`): `_accumulate_global_summary_row` upserts one row per
  *distinct* cross-group fit into a singleton `modelfit-globals-<rep>` series
  (keyed by the per-fit logical key → re-run replaces), carrying that fit's
  globals + χ²ᵣ + a monotonic first-seen `fit_index`; rows sit off both physical
  axes and are trended vs `fit_index` or a global via arbitrary-X.
  `_infer_x_key` now spans finite coordinates only (no all-NaN-slice warning for
  an entirely-off-axis series). Tests: two fits → 2 members; trendable vs
  `fit_index`; re-run replaces (index kept); JSON round-trip.
- **Phase B** (`a4ef172`): new panel `model_fit_completed` signal →
  `_record_single_model_fit_results_series`, one member per successful
  `ModelFitRange`, x = window centre (`effective_range_bounds` midpoint), carried
  as a `range_center` column and onto the physical axis slot for
  field/temperature x-keys; deterministic `modelfit-single-<sha1(param::x_key)>`
  id → re-run replaces. Extracted `_add_results_series` shared by the
  cross-group, Global-summary, and single-fit recorders. Tests: two-range fit →
  rows at the window centres; recursion succeeds; re-run replaces; JSON
  round-trip.
- **Phase A** (`76e4836`): `⊕` added to the parameter grammar only —
  `parse_component_expression` parameterised by `allowed_operators` (time-domain
  stays byte-identical and **rejects** `⊕`), tokenizer recognises the glyph,
  `_PARAMETER_ALLOWED_OPERATORS`, precedence = `+`/`-`, evaluated as √(lhs²+rhs²)
  in both the flat and parenthesised paths, additive-component aware, `⊕` keypad
  button via `extra_token_buttons`. **Oracle exact** (max abs diff `0.0`):
  `PowerLaw ⊕ Constant` (power-law `c`=0) == `PowerLawQuadBG`. Tests also cover
  associativity, precedence vs `+`/`*`, parentheses, `to_dict`/expression
  round-trip, operator validation, time-domain rejection, the keypad + build.

### Divergences from the planned design (recorded)
- **D's default-x ambition vs reality.** The plan hoped `fit_index` would be the
  *literal default* x for the Global summary series; a computed `FitSeries`
  carries no inferred-x hint, and forcing one would mean threading new state
  through the schema. Implemented instead as: rows off both physical axes
  (default x-inference returns `run`), with `fit_index` and every global exposed
  as selectable arbitrary-X columns. The decision-D requirement (globals
  *trendable across fits*) is met; the literal default is a deferred nicety.
- **B/D member keys vs `fit_index` ordering.** Member keys must be deterministic
  (sha1-derived) so re-running a fit replaces its row; that conflicts with using
  the key itself as a monotonic axis, so `fit_index` is stored as a separate
  values column rather than encoded in the key.

### Follow-ons retired / newly recorded
Retired (now shipped): single-fit-range export (#1 → B), cross-group
x-uncertainty (#2 → C), cross-fit global accumulation (#3 → D), quadrature
combinator (#4 → A). Still open: **single-fit accumulation across fits**,
**`⊕` in the time-domain grammar**, **python-user-functions** (Wave B). A small
UX nicety also recorded: a stale single-fit / cross-group results series is not
auto-removed when its source fit is deleted (consistent across both paths).
