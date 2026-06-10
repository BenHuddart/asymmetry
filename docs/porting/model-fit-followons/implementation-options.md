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
