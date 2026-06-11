# Comparison: Model-fit follow-ons vs Asymmetry surfaces

Date: 2026-06-10. WiMDA source: `$WIMDA_SRC/src` (read
directly; `__history/`/`__recovery/` ignored). Asymmetry worktree off
`origin/main` `8e36d34` (PR #32 merged at `d64820c`).

This file records (a) the WiMDA arbitrary-column behaviour transcribed from the
Pascal, (b) the exact Asymmetry seams each item touches with file:line, (c) the
item-3 recursion model and `_FitRow` schema analysis, and (d) the
effective-variance derivation for item 1's x-uncertainty option.

---

## 1. Item 1 — Arbitrary X column (param-vs-param)

### 1.1 WiMDA behaviour (verified)

See [README §"How WiMDA does the arbitrary-X part"](README.md). The operative
facts, transcribed from the Pascal:

- `columnscan(s, c)` (`Model.pas:318–345`): split `s` on space/comma/tab, parse
  the `c`-th (1-based) token as a double; return 0 if absent. No type semantics.
- `Xcolumn`/`Ycolumn`/`Ecolumn`/`X2column` are independent `TSpinEdit`s
  (`Model.pas:21,29,30,40`), clamped to the column count by `checkcolumns`
  (`Model.pas:405–437`).
- `getmodeldata` (`Model.pas:560–631`) builds `(xx, y, yerr)` by reading those
  column indices per row; the multi-range window test (line 595–602) is applied
  to the chosen **X** column. No x-error path exists — x is exact in the fit.

Divergence carried forward: Asymmetry treats x-uncertainty optionally via
effective variance (§3); WiMDA never does (x exact).

### 1.2 Asymmetry today (seams, `gui/panels/fit_parameters_panel.py`)

The "results table" *is* this panel — `self._rows: list[_FitRow]` and the
`self._table` view behind "Show fitted parameter table". There is no separate
results widget. Key seams:

| Seam | Location | Note |
|---|---|---|
| X-axis combo | `:241–243` | items `["Auto", "𝐵 (G)", "𝑇 (K)", "Run"]` |
| `_effective_x_key()` | `:1700–1708` | maps combo text → `"field"`/`"temperature"`/`"run"` (string keys) |
| `_inferred_x_key` | `:202` default, hint at `:1871` | "Auto" resolution |
| `_normalize_x_key()` | `:131–140` | persisted key → internal id; **only** knows field/temperature/run |
| `_x_value(row, x_key)` | `:3031–3036` | the only x mapper: returns `row.field`/`row.temperature`/`float(row.run_number)` |
| `_x_domain_for_sampling()` | `:3336–3343` | via `_x_value` |
| `_sample_fit_range_curve()` | `:3345–3390` | geomspace iff `x_key=="field"`, else linspace |
| GLE x-label | `_format_x_label_gle()` `:95–100` | hard-codes the three keys |
| MPL x-label | inside `_refresh_plot()` `~:2867` | dict over the three keys |
| Component scope | core `component_names_for_x()` `parameter_models.py:924–938` | **already** degrades to `{"common"}` for any key ≠ field/temperature |
| Dialog launch | `_open_model_fit_dialog()` `:2174–2227` | passes `x_key`, `x_values`, `y_values`, `y_errors` |
| Persistence | `get_state()` `:443–490` / `restore_state()` `:492–637`; per-fit `ParameterModelFit.x_key` via `_serialize_model_fits()` `:3132–3185` | `x_key` round-trips per model-fit |

**Gap.** `_x_value` and `_normalize_x_key` only understand three reserved keys;
the combo offers only those three. To trend param-vs-param, x must be allowed to
name a *fitted parameter* (a key in `row.values`). The core is already ready:
`fit_parameter_model` takes any x array, and `component_names_for_x` already
returns `{"common"}`-scope components for an unrecognised key — **no core change
for scoping**.

### 1.3 Port shape (item 1)

- **X-key encoding.** Reserve a namespace for parameter x-keys so they never
  collide with `field`/`temperature`/`run`: encode as `"param:<name>"`. Decode
  in `_effective_x_key`/`_normalize_x_key`/`_x_value`. `_x_value` gains: if
  `x_key.startswith("param:")` → `row.values.get(name, nan)` (and the matching
  error from `row.errors` for bars / effective variance).
- **Combo population.** Append the currently-trendable fitted parameter names
  (the union of `row.values` keys across the active series, i.e. the same set
  feeding the Y-selector) to the X combo, prefixed/labelled with their
  `get_param_info` symbol. Rebuilt whenever the Y-controls rebuild.
- **Scope degrade.** For `param:*` x-keys, the dialog/panel must request
  `component_names_for_x("param:<name>")` → `{"common"}`. Already automatic;
  add a test pinning it.
- **Sampling.** `param:*` is not log-natural like field → always linspace
  (the `x_key=="field"` geomspace branch must not fire).
- **Labels.** GLE/MPL x-label for `param:<name>` uses
  `_format_gle_label(name)` / `_format_plot_label(name)` (the existing
  per-parameter symbol helpers at `:79–88`).
- **Persistence.** `ParameterModelFit.x_key` already serialises any string, so
  `"param:lambda"` round-trips with no schema change; but `_normalize_x_key`
  currently collapses unknown keys to `"run"` (`:140`) — it must preserve
  `param:*`. **This is a legacy-state surface: add a round-trip test.**
- **x-errors in the fit** — off by default; see §3.

### 1.4 Cross-group inherits item 1 for free

`_run_cross_group_model_fit` (`:2229`) also calls `_effective_x_key()` and
`_x_value()` (`:2237,2250`), so param-vs-param x flows into cross-group fits with
no extra work — but note the per-group coordinate `_group_variable_value_for_rows`
(`:2325–`) assumes field↔temperature; for a `param:*` x it must fall back to a
sensible group coordinate (temperature, then field, then run). Recorded for the
plan.

---

## 2. Item 2 — Cross-group error modes + fit windows

### 2.1 The honesty gap (why the controls were hidden)

`CrossGroupFitDialog` (`gui/panels/cross_group_fit_dialog.py:43–50`) subclasses
`ModelFitDialog` and sets `_supports_error_modes = False`,
`_supports_windows = False`, hiding the inherited combo/value-field and the
"+ Window" button, *because the backend ignores them*:

`global_fit_parameter_model` (`parameter_models.py:1881–2089`) builds its cost
straight from `group.yerr` with only an `(e > 0)` finite mask
(`:2008,2016`) — no `apply_error_mode`, no `windows_mask`. Un-hiding the controls
without first making the backend honour them is the documented failure mode.

### 2.2 Core port (do this first)

`global_fit_parameter_model` gains `error_mode: ErrorMode | str =
ErrorMode.COLUMN`, `error_value: float | None = None`, and
`windows: Sequence[tuple[float, float]] | None = None`:

- In `cost_function` (`:2001`) and the `total_points`/`ndof` accounting
  (`:2030–2038`), replace the bare `(e > 0)` mask with
  `mask &= windows_mask(x, windows, x_min, x_max)` **and** derive σ from
  `apply_error_mode(y, group.yerr, error_mode, error_value)` per group (falling
  back to unit weights when it returns `None`). The window mask is applied
  identically across all groups (one shared model over the union — WiMDA
  semantics, §2.3 of the first study).
- **SCATTER** rescales *all* uncertainties (`global_unc` **and** every
  `local_unc[gid]`) by `√(χ²/ν)` after the fit, mirroring
  `fit_parameter_model`'s single-series scatter handling (`:1858–1877`); when
  `ν < 1`, report indeterminate errors (empty dicts + message), not collapsed
  ones.
- `CrossGroupFitResult` (`:1479–1491`) gains `error_mode: str =
  ErrorMode.COLUMN.value` and `n_points: int = 0` to match
  `ParameterModelFitResult` (lets item 3 carry them into the results rows and
  lets the verdict be suppressed for none/scatter).
- The window mask must be applied to the **stored** group arrays consistently
  with how the dialog already pre-slices by `[x_min,x_max]` in
  `CrossGroupFitDialog._run_fit` (`:451–471`); choose ONE place to slice
  (prefer core, so windows are honoured even for non-GUI callers) and have the
  dialog stop pre-slicing when windows are present — recorded for the plan to
  avoid double-masking.

### 2.3 GUI port (only after the core honours them)

- Flip `_supports_error_modes`/`_supports_windows` to `True` on
  `CrossGroupFitDialog`; the inherited combo, value field and "+ Window" UI then
  render via the base-class builders (`model_fit_dialog.py:400–421,643–689`).
  Keep the `_post_rebuild_ranges_ui` override (`:133–137`) that hides the
  per-range "active" checkbox (cross-group has no per-range activity concept).
- `CrossGroupFitDialog._run_fit` (`:425–540`) passes
  `error_mode=self._error_mode()`, `error_value=self._error_value()`,
  `windows=fit_range.windows` to `global_fit_parameter_model`.
- Verdict: the base `_quality_text_for_range` already suppresses for
  none/scatter and unknown `n_points`; with `CrossGroupFitResult.n_points`
  populated, the cross-group verdict renders correctly.
- **Config persistence** `_collect_config()` (`:139–169`) /
  `_apply_existing_config()` (`:212–283`): add `"error_mode"`, `"error_value"`,
  `"windows"` keys (windows as `[[lo,hi],…]`, read back with
  `parse_fit_windows`). Lenient legacy load (missing keys → Column / no
  windows). This reaches `.asymp` via the panel's
  `_cross_group_fit_configs` serialiser (`:2450–2506`).

### 2.4 Existing guard test to update

`tests/test_cross_group_fit_dialog.py::test_cross_group_dialog_hides_unsupported_controls`
(≈`:235–265`) asserts the controls are absent. It must be **inverted** to assert
they are now present *and wired*, with a new test asserting the backend honours
them (point counts change with windows; σ changes with mode).

---

## 3. Item 1 x-uncertainty — effective variance (decision C)

### 3.1 The estimator

Ordinary least squares minimises `χ² = Σ (yᵢ − f(xᵢ;θ))² / σ_yᵢ²`, assuming x is
exact. With per-point x-uncertainty σ_xᵢ the first-order (Orear/York) correction
inflates the denominator by the x-error propagated through the local slope:

```
σ²_eff,i(θ) = σ_yᵢ² + ( ∂f/∂x |_{xᵢ; θ} )² · σ_xᵢ²
χ²(θ)      = Σ (yᵢ − f(xᵢ;θ))² / σ²_eff,i(θ)
```

References: J. Orear, *Am. J. Phys.* **50**, 912 (1982); D. York *et al.*,
*Am. J. Phys.* **72**, 367 (2004). (APS-style list reproduced in the user-guide
section.)

### 3.2 Why it fits our architecture (and ODR does not)

- The slope is evaluated by **central finite difference**
  `(f(x+h) − f(x−h)) / 2h`, `h = max(|x|,1)·1e-6` per point — two extra model
  evaluations per residual, no analytic-derivative registry needed.
- σ²_eff depends on θ, but **iminuit re-evaluates the cost every step**, so the
  weighting is self-consistent at the minimum with **no outer iteration loop**.
- It is a pure modification of the cost in `_run_parameter_model_minuit`
  (`:1694–1767`), so it **keeps every existing feature**: box limits
  (`m.limits`), multistart seeding (`fit_parameter_model:1821–1849`), error
  modes, windows, the χ²/dof/verdict pipeline.
- When σ_x ≡ 0 (the field/temperature/run case, or the toggle off) σ²_eff = σ_y²
  **exactly** → byte-identical to today. This is the regression-safety
  guarantee.
- **ODR / total least squares** (`scipy.odr`, ODRPACK) was rejected: ODRPACK has
  no box constraints, and our parameter models rely on bounds (`Tc>0`, α,β≥0,
  `A_hf>0`, …); it is also a second engine to reconcile with `ParameterSet`/
  multistart. Effective variance gives ~the same answer for σ_x ≲ σ_y/slope at a
  fraction of the integration cost.

### 3.3 Plumbing

- New optional arg on `fit_parameter_model` and `_run_parameter_model_minuit`:
  `xerr: NDArray | None = None`. When present and finite, the minuit cost is a
  **custom cost** (not `iminuit.cost.LeastSquares`, which has no σ_x hook):
  `cost(θ) = Σ resid²` with `resid = (y − f(x;θ)) / sqrt(σ_y² + slope² σ_x²)`.
  `chi_squared = m.fval`, `ndof = N − n_free` as today.
- The dialog gains a checkbox **"Account for x uncertainty"** in the data
  section, enabled only when `x_key.startswith("param:")` and x-errors are
  finite; default unchecked. Read in `_run_fit` and passed through as
  `xerr = self._xerr if enabled else None`.
- σ_x for `param:<name>` x = `row.errors.get(name)` gathered alongside x_values
  in `_open_model_fit_dialog` (`:2194`), passed into the dialog as `x_errors`.
- **Persistence**: a per-`ParameterModelFit` (or per-range) boolean
  `use_x_errors` serialised in `_serialize_model_fits` (legacy → False). Treat as
  a save/reload surface from day one.
- **Plot bars**: `_refresh_plot` draws horizontal error bars from σ_x whenever
  `x_key` is a `param:*` key (independent of the fit toggle); GLE export likewise
  emits the x-error column. Vertical y-bars already exist.

### 3.4 Divergence

| # | Topic | WiMDA | Asymmetry |
|---|---|---|---|
| E1 | x-uncertainty in param-vs-param fits | x exact (no concept) | optional effective-variance weighting (default off → identical to WiMDA); horizontal x-bars shown |

---

## 4. Item 3 — Cross-group outputs → results table (recursion)

### 4.1 What a cross-group fit produces (seams)

`_run_cross_group_model_fit` (`fit_parameters_panel.py:2229–2323`) builds
`ParameterGroupData` per selected group (each with a `group_variable_value` — the
*orthogonal* coordinate from `_group_variable_value_for_rows`, e.g. group
temperature for a field-fit) and runs `CrossGroupFitDialog`. On success the
result is a `CrossGroupFitResult` (`parameter_models.py:1479–1491`):

- `global_parameters` / `global_uncertainties` — shared across groups (one set).
- `local_parameters[gid]` / `local_uncertainties[gid]` — one `ParameterSet` per
  group.
- `chi_squared`, `reduced_chi_squared`.

`_apply_cross_group_fit_to_groups` (`:1273–1322`) currently only attaches a model
*overlay* (`ParameterModelFit`) to each group; it does **not** surface the
local/global parameter *values* as trendable rows. That is the item-3 gap.

### 4.2 The recursion model (decision A + B)

After a successful cross-group fit, build a **new `_GroupFitData` series**
("*Model fit results*") and register it in `self._group_fit_results` under a
synthetic batch id, selectable via the existing group-tab / series machinery.
Its rows:

- **One `_FitRow` per group** (decision B — local params): `values` = that
  group's `local_parameters[gid]` (name→value), `errors` =
  `local_uncertainties[gid]`; the row's trend coordinate is
  `group.group_variable_value`. Provenance label = the source group's name.
- **Global params (decision B)**: the shared `global_parameters` are added to
  **every** row as constant-valued columns (so they appear in the table and are
  selectable as y/x), *and* a single global-summary representation is recorded —
  see the open design choice in §4.4.

The new series' `inferred_x_key` is the **group-variable axis** (temperature for
a field-fit, field for a temperature-fit — i.e. exactly what
`_group_variable_value_for_rows` already chose). With item 1 in place, the user
can then pick any column (a local model param, the group variable, or a global)
as the next x and fit again — recursion.

### 4.3 `_FitRow` schema for derived rows

`_FitRow` (`:143–152`) carries `run_number, run_label, field, temperature,
values, errors, combined_from, covariance`. A cross-group-output row has no run.
Minimal extension (open choice, see §4.4): map `group_variable_value` into the
slot named by the new series' x-axis (so the existing `_x_value` machinery works
unchanged), set a synthetic `run_number` (negative, unique) and a descriptive
`run_label` (the group name), and add an **`origin: str | None`** field
(`"cross_group"`) so the table/plot can mark provenance and persistence can
distinguish derived rows. `combined_from`/`covariance` stay `None`.

`_GroupFitData` already serialises via `_serialize_group_fit_results`
(`:1324–`); the new series persists for free **iff** `_FitRow`'s new `origin`
field is added to the row (de)serialiser — a save/reload surface.

### 4.4 Open design choices (to settle at the post-study checkpoint)

1. **Global-param representation** — (a) constant extra columns on the local
   rows (visible + selectable, but constant so not self-trendable); (b) an
   additional one-row global-summary entry in the same series; (c) accumulate
   one global-summary row per cross-group fit into a separate
   "*global summary*" series across successive fits (true global recursion, but
   needs an x that distinguishes successive fits). Recommendation: **(a) + (b)**
   now; (c) as a follow-on.
2. **`_FitRow` extension** — add `origin` only (reuse field/temperature slots
   for the group coordinate) vs. add a generic `x_value`+`x_label`. Reusing the
   existing slots keeps `_x_value`/labels unchanged; recommendation: **`origin`
   only**.
3. **Series identity & overwrite** — re-running the same cross-group fit should
   replace, not duplicate, its results series (key on
   `param_name + x_key + group set`, like `_cross_group_config_key`).

### 4.5 WiMDA correspondence

WiMDA's second-level **Model Fit Table** (`ModelFitTableUnit.pas`, 198 lines) is
a bare auto-saving text editor into which `UpdatePar` (`Model.pas:1395–1436`)
appends one row per model fit (χ²ᵣ + value/error pairs) — load/save/print only,
no analysis (model-function-parity comparison §2.5). Routing outputs into the
existing trend table is strictly more capable: the rows are immediately
re-trendable rather than parked in a separate widget.

---

## 5. Divergence summary (both behaviours stated)

| # | Topic | WiMDA | Asymmetry (this project) |
|---|---|---|---|
| E1 | x-uncertainty in fits | x exact, no concept | optional effective-variance (Orear/York), default off; reduces exactly to OLS when off; horizontal x-bars on plot |
| E2 | Cross-group error modes / windows | n/a (WiMDA has no cross-group fit) | `global_fit_parameter_model` honours `error_mode`/`error_value`/`windows`; SCATTER rescales global+local σ |
| E3 | Model-fit results table | separate auto-saving text table (`ModelFitTableUnit`) | cross-group outputs become a trendable series in the same panel (recursion); single-fit export deferred |
| E4 | Arbitrary X | any column index, x exact | any fitted parameter as `param:<name>`, common-scope components, optional x-error treatment |
