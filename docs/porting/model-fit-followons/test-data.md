# Test data: Model-fit follow-ons

Oracles are synthetic (the features are GUI/numerics, not new physics formulas);
the one real-data check reuses the EuO ν(T) trend fixture already frozen in
`tests/test_wimda_model_function_parity.py`. Compute oracle constants in the
test with numpy/scipy where noted, do not hand-transcribe.

## 1. Item 1 — arbitrary X (param-vs-param)

### 1.1 x-key encoding round-trip
`_normalize_x_key("param:lambda") == "param:lambda"` (must NOT collapse to
`"run"`); `"field"/"temperature"/"run"` unchanged; junk → `"run"`.
`_x_value(row, "param:lambda")` returns `row.values["lambda"]`, and the matching
σ from `row.errors["lambda"]`; missing key → NaN.

### 1.2 Component scope degrade
`component_names_for_x("param:lambda") == component_names_for_x("run")`
(both `{"common"}`-scope) — pins that arbitrary x offers common components only.

### 1.3 EuO λ vs ν real-data check (verification target)
Using the frozen EuO trend (PSI runs 2928–2943, per-run
`Oscillatory*Exponential + Constant`): trend the relaxation rate λ against the
precession frequency ν (both fitted per run, both in `row.values`). Assert the
fit runs, produces finite parameters, and that the x-array equals the per-run ν
values (not temperature). Qualitative (no golden numbers) — this exercises the
whole arbitrary-X path on real fitted parameters.

### 1.4 Persistence round-trip
Build a `ParameterModelFit(x_key="param:nu", …)`, serialise via
`_serialize_model_fits`, reload via `_deserialize_model_fits`; `x_key` survives.
Legacy state without the key loads with the documented default.

## 2. Item 1 — effective-variance x-uncertainty

### 2.1 σ_x = 0 is byte-identical to OLS
Linear data, finite σ_y, `xerr = zeros`. `fit_parameter_model(..., xerr=0)` must
return the *same* parameters and uncertainties (≲1e-12) as `xerr=None`. This is
the regression-safety oracle.

### 2.2 Effective-variance vs an independent reference
Synthetic line `y = m x + b`, `m=2, b=1`, seed 7: `x = linspace(0,10,15)`, add
`σ_y=0.3` and `σ_x=0.4` Gaussian scatter. For a straight line `∂f/∂x = m`, so
σ²_eff = σ_y² + m²σ_x² is *constant*; the effective-variance fit must equal a
plain weighted fit with that constant σ_eff (independent closed form) to ≲1e-6.
Also assert the effective-variance parameter errors are **larger** than the
σ_x-ignoring fit's (x-uncertainty inflates the variance).

### 2.3 Central-difference slope sanity
For a nonlinear model (`PowerLaw a=2,n=1.5`), assert the finite-difference slope
used internally matches the analytic `a·n·|x|^(n-1)` to ≲1e-5 on a positive grid
(guards the `h` choice).

## 3. Item 2 — cross-group error modes + windows

### 3.1 Windows honoured (point counts)
Two groups, `x = linspace(0,100,21)` each. Fit with `windows=[(0,40),(60,100)]`;
assert `CrossGroupFitResult.n_points` equals the count inside the union across
both groups (i.e. excludes (40,60)), and differs from the no-window count.

### 3.2 Error mode changes σ (and SCATTER rescales global+local)
- PERCENT vs COLUMN on the same two groups give different `reduced_chi_squared`
  and different uncertainties.
- SCATTER: assert χ²ᵣ-driven rescale is applied to **both** `global_uncertainties`
  and every `local_uncertainties[gid]` (each scaled by the same `√(χ²/ν)`); with
  ν<1 the uncertainties are empty and the message flags indeterminate errors.

### 3.3 Degenerate two-identical-groups equality (verification target)
Two groups with *identical* `(x, y, yerr)` and all-global params: the cross-group
COLUMN/SCATTER fit's global parameters must equal the single-series
`fit_parameter_model` result on one copy (same mode) to ≲1e-6 — proves the
cross-group error/window path reduces to the single-series path in the
degenerate case.

### 3.4 Config persistence round-trip
`CrossGroupFitDialog._collect_config()` → `_apply_existing_config()` preserves
`error_mode`, `error_value`, `windows`; legacy config (no keys) loads as
Column / no windows. Also through the panel's
`_serialize_cross_group_fit_configs` / `_deserialize_cross_group_fit_configs`.

## 4. Item 3 — results-table recursion

### 4.1 Local-param rows
After a 3-group cross-group fit (local params), assert the new *Model fit
results* series exists in `_group_fit_results` with **3 rows**, each row's
`values` carrying the group's local parameters and `errors` the local
uncertainties, the row coordinate equal to that group's `group_variable_value`,
and `origin == "cross_group"`.

### 4.2 Globals present
Assert the shared global parameters appear (constant) on every row and/or the
global-summary row per the chosen representation (§4.4 of comparison.md).

### 4.3 Recursion round-trip (verification target)
Take the new series, run a *second* trend fit (e.g. `Linear`) on a local model
parameter vs the group coordinate; assert it produces a finite result — "trend
the outputs of a trend fit".

### 4.4 Series persistence round-trip
`get_state()` → `restore_state()` preserves the derived series and its rows
(including the new `_FitRow.origin` field); legacy state (rows without `origin`)
loads with `origin=None`.

### 4.5 Series overwrite, not duplicate
Re-running the same cross-group fit (same param/x_key/group set) replaces the
existing results series rather than appending a second one.

## 5. Suite baseline
Full `validate` stays green; total passed ≥ **1823** (the model-function-parity
baseline) and rises by the new tests above. GUI tests under
`QT_QPA_PLATFORM=offscreen`.
