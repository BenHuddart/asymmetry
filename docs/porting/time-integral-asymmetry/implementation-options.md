# Time-integral asymmetry — implementation options

## Goal

Add a scriptable, Qt-free time-integral asymmetry observable to
`asymmetry.core`, assemble it across a series of runs into a field/temperature
scan, and surface the scan in the GUI — reusing existing primitives
(`compute_asymmetry`, good-bin windowing, `periods.py`, `FitSeries` ordering,
the trend-plot panel) rather than reimplementing them.

The behavioural contract is **Mantid's `PlotAsymmetryByLogValue`** (alpha-aware,
Integral + Differential types, single time window, red/green, sample-log x-axis),
with **WiMDA's count-integral as the default *Integral* formula**.

## The two layers

1. **Per-run reduction** (core transform): one run → `(value, error)` scalar.
2. **Series assembly** (representation/series): N runs → ordered scan curve
   `(x, y, σ)` with `x ∈ {field, temperature, run}`.

Keeping these separate matches the existing architecture and lets the scalar be
used on its own (e.g. a single-run "integral asymmetry" readout) as well as in a
scan.

## Option A — core transform + field-scan series, reusing FitSeries (chosen)

### Layer 1 — `core/transform/integral.py` (new)

```python
def integrate_asymmetry(
    forward, backward, *, alpha=1.0, time=None,
    t_min=None, t_max=None, method="integral",
) -> tuple[float, float]:
    """Reduce a run's grouped counts to one (value, error).

    method="integral":    sum counts in [t_min, t_max], then
                          (F_int − α·B_int)/(F_int + α·B_int)   (Mantid Integral
                          ≡ WiMDA count-integral with α)
    method="differential": compute_asymmetry per bin, then mean over the window
                          (Mantid Differential)
    """
```

- Reuses `compute_asymmetry` (Integral path: call it on the two summed scalars;
  Differential path: call per-bin then average) so the **error model is shared**
  with the rest of the app (Mantid-compatible) — no second error formula to drift.
- `method="integral"` is the default and is the WiMDA/Mantid-Integral behaviour
  **with alpha applied** (WiMDA omits alpha; we keep it because the kernel and the
  rest of Asymmetry are alpha-aware, and α=1.0 reproduces WiMDA exactly).
- Window defaults to the dataset's good-bin range when `t_min/t_max` are unset
  (matches Mantid "full range if unset" and WiMDA "full good-bin window").
- A thin `integrate_dataset(dataset, ...)` wrapper accepts a `MuonDataset`/`Run`
  and pulls grouping + good-bin window + alpha, so callers don't re-derive them.

### Layer 2 — field-scan series

A scan is a series of runs each reduced to one point and ordered by an
independent variable — which is exactly what `FitSeries` already models
(`order_key ∈ {field, temperature, run}`, `sort_members()`). Two sub-options:

- **A1 (recommended): a dedicated `FieldScan` representation/series.** Add
  `RepresentationType.FIELD_SCAN_INTEGRAL = "field_scan_integral"` (domain
  `"scan"`) and register it in `factory.py`. The per-run `compute()` returns a
  single-point `MuonDataset` (`time=[x_value]`, `asymmetry=[integral]`,
  `error=[σ]`); the series collects the points and orders them with the existing
  `FitSeries` machinery. Recipe = `{t_min, t_max, method, alpha, period_mode,
  log_function}`. The scan curve is the assembled series, persisted recipe-only
  like every other representation.
- **A2: reuse `FitSeries.results_by_run` directly**, storing the integral as a
  pseudo-parameter and letting `fit_parameters_panel.py` trend it. Less code, but
  conflates "fitted parameter" with "directly-computed observable" and muddies the
  schema. Prefer A1; fall back to A2 only if a new representation type proves too
  heavy.

### Red/green

For dual-period runs, reduce the period selected by `period_mode` (reusing
`periods.select_period` / `combine_period_asymmetry`), mirroring Mantid's
Red/Green/diff/sum and WiMDA's RG box. Default `period_mode="red"`.

### Differential-of-scan (WiMDA `dA/dB`)

Offer an **optional series-level transform** `differentiate_scan(x, y, σ)` →
`dA/dx` (forward difference, quadrature error), gated like WiMDA on a max
x-gap. This is a view on the assembled scan, not part of the per-run reduction.

**Pros**: every primitive already exists; shared error model; scriptable core;
GUI reuses the trend panel; α=1.0 reproduces WiMDA, α-aware matches Mantid.
**Cons**: a new representation type + domain (`"scan"`) is the largest new
surface; the GUI trend panel may need a "scan" domain branch.

## Option B — GUI-only ALC table (WiMDA-faithful)

Replicate WiMDA literally: batch-reduce a run list to a text fit-table, read it
back, plot. **Rejected**: violates the core-first invariant (logic in the GUI),
not scriptable, and duplicates the series machinery Asymmetry already has.

## Option C — fit-parameter trend only (musrfit-style)

Don't add an integral observable; tell users to fit each run and trend the
amplitude. **Rejected as the feature**: it is precisely what the corpus testing
found *insufficient* (it requires a model fit per run and does not give the
model-free integral that ALC/repolarisation analysis expects). Keep it as the
existing alternative, not the port.

## Scope split

- **This port (Phase 1)**: Layers 1 + 2 (Option A) — the observable and the scan.
  Integral + Differential methods; red/green; field/T/run ordering; optional
  `dA/dx`. Validated against Mantid `PlotAsymmetryByLogValue` and WiMDA ALC.
- **Follow-up (Phase 2)**: ALC **baseline subtraction + peak fitting** (Mantid's
  ALC interface steps 2–3). This is the `alc-avoided-level-crossing` candidate and
  *depends on* Phase 1; do not attempt it here.

## Validation rules (boundaries)

- `t_min < t_max`; both within the data's time span → else `ValueError`.
- `method ∈ {"integral", "differential"}`; `period_mode` validated by
  `periods.resolve_period_index`.
- A run lacking the chosen `order_key` log value (no field/temperature) →
  excluded from the scan with a recorded reason (mirror Mantid's per-run skip),
  not a hard failure of the whole scan.
- `alpha > 0`.

---

# GUI integration — chosen design (2026-06-07)

The core observable is implemented (see [README.md](README.md)). This section
records the GUI decisions taken with the maintainer (see
[gui-presentation.md](gui-presentation.md) for the WiMDA/Mantid UX study that
informed them) and the implementation plan.

## Decisions

| Question | Decision |
| --- | --- |
| **Entry point** | A **main-window ALC-mode toolbar toggle**, enabled only when the F-B asymmetry view is active. Enabling it **swaps the Fit and Parameters docks** for bespoke ALC widgets (build panel + scan view). Reuses the rep's grouping/α/period and the fit-range control. *(Refined 2026-06-07 — see the GUI rework note below; the earlier "checkbox on the Batch tab" framing was wrong.)* |
| **Depth (this pass)** | **Full**: build + display the scan, the `dA/dB` derivative, **and** baseline subtraction + peak fitting (resonance position/width). This absorbs the `alc-avoided-level-crossing` candidate. |
| **Options exposed** | **Integral reduction only** (the count-sum; `method="integral"`) plus a **`dA/dB` derivative** view toggle (WiMDA). The Mantid *Differential* reduction method is **not** surfaced in the UI (still available in core). |
| **Rendering** | The scan curve lives in the **trending-panel space**. **Batch** builds the scan across the series; **single** shows the selected run's integral read-out. The **main plot keeps the time spectrum** with the integration window shaded. |

## Key reuse insight

Asymmetry's **trending panel already fits composite parameter models** to an
arbitrary `(x, y, err)` curve, and the model library already ships the pieces ALC
needs. So the Mantid "baseline + peak fit" is *mostly existing machinery*:

- `core/fitting/parameter_models.py` — `fit_parameter_model(x, y, yerr, model,
  params, x_min, x_max)` fits a `ParameterCompositeModel` to any curve;
  components include **`Lorentzian`** (`a/(1+(B/B0)²)+c`, peak), **`Linear`**
  (`m·x+b`) and **`Constant`** (baseline).
- `gui/panels/fit_parameters_panel.py` — owns the x-axis selector (𝐵/𝑇/Run),
  the matplotlib canvas + mouse handlers, and the **model-fit-on-trend** flow
  (`trend_state.model_fits`).
- `gui/.../plot_panel.py` + `styles/plots.py::draw_fit_range_span` — the
  **draggable fit-range span** already used on the time spectrum; reused as the
  integration window, and (adapted) as baseline-region handles on the scan plot.
- `core/fitting/engine.py::FitEngine.fit` — domain-agnostic `(x, **params)`
  fitting if a richer engine is wanted.

So the ALC curve fit ≈ fitting a `Lorentzian + Constant/Linear` composite to the
scan with the **same** machinery the trend panel uses to fit a parameter trend.
The only genuinely new y-quantity is the **integral itself** (from
`build_field_scan`, not from per-run fit results).

## Components to build

**Core (Qt-free):**

1. `differentiate_scan` — **done** (WiMDA `dA/dB`).
2. A small **baseline helper** so a baseline can be fit on *disjoint non-resonant
   regions* (Mantid-style): `fit_scan_baseline(scan, regions, model="Linear")`
   masking to the union of `regions` before `fit_parameter_model`, returning the
   baseline curve + corrected `FieldScan`. (Refinement over a single-range fit;
   see open point below.)
3. A thin **`FieldScan` → fittable adapter** (x=`scan.x`, y=`scan.value`,
   err=`scan.error`) so the scan flows into `fit_parameter_model` / the trend
   panel unchanged.

**GUI:**

4. **Mode toggle** on the F-B asymmetry rep ("Time fit" ↔ "Integral scan / ALC").
   Stored in `recipe["scan_mode"]`.
5. **Integration window** = the existing draggable fit-range span on the time
   spectrum; in scan mode its value feeds `t_min/t_max`. Shaded as today.
6. **Batch → build scan**: in scan mode the batch action calls
   `build_field_scan(members, t_min, t_max, method="integral", order_key=<x-combo>,
   grouping_ref=<rep effective grouping>)`; **single** shows the selected run's
   `integrate_run` value.
7. **Scan view in the trend panel**: plot the scan (raw / baseline / corrected /
   `dA/dB` toggle); the y-quantity is the integral. Reuse the x-combo and canvas.
8. **ALC analysis in the trend panel**: baseline-region handles + baseline fit +
   subtract; then peak fit (`Lorentzian` [+ baseline]) via the existing trend
   model-fit; show **B₀ = resonance field** and width in the results table.
9. **Persistence**: `recipe["scan_mode"]`, integration window, `baseline_regions`,
   and the peak `model_fits` persist via the rep's `recipe`/`trend_state`
   (`representation.to_dict`); schema-version bump if a migration is needed.

## Baseline + peak relationship — decided: two-step (Mantid-faithful)

Fit a baseline on user-marked non-resonant **regions**, subtract to produce a
corrected curve, then fit the **peak** on the corrected curve. This matches
Mantid's mental model, is robust when the baseline is sloped, and gives a
distinct "corrected data" view. It needs the `fit_scan_baseline(scan, regions)`
region helper (#2) and the draggable baseline-region handles (#8).

The one-shot `Lorentzian + Linear` composite fit was **not** chosen (it has no
corrected-curve view and is weaker when the baseline is only constrained away
from the resonance), though the underlying composite-fit machinery is the same.

## Sequencing

- **G1 (core): DONE** (commit 6a61448) — `core/fitting/field_scan.py`:
  `fit_scan_baseline(scan, regions, model)` (region-masked baseline fit +
  subtract → `ScanBaselineResult`), `fit_scan_model(scan, model, …)` (the
  `FieldScan`→`fit_parameter_model` adapter for the peak fit), and
  `parameter_set_for_model` / `as_composite_model` helpers. Tests in
  `tests/test_field_scan_fitting.py` (recovers a known baseline + Gaussian
  resonance). `differentiate_scan` was done in the core API phase.
- **G2 (GUI): DONE** (commits 7a23f9d → ba31a8b → b5702de). Final design after a
  maintainer correction: a **main-window "ALC mode" toolbar toggle**, enabled only
  for the F-B asymmetry view, that **swaps the Fit and Parameters docks** for
  bespoke ALC widgets (`src/asymmetry/gui/panels/alc_panel.py`):
  - `ALCFitPanel` — integration-window echo (reuses the time-plot fit-range) +
    **Build Scan** (`build_requested`).
  - `ALCScanView` — plot + table of the scan (integral asymmetry **%** vs field).
  - `mainwindow._on_scan_requested` integrates the selected runs via
    `build_field_scan(order_key="field")`, scales to **percent (×100)**, records a
    **model-less `FitSeries`** (scan = first-class series in `ProjectModel`;
    `is_computed` gates it out of divergence + fit-clearing), and renders the scan
    view. The Fit dock uses the existing `_fit_stack`; the Parameters dock gained a
    `_parameters_stack`. Toggle auto-exits when leaving F-B. Tests:
    `tests/test_integral_scan_gui.py` (+ `is_computed` core test in
    `test_project_model.py`).
  - **Decided:** scan units = **percent** (matches the plots); ALC plots vs
    **field** (field-less runs dropped + logged). The Batch-tab checkbox first cut
    was reverted.
  - **G2b: DONE** (commit a9c138c) — the `ALCScanView` gained a control row: an
    **x-axis selector (B / T / Run)** and a **`dA/dB` derivative checkbox**
    (replace-toggle). `mainwindow._render_alc_scan` owns the per-run data
    (`_alc_scan_points`) and re-derives the display on build/options-change: map
    each run to the chosen x (drop runs missing that log), sort, scale to
    percent, optionally `differentiate_scan` (labelled %/G | %/K | %/run). Build
    now uses `order_key="run"` (include all integrable runs; the view picks the
    axis). The fit-range integration window also mirrors the regular machinery —
    draggable span + spinboxes (commit c71c9cb).
- **G3 (GUI ALC analysis): IN PROGRESS** — baseline regions + subtract + peak
  fit + results read-out. Decisions (2026-06-07): **bespoke controls in the ALC
  scan view** (call G1's `fit_scan_baseline` / `fit_scan_model`, not the trend
  panel); baseline regions defined **both** ways (draggable handles on the scan
  plot *and* an editable numeric table, kept in sync); **multiple peaks** with
  **Gaussian + Lorentzian** shapes.
  - **G3a: DONE** (commit 046ea14) — added the centred **`LorentzianLCR`** peak
    component (`f/(1+((B-B0)/Bwid)²)`, sharing `f/B0/Bwid` with `GaussianLCR` so
    the two are interchangeable). Tests: single-peak recovery + a two-peak
    composite recovering both resonance fields.
  - **G3b: DONE** (commit bab4e8f) — baseline UI: a "Baseline" group (Linear/
    Constant model + editable non-resonant regions table + Fit baseline). Regions
    shade on the plot; `fit_scan_baseline` over the region union overlays the
    baseline and stores the corrected scan. Layout = **stacked sections below the
    plot** (chosen with the maintainer).
  - **G3c: DONE** (commit 246a6bf) — peak UI: a "Peaks" group (+ Gaussian /
    + Lorentzian / − peak, an editable peaks table, Fit peaks). Builds the peak
    composite (multiple peaks, both shapes), fits via `fit_scan_model` on the
    corrected scan, overlays the total (baseline + peaks) fit, and reads off
    B₀ ± err, FWHM, and amplitude per peak. The scan data is now drawn as
    **markers only** (fits/baseline remain lines).
  - **Remaining in G3:** draggable region/peak handles on the plot (a usability
    layer over the numeric tables — the "both" choice; the table path is done).
  - Note (from review): the bespoke view duplicates the trend panel's
    plot/x-selector and will re-implement the model-fit overlay; this is the
    accepted cost of the bespoke-panel decision. The fit-range spinboxes could be
    extracted into a shared `FitRangeSpinBoxes` widget (deferred cleanup).
- **G4 (persistence):** scan/baseline/peak state in `.asymp` (+ migration, gui-smoke).
