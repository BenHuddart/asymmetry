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
