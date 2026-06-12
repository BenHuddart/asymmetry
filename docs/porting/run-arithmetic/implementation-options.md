# Implementation options: run-arithmetic

## Chosen shape

A single Qt-free kernel `core/data/combine.py` with one public entry point:

```python
def combine_runs(
    runs: Sequence[Run],
    *,
    sign: int = +1,
    scales: Sequence[float] | None = None,
    run_number: int | None = None,
    label: str | None = None,
) -> Run:
    """Combine raw histograms of `runs` at the count level.

    sign=+1 → co-add; sign=-1 → co-subtract (runs[0] − Σ scale·runs[1:]).
    Period-aware (sums period_histograms via periods.sum_period_histograms),
    accumulates good frames, event-weights scalar T/field, records spread +
    a `combination` provenance block. Co-subtract per-detector arithmetic goes
    through transform.background.subtract_scaled_counts (variances add).
    """
```

`Run` carries no error array (errors are recomputed at reduction from Poisson
counts), so co-add needs no error plumbing — it sums counts and lets the normal
reduction produce errors (divergence RA2). Co-subtract is the only path that
must *carry* an error, because the difference of two count spectra is no longer
Poisson; `subtract_scaled_counts` returns the propagated per-bin error, which we
attach to the combined Run via a `combination["subtract_errors"]` payload that
the reduction consumes (see "Co-subtract error plumbing" below).

### Validation / compatibility

Hard errors (raise `ValueError`): fewer than 2 runs; mismatched bin width
(per detector, within a tolerance); mismatched detector count; empty
histograms. These are the count-level invariants — everything else (alpha,
good-bin window, deadtime) is a *grouping* concern handled at reduction and
mirrored from the first run.

The GUI co-add path keeps its existing stricter `_coadd_compatibility_error`
gate (identical full grouping signature) so interactive behaviour is
predictable; `combine_runs` itself only enforces the count-level invariants so
it stays scriptable and reusable by the reference-run subtract (where the
reference is deliberately a *different* run with its own grouping).

### t0 alignment (decision 3, RA7)

For each detector, if constituent `t0_bin`s differ, align via the same rule as
`transform.grouping.apply_grouping_aligned`: shift each detector so its local
t0 lands on a common bin (max of the constituents' t0), zero-padding the front,
then sum on the common length. Equal-t0 runs (the ISIS norm) take the trivial
path. The chosen `common_t0_bin` and any per-run shifts are recorded in
`combination["alignment"]`. The combined histograms adopt the common t0.

### Frame / exposure accumulation (W12 note)

`Run` has **no `good_frames` field**; good frames live in
`grouping["good_frames"]` (run-level) and `grouping["period_good_frames"]`
(per period). Accumulate them where they live:

- co-add: summed good frames = Σ frames over constituents;
- co-subtract: the reference is frame-scaled to the sample, so the combined
  exposure is the *sample's* frames (the reference is consumed, not added).

Per-period good frames sum per period set (reusing the periods helpers'
conventions). Deadtime tables: equal tables pass through; differing tables get
a frame-weighted mean, mirroring `periods._combined_dead_times`.

### Event-weighted metadata (W3, RA6)

Weight by good events per constituent (good frames are the available proxy for
events; use `grouping.good_frames`). Scalars written to
`metadata["temperature"]`/`["field"]` (kept scalar floats). New keys
`metadata["temperature_spread"]` / `["field_spread"]` record (min, max) — or a
single spread scalar — so inhomogeneous groups are visible. `combined_from`
kept verbatim. The nested `metadata["combination"]` block mirrors
`metadata["simulation"]`:

```python
combination = {
    "method": "coadd" | "subtract_reference",
    "sign": +1 | -1,
    "constituents": [{"run_number", "source_file", "good_frames",
                      "temperature", "field"}, ...],
    "scales": [...],            # frame-ratio scales applied (subtract)
    "alignment": {"common_t0_bin", "shifts": {run: bins}},
    "negative_count_bins": int, # guard (RA5)
    "reference_run_number": int | None,   # subtract
}
```

### Co-subtract error plumbing

`subtract_scaled_counts(sample, scale·reference)` gives per-detector difference
+ error. We cannot reduce a difference spectrum as Poisson, so the combined Run
stores the propagated detector errors and the reduction for a
subtraction-combined run uses them instead of `√counts`. Cleanest minimal seam:
store grouped (or per-detector) difference errors under
`combination["subtract_errors"]` and have the GUI reduction / `combine_runs`'s
own `reduce` helper consume them. Because the reference-run subtract operates on
*grouped* forward/backward counts (the chokepoint's natural granularity), the
simplest correct implementation subtracts at the grouped-count level (resolve
reference → group both → `subtract_scaled_counts` on F and B → combine to
asymmetry with the propagated errors), producing the combined dataset directly,
and *also* stores summed-then-differenced histograms on the Run for downstream
display. (Detail finalised in code; the principle — one chokepoint, variances
add — is fixed.)

### GUI wiring (W11)

- `_coadd_datasets` → builds a `Run` via `combine_runs(sign=+1)` then reduces
  it (so the combined row has real histograms); keeps negative
  `_next_combined_id`, `_combined_datasets`, `_combined_source_datasets`.
- `rebuild_combined_dataset` → same kernel (silent recompute on `.asymp` load).
- New `_subtract_reference_run` action (flat, single-run selection): opens a
  small picker of the other loaded runs; on accept resolves the reference via
  `resolve_background_reference` and builds a subtraction-combined row.
- The combined-row info dialog already lists constituents via `combined_from`;
  extend it to show the weighting/scale and spread.

## Alternatives considered (rejected)

- **Keep curve-mean co-add, just fix errors** — rejected: leaves
  `histograms=[]`, so the combined row still can't be regrouped/fitted. Misses
  the whole point.
- **Separate co-subtract implementation** — rejected by F9: must reuse
  `subtract_scaled_counts` + `resolve_background_reference`.
- **Second period-summing routine** — rejected by W12: reuse
  `periods.sum_period_histograms`.
- **New schema block for combined curves** — rejected by W1/W11: definitions
  already round-trip via `combined_datasets`; recompute on load.
- **Symmetric N-run co-subtract now** — deferred (decision 1).
</content>
