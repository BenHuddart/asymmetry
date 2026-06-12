# Comparison: run arithmetic — WiMDA vs Asymmetry

WiMDA source verified at `$WIMDA_SRC/src/muondata.pas` (co-add/co-subtract in
the `loadrun` path, lines 2418–2491) and `$WIMDA_SRC/src/BatchFit.pas`
(in-batch consumption, lines 300–324). Physics primary source: Blundell, De
Renzi, Lancaster and Pratt, *Muon Spectroscopy: An Introduction* (OUP, 2022) —
"the textbook" below.

## WiMDA mechanism (verified)

`loadrun` loads a "master" run into `mrun`; each subsequent run arrives as
`thisrun` with `coadd`/`cosub` flags. The arithmetic:

```pascal
if cosub then cosign := -1 else cosign := 1;          { line 2418 }

{ frame / spill accumulation carries the sign }
totalframes := totalframes + cosign * mrun.info.framestotal;   { 2429 }

{ non-RG, non-period: per-detector, per-bin, with ACC tshift }
mrun.histos[i, ii] := mrun.histos[i, ii] + cosign * thisrun.histos[i, ii + tsh]; { 2441 }

{ RG mode (no periods): R and G histograms summed separately }   { 2455-2458 }
{ period mode: periods.cycles, periods.frames[1..8] and          }
{   periods.periodhistos[k, i, ii] all summed with cosign         } { 2469-2490 }
```

The `tsh = cgrp.tshift[i]` term is the **ACC alignment shift** computed at
lines 2364–2416 when *Grouping → ACC shift* is enabled: per detector, WiMDA
finds the prompt-peak bin (max logarithmic derivative) and shifts detectors
whose peak deviates by > 8 bins from the mean, so detectors are time-aligned
before the bin-by-bin sum. With ACC off, `tshift[i] = 0` and the sum is a plain
detector-wise add.

In-batch co-add (`BatchFit.pas:300–324`) reuses exactly this machinery: with
*batch co-add* enabled it loads the master, then loads the next
`CoAddUpDown.position` runs with `coadd := true` before fitting — i.e. there is
one co-add implementation, consumed by both the interactive and batch paths.
The combined-run label is `"<master>+<next>"` (e.g. ALC averaging,
`muondata.pas:787`: `runnumber.caption + '+' + runstr`).

## Key divergences

| # | Topic | WiMDA | Asymmetry | Rationale |
|---|---|---|---|---|
| RA1 | **Combine domain** | Raw counts summed; reduce afterwards | Same — `combine_runs` sums histograms, reduction happens after | Correctness: errors and nonlinear corrections must see total statistics. Replaces Asymmetry's old curve-mean co-add. |
| RA2 | **Co-add errors** | Implicit — Poisson errors recomputed from summed counts at reduction | Same: errors come from summed counts via the normal reduction (`compute_asymmetry_with_count_errors`), never from averaging input errors | The old curve-mean path divided summed errors by N; wrong at low counts (see RA8). |
| RA3 | **Co-subtract errors** | `thisrun.histos` subtracted **without touching errors** (study divergence D7, already recorded for `subtract_scaled_counts`) | Variances add: `√(a + s²·r)` via `subtract_scaled_counts` | Independent Poisson channels; WiMDA underestimates the subtracted-spectrum error. |
| RA4 | **Subtraction scale** | `cosign·thisrun.histos` — **no frame scaling** between the two runs (frames merely accumulate with sign) | Reference-run subtract scales the reference by the good-frame ratio sample/reference (WiMDA's *background-run* exposure scale, reused here) | A co-subtract of two unequal-exposure runs should match exposures; the frame-ratio scale is the physically meaningful normaliser and is already the chokepoint's contract. Plain (scale=1) subtraction of equal-exposure runs is the special case. |
| RA5 | **Negative counts** | Allowed silently (a bin can go negative after `cosign·`) | Guard: warn + clip the variance radicand to ≥ 0; the difference array keeps its (possibly negative) value but provenance records that bins went negative | Negative *expected* counts are unphysical and break downstream Poisson reduction; surfacing them is the defensible choice. |
| RA6 | **T / field metadata** | Load/coadd routine keeps the **master run's** scalar temperature/field (no averaging in `muondata.pas`) | Event-weighted scalar mean over constituents, weighted by good events; spread recorded under `temperature_spread`/`field_spread` (W3) | The master-only value misrepresents a combined group; the event-weighted mean is the defensible summary, and the spread lets users spot inhomogeneous groups. The brief's "event-weighted averaging" is thus an *improvement over*, not a copy of, WiMDA's code. |
| RA7 | **t0 alignment** | Optional ACC peak-shift (`tshift`), instrument-gated | Per-detector align to a common t0 via `apply_grouping_aligned` (max-t0 convention), always applied where t0 differs; recorded in provenance | Same intent (align before summing) with a deterministic, source-agnostic rule; equal-t0 ISIS data is unaffected. |
| RA8 | **Combined-row capability** | Combined `mrun` is a normal run — fully analysable | Combined Run now carries histograms + grouping → regroup/deadtime/count-fit/MaxEnt all work (was impossible with `histograms=[]`) | The motivating correctness gap. |
| RA9 | **Period summation** | `periodhistos[k,i,ii]` summed in-place with `cosign` | `core/io/periods.sum_period_histograms` per period set; writes existing `period_histograms`/`period_good_frames`/`period_dead_time_us`/`period_mode` keys (W12) | One period-summing implementation; no parallel path. Subtraction across period runs is out of the reference-run scope (recorded follow-on). |
| RA10 | **Label** | `"<a>+<b>"` | `combined_from` list kept verbatim; `run_label` is `" + "`-joined; subtraction label `" − "`-joined | Trend panel + GLE export consume `combined_from`; display label distinguishes add vs subtract. |

## Quantified co-add correction (low-count pair)

The old curve-mean co-add and the new count-sum co-add agree in the
**asymmetry value** for identical grouping (both linear in counts at α = 1),
but differ in the **error bars** whenever the constituent runs have unequal
statistics, and in *any* nonlinear correction. For two runs with forward/back
totals (F₁,B₁) and (F₂,B₂):

- count-sum reduces (F₁+F₂, B₁+B₂) → one Poisson error from the pooled counts;
- curve-mean takes ½√(σ₁² + σ₂²), which only equals the pooled error when the
  two runs have identical counts.

**Quantified (synthetic, fixed seeds, relaxing-TF cosine).** Co-adding a
low-statistics run with one carrying 10× the events, the old curve-mean error
bar **over-estimates** the combined error by **53 %** (median over the good
window) relative to the correct pooled Poisson error. For two equal-statistics
runs the two routes agree to 0.1 % (ratio 0.999), confirming the divergence is
purely the unequal-statistics term and the value itself is unchanged at
α = 1. This is the headline number the brief asks for; it is quoted in the
user-guide page. (The synthetic pair is the controlled oracle; the corpus pair
in [test-data.md](test-data.md) exercises the same path on real data behind the
`$ASYMMETRY_*` env gate.)

## When to use which (for the user guide)

- **Co-add** — same physical measurement repeated (more statistics). Sum
  counts; the result is one higher-statistics run.
- **Co-subtract (reference run)** — remove a *signal* present in a reference
  exposure: laser-OFF from laser-ON in photo-µSR, or a known-empty/background
  reference. Frame-scaled, variances add.
- **Background-run correction** (separate, already shipped) — subtract a
  scaled background *as a reduction step* (does not create a combined dataset);
  use when the background is a steady detector floor, not a co-measured signal.
  Shares `subtract_scaled_counts` arithmetic with co-subtract.
</content>
