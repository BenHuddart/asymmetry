# Comparison: correction order vs. alpha estimation

Cross-program comparison of where deadtime, background, and grouping sit relative
to the `alpha` estimate, and the physics that fixes the required order.

## Current Asymmetry behaviour (the divergence)

| Path | Deadtime | Grouping | Background | alpha applied |
| --- | --- | --- | --- | --- |
| `reduce_grouped_asymmetry` (reduction) | yes (`reduce.py:121`) | yes (`:133`) | yes (`:184`) | yes (`:233`) |
| `estimate_alpha_detailed` (calibration dialog) | **no** | yes (`grouping.py:547`) | **no** | n/a (estimated) |
| `_estimate_run_alpha` (per-run policy) | **no** | yes (`profiles.py:1183`) | **no** | n/a (estimated) |

`group_forward_backward` (`grouping.py:500`) sums raw `Histogram.counts`; the
calibration dialog's grouping dict (`alpha_calibration_dialog.py:452`) carries
only `groups`, `forward_group`, `backward_group`, `excluded_detectors` — it is
not even handed the deadtime/background policy, so it cannot apply them.

## Required order (approved direction)

```
raw per-detector Histogram.counts
  → deadtime correction   (per-detector, nonlinear, BEFORE grouping)
  → t0 alignment          (bin relabelling; before grouping when per-detector t0)
  → grouping              (sum F, sum B)
  → background subtraction (on grouped F, B; reference run pre-corrected — see below)
  → [ alpha ESTIMATE reads HERE ]     ← the estimator input seam
  → binned_fb_asymmetry(alpha)        (reduction only)
```

The estimator must consume the spectra one step before the asymmetry is formed —
i.e. the same corrected F/B the reduction uses.

## Why each correction must precede the estimate

### Deadtime — yes, before grouping

- `alpha` is solid angle × efficiency × geometry for the *signal* channel. F and
  B essentially never see the same rate (different solid angle, sample-dependent
  absorption, deliberately asymmetric LF coverage), so deadtime compresses the
  hotter group more and pulls raw `ΣF/ΣB` toward unity.
- The sums are exponentially weighted toward early times where rates — and hence
  deadtime losses — are highest, so the bias concentrates exactly there.
- Deadtime is nonlinear in `N`, so correcting a group sum with an effective `τ`
  ≠ summing per-detector-corrected counts, and each detector has its own `τ`.
  **Correct per detector, on raw histograms, before grouping.** Asymmetry's
  existing `deadtime → group` order is right; only the estimator is not fed from
  it.

### Background — yes, before the estimate

- The flat pedestal (beam-uncorrelated events, dark counts, cosmics) shares
  neither the muon lifetime nor the precession, and its F/B ratio is set by
  ambient geometry, not positron efficiency.
- Estimating on `(signal + pedestal)` yields an `alpha` that balances totals;
  after subtraction the asymmetry gains a constant offset of order `δα/2` with
  `δα ≈ b·(r_bg/r_sig − 1)` (`b` = background fraction, `r` = F/B ratios).
- The pedestal fraction grows like `exp(t/τ)` relative to signal, so a
  totals-calibrated `alpha` is a time-weighted compromise — precisely the
  "calibration looks centred, reduction doesn't" symptom. A wTF calibration
  whose reduced asymmetry is not centred is a failed calibration **by
  definition**.

### Deadtime before background — physical necessity, not convention

- Background events occupy detector deadtime just as signal events do; the
  deadtime correction reconstructs the true incident rate *including* background,
  and subtraction is only valid in that linearised space.
- Subtract-then-correct **under-corrects** by a rate-dependent amount (you have
  removed counts that were causing the deadtime). Deadtime is nonlinear,
  subtraction is linear → the linear op must live inside the linearised space.
- If the pedestal is estimated from the corrected data itself (pre-t0 window or
  late-time tail fit on corrected grouped counts) it is automatically in
  corrected units. A **reference-run** pedestal must be corrected independently
  (below).
- t0 alignment commutes with deadtime (bin relabelling) but must precede grouping
  when detectors carry individual t0s.

## Per-method sensitivity (all three still require corrected input)

| Method | Background sensitivity | Deadtime sensitivity | Notes |
| --- | --- | --- | --- |
| `ratio` (`ΣF/ΣB`) | real but count-weighted toward early times (diluted) | worst of the three | most forgiving — likely why the bug went unnoticed |
| `diamagnetic` (min `Σ(A/σ)²`) | absorbs pedestal baseline into `alpha` with high confidence | nonlinear in counts → sensitive | needs honest `σ`: raw-count Poisson variance **plus** pedestal-estimate variance, not post-subtraction counts |
| `general` (lifetime-corrected flatness) | near-meaningless un-subtracted: `×exp(t/τ)` makes a flat pedestal diverge at late t | degrades if deadtime distorts the early exponential | most fragile |

## Reference-run background: required pre-correction

A scaled empty-sample reference must be treated identically to the sample run
*before* subtraction:

1. **Deadtime-correct with its own rates** — an empty holder stops fewer muons,
   so its correction factor differs; cannot reuse the sample's factor nor
   subtract-raw-then-correct (nonlinearity).
2. **t0-align to its own t0s** — t0 drifts between runs; a misaligned reference
   smears sharp prompt/pedestal structure into the good-bin region.
3. **Apply the identical grouping** — same detector membership and dead-detector
   masking; a differently-grouped reference compares different solid angles.
4. **Scale bin-by-bin per incident muon** — accumulated frames (or integrated
   beam / muon counts where available), with an optional user trim factor.
   Propagate the scaled reference's Poisson variance in quadrature into the
   grouped errors (the `diamagnetic` weights depend on it).

Physical caveat worth a UI tooltip: a reference run subtracts not only the flat
pedestal but any *signal-like* background (muons stopping in the holder/cryostat
that decay with `τ_μ` and may precess). That is desired — but it makes estimating
`alpha` *after* subtraction essential, because the pre-subtraction spectra carry
a second asymmetric signal with its own effective `alpha`.

## Reference-program behaviour

| Program | Deadtime vs grouping | alpha input | Background treatment |
| --- | --- | --- | --- |
| **Mantid** | `MuonPreProcess` deadtime + time-zero + rebin **before** grouping | `AlphaCalc` runs on the pre-processed grouped workspaces (corrected-counts-first) | largely fits flat background rather than subtracting pre-asymmetry, so Mantid is mainly the *deadtime* precedent |
| **musrfit** | file-based deadtime before asymmetry | asymmetry formed from **background-subtracted** histograms `(N_F − bkg_F)`, `(N_B − bkg_B)`; `alpha` (and `β`) determined against subtracted histograms by construction (fitted or fixed) | cleanest precedent for the background half |
| **WiMDA** | deadtime and background applied per current reduction settings | estimates `alpha` from the spectra **as corrected by the reduction settings** (deadtime + background applied) | the `diamagnetic` port's reference — Asymmetry has diverged here |
| **Asymmetry (today)** | deadtime before grouping in reduction only | estimates on **raw** grouped counts | subtracts in reduction only — not seen by the estimator |

**Conclusion:** corrected-counts-first for `alpha` estimation is the established
convention across all three references. Asymmetry's estimator-on-raw-counts is a
divergence from WiMDA specifically (the `diamagnetic` method's source), and
should be recorded as a fidelity bug, not merely a code bug.

## Divergences / caveats to carry into implementation

- **D1** — Mantid sidesteps the pedestal question (fit not subtract), so it
  validates only the deadtime half of the ordering; use musrfit/WiMDA for the
  background half.
- **D2** — WiMDA's estimate follows the *live* reduction settings, which means
  changing deadtime/background after estimating should invalidate `alpha`
  (staleness — see UI direction).
- **D3** — `general` method on any residual un-subtracted pedestal is unstable;
  verification must include a background-heavy case for this method
  (`verification-plan.md`).
