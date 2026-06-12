# Spectral moments — study

Umbrella: [`wimda-parity-gap`](../wimda-parity-gap/README.md) · Wave B project 7 ·
Size S–M · **supersedes** the [`moments-analysis`](../candidates/moments-analysis/README.md)
candidate (status `candidate`, kept for provenance; this study is the
implementation-grade successor under the unique slug `spectral-moments`).

Date: 2026-06-12. Branch: `feat/spectral-moments`.

## What this feature is

Moments are the *quantitative consumer* of a muon field/frequency spectrum. Once
a MaxEnt or phase-corrected FFT spectrum exists, this feature reduces its
lineshape to a handful of numbers — peak field, mean field, RMS widths,
skewness, lineshape asymmetry — and exports them as a trendable series so that
`B_rms(T)`, `β(T)` etc. fit like any other parameter. The physics payoff is
direct:

- **`B_rms`** of a vortex-lattice field distribution sets the magnetic
  penetration depth, `B_rms ∝ 1/λ²` in the London limit — the headline number
  for a superconductor's superfluid density.
- **Skewness `α`** and **lineshape asymmetry `β`** encode the *shape* of the
  vortex-lattice field distribution (the long high-field tail from cores, the
  cutoff at the minimum-field saddle point) and so probe the lattice geometry
  and its disorder.
- **`B_pk` / `B_ave`** locate the line and its diamagnetic shift.

Reference physics: Blundell, De Renzi, Lancaster & Pratt, *Muon Spectroscopy: An
Introduction* (OUP, 2022) — "the textbook" below — particularly its treatment of
the field distribution `p(B)` in the mixed state and its connection to `λ`.

## WiMDA reference (the behavioural oracle)

`$WIMDA_SRC/src/Moments.pas:152–423` (`TMoment.FormShow`, `parabpkextrap`,
`Button1Click`). WiMDA computes moments of the **MaxEnt** spectrum only, in
**Gauss**, over a user `[xmin, xmax]` window with an amplitude **cutoff as a
percentage of the peak**. Full arithmetic transcription and our divergences are
in [comparison.md](comparison.md). The unit `Moments.pas` is GPL; we use it as a
*behavioural oracle* (transcribe the arithmetic, re-derive in our own idiom and
tests), never as copied code.

The moment set (WiMDA names → ours):

| WiMDA | Symbol | Meaning | Ours |
|---|---|---|---|
| `Bpk` | `B_pk` | parabolically-refined peak field | `b_pk` |
| `Bave` | `B_ave` | amplitude-weighted mean field | `b_ave` |
| `Bave-Bpk` | `⟨B_ave−B_pk⟩` | mean–peak shift | `b_diff` |
| `Brms (vs Bave)` | `√m₂` | RMS width about the mean | `b_rms_mean` |
| `Brms (vs Bpk)` | `√m₂,pk` | RMS width about the peak | `b_rms_peak` |
| `Alpha` | `α = sign(m₃)·∛\|m₃\| / √m₂` | skewness (field units → dimensionless ratio) | `skewness` |
| `Beta` | `β = (B_ave−B_pk) / √m₂,pk` | lineshape asymmetry | `beta` |
| `Points` | `n` | points inside the window above cutoff | `n_sample` |

## Decisions settled with Ben

### Pre-study scope (2026-06-12)

- **Moment unit — field-default with a selector.** Moments are reported in
  **Gauss by default** (the penetration-depth/vortex reading, matching WiMDA),
  but the widget carries a `G / T / MHz` selector and the chosen unit is recorded
  per series so a series stays unit-consistent. The core
  ([`core/fourier/moments.py`](../../../src/asymmetry/core/fourier/moments.py)) is
  **array-in / array-out and unit-agnostic**; the GUI converts the active
  spectrum's axis to the chosen unit (via [`FieldUnit`](../../../src/asymmetry/core/fourier/units.py))
  before calling it. `α` and `β` are dimensionless and invariant under the linear
  field↔frequency rescaling; only the `B_*` moments scale.
- **Run-averaging — per-spectrum members only.** A selection of *N* spectra
  produces **one computed `FitSeries` with one member per spectrum** (moments as
  the member's parameters). Run-to-run scatter and averaging are left to the
  existing trend-fit layer (the brief's guidance). WiMDA's stateful
  averaging accumulator (a running mean ± population-σ over runs you step
  through, with a Reset button) is **not** replicated — recorded as a divergence
  in [comparison.md](comparison.md) §"Run-averaging".

### Input eligibility (binding Wave-B collision directive)

Moments compute **only from lineshape-faithful spectra**: the MaxEnt
reconstruction and the phase-corrected real FFT modes (`phase_corrected` /
`phase_opt_real`). For every other display mode — `power`, `power_sqrt`,
`magnitude`, `phase_spectrum`, `cos`/`sin`/`imaginary`, raw un-phased `real`,
`burg`, `correlation` — the moments UI **greys out** with a tooltip explaining
why: squared and diagnostic lineshapes (power, magnitude, Burg, correlation) and
dispersion-mixed channels bias `B_rms` and the skewness. The eligibility test
lives at the GUI layer; the core never sees a display mode.

### Trend integration (binding W8/F10 directive)

"Send to trend" records a **computed `FitSeries` and nothing else** — rows in the
[`fit_result_summary`](../../../src/asymmetry/core/fitting/result_summary.py)
shape (`success`, `parameters` = moment columns, `uncertainties`, explicit
`field`/`temperature`/`run_label`), `canonical_model=None`, registered through
`MainWindow._add_results_series` + `_refresh_trend_panel`. The batch id is a
deterministic hash of *(moment recipe + member set)* mirroring
`_cross_group_batch_id`, so re-sending **replaces** rather than duplicates. The
series carries the `rep_type` of the originating representation. The generating
recipe (range, cutoff, unit, mode) rides `FitSeries.extra` (the PR #51 home). No
new container, no new top-level project key, no new panel.

### Persistence (binding W1 directive)

**No `schema_version` bump.** Live widget settings persist as an additive,
namespaced sub-dict inside the host panel's existing `get_state()` /
`restore_state()` (the Fourier panel's `fourier_state`); `restore_state`
tolerates its absence. `CURRENT_SCHEMA_VERSION` stays at 8.

### Implementation choices (settled with Ben, 2026-06-12)

- **Uncertainty — bootstrap (primary) + linear fallback.** When the spectrum
  carries per-point errors, uncertainties come from a seeded bootstrap (resample
  `amplitude[i] ~ N(amplitude[i], σ[i])`, recompute all moments, take the sample
  std) — the one method that propagates correctly through the nonlinear `b_pk`,
  `α`, `β` and *exposes* `b_pk`'s fragility rather than asserting it. Analytic
  linear propagation is the deterministic fallback for the linear moments; errors
  are `NaN` (greyed) when the spectrum has no error array. The bootstrap seed is
  recorded in the recipe for reproducibility.
- **GUI hosting — one shared widget in both panels.** A single
  `SpectralMomentsWidget` class is mounted in *both* the Fourier advanced stack
  and the MaxEnt panel (each feeding it via the same W15 accessor), covering
  WiMDA's canonical MaxEnt input with no duplication and ≤ one-line hooks (W10).
- **β sign — WiMDA's, cited to Brandt + the textbook.** `β = (B_ave −
  B_pk)/√m₂,pk > 0` for the high-field-tailed mixed-state lineshape (mean above
  peak), matching WiMDA and the skewness sign. Primary citation: E. H. Brandt's
  vortex-lattice field distribution `p(B)`; secondary: the textbook's mixed-state
  field-distribution treatment. Pinned in [comparison.md](comparison.md) §3 and
  the user guide.

## The B_pk fragility caveat (stated plainly)

`B_pk` is the **weakest member of the set**. It is a parabola fitted to the five
spectral points around the discrete maximum; on a noisy or near-flat spectrum the
maximum can hop between bins and the parabolic vertex can swing well outside the
five-point span. Everything derived from it inherits the fragility — `b_diff`
and, especially, `β`, which divides by `√m₂,pk` (itself measured *about* `B_pk`).
The robust members are `B_ave`, `√m₂` and (to the extent the third moment
converges) `α`, which are amplitude-weighted integrals over the whole window and
average noise down. When in doubt, read `B_ave`/`B_rms`; treat `B_pk`/`β` as
indicative. The user guide and the widget both say so, and our chosen uncertainty
method is required to *expose* this (see implementation-options.md).

## When to use moments vs fitting the lineshape

Moments are **model-free**: they summarise whatever `p(B)` the spectrum shows
without assuming a functional form, which is exactly right for a first look, for
tracking a width across a temperature scan, and for distributions with no clean
analytic form. **Fit the lineshape instead** when you have a physical model of
`p(B)` (a Brandt vortex-lattice distribution, a Gaussian-broadened London model,
a sum of diamagnetic + background lines) and want its *parameters* with proper
covariances, or when the spectrum's tails are too noisy for a stable third
moment. The two are complementary: moments give the quick, assumption-light
trend; a lineshape fit gives the interpreted physics. The user guide carries this
"when to use" box.

## Study artifacts

- [comparison.md](comparison.md) — WiMDA arithmetic transcription, our moment
  definitions, and every documented divergence (both behaviours shown).
- [implementation-options.md](implementation-options.md) — core API, the five GUI
  integration seams (with file:line), and the open step-3 choices with
  recommendations.
- [test-data.md](test-data.md) — synthetic distributions (closed-form), the
  WiMDA shared-spectrum oracle, and the external-corpus gate.
- [verification-plan.md](verification-plan.md) — how each claim is checked.
</content>
</invoke>
