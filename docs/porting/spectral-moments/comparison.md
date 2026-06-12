# Spectral moments — WiMDA comparison

Oracle: `$WIMDA_SRC/src/Moments.pas` (GPL; transcribed as a *behavioural*
oracle — arithmetic re-derived in our own idiom, never copied). Cross-references
to the field-resolution and default-level definitions are in
`$WIMDA_SRC/src/MaxControl.pas`.

## 1. WiMDA's pipeline (`TMoment.FormShow`, lines 152–393)

WiMDA builds a **field** spectrum from the MaxEnt frequency spectrum and computes
moments over a windowed, cutoff-thresholded subset.

### 1.1 Build the field spectrum (lines 169–176)

```pascal
for i := nmin to nmax do begin
  x^[np, 1] := (i - 1) * bres;     { field axis, Gauss }
  x^[np, 2] := F[i] - DEF;         { amplitude, MaxEnt level less default }
  inc(np)
end;
```

- `bres := fres / 0.01355342` (`MaxControl.pas:271`) — field resolution in Gauss,
  i.e. the frequency resolution divided by `γ_μ/2π = 0.01355342 MHz/G`. The field
  axis is therefore `B = (i−1)·bres`.
- `F[i]` is the MaxEnt spectral amplitude; `DEF` is the MaxEnt flat **default**
  (prior) level, subtracted so the baseline sits at zero.
- `nmin`/`nmax` are the index bounds of the user frequency window
  (`MaxControl.pas:295,302`: `nmin := 1 + round(Fmin/bres)` etc.).

**Our equivalent.** Asymmetry's spectra already carry a field/frequency axis
(`FieldUnit`) and a baseline-subtracted amplitude (the Fourier conditioning
ladder — σ-clip / WiMDA baseline — and the MaxEnt reconstruction both deliver a
zero-baselined lineshape). Per the collision directive we add **no baseline or
default handling of our own**; the W15 accessor hands the core the
already-conditioned `(x, amplitude, errors, unit)`. The core treats `x` as the
field/frequency axis in whatever unit the GUI passes (Gauss by default).

### 1.2 Discrete peak (lines 178–188)

```pascal
ppk := 0; Bpk := 0; ipk := 0;
for i := 1 to np do
  if x^[i,2] > ppk then begin ppk := x^[i,2]; Bpk := x^[i,1]; ipk := i end;
```

`ppk` = peak amplitude, `ipk` = its index, `Bpk` = its field. **`ppk` is the
discrete peak**, and it is what the cutoff threshold is later measured against.

> **WiMDA indexing bug (divergence D1).** The spectrum is filled `np = 0,1,2,…`
> (0-based; `inc(np)` after each store), but the peak search runs `for i := 1 to
> np` — it **skips index 0 and reads index `np`**, one past the last stored
> point (stale/zero memory). The moment sums below run `for i := 0 to np-1`
> (correct 0-based). We use the **consistent** `0 … n-1` range everywhere; on any
> spectrum whose true peak is the first bin, or whose last+1 slot held a large
> stale value, WiMDA could pick a different peak. Documented; we do not
> reproduce the bug.

### 1.3 Parabolic peak refinement (`parabpkextrap`, lines 87–147)

A 5-point quadratic least-squares fit around `ipk`, evaluated at its vertex:

- normalise `xx = (B[i] − B[ipk]) / Δ` with `Δ = B[ipk+1] − B[ipk]`, over
  `i = ipk−2 … ipk+2`;
- least-squares parabola `y = a·xx² + b·xx + c`;
- vertex `xx* = −b/(2a)`, then `B_pk = xx*·Δ + B[ipk]`.

Guarded by `(index < n − nexpts/2) and (index > nexpts/2)` (with `nexpts = 5`,
`nexpts div 2 = 2`): **if the discrete peak is within 2 bins of either end, the
parabolic step is skipped** and `B_pk` stays the discrete peak. We reproduce this
exactly (vertex of the 5-point LSQ parabola, edge guard, fall back to the
discrete bin), and re-derive the closed-form normal-equation solution rather than
copying WiMDA's hand-unrolled `m1…m6` accumulators.

### 1.4 Window + cutoff mask (lines 227–229, 258–260)

A point `i` contributes iff

```
x[i,2] > cutoff·ppk    AND    xnmin ≤ x[i,1] ≤ xnmax
```

with `cutoff = CutoffEdit/100` (a **percentage of the discrete peak amplitude**)
and `[xnmin, xnmax]` the user field range. Strict `>` on the cutoff (so points
exactly at threshold, and all negatives, are excluded; with `ppk>0` and
`cutoff∈[0,1]` the threshold is positive). We match: `amplitude > cutoff_fraction
· peak_amplitude` and `x_range[0] ≤ x ≤ x_range[1]`.

### 1.5 Mean (first pass, lines 222–251)

```
m0 = Σ p ;  m1 = Σ p·B ;  Bave = m1/m0
```

amplitude-weighted mean field over the masked points. If `m0 = 0` (empty window)
WiMDA blanks all labels and exits with `n_sample` reported. We return a result
flagged `n_sample = 0` / `success`-style empty so the GUI can grey the readout.

### 1.6 Central moments (second pass, lines 253–281)

```
about Bave:  m2  = Σ p·(B−Bave)²/m0 ,  m3 = Σ p·(B−Bave)³/m0
about Bpk:   m2pk = Σ p·(B−Bpk )²/m0
√m₂   = sqrt(m2)            { Brms vs Bave }
√m₂,pk = sqrt(m2pk)          { Brms vs Bpk  }
α = power(abs(m3), 1/3)/√m₂ , negated if m3<0    { skewness }
β = (Bave − Bpk)/√m₂,pk                           { lineshape asymmetry }
```

Notes:

- `m1` about `Bave` is computed but ≈0 by construction (first central moment) and
  unused; we skip it.
- **`α`'s cube-root normalisation is WiMDA's own convention.** The conventional
  dimensionless skewness is `γ₁ = m₃/m₂^{3/2}`. WiMDA instead reports `α =
  ∛|m₃|/√m₂` (sign of `m₃`). Dimensionally `∛|m₃| ∼ B` and `√m₂ ∼ B`, so `α` is
  dimensionless, but it is **not** `γ₁` — it is `sign(m₃)·|γ₁|^{1/3}·…` no, it is
  its own quantity: `α = sign(m₃)·|m₃|^{1/3}/m₂^{1/2}`. For parity we report
  WiMDA's `α`; the core **also** exposes the textbook `γ₁ = m₃/m₂^{3/2}` as
  `skewness_g1` so users have the standard quantity. Divergence D2.
- `β` measures how far the mean sits from the peak in units of the peak-referenced
  RMS — positive when the mean lies above the peak (long high-field tail, the
  vortex-lattice signature). Sign convention discussed in §3.

### 1.7 Run-averaging accumulator (lines 297–387)

When the moments window is visible and a *new* spectrum is shown, WiMDA folds the
current run's seven moments into running sums `Σ` and `Σ²`, increments `nave`, and
— **only once `nave > 2`** — reports, per moment, a mean and a **population**
standard deviation:

```
mean = Σ/nave
var  = Σ² − Σ²/nave ;  std = sqrt(var/nave) if var>0 else 0
```

A `Reset` button zeroes the accumulators. The export writes either the
single-run row (`Button1`) or the averaged row (`Button3`).

**Divergence D3 — we do not replicate the stateful accumulator.** Per Ben's
decision, a selection of spectra becomes one computed `FitSeries` with one member
per spectrum; run-to-run averaging and its scatter error are handled by the
existing parameter-trend layer (which already averages/fits a series and is the
modern equivalent of WiMDA's fit-table export). This is strictly more capable:
the trend layer keeps every member (not just running sums), can fit `B_rms(T)`,
and round-trips through `.asymp`. We note WiMDA's `std = sqrt(var/nave)` is the
*population* σ (divides by `nave`, not `nave−1`) and is only shown for `nave>2`.

### 1.8 Export row (`Button1Click`, lines 395–423)

```
! Run  Field  Temp   Bave   Bpk  Bave-Bpk  RMSa  RMSp  Alpha  Beta
```

i.e. run number, applied field, temperature, then the seven-moment row. Our
trend-series member carries exactly these as `parameters` (plus `run_number`,
`run_label`, `field`, `temperature` in the frozen six-key row shape), so the
fit-table export is reproduced as a first-class trendable series.

## 2. Our moment definitions (Qt-free core)

`spectrum_moments(x, amplitude, *, x_range, cutoff_fraction, errors=None,
method=…)` returns a `SpectrumMoments` dataclass with, for the masked window:

| field | definition |
|---|---|
| `b_pk` | parabolic-refined peak (discrete peak if edge-guarded) |
| `b_ave` | `Σ p·x / Σ p` |
| `b_diff` | `b_ave − b_pk` |
| `b_rms_mean` | `√(Σ p·(x−b_ave)² / Σ p)` |
| `b_rms_peak` | `√(Σ p·(x−b_pk)² / Σ p)` |
| `skewness` | WiMDA `α = sign(m₃)·∛\|m₃\| / b_rms_mean` |
| `skewness_g1` | textbook `γ₁ = m₃ / m₂^{3/2}` (our addition) |
| `beta` | `b_diff / b_rms_peak` |
| `n_sample` | masked point count |
| `*_err` | uncertainties per the agreed method (WiMDA gives none) |
| `recipe` | `(x_range, cutoff_fraction, unit, mode)` provenance |

The mask, the cutoff-vs-discrete-peak semantics, the parabolic peak with edge
guard, and the amplitude-weighting all match WiMDA. The additions over WiMDA are:
the textbook `γ₁`, real per-moment **uncertainties**, and explicit window
provenance.

## 3. The β sign convention (and its citation)

`β = (B_ave − B_pk)/√m₂,pk`. With the mixed-state field distribution `p(B)` —
which has a sharp low-field cutoff at the lattice's saddle-point field and a long
tail to high field near the vortex cores — the **mean sits above the peak**, so
`B_ave − B_pk > 0` and **β > 0** denotes the physically expected
positive-skew (high-field-tailed) lineshape. This matches WiMDA's sign and the
sign of the skewness `α` for the same distribution. The literature citation for
the asymmetric `p(B)` and the sign of its skewness is recorded with Ben at
step-3 and pinned here and in the user guide (the textbook's mixed-state field
distribution treatment; Brandt's vortex-lattice `p(B)` is the canonical primary
reference). Verified on a synthetic vortex-lattice-like lineshape in
[verification-plan.md](verification-plan.md).

## 4. Divergence ledger

| id | WiMDA | Asymmetry | rationale |
|---|---|---|---|
| D1 | peak search `for i:=1 to np` (skips bin 0, reads bin `np`) | consistent `0…n−1` over the stored window | WiMDA off-by-one over stale memory; physical correctness |
| D2 | reports only `α = sign(m₃)·∛\|m₃\|/√m₂` | reports `α` **and** textbook `γ₁ = m₃/m₂^{3/2}` | give the standard skewness alongside parity |
| D3 | stateful run-average accumulator (running Σ, Σ², population σ, Reset) | per-spectrum members in one computed `FitSeries`; scatter → trend layer | modern, lossless, round-trips, fits trends |
| D4 | single-spectrum moments carry **no** error (`errsread=false`) | real per-moment uncertainties (propagation/bootstrap, step-3) | "we should do better than WiMDA" |
| D5 | MaxEnt spectrum only | MaxEnt **and** phase-corrected real FFT (eligibility-gated) | both are lineshape-faithful; broadens applicability |
| D6 | field axis hard-wired to Gauss | field-default Gauss + `G/T/MHz` selector, unit recorded | flexibility; `α`,`β` invariant, `B_*` rescale |

WiMDA's commented-out analytic error block (`Moments.pas:195–217`, the
`errsread` branch) is a useful precedent for D4 — it does standard linear
propagation of per-point errors through `m0…m3`. We document it in
implementation-options.md as the linear-propagation reference and decide
propagation-vs-bootstrap at step-3.
</content>
