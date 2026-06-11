# Muoniated-radical correlation spectrum — test data

Synthetic-first (checkpoint-1 decision). No real muoniated-radical TF data is
available in the corpus (`~/Documents/radical/` holds only the reference PDFs),
and the method is verifiable end-to-end from first principles, so the gating
tests are synthetic. A WiMDA-transcribed `rmatch`/`CorrFn` oracle anchors the
parity claims.

## 1. Synthetic radical TF run (primary gating fixture)

Generated through `core/simulate` (the synthetic-run generator from PR #33), as
a grouped TF run with a known applied field B and a known muon hyperfine
coupling A_µ:

- **Signal model:** the diamagnetic line at `γ_µ·B` plus the two radical
  precession lines at the exact Breit–Rabi frequencies `ν₁₂(B, A_µ)` and
  `ν₃₄(B, A_µ)` from `muonium.py._tf_levels` (so `ν₁₂ + ν₃₄ = A_µ`), each with a
  realistic transverse relaxation and comparable amplitude, on top of Poisson
  counting statistics.
- **Worked-example parameters:** cyclohexadienyl radical, **A_µ = 514.4 MHz**,
  at **B = 2900 G** (one of McKenzie's Fig. 3 fields). At this field
  `ν₁₂ ≈ 218 MHz`, `ν₃₄ ≈ 296 MHz`, diamagnetic line `≈ 39 MHz`.
- **Expected result:** the correlation spectrum has a single dominant peak at
  the hyperfine axis position **A_µ = 514.4 MHz** (tolerance set by the
  spectrum's frequency resolution, a few MHz), and the diamagnetic line and
  spectral noise do **not** produce a comparable peak.

Variants for robustness:
- A **second field** (B = 14500 G) for the same A_µ — the correlation peak must
  stay at 514.4 MHz (field-independence of the recovered coupling).
- A **two-radical** mixture (two distinct A_µ) — two correlation peaks at the
  two couplings.
- A **field sweep** of A_µ values (e.g. 200, 514, 700 MHz) — peak tracks A_µ.

## 2. WiMDA `rmatch` / `CorrFn` numerical oracle

Transcribed verbatim from the Pascal as fixed-input/fixed-output checks (these
guard the *parity* claim and document the rounded-constant divergence):

- `rmatch(freq, field)` (`Plot.pas:515-523`) at the table points in
  [comparison.md §2.2](comparison.md): e.g. `rmatch(−285.5759, 1000) ≈ −214.42`,
  giving `|f₁+f₂| ≈ 500.0001` for true A = 500. The Asymmetry exact forward map
  must agree with `rmatch` to **~0.03 MHz** (the documented WiMDA approximation
  error), and with the true A_µ to **machine precision**.
- `CorrFn(y1, y2, order)` (`Plot.pas:1387-1394`): the order-weighting and the
  `order = 0 → |y1·y2|` fallback, at hand-computed points (e.g.
  `CorrFn(2, 2, 2) = 4`, `CorrFn(4, 1, 2) = 2·4/(16 + 1/16)`,
  `CorrFn(4, 1, 0) = 4`).

## 3. Breit–Rabi cross-checks (already in this study)

The `w12 + w34 = A` identity from `muonium.py._tf_levels`, verified to machine
precision over A ∈ {330, 500, 514, 1200} MHz, B ∈ {1000…5000} G
([comparison.md §2.1](comparison.md)). The implementation test re-runs this as a
property check (the correlation core must inherit it).

## 4. Regression guards (must not change)

- Existing `fourier`/`maxent` tests (the new `correlation` display mode must not
  alter any other mode's output).
- The project-file FFT-recipe round-trip: a `.asymp` saved with a correlation
  config reloads to an identical spectrum (additive `GroupSpectrumConfig` keys).
- `compute_average_group_spectrum` for every non-correlation mode is byte-for-
  byte unchanged.

## What is *not* tested here

- Real radical data (none available; not required for correctness).
- ALC (out of scope; already covered by the time-integral-asymmetry suite).
- Anisotropic / low-field correlation (out of scope, §comparison).
