# Dynamic relaxation: test data & golden cases

All golden values are analytic — no external program output is required to verify
correctness (Mantid/musrfit can be used as an *optional* extra oracle but are
GPL, so reference-only). Times in µs, Δ/σ/a in µs⁻¹, ν in MHz (≡ µs⁻¹), B_L in G.

## 1. Analytic limit cases (exact — unit-test assertions)

| ID | Function | Inputs | Expected |
|---|---|---|---|
| L1 | dynamic Gaussian KT | ν=0, B_L=0 | == `static_gkt_zf(t,Δ)` (to 1e−9) |
| L2 | dynamic Gaussian KT | ν=0, B_L=50 | == `longitudinal_field_kubo_toyabe(t,Δ,50)` |
| L3 | dynamic Gaussian KT (ZF) | ν ≫ Δ (e.g. Δ=0.3, ν=20) | ≈ exp(−2Δ²t/ν), the 1/3 tail washed out (rel. err < few % over 0–8 µs) |
| L4 | dynamic Lorentzian KT | ν=0, B_L=0 | == `1/3 + 2/3(1−at)e^{−at}` |
| L5 | Keren | ω₀=0 (B_L=0) | == exp[−(2Δ²/ν²)(e^{−νt}−1+νt)] |
| L6 | Abragam | ν→0 | → exp(−σ²t²/2) (Gaussian); check t small |
| L7 | Abragam | ν≫σ | → exp(−(σ²/ν)t) (exponential) |
| L8 | any dynamic, LF | B_L → ∞ | → 1 (full decoupling) |
| L9 | t=0, all | — | G(0)=A0 exactly |

## 2. Continuity / monotonicity / sanity (property tests)

- All G_d(t) ∈ [−A0/3, A0] envelope; finite, no NaN/Inf for Δ,a,σ ∈ (0,5],
  ν ∈ [0,50], B_L ∈ [0,1e4], t ∈ [0,16] µs.
- ν→0 limit is continuous: |G_d(ν=1e−6) − G_static| < 1e−4.
- Increasing ν monotonically removes the ZF dip and slows long-time decay toward
  the motional-narrowed exponential.
- Increasing B_L raises the long-time asymptote toward 1.
- Keren overlays the numerical dynamic Gaussian KT (LF) within a few % in the
  fast/intermediate regime (ν ≳ Δ); diverges (expectedly) for ν ≪ Δ.

## 3. Cross-consistency (internal oracle)

- **C1:** Keren(ω₀=0) == 2× Abragam-exponent form with σ=Δ (definition check).
- **C2:** dynamic Gaussian KT(ZF) and Keren(ZF) agree in the fast regime
  (both → exp(−2Δ²t/ν)).

## 4. Synthetic fitting round-trip (golden recovery)

Generate noisy asymmetry from each model at known parameters, fit back with
`FitEngine`, assert recovery within uncertainties:

- **Copper-like:** dynamic Gaussian KT, Δ=0.37 µs⁻¹, ν stepped 0.05→5 MHz, ZF,
  A0=23 (%) — recovers Δ (≈const) and ν (rising), the motional-narrowing series.
- **Ionic-motion-like:** simultaneous 0/5/10 G LF, shared Δ, ν; per-field B_L
  fixed — recovers Δ, ν (uses the existing global-fit engine).
- **Abragam TF-like:** σ=0.3 µs⁻¹, ν 0.1→10 MHz — recovers the Gaussian→Lorentzian
  line-shape crossover.

## 5. Real corpus spot-checks (manual, post-implementation)

- Copper ZF 40 K (run 20886): dynamic Gaussian KT should give Δ≈0.37 µs⁻¹, ν≈0
  (static); higher-T runs (≥120 K) should return a finite, rising ν.
- Ionic-motion Al-LLZ 160 K triple (51341–51343): simultaneous fit, finite ν.
  Folder: `~/Documents/WiMDA muon school/Nuclear magnetism and ionic motion/`.
