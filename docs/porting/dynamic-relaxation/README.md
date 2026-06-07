# Dynamic Relaxation Functions Study

Status: study pass complete; implementation not started (no engine code in this pass).

This study covers a coherent family of **dynamic (fluctuating-field) muon
relaxation functions** that Asymmetry currently lacks, identified as the top
functionality gap by the 2026-06 WiMDA-corpus evaluation
(`docs/testing/reports/wimda-eval-2026-06/REPORT.md`, finding G1):

1. **Dynamic Gaussian Kubo–Toyabe** (ZF **and** LF) — strong-collision
   generalisation of `StaticGKT_ZF` / `LFKuboToyabe`.
2. **Dynamic Lorentzian Kubo–Toyabe** (ZF and LF) — for dilute-moment /
   spin-glass field distributions.
3. **Keren** LF relaxation function — analytic dynamic Gaussian relaxation in a
   longitudinal field.
4. **Abragam** function — single-component relaxation interpolating Gaussian ↔
   Lorentzian (TF line-shape, copper-style hop-rate extraction).

They are studied together because they share one piece of physics — a static
local-field distribution dephasing under stochastic fluctuations at rate **ν** —
and because three of the four are analytic special cases / limits of each other
(see "Internal consistency" below), so they must use one consistent parameter
convention.

## Why (corpus motivation)

Every system in the WiMDA "Nuclear magnetism and ionic motion" section requires
one of these, and none can currently be analysed quantitatively in Asymmetry:

- **Copper** (`Copper 2026.docx`): ZF **dynamic Gaussian KT** for the hop rate,
  TF **Abragam** for the hop rate above ~100 K; Arrhenius → activation energy.
  (The static-KT Δ=0.37 µs⁻¹ is reproduced for T≲90 K; above that the static fit
  degrades exactly where dynamic KT is needed.)
- **Ionic motion, Al-LLZ** (`Ionic motion 2026.docx`): explicitly *"two main
  options are a dynamic Kubo–Toyabe function and a Keren function"* for the
  simultaneous 0/5/10 G LF-decoupling fit; Δ(T) and ν(T) → activation energy.
- **Spin glass / dilute magnets**: dynamic **Lorentzian** KT.

## The physics (one paragraph)

A muon in a static random local field **B** with distribution width Δ (Gaussian)
or a (Lorentzian) precesses and dephases, giving the static KT function G_s(t).
If the field reorients stochastically with correlation time τ_c = 1/ν
("strong-collision": each collision draws a new field from the same
distribution, instantaneously and isotropically), the polarisation becomes the
**dynamic** G_d(t). Two universal limits bound it: **ν → 0** recovers the static
G_s(t) (with its 1/3 ZF tail); **ν → ∞** gives motional narrowing — exponential
decay with rate λ = 2Δ²/ν (Gaussian) — washing the 1/3 tail away. A longitudinal
field B_L adds a Larmor term ω₀ = γ_µ B_L that decouples the muon (G_d → 1 as
B_L → ∞).

## References (anchored to original papers, per design decision)

- R. Kubo and T. Toyabe, in *Magnetic Resonance and Relaxation* (1967) — static KT.
- R. S. Hayano, Y. J. Uemura, J. Imazato, N. Nishida, T. Yamazaki, R. Kubo,
  **Phys. Rev. B 20, 850 (1979)** — Gaussian KT in longitudinal field + the
  strong-collision dynamic generalisation (the canonical dynamic-KT reference).
- Y. J. Uemura, T. Yamazaki, D. R. Harshman, M. Senba, E. J. Ansaldo,
  **Phys. Rev. B 31, 546 (1985)** — Lorentzian (dilute) KT, ZF and LF.
- A. Keren, **Phys. Rev. B 50, 10039 (1994)** — analytic generalisation of the
  Abragam function to dynamic Gaussian relaxation in a longitudinal field.
- A. Abragam, *The Principles of Nuclear Magnetism* (Oxford, 1961), Ch. X — the
  Abragam (exp of exp) relaxation function.
- Textbook: Blundell, De Renzi, Lancaster, Pratt, *Muon Spectroscopy* (2021),
  §5.3 — dynamic KT limits used as unit tests.

## Internal consistency (cross-checks the implementation must satisfy)

- **Keren at ω₀ = 0** reduces to the **Abragam** exponent:
  Γ(t) = (2Δ²/ν²)(e^{−νt} − 1 + νt). (Same algebra; factor 2 = two transverse
  ZF components. The single-component Abragam drops the 2.)
- **Dynamic Gaussian KT at ν → 0** = `StaticGKT_ZF` (ZF) / `LFKuboToyabe` (LF).
- **Dynamic Gaussian KT, ZF, ν ≫ Δ** → exp(−2Δ²t/ν) (motional narrowing).
- **Keren** ≈ **dynamic Gaussian KT (LF)** in the fast/intermediate regime
  (Keren is the analytic fast-fluctuation approximation; cross-plot must overlay
  the numerical strong-collision result where both are valid).

These four relations are the backbone of the verification plan.

## Prior art (see comparison.md for detail)

- **Mantid:** `DynamicKuboToyabe.cpp` (iterative time-step convolution; ZF+LF
  Gaussian). No Keren/Abragam as named functions.
- **musrfit:** `PTheory` `dynGssKT`/`statGssKT`/`dynLrKT`/`statLrKT`, plus a
  `combiLGKT` and the `Abragam` user function; LF via numerical convolution.
- **WiMDA:** dynamic KT + Keren via its user-fitting function registry (Pascal
  source in the corpus at `wimda installation/user fitting/`).

## Design decisions (confirmed with maintainer, 2026-06)

- **Scope:** Gaussian **and** Lorentzian dynamic KT, both **ZF and LF**; plus
  Keren and Abragam. All four in **one PR**.
- **Fluctuation/hop rate ν in MHz** (≡ µs⁻¹ numerically for a rate; matches the
  guides' "fluctuation rate 0.2 MHz" and the existing `nu` ParamInfo). Static
  widths Δ/σ stay in µs⁻¹.
- **References:** original papers (above), cited in each component description.
