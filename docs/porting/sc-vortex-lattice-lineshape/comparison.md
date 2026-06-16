# Comparison

## Reference behaviour

| Program | Vortex-lattice lineshape? | Notes |
|---|---|---|
| **WiMDA** | No | Fits `Oscillatory * Gaussian` to the TF line and reports the Gaussian `σ`. The non-Gaussian VL line is approximated as a Gaussian; the systematic from that approximation is acknowledged in the literature (Pratt 2009, Ref. 29 footnote). |
| **musrfit** | Yes (`userFcn`) | Ships vortex-lattice field-distribution user functions (analytic and Brandt-iterative `p(B)`), cosine-transformed to the time domain. This component is the in-repo, scriptable equivalent. |
| **Asymmetry (before)** | No | Only the field-domain `SC_Brandt_VortexLattice[_Powder]` trend models (σ→λ). No way to fit the lineshape itself. |
| **Asymmetry (this study)** | Yes | `VortexLattice` / `VortexLatticePowder` time-domain components, width-calibrated to the existing Brandt trend models. |

## Why not just keep using a single Gaussian

Measured on the LiFeAs corpus (Sample 1, run 3366, 1.5 K, 400 G, Up/Down):

| Fit choice | single-Gaussian `σ` (µs⁻¹) |
|---|---|
| rebin 1, t_max 8 µs | 1.72 |
| rebin 4, t_max 0.7 µs | 1.52 |
| rebin 20, t_max 8 µs | 0.73 |
| rebin 1, t_max 0.5 µs | 3.06 |

A true Gaussian is invariant to window/binning; this spans 0.7–3.1, i.e. the line
is **not** Gaussian and the Gaussian rate is a fit artefact. The model-free
envelope is concave (fast VL decay on a slow nuclear tail) and the line is
positively skewed — exactly the modified-London `p(B)`.

## What the new component changes — and what it does not

- **Does** provide a window-independent lineshape with a well-defined second
  moment, so on clean data (single crystal, or synthetic) `λ` is recovered
  robustly. Synthetic round-trip recovers `λ = 195.1 ± 0.1 nm`.
- **Does not** by itself resolve the LiFeAs *powder* headline (`λ_ab = 195 nm`).
  That signal is a three-component mix — fast VL ⊗ slow nuclear ⊗ persistent Ag
  background — that is mutually degenerate in a single run regardless of how good
  the VL lineshape is. Separating them needs either a normal-state-constrained
  nuclear rate or the field-dependence (the paper's Fig. 2(a) method). This is a
  **data-degeneracy** limit, not a model limit; documented in the LiFeAs
  cookbook (`wimda-eval`).

## Powder-average approximation

The powder variant uses the single-crystal modified-London shape evaluated at
`λ_eff = 3^{1/4} λ_ab` (correct second moment, per Pratt Eq. (3)). A full
orientation average over the anisotropic `λ(θ)` of a uniaxial superconductor
would smear the line further but needs `λ_c` (not determinable from a powder TF
measurement); Pratt's treatment collapses to the `3^{1/4}` second-moment factor,
which this model reproduces exactly.
