# Comparison — WiMDA RigiWorkshopFit vs Asymmetry RFResonanceMuP

## The spin Hamiltonian (exact path)

WiMDA `muproton1` (in `RigiWorkshopfit.dpr`) builds an **8×8 real-symmetric**
matrix in the `|m_e, m_µ, m_p⟩` product basis and diagonalises it with
`eigen4` (real-symmetric Householder + QL from `Eigenuni.pas`). The diagonal
holds each product state's Zeeman and longitudinal hyperfine energy; the
off-diagonals are the electron–muon flip-flop (A/2, proton spectator) and the
electron–proton flip-flop (A_p/2, muon spectator).

This is exactly the **isotropic contact Hamiltonian**

    H = A_µ·Sₑ·S_µ + A_p·Sₑ·S_p + (γₑ Sₑz − γ_µ S_µz − γ_p S_pz)·B    (MHz),

with the proton coupling to the **electron** (not directly to the muon) — the
standard radical topology. Asymmetry builds the same operator with Kronecker
products and diagonalises with `numpy.linalg.eigvalsh`.

**Eigensolver substitution is exact.** The spectrum of a Hermitian matrix is
basis-independent, so the sorted eigenvalues `E₁ ≤ … ≤ E₈` — and therefore every
level *difference* WiMDA selects — are identical to `eigvalsh`'s output to machine
precision. `Eigenuni.pas` is infrastructure superseded by NumPy, exactly as the
earlier parity programme reframed it; this study covers the **RF-fitting
workflow** that infrastructure served, which was not previously ported.

**Cross-check (Aₚ = 0).** With `A_p = 0` the proton decouples; the 8 levels are
the 4 Breit-Rabi muonium levels each split by the bare proton Zeeman `±γ_p·B/2`.
Their pairwise **midpoints reproduce `muonium._tf_levels` to < 1e-6 MHz** and the
pair **splittings equal `γ_p·B`** — asserted in `test_rf_musr_resonance.py`. This
validates the Hamiltonian against the independently-ported muonium levels.

## The RF resonance condition

WiMDA `RFresonanceMuPlusProtonExact` finds the two resonance fields B₁ and B₂
with `zbrent` on 0–2000 G, then returns a flat background plus two Lorentzians,

    BG + Σₖ aₖwₖ²/(wₖ² + (B − Bₖ)²).

The two driven transitions are between **sorted-eigenvalue** pairs `E₇−E₅` and
`E₈−E₆`. Asymmetry mirrors this exactly: `rf_transition_freqs` returns
`(E₇−E₅, E₈−E₆)` (0-indexed `e[6]-e[4]`, `e[7]-e[5]`), and `rf_resonance_fields`
root-solves each for the field where it equals `ν_RF`.

**Transition-selection is unambiguous.** At the benzene couplings
(A_µ=514.78, A_p=124.6, ν_RF=218.5 MHz), a sweep over all 28 level pairs finds
that **only `E₇−E₅` (→ 893.9 G) and `E₈−E₆` (→ 796.7 G)** cross `ν_RF` inside the
experimental window (500–1100 G) — confirming WiMDA's `75`/`86` selectors are the
physically driven pair (asserted in the test suite).

**Root-finding deviation (robustness).** WiMDA calls `zbrent(…,0,2000,…)` once
per transition. The RF transitions rise through the resonance window then
saturate (and can turn over above ~1.9 kG), so there can be a spurious second
crossing. Asymmetry takes the **lowest ascending crossing** (coarse scan +
`brentq`) and returns `nan` when a transition cannot be bracketed in
`[1, 2000] G`; the model then drops that Lorentzian rather than raising, keeping
the curve finite for any trial parameters a least-squares minimiser probes. This
is a deliberate, documented hardening of WiMDA's single-`zbrent` call.

## Exact vs analytic (why only the exact variant is ported)

WiMDA also ships an **analytic** model `RFresonanceMuPlusProton` built on
first-order-in-`A_p` Breit-Rabi levels (`MuoniumPlusProton`: `E11, E21, E12,
E22`). Comparing the two at the benzene couplings:

| Model | B₁, B₂ (G) | mean (G) | split (G) |
|---|---|---|---|
| **Exact** (`eigvalsh`) | 893.9 / 796.7 | 845.3 | **97.1** |
| Analytic (1st-order Aₚ) | 888.2 / 810.4 | 849.3 | 77.8 |
| Digitised paper Fig. 3a | 865 / 773 | 819 | ~92 |

The **splitting** encodes `A_p`; the exact split (97 G) is within ~5 % of the
digitised paper split (92 G), while the analytic split (78 G) is ~15 % low —
exactly the paper's stated reason for using full numerical diagonalisation at
these low fields. We therefore port **only the exact variant** as a component
(`analytic_rf_transition_freqs` is kept as a module function for cross-checking
and the test that demonstrates the analytic under-splitting). The ~26 G offset
of the **mean** between our exact model and the digitised trace is attributed to
figure-tracing / absolute-field-calibration error and the paper's use of an
independent quantum-simulation code (ref 24); see verification-plan.md.

## Conventions and constants

| Quantity | WiMDA literal (MHz/G) | Asymmetry source | value (MHz/G) |
|---|---|---|---|
| `g_µ` | 0.01355342 | `MUON_GYROMAGNETIC_RATIO_MHZ_PER_T · 1e-4` | 0.013553882 |
| `g_e` | 2.80249514 | `ELECTRON_…_RAD_PER_US_PER_G / 2π` | 2.802495142 |
| `g_p` | 0.00425764 | `PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T · 1e-4` | 0.004257748 |

They agree to ~5 significant figures; the differences shift resonance fields by
far less than 1 G (negligible vs the data spacing). Energies/couplings in MHz,
field in Gauss, as in WiMDA. Parameter names map directly:
`A → A_mu`, `Ap → A_p`, `RF → nu_RF`, `ampl1/wid1/ampl2/wid2/BG` identical.

## Well-conditioning of the inverse problem

The position→coupling map `(B₁, B₂) → (A_µ, A_p)` is **well-conditioned**:
given the two exact resonance fields, a least-squares inversion recovers both
couplings to machine precision from any starting guess whose trial resonances
fall in the field window. The practical RF-µSR difficulty — and the reason the
paper quotes `A_p = 124.6(14)` with a large uncertainty while
`A_µ = 514.78(4)` is tight — is pinning the **dip positions** (especially their
**splitting**) from noisy asymmetry data, not the inversion itself. `A_µ` (the
mean) is the robust primary observable; `A_p` (the split) benefits from a
complementary ALC measurement.
