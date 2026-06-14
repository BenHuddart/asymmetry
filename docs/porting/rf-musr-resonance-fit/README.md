# RF-µSR resonance fit (muon + electron + proton spin Hamiltonian)

Study-first port of WiMDA's **`RigiWorkshopFit`** RF-µSR resonance model: fit a
**field-swept RF-µSR resonance scan** of a muoniated radical to the full
muon + electron + proton spin Hamiltonian and extract the **muon hyperfine
coupling `A_µ` and the proton hyperfine coupling `A_p` simultaneously**.

This closes the confirmed parity gap **PC1** (`docs/testing/parity-checks.md` on
the `testing/wimda-eval` branch): Asymmetry had no way to fit RF-resonance
positions to the spin Hamiltonian. The high-field-limit linear formula
`B_res = (ν_RF ± ½A_µ)/γ_µ` is inaccurate at the few-hundred-G fields these scans
use (it gives ~416 MHz instead of the paper `A_µ = 514.78 MHz` on the benzene
example), and `A_p` was entirely inaccessible.

## What RF-µSR measures

A muoniated radical such as **cyclohexadienyl C₆H₆Mu** (Mu addition to benzene)
couples the **muon** spin and one dominant **proton** spin to the unpaired
**electron**. In RF-µSR a fixed-frequency RF field `ν_RF` is applied while the
static longitudinal field `B` is swept; the (Red − Green) integral asymmetry
shows a **W-shaped double dip**. Each dip is a resonance where an RF-driven
muon-spin-flip transition of the three-spin system matches `ν_RF`. The two
resonance fields `B₁, B₂` have a **mean that tracks `A_µ`** and a **splitting
that tracks `A_p`**, so one field scan determines both couplings.

## WiMDA reference

`wimda installation/user fitting/` (in the corpus
`C:\Users\benhu\iCloudDrive\Documents\WiMDA muon school`):

- **`RigiWorkshopfit.dpr`** — the fit-function DLL. Relevant functions:
  - `muproton1` — builds the **8×8 real-symmetric Hamiltonian** in the
    `|m_e, m_µ, m_p⟩` product basis and diagonalises it (the exact path).
  - `MuProtLevels` / `MuProtAsym` — energy levels / LF asymmetry from `muproton1`.
  - `RFresonanceMuPlusProtonExact` — the RF resonance model: root-solve the two
    sorted-level transitions `E₇−E₅` (`FreqDiff1exact`, selector `75`) and
    `E₈−E₆` (`FreqDiff2exact`, selector `86`) for the fields where they equal
    `RF`, then return `BG + Σ aᵢ·wᵢ²/(wᵢ² + (B−Bᵢ)²)`.
  - `MuoniumPlusProton` / `RFresonanceMuPlusProton` — a first-order **analytic**
    (Breit-Rabi + perturbative `A_p`) variant; **intentionally not ported as a
    component** because it is inaccurate at low field (see comparison.md).
- **`Eigenuni.pas`** — WiMDA's hand-written Hermitian eigensolvers
  (`eigen`..`eigen4`, NAG/Numerical-Recipes Householder + QL / Jacobi). These are
  **infrastructure, not behaviour**: the spectrum of a Hermitian matrix is
  basis-independent, so we replace them with `numpy.linalg.eigvalsh` and the
  sorted eigenvalues — hence every level *difference* WiMDA selects — are
  identical to machine precision.

## Building blocks already in-tree

- `core/fitting/muonium.py` `_tf_levels` — Breit-Rabi muonium levels; with
  `A_p = 0` the 8-level system collapses pairwise onto these (the test cross-check).
- gyromagnetic ratios in `core/utils/constants.py`
  (`MUON_…`, `PROTON_GYROMAGNETIC_RATIO_MHZ_PER_T`) and `muonium.py`
  (`G_E_MHZ_PER_G`, `G_MU_MHZ_PER_G`).
- the parameter-vs-field trend framework in `core/fitting/parameter_models.py`
  (`ParameterCompositeModel`, field-scope components) — the natural seam, since
  the observable is asymmetry **vs swept field**.
- `numpy.linalg.eigvalsh` (batched) and `scipy.optimize.brentq`.

## Chosen design (implemented)

A new core module **`core/fitting/muon_proton.py`** holds the physics:

- `mu_proton_hamiltonian` / `mu_proton_levels` — the 8×8 Hamiltonian
  `H = A_µ·Sₑ·S_µ + A_p·Sₑ·S_p + (γₑSₑz − γ_µS_µz − γ_pSₚz)·B` (MHz, field in G),
  diagonalised with `eigvalsh` (batched over field).
- `rf_transition_freqs` — the two RF transitions `E₇−E₅`, `E₈−E₆` (WiMDA parity).
- `analytic_rf_transition_freqs` — the analytic variant, for cross-checks only.
- `rf_resonance_fields` — robust root finder (coarse scan + `brentq`) for the two
  resonance fields; returns `nan` for an unbracketable resonance so the minimiser
  never sees a non-finite curve.
- `rf_resonance_mup` — the two-Lorentzian field-swept model.

This is registered as a **field-scope parameter-trend component
`RFResonanceMuP`** in `parameter_models.py` (params `A_mu, A_p, nu_RF, ampl1,
wid1, ampl2, wid2, BG`), with applicability + APS references in
`component_docs.py` and a user-guide section in
`docs/user_guide/parameter_trending.rst`.

See [comparison.md](comparison.md) for the WiMDA physics and the exact/analytic
difference, [implementation-options.md](implementation-options.md) for the seams
considered, [test-data.md](test-data.md) for fixtures and corpus data, and
[verification-plan.md](verification-plan.md) for the verification done.

**Status: IMPLEMENTED** — `RFResonanceMuP` field-trend component backed by exact
8×8 diagonalisation; verified by exact round-trip and against the benzene corpus
(McKenzie 2013, paper-graded).

## Reference paper

I. McKenzie, R. Scheuermann, S. P. Cottrell, J. S. Lord, and I. M. Tucker,
*Hyperfine Coupling Constants of the Cyclohexadienyl Radical in Benzene and
Dilute Aqueous Solution*, J. Phys. Chem. B **117**, 13614 (2013).
DOI 10.1021/jp4068763. (Corpus: `Chemistry/Muon spectroscopy of benzene/RFpaper.pdf`.)
