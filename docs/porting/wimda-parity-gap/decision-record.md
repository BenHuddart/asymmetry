# Consolidated decision record — WiMDA parity programme (PRs #30–#43)

Date: 2026-06-11. Compiled at Wave A closeout from the per-project study
docs; citations point at the owning study. Programme rule throughout:
*parity of functionality, not implementation — favour modern numerics and
physical correctness; document every divergence with both behaviours.*

This is the durable reference. The point-in-time closeout (status, Wave B
impact, collision watchlist) is [wave-a-closeout.md](wave-a-closeout.md);
the authoritative final programme status is
[programme-complete.md](programme-complete.md) (2026-06-13). The per-project
"open follow-ons" lists below are point-in-time at Wave A closeout — for the
current open/closed picture see programme-complete.md and
[follow-on-triage.md](follow-on-triage.md).

## 1. Programme-wide exclusions

| Item | Decision | Rationale |
|---|---|---|
| Eigen.pas eigensolvers | Out, permanently | Old-roadmap mislabel: Hermitian eigensolvers serving the F–μ–F DLL models, superseded by `np.linalg.eigh`; never a spectral estimator |
| Kramers–Kronig transform | Out | Optical-spectroscopy utility; transforms no μSR observable |
| HDF4 `.nxs` | Out (standing decision reaffirmed) | Coverage boundary, not a bug |
| Burg all-poles MEM | **In** (reversal), diagnostic framing only | Shipped in PR #42 as "Resolution (Burg)" with documented pathologies |
| Legacy formats (`.tri`, `.kek`, KEK binary, DeltaT, VMS PSI, MCS, 16 ns `.raw`, ARGUS/CHRONUS `.ral`), zip/bz2 loading, legacy `alc*` files, laser/aux `.mon` logs | Drop | Superseded by NeXus; no active data sources |
| TRIUMF MUD loader | Not a parity item | WiMDA's own MUD support is a non-functional stub; stays on the general roadmap |
| Printing; in-app GLE editor | Drop | PDF export / external editors supersede |
| Fit-table resampling; animated fitting; fit-vars/save-vars two-stage selections; cursor-point→fit-table | Drop | Legacy spectacle / superseded by structured trending |
| ALCscans form, SetALCthresh, CommentForm, registration/licensing | Drop | Dead or vestigial in WiMDA itself |
| Negative-muon elemental analysis | Deferred brief (adapt-not-port) | `projects/negative-muon-analysis.md`; needs μ⁻ data + user pull |
| Live current-run monitoring | Optional late phase | Beamline access required for testing |

## 2. Per-project record

Format per project: **left behind** (deliberate non-ports) / **deviations**
(Asymmetry ≠ WiMDA, on purpose) / **open follow-ons** (still unshipped at
closeout).

### wimda-fit-function-parity (PR #30)

- **Left behind**: `otScaledFRotation`, `rtFstr` (expr/link-group recipes
  documented instead); `rtGau2`/`rtSig2` (reparameterisations, σ-mapping
  documented); BeCu pressure-cell pair (composite recipe / single-instrument
  calibration polynomial); `Fequitriangle` (folded into `FmuF_Triangle`);
  WiMDA's own withdrawn functions (`rtHH`, `otKTdist`, `otDelayRot`).
- **Deviations**: RischKehr via `erfcx` (no unphysical Γ<0 mirror);
  MuoniumHighTFAniso exact 4-level diagonalisation (**fitted D not
  WiMDA-comparable**); MuoniumLFRelax exact Breit–Rabi λ(B) — the WiMDA
  `(1−δ)` prefactor was checked against Kadono and **removed as spurious**
  (PR #71, 2026-06-13); the rate is now the clean BPP/Redfield form
  `λ(B)=δ_ex²τ_c/(1+(ω₁₂τ_c)²)`; DynamicFmuF Abragam-form crossover
  (WiMDA seam up to ~30%); FmuF_Triangle all six couplings + full powder
  average (**fitted distances not comparable**); DipolarSpinJ signed mixing
  angle (WiMDA wrong for J > ½); Bessel re-introduced (beyond parity);
  radians/positive-frequency/CODATA conventions throughout; GBKT via
  Gauss–Hermite.
- **Closed (PR #71, 2026-06-13)**: fit-wizard opt-in for the new components;
  reviewer sign-off on the non-textbook sources completed; the MuLFRelax
  (1−δ) prefactor resolved (removed as spurious, see Deviations above).

### model-function-parity + follow-ons (PRs #32, #38, #39)

- **Left behind**: x2 second analytic variable (cross-group fitting is
  strictly more general; requirements recorded); `*fit.dll` loading (→
  python-user-functions); ReloadFit (.asymp supersedes); Estimate-mode
  *iteration* (one-step fixed point instead); composite-expressible
  functions as recipes not components; Model Fit Table widget (results-table
  recursion supersedes).
- **Deviations** (D1–D12, E1–E7): |x| power-law guard; Polynomial in
  absolute x (**WiMDA cubic-BG coefficients not 1:1 transferable**);
  activation energies in meV/CODATA (WiMDA e/k 0.089% low);
  MuRepolarisation fits A_hf with B₀ derived; scatter-rescale error mode
  with suppressed-verdict explanation; χ² band via `chi2.ppf`; opt-in
  effective-variance x-uncertainty (beyond WiMDA); results recursion + `⊕`
  quadrature operator (beyond WiMDA).
- **Open**: single-fit accumulation across fits; `⊕` in the time-domain
  grammar (only on physical demand); stale-series auto-removal; fit_index
  default-x nicety.

### simulate-mode + follow-ons (PRs #33, #37)

- **Left behind**: sample-environment log simulation (provenance hazard);
  event-mode; PSI/ROOT writers; WiMDA template-copy NeXus writing (minimal
  V1 writer instead).
- **Deviations**: seeded PCG64 with seed in provenance; degrade = exact
  binomial thinning for f<1 (WiMDA's Poisson(k·f) is over-dispersed);
  pre-t0 bins background-only; exact `time_zero`; derived run not in-place
  overwrite; zero deadtimes recorded as *correct* for synthetic counts;
  beyond-WiMDA: instrument templates, archetype gallery, multi-group
  simulation, pull diagnostic.
- **Open**: deadtime-distortion injection; **two-period and count-mode
  simulation (handed to count-domain, NOT delivered by PR #41 — stray, see
  closeout)**; more instrument presets; pull-distribution batch mode.

### data-reduction-parity + follow-ons (PRs #34, #36)

- **Left behind**: KEK spill deadtime; fitted-baseline Set BG (deferred);
  per-directory `default.mgp`/`.exclude` auto-load (dead code in WiMDA
  itself); ARGUS/KEK fixers and N→32 mapping.
- **Deviations** (D1–D14): **D14 — WiMDA's General-α functional has no
  interior minimum at realistic statistics** (closed-form two-window solution
  instead, informative failure); **D4 — tail-fit by Poisson MLE keeping all
  bins** (WiMDA amputates ≤4-count bins, deleting fine-binned tails);
  bounded ln-α minimisation with σ_α on deterministic windows (D1–D3);
  background-run subtraction deadtime-corrects both runs and propagates
  variance (D6/D7); exact variable-bin law (D8); pulsed t0 = half-maximum
  rising edge (D9); grouping-time zero-weight exclusion (D10);
  asymmetry-level G−R (D11); arbitrary N periods (D12).
- **Open**: plot-side quick binning control (UI-polish pass); optional live
  WiMDA spot-checks; TF-phase fine-t0 refinement (beyond WiMDA); period
  mapping in the integral/ALC path; **tail-fit in the Fourier input path
  (handed to frequency-domain-finishers, NOT delivered by PR #42 — stray)**;
  co-subtract reuse of `subtract_scaled_counts` (→ run-arithmetic).

### asymmetry-error-propagation (PR #35, emergent project)

- **Left behind**: selectable `error_model` Mantid-parity escape hatch
  (deferred; would enshrine a wrong formula).
- **Deviations**: exact Poisson σ_A = 2|α|√(FB(F+B))/(F+αB)² replaces
  Mantid's (1+A²) independent-propagation over-estimate — **Mantid is the
  outlier**; WiMDA/musrfit/textbook agree with the exact form. One-sided
  bins keep the 1.0 no-information sentinel. Measured pre-fix impact:
  amplitude σ +3.2%, χ²ᵣ biased low.
- **Closed (2026-06-13)**: testing-worktree σ_A goldens audited against the
  exact-Poisson change — no `.asymp` goldens exist, and the error-bearing CSV
  reports were regenerated on post-fix `main` (χ²ᵣ centres ≥1.0, the exact-σ
  signature). See the audit note on the `testing/wimda-eval` branch.

### maxent-completion (PR #40)

- **Left behind**: spectral deconvolution `Sconv` (unbounded 1/Sconv adjoint
  — deferred); looseness/phase-acceleration knobs (**out permanently** —
  they target WiMDA's Skilling–Bryan kernel, dead controls on our
  projected-gradient engine); `.max` binary export / auto-save-per-cycle.
- **Deviations**: reconstruction overlay in normalised χ² space + combined
  view + residuals strip (beyond WiMDA); pulse response folded into the
  cos/sin kernel preserving the OPUS/TROPUS adjoint (Mantid oracle:
  machine-precision single-pulse); deadtime fit reuses
  `calibrate_deadtime_from_histograms` with **suggest-only promotion**
  (WiMDA auto-applies silently); σ-inflation ×1e8 exclusion with exact
  (1−A²) errors (not WiMDA's √(N+2)); SpecBG ×1.201 constant kept but
  anchored in-window, display-only; phase exchange matched by group id with
  provenance (WiMDA's index-matched buffer footgun removed); Tesla axis via
  shared `core/fourier/units.py`.
- **Open**: plot-workspace "view band" abstraction; loader capture of pulse
  half-width/separation (speculative).

### count-domain-fit-modes (PR #41, incl. six follow-ons)

- **Left behind**: negative-muon mode (deferred brief); RRF coupling (→
  rrf); KEK spill; Set BG (out by decision).
- **Deviations**: **Poisson (Cash) cost by default on raw counts** (Gaussian
  √N selectable) — fixes WiMDA's documented late-time underweighting;
  baseline drift multiplies the whole polarization; dpsep by coarse→fine
  scan (the t>dpsep/2 gate defeats migrad); free-τ option (beyond WiMDA);
  DT0-only drives the reduction with DT1/C2–C4 as provenance; √α
  parameterisation kept deliberately (α–amplitude correlation falls out).
- **Open**: unify `fgAll` onto the Poisson cost factory (→
  fit-workflow-diagnostics); MINOS on α (same project).

### frequency-domain-finishers (PR #42)

- **Left behind**: FB t=0 extrapolation (moot under grouped-counts source);
  per-detector FFT (standing deferral); N₀ single-histogram input
  (deferred); field-axis display was verify-only (already better than
  WiMDA).
- **Deviations**: pulse compensation ×1/R(ν) from the shared
  `core/maxent/pulse.py` model, capped and zeroed beyond the first node
  (WiMDA's exp((πfτ)²) grows unboundedly); iterative σ-clip robust baseline
  (WiMDA single-pass reachable as max_iter=1); per-bin peak S/N;
  conditioning on the averaged display channel; Burg as badged diagnostic;
  diamag removal via bounded windowed `curve_fit`.
- **Open**: N₀ single-histogram input; field-axis probe override (¹⁹F/¹H);
  inherited stray: tail-fit in the Fourier input path (see closeout).

### radical-correlation-spectrum (PR #43, emergent from #42 optional scope)

- **Left behind**: ALC code (already shipped; docs teach complementarity);
  anisotropic hyperfine; DFT prediction; low-field correlation pair.
- **Deviations**: **exact Breit–Rabi forward map** (uniform A axis,
  ν₁₂+ν₃₄ = A by construction) instead of WiMDA `rmatch`'s rounded-constant
  high-field inverse and non-uniform axis; correlate-the-averaged-spectrum
  (CorrFn is non-linear, so not bit-identical to WiMDA's per-group AvCorr —
  documented, better noise behaviour); A_μ treated as a coupling axis,
  excluded from field-unit conversion (physically meaningless via γ_μ);
  CorrFn order-weighted combiner ported verbatim.
- **Open**: per-group AvCorr bit-parity mode on demand; hyperfine-axis
  range/step controls; low-field variant; link to the
  muonium-radical-hyperfine fit workflow.

## 3. Reference-program bug ledger — "do not oracle against"

WiMDA unless noted. Discovered and documented during the programme:

1. `KTBArray` low-field branch writes its result outside the loop — array
   static KT wrong for 0 < B < 2Δ (`KuboToyabe.pas:151–166`).
2. `ZFdipgen` mixing angle from cos²2α discards the sign — **wrong for every
   J > ½** (deviations to ~0.56 of polarization).
3. `Fequitriangle` wrapper geometry internally inconsistent; triangle omits
   all F–F couplings.
4. `MuLFrel` mixes MHz and rad/μs in its field expression.
5. `ZFprotondipole` constant `c = 5.05` flagged wrong in WiMDA's own source.
6. General-α estimator: no interior minimum at realistic statistics — grid
   walk silently returns the clamp (pinned in `tests/test_alpha_estimation.py`).
7. t0 search: scan ceiling uses display-bin count (shrinks under bunching);
   pulse-peak max used even on pulsed data.
8. Tail-fit background amputates ≤4-count bins via σ=10¹⁰ — deletes
   fine-binned tails.
9. `rmatch` constants rounded at the 5th significant figure (and
   inconsistent with WiMDA's own constants elsewhere); recovered A drifts
   ~0.01–0.03 MHz.
10. Variable-binning exponent off by ×1.0014; thermal-activation e/k 0.089%
    low; `gmu2` rounded at the 6th digit (all cosmetic-to-minor).
11. Simulate: unphysical growing exponential before t0; unseeded RNG;
    `time_zero` quantised to whole μs; degrade over-dispersed Poisson(k·f).
12. **Mantid** `AsymmetryCalc`: (1+A²) error over-estimate — the outlier
    among all references; Mantid `start.py` truncates τ_μ (harmless ~1e-3).
13. WiMDA `FEATURE_MAP.json` untrusted — misroutes "maxent" to the Burg code.
14. Dead code: `default.mgp`/`.exclude` auto-load commented out; MUD loader
    stub.
