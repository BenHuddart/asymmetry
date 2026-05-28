# μSR practical-workflow catalogue

This is the canonical inventory of the analysis workflows that the
Asymmetry documentation should teach. Each entry distills how a real
muon-spectroscopy experiment is analysed end-to-end. Sources are
two modern textbooks:

- **Blundell, De Renzi, Lancaster, Pratt (eds.),** *Muon Spectroscopy:
  An Introduction*, Oxford University Press, 2022. (Abbreviated
  "Blundell" below.)
- **Amato & Morenzoni,** *Introduction to Muon Spin Spectroscopy:
  Applications to Solid State and Material Sciences*, Springer
  Lecture Notes in Physics 961, 2024. (Abbreviated
  "Amato-Morenzoni" below.)

Further-reading pointers (no extracts) for users who want
training-school material:

- [PaN-Learning](https://e-learning.pan-training.eu/) — hosts ISIS
  Muon Training School lectures and quizzes.
- [MuonSources.org Resources](https://muonsources.org/resources/)
  — community portal with links into all major facility training
  programmes.
- [ISMS meetings](https://www.musr.org/meetings) — proceedings of
  the triennial International μSR Conference series.

Each catalogue entry uses a fixed format so the case-study chapters
in `docs/user_guide/workflows/` can derive directly from it.

## Catalogue conventions

For each workflow we record:

- **Slug** — kebab-case identifier referenced by case studies and
  candidate folders.
- **Physics question** — one sentence on what the experiment is
  designed to answer.
- **Instrument requirement** — ZF / LF / TF, sample environment
  needs (cryostat, magnet, oven), pulsed vs continuous beam.
- **Data signature** — what the raw asymmetry "looks like" so a
  user can recognise the regime from their own data.
- **Analysis steps** — ordered list of the standard pipeline.
- **Asymmetry support** — which existing GUI features / API
  functions support each step; what's missing.
- **Textbook anchors** — chapter / figure references.
- **Roadmap links** — candidates this workflow surfaces or relies
  on.

---

## 1. Order-parameter temperature scan (`order-parameter-tscan`)

**Physics question.** Locate the critical temperature *T*\ :sub:`c`
(or *T*\ :sub:`N`) of a magnetic transition and measure the
temperature dependence of the order parameter
*M*\ (*T*) ∝ ν\ :sub:`μ`\ (*T*).

**Instrument requirement.** Zero-field (or weak TF/LF for cross-
check); continuous or pulsed beam; cryostat covering the transition.

**Data signature.** Below *T*\ :sub:`c`: clear spontaneous-field
precession at a single (or few) frequencies. Above *T*\ :sub:`c`:
exponential paramagnetic relaxation that broadens sharply through
*T*\ :sub:`c` and decays away above.

**Analysis steps.**

1. Load the temperature scan as a logbook (~6-20 runs).
2. Group and bunch each run identically; verify alpha is consistent
   across temperatures.
3. Fit each run individually. Below *T*\ :sub:`c`: oscillatory +
   constant; above: exponential + constant. Around *T*\ :sub:`c`:
   composite or stretched envelope.
4. Assemble parameter trend: ν(*T*) below *T*\ :sub:`c`, λ(*T*)
   throughout.
5. Fit ν(*T*) to a power law *M*\ (*T*) =
   *M*\ :sub:`0`\ (1 − *T*/*T*\ :sub:`c`)\ :sup:`β` and extract
   *T*\ :sub:`c` and the critical exponent β.

**Asymmetry support.**

- ✅ Data browser with sort / filter on temperature.
- ✅ `Oscillatory`, `Exponential`, `Constant` components.
- ✅ Fit Wizard ranks candidates automatically.
- ✅ Parameter trending panel for ν(*T*).
- ❌ No built-in power-law fit for the order parameter (parametric
  registry has `SC_*` for superconductors but no
  `LandauPowerLaw` model). Workaround: use
  `core.fitting.parameter_models` with a custom expression, or
  fall back to scipy.
- ◐ The Fit Wizard's portfolio is rich enough below *T*\ :sub:`c`
  but may underweight stretched-exponential near *T*\ :sub:`c`.

**Textbook anchors.** Blundell Ch 6 (Fig 6.6 EuO ferromagnet); Ch 6.2
(Landau model, critical exponents); Amato-Morenzoni Ch 5 (sections
on magnetic ordering and critical-exponent extraction).

**Roadmap links.** Closely related to
[`dynamic-kubo-toyabe`](../../candidates/dynamic-kubo-toyabe/) and
[`theory-library-expansion`](../../candidates/theory-library-expansion/)
(SpinGlass for near-Tc fluctuations).

---

## 2. LF Kubo–Toyabe field-decoupling series (`lf-kt-decoupling`)

**Physics question.** Distinguish static from dynamic local-field
distributions; measure the width Δ of the static field distribution
in a nonmagnetic host (e.g. Ag, Cu).

**Instrument requirement.** ZF + a series of LFs spanning
γ\ :sub:`μ`\ *B*\ :sub:`L`/Δ from 0 to ~10; continuous beam preferred
for the dense field sampling.

**Data signature.** At *B*\ :sub:`L` = 0: characteristic Gaussian
KT dip and 1/3 tail recovery. As *B*\ :sub:`L` increases, the dip
fills in and the tail rises toward unity. Full decoupling at
γ\ :sub:`μ`\ *B*\ :sub:`L` ≫ Δ produces a flat-or-slowly-decaying
asymmetry.

**Analysis steps.**

1. Load the LF series as a group in the data browser.
2. Verify identical grouping and alpha across runs.
3. Global fit: shared Δ, A\ :sub:`0`, baseline; per-run *B*\ :sub:`L`
   fixed to the applied value.
4. Compute γ\ :sub:`μ`\ *B*\ :sub:`L`/Δ for each run and verify the
   decoupling progression matches Hayano (1979).
5. If the 1/3 tail decays in ZF, suspect dynamics — flag for a
   follow-up dynamic-KT analysis.

**Asymmetry support.**

- ✅ Data browser groups.
- ✅ `LFKuboToyabe` component.
- ✅ Global Fit tab with shared parameters.
- ❌ `DynamicKuboToyabe` not yet available — must be flagged as
  "static only" if dynamics are suspected.

**Textbook anchors.** Blundell Ch 5.2 (Fig 5.6 LF-KT decoupling
curves) and Ch 5.2 (Hayano et al. PRB 20, 850, 1979); Amato-Morenzoni
Ch 4 (LF formalism).

**Roadmap links.**
[`dynamic-kubo-toyabe`](../../candidates/dynamic-kubo-toyabe/)
unlocks the dynamic follow-up.

---

## 3. Superconductor σ(T) → λ(T) penetration depth (`sc-sigma-to-lambda`)

**Physics question.** Extract the temperature dependence of the
magnetic penetration depth λ\ :sub:`L`\ (*T*) and, from it, the
superconducting gap structure (s-wave / d-wave / multi-gap).

**Instrument requirement.** TF above *H*\ :sub:`c1` and below
*H*\ :sub:`c2` (typically 100-2000 G for high-Tc cuprates;
20-200 mT for type-II superconductors generally); cryostat covering
*T*\ :sub:`c` and well into the superconducting state.

**Data signature.** Above *T*\ :sub:`c`: ordinary Larmor precession
with weak Gaussian damping from nuclear dipoles. Below *T*\ :sub:`c`:
the FFT lineshape becomes asymmetric (vortex-lattice P(B)); the
time-domain envelope shows an enhanced Gaussian damping rate σ(*T*).

**Analysis steps.**

1. Load all temperature points as a series.
2. Fit each TF run as a damped cosine `Oscillatory * Gaussian`
   (Asymmetry composite: `(Oscillatory){frac} + Constant` with the
   Oscillatory carrying a Gaussian envelope).
3. Extract σ(*T*) and ν(*T*) per run.
4. Switch to the parameter-trending panel: σ(*T*) vs *T*.
5. Convert σ → λ via σ = 0.609 (μ\ :sub:`0`\ γ\ :sub:`μ`)/(λ²·4π²)
   — implemented by the SC parametric models.
6. Fit σ(*T*) with `SC_SWave`, `SC_TwoGap_SS`, or `SC_TwoGap_SD`
   depending on the expected gap structure.
7. Report Δ\ :sub:`s`/k\ :sub:`B`\ *T*\ :sub:`c` and the gap weights.

**Asymmetry support.**

- ✅ Multi-run TF fits with `Oscillatory` and `Gaussian` damping
  composites.
- ✅ Parameter-trending panel.
- ✅ `SC_SWave`, `SC_TwoGap_SS`, `SC_TwoGap_SD`,
  `SC_TwoGap_DD` parametric models.
- ◐ Vortex-lattice TF time-domain forward model not present — users
  must approximate as Gaussian-damped Larmor (acceptable for clean
  type-II SCs; biased for cuprates near *H*\ :sub:`c1`).
- ❌ Brandt-form vortex P(B) fit is not directly available.

**Textbook anchors.** Blundell Ch 9 (Fig 9.5 MgB₂ σ(T)); Sonier RMP
72, 769 (2000); Amato-Morenzoni Ch 6 (multi-gap models and
penetration-depth analysis).

**Roadmap links.**
[`theory-library-expansion`](../../candidates/theory-library-expansion/)
adds the time-domain Brandt vortex model;
[`maxent-spectrum`](../../candidates/maxent-spectrum/) sharpens the
P(B) lineshape extraction.

---

## 4. F–μ–F entanglement identification (`fmuf-identification`)

**Physics question.** Detect muon coordination by fluorine nuclei
(or other I = 1/2 spins) via the characteristic F–μ–F dipolar beat
pattern; extract the μ–F bond length.

**Instrument requirement.** ZF in a diamagnetic fluoride host;
continuous or pulsed beam; sample environment usually dilution
refrigerator or 4He cryostat to suppress thermal phonons that wash
out the entanglement.

**Data signature.** Slow oscillatory beat with a characteristic
3-frequency envelope (for linear F–μ–F) or 5-frequency for general
geometry. Period of ~5-10 μs; high time-resolution required.

**Analysis steps.**

1. Load the ZF run.
2. Recognise the F–μ–F pattern: characteristic dip-and-recovery
   over the first ~5 μs followed by slow oscillation.
3. Fit with `FmuF_Linear` (collinear) or `FmuF_General` (powder-
   averaged with two distances and an angle).
4. Extract *r*\ :sub:`μF` (typically 1.0-1.3 Å for fluorides).
5. Cross-check against DFT muon-site calculations
   (`MuFinder` / `mu-LFC` from the broader Source/ ecosystem).

**Asymmetry support.**

- ✅ `MuF`, `FmuF_Linear`, `FmuF_General` components.
- ✅ Fit Wizard portfolio includes F–μ–F candidates.
- ❌ No FFT-based "is this F–μ–F or just two oscillation frequencies"
  diagnostic tool.

**Textbook anchors.** Blundell Ch 4.6 (Brewer et al. PRB 33, 7813,
1986; Lancaster et al. PRB 75, 094421, 2007); Amato-Morenzoni Ch 4
(short discussion of multi-spin polarisation functions).

**Roadmap links.** None new — feature is mature.

---

## 5. Vortex-lattice TF Fourier analysis (`vortex-tf-fourier`)

**Physics question.** Map the internal-field distribution P(*B*) of
a type-II superconductor in the mixed state; identify lattice
symmetry (triangular vs square vs glassy) from the lineshape.

**Instrument requirement.** TF well above *H*\ :sub:`c1` and below
*H*\ :sub:`c2`; field-cooled measurement preferred to ensure a
well-ordered vortex lattice.

**Data signature.** Asymmetric FFT line: sharp van Hove peak at
the saddle-point field, long high-field tail to the vortex-core
field, suppressed low-field shoulder at the minimum-field
saddle.

**Analysis steps.**

1. Load the field-cooled TF run.
2. Switch the central plot to the Frequency domain.
3. Compute the grouped FFT spectrum.
4. Inspect the lineshape: asymmetric → vortex lattice; symmetric →
   may indicate disorder.
5. Extract the second moment σ\ :sub:`VL`² of P(*B*) which is
   directly related to λ via the Brandt formula.

**Asymmetry support.**

- ✅ Frequency-domain view, grouped FFT pipeline.
- ✅ Apodisation modes (Lorentz, Gauss, none) for sharpening /
  smoothing.
- ❌ MaxEnt for super-resolved lineshape (currently a stub).
- ❌ Direct fit to Brandt P(*B*) profile.

**Textbook anchors.** Blundell Ch 9.5 (Sonier RMP 72, 769, 2000);
Amato-Morenzoni Ch 6 (the section on the vortex-state field
distribution).

**Roadmap links.**
[`maxent-spectrum`](../../candidates/maxent-spectrum/);
[`theory-library-expansion`](../../candidates/theory-library-expansion/)
(Brandt P(*B*) model).

---

## 6. Paramagnetic Knight-shift TF measurement (`paramagnetic-knight-shift`)

**Physics question.** Measure the muon Knight shift
*K* = (ν − ν\ :sub:`ref`)/ν\ :sub:`ref` in a metal to probe the
local spin susceptibility χ\ :sub:`P`\ (*T*).

**Instrument requirement.** Moderate TF (typically 0.1-0.6 T);
high-statistics detector configuration; reference sample (Ag or
Pt) measured under identical conditions.

**Data signature.** Cleanly precessing TF asymmetry with a tiny
frequency offset (parts per 10³-10⁴) relative to the reference.

**Analysis steps.**

1. Load the sample and reference TF runs.
2. Resolve the data into per-detector or per-group histograms.
3. Switch the central plot to the **Individual Groups** domain so
   per-detector traces are visible.
4. Open the **Multi-Group Fit** window — this engages automatically
   when the Individual Groups domain is active.
5. Fit per-group amplitudes, baselines, and relative phases as local
   parameters; share the Larmor frequency and damping rate as global.
6. Compute *K* = (ν\ :sub:`sample` − ν\ :sub:`ref`)/ν\ :sub:`ref` ×
   10⁶ (units: ppm).
7. Trend *K*(*T*) and compare to bulk susceptibility χ(*T*).

**Asymmetry support.**

- ✅ Individual Groups domain.
- ✅ Multi-Group Fit window with shared/local parameter
  classification.
- ✅ Parameter trending panel for *K*(*T*).
- ◐ No built-in "compute *K* against a reference run" helper;
  user must subtract frequencies manually.

**Textbook anchors.** Blundell Ch 6.3 (the section on the local
field; Knight-shift contribution from electrons); Amato-Morenzoni
Ch 4 (Knight-shift formalism) and Ch 7 (Knight shift in muonium-
hosting semiconductors).

**Roadmap links.**
[`phase-auto-calibration`](../../candidates/phase-auto-calibration/)
gives better initial phases for the per-group fit;
[`minos-error-analysis`](../../candidates/minos-error-analysis/)
delivers the accurate uncertainty needed to quote ppm-level shifts.

---

## 7. Muonium-radical hyperfine analysis (`muonium-radical-hyperfine`)

**Physics question.** Identify the formation and structure of
muoniated radicals in organic / chemical systems; extract hyperfine
coupling constants *A*\ :sub:`μ`.

**Instrument requirement.** Higher TF (0.5-2 T) preferred to resolve
the muonium hyperfine pair frequencies; ALC sometimes used instead
(see workflow 8).

**Data signature.** Two precession lines symmetric about
ν\ :sub:`μ` ± *A*\ :sub:`μ`/2; or, in lower fields, complex multi-
frequency Larmor signals corresponding to muonium hyperfine sub-
levels.

**Analysis steps.**

1. Load the TF run.
2. Compute the FFT to identify the hyperfine pair frequencies.
3. Fit time-domain composite of two `Oscillatory` components plus
   constant background.
4. Compute *A*\ :sub:`μ` = ν\ :sub:`+` − ν\ :sub:`−`.
5. Trend *A*\ :sub:`μ`\ (*T*) for dynamic effects (rotation,
   tunnelling).

**Asymmetry support.**

- ◐ Composite of two `Oscillatory` components possible via the
  composite-model expression syntax.
- ❌ No dedicated muonium-pair fit function (Mantid ships
  `HighTFMuonium`, `LowTFMuonium`, `TFMuonium`, `ZFMuonium`,
  `MuoniumDecouplingCurve`).
- ❌ No ALC-style analysis for hyperfine *transitions*.

**Textbook anchors.** Blundell Ch 12 (muoniated radicals);
Amato-Morenzoni Ch 7 (muonium in semiconductors — same physics
applies in solid-state chemistry).

**Roadmap links.** New candidate
[`muonium-radical-hyperfine`](../../candidates/muonium-radical-hyperfine/)
needed (logged in Phase 6);
[`alc-avoided-level-crossing`](../../candidates/alc-avoided-level-crossing/)
provides the complementary resonance technique.

---

## 8. Avoided level-crossing resonance scan (`alc-resonance-scan`)

**Physics question.** Identify and quantify quadrupolar or hyperfine
resonances in a sample by scanning the applied field through a
level-crossing condition and observing the suppression of muon
polarisation.

**Instrument requirement.** Scan-capable LF magnet (typically up
to 5 T); a high-stability cryostat for slow scans (~hours per
field point).

**Data signature.** Plot of integrated asymmetry vs applied field
shows broad smooth baseline with sharp dips at resonance fields.

**Analysis steps.**

1. Load the field-scan as a series of runs.
2. Integrate the asymmetry over a common time window per run.
3. Fit the baseline (polynomial or spline) excluding resonance
   regions.
4. Subtract the baseline and fit Lorentzian / Gaussian peaks to
   the residual.
5. Extract resonance positions, widths, integrated areas.

**Asymmetry support.** ❌ None. This is a Mantid-only workflow
today. Logged as the
[`alc-avoided-level-crossing`](../../candidates/alc-avoided-level-crossing/)
roadmap candidate (Later tier, heaviest GUI surface in the
roadmap).

**Textbook anchors.** Blundell Ch 19.4 (hyperfine ALC); Amato-
Morenzoni Ch 7 (ALC in semiconductors).

**Roadmap links.**
[`alc-avoided-level-crossing`](../../candidates/alc-avoided-level-crossing/).

---

## 9. Spin-glass freezing curve (`spin-glass-freezing`)

**Physics question.** Identify a spin-glass transition
*T*\ :sub:`f` and characterise the freezing dynamics via a stretched
exponential or Uemura-form fit.

**Instrument requirement.** ZF + LF cross-checks; cryostat to base.

**Data signature.** Above *T*\ :sub:`f`: exponential paramagnetic
relaxation. Through *T*\ :sub:`f`: relaxation rate λ peaks and
the stretching exponent β drops below unity. Below *T*\ :sub:`f`:
quasi-static field distribution emerges; near-Gaussian Kubo–Toyabe
or Uemura form.

**Analysis steps.**

1. Load the temperature scan.
2. Fit each run with `StretchedExponential + Constant`.
3. Trend λ(*T*) and β(*T*); identify the *T*\ :sub:`f` peak.
4. Verify spin-glass character with LF decoupling: at base *T*,
   apply LF — the relaxation should remain Gaussian-KT-like
   rather than fully decouple.

**Asymmetry support.**

- ✅ `StretchedExponential` component.
- ✅ Parameter trending.
- ❌ No dedicated **Uemura-form spin-glass model** (covered by
  the `theory-library-expansion` candidate).

**Textbook anchors.** Blundell Ch 5.4 (stretched exponential); Ch 6
(spin-glass examples); Uemura et al. PRB 31, 546 (1985).

**Roadmap links.**
[`theory-library-expansion`](../../candidates/theory-library-expansion/)
adds the Uemura SpinGlass component.

---

## 10. Diffusion / motional-narrowing measurement (`muon-diffusion`)

**Physics question.** Measure the muon (or muonium) hopping rate
in a host lattice via the temperature dependence of relaxation.

**Instrument requirement.** ZF + LF; cryostat covering the
diffusion-onset temperature.

**Data signature.** At low *T*: static Gaussian KT (immobile muon).
Above the activation temperature: relaxation accelerates then
narrows as the muon hops faster than the dipolar coupling
(motional narrowing → exponential decay with λ ∝ Δ²/ν).

**Analysis steps.**

1. Load the temperature scan.
2. At each *T*: fit dynamic-KT (or static + small exponential)
   and extract ν(*T*).
3. Trend ν(*T*) and fit an Arrhenius or BPP model to extract
   *E*\ :sub:`a`.

**Asymmetry support.**

- ✅ Parameter trending.
- ❌ `DynamicKuboToyabe` (covered by the `dynamic-kubo-toyabe`
  candidate).
- ❌ Arrhenius / BPP parametric models not yet in the
  `core.fitting.parameter_models` registry.

**Textbook anchors.** Blundell Ch 8 (muon diffusion; BPP
relaxation); Amato-Morenzoni Ch 5 (motional narrowing).

**Roadmap links.**
[`dynamic-kubo-toyabe`](../../candidates/dynamic-kubo-toyabe/)
unlocks the time-domain fit. A new candidate
[`bpp-relaxation`](../../candidates/bpp-relaxation/) (logged in
Phase 6) adds the BPP parametric model.

---

## 11. Pulsed-source period merging (`period-merging-pulsed`)

**Physics question.** Combine multi-period histograms from a
pulsed source (e.g. ISIS) into a single asymmetry channel before
analysis.

**Instrument requirement.** Pulsed beam; multi-period acquisition
mode.

**Data signature.** Multiple histograms per detector per run
in the raw file; per-period statistics typically lower than the
combined.

**Analysis steps.**

1. Load the multi-period run.
2. Apply period arithmetic (e.g. `1+2-3` for sum of periods 1+2
   minus a background period 3).
3. Continue with the standard asymmetry workflow.

**Asymmetry support.** ❌ Asymmetry's NeXus loader currently
exposes only the default period. Logged as the
[`period-arithmetic`](../../candidates/period-arithmetic/)
candidate.

**Textbook anchors.** Blundell Ch 14 (pulsed sources); Ch 15.1
(experimental setup).

**Roadmap links.**
[`period-arithmetic`](../../candidates/period-arithmetic/).

---

## 12. Low-energy μSR thin-film depth profiling (`lem-thin-film`)

**Physics question.** Probe the local magnetic / electronic
environment as a function of depth in a thin film, multilayer,
or surface.

**Instrument requirement.** Low-energy muon (LEM) beam — only
available at PSI; sample environment with depth-controlled
implantation energy.

**Data signature.** Series of runs at different implantation
energies (typically 1-30 keV mapping to ~5-200 nm depth) showing
depth-dependent relaxation or precession.

**Analysis steps.**

1. Load the depth scan.
2. Convert each implantation energy to a mean implantation depth
   (TrimSP simulation or Monte Carlo).
3. Fit each run independently.
4. Trend extracted parameters vs depth.

**Asymmetry support.**

- ✅ Multi-run loading.
- ✅ Parameter trending.
- ❌ No built-in implantation-energy → depth conversion.
- ❌ No coupling to TrimSP or other range-stopping simulators.

**Textbook anchors.** Blundell Ch 18 (low-energy μSR); Amato-
Morenzoni Ch 8 (LEM, depth profiling).

**Roadmap links.** New candidate
[`lem-depth-profiling`](../../candidates/lem-depth-profiling/)
(logged in Phase 6) for the depth-conversion helper.

---

## 13. Negative muon X-ray elemental analysis (`mu-minus-xrf`)

**Physics question.** Determine elemental composition by detecting
characteristic X-rays from muonic atoms; depth-resolved by tuning
the negative muon momentum.

**Instrument requirement.** Negative muon beam with X-ray
detectors; ISIS, RIKEN-RAL, J-PARC, MuSIC.

**Data signature.** X-ray spectrum (not asymmetry vs time) with
sharp lines characteristic of each element present in the sample.

**Analysis steps.** Out of scope for Asymmetry — this is the
domain of Mantid's `Elemental Analysis` interface. Catalogued
here for completeness but **not on the Asymmetry roadmap**.

**Textbook anchors.** Blundell Ch 22 (negative muon techniques);
Amato-Morenzoni Ch 9.

**Roadmap links.** **Out of scope** by team decision (see
ROADMAP.md "What's deliberately not on the roadmap").

---

## 14. Structural / non-magnetic phase transitions
(`structural-transition`)

**Physics question.** Detect lattice / charge-density-wave / orbital-
order transitions through subtle changes in the muon precession
frequency, damping, or volume fraction.

**Instrument requirement.** TF or ZF; high-resolution temperature
scans.

**Data signature.** Discontinuities or kinks in fitted parameters
(λ, ν, A\ :sub:`0`) at the transition temperature; sometimes
volume-fraction splits between two components.

**Analysis steps.**

1. Standard temperature-scan fit (as in workflow 1).
2. Trend several parameters; look for kinks rather than
   power-law trends.
3. If volume-fraction splitting suspected, fit two-component
   composite with shared physics and a fraction parameter.

**Asymmetry support.**

- ✅ Composite-model expression syntax handles the two-component
  case via fraction groups.
- ✅ Parameter trending.
- ❌ No dedicated "transition finder" — user has to spot the
  kink visually.

**Textbook anchors.** Blundell Ch 11 (ionic motion); Amato-
Morenzoni Ch 5 (subtle structural transitions in correlated
materials).

**Roadmap links.** New candidate
[`structural-transitions`](../../candidates/structural-transitions/)
(logged in Phase 6) catalogues the workflow but the main need is a
better "trend analysis" toolkit.

---

## 15. Project-file portability / reproducibility
(`project-reproducibility`)

**Physics question.** Not a physics workflow — instead, how do
users save / share / reload an analysis state so a collaborator can
reproduce the result?

**Instrument requirement.** N/A.

**Data signature.** N/A.

**Analysis steps.**

1. After finishing an analysis, File → Save Project → `.asymp`.
2. Share the `.asymp` + the original data files.
3. Collaborator: File → Open Project to restore everything (data
   selections, grouping, fit configurations, parameter trends).

**Asymmetry support.**

- ✅ Schema-versioned JSON `.asymp` project files.
- ✅ Forward-compatible loading via documented migrations.
- ◐ No "package data + project into a zip" helper (the data files
  are referenced by path).

**Textbook anchors.** N/A — this is a workflow-management concern
not covered in the physics textbooks. Worth mentioning anyway as
many users come from `.msr` or `.mantid` ecosystems and expect to
ask "where's my workflow file?"

**Roadmap links.**
[`msr-import`](../../candidates/msr-import/) bridges from
musrfit's `.msr` files.

---

## Summary table

The 15 workflows above map onto Asymmetry's current capability as
follows:

| Workflow | Fully supported | Partial / workaround | Blocked on roadmap |
|---|:---:|:---:|:---:|
| order-parameter-tscan | ✅ | | |
| lf-kt-decoupling | ✅ | (no dynamic KT) | |
| sc-sigma-to-lambda | ✅ | | |
| fmuf-identification | ✅ | | |
| vortex-tf-fourier | ◐ | (no MaxEnt) | |
| paramagnetic-knight-shift | ✅ | | |
| muonium-radical-hyperfine | | ◐ via composites | ❌ specialised models |
| alc-resonance-scan | | | ❌ |
| spin-glass-freezing | ◐ stretched only | | |
| muon-diffusion | | | ❌ dynamic KT + BPP |
| period-merging-pulsed | | | ❌ |
| lem-thin-film | ◐ runs load fine | (no depth conv.) | |
| mu-minus-xrf | | | (out of scope) |
| structural-transition | ◐ via parameter trends | | |
| project-reproducibility | ✅ | | |

The case-study chapters in `docs/user_guide/workflows/` cover the
**fully supported** rows that have the richest pedagogical content:
order-parameter-tscan, sc-sigma-to-lambda, lf-kt-decoupling.

Additional workflows are mentioned in per-page additions
(Phase 4) and surfaced as roadmap candidates (Phase 6).
