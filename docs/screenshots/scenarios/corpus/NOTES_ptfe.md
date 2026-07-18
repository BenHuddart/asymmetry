# NOTES — FµF state in PTFE (`Nuclear magnetism and ionic motion/The FmuF state in PTFE`)

Corpus-driven documentation screenshots for the quantum-entangled **F⁻–µ⁺–F⁻**
three-spin state in PTFE (Teflon). Data: MuSR runs 17293–17322 (ISIS pulsed
source, 2008), loaded from the HDF4 `.nxs` files in
`Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data/` through the
real Asymmetry loader. 17293 is a TF 20 G calibration run; 17294–17322 are
zero-field over ≈ 20–200 K.

Module: `ptfe_fmuf.py`. Capture (serialized under the shared lock — concurrent
offscreen-Qt captures deadlock, wave-1 finding):
`flock /tmp/asymmetry-capture.lock .venv/bin/python -m docs.screenshots.capture_corpus --only <name>`.

## Scenarios registered

| Scenario | Render | Intended docs use |
|---|---|---|
| `corpus_ptfe_zf_signature` | Raw ZF run 17294 (20 K, 41.6 MEv), no fit, framed 0–8 µs. | The raw FµF signature: the characteristic non-exponential dip (≈1.4 µs) → recovery bump (≈2.4 µs) → second dip of the collinear three-spin oscillation. |
| `corpus_ptfe_fmuf_fit` | **Headline.** `FmuF_Linear * Gaussian + Constant` converged on 17294; fit panel shows r_µF ≈ 1.30 Å, first 10 µs with the fit overlay. | The muon-specific FµF analytical polarisation model fitted on real data → the µ–F distance from the dipolar oscillation. |
| `corpus_ptfe_fft` | Frequency-domain FFT of 17294 with the Fourier inspector tab (apodisation τ = 4 µs, Lorentzian) raised; framed 0–1.2 MHz. | The sub-MHz FµF line cluster (broad combination lines, peak ≈0.47 MHz) that encodes the dipolar frequency; the FFT/apodisation controls. |
| `corpus_ptfe_tf_calibration` | TF 20 G run 17293, no fit, framed 0–12 µs. | The prescribed calibration step (GROUND_TRUTH §4): the slow ~0.3 MHz transverse precession used to fix the detector balance. |

Only `corpus_ptfe_fmuf_fit` runs a real iminuit fit at capture time → it alone
carries `requires_fit = True`.

## Run selection & workflow (GROUND_TRUTH.md § refs)

- **ZF signature + fit** (§3, §4, §6): run **17294** — the highest-statistics ZF
  run (20 K, 41.6 MEv; §3 run table), which gives the cleanest FµF beating in the
  set. Model `FmuF_Linear * Gaussian + Constant`
  (G(t) = A₁·G_FµF(t; r_µF)·exp(−σ²t²/2) + A_bg). `FmuF_Linear` is the collinear
  three-spin ZF polarisation, parameterised **directly by r_µF (Å)** (§4, §7).
- **Gaussian envelope is required** (§7): a bare `FmuF_Linear + Constant` pins
  r_µF at its bound (the reference-program file records χ²ᵣ ≈ 62); the more distant
  fluorines damp the oscillation, so the Gaussian damping term is what lets it fit.
- **Calibration** (§4): TF 20 G run **17293** (taken while cooling) — the only
  non-ZF run in the set; the guide names it as the detector-balance calibration.
- Fits run over the full time range (0.098–31.44 µs), which is what the GUI
  single-fit `Fit` button does. The fit reproduces from a literature-guidance
  seed (r_µF = 1.15 Å) without warm-starting.

## Fitted values vs guidance (⚠ no numeric ground truth — GT §6/§9)

The guide gives **no numeric target** for the dipolar frequency or r_µF. The
1.1–1.2 Å figure is a *literature expectation* (Brewer 1986 CaF₂ = 1.172 Å), not
a guide answer; the reference-program note (§7) is Asymmetry's own prior output,
not ground truth. Grade on producing a physically sensible FµF fit.

| Quantity | This capture (17294, 20 K) | Guidance / context | Verdict |
|---|---|---|---|
| **r_µF** | **1.296(1) Å** | 1.1–1.2 Å literature band (§6/§10); refprog 1.30 Å (§7) | Lands at ~1.30 Å — **above** the 1.1–1.2 band, but reproduces the prior Asymmetry result exactly. |
| σ (Gaussian damping) | 0.396(3) µs⁻¹ | none prescribed (§9) | Envelope needed to fit (§7). |
| A₁ (amplitude) | 14.68(5) % | — | ~15 % initial ZF asymmetry. |
| A_bg | 0.51(3) % | — | Small residual baseline. |
| χ²ᵣ | **1.42** (ndof 1956; panel flags "poor") | refprog ≈ 1.8 (§7) | ✓ comparable, slightly better. |
| Implied ν_d | ≈ 0.16 MHz → lines ~0.10/0.28/0.39 MHz | ~0.14/0.39/0.53 MHz at r=1.17 (§6) | FFT cluster peaks ~0.47 MHz — right sub-MHz regime. |

**Honest read on r:** the fit sits at **r_µF ≈ 1.30 Å**, ~8–15 % above the
literature 1.1–1.2 Å band, exactly matching the previous Asymmetry analysis
(§7). This is a real model/geometry effect, not a fluke — see caveats.

## Model-choice caveats (honest)

- **r lands high (≈1.30 vs 1.1–1.2 Å).** The collinear `FmuF_Linear` assumes a
  *symmetric, exactly linear* F–µ–F unit with two equivalent F at distance r_µF.
  In PTFE the geometry is not perfectly collinear/symmetric; forcing the collinear
  model onto a slightly bent or asymmetric unit, together with the Gaussian
  envelope absorbing part of the early-time curvature, biases r upward. GT §6
  notes the refprog value is "slightly larger than" the literature expectation.
  `FmuF_General` (r₁, r₂, θ) or `FmuF_Triangle` (adds a third F) are available for
  non-collinear geometries and would be the route to test whether the excess is
  geometric — left as a refinement (they are EXPENSIVE powder-averaged models).
- **Gaussian vs Exponential envelope.** The synthetic PbF₂ scenario
  (`muon_fluorine_pbf2`) uses `FmuF_Linear * Exponential + Constant` (λ = 0.3
  µs⁻¹); this corpus example follows the reference-program's `* Gaussian`
  (GT §7), which fits PTFE better (χ²ᵣ 1.42). The envelope choice is **not**
  prescribed by the guide (§9) — it is a modelling decision that follows the
  refprog note.
- **Single run, not a T-trend.** The guide poses no temperature-dependence
  question (§9); the FµF coupling is nuclear and only weakly T-dependent, so a
  headline r_µF(T) trend would be nearly flat and low-value. Warm-starting is also
  needed off base T: with fixed guidance seeds, run 17319 (25 K) walks r_µF to the
  2.0 Å bound (bad local minimum). The single base-T fit is the robust, defensible
  headline; no trend scenario is shipped.

## Feature-demonstration opportunities

Captured: raw muon-specific FµF beating on real data, the converged
`FmuF_Linear * Gaussian + Constant` single fit (a distinctive, muon-specific
analytical model that few of the other corpus examples exercise), a sub-MHz ZF
FFT with the apodisation controls, and a TF calibration precession.

Not captured but available on this example:
- **`FmuF_General` / `FmuF_Triangle` fit** — testing whether relaxing the
  collinear/symmetric constraint pulls r toward 1.1–1.2 Å; the natural
  "geometry matters" companion figure.
- **MaxEnt of the ZF run** — the sub-MHz FµF line cluster is broad under FFT;
  MaxEnt (Frequency → MaxEnt tab) could sharpen the three combination lines.
- **Temperature waterfall** (17294 → 17309, 20 → 200 K) — the FµF oscillation
  persisting across the whole ZF scan (weak T-dependence), a contrast to the EuO
  order-parameter waterfall.

## Problems hit

1. **No LF data → the optional "LF decoupling" scenario is not possible.** Every
   analysis run in this set is zero-field (GT §3: field = 0 for 17294–17322); the
   only applied-field run is the TF 20 G calibration 17293. The brief's optional
   LF-evolution figure would require longitudinal-field data that this corpus
   example does not contain — so `corpus_ptfe_tf_calibration` (a real, guide-
   prescribed workflow step, §4) is shipped as the 4th scenario instead. Flagged
   rather than faked.
2. **`Data_hdf5/` mirror gone.** The reference-program note (`ANALYSIS_asymmetry.md`)
   ran on a `Data_hdf5/` copy that is no longer on disk (GT §2, §8); the HDF4
   `.nxs` files in `Data/` load natively and reproduce the refprog r_µF exactly,
   so this is cosmetic.
3. **FFT DC skirt clips the top of the frame.** The FµF combination lines sit
   sub-MHz on the decaying-envelope DC peak; Y is framed to 1.25× the cluster peak
   (away from the 0 MHz bin) so the FµF structure is legible, which lets the DC
   skirt run off the top edge. Intentional — the caption is about the cluster near
   0.3–0.5 MHz, not the DC term.
4. **Off-base-T fits need warm-starting** (see caveats): fixed guidance seeds send
   run 17319 to the r_µF bound. Only the base-T single fit is captured, which is
   robust from the 1.15 Å seed.

## Top pick for the docs

`corpus_ptfe_fmuf_fit` — the converged `FmuF_Linear * Gaussian + Constant` fit
with r_µF ≈ 1.30 Å visible in the panel is the headline: it exercises Asymmetry's
distinctive muon-specific FµF analytical model on real entangled-spin data and
delivers the bond length the guide asks for. `corpus_ptfe_zf_signature` (the raw
dip-and-recovery beating) is the strongest supporting frame — the cleanest visual
of the FµF three-spin oscillation in the set.
