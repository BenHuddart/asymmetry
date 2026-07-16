# NOTES — Muon spectroscopy of benzene (Chemistry, multi-technique)

Module: `benzene_multi.py` · Example: `Chemistry/Muon spectroscopy of benzene`
Spec: the example's `GROUND_TRUTH.md` (guide `Benzene 2026.docx` + `RFpaper.pdf`
= McKenzie *et al.*, *J. Phys. Chem. B* **117**, 13614 (2013)).

The corpus's four-technique showcase for the muoniated **cyclohexadienyl radical**
C6H6Mu: High-TF muon spin rotation (GPD@PSI `.bin`), ALC resonance (HiFi@ISIS
`.nxs`), low-field repolarisation (EMU@ISIS `.nxs`), and RF resonance (MUT@ISIS
`.nxs`). One radical, four measurements, three instruments — every scenario
reaches the same hyperfine coupling A_µ from a different angle.

## Hard targets (GROUND_TRUTH §4 / §11)
- **A_µ (benzene, RF-µSR) = 514.78(4) MHz** — paper Table 1 (primary grading value).
- **A_p (benzene, RF-µSR) = 124.6(14) MHz** — paper Table 1.
- High-TF FFT guide reading: diamagnetic ≈41 MHz, radical lines ≈209 / ≈306 MHz,
  ν1+ν2 = A_µ ≈ 515 MHz; PSI cyclotron artefact ≈51 / ≈102 MHz (GT §3A).
- Liquid ring-proton Δ0 dips (ref `test.sel`, GT §7): ≈28.94 / ≈29.54 kG
  (digitised targets ≈28.97 / ≈29.57 kG, GT §3B narrative).

## Scenarios registered

| Scenario | Technique | Render (intended docs use) |
|---|---|---|
| `corpus_benzene_hightf_fft` | High-TF | Real GUI **Fourier panel** on the 3000 G co-add (runs 3678–3682): diamagnetic + two radical precession lines annotated; ν1+ν2 = A_µ. Frequency-domain data-analysis step. |
| `corpus_benzene_correlation` | High-TF | Real GUI Fourier panel, WiMDA **radical correlation** display: the Breit–Rabi line pair collapsed onto a single A_µ peak on the hyperfine axis. The distinctive high-TF render + headline A_µ. |
| `corpus_benzene_liquid_alc` | ALC | Real GUI **ALC integral-scan view**: 76 HiFi ring-proton runs → integral asymmetry vs field, Cubic BG + 2 Lorentzians; the two Δ0 dips fitted. Core ALC analysis step (`requires_fit`). |
| `corpus_benzene_repolarisation` | Repolarisation | Standalone figure: EMU integral asymmetry vs LF field (3–4000 G) — the muonium repolarisation rise. Deliverable/overview. |
| `corpus_benzene_rf_resonance` | RF | Standalone figure: RF field scan + `RFResonanceMuP` (muon+proton spin-Hamiltonian) fit of the W-shaped double dip → A_µ, A_p (`requires_fit`). |

All four techniques covered; two headline A_µ routes (correlation peak; RF fit).

## Run selection & workflow (GT § refs)

- **High-TF (§3A).** Co-add the five 3000 G high-statistics runs
  `deltat_tdc_gpd_3678–3682` at the count level (`combine_runs`, sign=+1), then
  the averaged grouped FFT (`compute_average_group_spectrum`) over all four GPD
  groups: Lorentzian apodisation τ=3 µs, 8× zero-padding, t = 0–7 µs (GT §3A
  prescribes apodise 1.5→7 µs, 8× pad). Plain `(Power)^1/2` display gives the
  three lines; `correlation` display (WiMDA `Corr`/`AvCorr`, exact Breit–Rabi
  forward map) collapses the radical pair onto the single A_µ peak. dt = 0.586 ns
  → Nyquist ≈ 853 MHz, so the 306 MHz line is fully resolved (no rotating frame).
- **ALC liquid (§3B / §7).** Ring-proton (Δ0) window `hifi00029723–29798`
  (≈28.5–30.0 kG, HiFi titles "o-p scan"). Integral-asymmetry field scan
  (`build_field_scan`, method=integral) in the GUI ALC view; Cubic baseline over
  the two non-resonant edges `[(28500,28850),(29650,30000)]` + two Lorentzian
  dips seeded near 28.92 / 29.52 kG. This is exactly the window the reference
  `test.sel` covers (GT §7).
- **Repolarisation (§3C).** EMU LF sweep `EMU00015958–16001` (3–4000 G). Built
  from the curated **primary ascending sweep** (`_REPOL_PRIMARY`, one clean run
  per field). GT §7: the shipped `Repolarisation/test.sel` is a **misfiled
  liquid-ALC scan** (28.5–30.0 kG) — there is no reference repolarisation trend,
  so the curve is built directly from the EMU runs (no numeric target; GT §6).
- **RF (§3D / §11).** MUT field scan `56426–56462` at fixed ν_RF = 218.5 MHz,
  560–1080 G, 293 K. Integral-asymmetry field scan + `fit_rf_resonance`
  (`RFResonanceMuP`, ν_RF held fixed): dip **mean → A_µ**, dip **splitting → A_p**.
  Digitised Fig. 3a dips (773 / 865 G, GT §11) marked for reference.

## Fitted values vs targets

| Quantity | This module | Target (GT) | Source |
|---|---|---|---|
| High-TF diamagnetic line | **40.7 MHz** | ≈41 MHz | GT §3A |
| High-TF radical ν1 | **208.7 MHz** | ≈209 MHz | GT §3A |
| High-TF radical ν2 | **305.5 MHz** | ≈306 MHz | GT §3A |
| High-TF A_µ = ν1+ν2 | **514.2 MHz** | ≈515 / 514.78 | GT §3A / §4 |
| Correlation A_µ peak | **514.4 MHz** | 514.78(4) | GT §4/§11 |
| ALC Δ0 dip 1 (B0_1) | **28.94 kG** (±0.4 G) | ≈28.94 / 28.97 | test.sel / §3B |
| ALC Δ0 dip 2 (B0_2) | **29.54 kG** (±0.5 G) | ≈29.54 / 29.57 | test.sel / §3B |
| RF A_µ (dip mean) | **516.2 MHz** | 514.78(4) | GT §11 |
| RF A_p (dip splitting) | **135 MHz** | 124.6(14) | GT §11 |
| Repolarisation | shape only (19→35 %) | no numeric target | GT §3C/§6 |

Three independent A_µ determinations (correlation 514.4; FFT sum 514.2; RF fit
516.2) bracket the 514.78 MHz target within ~1.5 MHz — a strong multi-technique
consistency story.

## Technique-coverage & feature-demonstration notes

- **Radical correlation spectrum** (`corpus_benzene_correlation`) is the standout:
  a WiMDA `Corr`/`AvCorr` port that no other corpus example exercises, and it
  reads A_µ straight off the hyperfine axis at 514.4 MHz. Top pick.
- **`RFResonanceMuP`** is the model behind GROUND_TRUTH's parity check **PC1** —
  and it is **present in Asymmetry**, both as `core.fitting.field_scan.fit_rf_resonance`
  and as an **"RF RESONANCE (A_M, A_P)"** section visible in the GUI ALC
  integral-scan sidebar (see the `corpus_benzene_liquid_alc` render). PC1 is
  therefore *not* a missing-parity gap for this program; worth recording against
  PC1 in `parity-checks.md`.
- **Count-level co-add** (`combine_runs`) of five PSI `.bin` runs feeding a single
  FFT/correlation is a data-handling capability the FFT scenarios demonstrate.
- All four ISIS/PSI formats load cleanly through the one `load()` path:
  GPD `.bin`, HiFi/EMU/MUT `.nxs` (two-period RF included).

## Problems / honesty

- **GUI Fourier-panel annotations are wiped by `widget.show()`.** The panel
  replots on show (it even overrides a manual `set_xlabel`), so build()-time
  axvlines/labels vanished on first capture. Fixed by moving framing + annotation
  into `settle()` (after the show-time replot); values are computed in `build()`
  and stashed on the instance. (New lesson-learned candidate for the README.)
- **RF (Green − Red) difference scan is too noisy to render** (red≈green means,
  ~0.5 % difference → scatter swamps the W-dip). The scenario instead scans the
  **loaded single period (RF-on)** integral asymmetry, which shows the clean
  W-dip; `RFResonanceMuP` fits it to A_µ = 516.2, A_p = 135. Consequences: A_µ
  sits ~1.5 MHz high and A_p ~10 MHz high vs Table 1, and the flat-BG model does
  not capture the real upward baseline slope on the wings (dip *positions* are
  robust, wing baseline is not). The migrad errors are correspondingly
  overconfident (sub-0.1 MHz), so the annotation quotes values only, not errors.
  A_p from RF is intrinsically weak here — the paper itself uses RF for A_µ and
  takes A_p from the ALC resonance field (GT §6). Building a true Green−Red scan
  with proper period handling is a possible future refinement.
- **Repolarisation 100 G outlier.** Eight interspersed "start of cycle" 100 G
  monitor runs read a distinct low value (~13.5 %, evidently a different
  acquisition state) versus the physical ~20 % point (runs 15972/15999). A naive
  keep-first-per-field dedup grabbed the wrong 100 G run and left a spurious dip;
  the module uses an explicit curated ascending sweep (`_REPOL_PRIMARY`, 32 runs)
  instead. No repolarisation-function fit is shown (GT §6: no numeric B₀ target,
  and no repolarisation model is registered in the field-scan fitters).
- **ALC interleaved sawtooth.** The ring-proton window is five interleaved 100 G
  sub-scans; sorted by field, adjacent points come from different sub-scans and
  show a small baseline-drift zigzag (the guide/GT §7 suggest "Make differential
  ALC" for this). The Cubic BG + 2-Lorentzian fit averages through it cleanly and
  recovers both dip fields to <0.5 G, so no differencing was needed here. The
  **methylene Δ0 window** (~20.79 kG, runs 29592–29721) and the **solid-phase D1**
  scans (D_µ) were not captured — single-dip/second-render opportunities left for
  a future pass; ring-proton was chosen for its clean two-dip render matching the
  reference `test.sel`.

## Rebase onto main (PR #264) — 2026-07-16 — FFT annotations kept in settle()

- **`corpus_benzene_hightf_fft` annotations stay in `settle()`.** #264's
  persistent-marker API (`add_persistent_frequency_marker` /
  `set_custom_x_axis_label`) survives the show-time replot, but it only covers
  **vertical frequency markers**. This scenario's annotation set also includes
  the `A_µ = ν₁ + ν₂` **axes-fraction summary box** and per-line text labels,
  which the persistent API cannot carry — arbitrary `ax.annotate` artists are
  still wiped by the replot. So drawing them in `settle()` (after the replot) is
  still required; no simplification. Verified the recapture still shows all
  annotations.
