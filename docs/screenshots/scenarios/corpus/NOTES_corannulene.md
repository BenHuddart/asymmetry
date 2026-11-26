# NOTES — Molecular dynamics of corannulene (Chemistry)

Module: `corannulene_ulcr.py` · Example: `Chemistry/Molecular dynamics of corannulene`
Spec: `GROUND_TRUTH.md` (Gaboardi et al., *Carbon* **155** (2019) 432–437).
Data: HiFi `HIFI00118xxx.nxs` (HDF4), runs 118133–118515 (383 files — the
largest set in the corpus).

## Scenarios registered

| Scenario | Render | Intended docs use | GT § |
|---|---|---|---|
| `corpus_corannulene_ulcr_scan` | **Real GUI** integral-scan view: 40 K wide µLCR scan (158 runs, 0.5–3.0 T) with the R3 dip carved out of the rising repolarisation baseline; ALC-scan fit panel visible. | Program-in-action / core-analysis surface on the corpus's widest field scan. | §3, §4.1–4.2 |
| `corpus_corannulene_resonance_fit` | Matplotlib: background-subtracted 40 K Δα scan, four Gaussian |ΔM|=1 dips fitted (R2/R3 joint doublet), each B_r → A_µ annotated vs paper. | **Headline** — hyperfine deliverable reproducing Table 1. | §4.3–4.4, §6a/§6b |
| `corpus_corannulene_temperature` | Matplotlib: 40 K vs 410 K subtracted Δα overlaid (offset, Fig. 4 style); R4/R3 narrow, R1/R2 broaden. | The "molecular dynamics" story (motional narrowing). | §4.6, §6a note |
| `corpus_corannulene_repolarisation` | Matplotlib: normalised integral asymmetry at 40 K and 410 K on a log field axis with B½ markers. | Half recovery in the low-field scans. | §4.5, §6d |

All three matplotlib scenarios are `requires_fit` where an iminuit fit runs at
capture time (resonance_fit, temperature). Capture is fast: GUI scan ≈ 15 s
(158-run load + reduction), the three figures ≈ 2–6 s each; whole flock run
≈ 27 s, all PNGs 76–370 KB (≤ 600 KB budget).

## Run selection (GROUND_TRUTH §3, confirmed against the loader field/T log)

The full-383 loader census reproduces §3's block table exactly (setpoint-T →
#runs, field span): 50 K → 176 (0–3.0 T); 420 K → 138 (0–2.46 T); 350/400/440 K
→ 19 each (≤0.5 T); 300 K → 7; single ZF points at 320/340/360 K, 2 at 380 K.

- **40 K** = setpoint 50 K, measured sample-T ≈ 42.7 K (`selog/Temp_Sample`).
  - Wide µLCR scan: **118259–118416** (5000–30000 G = 0.5–3.0 T; 100 G steps to
    11.2 kG then 200 G). All four resonances live here.
  - Repolarisation: **118242–118259** (0–5000 G, log-spaced 0,1,2,3,5…3500,5000 G).
- **410 K** = setpoint 420 K, measured ≈ 410 K.
  - Wide µLCR scan: **118417–118515** (5000–24600 G = 0.5–2.46 T; 200 G steps —
    "pure 200 G steps" as flagged).
  - Repolarisation: **118204–118221** (0–5000 G, log-spaced).

Note: the wide 410 K scan file block (118417–118515) only spans 0.5–2.46 T; its
low-field <0.5 T half is the separate log-spaced block 118204–118221 (used as
the 410 K repolarisation curve). Both dense scans start at 5000 G, so the T
overlay compares like-for-like from 0.5 T.

## Workflow followed

1. Per-run integral asymmetry (WiMDA integral method, `build_field_scan`,
   field-ordered) — same reduction the GUI's integral-scan view runs (GT §4.1).
2. Fit + subtract a **Quartic** repolarisation background over off-resonance
   windows bracketing the four dips (GT §4.2). Off-res windows (G):
   `[(5000,6200),(9000,13500),(20200,22800),(26500,30000)]` (40 K);
   drop the top window for 410 K (scan ends at 24600 G).
3. Fit the four |ΔM|=1 dips as **GaussianLCR** lines (all Gaussian at 40 K, GT
   §4.3): R4 (0.7 T) and R1 (2.44 T) singly; the overlapping **R2/R3 doublet**
   (1.53 / 1.8 T) jointly (two GaussianLCR) so the shoulder separates.
4. B_r → hyperfine via **eq. 2**, `A_µ[MHz] = B_r[G] / 36.713`
   (≡ B_r[T] = 0.0036713·A_µ), GT §1/§6.

## Fitted values vs ground truth

### µLCR resonances (40 K)

| Radical | B_r fitted (T) | A_µ fitted (MHz) | A_µ paper (MHz) | Δ from paper |
|---|---|---|---|---|
| **R4** | 0.696 | **190** | 192(11) | −2, within 1σ |
| **R3** | 1.536 | **418** | 419(10) | −1, within 1σ |
| **R2** | 1.780 | **485** | 484(20) | +1, within 1σ |
| **R1** | 2.449 | **667** | 665(15) | +2, within 1σ |

All four hyperfine couplings land **inside the paper's stated uncertainties** —
the joint doublet fit is essential for R2/R3 (a single-line fit over the R2
window collapses onto R3). Resonance fields match §6a (0.7/1.53/1.8/2.44 T) to
≈ 1 %.

### Temperature contrast (40 K vs 410 K, qualitative — GT §6a note, §6b widths)

Reproduced in the raw and subtracted scans: at 410 K the **R4 (0.7 T) and R3
(1.48 T) lines narrow into needle-sharp dips** (Lorentzian, molecular rotation /
pre-melting), while **R1 (2.44 T) all but vanishes** and R2 weakens — matching
"R4 narrows ≈ ×4, R3 narrows; R1, R2 broaden." R3 also shifts 1.53 → ~1.48 T at
410 K, as the paper reports. Widths are not point-graded (paper Fig. 4 y-axis is
arbitrary units, offset ±1); the narrowing/broadening *contrast* is the target,
and it comes through clearly.

### Half recovery

The [walkthrough](../../../workflows/alc_scan_tcnq.rst) gives the reference
intervals, rising brackets, and sensitivity of the half-recovery fields.

## Problems / caveats (honest)

- **Quartic baseline overshoots slightly** between R2 and R1 (a small positive
  bump near 2.1 T in the subtracted 40 K scan) and merges the R2/R3 region at
  410 K; it's a smooth-background artifact, not a resonance. The four dip
  positions are unaffected.
- **Field mislabelled `TF` in the NeXus metadata** (GT §9.7) — all 383 runs
  report `field_state = "TF"` though the µLCR/ALC scans are longitudinal-field
  sweeps. The applied-field *magnitude* is reliable; we treat the scans as LF
  (field-domain integral scan), which is correct.
- **Setpoint vs measured T** (GT §9.1): logbook/titles store the setpoint
  (50 K, 420 K); files' measured sample-T ≈ 42.7 K / 410 K = the paper's
  "40 K / 410 K". NeXus headers carry `experiment_identifier = 1700049` and the
  paper's own author list — this corpus data *is* the Gaboardi et al. experiment.
  Graded against the paper's 40 K / 410 K throughout.

## Feature-demonstration opportunities spotted (not all captured)

- The **GUI ALC-scan panel** (Baseline / Peaks / RF-resonance) is fully wired for
  this dataset — a future scenario could drive the in-GUI Quartic-baseline +
  Gaussian-peak fit (as `tcnq_alc.py` does the Cubic+Lorentzian path) instead of
  the standalone matplotlib fit, for an "analysis in the program" render.
- **Widest field scan in the corpus** (0–3 T / 30 kG) — good stress test for the
  integral-scan x-axis and the 158-run browser group.
- A **per-radical A_µ(site) bar chart** vs the DFT column (GT §6b, R1–R4 + the
  unobserved R5) would make a compact "assignment" figure if wanted.
