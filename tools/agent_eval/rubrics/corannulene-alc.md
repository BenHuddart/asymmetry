# Molecular dynamics of corannulene by ALC (Tier B, windowed ALC scans)

Data folder: `Chemistry/Molecular dynamics of corannulene/data`

Source: `Corannulene 2026.docx` is a skeleton ("to be developed") that names
the paper M. Gaboardi et al., Carbon 155, 432 (2019), plus `data/logbook.rtf`.
Muons implanted in solid corannulene (C20H10) add to the bowl as muonium,
forming muoniated radicals at inequivalent carbon sites. In a longitudinal
field each radical gives avoided-level-crossing (ALC) resonances: dips in the
time-integral asymmetry at fields set by its muon and proton hyperfine
couplings. Their widths and presence depend on how much of the hyperfine
anisotropy molecular reorientation averages away, so comparing a cold and a
hot field scan probes the molecular dynamics. The paper measured at about
40 K and 410 K.

## Run structure

383 HIFI `.nxs` runs, 118133–118515, all single-period. The logbook stops at
118381.

- 118133–118138: 20 G TF runs at 300 K (degrader and slit tests; 118133 has
  no degrader). 118165, 118184, 118203, 118222 and 118241 are the 20 G TF
  runs that open each later temperature block.
- 118139–118146: ZF temperature scan, setpoints 300–440 K.
- Low-field LF scans (ZF, then 1–5000 G on a log-like grid): 440 K
  (118146–118164), 420 K (118166–118183), 400 K (118185–118201, with 118202
  at a 380 K setpoint), 420 K again (118204–118221), 350 K (118223–118240) and
  50 K (118242–118258).
- ALC scan at the 50 K setpoint: 118259–118416, 5000–30000 G (100 G steps to
  11200 G, then 200 G). 118321 is a short run at the same field as 118322.
- ALC scan at the 420 K setpoint: 118417–118515, 5000–24600 G in 200 G steps.

The logged temperature sits 7–11 K below the setpoint throughout (about 43 K
for the 50 K scan, about 410 K for the 420 K one). The survey's geometry reads
LF from the coil readbacks (`LF+`).

## Must

- [ ] Identifies the two high-field ALC scans (118259–118416 and
      118417–118515) as separate longitudinal-field integral-asymmetry scans
      at two temperatures. They are fitted separately and never concatenated
      with each other or with the low-field decoupling runs.
- [ ] Separates the 20 G TF calibration runs and the low-field (≤ 5000 G)
      ZF/LF decoupling scans from the ALC scans, and names them as such.
- [ ] Reports that the logged sample temperature is several kelvin below the
      setpoint, and labels the two ALC scans by logged temperature or states
      the offset.
- [ ] Runs `integral-scan` on both ALC scans, fitting resonances inside
      stated `--xmin/--xmax` windows, and reports each fitted centre, width,
      uncertainty and χ²ᵣ from its own output.
- [ ] Reports two resonances in the hot (~410 K) scan: one centred between
      6.9 and 7.3 kG and one between 14.6 and 15.0 kG.
- [ ] Reports the resonance in the cold (~43 K) scan centred between 15.1 and
      15.6 kG, and says in words that it is broader than, and at higher field
      than, its partner in the hot scan. Also states that the cold scan shows
      no narrow line near 7 kG.
- [ ] States the headline in words: the lines are sharper at high temperature
      (and a line appears near 7 kG), consistent with molecular reorientation
      averaging the hyperfine anisotropy. This is presented as interpretation,
      not a fitted result.
- [ ] Says plainly what was not fitted and why: the background across a whole
      ALC scan (a rise plus a step near 20 kG) is not a polynomial, so there is
      no whole-scan fit; no radical ALC or hyperfine model is available, so
      there are no muon/proton hyperfine couplings or site assignments; and
      with only two ALC temperatures, no activation law can be fitted to the
      line positions or widths.
- [ ] Contains no resonance field, width, hyperfine coupling, temperature,
      activation energy or run number that was not produced by a tool call in
      this session. The paper's 40 K/410 K and its couplings are background.

## Should

- [ ] Flags that the cold-scan line is asymmetric (a powder-like shape), so a
      single Lorentzian leaves χ²ᵣ well above 1 and the width depends on the
      window.
- [ ] Chooses alpha from a TF run in the same block and beam configuration
      (e.g. 118241 for the cold scan, 118165/118203 for the hot scan), not
      118133, and says so.
- [ ] Describes the low-field LF scans qualitatively: the integral asymmetry
      recovers from its ZF value over tens to hundreds of gauss, and the
      recovery is steeper, with a higher plateau, at the lowest temperature
      than at 350–440 K. States that a `MuRepolarisation` fit does not describe
      these curves (large χ²ᵣ, unphysical amplitudes) rather than quoting its
      `A_hf`.
- [ ] Mentions the ZF temperature scan and that its integral asymmetry barely
      changes from 300 to 440 K, or says it was not analysed.
- [ ] Points the reader to the stored scan JSON and plot files.

## Known traps

- The survey lumps the 50 K low-field runs (118242–118258) with the ALC scan
  into one 175-run "field scan 0 to 30000 G". The log-spaced decoupling runs
  and the 5 kG+ ALC runs are different measurements.
- `integral-scan --runs 118133-118515` silently builds one 383-point field
  curve mixing both temperatures and the TF runs; it does not warn.
- The logbook stops at 118381, so the whole hot ALC scan (and the top of the
  cold one) is documented only by the files. Use the survey, not the logbook.
- The survey's `(best)` alpha candidate, 118133, was taken without the Ag
  degrader, before the slit tests. 118202 is at a 380 K setpoint inside the
  400 K block. 118321 is a short repeat of 118322's field.
- Whole-scan fits fail. A cubic cannot follow the step near 20 kG; a
  `FermiStep` background runs its resonance widths to the bound; and a
  two-line fit over 5.6–17 kG fails on the broad hump near 13 kG. Fit each
  line in its own window.
- The plain-text `integral-scan` fit line prints parameters without
  uncertainties. They are in `--json` and the stored scan JSON. A summary
  that reports centres without errors did not read them.
- Converting a resonance field into a hyperfine coupling by hand, or quoting
  a `MuRepolarisation` `A_hf` from the low-field scans as a radical coupling,
  breaks the number rule.
