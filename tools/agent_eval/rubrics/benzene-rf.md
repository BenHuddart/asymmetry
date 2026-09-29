# Muoniated radical in benzene: RF resonance (Tier B, integral scan)

Data folder: `Chemistry/Muon spectroscopy of benzene/data/RF resonance`

Worksheet: `Benzene 2026.docx`, "Analysis of RF Data". The C6H6Mu radical in
liquid benzene is swept through a longitudinal field while an RF field at
218 MHz (the notes' value) drives muon spin transitions; on resonance the
muon polarisation drops. The data are red/green two-period runs (the worksheet
says green = RF off, red = RF on) and the resonance lives in the green − red
difference of the time-integral asymmetry: two peaks, whose mean position
sets the muon coupling A_μ and whose separation sets the methylene proton
coupling A_p. The field is low enough that the high-field linear relation
fails; the worksheet asks for a full-Hamiltonian fit giving A_μ and A_p, and
a comparison with the analytic approximation.

## Run structure

37 ISIS DEVA (`MUT`) `.nxs` runs, 56426–56462, all two-period, 293 K, fields
560–1080 G, notes "Benzene in cell, 218MHz, 60W". Three interleaved passes,
so run order does not track field:

- 56426–56439: 560–1080 G in 40 G steps.
- 56440–56452: 580–1060 G, filling between them.
- 56453–56462: 770–950 G in 20 G steps, the infill across the resonances.

No run precesses (`prec none`, geometry `-`): the survey's single field scan
reads "unknown geometry". There are no alpha-calibration candidates. The files
carry deadtime values.

## Must

- [ ] Identifies one longitudinal-field RF-resonance field scan of 37 runs at
      293 K, two periods per run, with the 218 MHz RF from the notes, and says
      that run order is not field order.
- [ ] States that no run in the folder can calibrate alpha and that alpha was
      assumed to be 1.0; states the deadtime treatment.
- [ ] Runs `integral-scan` with `--period green-red` and describes two
      resonance peaks in the difference, both between about 0.74 and 0.90 kG,
      on a flat baseline outside them.
- [ ] Fits `RFResonanceMuP` with `nu_RF` held at 218 MHz and reports A_μ
      between 510 and 520 MHz and A_p between 115 and 135 MHz, each with its
      uncertainty, and the fit's χ²ᵣ.
- [ ] Contains no A_μ, A_p or resonance field that was not produced by a tool
      call in this session: no couplings from the mean and difference of two
      fields by a formula, and the published couplings are background.

## Should

- [ ] Also fits the worksheet's model, two Lorentzians on a constant, seeded
      near the two peaks, and reports the two fitted centres; notes the fit
      agrees in quality with `RFResonanceMuP`.
- [ ] Says `RFResonanceMuP` is the exact-diagonalisation model and that the
      analytic high-field approximation the worksheet compares against is not
      available in the CLI (or was not run).
- [ ] Says which period is RF-on and on what evidence (the sign of the
      green − red peaks, or an assumption stated as such).
- [ ] Compares A_μ and A_p with the published or worksheet values in words,
      attributing them.
- [ ] Points to the stored scan JSON and PNG.

## Known traps

- Survey `unknown geometry` and `prec none` do not mean a failed run: an RF
  scan is longitudinal, and nothing precesses at the applied field.
- A single-period integral (no `--period`) mixes RF-on and RF-off counts; the
  resonance is the green − red difference.
- The two-Lorentzian model auto-seeded as dips fits the positive peaks upside
  down (χ²ᵣ ~22, widths at the upper bound); it needs `--initial` seeds; its
  widths print `(at bound)`.
- The worksheet says the high-field linear relation is inaccurate here, so a
  hand conversion of the peak fields into couplings is wrong physics as well
  as a number-rule failure.
