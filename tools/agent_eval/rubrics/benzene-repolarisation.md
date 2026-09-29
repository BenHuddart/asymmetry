# Muoniated radical in benzene: LF repolarisation (Tier B, integral scan)

Data folder: `Chemistry/Muon spectroscopy of benzene/data/Repolarisation`

Worksheet: `Benzene 2026.docx`, "Analysis of Repolarisation Data". In zero
field the muon in the C6H6Mu radical shares its polarisation with the unpaired
electron (and the protons); a longitudinal field decouples them, and the
time-integral asymmetry rises towards full polarisation. The worksheet asks
for a fit of the integral asymmetry against field with the muonium
repolarisation function, giving the characteristic field B0 and an effective
hyperfine parameter A — "a broad overall picture of the coupling strengths".
The CLI's `MuRepolarisation` reports that A (`A_hf`) directly; it does not
print B0. A repolarisation model for a radical with proton couplings (the
worksheet's reference) is not in the CLI.

## Run structure

44 EMU `.nxs` runs, 15958–16001, liquid benzene, 300 K (logged ~300.4–300.8 K),
deadtime values in the files:

- Six short 100 G weak-TF runs, `TF* larmor`, interleaved through the
  session: 15958, 15966, 15976, 15986, 15994, 16000. These are the alpha
  calibrations (alpha 1.31–1.33 on each), not scan points.
- 38 longitudinal runs, 3–4000 G, rising: 15959–15965, 15967–15975,
  15977–15985, 15987–15993; then a return pass repeating 3000, 2000, 1000,
  500, 100 G (15995–15999) and 10 G (16001).

The survey's field scan counts the 38 LF runs but prints "TF on 4 of 38
runs": 15991–15993 and 15995 (2700–4000 G) are file-stamped TF above the
Nyquist limit (`prec -`), not measured.

## Must

- [ ] Identifies a longitudinal-field scan of the integral asymmetry from a
      few gauss to 4 kG at 300 K, the six interleaved 100 G transverse runs as
      calibration (excluded from the scan), and the repeated fields of the
      return pass.
- [ ] Establishes the LF geometry of the high-field runs despite their TF
      stamp, and says how.
- [ ] Measures alpha on one of the 100 G TF runs, names it and the value, and
      states the deadtime treatment.
- [ ] Runs `integral-scan` with `--model MuRepolarisation` over the LF runs
      and reports `A_hf` with its uncertainty, the χ²ᵣ and the field range
      fitted; says plainly how well the function describes the curve (a
      one-term fit over the whole range is poor).
- [ ] Describes the curve itself: the integral asymmetry rises steadily with
      field over the whole scan and is still rising at the highest field.
- [ ] Contains no B0, A, coupling or fraction that was not produced by a tool
      call in this session: no B0 converted from `A_hf` (or back) by hand, and
      no literature coupling presented as a result.

## Should

- [ ] Says the muonium repolarisation function treats the radical as a
      two-spin muon–electron system, so `A_hf` is only an effective coupling;
      the radical's proton couplings broaden the curve, and a radical
      repolarisation model is not available in the CLI.
- [ ] Notes the fitted diamagnetic term `a_Dia` is negative, i.e. unphysical
      as a fraction, as a symptom of that mismatch.
- [ ] If a sum of two `MuRepolarisation` terms is fitted, reports both with
      χ²ᵣ but calls it an empirical description, and notes the two `a_Dia`
      terms are the same constant (degenerate).
- [ ] Notes that `A_hf` moves with the fitted field range (`--xmax`), so the
      number is not robust.
- [ ] Notes that repeated fields from the return pass differ by more than
      their errors (a drift between passes), limiting χ²ᵣ.

## Known traps

- The skill says a sum of two `MuRepolarisation` terms "describes two
  muoniated species". Benzene gives one radical (no free muonium survives);
  the two-term fit gives a second `A_hf` above the vacuum muonium coupling,
  which no species has. Calling the two terms two species is wrong.
- Including the 100 G TF runs as scan points puts precessing (near-zero
  integral) points into the curve.
- The survey's "TF on 4 of 38 runs" is the file stamp on above-Nyquist runs,
  not a mixed-geometry scan.
- The worksheet's "A = (ge + gμ) B0" is a conversion; doing it by hand on a
  printed number fails the number rule.
