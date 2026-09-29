# Muoniated radical in benzene: ALC resonances (Tier B, sub-folders + integral scans)

Data folder: `Chemistry/Muon spectroscopy of benzene/data/ALC resonance`

Worksheet: `Benzene 2026.docx`, "Analysis of ALC Data" (liquid and solid; the
`solution` sub-folder is not described in it). At high longitudinal field the
C6H6Mu radical's muon–electron–nuclear levels cross; each avoided crossing is
a dip in the time-integral asymmetry against field. Δ0 resonances (muon and
one proton flip-flop) are narrow and set by A_μ − A_p, so each proton gives
its own line; the Δ1 resonance (muon alone) needs an anisotropic coupling and
appears only in the solid. The worksheet asks for the resonance fields and
widths and converts them to A_μ, D_μ and A_p with formulas — the CLI prints
fields and widths, not couplings.

## Run structure

The folder holds no runs directly: `survey` exits 1 naming `liquid` (263
runs), `solid` (123) and `solution` (360). All HIFI `.nxs`, deadtime in files.

- **liquid**, 300 K. 29592–29721: "CHMu(0) scan", 19.0–22.475 kG, four
  interleaved passes. 29723–29798: "o-p scan", 28.5–30.0 kG, five passes.
  29800–29854: "o-p scan repeat with RG -5A", two-period red/green runs over
  28.5–30.0 kG (29800–29808 repeat the fields of 29809–29817; 29829–29831 are
  stamped TF). 20 G TF calibrations 29722 and 29799 (alpha about 1.19).
- **solid**, setpoint 200 K, notes "warming". 59732–59853: one 17.0–23.05 kG
  scan in two interleaved passes. 59731: 20 G TF calibration (alpha about
  1.27), logged 157 K; the first scan runs log 2–3.5 K above setpoint.
- **solution**, three samples, each a set of short (~200 G) scans at several
  temperatures, taken by sweeping a Z coil about a persistent main field:
  "Benzene(aq)" 77766–77854 (275–350 K, main field 20900 G), benzene in
  hexane "C6H6/C6H14" 83929–84120 (180–335 K), "C6H6/H2O 20mM" 84122–84200
  (280–360 K; main field 20800 G for both June samples). Titles' `F=` is the
  coil offset; the survey's B/G is the total. No calibration run.

## Must

- [ ] Recognises the three sub-folders and analyses each separately, with its
      own work directory, never mixing runs between them.
- [ ] Treats every folder as longitudinal-field ALC scans fitted with
      `integral-scan`, uses the 20 G TF runs only for alpha (naming the run
      and value), and says alpha was assumed 1.0 for `solution`.
- [ ] Liquid: reports from `integral-scan` fits one resonance in the CHMu(0)
      scan between 20.7 and 20.9 kG and two in the 28.5–30 kG scan, one
      between 28.9 and 29.0 kG and one between 29.5 and 29.6 kG, with
      uncertainties or fit quality.
- [ ] Does not integrate the red/green repeat 29800–29854 as an ordinary
      scan: either fits it as `--period green-red` with a pair model and the
      red–green field step held from the command's printed offset, or sets it
      aside and says why.
- [ ] Solid: reports two resonances from converged fits (a joint fit, or
      one windowed fit per resonance), one between 19.3 and 19.7 kG and one
      between 21.3 and 21.6 kG. A failed fit's position is not a fitted
      resonance.
- [ ] Solution: names the three samples and fits at least one scan, stating
      sample and temperature, with a resonance between 20.4 and 21.2 kG.
- [ ] Contains no A_μ, A_p, D_μ or resonance field that was not produced by a
      tool call in this session: no couplings from resonance fields by the
      worksheet's formulas.

## Should

- [ ] Solid: from a converged joint fit of both resonances, notes that the
      lower one is the broader (separately windowed fits with their own
      backgrounds can hide the difference).
- [ ] Assigns in words: liquid lines are Δ0 (the 20.8 kG one the methylene
      CHMu proton, the pair near 29 kG ring protons); the solid's broad lower
      line is the Δ1 muon resonance and the narrower upper one a Δ0 line.
- [ ] Solution: fits several temperatures of one sample and reports the
      resonance field rising with temperature, as a list of fitted fields.
- [ ] Solid: uses the logged sample temperature and notes the calibration run
      was taken colder than the scan.
- [ ] Says the worksheet's time-domain questions (relaxation vs field,
      on-resonance oscillations) were or were not examined.
- [ ] Names the stored scan JSON and PNG for each fitted scan.

## Known traps

- `--alpha-from 29799` with any `--period` fails — "The green − red
  difference needs a two-period (red/green) run", or "Unknown period selector"
  for `--period green`: the period is applied to the single-period
  calibration run too. Pass `--alpha` with the value `alpha` printed instead.
- The two-period repeat integrated without `--period` places the ring-proton
  lines ~35 G higher than the single-period scan: red and green sit ~44 G
  apart. With the pair offset free the fit fails; the command prints the fix.
- In the short solution scans the Lorentzian width sits at its bound on
  nearly every scan (`integral-scan` prints `(at bound)`): those widths are
  not measurements.
- The survey lists 91 "temperature scans" through the solution field scans'
  points; they are cross-sections, not measurements.
- The two solution campaigns use different persistent main fields ("new
  persistent ref"); absolute fields compared across samples need that caveat.
  Within one sample the temperature trend is sound.
