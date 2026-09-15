# The F-mu-F state in PTFE (Tier A)

Data folder: `Nuclear magnetism and ionic motion/The FmuF state in PTFE/Data`

## Experiment

PTFE has no double bonds, so implanted muons stay diamagnetic and couple
to nearby fluorine nuclei, forming a closely-coupled F-mu-F three-spin
system. The worksheet asks for the F-mu-F relaxation function to be fit
across a temperature range to extract the dipolar coupling frequency and
check it is roughly temperature independent.

## Run structure

30 MUSR `.nxs` runs, 17293-17322:

- Calibration, 17293: TF 20 G, 5 K.
- ZF temperature series, 17294-17322 (29 runs): 20-200 K, taken out of
  temperature order (coarse pass to 200 K then several revisit/infill
  temperatures such as 115, 125, 135, 145, 155, 165, 175, 185, 195,
  25, 35, 45, 55 K).

## Must

- [ ] Identifies run 17293 as the TF calibration run and the remainder
      as the ZF temperature series.
- [ ] States that alpha was determined from 17293 (or explains why not,
      if it used a different run).
- [ ] Recommends or applies an F-mu-F–family relaxation function
      (three-spin dipolar / Brewer-type), not a generic Kubo-Toyabe or
      exponential, for the ZF data.
- [ ] Reports the muon–fluorine dipolar coupling extracted from the fits,
      as the coupling frequency or as the muon–fluorine distance `r_muF`
      (the CLI parameterises the F–μ–F family by `r_muF`, with ω_D ∝ r⁻³).
- [ ] States whether that dipolar coupling frequency is roughly
      constant (temperature independent) or varies across the scan.
- [ ] Contains no coupling-frequency, alpha, or temperature number that
      was not produced by a tool call in this session.

## Should

- [ ] Notes that the ZF runs are not in monotonic temperature order and
      that later runs revisit/infill earlier temperature gaps, rather
      than assuming run number tracks temperature.
- [ ] Mentions the sample-temperature column (with its uncertainty) as
      distinct from the nominal set-point temperature.
- [ ] References the muon-fluorine "hydrogen bonding" / entanglement
      picture from the worksheet background when interpreting the
      result.

## Known traps

- ISIS MUSR files can stamp a nonzero field even on runs intended as
  ZF; the survey's field column, not the filename or position in the
  sequence, should decide which run is the TF calibration.
- Run order is not temperature order — sorting by run number and
  reading off "the scan" without checking the temperature column will
  misdescribe the coverage.
- The single calibration run (17293) is at 5 K, not at a
  representative mid-range temperature; alpha derived there still
  applies across the set, but the summary should not imply the
  calibration was repeated at each temperature.
