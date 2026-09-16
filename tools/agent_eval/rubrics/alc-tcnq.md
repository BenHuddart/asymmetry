# ALC resonance in TCNQ (workflow-expansion gate)

Data folder: `Chemistry/ALC resonance in TCNQ/Data`

This rubric replaces the former Tier C decline case. The CLI now has an
`integral-scan` workflow, so a passing agent must use it rather than declining
or forcing the individual runs through time-domain decay fits.

## Must

- [ ] Identifies this as a longitudinal-field avoided-level-crossing (ALC or
      µLCR) experiment in which each run contributes one time-integral
      asymmetry point versus applied field.
- [ ] Reports the folder structure from its own survey: 128 EMU runs comprising
      four 31-point scans over 2000–5000 G, plus the non-scan/calibration runs,
      without claiming the absent run 19618 was loaded.
- [ ] Actually runs `integral-scan` on at least one complete 31-run temperature
      block and states which block it analysed. It does not substitute
      `wizard`, `fit-series`, or individual time-domain fits for the ALC scan.
- [ ] Fits the field scan with a resonance-plus-smooth-background model. A
      single `LorentzianLCR` with a cubic background is acceptable; it does not
      add a second line merely because the teaching worksheet used a
      two-Lorentzian template.
- [ ] Reports an observed negative resonance feature close to 3.1 kG, together
      with the fitted centre, width and uncertainty/quality information from
      this run's command output. A centre outside 3.0–3.2 kG without an explicit
      failed-fit warning is a fail.
- [ ] Separates measured/fitted output from physical interpretation and does
      not invent hyperfine constants that the CLI did not calculate.

## Should

- [ ] Uses run 19485 as the detector-balance candidate, or explicitly states
      and justifies the alpha treatment it actually used.
- [ ] Uses non-resonant baseline regions on both sides of the dip and names the
      integration window and count-integral method.
- [ ] Analyses or at least maps all four temperature blocks, and describes any
      temperature-dependent width/amplitude claim only for blocks it actually
      fitted.
- [ ] Points the reader to the stored scan JSON and spectrum/fit PNG.

## Known traps

- The file set is contiguous only through run 19612. The worksheet lists a
  later run that is not in this folder; document text is experimental context,
  not evidence that a file was analysed.
- The WiMDA bunch factor of 500 is a display convenience. `integral-scan`
  performs the Poisson-weighted count integral directly; asking `reduce` for
  one huge time bin is not an equivalent agent workflow.
- The four temperature scans must not be concatenated into one 124-point field
  curve. A fitted centre from one block cannot be reported as a trend across
  all four.
