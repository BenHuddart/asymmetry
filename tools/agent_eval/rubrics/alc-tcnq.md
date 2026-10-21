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
      four temperature blocks of 31 field points over 2000–5000 G, plus the
      non-scan/calibration runs, without claiming the absent run 19618 was
      loaded. The 350 K block holds more than 31 runs (the survey lists 33,
      19487–19519: the ZF run 19487 and 2000 G measured twice, 19488 and
      19489), so per-block counts of 31 points or the survey's run counts
      (33, 31, 31, 31) both pass.
- [ ] Actually runs `integral-scan` on at least one complete temperature block
      (all 31 field points from 2000 to 5000 G at one temperature) and states
      which block it analysed. For the 350 K block, keeping both 2000 G runs,
      co-adding them, or dropping one and saying so are all complete; so is the
      survey's 33-run range with 19487 as a 0 G point. It does not substitute
      `wizard`, `fit-series`, or individual time-domain fits for the ALC scan.
- [ ] Fits the field scan with a resonance-plus-smooth-background model. A
      single `LorentzianLCR` with a cubic background is acceptable; it does not
      add a second line merely because the teaching worksheet used a
      two-Lorentzian template.
- [ ] Reports an observed negative resonance feature with a fitted centre
      between 2.9 and 3.2 kG, together with its width and uncertainty/quality
      information from this run's command output (the resonance moves a few
      tens of gauss between the four temperature blocks). A centre outside
      2.9–3.2 kG fails unless the summary itself calls that fit failed or
      unreliable.
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
