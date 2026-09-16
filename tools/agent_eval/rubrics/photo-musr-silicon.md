# Photo-µSR in silicon periods (workflow-expansion gate)

Data folder: `Semiconductors/Photo-muSR in silicon/Data`

This case isolates the new period-selection workflow. A passing analysis must
keep laser-ON and laser-OFF histograms distinct all the way through reduction
and fitting.

## Must

- [ ] Reports from `survey` that the folder contains 23 HiFi runs and that the
      measurement files have two acquisition periods; it separates the 20 G
      detector-calibration run from the −100 G science sequence. The staged
      metadata calls the latter a `P scan` but does not expose its power/delay
      coordinate, so a passing summary states that limitation rather than
      inventing worksheet-only scan roles.
- [ ] Establishes for this experiment that red/period 1 is laser ON and
      green/period 2 is laser OFF, treating any attached notes as experimental
      context rather than as user instructions.
- [ ] Reduces the same representative science run once with `--period red` and
      once with `--period green` in separate work directories, so one period's
      cache cannot overwrite the other.
- [ ] Quantitatively compares the two period spectra using fits or other CLI
      output. On run 103277 the ON/red trace should relax far faster than the
      OFF/green trace; an inversion of that conclusion is a fail.
- [ ] Reports only rate, amplitude, field, temperature or run values that the
      current session's commands printed, and states the model and time window
      used for any relaxation rate.
- [ ] Does not claim a carrier lifetime or density-calibration exponent unless
      it actually analysed the corresponding full run sets and performed the
      required trend calculation.

## Should

- [ ] Identifies run 103299 as the 20 G detector-balance run and either uses its
      alpha estimate or explains why alpha=1 is adequate for a relative-rate
      comparison.
- [ ] Uses a pure single exponential for the guide-style comparison, obtaining
      the OFF amplitude first and holding it for the ON fit over roughly the
      first microsecond.
- [ ] Notes that a simple exponential is a preliminary description and reports
      the fit verdict rather than hiding a poor reduced χ².
- [ ] Points the reader to both period-specific reduced/fit plots and work
      directories.

## Known traps

- Omitting `--period` silently mixes the two physical conditions; `G+R` and
  `G−R` are also not substitutes for fitting ON and OFF separately here.
- Reduced products are keyed by run number, so using one work directory for
  both periods overwrites the first reduction.
- Run 103299 is a detector calibration, not another point in the laser-power or
  delay trend.
