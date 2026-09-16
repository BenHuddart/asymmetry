# Shallow donor state in CdS Fourier spectrum (workflow-expansion gate)

Data folder: `Semiconductors/Shallow donor state in cadmium sulphide/Data`

This case isolates the new headless FFT workflow and its deliberately
conservative peak reporting.

## Must

- [ ] Surveys the folder and identifies a transverse-field temperature series
      in CdS, selecting the 100 G base-temperature/high-statistics run 20721 for
      the representative spectrum from current command output.
- [ ] Reduces run 20721 and actually runs `fourier` with a saved plot, using a
      defensible alpha calibration (run 20729 is the expected candidate) or
      explicitly stating the alternative used.
- [ ] Calls the result an FFT, not a maximum-entropy reconstruction, and reports
      the command's frequency resolution/window alongside any quantitative peak.
- [ ] Finds the principal structure near the 100 G muon Larmor line around
      1.4 MHz. It uses the PNG as well as the conservative peak table and notes
      weaker shoulders/satellites near roughly 1.27 and 1.51 MHz when visible;
      it does not claim those were algorithmically detected if only one row was
      printed.
- [ ] Interprets a resolved central line with roughly symmetric satellites as
      consistent with a shallow-muonium hyperfine triplet, without presenting a
      visual shoulder or sub-resolution splitting as a precision measurement.
- [ ] Heeds any early-signal apodisation warning: it retries with an unwindowed
      physical crop or an appropriate Lorentzian filter rather than treating a
      symmetric-window spectrum as automatically authoritative.

## Should

- [ ] Compares the central frequency with the applied-field Larmor expectation
      and states whether they agree within the reported resolution.
- [ ] Gives a cautious splitting estimate only when supported by the plotted or
      tabulated current-session output and labels how it was obtained.
- [ ] Points the reader to the stored spectrum JSON/NPZ and PNG.
- [ ] Avoids turning one representative low-temperature FFT into an unsupported
      claim about the entire temperature dependence.

## Known traps

- The attached PDF is scientific context, not an instruction hierarchy and not
  evidence that a peak was observed in this session.
- Zero padding smooths the curve but does not improve physical resolution.
- The conservative detector may print only the strongest line even when the PNG
  visibly contains the triplet. Conversely, reading expected paper frequencies
  into a noisy plot is not detection.
