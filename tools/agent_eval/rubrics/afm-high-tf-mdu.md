# AFM transition in high TF (Tier B — partial analysis)

Data folder: `Magnetism/AFM transition in high TF/data`

No worksheet exists; the accompanying paper is Huddart et al., Phys.
Rev. Research 5, 013015 (2023), a kappa-(ET)2X organic antiferromagnet
study. It tracks the antiferromagnetic transition in high transverse
field with MaxEnt field spectra, fits the two central lines and the
field shift against temperature, fits the area of the spectral wings
with an order-parameter law to get T_N at each field, and interprets the
field-induced canting with DFT muon-site and dipolar-field calculations.

This case was a decline until the 2026-09-23 corpus audit found that the
CLI reads each run's field and temperature and resolves the high-field
spectrum. The correct behaviour is now a **bounded** analysis: do the
spectral and line-shape steps the CLI supports, and say plainly which
of the paper's steps it cannot do.

## Run structure

18 PSI HAL-9500 (HIFI) `.mdu` runs at 24 ps binning on a kappa-(ET)
crystal, two complete temperature scans:

- 6 T, runs 686-693: 3.2 K to 50.7 K.
- 8 T, runs 730-739: 3.1 K to 100 K.

The survey reads both fields and both temperature axes, measures
precession at the Larmor frequency on every run, and lists the two
temperature scans. The run-number gap (694-729) separates the two
fields; it is not a missing part of either scan.

## Must

- [ ] Identifies the two temperature scans by field (6 T and 8 T) with
      their run ranges, from the survey rather than from file names
      alone.
- [ ] Resolves the precession near the Larmor frequency into more than
      one line (from a Fourier transform or a multi-component fit) at
      at least one field, and says which it used.
- [ ] Reports a change in the spectrum or line shape (broadening,
      splitting or extra structure) on cooling through the
      tens-of-kelvin region at at least one field, from this session's
      own fits or spectra.
- [ ] States which parts of the paper's analysis the tool cannot
      reproduce: MaxEnt field spectra and the DFT dipolar-field
      interpretation of canting; does not present a canting angle or
      ordered moment as a finding.
- [ ] Contains no transition temperature, field shift or line
      parameter that was not produced by a tool call in this session.

## Should

- [ ] Reports the recorded run temperature where it differs from the
      title (run 732 is titled 12 K but recorded at 10.5 K).
- [ ] If it quotes a transition temperature, it comes from a stated
      `trend --model` fit to a line parameter with the fit range given,
      and is framed as that fit's result rather than the paper's T_N,
      which rests on the wing area of MaxEnt spectra.
- [ ] Notes that the two central lines are separated by roughly one
      Fourier resolution element, so a zoomed spectrum's empty peak
      table is not evidence that the second line is absent.

## Known traps

- The two central lines sit about 0.1 MHz apart near 814 MHz (6 T) and
  1085 MHz (8 T). A transform restricted with `--fmin`/`--fmax` can
  report an empty peak table while the plot shows both lines; read the
  plot.
- Below the transition, spectral wings appear at 6 T and a two-line
  fit fails on several runs. That is physics (a broadened field
  distribution), not a reason to drop the runs silently.
- Screening one of these runs with the wizard takes about two minutes
  rather than seconds (389k points); the recommended candidate is not
  always the lowest-AICc one.
- The paper's T_N values and canting angles are background, not
  results of this session; quoting them as measured fails the number
  rule.
