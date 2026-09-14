# Magnetic ordering in EuO (Tier B)

Data folder: `Magnetism/Magnetic ordering in EuO/data`

No worksheet exists for this dataset — the source is a published paper
(Blundell et al., PRB 81, 092407 (2010)) whose abstract is the rubric
basis: muSR on the localized ferromagnet EuO, arguing implanted muons
sense the internal field mainly through hyperfine and Lorentz
contributions, with the temperature dependence of the internal field
and the relaxation rate measured and compared to theoretical
predictions.

## Run structure

51 PSI GPS `.bin` runs (`deltat_pta_gps_2923.bin` ... `_2973.bin`,
contiguous, no gaps), no ISIS-style field stamp in the filename:

- ZF block, 2923-2960 (38 runs): temperature swept from 5 K up through
  the 60-95 K region with several revisits clustered near 68-71 K
  (the expected transition region), then a final low-temperature run
  (2960) at 1.5 K taken out of sequence at the end of the block.
- TF 60 G block, 2961-2973 (13 runs): temperature swept down from
  200 K, again with several closely-spaced revisits near 69-71 K.

## Must

- [ ] Identifies this as a PSI/GPS `.bin` dataset, distinct from the
      ISIS `.nxs` datasets, and confirms the PSI loader path was used.
- [ ] Separates the ZF block from the TF 60 G block as two distinct
      scans.
- [ ] Reports an internal field (from the TF block, or from ZF
      oscillations if present) that changes with temperature, growing
      as the sample is cooled below the ordering region.
- [ ] Reports a relaxation rate that changes with temperature,
      consistent with critical slowing down near the ordering region.
- [ ] Contains no internal-field value, relaxation rate, or Curie
      temperature that was not produced by a tool call in this
      session.

## Should

- [ ] Notes that both blocks revisit a narrow temperature window
      several times rather than assuming a single monotonic sweep.
- [ ] Treats run 2960 (1.5 K, out of sequence at the end of the ZF
      block) as a low-temperature reference point rather than ignoring
      it or assuming it belongs elsewhere in the sequence.
- [ ] Frames the comparison to theory (e.g. mean-field/Heisenberg
      critical behaviour) as the paper's stated goal, without
      inventing a critical exponent value.

## Known traps

- The paper's introduction quotes EuO's literature Curie temperature
  (69 K) as background; this is a citation, not a result of fitting
  this dataset, and must not be echoed as an experimental finding
  unless a tool call actually produced a matching number.
- PSI `.bin` files carry field and temperature in the file header, not
  in a `.nxs`-style filename or an ISIS-convention stamp — grouping by
  filename alone (as for the ISIS datasets) will not separate the ZF
  and TF blocks; the survey's own field metadata must be used.
- Run order is not monotonic in temperature in either block; treating
  "first run to last run" as "high T to low T" (or vice versa)
  throughout will misdescribe the coverage.
