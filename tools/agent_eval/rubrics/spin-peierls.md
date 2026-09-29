# A spin-Peierls transition (Tier B)

Data folder: `Magnetism/A spin-Peierls transition/Data`

## Experiment

Potassium TCNQF4 is a quasi-1D S=1/2 charge-transfer salt that
dimerises at approximately 150 K, undergoing a spin-Peierls transition
into a singlet-triplet excitation regime. The worksheet asks the
student to fit the ZF spectra across this transition and notice that
the spectral line shape itself changes as the sample is cooled through
it.

## Run structure

26 EMU `.nxs` runs, one contiguous block 29919-29944, no dedicated
low-field calibration run:

- TF 100 G runs, interleaved with the ZF scan: 29919 (300 K, the
  worksheet's only field run), 29922, 29925-29930, 29935 and 29942; the
  survey measures each precessing at the Larmor frequency and offers them
  as alpha candidates.
- ZF runs: 29920-29921, 29923-29924, 29931-29934, 29936-29941, 29943-29944;
  the worksheet documents 29931-29944 (130-185 K) and nothing below.
- The worksheet's table additionally lists runs 29945-29951 (110-127 K)
  that are not present in this Data folder.

## Must

- [ ] Separates the TF 100 G runs from the ZF scan rather than fitting
      them as one series.
- [ ] States what was used for alpha (a TF 100 G run, named), rather
      than silently assuming 1.0.
- [ ] Reports on the full run range actually present (29919-29944),
      not only the subset the worksheet happens to tabulate.
- [ ] Does not report findings for runs 29945-29951 as if they were
      analyzed — they are not present in this data folder.
- [ ] Reports a change in ZF line shape between the highest- and
      lowest-temperature runs actually available.
- [ ] Contains no relaxation rate, frequency, or transition-temperature
      number not produced by a tool call in this session.

## Should

- [ ] Accounts for every run in the block — the ZF runs below 130 K and
      the interleaved TF runs — rather than silently dropping any.
- [ ] Discusses candidate fit functions for above vs below the
      transition, without inventing fitted values.
- [ ] Connects the observed line-shape change to the singlet-triplet /
      dimerisation physics described in the worksheet background.

## Known traps

- The worksheet's table references seven runs (29945-29951) that do
  not exist in the supplied Data folder — a summary built from the
  worksheet text alone, rather than the actual file survey, will claim
  results for data it never had.
- Eleven present runs (29920-29930) plus two more inside the
  documented block (29935, 29942) have no worksheet entry; a summary
  that only discusses the worksheet-named runs silently discards a
  large fraction of the real dataset.
- ISIS EMU files can stamp a nonzero field on a ZF-intended run (or
  vice versa); group by the survey's own field column, not by
  assuming the worksheet's ZF/TF split matches every file exactly.
