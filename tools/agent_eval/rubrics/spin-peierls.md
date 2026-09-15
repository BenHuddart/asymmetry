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

- Reference run, 29919: TF 100 G, 300 K (worksheet's only field run).
- Worksheet-documented ZF runs within the block, 29931-29944 (skipping
  29935 and 29942): 130-185 K.
- Runs 29920-29930, plus 29935 and 29942, are present on disk but have
  no entry in the worksheet's temperature table at all.
- The worksheet's table additionally lists runs 29945-29951 (110-127 K)
  that are not present in this Data folder.

## Must

- [ ] Identifies 29919 as the TF 100 G reference/calibration run,
      separate from the ZF runs.
- [ ] States what was used for alpha (29919, or an explicit
      alternative), rather than silently assuming 1.0.
- [ ] Reports on the full run range actually present (29919-29944),
      not only the subset the worksheet happens to tabulate.
- [ ] Does not report findings for runs 29945-29951 as if they were
      analyzed — they are not present in this data folder.
- [ ] Reports a change in ZF line shape between the highest- and
      lowest-temperature runs actually available.
- [ ] Contains no relaxation rate, frequency, or transition-temperature
      number not produced by a tool call in this session.

## Should

- [ ] Flags runs 29920-29930 (and 29935, 29942) as present but
      undocumented by the worksheet, and reports what the survey found
      for them rather than silently dropping them.
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
