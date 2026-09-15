# A molecular antiferromagnet (Tier B)

Data folder: `Magnetism/A molecular antiferromagnet/Data`

## Experiment

Ni(9S3)2[Ni(bdt)2]2 is a molecular magnet with two distinct Ni
environments forming both ferrimagnetic and antiferromagnetic chains.
Cooling produces clear oscillations in the ZF spectra as long-range
order sets in; the worksheet's aim is to track the oscillation
frequencies through the transition to estimate a Neel temperature, TN.

## Run structure

11 MUSR `.nxs` runs, 17094-17104, all documented in the worksheet:

- ZF temperature sweep, 17094-17103 (10 runs): 1.2-10.0 K, monotonic
  with run number.
- Calibration, 17104: TF 20 G, 10.0 K, explicitly "for alpha
  determination".

## Must

- [ ] Identifies 17104 as the TF 20 G calibration run and states alpha
      was determined from it.
- [ ] Treats 17094-17103 as the ZF temperature sweep used for the
      ordering analysis, separate from the calibration run.
- [ ] Reports oscillation(s) present in the low-temperature ZF spectra
      that become more relaxed/damped as temperature rises through
      the sweep.
- [ ] Comments on whether more than one oscillation frequency is
      present (the worksheet flags this as something to check for),
      rather than assuming a single frequency without looking.
- [ ] Contains no frequency, TN, or relaxation-rate number not
      produced by a tool call in this session.

## Should

- [ ] Notes the "Cryostat Temperature" vs "Sample Temperature" columns
      logged for these runs are identical in the worksheet table, and
      does not imply a discrepancy that isn't there.
- [ ] Discusses whether it fit from low to high temperature or vice
      versa, and why (the worksheet explicitly raises this as a
      design choice).
- [ ] Chooses a relaxing-oscillation function appropriate to magnetic
      order (e.g. damped cosine terms) rather than a generic
      unstructured relaxation function.

## Known traps

- This dataset is "ZF only" in the sense that matters physically, but
  one of its 11 runs (17104) is a TF calibration run — a summary that
  describes the whole folder as pure ZF data and skips calibration
  entirely misses that run's purpose.
- Event counts vary sharply across the sweep (100 M down to ~4 M
  events for the last ZF runs); very low statistics runs should not be
  treated as equally reliable as the high-statistics ones without
  comment.
- ISIS MUSR files can stamp a nonzero field on a ZF-intended run (or
  vice versa); the calibration run should be identified from the
  survey's field column, not from its position at the end of the run
  list.
