# A high-Tc cuprate (Tier A)

Data folder: `Superconductivity/A high-Tc cuprate/Data`

## Experiment

BiSCCO is measured in transverse field to map the vortex-lattice field
distribution below Tc via the Gaussian TF relaxation rate sigma, which
is related to the London penetration depth. The worksheet asks for
sigma(T) at each field, comparison of time- and frequency-domain
analysis, and consideration of the non-superconducting signal fraction.

## Run structure

Two instruments in one folder, 48 runs total, no separate calibration
run (each TF run is itself a precessing calibration-style measurement):

- EMU, TF 150 G, 76976-76993 (18 runs): 5-80 K in 5 K steps; the
  worksheet's own table only tabulates 76976-76992 (17 runs) and
  omits 76993.
- MUSR, TF 400 G and 200 G, 1274-1303 (30 runs): the worksheet's table
  starts at 1276 ("Above Tc Run" at 125 K) and omits 1274-1275; the
  tabulated runs cover 400 G from 10-125 K then 200 G from 10-120 K.

## Must

- [ ] Separates the two instruments' runs (EMU vs MUSR) into distinct
      scans rather than treating the folder as one series.
- [ ] Names both applied fields present on MUSR (400 G and 200 G) as
      separate field groups, not a single 400 G scan.
- [ ] Reports a Gaussian (or Gaussian-like) TF relaxation rate that
      rises as temperature drops below the transition, for at least
      one of the field/instrument combinations.
- [ ] Discusses how the non-superconducting fraction of the signal was
      handled (e.g. a second, slowly-relaxing or non-relaxing
      component), rather than fitting a single Gaussian and ignoring
      it.
- [ ] Contains no sigma, Tc, or penetration-depth number not produced
      by a tool call in this session.

## Should

- [ ] Notes that the data folders contain more runs than the
      worksheet's own tables list (76993 on EMU; 1274-1275 on MUSR)
      and reflects the survey's actual range rather than only the
      worksheet's table.
- [ ] Compares sigma(T) trends across the different fields/instruments
      rather than reporting only one.
- [ ] Mentions the "above Tc" run (1276, 125 K) as a normal-state
      reference distinct from the temperature scan proper.

## Known traps

- Two instruments sharing one folder is easy to survey as a single
  run list; grouping by field and instrument prefix (EMU vs MUSR) is
  required before any per-scan fit makes sense.
- There is no dedicated low-field calibration run for either
  instrument; alpha handling should be stated explicitly rather than
  assumed silently.
- The worksheet's tabulated run lists are each missing a few runs that
  exist on disk (76993; 1274-1275) — a summary built only from the
  worksheet text rather than the survey will under-report coverage.
