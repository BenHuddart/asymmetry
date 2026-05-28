# Period arithmetic

**Status:** candidate.

## What

First-class support for sum / difference operations on pulsed-beam
period data. At ISIS the muon beam is pulsed and each "run" can
contain multiple acquisition periods (typically a red set vs. green
set or pulse vs. inter-pulse background). Period arithmetic combines
these intelligently before grouping and asymmetry calculation.

## Why

- ISIS-side users currently cannot fully exploit Asymmetry on pulsed
  data without manual histogram surgery.
- Period handling is the most common single source of confusion when
  Mantid users try Asymmetry and find their multi-period data
  collapsed into a single channel.

## Prior art

- **Mantid:** `SummedPeriodSet` and `SubtractedPeriodSet` syntax in
  `MuonProcess`. Each NeXus file is loaded as a workspace group with
  one workspace per period; the syntax `"1+2-3"` produces the
  combined channel.
- **WiMDA:** `period-mapping` slug; eight-period radio-button UI
  assumption; logic in `PeriodMappingUnit.pas` and `muondata.pas`.
  Less flexible than Mantid's syntax.
- **musrfit, Asymmetry:** ❌. (Asymmetry's NeXus loader reads only
  the default period.)
