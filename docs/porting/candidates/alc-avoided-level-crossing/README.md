# ALC: Avoided Level Crossing interface

**Status:** candidate.

## What

A dedicated workflow for analysing muon Avoided Level Crossing
resonances: scan a sample log (typically applied field) → fit a
baseline polynomial → fit Lorentzian / Gaussian peaks on the
baseline-corrected residual → extract resonance positions, widths,
heights.

ALC is the canonical technique for identifying muonium hyperfine
states in semiconductors and for probing nuclear quadrupolar effects.

## Why

- The only Mantid-exclusive interface in the comparison matrix —
  Asymmetry porting it would close a real capability gap.
- The chemistry / semiconductor μSR community (Lord, Cottrell,
  Cox papers) relies on ALC; the textbook chapter 19.4 covers it.

## Prior art

- **Mantid ALC interface** — strict MVP architecture:
  `qt/scientific_interfaces/Muon/ALCInterface.cpp`,
  `ALCDataLoading{Presenter,Model,View}.{cpp,h}`,
  `ALCBaselineModelling{Presenter,Model,View}.{cpp,h}`,
  `ALCPeakFitting{Presenter,Model,View}.{cpp,h}`.

- **WiMDA:** ❌. Has `ALCmode` flag in `muondata.pas` for ingest
  routing but no analysis surface.
- **musrfit, Asymmetry:** ❌.

## Why this candidate is heavy

- ALC is not a single algorithm — it's a multi-stage workflow with
  its own GUI tab, multiple plot panes, baseline editor, peak
  picker, results table.
- The closest Asymmetry analogue is the parameter-trending panel,
  but ALC's baseline-then-peak workflow is structurally distinct.
- Estimated 1500-3000 lines of code (GUI + workflow) on top of
  existing infrastructure.

## Out of scope

- Real-time ALC monitoring (Mantid doesn't do this either).
- ALC simulation from a calculated hyperfine Hamiltonian (separate
  candidate — `mu_finder` / `muLFC` integration in `Source/`).
