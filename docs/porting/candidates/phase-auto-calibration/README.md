# Automatic phase calibration

**Status:** candidate (partially absorbed).

**Note (2026-06-10):** the WiMDA/MaxEnt slice of this candidate — exchanging
fitted phases between a grouped time-domain fit and MaxEnt ("Use fitted phases"
/ "Send phases to fit", matched by group id, with provenance) — has been
implemented in the `maxent-completion` project (Phase 3,
`docs/porting/maxent-completion/`). What remains here is the general
auto-calibration of phases from a reference TF run
(Mantid `CalMuonDetectorPhases`-style), which is unaffected.

## What

Estimate per-detector or per-group phase angles automatically by
fitting a sinusoid to the early-time TF signal. Output is a phase
table that can be inspected and edited in the GUI, then consumed by
the asymmetry-calculation and FFT pipelines.

## Why

- Phase is currently treated as a manual input or as a fit parameter
  in Asymmetry. Either path is friction: manual entry is slow and
  error-prone; fitting phase per dataset couples it to other model
  parameters and inflates uncertainties.
- Auto-calibration is the standard practice at every facility — the
  data-reduction pipeline computes initial phase estimates from a
  reference TF run, then reuses them across the experiment.
- The MultiGroupFitWindow's per-group "relative phase" parameter
  would benefit from sensible initial values rather than starting
  at zero.

## Prior art

- **Mantid `CalMuonDetectorPhases`** — fits a damped-cosine
  `A·cos(2πf·t + φ_d)·exp(-λ·t)` to each detector's early-time data
  with shared `f` and `λ` and per-detector `A` and `φ`. Outputs a
  TableWorkspace mapping detector index → (A, φ). Flagship feature
  in the Mantid Muon Analysis "Phase Calculation" tab.
- **musrfit, WiMDA:** ❌. Phases are user-supplied in the `.msr`
  RUN block or read from a calibration phase table file.

## Why this is roadmap-tractable

- The fit machinery for the per-detector cosine already exists —
  the `Oscillatory` component plus a global-fit pattern with
  shared (f, λ) and local (A, φ) is exactly the YBCO Knight scenario.
- API: `calibrate_phases(dataset, *, fit_window_us=(0, 4)) → dict[int, float]`.
- GUI: a "Calibrate phases" button in the grouping dialog or the
  multi-group fit window.

## Out of scope

- Cross-run phase calibration (using a reference TF run to seed
  multiple sample runs). Defer to follow-up.
- Phase calibration in the frequency domain (Mantid's MuonMaxent
  refines phases iteratively — that path lands with the
  maxent-spectrum candidate).
