# Expert consultation — β estimator protocol

Status: consultation complete, 2026-07-19. Outcome adopted in
`estimator-plan.md` and implemented on `feat/beta-estimator`.

## Context

The scalar β correction shipped in PR #272 (v0.15.0) with the estimator
deliberately deferred pending expert consult: three candidate measurement
protocols had been drafted (`implementation-options.md`), and Ben sought
outside opinion before committing to one as the shipped default.

## Outcome

- **Protocol A** (a simultaneous forward+backward count-domain fit, sharing
  the physics between the two histograms, with β free on the backward
  amplitude) is **endorsed as the measurement protocol** and is the default.
- Some instrument scientists prefer **Protocol B** (paired independent
  single-histogram fits, β̂ = A₀,b/A₀,f by ratio); it is offered as an
  alternative and cross-check, not the default.
- β is measured on a **weak-TF calibration run** — the same run type already
  used to calibrate α.
- β is broadly an **instrument-dependent** property of a detector pair.
  Typical values: ~0.8–0.9 on FLAME; just under 1 on GPS.
- **Protocol C** (an asymmetry-space fit) was considered and **dropped**.

These points are recorded here as the consultation's outcome; the full
protocol specifications, the core API, and the persistence/GUI design built
on them are in `estimator-plan.md`.
