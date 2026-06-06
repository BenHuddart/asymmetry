# Maximum Entropy Implementation Options

## Option 1: Port The Generic Skilling–Bryan Engine (Mantid `MaxEnt-v1` Style) On Asymmetry Data

Implement a per-spectrum MaxEnt over already-processed asymmetry, like
Mantid's generic algorithm.

Pros:

- simplest data contract — matches the existing stub signature in
  `src/asymmetry/core/fourier/maxent.py` (processed `MuonDataset` in,
  spectrum out)
- no raw-counts, frames, or deadtime plumbing needed
- PosNeg entropy available for signed asymmetry data
- best-tested reference implementation to compare against

Cons:

- loses the defining capabilities of μSR MaxEnt: joint multi-group
  reconstruction and outer-loop phase/amplitude/background/deadtime fitting
- Mantid's own muon GUI abandoned this engine for `MuonMaxent` in v3.12 —
  evidence it does not serve muon users
- would not reproduce WiMDA results, breaking the project's WiMDA-parity
  precedent

Assessment: reject as the primary engine. Revisit only if a future use case
needs MaxEnt on data whose raw counts are unavailable.

## Option 2: Port The MULTIMAX Algorithm With WiMDA As The Behavioral Contract

Implement the joint multi-group algorithm once in `asymmetry.core`, following
WiMDA's `Wimdamax.pas` behavior, verified against Mantid's `MaxentTools`
numpy oracle, with the WiMDA control surface (incremental cycles, window,
fit-phases/BG/deadtime toggles) layered on top.

Pros:

- implements the method both reference programs actually ship and musrfit's
  roadmap aspires to
- one engine, one user-facing "MaxEnt" — no confusing variant choice
- WiMDA-parity matches the project precedent and the richest feature set
  (MODCONST constant background, auto window, interactive convergence,
  moments follow-on)
- Mantid's plain-numpy port gives a line-level executable oracle the Fourier
  study never had
- WiMDA-only extras (apodisation-as-σ, exclusion window, phase relaxation)
  slot in as options without architectural change

Cons:

- requires raw grouped counts, frames, t0, and deadtime metadata through the
  core seam — the existing stub contract must change
- stateful incremental-cycle API is more design work than a one-shot function
- license discipline needed: Mantid code (GPL-3) may be executed as an
  oracle but not copied into MIT-licensed Asymmetry

Assessment: **choose.**

## Option 3: Port MULTIMAX With Mantid `MuonMaxent` As The Behavioral Contract

Same algorithm, but adopt Mantid's parameter set (one-shot
Outer×Inner batch, `MaxField` clamp, phase tables, no constant-BG fit).

Pros:

- contract is testable today against Mantid's system/doc tests
- phase-table I/O matches Asymmetry's existing per-group phase tables

Cons:

- loses WiMDA features users of this project expect (constant background fit
  for continuous sources, auto window, interactive convergence control)
- Mantid's batch model is a strict subset of WiMDA's incremental model
- inherits Mantid quirks (Npts validator default bug, truncated constants)

Assessment: reject as the contract, but adopt its good parts inside
Option 2: phase-table seeding, phase-convergence diagnostics, reconstructed
spectra as first-class outputs, and the data-derived spectrum-points default.

## Option 4: Also Implement Burg All-Poles MEM (WiMDA `MaxEnt.pas`)

Pros:

- cheap (~200 lines); WiMDA users may recognize it

Cons:

- a different statistical method that merely shares the name "maximum
  entropy" — the canonical confusing-second-choice
- WiMDA itself keeps it in the Fourier window, not the MaxEnt feature
- no other reference program has it; no test coverage anywhere

Assessment: reject from this feature. If ever wanted, file it under the
Fourier feature family as an "all-poles (Burg)" transform mode.

## Selected Direction

Option 2, staged:

1. **Core engine slice** — `src/asymmetry/core/maxent/` (new package; the
   existing `src/asymmetry/core/fourier/maxent.py` stub is superseded and its
   contract redefined):
   - input dataclass: per-group raw histograms, σ model
     (`sqrt(N+2)`-style with normalization internal), t0/good-bin ranges,
     frames, optional deadtimes/phases/alpha
   - forward/adjoint maps (OPUS/TROPUS equivalents) over a windowed default
     map; Skilling–Bryan 3-direction inner loop with Cholesky subproblem,
     χ²-target staging, trust region, negativity clamp
   - outer cycle: phase+amplitude refit (free or amplitude-only), exponential
     background renormalization, constant-background fit; explicit
     excluded-bin mask instead of σ-sentinel magic
   - resumable `MaxEntState` so the GUI can do +1/+5/+25/Converge; explicit
     restart-required semantics per parameter
   - diagnostics record per cycle: χ², entropy, TEST, sconv, phases,
     amplitudes, backgrounds
2. **Verification slice** — golden data from the Mantid oracle and synthetic
   cases per `test-data.md` / `verification-plan.md`.
3. **GUI slice** — MaxEnt panel filling the reserved `FrequencyMaxEnt`
   representation; reuse the Fourier panel's field/frequency axis machinery
   and per-group phase tables; time-domain reconstruction overlay on the
   time-domain tab.
4. **Options slice** — deadtime fitting, pulse-shape response, apodisation,
   exclusion window, phase relaxation, smooth errors (defaults per
   `comparison.md` §"Behaviors to offer as user options").
5. **Follow-ons (separate slices, not blocking)** — moments analysis window,
   ZF/LF alpha-tied mode, spectral deconvolution (needs a regularized
   adjoint), muonium-correlation display.

Decisions locked by `comparison.md`: single engine (no WiMDA-vs-Mantid or
generic-engine choice), WiMDA window mechanism, CODATA constants, WiMDA
incremental convergence model, Burg MEM excluded.

Open questions for the implementing agent are listed at the end of
`comparison.md` §"Aspects The Implementing Agent Should Explore"; resolve
items 1–3 (provenance papers, oracle importability, raw-counts availability)
before writing the core engine.
