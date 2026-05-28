# Theory library expansion

**Status:** candidate. Umbrella for multiple individual function ports
that share registration scaffolding.

## What

Expand Asymmetry's MODELS / COMPONENTS registries from ~17 entries to
~30 by porting widely-used theory functions that musrfit and Mantid
ship but Asymmetry lacks. First wave (in suggested implementation
order):

1. **`Keren`** — exchange-coupled paramagnet relaxation. Mantid only.
2. **`Meier`** — hyperfine-coupled muonium / radical model. Mantid only.
3. **`Abragam`** — interpolation between Gaussian and exponential
   envelopes; the canonical motional-narrowing model. musrfit only.
4. **`Bessel`** — J₀ oscillation (incommensurate magnetism, spin
   density waves). musrfit only.
5. **`MuoniumDecouplingCurve`** — field-dependence of the hyperfine
   transition probability. Mantid only.
6. **`SpinGlass`** — Uemura-form spin-glass relaxation. musrfit only.
7. **`SuperconductorVortexLattice`** — time-domain analogue of the
   Brandt P(B) (currently only present as a frequency-domain
   parametric model in Asymmetry). musrfit only.

## Why

- The user-guide currently has to defer or hand-wave several
  archetypes — the comparison matrix shows Asymmetry as the
  smallest theory library of the four reference programs.
- Each new function is ~20–60 lines of numpy and a registry entry,
  with minimal risk to the rest of the codebase.
- The Fit Wizard's portfolio breadth improves linearly with the
  library size.

## Why this is roadmap-tractable

- Pure addition. Existing registries
  (`src/asymmetry/core/fitting/models.py` and `composite.py`) accept
  one new entry per function with no architecture work.
- Each function is independently testable against the corresponding
  musrfit or Mantid output.
- The Fit Wizard's AICc ranking already handles arbitrary candidate
  counts.

## Scope note

This candidate is an umbrella: the implementation pass should break
into one PR per function rather than landing all seven together. The
study pass produces one combined `comparison.md` documenting all
seven against the reference implementations.

## Out of scope for this candidate

- Mantid-specific muonium variants (`HighTFMuonium`, `LowTFMuonium`,
  `TFMuonium`, `ZFMuonium`) — those need a separate composition
  framework for the high-field hyperfine structure; tracked as a
  Later-tier item.
- `MuonFInteraction` — Mantid's specific TF-with-F-coupling
  function; Asymmetry already has the F-μ-F family (`MuF`,
  `FmuF_Linear`, `FmuF_General`) covering similar physics.
