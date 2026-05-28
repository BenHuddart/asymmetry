# MaxEnt frequency reconstruction: scoring

## Impact (1-5)

**Score: 4**

- *Breadth of user benefit:* moderate-to-high. Users with multi-site
  magnets, short data runs, or closely-spaced precession peaks
  benefit substantially; casual users not affected.
- *Pedagogical value:* high. The textbook discusses MaxEnt as a
  resolution-improvement technique in chapter 15.5; Asymmetry can
  currently only show apodised FFT.
- *Alignment with Asymmetry strengths:* the stub already exists;
  promoting it to working implementation is consistent with the
  existing API.
- *Why not 5:* the technique is more specialised than e.g.
  dynamic KT — not every workflow uses MaxEnt, but those that do
  rely on it heavily.

## Ease (1-5)

**Score: 3**

- *Registry / API readiness:* the stub at
  `core/fourier/maxent.py` defines the signature; the Fourier panel
  already wires `_on_compute_fourier` to dispatch to this module.
- *Model complexity:* moderate-to-high. Burg is ~200 lines of
  numpy; iterative entropy is ~600 lines plus tuning.
- *GUI surface required:* small (one Fourier mode toggle).
- *Test-data availability:* synthetic delta-comb + noise tests are
  easy to generate; cross-validation against Mantid output is
  possible.
- *Risk:* moderate. Iterative convergence on real data can be
  finicky; need to expose convergence diagnostics.

## Score = impact × ease = **12**

Tier: **Next** (mid-roadmap). Burg-first implementation pass can land
in a couple of weeks; iterative entropy is a follow-on.
