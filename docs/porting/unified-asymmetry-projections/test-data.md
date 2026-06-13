# Test data

This is an internal unification, exercised with existing instrument layouts and
synthetic data rather than a new reference corpus.

## Projection-bearing groupings

- **EMU vector polarization** — `core/instrument.py` EMU `Vector Polarization`
  preset (6 groups → 3 projections `P_x/P_y/P_z`). Primary exercise for the chip
  bar, frame-tint, stacked subplots, and per-projection fits.
- **MuSR TF dual grouping** — `Transverse (Top–Bottom)` and
  `Transverse (Forward–Backward)` presets, to be re-expressed as a single preset
  declaring **two** projections. Proves the unification generalizes beyond three
  Cartesian axes and beyond EMU.
- **HiFi TF dual grouping** — `Transverse (Left–Right)` and `(Top–Bottom)`,
  secondary check.
- **Longitudinal (any instrument)** — the degenerate 1-projection case: chip bar
  and subplot-selection UI must *not* appear; behaviour unchanged.

## Synthetic signals

Reuse the simulate-mode builders (`core/transform/simulate*`) for a controlled
three-axis signal: dominant slow exponential on `P_z`, weak transverse
oscillation on `P_x`, near-zero noise on `P_y` (the textbook EMU signature used by
`docs/user_guide/vector_polarization.rst`). Lets per-projection alpha and
per-projection fits be checked against known inputs.

## Persistence fixtures

- A pre-v9 `.asymp` project containing an EMU vector dataset with a single fit:
  asserts migration lands that fit on the default projection and that
  `alpha_x/y/z` / legacy `alpha_px/py/pz` resolve.
- A v9 project with **distinct** fits on `P_x`, `P_y`, `P_z` of the same run:
  asserts each round-trips independently and the active fit target is restored.

## RG-mode colour check

Multi-run EMU overlay in RG mode with ≥2 projections shown: assert trace colours
track runs (consistent across subplots) and frame tints track projections — i.e.
the two colour systems stay orthogonal.
