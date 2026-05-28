# Theory library expansion: scoring

## Impact (1-5)

**Score: 4**

- *Breadth of user benefit:* high. Each function unlocks a class of
  studies that currently cannot be done in Asymmetry. Keren and
  SpinGlass alone cover most magnetic-fluctuation work.
- *Pedagogical value:* high. The user-guide can include archetypes
  that match real μSR literature (Uemura plot for spin glasses,
  Keren analysis for paramagnets near Tc, Bessel for incommensurate
  magnets).
- *Alignment with Asymmetry strengths:* perfect — drops into the
  existing registries with no architectural work.
- *Note:* the impact per function varies. Keren / Abragam / SpinGlass
  are widely used; SuperconductorVortexLattice is narrower. Aggregate
  4 reflects the bundle.

## Ease (1-5)

**Score: 4**

- *Registry / API readiness:* perfect. One new entry per function in
  `src/asymmetry/core/fitting/models.py` (or `composite.py` /
  `parameter_models.py` as appropriate).
- *Model complexity:* low per function. ~20–60 lines of numpy each.
- *GUI surface required:* none. Fit Wizard picks up new entries
  automatically.
- *Test-data availability:* Mantid and musrfit ship regression
  curves we can use directly.
- *Risk:* low. Each function is independent and can be reverted
  without affecting others.

## Score = impact × ease = **16**

Tier: **Now**. Recommend an umbrella study pass plus PRs
function-by-function in order:

1. Keren
2. Abragam
3. Bessel
4. SpinGlass
5. Meier
6. MuoniumDecouplingCurve
7. SuperconductorVortexLattice

Items 1–4 are likely a single PR each. Items 5–7 are slightly more
involved due to literature-convention ambiguities and the parameter-
domain placement.
