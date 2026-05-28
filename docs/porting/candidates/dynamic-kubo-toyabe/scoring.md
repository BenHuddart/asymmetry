# Dynamic Kubo–Toyabe: scoring

## Impact (1-5)

**Score: 5**

- *Breadth of user benefit:* every magnetic-fluctuation μSR study uses
  dynamic KT or its motional-narrowing limits. Universal.
- *Pedagogical value:* the static-→-dynamic crossover is the canonical
  ν/Δ teaching example in chapter 5 of the textbook. Asymmetry's
  user guide currently has to skip it.
- *Alignment with Asymmetry strengths:* drops into the existing
  MODELS registry and Fit Wizard portfolio without architectural
  change.
- *Frequency of "missing feature" requests:* this is the single
  most commonly-flagged absence when Asymmetry is benchmarked against
  musrfit / Mantid.

## Ease (1-5)

**Score: 4**

- *Registry / API readiness:* the static analogues already exist; the
  new function plugs into the same `_register(...)` block in
  `src/asymmetry/core/fitting/models.py`.
- *Model complexity:* moderate — the time-step convolution is ~30
  lines of numpy with cached static-KT evaluations.
- *GUI surface required:* none. The Fit Wizard picks up the new
  registry entry automatically.
- *Test-data availability:* Mantid's regression curves + textbook
  asymptotic limits provide validation oracle.
- *Risk:* low. Worst case is performance — N² convolution at
  N=10⁴ is still sub-second in vectorised numpy.

## Score = impact × ease = **20**

Tier: **Now** (top quartile). Recommend promoting to a full study
folder in the first roadmap iteration.
