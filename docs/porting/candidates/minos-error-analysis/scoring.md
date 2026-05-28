# MINOS error analysis: scoring

## Impact (1-5)

**Score: 4**

- *Breadth of user benefit:* high. Every published μSR paper with a
  non-trivial fit landscape benefits — and the most common
  Asymmetry-fittable regime (nuclear-dipolar Δ near zero,
  damped-oscillation Λ near zero) is exactly where Hessian errors
  mislead.
- *Pedagogical value:* moderate. Asymmetric errors are a recognised
  hallmark of rigorous fitting; expert readers expect them.
- *Alignment with Asymmetry strengths:* perfect. iminuit already
  supports it; the change is API exposure, not new algorithms.
- *Why not 5:* most casual users never look beyond a single
  significant figure; the impact is concentrated in publication
  workflows.

## Ease (1-5)

**Score: 5**

- *Registry / API readiness:* iminuit's `Minuit.minos()` is one
  method call.
- *Model complexity:* none. Pure post-processing of an existing fit.
- *GUI surface required:* small (one checkbox, one display extension).
- *Test-data availability:* easy to validate — generate a fit with
  a bounded parameter, confirm Hessian and MINOS differ as expected.
- *Risk:* very low. Worst case is MINOS fails → fall back to Hessian.

## Score = impact × ease = **20**

Tier: **Now**. This is the highest-leverage roadmap item: an
existing dependency, one method call, immediate quality improvement.
Should be promoted to a full study + implementation pass first.
