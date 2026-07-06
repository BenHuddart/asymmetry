# Prototype results: BED next-point acquisition math

Companion to [bed-next-point-suggestion.md](bed-next-point-suggestion.md).
A standalone deterministic script (numpy + iminuit from the project venv,
seeded RNG) implemented the §3.1–3.2 acquisition math and ran it against
the closed-form design oracles before any production code was written.
Run date: 2026-07-06. Note on units: the prototype labels the statistics
scale factor `t/t_ref`; the production design expresses the identical
quantity as an event-count ratio `N/N_ref`.

## Oracle outcomes (all as predicted)

| Case | Setup | Expected | Observed |
|------|-------|----------|----------|
| 1. Straight line | (m,b)=(0.5,1.0), 8 pts on [1,9], candidates [0,10] | D- and c-opt(m) at a boundary; c-opt(b) near x=0 | D and c(m) at boundary (tie between 0 and 10, symmetric utilities); c(b) at x=0 |
| 2. Arrhenius | a=10, E_a=20 meV, T∈[100,300], candidates [80,320], heteroscedastic 3% errors | c-opt(E_a) at an extreme of reachable 1/T | argmax at 80 K (low-T end); low-T end 3.8× more informative than high-T end |
| 3. Order parameter | (y0,T_c,α,β)=(1,100,2,0.35), 10 pts on [10,95], candidates [5,120] | c-opt(T_c) just below T_c | argmax at 99.9 K; genuine secondary peak at 86 K (4.3× smaller), tertiary ~47 K; utility ~0 above T_c |
| 5. Event-count solve | target σ(T_c) ≤ 0.5 K at x*, and aggressive 0.01 K | closed-form solve + unreachable detection | 0.5 K reachable at N/N_ref ≈ 0.12; 0.01 K correctly UNREACHABLE — single-point N→∞ floor is σ = 0.176 K, set by the other parameters' residual uncertainty |

## Case 4 — the important negative result

Rank-one predicted post-variance vs Monte-Carlo reality (200 simulated
refits per candidate, 0 fit failures):

| Candidate | Predicted post-Var(T_c) | Realized (MC mean) | Ratio |
|-----------|------------------------|--------------------|-------|
| x* = 99.87 K (informative) | 0.058 | 0.330 | 0.18 |
| T = 20 K (flat region) | 11.53 | 117 | 0.10 |

**Ranking is preserved** — both agree the near-T_c point is transformative
(prior Var = 11.6) and the flat-region point is useless. But the
**magnitude is off ~5.6×** at the informative point, and got *worse* with
a tighter prior. Root cause (verified not to be a finite-difference
artifact): for β < 1 the order-parameter curve has infinite slope as
T→T_c⁻, so ∂f/∂T_c diverges just below T_c and collapses discontinuously
to 0 above it. The Laplace/rank-one update assumes a locally quadratic
likelihood; that assumption fails hardest exactly where the utility peaks.
This is structural to critical-exponent models — the μSR bread-and-butter
case — not a bug.

Consequence for the design: the utility curve and argmax are trustworthy;
the **predicted post-σ and events-to-target figures are not**, near
critical points. The v1 counting recommendation therefore includes a cheap
Monte-Carlo calibration pass: shortlist top candidates analytically, then
validate/calibrate with ~50 simulated add-point-and-refit trials
(sub-second per trial; run off-thread). See plan task 1.4.

## Numerical gotchas for production

1. **Near-boundary gradients**: the anticipated failure (NaN/inf at
   T ≥ T_c) never occurred — gradients stay finite but are *unstable*
   within one finite-difference step of the fitted T_c (huge or exactly
   zero depending on which side the step lands). `nan_to_num` alone would
   not catch this. Guard: one-sided differences (or candidate
   deprioritisation) when a step would straddle a model domain boundary.
2. **Covariance conditioning**: the 4-parameter order-parameter Σ had
   condition number ~1.1e5 with |corr(T_c, β)| ≈ 0.95. Production should
   check `np.linalg.cond` (warn above ~1e4) and require iminuit
   `valid`/`accurate` before trusting Σ.
3. **Gradient step**: `max(1e-6, 1e-6·|θ_j|)` central differences were
   stable away from singularities (values varied &lt;10% across step sizes
   1e-6×–1e-2×). A 10×-step stability cross-check is worthwhile only for
   candidates near domain edges, not unconditionally.
4. **Fit robustness was a non-issue**: Migrad converged in all 400 MC
   refits — the inaccuracy lives in the covariance-based *prediction*,
   not the refitting.
5. Seed per-trial RNG (`default_rng(base + trial)`) so individual MC
   trials are independently replayable.

Prototype script: [bed_prototype.py](bed_prototype.py) (deterministic,
seeded; run with the project venv python — it needs the pinned
numpy 2.2.x + iminuit).
