# Test data

No reference-program output exists; verification uses independent numerical
references and exact limits, all synthetic.

- **Helix-phase average**: the field (f cos ψ, r f sin ψ) averaged over 8 192
  uniform ψ samples — independent of the density and the series — for
  r ∈ {0.05, 0.3, 0.46, 0.8, 0.999}, phases 0 and 0.7, times up to Δω t ≈ 200
  (both evaluation paths), and five crystal orientations.
- **B² arcsine reference**: the zero-phase line as a 60 000-node midpoint rule in
  u = B², converged for the powder down to r = 1e-5 and for the crystal weight
  down to r = 1e-3.
- **Closed-form limits**: J₀ and J₀/H₀ at r = 0; a cosine at r = 1; W₀ =
  (a² + c² r)/(1 + r); the magic orientation reproducing the powder split.
- **Calibration** (scratch, not in the suite): coefficient tails from 8× bound
  DCTs and minimal node counts against 60 000-node references, for r from 1e-5
  to 0.95, for both weights.
- **Real data**: private zero-field powder data, used only to confirm fitting
  behaviour; not in the repository and not a regression corpus.
