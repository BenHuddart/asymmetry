# MaxEnt frequency reconstruction: comparison

| Aspect | WiMDA | musrfit | Mantid | Asymmetry |
|---|---|---|---|---|
| Algorithm | Burg pole-scan + inverse cosine transform | limited / not flagship | Iterative entropy maximisation with phase refinement | stub |
| Phase optimisation | external (phase table) | n/a | ✅ internal | n/a |
| Group-resolved | ◐ via plot pipeline | ❌ | ★ native | ❌ |
| Pole-count auto-selection | ✅ FPE criterion | n/a | n/a | n/a |
| Speed | fast (single-pass) | n/a | slow (iterative) | n/a |
| Reference | `MaxEnt.pas`, `Fourier.pas` | | `MuonMaxent.cpp` | `core/fourier/maxent.py` (stub) |

## Algorithm summary

### Burg pole-scan (WiMDA)

1. Estimate the autoregressive (AR) coefficients of order *p* from
   the time-domain signal via Burg's recursion (`memcof`).
2. Compute the FPE criterion as a function of *p*; pick the *p* that
   minimises FPE.
3. Evaluate the AR power spectrum on a chosen frequency grid:
   `S(f) = σ² / |1 - Σₖ aₖ exp(-2π i f k Δt)|²`.
4. Optionally inverse-transform back to time domain for residual
   inspection.

Pros: fast, self-contained, has a robust pole-count criterion.
Cons: less expressive for very long signals; no phase optimisation.

### Iterative entropy maximisation (Mantid)

1. Initialise a flat positive prior over frequency bins.
2. At each iteration, compute the chi-squared between data and the
   inverse-FT of the current spectrum, then perturb spectrum bins
   in the direction that minimises chi-squared while maximising
   the Shannon entropy `S = -Σ pᵢ log pᵢ`.
3. Stop when chi-squared/Ndof ~ 1.
4. Optionally re-fit per-detector phases at each iteration.

Pros: produces phase-consistent group-resolved output; more
expressive than Burg.
Cons: slow (typically 20-200 iterations); needs care to avoid
runaway iterations on noisy data.

## Recommendation for Asymmetry

Implement **Burg first** as the canonical single-channel MaxEnt:
- Faster to land (~200 lines numpy)
- Easier to validate (analytic test cases available)
- Sufficient for the common pedagogical use case

Then add **iterative entropy** as a second engine selectable from
the Fourier panel. The two engines share the same API contract
(`maxent(dataset, ...)`).

## Validation oracle

Synthesise a known dataset (sum of N delta peaks + noise), apply
both engines, confirm:
- Peak positions recovered within 1 frequency-bin.
- Peak heights within 5% of input.
- For Mantid-iterative: chi²/Ndof → 1.
