# Dynamic Kubo–Toyabe

**Status:** candidate. Not yet promoted to a full study.

## What

Implement the **dynamic Kubo–Toyabe** polarisation function — the
strong-collision-model generalisation of the static GKT that captures
muon dephasing in the presence of fluctuating local fields with
correlation time `ν⁻¹`. The function reduces to the static GKT for
`ν → 0` and to motional-narrowing exponential decay for `ν → ∞`.

Reference: Hayano et al. PRB 20, 850 (1979); textbook chapter 5.3
of Blundell, De Renzi, Lancaster, Pratt.

## Why

Dynamic KT is the canonical model for *every* μSR study of magnetic
systems with thermal fluctuations — spin glasses, paramagnets near a
transition, frustrated magnets, hopping muons. It is the most
frequently-cited "missing model" in Asymmetry against the
`docs/user_guide/lf_kubo_toyabe.rst` page, which currently has to
stay in the static regime.

## Prior art

- **Mantid:** `DynamicKuboToyabe` fit function in
  `Framework/CurveFitting/src/Functions/DynamicKuboToyabe.cpp`.
- **musrfit:** `PTheory` includes `dynKTLF` and `dynKTZF`.
- **WiMDA:** dynamic KT exposed via the `musr-function-registry`
  DLL.

All three implement the strong-collision integral; numerical strategies
differ. Mantid uses an iterative time-step convolution; musrfit uses a
quadrature over the static-KT response convolved with an exponential
fluctuation kernel.

## Why this is roadmap-tractable

- The static analogue (`StaticGKT_ZF`, `LFKuboToyabe`) already exists
  in `src/asymmetry/core/fitting/models.py`. The new function fits
  the same registration pattern.
- The synthesis pipeline used for screenshot scenarios
  (`docs/screenshots/data/archetypes.py`) needs only a single extra
  generator to produce a fluctuation series.
- The Fit Wizard's candidate portfolio already lists "static vs.
  dynamic KT" as a comparison; adding the dynamic branch makes the
  portfolio recommendation honest.

## Notes for the study pass

- Choose a numerical strategy: time-step convolution (Mantid-style)
  vs. memory-kernel quadrature (musrfit-style). Time-step is easier
  to validate; quadrature is faster.
- Cross-check against Mantid output on identical synthetic data.
- The textbook lists exact analytic forms in the slow-fluctuation
  (`νt → 0`) and fast-fluctuation (`νt → ∞`) limits — use those as
  unit tests.
