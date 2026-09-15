# Fermi-function transition step — study

**Status:** implemented (2026-09-15).

**Slug:** `fermi-step-transition`

**Implementation:** `FermiStep` in `src/asymmetry/core/fitting/parameter_models.py`
(function `_fermi_step`, seeds `_fermi_step_seeds` via `suggest_trend_seeds`),
parameter metadata `A1`/`A2`/`dT` in `parameters.py`, applicability text in
`component_docs.py`, user docs in `docs/reference/parameter_trending.rst`
§ "Transition step". Tests: `tests/core/test_fermi_step_parameter_model.py`.

## Why

The initial (paramagnetic) asymmetry across a magnetic transition is a step, not
an order parameter: in weak transverse field the oscillating amplitude falls from
the full sample asymmetry to the background on cooling, and in zero field a powder
loses 2/3 of its sample asymmetry. The trend library had `OrderParameter` and
`CriticalDivergence` but no step, so users could not extract the midpoint and
width of a volume-fraction transition. The μSR literature fits these trends with
a Fermi-function (logistic) step.

## References studied

- Mantid `SmoothTransition` (`$MANTID_SRC/Framework/CurveFitting/src/Functions/SmoothTransition.cpp`
  and `docs/source/fitting/fitfunctions/SmoothTransition.rst`) — the only
  reference-program implementation found. WiMDA and musrfit ship no step trend.
- R. Khasanov *et al.*, Phys. Rev. B **102**, 094504 (2020) — wTF A(0,T).
- A. J. Steele *et al.*, Phys. Rev. B **84**, 064412 (2011) — ZF late-time amplitude.
- R. Khasanov *et al.*, arXiv:2512.22371 (2025) — sum of steps for multiple phases.
- A. Hernández-Melián *et al.*, arXiv:2211.15560 (2022) — tanh form.
- E. Nocerino *et al.*, arXiv:2209.11966 (2022) — unspecified "sigmoid".
- B. A. Frandsen *et al.*, Nat. Commun. **7**, 12519 (2016) — wTF/ZF volume-fraction normalisation.
- S. Kamusella *et al.*, Phys. Rev. B **95**, 094415 (2017) — error-function (Gaussian T_c distribution) variant.

## Decision

Plateau parameterisation `y = A2 + (A1 − A2)/(exp((T − Tc)/dT) + 1)` (Steele,
Mantid), temperature scope, category "Critical behaviour", data-aware seeds.
Deferred: an error-function sibling and a first-class multi-step sum (see
`implementation-options.md`).
