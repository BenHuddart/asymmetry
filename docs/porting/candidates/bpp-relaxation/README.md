# BPP relaxation model

**Status:** candidate. Surfaced by practical-workflow #10 (muon
diffusion).

## What

Add a Bloembergen-Purcell-Pound parametric model to
`core/fitting/parameter_models.py` for fitting the temperature
dependence of the muon relaxation rate λ(T) when the muon is
diffusing.

## Why

- Diffusion / motional-narrowing studies are a major μSR
  application (Blundell Ch 8).
- The standard parametric model — λ(T) ∝ τ(T)/[1 + ω²τ²(T)] with
  τ(T) = τ₀·exp(E_a/k_BT) — is in every textbook but absent from
  Asymmetry's parameter_models registry.
- Currently users fall back to scipy with a hand-coded model.

## Prior art

- **musrfit**: implicit via the FUNCTIONS block; no dedicated
  parametric model.
- **WiMDA**: ❌.
- **Mantid**: ❌.

## Roadmap position

Low complexity (~30 lines of numpy + registry entry), modest
impact. Likely Later tier; ship opportunistically alongside the
theory-library expansion.
