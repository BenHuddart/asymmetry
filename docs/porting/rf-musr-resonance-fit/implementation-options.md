# Implementation options

## Where the feature lives (chosen: field-scope parameter-trend component)

The RF-µSR observable is **integral asymmetry vs swept field**, fit with two
Lorentzian resonances whose centres are fixed by `(A_µ, A_p, ν_RF)`. That is a
**parameter-vs-field trend**, not a time-domain line shape — so the natural seam
is `core/fitting/parameter_models.py` with `scopes=("field",)`, alongside the
existing field components (`Lorentzian`, `LorentzianLCR`, `Redfield`,
`MuRepolarisation`, the diffusion/ballistic LF models).

| Option | Seam | Verdict |
|---|---|---|
| **A. Field-scope parameter-trend component** | `parameter_models.py` `RFResonanceMuP` | **Chosen.** Matches the observable (asymmetry vs field), reuses `ParameterCompositeModel` + `fit_parameter_model`, composes with other field terms, and is the same surface as the already-ported `MuRepolarisation` / `LorentzianLCR` resonance fits. |
| B. Time-domain fit component | `composite.py` `COMPONENTS` | Rejected. RF resonance is not a `f(t)` line shape; it is a field-domain peak model. Would misuse the time-domain framework and the picker category. |
| C. Standalone analysis function only | `core/fitting/muon_proton.py` helper | Insufficient on its own — would not appear in the fit builder / trend panel. Kept *in addition* as the physics core that the component wraps. |

## Module layout (chosen)

- **`core/fitting/muon_proton.py`** (new) — pure physics, no Qt/matplotlib:
  - operators built once at import (electron ⊗ muon ⊗ proton);
  - `mu_proton_hamiltonian(field, A_mu, A_p)` and
    `mu_proton_levels(field, A_mu, A_p)` (batched `eigvalsh` over field);
  - `rf_transition_freqs` (exact, `E₇−E₅`/`E₈−E₆`) and
    `analytic_rf_transition_freqs` (cross-check only);
  - `rf_resonance_fields(A_mu, A_p, nu_RF, …)` — coarse scan + `brentq`, `nan`
    when unbracketable;
  - `rf_resonance_mup(x, A_mu, A_p, nu_RF, ampl1, wid1, ampl2, wid2, BG)` — the
    two-Lorentzian field-swept model (the registered `function`).
- **`parameter_models.py`** — import `rf_resonance_mup`, register
  `RFResonanceMuP` (field scope, 8 params, defaults at the benzene values).
- **`parameters.py`** — `ParamInfo` registry entries for `A_mu, A_p, nu_RF,
  ampl1, ampl2, wid1, wid2` (display/units; `BG` already present).
- **`component_docs.py`** — `PARAMETER_MODEL_APPLICABILITY["RFResonanceMuP"]`
  and `PARAMETER_MODEL_REFERENCES["RFResonanceMuP"]` (McKenzie 2013 + Roduner).
- **`docs/user_guide/parameter_trending.rst`** — user-facing section + a row in
  the "Migrating WiMDA Model Functions" table.

This mirrors how `sc/`, `diffusion.py`, `ballistic.py` keep their physics in a
core module and `parameter_models.py` only registers a thin wrapper.

## Design decisions

1. **Eigensolver:** `numpy.linalg.eigvalsh` (batched) instead of porting
   `Eigenuni.pas`. Justified by basis-independence of the Hermitian spectrum;
   only eigen*values* (level differences) are needed for the RF fit, so the
   slower eigen*vector* routines are unnecessary here.
2. **Exact only as a component.** The analytic variant is kept as a module
   function for cross-checking but is **not** registered, because it is
   inaccurate at the low fields of RF-µSR (comparison.md). One component, no
   user-facing footgun.
3. **Robust root finding.** Lowest *ascending* crossing + `nan` fallback, so the
   model is finite for any minimiser trial (vs WiMDA's single `zbrent`). The
   coarse-scan resolution (256 points over `[1, 2000] G`) brackets every
   physical crossing; `brentq` (`xtol=1e-10`) then makes `B₁(A,Aₚ)`/`B₂(A,Aₚ)`
   smooth enough for a finite-difference Jacobian.
4. **Parameter naming.** `A_mu`/`A_p` (not the existing asymmetry-percent `A`),
   `nu_RF` (the fixed applied frequency, normally held fixed), and WiMDA's
   `ampl1/wid1/ampl2/wid2/BG` for the two Lorentzians + background.

## Follow-ons (not in scope here)

- An optional **starting-value helper** (estimate `A_µ, A_p` from the two
  observed dip fields via `rf_resonance_fields` inversion) would make the GUI fit
  one-click; the physics for it already exists (`rf_resonance_fields` is
  invertible). Deferred.
- A registered **`MuProtLevels` / `MuProtAsym`** energy-level/asymmetry component
  (WiMDA's other two RigiWorkshopFit registrations) could reuse `mu_proton_levels`
  directly; deferred, as PC1 is specifically the RF-resonance fit.
