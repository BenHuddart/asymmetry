# Muonium triplet — implementation options & chosen design

## Option 1 — three independent lines + link groups (already shipped, NOT this)

`Oscillatory*Exponential ×3 + Constant` with link groups sharing
amplitude/phase/relaxation. Frequencies free; symmetry only recovered, not
enforced; hyperfine = `f₅ − f₃` post-fit. This is the flexible baseline; the new
component is the constrained complement, not a replacement.

## Option 2 — faithful WiMDA `TFMuonium` (4-level, field-parameterised) — deferred

Port `TFMuonium`/`LowTFMuonium` verbatim, params `(B, A, phase)`. Maximum
fidelity, but field-coupled and 4-line; overkill for the shallow-donor small-A
CdS case and awkward to seed. Recorded as a future high-field addition.

## Option 3 — self-contained symmetric triplet component (CHOSEN)

A new baseline-free `ComponentDefinition` `MuoniumTriplet`, used additively
(`MuoniumTriplet + Constant`).

### Function (top-level, picklable — required for batch/global)

```python
def _muonium_triplet_component(
    t, A_centre, A_sat, f_centre, hyperfine, Lambda, phase
):
    t = np.asarray(t, dtype=float)
    damp = np.exp(-Lambda * t)
    two_pi = 2.0 * np.pi
    f_lo = f_centre - 0.5 * hyperfine
    f_hi = f_centre + 0.5 * hyperfine
    return damp * (
        A_centre * np.cos(two_pi * f_centre * t + phase)
        + A_sat * np.cos(two_pi * f_lo * t + phase)
        + A_sat * np.cos(two_pi * f_hi * t + phase)
    )
```

Lives in `src/asymmetry/core/fitting/` (new small module `muonium.py`, or
alongside the other component fns in `composite.py`). Must be module-level.

### Registry & metadata

- Add to `COMPONENTS` in `composite.py`:
  `param_names = ["A_centre","A_sat","f_centre","hyperfine","Lambda","phase"]`,
  `param_defaults`, `param_info`, `formula_template`, `latex_equation`,
  `category="Muonium"` (a new category; the builder groups by `category`
  automatically), `domain="time"`.
- Parameter metadata in `parameters.py` `PARAM_INFO_REGISTRY` / `get_param_info`:
  reuse `frequency`, `Lambda`, `phase`; **add** `A_centre`, `A_sat` (amplitude,
  `default_min=0.0`, unit "%") and `hyperfine` (label "Aµ", unit "MHz",
  `default_min=0.0`, the satellite splitting). Names are distinct from `A`/`A_bg`
  so `_is_scaling_parameter` does not fold them into the chain-amplitude logic.
- Component applicability blurb in `component_docs.py`
  (`FIT_COMPONENT_APPLICABILITY`) for the Component-Info dialog.

### No GUI code changes

The builder dropdown (`fit_function_builder.py` `_build_components_by_category`)
and the Component-Info dialog read `COMPONENTS`/metadata, so the component
appears and renders its equation automatically. It is reachable via typed
expressions too (`MuoniumTriplet + Constant`).

### Seeding & robustness notes

- `hyperfine` and `Lambda` and the amplitudes get `default_min = 0.0`.
- Default seeds chosen for visibility (e.g. `f_centre = 1.0` MHz, `hyperfine =
  0.2` MHz, `A_centre = 25`, `A_sat = 10`, `Lambda = 0.3`, `phase = 0`). Seeding
  `hyperfine` away from 0 matters — at Δ=0 the three lines collapse and the
  splitting gradient vanishes (same class of trap as the OrderParameter Tc seed).
- `.asymp` round-trip is automatic (model serialises via `CompositeModel.to_dict`
  by component name; params via the existing dict path).

### Design decisions taken

- **Full splitting** `hyperfine` Δ (= f₊ − f₋) as the param, not the half δ, so
  the fitted value is the hyperfine constant directly (acceptance `2δ ≈ 0.242`).
- **Shared** `Lambda` and `phase` across all three lines (matches the linked CdS
  fit that gave χ²ᵣ=1.35). Per-line satellite phase / central-vs-satellite
  damping are recorded as future refinements (the docx hints satellite phases may
  vary), kept out to stay minimal and symmetry-pure.
- **Free** `A_centre`, `A_sat` (not a single A + ratio) so `A_sat` trends
  directly in the Arrhenius batch workflow.
