# Brandt field-dependent vortex-lattice penetration depth — study

**Status:** study (implementation pending user confirmation of the design).

**Slug:** `sc-brandt-penetration-depth`

**References studied:**

- E. H. Brandt, *Phys. Rev. B* **37**, 2349 (1988) — second moment of the
  field distribution of an ideal triangular vortex lattice.
- E. H. Brandt, *Phys. Rev. B* **68**, 054506 (2003) — refined numerical
  field-dependence interpolation for the FLL second moment, valid over the
  whole field range `0 < b < 1`.
- F. L. Pratt *et al.*, *Phys. Rev. B* **79**, 052508 (2009) — LiFeAs TF-µSR
  validation example (corpus `GROUND_TRUTH.md`). Eq. (2)/(3) and Fig. 2.
- J. E. Sonier, J. H. Brewer, R. F. Kiefl, *Rev. Mod. Phys.* **72**, 769
  (2000) — review of the London/Brandt linewidth–`λ` relation.
- Existing in-repo SC machinery: `asymmetry.core.fitting.sc.constants`
  (`BRANDT_COEFFICIENT = 0.0609`, `lambda_nm_to_sigma_us`) and the σ(T) SC
  param-models in `parameter_models.py`.

This is a **theory addition**, not a port from WiMDA/musrfit/Mantid — those
programs do not ship a field-dependent Brandt B_rms(B₀) model. The study
documents the physics, the in-repo seam, and the chosen parameterisation.

## Why this study exists — the gap

The superconducting models already in `PARAMETER_MODEL_COMPONENTS`
(`SC_SWave`, `SC_DWave`, `SC_TwoGap_*`, …) are all **temperature-domain**
σ(T) gap-function models: they fit the line width as a function of `T` at
fixed field and map it onto the normalised superfluid density ρ_s(T) ∝ λ⁻²(T).

There is **no field-domain** model: nothing fits the line width measured
across applied field σ(B₀) to extract the absolute penetration depth λ and
the upper critical field B_c2 of a type-II superconductor. That field
dependence is exactly Brandt's Ginzburg–Landau result for the vortex lattice,
and it is what Pratt *et al.* used to extract λ_ab = 195/244 nm in LiFeAs.

`sc.constants.lambda_nm_to_sigma_us` already encodes the **field-independent
London limit** (the `b → 0` value); this study adds the field dependence
`g(b)` that multiplies it, with `b = B₀/B_c2` as a second fitted axis.

## The physics

For an ideal triangular flux-line lattice in an extreme type-II
superconductor (κ ≫ 1), Brandt's numerical result for the rms field width is

```
B_rms(b) = 0.0609 · (Φ₀ / λ²) · (1 − b) · [1 + 1.21 (1 − √b)³]
```

with `b = B₀/B_c2`, `Φ₀ = 2.0678×10⁻¹⁵ Wb` the flux quantum, and λ the
London penetration depth. The Gaussian muon depolarisation rate is

```
σ(B₀) = γ_µ · B_rms(B₀),   γ_µ = 2π·135.5 MHz/T = 0.8516 µs⁻¹ mT⁻¹.
```

Putting σ in µs⁻¹ and λ in nm, the single-crystal relation is the widely
cited form

```
σ(B₀) [µs⁻¹] = A · (1 − b) · [1 + 1.21 (1 − √b)³] · λ⁻² [nm⁻²]
```

with `A ≈ 4.85×10⁴ µs⁻¹·nm²` (= 0.0609·γ_µ·Φ₀ in nm/µs units; the commonly
quoted literature value is 4.83×10⁴, differing only by rounding of 0.0609).

### Consistency with the existing London-limit helper

The bracket at zero field is `(1−0)·[1 + 1.21·1] = 2.21`. Therefore

```
σ(b → 0) = A · 2.21 · λ⁻² = 0.0609 · γ_µ · Φ₀ / λ² = lambda_nm_to_sigma_us(λ).
```

So the existing `lambda_nm_to_sigma_us` is exactly the `b → 0` maximum of the
Brandt curve. The new model is its field-dependent generalisation. The
implementation reuses `lambda_nm_to_sigma_us(λ)` for the scale and multiplies
by the **normalised** field factor

```
g(b) = (1 − b) · [1 + 1.21 (1 − √b)³] / 2.21,   g(0) = 1, g(1) = 0,
```

guaranteeing the new model is numerically consistent with the σ↔λ helpers
already in the codebase.

### Powder average

LiFeAs (and most µSR penetration-depth samples) are **polycrystalline**. For a
uniaxial superconductor the powder average replaces λ by an effective length;
Pratt's Eq. (3) writes the denominator as `(3^{1/4} λ_ab)²`, i.e. the powder
σ is smaller than the single-crystal value at the same λ by a factor √3
(equivalently a single-crystal fit of powder data under-estimates λ by
`3^{1/4} = 1.316`). The model therefore needs a geometry option so the LiFeAs
validation recovers λ_ab = 195/244 nm rather than ~148/185 nm.

## Entry points / data flow / seam

- Core math: new functions in `src/asymmetry/core/fitting/sc/models.py`
  (`brandt_field_width_sigma`, sitting beside the σ(T) `sc_*` models), reusing
  `sc.constants.lambda_nm_to_sigma_us` and `BRANDT_COEFFICIENT`.
- Registry: new `field`-scope entries in
  `PARAMETER_MODEL_COMPONENTS` inside
  `_register_superconducting_components()` of `parameter_models.py`
  (the σ(T) ones are `temperature`-scope; these are `field`-scope, so they
  surface only for field-trend fits — see `component_names_for_x`).
- Param metadata: add `lambda_ab` (nm) and `Bc2` (T) to
  `PARAM_INFO_REGISTRY` in `parameters.py`; reuse `sigma_bg` (µs⁻¹).
- Consumption: unchanged. `ParameterCompositeModel` / `fit_parameter_model`
  call `component.function(x, **params)`; x is the **field in Gauss** (the
  field-scope convention, cf. the Redfield/`λ0D` component). B_c2 is a
  parameter in **Tesla**; the model forms `b = (x·1e-4)/Bc2`.

## Edge cases

- `b ≥ 1` (B₀ ≥ B_c2): above B_c2 there is no vortex lattice; clamp `g(b)=0`
  for `b ≥ 1` and clip `√b` domain so no NaNs leak into the fit residuals.
- λ ≤ 0 or B_c2 ≤ 0: guard with the same `max(·, tiny)` pattern used by
  `lambda_nm_to_sigma_us`; default_min = 0 on both params.
- Optional field-independent nuclear/background channel σ_bg added in
  quadrature `√(σ² + σ_bg²)`, matching Pratt Eq. (2) and the existing
  `sc_*_q` quadrature convention. Default 0 (pure σ_VL field dependence).

See `comparison.md`, `implementation-options.md`, `test-data.md`,
`verification-plan.md` for the rest of the study pass.
