# Implementation options

## A. Parameterisation of the field factor

- **A1 (chosen): normalised field factor reusing the existing helper.**
  `σ(B₀) = lambda_nm_to_sigma_us(λ) · g(b)`,
  `g(b) = (1−b)[1+1.21(1−√b)³]/2.21`. Fitted params: λ (nm), B_c2 (T).
  Pros: guaranteed consistent with the in-repo σ↔λ helpers
  (`g(0)=1` ⇒ `σ(b→0)=lambda_nm_to_sigma_us(λ)`); λ comes out directly in nm.
- A2: hard-code the literature prefactor `4.83×10⁴·(1−b)[1+1.21(1−√b)³]·λ⁻²`.
  Equivalent, but duplicates the constant and risks drifting from the
  codebase's `BRANDT_COEFFICIENT`/`lambda_nm_to_sigma_us`.
- A3: fit a free amplitude `σ_0` instead of λ. Rejected — defeats the purpose
  (λ would not be a fitted output); the σ(T) models already cover free-σ_0
  trends.

## B. Geometry (single crystal vs powder)

- **B1 (chosen): two registered components sharing one core function.**
  `SC_Brandt_VortexLattice` (single crystal) and
  `SC_Brandt_VortexLattice_Powder` (applies the `3^{1/4}` ab-plane powder
  length, σ → σ/√3), the powder one bound via `functools.partial(..., powder=True)`.
  Pros: powder samples (the common µSR case, incl. LiFeAs) are first-class and
  testable; the only difference is one documented constant.
- B2: single-crystal only, document the powder correction in prose.
  Rejected — the motivating LiFeAs example is a powder; users would silently
  under-estimate λ by 1.316×.
- B3: a continuous `geometry` float param. Rejected — geometry is discrete,
  and a free/continuous flag invites nonsense intermediate values.

## C. Background / nuclear channel

- **C1 (chosen): optional σ_bg added in quadrature**, `√(σ_VL² + σ_bg²)`,
  default 0. Matches Pratt Eq. (2) `σ² = σ_VL² + σ_n²` and the existing
  `sc_*_q` quadrature convention. With the default 0 the model is the pure
  vortex-lattice field dependence you fit for λ.
- C2: additive background. Rejected — second moments of independent Gaussian
  channels add in quadrature, not linearly; quadrature is the physical choice.

## D. Units / registry plumbing

- x (field) in **Gauss** — the established field-scope convention
  (`λ0D`/Redfield component), so the model converts internally
  `B₀[T] = x·GAUSS_TO_TESLA`.
- `Bc2` parameter in **Tesla** (type-II B_c2 is tens of T; Gauss would be
  awkward five-digit values). default_min = 0.
- `lambda_ab` parameter in **nm**, default_min = 0. New `PARAM_INFO_REGISTRY`
  entries: `lambda_ab` (nm), `Bc2` (T). Reuse `sigma_bg` (µs⁻¹).
- scope = `("field",)` so it appears only in field-trend fits.

## Public surface (proposed — confirm in CONSULT)

Core (GUI-free), `sc/models.py`:

```python
def brandt_field_width_sigma(
    B0_gauss, *, lambda_nm, Bc2_tesla, sigma_bg=0.0, powder=False
) -> NDArray[np.float64]:
    ...
```

Registry components (`parameter_models.py`):

- `SC_Brandt_VortexLattice`        params [lambda_ab, Bc2, sigma_bg]
- `SC_Brandt_VortexLattice_Powder` params [lambda_ab, Bc2, sigma_bg]

Exposed for fitting through the documented `fit_parameter_model` path exactly
like every other `SC_*` component (no new entry point).
