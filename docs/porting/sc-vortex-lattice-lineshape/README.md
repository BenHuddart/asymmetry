# Vortex-lattice field-distribution lineshape — study

**Status:** implemented.

**Slug:** `sc-vortex-lattice-lineshape`

**References studied:**

- E. H. Brandt, *Phys. Rev. B* **68**, 054506 (2003) — FLL second moment and its
  field dependence.
- J. E. Sonier, J. H. Brewer, R. F. Kiefl, *Rev. Mod. Phys.* **72**, 769 (2000)
  — review of the modified-London model and TF-µSR lineshape analysis.
- F. L. Pratt *et al.*, *Phys. Rev. B* **79**, 052508 (2009) — LiFeAs powder
  validation example (`Superconductivity/LiFeAs/GROUND_TRUTH.md`).
- Existing SC stack: `sc-brandt-penetration-depth` (the σ↔λ field-domain trend
  models) and `sc.constants.lambda_nm_to_sigma_us`.

This is a **theory addition** (a new time-domain fit component), not a port from
WiMDA — WiMDA fits a Gaussian to the TF line and does not ship a vortex-lattice
lineshape. musrfit exposes VL field-distribution `userFcn`s; this is the
in-repo, scriptable equivalent.

## Why this study exists — the gap

The `sc-brandt-penetration-depth` study added field-domain trend models that map
a **Gaussian rate** `σ_VL` to `λ` and `B_c2`. That mapping is only as good as the
`σ_VL` fed to it, and **a single Gaussian fitted in the time domain does not give
a well-defined `σ_VL` for a vortex lattice.**

The flux-line-lattice field distribution `p(B)` is strongly non-Gaussian: a sharp
low-field cutoff at the saddle point, a most-probable field below the mean, and a
long tail to high field near the vortex cores (a positively skewed line). Fitting
a symmetric `Oscillatory * Gaussian` to the resulting signal returns a rate that
**depends on the fit window and binning, not the true second moment**. On the
LiFeAs corpus the fitted single-Gaussian `σ` for one run spans **0.7–3.1 µs⁻¹**
across reasonable (rebin, t_max) choices — so `λ_ab` is unidentifiable from a
Gaussian fit (it lands ~240 nm at the robust default vs the published 195 nm).

This study adds the missing piece: a fit component that **is** the vortex-lattice
lineshape, so the line is fitted directly instead of via a window-dependent
Gaussian proxy.

## The physics

Spatial field of an ideal triangular FLL in the modified-London model:

```
B(r) = B0 * sum_G  exp(-xi^2 G^2 / 2) / (1 + lambda^2 G^2) * exp(i G.r)
```

over the reciprocal lattice `G` of the triangular lattice (spacing set by flux
quantisation, `area/vortex = Phi0/B0`), with core cutoff
`xi = sqrt(Phi0 / 2*pi*Bc2)`. The field distribution is
`p(B) = <delta(B - B(r))>_r`, sampled on a real-space grid over one unit cell.
The muon relaxation is its characteristic function

```
R(t) = < exp(i 2*pi*gamma_mu (B(r) - B0) t) >_r ,   R(0) = 1,
```

and the measured asymmetry is
`P_x(t) = A * Re[ exp(i(2*pi*gamma_mu*B0*t + phase)) * R(t) ]`. `|R(t)|` is the
envelope; `arg R(t)` carries the skew.

### Calibration to the existing Brandt width (key decision)

The modified-London reciprocal-lattice sum gives a second-moment coefficient
~3 % different from Brandt's `0.0609` (encoded in `lambda_nm_to_sigma_us`). To
keep the new lineshape **numerically consistent** with the rest of the SC stack,
the line's second moment is rescaled to equal
`brandt_field_width_sigma[_powder]` exactly at every `(B0, λ, B_c2)`. The
modified-London computation then supplies only the **shape** (skew, higher
moments); the **width** — hence the extracted `λ` and `B_c2` — is the validated
Brandt result. Fitting this lineshape and reading `σ_VL → λ` through the existing
converters therefore agree by construction.

### Powder average

LiFeAs (and most µSR penetration-depth samples) are polycrystalline. As in
`sc-brandt-penetration-depth`, the powder variant replaces `λ_ab` by
`3^{1/4} λ_ab` (Pratt Eq. (3)), reusing `_POWDER_LAMBDA_FACTOR`. The line shape
uses the single-crystal form at that effective length — the standard practical
powder approximation (correct second moment; the true orientation average is
slightly more smeared). See `comparison.md`.

## Entry points / data flow / seam

- Core math (GUI-free): `src/asymmetry/core/fitting/sc/lineshape.py`
  (`vortex_lattice_relaxation`, `vortex_lattice_component`,
  `vortex_lattice_powder_component`), reusing `sc.models.brandt_field_width_sigma`
  for the width and `sc.constants.FLUX_QUANTUM_WB`.
- Registry: two **time-domain** `Oscillation`-category components in
  `COMPONENTS` (`composite.py`): `VortexLattice` and `VortexLatticePowder`,
  params `[A, field, phase, lambda_ab, Bc2]`, `field` pre-fixed.
- Docs: `docs/user_guide/fit_functions/oscillation.rst` +
  `component_docs.py` applicability/references.
- Consumption: unchanged. `CompositeModel.from_expression(...)` composes them,
  e.g. `VortexLatticePowder * Gaussian + Oscillatory + Constant`.

## Edge cases

- `B0 >= B_c2`, `λ <= 0`, `B_c2 <= 0`: no vortex lattice → `R(t) = 1` (undamped),
  matching the Brandt `g(b>=1)=0` clamp.
- `field` is the applied field and starts fixed; `B_c2` is weakly constrained by
  a single low-field run and is best fixed or fit across the field dependence.
- `lambda_ab` is strongly correlated with the nuclear Gaussian rate; constrain
  the latter from a normal-state run (see `verification-plan.md` and the LiFeAs
  cookbook).

See `comparison.md`, `implementation-options.md`, `test-data.md`,
`verification-plan.md` for the rest of the study pass.
